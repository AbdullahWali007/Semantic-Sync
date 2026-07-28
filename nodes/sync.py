# nodes/sync.py
import logging
import time
from typing import Optional, cast

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from langchain_core.runnables import RunnableConfig
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from core.state import SyncState
from core.schemas import CDCPayload
from core.config import AppConfig

logger = logging.getLogger(__name__)

# Retry policy for Qdrant network errors
QDRANT_RETRY_POLICY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((UnexpectedResponse, ResponseHandlingException, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)


@QDRANT_RETRY_POLICY
def _delete_point(qdrant_client, collection_name: str, point_id: str) -> None:
    """Idempotent delete with retry."""
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=[point_id]
    )


@QDRANT_RETRY_POLICY
def _upsert_point(qdrant_client, collection_name: str, point: PointStruct) -> None:
    """Idempotent upsert with retry."""
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point]
    )


def execute_vector_sync(state: SyncState, config: RunnableConfig) -> SyncState:
    """
    Node 3: Executes the final mutation against the Vector Database.
    Handles upserts (generate embeddings) and deletions (tombstoning).
    """
    event_id = state.get("event_id", "unknown")
    mutation_type = state.get("mutation_type")
    sync_status = state.get("sync_status", "PENDING")
    raw_payload = state.get("cdc_payload", {})

    # Helper to return a full SyncState by merging updates into the original state
    def updated_state(updates: dict) -> SyncState:
        new_state = dict(state)          # shallow copy includes all required keys
        new_state.update(updates)
        return cast(SyncState, new_state)

    # ========== EARLY EXIT GUARDS ==========
    # 1. If already finalised, do nothing (prevents duplicate execution)
    if sync_status in ("COMPLETED", "SKIPPED", "FAILED"):
        logger.info(f"[{event_id}] Sync node skipped – state already finalised ({sync_status}).")
        return state  # no changes

    # 2. Load configuration
    app_config: Optional[AppConfig] = config.get("configurable", {}).get("app_config")
    if not app_config:
        raise RuntimeError(f"[{event_id}] 'app_config' missing from RunnableConfig.")

    # 3. Extract and validate payload
    try:
        payload = CDCPayload(**raw_payload)
    except Exception as e:
        logger.error(f"[{event_id}] Invalid CDC payload in sync node: {e}")
        return updated_state({
            "sync_status": "FAILED",
            "error_trace": f"Payload validation error: {e}"
        })

    # 4. Validate dependencies
    embedder = config.get("configurable", {}).get("embedding_client")
    qdrant_client = config.get("configurable", {}).get("qdrant_client")
    collection_name = config.get("configurable", {}).get("collection_name")
    if not collection_name:
        collection_name = "enterprise_docs"  # fallback, but should be configured
        logger.warning(f"[{event_id}] collection_name not set, using default: {collection_name}")

    if not embedder or not qdrant_client:
        logger.error(f"[{event_id}] Missing injected clients for Vector Sync.")
        return updated_state({
            "sync_status": "FAILED",
            "error_trace": "Missing embedder or Qdrant client."
        })

    # ========== DELETE ROUTE ==========
    if mutation_type == "DELETE":
        try:
            logger.info(f"[{event_id}] Tombstoning vector ID: {payload.document_id}")
            _delete_point(qdrant_client, collection_name, payload.document_id)
            return updated_state({"sync_status": "COMPLETED"})
        except Exception as e:
            logger.error(f"[{event_id}] Delete failed after retries: {e}")
            return updated_state({
                "sync_status": "FAILED",
                "error_trace": f"Delete error: {e}"
            })

    # ========== UPSERT ROUTE (INSERT, UPDATE, LARGE_UPDATE) ==========
    if mutation_type not in ("INSERT", "UPDATE", "LARGE_UPDATE"):
        logger.warning(f"[{event_id}] Unhandled mutation_type '{mutation_type}' in sync node – skipping.")
        return updated_state({"sync_status": "SKIPPED"})

    # Validate content
    text_after = payload.extracted_text_after
    if not text_after or not text_after.strip():
        logger.warning(f"[{event_id}] Aborting upsert – 'after' text is empty.")
        return updated_state({"sync_status": "SKIPPED"})

    try:
        logger.info(f"[{event_id}] Generating embeddings for vector ID: {payload.document_id} "
                    f"(content length: {len(text_after)} chars)")
        start_time = time.perf_counter()
        vector_embedding = embedder.embed_query(text_after)
        embedding_latency = (time.perf_counter() - start_time) * 1000
        logger.debug(f"[{event_id}] Embedding generated in {embedding_latency:.1f}ms")

        # Build point with metadata
        point = PointStruct(
            id=payload.document_id,
            vector=vector_embedding,
            payload={
                "content": text_after,
                "source_ts_ms": payload.ts_ms,
                "last_mutation_type": mutation_type,
                "semantic_drift_detected": state.get("semantic_drift_detected", False),
                "evaluation_reasoning": state.get("evaluation_reasoning", ""),
            }
        )

        logger.info(f"[{event_id}] Upserting vector ID: {payload.document_id} into '{collection_name}'")
        _upsert_point(qdrant_client, collection_name, point)

        return updated_state({"sync_status": "COMPLETED"})

    except Exception as e:
        logger.error(f"[{event_id}] Vector DB transaction failed: {e}")
        return updated_state({
            "sync_status": "FAILED",
            "error_trace": f"Upsert error: {e}"
        })