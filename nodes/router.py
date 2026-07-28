# nodes/router.py
import logging
from typing import Dict, Any, cast
from pydantic import ValidationError
from langchain_core.runnables import RunnableConfig

from core.state import SyncState
from core.schemas import CDCPayload
from core.config import AppConfig

logger = logging.getLogger(__name__)


def route_event(state: SyncState, config: RunnableConfig) -> SyncState:
    """
    Node 1: Evaluates the incoming CDC payload and determines the mutation route.
    Returns a full state update (merge of incoming state + changes).
    """
    event_id = state.get("event_id")
    raw_payload = state.get("cdc_payload", {})

    # Helper to return a full SyncState by merging updates into the original state
    def updated_state(updates: dict) -> SyncState:
        new_state = dict(state)          # shallow copy includes all required keys
        new_state.update(updates)
        return cast(SyncState, new_state)

    # Load config
    app_config: AppConfig = config.get("configurable", {}).get("app_config")
    if not app_config:
        raise RuntimeError(f"[{event_id}] 'app_config' missing from RunnableConfig.")

    try:
        payload = CDCPayload(**raw_payload)
    except ValidationError as e:
        logger.error(f"[{event_id}] Schema validation failed: {e}")
        return updated_state({
            "mutation_type": "INVALID",
            "error_trace": f"Validation error: {e}",
            "sync_status": "FAILED"
        })

    # Route DELETE
    if payload.op == 'd':
        logger.info(f"[{event_id}] Routing DELETE for document {payload.document_id}")
        return updated_state({"mutation_type": "DELETE", "sync_status": "COMPLETED"})

    # Route INSERT
    if payload.op == 'c':
        logger.info(f"[{event_id}] Routing INSERT for document {payload.document_id}")
        return updated_state({"mutation_type": "INSERT", "sync_status": "COMPLETED"})

    # Route UPDATE
    if payload.op == 'u':
        text_before = payload.extracted_text_before
        text_after = payload.extracted_text_after

        # Need LLM service for token counting? We can instantiate a quick token counter.
        # To avoid circular dependencies, we can import the service or use a utility.
        # For simplicity, we'll rely on the LLMService from evaluator, but we can also use a separate token counter.
        # We'll use a minimal token counter here (could use same as LLMService).
        # For this node, we only need token counting; we can create a temporary counter.
        from services.llm_service import LLMService  # local import to avoid circular
        # We need the LLM client? For token counting, we don't need LLM, only the tokenizer.
        # We can create a dummy service without LLM.
        # Better: extract a separate TokenCounter class.
        # For now, we'll use a simple character heuristic if we can't get the service.
        # But to be accurate, we'll get the LLMService from config if available.
        llm_service = config.get("configurable", {}).get("llm_service")
        if llm_service:
            total_tokens = llm_service.count_tokens(text_before) + llm_service.count_tokens(text_after)
        else:
            # Fallback: rough token count
            total_tokens = (len(text_before) + len(text_after)) // 4
            logger.warning(f"[{event_id}] LLMService not available for token counting; using heuristic.")

        if total_tokens > app_config.max_context_tokens:
            logger.warning(
                f"[{event_id}] Payload exceeds token threshold ({total_tokens} > {app_config.max_context_tokens}). "
                f"Bypassing evaluator for forced sync."
            )
            return updated_state({"mutation_type": "LARGE_UPDATE", "sync_status": "COMPLETED"})

        logger.info(f"[{event_id}] Routing UPDATE to Semantic Evaluator. Tokens: {total_tokens}")
        return updated_state({"mutation_type": "UPDATE", "sync_status": "PENDING"})

    # Fallback
    return updated_state({
        "mutation_type": "INVALID",
        "sync_status": "FAILED",
        "error_trace": "Unrecognized operation."
    })