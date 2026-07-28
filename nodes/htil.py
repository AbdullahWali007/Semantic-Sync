# nodes/hitl.py
import logging
from typing import Optional, cast

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from core.state import SyncState
from core.schemas import CDCPayload
from core.config import AppConfig

logger = logging.getLogger(__name__)


def human_review_node(state: SyncState, config: RunnableConfig) -> SyncState:
    """
    Node 4: Suspends execution for human oversight on high-risk updates.
    Only triggers if:
      - HITL is enabled in AppConfig.
      - semantic_drift_detected is True.
      - mutation_type is an update (UPDATE or LARGE_UPDATE).
    """
    event_id = state.get("event_id", "unknown")
    mutation_type = state.get("mutation_type")
    drift_detected = state.get("semantic_drift_detected", False)
    sync_status = state.get("sync_status", "PENDING")

    # Helper to return a full SyncState by merging updates into the original state
    def updated_state(updates: dict) -> SyncState:
        new_state = dict(state)          # shallow copy includes all required keys
        new_state.update(updates)
        return cast(SyncState, new_state)

    # ========== EARLY EXIT GUARDS ==========
    # 1. If already finalised, skip
    if sync_status in ("COMPLETED", "SKIPPED", "FAILED"):
        logger.info(f"[{event_id}] HITL node skipped – state already finalised ({sync_status}).")
        return state   # no changes

    # 2. Load configuration
    app_config: Optional[AppConfig] = config.get("configurable", {}).get("app_config")
    if not app_config:
        raise RuntimeError(f"[{event_id}] 'app_config' missing from RunnableConfig.")

    # 3. Check if HITL is enabled globally
    # (We can add an attribute to AppConfig, e.g., hitl_enabled: bool = True)
    # For safety, we default to True if not set, but we should read from env.
    hitl_enabled = getattr(app_config, "hitl_enabled", True)
    if not hitl_enabled:
        logger.info(f"[{event_id}] HITL disabled by configuration. Proceeding with drift decision.")
        return state   # do nothing, graph will proceed based on current drift flag

    # 4. Only interrupt if we have a risky update with drift
    if mutation_type not in ("UPDATE", "LARGE_UPDATE"):
        logger.info(f"[{event_id}] HITL node skipped – mutation_type '{mutation_type}' does not require review.")
        return state

    if not drift_detected:
        logger.info(f"[{event_id}] HITL node skipped – no semantic drift detected.")
        return state

    # ========== PAYLOAD VALIDATION ==========
    raw_payload = state.get("cdc_payload", {})
    try:
        payload = CDCPayload(**raw_payload)
    except Exception as e:
        logger.error(f"[{event_id}] Invalid CDC payload in HITL node: {e}")
        return updated_state({
            "sync_status": "FAILED",
            "error_trace": f"Payload validation error: {e}"
        })

    # ========== SUSPEND EXECUTION ==========
    logger.info(f"[{event_id}] Entering HITL suspension. Awaiting human decision for doc {payload.document_id}.")

    # Optional: fire external webhook asynchronously (stubbed)
    # fire_slack_alert(payload.dict(), event_id)

    # The thread freezes here until an external system resumes it via the LangGraph API.
    human_decision = interrupt({
        "message": "Pending manual approval for high-risk semantic drift.",
        "thread_id": event_id,
        "document_id": payload.document_id,
        "before_text_preview": payload.extracted_text_before[:app_config.log_text_max_length],
        "after_text_preview": payload.extracted_text_after[:app_config.log_text_max_length],
        "evaluation_reasoning": state.get("evaluation_reasoning", ""),
        "timestamp_ms": payload.ts_ms,
    })

    # --- Code resumes ONLY after external resume call ---
    logger.info(f"[{event_id}] Graph resumed with decision: {human_decision}")

    # Safely parse the human decision
    if not isinstance(human_decision, dict):
        logger.error(f"[{event_id}] Invalid HITL response type: {type(human_decision)}. Expected dict.")
        # Fail safe: treat as REJECT to prevent accidental sync.
        return updated_state({
            "semantic_drift_detected": False,
            "sync_status": "SKIPPED"
        })

    action = human_decision.get("action", "REJECT").upper()

    if action == "APPROVE":
        logger.info(f"[{event_id}] Human APPROVED sync. Proceeding to Vector DB.")
        # Keep drift_detected=True so that downstream sync node executes the upsert.
        # Sync node also checks `sync_status`, which we set to PENDING so it runs.
        return updated_state({
            "semantic_drift_detected": True,
            "sync_status": "PENDING",
            "evaluation_reasoning": f"{state.get('evaluation_reasoning', '')} [HITL APPROVED]"
        })
    else:
        logger.info(f"[{event_id}] Human REJECTED sync. Bypassing Vector DB.")
        return updated_state({
            "semantic_drift_detected": False,   # downstream nodes will see no drift
            "sync_status": "SKIPPED",          # sync node will skip
            "evaluation_reasoning": f"{state.get('evaluation_reasoning', '')} [HITL REJECTED]"
        })