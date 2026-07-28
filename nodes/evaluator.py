# nodes/evaluator.py
import logging
import time
from typing import Optional, cast
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from core.state import SyncState
from core.schemas import CDCPayload
from core.config import AppConfig
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SemanticDriftEvaluation(BaseModel):
    """
    Strict output schema enforced on the LLM.
    """
    drift_detected: bool = Field(
        ...,
        description="True if semantic meaning, facts, or instructions changed. False otherwise."
    )
    reasoning: str = Field(
        ...,
        description="A concise, one-sentence justification for the boolean decision."
    )


def evaluate_semantic_drift(state: SyncState, config: RunnableConfig) -> SyncState:
    """
    Node 2: Compares BEFORE and AFTER states to detect semantic drift.
    Updates the state with a boolean flag and reasoning.
    """
    event_id = state.get("event_id")
    raw_payload = state.get("cdc_payload", {})

    # Helper to return a full SyncState by merging updates into the original state
    def updated_state(updates: dict) -> SyncState:
        new_state = dict(state)          # shallow copy includes all required keys
        new_state.update(updates)
        return cast(SyncState, new_state)

    # Re-hydrate and validate
    try:
        payload = CDCPayload(**raw_payload)
    except Exception as e:
        logger.error(f"[{event_id}] Invalid CDC payload: {e}")
        return updated_state({
            "semantic_drift_detected": True,  # fail safe
            "error_trace": f"Invalid payload: {e}",
            "sync_status": "FAILED"
        })

    # Only evaluate for UPDATE operations; for others, we don't call this node.
    if payload.op != 'u':
        logger.warning(f"[{event_id}] Evaluator called on non-Update operation: {payload.op}. Skipping.")
        return state   # no changes

    text_before = payload.extracted_text_before
    text_after = payload.extracted_text_after

    # Extract dependencies from config
    app_config: Optional[AppConfig] = config.get("configurable", {}).get("app_config")
    if not app_config:
        raise RuntimeError(f"[{event_id}] 'app_config' missing from RunnableConfig.")

    llm_service: Optional[LLMService] = config.get("configurable", {}).get("llm_service")
    if not llm_service:
        raise RuntimeError(f"[{event_id}] 'llm_service' missing from RunnableConfig.")

    # Check token budget
    if not llm_service.check_token_budget(text_before, text_after):
        logger.warning(f"[{event_id}] Token budget exceeded. Forcing drift detection (large update).")
        return updated_state({
            "semantic_drift_detected": True,
            "evaluation_reasoning": "Token budget exceeded; forced sync.",
            "sync_status": "COMPLETED"
        })

    # Perform evaluation with retries
    try:
        start_time = time.perf_counter()
        drift, reasoning = llm_service.evaluate_drift(text_before, text_after, event_id)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return updated_state({
            "semantic_drift_detected": drift,
            "evaluation_reasoning": reasoning,
            "evaluation_latency_ms": latency_ms,
            "sync_status": "COMPLETED" if not drift else "PENDING"
        })

    except Exception as e:
        logger.error(f"[{event_id}] Evaluation failed after retries: {e}")
        # Apply fallback strategy
        if app_config.fallback_assume_drift_on_error:
            return updated_state({
                "semantic_drift_detected": True,
                "error_trace": f"Evaluator fallback (drift=True) due to: {str(e)}",
                "sync_status": "FAILED"
            })
        else:
            # Raise to let the graph handle it (e.g., retry entire node)
            raise RuntimeError(f"[{event_id}] Evaluation failed and fallback is disabled.") from e