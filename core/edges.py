# core/edges.py
import logging
from typing import Literal
from core.state import SyncState
from core.schemas import CDCPayload

logger = logging.getLogger(__name__)

def router_edge(state: SyncState) -> str:
    """
    Determines the next step after the Debezium payload is parsed.
    """
    mutation_type = state.get("mutation_type")
    event_id = state.get("event_id", "unknown")
    
    if mutation_type in ["INSERT", "DELETE", "LARGE_UPDATE"]:
        # Bypass evaluation entirely. Forced sync required.
        return "execute_vector_sync"
    elif mutation_type == "UPDATE":
        # Standard update requires LLM semantic evaluation
        return "evaluate_semantic_drift"
    else:
        # Catch-all for INVALID or malformed payloads
        logger.warning(f"[{event_id}] Routing to END due to INVALID mutation_type.")
        return "__end__"


def evaluator_edge(state: SyncState) -> str:
    """
    Determines the next step based on the LLM's binary decision.
    Implements the conditional HITL (Human-In-The-Loop) check.
    """
    drift_detected = state.get("semantic_drift_detected")
    event_id = state.get("event_id", "unknown")
    
    # If no drift, terminate the graph immediately.
    if not drift_detected:
        logger.info(f"[{event_id}] No drift detected. Short-circuiting graph.")
        return "__end__"
    
    # --- Business rule: HITL for specific departments ---
    # This can be externalised to config later, but kept as hard-coded for now.
    raw_payload = state.get("cdc_payload", {})
    try:
        payload = CDCPayload(**raw_payload)
        department = payload.after.get("department", "").upper() if payload.after else ""
    except Exception:
        # If payload is invalid, we cannot determine department; proceed to sync (fail safe?)
        logger.warning(f"[{event_id}] Could not parse payload for department check. Proceeding to sync.")
        return "execute_vector_sync"

    if department == "LEGAL":
        logger.info(f"[{event_id}] High-risk semantic drift detected (LEGAL). Routing to HITL.")
        return "human_review_node"
    
    # Standard drift detected. Proceed to sync.
    return "execute_vector_sync"