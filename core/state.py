# core/state.py
from typing import TypedDict, Optional, Literal, Dict, Any

class SyncState(TypedDict):
    """
    The shared state dictionary for the LangGraph execution thread.
    """
    event_id: str
    cdc_payload: Dict[str, Any]          # Raw JSON injected by Kafka
    mutation_type: Optional[Literal["INSERT", "UPDATE", "DELETE", "LARGE_UPDATE", "INVALID"]]
    semantic_drift_detected: Optional[bool]
    sync_status: Literal["PENDING", "COMPLETED", "SKIPPED", "FAILED", "PENDING_HITL"]
    error_trace: Optional[str]
    # Additional fields for observability (optional)
    evaluation_reasoning: Optional[str]   # store the LLM's reasoning
    evaluation_latency_ms: Optional[float]