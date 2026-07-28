# core/schemas.py
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, Dict, Any

class CDCPayload(BaseModel):
    """
    Strict validation schema for Debezium logical replication envelopes.
    """
    op: str = Field(..., description="Operation type: 'c' (create), 'u' (update), 'd' (delete)")
    ts_ms: int = Field(..., description="Transaction timestamp for Optimistic Concurrency Control")
    before: Optional[Dict[str, Any]] = Field(None, description="Row state before mutation")
    after: Optional[Dict[str, Any]] = Field(None, description="Row state after mutation")

    @field_validator("op")
    @classmethod
    def validate_op(cls, v: str) -> str:
        if v not in {"c", "u", "d"}:
            raise ValueError(f"Invalid operation '{v}'. Must be one of 'c', 'u', 'd'.")
        return v

    @field_validator("ts_ms")
    @classmethod
    def validate_ts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"ts_ms must be positive, got {v}")
        return v

    @property
    def document_id(self) -> str:
        """Extracts primary key safely depending on operation type."""
        if self.op == 'd':
            if not self.before or "id" not in self.before:
                raise ValueError("Delete operation requires 'id' in 'before'.")
            return str(self.before["id"])
        if self.after and "id" in self.after:
            return str(self.after["id"])
        raise ValueError("Malformed CDC payload: Missing document ID in 'after' for non-delete.")

    @property
    def extracted_text_before(self) -> str:
        """Extracts the semantic content payload for evaluation."""
        return self.before.get("content", "") if self.before else ""

    @property
    def extracted_text_after(self) -> str:
        return self.after.get("content", "") if self.after else ""