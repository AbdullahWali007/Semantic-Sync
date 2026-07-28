# core/config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AppConfig:
    # LLM settings
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_max_tokens: Optional[int] = int(os.getenv("LLM_MAX_TOKENS", "512")) if os.getenv("LLM_MAX_TOKENS") else None
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
    llm_retry_max_attempts: int = int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))
    llm_retry_base_delay_seconds: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
    
    # Token budget
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "100000"))
    
    # Cache TTL (seconds) – 0 disables caching
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Logging / truncation
    log_text_max_length: int = int(os.getenv("LOG_TEXT_MAX_LENGTH", "200"))

    # Fallback behaviour on LLM failure: True = assume drift, False = raise error
    fallback_assume_drift_on_error: bool = os.getenv("FALLBACK_ASSUME_DRIFT", "true").lower() == "true"

    # Tokeniser encoding – override if needed
    tokenizer_encoding: str = os.getenv("TOKENIZER_ENCODING", "cl100k_base")

    # HITL toggle
    hitl_enabled: bool = os.getenv("HITL_ENABLED", "true").lower() == "true"

    # Kafka consumer settings (added for the worker)
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "debezium.public.documents")
    kafka_group_id: str = os.getenv("KAFKA_GROUP_ID", "semantic-sync-engine")