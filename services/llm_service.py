# services/llm_service.py
import logging
import time
import hashlib
from typing import Optional, Dict, Tuple, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import tiktoken

from core.config import AppConfig
from core.prompts import build_evaluator_messages
from nodes.evaluator import SemanticDriftEvaluation

logger = logging.getLogger(__name__)


class LLMService:
    """
    Encapsulates LLM interaction with retry, caching, and token counting.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config: AppConfig,
        cache: Optional[Dict[str, Tuple[float, bool, str]]] = None
    ):
        self.llm = llm
        self.config = config
        self.cache = cache or {}

        self._tokenizer = None
        try:
            self._tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
        except Exception as e:
            logger.warning(
                f"Failed to load tiktoken encoding '{config.tokenizer_encoding}': {e}. "
                "Falling back to character count."
            )
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken; fallback to len(text)//4 (rough)."""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        return len(text) // 4

    def _get_cache_key(self, before: str, after: str) -> str:
        return hashlib.md5(f"{before}||{after}".encode("utf-8")).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        if self.config.cache_ttl_seconds <= 0:
            return False
        return (time.time() - timestamp) < self.config.cache_ttl_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def evaluate_drift(self, before_text: str, after_text: str, event_id: str) -> Tuple[bool, str]:
        # 1. Check cache
        cache_key = self._get_cache_key(before_text, after_text)
        cached = self.cache.get(cache_key)
        if cached and self._is_cache_valid(cached[0]):
            logger.info(f"[{event_id}] Cache hit for evaluation.")
            return cached[1], cached[2]

        # 2. Prepare prompt
        messages = build_evaluator_messages(before_text, after_text)
        prompt = ChatPromptTemplate.from_messages(messages)

        # 3. Bind structured output
        structured_llm = self.llm.with_structured_output(SemanticDriftEvaluation)

        # 4. Invoke with timeout – cast config to satisfy type checker
        start = time.perf_counter()
        try:
            # The returned value is either a dict or the model instance;
            # with_structured_output with a Pydantic model returns the instance.
            result = structured_llm.invoke(
                prompt.format_prompt(),
                config=cast(RunnableConfig, {"timeout": self.config.llm_timeout_seconds})
            )
            # Ensure it's our expected type (cast to the model)
            evaluation: SemanticDriftEvaluation = cast(SemanticDriftEvaluation, result)
        except Exception as e:
            logger.error(f"[{event_id}] LLM invocation failed: {e}")
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        # 5. Cache the result
        self.cache[cache_key] = (time.time(), evaluation.drift_detected, evaluation.reasoning)

        logger.info(
            f"[{event_id}] Drift evaluation: {evaluation.drift_detected} "
            f"(latency={latency_ms:.1f}ms) | Reason: {evaluation.reasoning}"
        )

        return evaluation.drift_detected, evaluation.reasoning

    def check_token_budget(self, before_text: str, after_text: str) -> bool:
        total = self.count_tokens(before_text) + self.count_tokens(after_text)
        return total <= self.config.max_context_tokens