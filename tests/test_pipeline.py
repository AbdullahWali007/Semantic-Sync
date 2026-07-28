# tests/test_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from core.config import AppConfig
from core.graph import build_semantic_sync_graph
from core.state import SyncState
from services.llm_service import LLMService


@pytest.fixture
def app_config():
    config = AppConfig()
    # Disable HITL by default for most tests
    config.hitl_enabled = False
    config.llm_retry_max_attempts = 1
    config.max_context_tokens = 1000
    return config


@pytest.fixture
def mock_llm_service(app_config):
    service = MagicMock(spec=LLMService)
    # Default: no drift
    service.evaluate_drift = AsyncMock(return_value=(False, "No drift"))
    service.count_tokens = MagicMock(return_value=10)
    return service


@pytest.fixture
def mock_embedder():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_qdrant():
    mock = MagicMock()
    mock.upsert.return_value = None
    mock.delete.return_value = None
    return mock


@pytest.fixture
def compiled_graph(app_config, mock_llm_service, mock_embedder, mock_qdrant):
    """Compile graph with in‑memory checkpointer and injected mocks."""
    workflow = build_semantic_sync_graph()
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # Return a wrapper that builds config and invokes
    async def run(initial_state: dict, thread_id: str):
        config = {
            "configurable": {
                "thread_id": thread_id,
                "app_config": app_config,
                "llm_service": mock_llm_service,
                "embedding_client": mock_embedder,
                "qdrant_client": mock_qdrant,
                "collection_name": "test_collection",
            }
        }
        return await graph.ainvoke(initial_state, config=config)

    return run


# ---------- Test Cases ----------

@pytest.mark.asyncio
async def test_insert(compiled_graph):
    payload = {
        "op": "c",
        "ts_ms": 123,
        "before": None,
        "after": {"id": "1", "content": "New doc", "department": "ENG"}
    }
    state = {
        "event_id": "evt1",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread1")
    assert result["mutation_type"] == "INSERT"
    assert result["sync_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_update_no_drift(compiled_graph, mock_llm_service):
    mock_llm_service.evaluate_drift = AsyncMock(return_value=(False, "Cosmetic only"))
    payload = {
        "op": "u",
        "ts_ms": 124,
        "before": {"id": "2", "content": "Hello", "department": "ENG"},
        "after": {"id": "2", "content": "Hello  ", "department": "ENG"}
    }
    state = {
        "event_id": "evt2",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread2")
    assert result["mutation_type"] == "UPDATE"
    assert result["semantic_drift_detected"] is False
    # sync should not have been called (graph ends early)
    assert result.get("sync_status") != "COMPLETED"


@pytest.mark.asyncio
async def test_update_with_drift(compiled_graph, mock_llm_service):
    mock_llm_service.evaluate_drift = AsyncMock(return_value=(True, "Price changed"))
    payload = {
        "op": "u",
        "ts_ms": 125,
        "before": {"id": "3", "content": "Price $100", "department": "ENG"},
        "after": {"id": "3", "content": "Price $200", "department": "ENG"}
    }
    state = {
        "event_id": "evt3",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread3")
    assert result["mutation_type"] == "UPDATE"
    assert result["semantic_drift_detected"] is True
    assert result["sync_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_delete(compiled_graph):
    payload = {
        "op": "d",
        "ts_ms": 126,
        "before": {"id": "4", "content": "Delete me"},
        "after": None
    }
    state = {
        "event_id": "evt4",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread4")
    assert result["mutation_type"] == "DELETE"
    assert result["sync_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_legal_hitl(compiled_graph, mock_llm_service, app_config):
    # Enable HITL for this test
    app_config.hitl_enabled = True
    mock_llm_service.evaluate_drift = AsyncMock(return_value=(True, "Legal change"))
    payload = {
        "op": "u",
        "ts_ms": 127,
        "before": {"id": "5", "content": "Old clause", "department": "LEGAL"},
        "after": {"id": "5", "content": "New clause", "department": "LEGAL"}
    }
    state = {
        "event_id": "evt5",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    # We need to capture the interrupt
    try:
        await compiled_graph(state, "thread5")
        pytest.fail("Expected GraphInterrupt")
    except GraphInterrupt:
        # Resume with APPROVE
        config = {
            "configurable": {
                "thread_id": "thread5",
                "app_config": app_config,
                "llm_service": mock_llm_service,
                "embedding_client": None,  # not needed for resume
                "qdrant_client": None,
                "collection_name": "test_collection",
            }
        }
        # We need to invoke with Command on the compiled graph directly (not the wrapper)
        # So we rebuild the graph and run separately, or we use the graph instance from the fixture.
        # Simpler: we re‑fetch the graph and resume.
        workflow = build_semantic_sync_graph()
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)
        # Re‑create the same state and config for resume
        # Actually, the interrupt already stored the state; we just resume.
        final_state = await graph.ainvoke(
            Command(resume={"action": "APPROVE"}),
            config=config
        )
        assert final_state["sync_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_large_update(compiled_graph, app_config):
    # Force token threshold
    app_config.max_context_tokens = 1
    payload = {
        "op": "u",
        "ts_ms": 128,
        "before": {"id": "6", "content": "a" * 1000, "department": "ENG"},
        "after": {"id": "6", "content": "b" * 1000, "department": "ENG"}
    }
    state = {
        "event_id": "evt6",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread6")
    assert result["mutation_type"] == "LARGE_UPDATE"
    assert result["sync_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_invalid_payload(compiled_graph):
    payload = {"op": "x", "ts_ms": 129}
    state = {
        "event_id": "evt7",
        "cdc_payload": payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
    }
    result = await compiled_graph(state, "thread7")
    assert result["mutation_type"] == "INVALID"
    assert result["sync_status"] == "FAILED"