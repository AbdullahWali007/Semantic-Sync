# api/server.py
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from core.config import AppConfig
from core.graph import build_semantic_sync_graph
from services.llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all infrastructure connections and graph once."""
    config = AppConfig()
    
    # Validate required env vars
    required = {
        "POSTGRES_CHECKPOINT_URL": os.getenv("POSTGRES_CHECKPOINT_URL"),
        "EVALUATOR_MODEL": os.getenv("EVALUATOR_MODEL"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
        "VECTOR_DB_URL": os.getenv("VECTOR_DB_URL"),
        "VECTOR_DB_API_KEY": os.getenv("VECTOR_DB_API_KEY"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    # PostgreSQL checkpointer
    pool = AsyncConnectionPool(conninfo=os.getenv("POSTGRES_CHECKPOINT_URL"), max_size=20, timeout=30.0)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    logger.info("PostgreSQL checkpointer ready.")

    # Build graph
    workflow = build_semantic_sync_graph()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Graph compiled.")

    # Clients
    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
        timeout=config.llm_timeout_seconds,
    )
    embedder = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    qdrant = QdrantClient(
        url=os.getenv("VECTOR_DB_URL"),
        api_key=os.getenv("VECTOR_DB_API_KEY")
    )
    llm_service = LLMService(llm=llm, config=config)

    app_state["graph"] = compiled_graph
    app_state["app_config"] = config
    app_state["llm_service"] = llm_service
    app_state["embedding_client"] = embedder
    app_state["qdrant_client"] = qdrant
    app_state["collection_name"] = os.getenv("COLLECTION_NAME", "enterprise_docs")

    logger.info("All clients ready.")
    yield
    await pool.close()
    logger.info("Server shutdown.")

app = FastAPI(lifespan=lifespan)


# ----- Pydantic models -----
class ResumePayload(BaseModel):
    action: str  # "APPROVE" or "REJECT"

class EventIngestRequest(BaseModel):
    op: str
    ts_ms: int
    before: Dict[str, Any] | None = None
    after: Dict[str, Any] | None = None


# ----- Background task for graph invocation -----
async def run_graph_for_event(event_payload: Dict[str, Any], thread_id: str):
    graph = app_state["graph"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "app_config": app_state["app_config"],
            "llm_service": app_state["llm_service"],
            "embedding_client": app_state["embedding_client"],
            "qdrant_client": app_state["qdrant_client"],
            "collection_name": app_state["collection_name"],
        }
    }
    initial_state = {
        "event_id": thread_id,
        "cdc_payload": event_payload,
        "mutation_type": None,
        "semantic_drift_detected": None,
        "sync_status": "PENDING",
        "error_trace": None,
        "evaluation_reasoning": None,
        "evaluation_latency_ms": None,
    }
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        logger.info(f"[{thread_id}] Graph completed with status: {final_state.get('sync_status')}")
    except Exception as e:
        logger.error(f"[{thread_id}] Graph execution failed: {e}", exc_info=True)


# ----- Endpoints -----
@app.post("/api/v1/event")
async def ingest_event(request: EventIngestRequest, background_tasks: BackgroundTasks):
    thread_id = str(uuid.uuid4())
    logger.info(f"Received event op={request.op}, thread={thread_id}")
    if request.op not in ("c", "u", "d"):
        raise HTTPException(400, "Invalid op")
    if request.op == "d" and not request.before:
        raise HTTPException(400, "Delete requires 'before'")
    if request.op in ("c", "u") and not request.after:
        raise HTTPException(400, "Create/Update requires 'after'")

    background_tasks.add_task(run_graph_for_event, request.dict(), thread_id)
    return {"status": "accepted", "thread_id": thread_id}

@app.post("/api/v1/graph/resume/{thread_id}")
async def resume_graph(thread_id: str, payload: ResumePayload):
    graph = app_state["graph"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "app_config": app_state["app_config"],
            "llm_service": app_state["llm_service"],
            "embedding_client": app_state["embedding_client"],
            "qdrant_client": app_state["qdrant_client"],
            "collection_name": app_state["collection_name"],
        }
    }
    try:
        await graph.ainvoke(Command(resume={"action": payload.action}), config=config)
        logger.info(f"Thread {thread_id} resumed with {payload.action}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}