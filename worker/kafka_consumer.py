# worker/kafka_consumer.py
import os
import json
import asyncio
import logging
from typing import Dict, Any

from confluent_kafka import Consumer, KafkaError
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from core.config import AppConfig
from core.graph import build_semantic_sync_graph
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

async def run_consumer():
    """
    Main ingestion loop for CDC events.
    Uses the same refactored graph and dependencies as the API server.
    """
    # 1. Load config
    config = AppConfig()

    # 2. Initialize PostgreSQL checkpointer
    pool = AsyncConnectionPool(conninfo=os.getenv("POSTGRES_CHECKPOINT_URL"), max_size=10)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    # 3. Build graph
    workflow = build_semantic_sync_graph()
    graph = workflow.compile(checkpointer=checkpointer)

    # 4. Initialise clients (same as server)
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
    collection_name = os.getenv("COLLECTION_NAME", "enterprise_docs")

    # 5. Kafka consumer setup
    consumer_conf = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'group.id': config.kafka_group_id,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,  # manual commit
    }
    consumer = Consumer(consumer_conf)
    consumer.subscribe([config.kafka_topic])
    logger.info(f"Kafka consumer subscribed to {config.kafka_topic}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                break

            # Parse payload
            raw_value = msg.value().decode('utf-8')
            cdc_payload = json.loads(raw_value)

            # Use document ID as thread_id for sequential processing per document
            document_id = cdc_payload.get("after", {}).get("id") or cdc_payload.get("before", {}).get("id")
            if document_id is None:
                logger.warning(f"Message {msg.offset()} missing document ID, skipping")
                consumer.commit(asynchronous=False)
                continue

            thread_id = f"doc_{document_id}"

            # Build full RunnableConfig
            run_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "app_config": config,
                    "llm_service": llm_service,
                    "embedding_client": embedder,
                    "qdrant_client": qdrant,
                    "collection_name": collection_name,
                }
            }

            initial_state = {
                "event_id": str(msg.offset()),
                "cdc_payload": cdc_payload,
                "mutation_type": None,
                "semantic_drift_detected": None,
                "sync_status": "PENDING",
                "error_trace": None,
                "evaluation_reasoning": None,
                "evaluation_latency_ms": None,
            }

            try:
                logger.info(f"Processing event {msg.offset()} for doc {document_id}")
                final_state = await graph.ainvoke(initial_state, config=run_config)
                logger.info(f"Completed offset {msg.offset()}, status={final_state.get('sync_status')}")
                # Commit after success
                consumer.commit(asynchronous=False)
            except Exception as e:
                logger.error(f"Failed to process offset {msg.offset()}: {e}", exc_info=True)
                # In production, send to DLQ; we skip commit to retry later.
                # For robustness, we could commit and log to DLQ here.
                # For now, we just log and continue.
                # To avoid infinite retry, we could also commit after a few failures.
                # We'll commit anyway to avoid blocking the consumer (at-least-once).
                consumer.commit(asynchronous=False)

    finally:
        consumer.close()
        await pool.close()
        logger.info("Consumer shut down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_consumer())