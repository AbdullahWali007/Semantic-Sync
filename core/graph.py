# core/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.state import SyncState
from core.edges import router_edge, evaluator_edge

# Import Nodes
from nodes.router import route_event
from nodes.evaluator import evaluate_semantic_drift
from nodes.sync import execute_vector_sync
from nodes.hitl import human_review_node

def build_semantic_sync_graph():
    """
    Compiles the LangGraph StateMachine.
    """
    workflow = StateGraph(SyncState)

    # 1. Add Core Nodes
    workflow.add_node("route_event", route_event)
    workflow.add_node("evaluate_semantic_drift", evaluate_semantic_drift)
    workflow.add_node("human_review_node", human_review_node)
    workflow.add_node("execute_vector_sync", execute_vector_sync)

    # 2. Define the Entry Point
    workflow.add_edge(START, "route_event")

    # 3. Apply Conditional Routing from the Router Node
    workflow.add_conditional_edges(
        "route_event",
        router_edge,
        {
            "execute_vector_sync": "execute_vector_sync",
            "evaluate_semantic_drift": "evaluate_semantic_drift",
            "__end__": END
        }
    )

    # 4. Apply Conditional Routing from the Evaluator Node
    workflow.add_conditional_edges(
        "evaluate_semantic_drift",
        evaluator_edge,
        {
            "human_review_node": "human_review_node",
            "execute_vector_sync": "execute_vector_sync",
            "__end__": END
        }
    )

    # 5. Define Deterministic Exits
    # Once human review is complete (or approved), route back to sync
    workflow.add_edge("human_review_node", "execute_vector_sync")
    # Once sync is complete, terminate
    workflow.add_edge("execute_vector_sync", END)

    return workflow