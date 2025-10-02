"""
EDA Workflow Graph Definition

This module creates and configures the complete LangGraph EDA workflow.
"""

from langgraph.graph import StateGraph, END
from EDA.workflow.eda_workflow_state import EDAWorkflowState
from EDA.workflow.workflow_nodes import (
    univariate_analysis_node,
    hypothesis_generation_node,
    eda_worker_loop_node,
    summarizer_node,
    error_handler_node,
    should_continue_workflow,
    should_retry_after_error
)

def create_eda_workflow() -> StateGraph:
    """Create the complete EDA workflow graph"""
    
    # Initialize workflow
    workflow = StateGraph(EDAWorkflowState)
    
    # Add nodes
    workflow.add_node("univariate_analysis", univariate_analysis_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("eda_worker_loop", eda_worker_loop_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define conditional edges
    workflow.add_conditional_edges(
        "univariate_analysis",
        should_continue_workflow,
        {
            "continue": "hypothesis_generation",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "hypothesis_generation",
        should_continue_workflow,
        {
            "continue": "eda_worker_loop",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "eda_worker_loop",
        should_continue_workflow,
        {
            "continue": "summarizer",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "summarizer",
        should_continue_workflow,
        {
            "continue": END,
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "error_handler",
        should_retry_after_error,
        {
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("univariate_analysis")
    
    # Compile workflow
    return workflow.compile()

def create_simplified_workflow() -> StateGraph:
    """Create a simplified workflow for testing without all agents"""
    
    # Initialize workflow
    workflow = StateGraph(EDAWorkflowState)
    
    # Add only essential nodes for testing
    workflow.add_node("univariate_analysis", univariate_analysis_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define edges
    workflow.add_conditional_edges(
        "univariate_analysis",
        should_continue_workflow,
        {
            "continue": "hypothesis_generation",
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "hypothesis_generation",
        should_continue_workflow,
        {
            "continue": END,
            "error_handler": "error_handler",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "error_handler",
        should_retry_after_error,
        {
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("univariate_analysis")
    
    # Compile workflow
    return workflow.compile()
