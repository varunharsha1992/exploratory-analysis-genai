"""
EDA Workflow Node Functions

This module contains all the node functions for the LangGraph EDA workflow.
"""

import logging
from datetime import datetime
from turtle import update
from typing import Dict, Any, List
from EDA.workflow.eda_workflow_state import EDAWorkflowState
# Import agent functions - these will be imported dynamically to avoid dependency issues
# from EDA.agents.univariate_analysis.univariate_analysis import univariate_analysis_agent
# from EDA.agents.hypothesis_generation.hypothesis_generation import hypothesis_generation_agent
# from EDA.agents.eda_worker_loop.eda_worker_loop import eda_worker_loop_agent
# from EDA.agents.summarizer.summarizer import summarizer_agent

def univariate_analysis_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Execute univariate analysis on target variable"""
    
    try:
        logging.info(f"Starting univariate analysis for target: {state['target_variable']}")
        
        # Update state with current agent
        state["current_agent"] = "univariate_analysis"
        state["execution_status"] = "running"
        
        # Dynamic import to avoid dependency issues
        try:
            from EDA.agents.univariate_analysis.univariate_analysis import univariate_analysis_agent
            # Execute univariate analysis agent
            updated_state = univariate_analysis_agent(state)
            
            # Check if analysis was successful
            if updated_state.get("execution_status") == "completed":
                logging.info("Univariate analysis completed successfully")
                state["univariate_results"] = updated_state.get("univariate_results")
                state["execution_status"] = "completed"
            else:
                logging.error("Univariate analysis failed")
                state["error_messages"] = updated_state.get("error_messages", [])
                state["execution_status"] = "failed"
        except ImportError as ie:
            logging.warning(f"Univariate analysis agent not available: {str(ie)}")
            state["error_messages"].append(f"Univariate analysis agent not available: {str(ie)}")
            state["execution_status"] = "failed"
        
        return state
        
    except Exception as e:
        logging.error(f"Univariate analysis node failed: {str(e)}")
        state["error_messages"].append(f"Univariate analysis failed: {str(e)}")
        state["execution_status"] = "failed"
        return state

def hypothesis_generation_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Generate hypotheses about influencing variables"""
    
    try:
        logging.info(f"Starting hypothesis generation for target: {state['target_variable']}")
        logging.info(f"Number of hypotheses to generate as read by hypothesis generation node: {state['hypothesis_limit']}")

        # Update state with current agent
        state["current_agent"] = "hypothesis_generation"
        state["execution_status"] = "running"
        
        # Dynamic import to avoid dependency issues
        try:
            from EDA.agents.hypothesis_generation.hypothesis_generation import hypothesis_generation_agent
            # Execute hypothesis generation agent
            updated_state = hypothesis_generation_agent(state)
            
            # Check if generation was successful
            if updated_state.get("execution_status") == "completed":
                logging.info("Hypothesis generation completed successfully")
                state["generated_hypotheses"] = updated_state.get("generated_hypotheses")
                state["execution_status"] = "completed"
            else:
                logging.error("Hypothesis generation failed")
                state["error_messages"] = updated_state.get("error_messages", [])
                state["execution_status"] = "failed"
        except ImportError as ie:
            logging.warning(f"Hypothesis generation agent not available: {str(ie)}")
            state["error_messages"].append(f"Hypothesis generation agent not available: {str(ie)}")
            state["execution_status"] = "failed"
        
        return state
        
    except Exception as e:
        logging.error(f"Hypothesis generation node failed: {str(e)}")
        state["error_messages"].append(f"Hypothesis generation failed: {str(e)}")
        state["execution_status"] = "failed"
        return state

def eda_worker_loop_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Orchestrate parallel hypothesis testing"""
    
    try:
        logging.info("Starting EDA worker loop for hypothesis testing")
        
        # Update state with current agent
        state["current_agent"] = "eda_worker_loop"
        state["execution_status"] = "running"
        
        # Dynamic import to avoid dependency issues
        try:
            from EDA.agents.eda_worker_loop.eda_worker_loop import eda_worker_loop_agent
            # Execute EDA worker loop agent
            updated_state = eda_worker_loop_agent(state)
            
            # Check if testing was successful
            if updated_state.get("execution_status") == "completed":
                logging.info("EDA worker loop completed successfully")
                print(f"Successfully completed EDA worker loop")
            else:
                logging.error("EDA worker loop failed")
        except ImportError as ie:
            logging.warning(f"EDA worker loop agent not available: {str(ie)}")
        
        return updated_state
        
    except Exception as e:
        logging.error(f"EDA worker loop node failed: {str(e)}")
        state["error_messages"].append(f"EDA worker loop failed: {str(e)}")
        state["execution_status"] = "failed"
        return state

def summarizer_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Generate final summary and recommendations"""
    
    try:
        logging.info("Starting summarization and recommendation generation")
        
        # Update state with current agent
        state["current_agent"] = "summarizer"
        state["execution_status"] = "running"
        
        # Dynamic import to avoid dependency issues
        try:
            from EDA.agents.summarizer.summarizer import summarizer_agent
            # Execute summarizer agent
            updated_state = summarizer_agent(state)
            
            # Check if summarization was successful
            if updated_state.get("execution_status") == "completed":
                logging.info("Summarization completed successfully")
            else:
                logging.error("Summarization failed")
        except ImportError as ie:
            logging.warning(f"Summarizer agent not available: {str(ie)}")
        
        return updated_state
        
    except Exception as e:
        logging.error(f"Summarizer node failed: {str(e)}")
        state["error_messages"].append(f"Summarization failed: {str(e)}")
        state["execution_status"] = "failed"
        return state

def error_handler_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Handle workflow errors and provide recovery options"""
    
    try:
        logging.error(f"Error handler activated for agent: {state['current_agent']}")
        
        # Create error summary
        error_summary = {
            "failed_agent": state["current_agent"],
            "error_messages": state["error_messages"],
            "recovery_options": [],
            "partial_results": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect partial results
        if state.get("univariate_results"):
            error_summary["partial_results"]["univariate_analysis"] = state["univariate_results"]
        
        if state.get("generated_hypotheses"):
            error_summary["partial_results"]["hypotheses"] = state["generated_hypotheses"]
        
        if state.get("hypothesis_testing_results"):
            error_summary["partial_results"]["hypothesis_testing"] = state["hypothesis_testing_results"]
        
        # Provide recovery options based on failed agent
        if state["current_agent"] == "univariate_analysis":
            error_summary["recovery_options"] = [
                "Retry with different data sampling",
                "Skip univariate analysis and proceed with basic hypotheses",
                "Use cached univariate results if available"
            ]
        elif state["current_agent"] == "hypothesis_generation":
            error_summary["recovery_options"] = [
                "Retry with reduced hypothesis limit",
                "Use manual hypothesis specification",
                "Proceed with basic correlation analysis"
            ]
        elif state["current_agent"] == "eda_worker_loop":
            error_summary["recovery_options"] = [
                "Retry with reduced worker count",
                "Process hypotheses sequentially",
                "Skip failed hypotheses and continue"
            ]
        elif state["current_agent"] == "summarizer":
            error_summary["recovery_options"] = [
                "Generate basic summary from available results",
                "Retry with simplified summarization",
                "Return partial results"
            ]
        
        # Update state
        state["final_summary"] = error_summary
        state["execution_status"] = "error_handled"
        state["end_time"] = datetime.now()
        
        return state
        
    except Exception as e:
        logging.error(f"Error handler failed: {str(e)}")
        state["error_messages"].append(f"Error handler failed: {str(e)}")
        state["execution_status"] = "critical_error"
        return state

def should_continue_workflow(state: EDAWorkflowState) -> str:
    """Determine if workflow should continue based on execution status"""
    
    if state["execution_status"] == "failed":
        return "error_handler"
    elif state["current_agent"] == "summarizer" and state["execution_status"] == "completed":
        return "end"
    else:
        return "continue"

def should_retry_after_error(state: EDAWorkflowState) -> str:
    """Determine if workflow should retry after error handling"""
    
    # For now, always end after error handling
    # In a production system, this could implement retry logic
    return "end"
