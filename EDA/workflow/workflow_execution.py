"""
EDA Workflow Execution

This module handles the execution of the EDA workflow with monitoring and error handling.
"""

import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from EDA.workflow.eda_workflow_state import EDAWorkflowState
from EDA.workflow.eda_workflow import create_eda_workflow, create_simplified_workflow
import pandas as pd
from EDA.tools import intugle_agent_tools

def dump_hypotheses_to_file(hypotheses: list, target_variable: str, workflow_type: str = "full") -> str:
    """Dump hypotheses to a file in the hypothesis_outputs folder"""
    
    # Create hypothesis_outputs directory if it doesn't exist
    output_dir = "hypothesis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hypotheses_{target_variable}_{workflow_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Create output data structure
        output_data = {
            "metadata": {
                "target_variable": target_variable,
                "workflow_type": workflow_type,
                "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_hypotheses": len(hypotheses)
            },
            "hypotheses": hypotheses
        }
        
        # Write JSON to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Hypotheses dumped to: {filepath}")
        return filepath
        
    except Exception as e:
        logging.error(f"Failed to dump hypotheses: {str(e)}")
        return ""

async def execute_eda_workflow(
    target_variable: str,
    eda_request: str,
    domain_context: str = "",
    hypothesis_limit: int = 1,
    config: Optional[Dict[str, Any]] = None,
    kb: Optional[Any] = None,
    target_data_path: str = "",
    simplified: bool = False,
    full_data_path: str = "",
    target_file_name: str = "",
    files_to_process: list = []
) -> Dict[str, Any]:
    """Execute the complete EDA workflow"""
    
    try:
        intugle_tools = intugle_agent_tools.IntugleAgentTools(full_data_path=full_data_path, files_to_process=files_to_process)
        # Initialize workflow
        if simplified:
            workflow = create_simplified_workflow()
        else:
            workflow = create_eda_workflow()

        target_data = pd.read_csv(os.path.join(full_data_path, target_file_name))    
        # Create initial state
        if intugle_tools.is_available():
            initial_state = EDAWorkflowState(
                target_variable=target_variable,
                eda_request=eda_request,
                domain_context=domain_context,
                hypothesis_limit=hypothesis_limit,
                univariate_results=None,
                generated_hypotheses=None,
                hypothesis_testing_results=None,
                final_summary=None,
                execution_status="initialized",
                error_messages=[],
                performance_metrics={},
                current_agent="",
                start_time=datetime.now(),
                end_time=None,
                data=target_data,
                config=config or {},
                kb=kb,
                intugle_tools=intugle_tools
            )
            
            logging.info(f"Starting EDA workflow for target: {target_variable}")
            logging.info(f"Number of hypotheses to generate as read by workflow execution step: {hypothesis_limit}")
            # Execute workflow
            final_state = await workflow.ainvoke(initial_state)
            # Dump hypotheses to file if they exist
            hypotheses_file = ""
            if final_state.get("generated_hypotheses"):
                workflow_type = "simplified" if simplified else "full"
                hypotheses_file = dump_hypotheses_to_file(
                    final_state["generated_hypotheses"], 
                    target_variable, 
                    workflow_type
                )
            
            # Calculate execution metrics
            execution_time = None
            if final_state.get("start_time") and final_state.get("end_time"):
                execution_time = (final_state["end_time"] - final_state["start_time"]).total_seconds()
            
            # Prepare result
            result = {
                "status": "success" if final_state["execution_status"] == "completed" else "failed",
                "generated_hypotheses": final_state.get("generated_hypotheses", []),
                "eda_analysis_results": final_state.get("hypothesis_testing_results", []),
                "results": final_state.get("final_summary", {}),
                "execution_metrics": {
                    "execution_time_seconds": execution_time,
                    "workflow_path": final_state.get("current_agent", ""),
                    "error_count": len(final_state.get("error_messages", [])),
                    "hypotheses_generated": len(final_state.get("generated_hypotheses", [])),
                    "performance_metrics": final_state.get("performance_metrics", {}),
                    "hypotheses_file": hypotheses_file
                },
                "workflow_path": final_state.get("current_agent", ""),
                "error_messages": final_state.get("error_messages", [])
            }
            print(f"Final Result: {result}")
            # For simplified workflow, create a basic summary if no final_summary exists
            if not result["results"] and final_state.get("generated_hypotheses"):
                result["results"] = {
                    "workflow_type": "simplified",
                    "hypotheses_generated": len(final_state.get("generated_hypotheses", [])),
                    "univariate_analysis_completed": final_state.get("univariate_results") is not None,
                    "hypothesis_generation_completed": final_state.get("generated_hypotheses") is not None,
                    "summary": f"Simplified EDA workflow completed successfully. Generated {len(final_state.get('generated_hypotheses', []))} hypotheses."
                }
            
            logging.info(f"EDA workflow completed with status: {result['status']}")
            return result
        else:
            logging.error("Intugle tools are not available")
            return {
                "status": "failed",
                "error": "Intugle tools are not available",
                "partial_results": {},
                "error_messages": ["Intugle tools are not available"],
                "execution_metrics": {
                    "execution_time_seconds": None,
                    "workflow_path": "failed",
                    "error_count": 1
                }
            }
    except Exception as e:
        logging.error(f"EDA workflow execution failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "partial_results": {},
            "error_messages": [str(e)],
            "execution_metrics": {
                "execution_time_seconds": None,
                "workflow_path": "failed",
                "error_count": 1
            }
        }

def monitor_workflow_progress(state: EDAWorkflowState) -> Dict[str, Any]:
    """Monitor workflow execution progress"""
    
    progress = {
        "current_step": state["current_agent"],
        "completed_steps": [],
        "remaining_steps": [],
        "progress_percentage": 0,
        "estimated_completion": None,
        "execution_status": state["execution_status"]
    }
    
    # Define workflow steps
    workflow_steps = [
        "univariate_analysis",
        "hypothesis_generation", 
        "eda_worker_loop",
        "summarizer"
    ]
    
    current_index = workflow_steps.index(state["current_agent"]) if state["current_agent"] in workflow_steps else 0
    
    progress["completed_steps"] = workflow_steps[:current_index]
    progress["remaining_steps"] = workflow_steps[current_index:]
    progress["progress_percentage"] = (current_index / len(workflow_steps)) * 100
    
    # Estimate completion time
    if state.get("start_time"):
        elapsed = (datetime.now() - state["start_time"]).total_seconds()
        if current_index > 0:
            avg_time_per_step = elapsed / current_index
            remaining_steps = len(progress["remaining_steps"])
            estimated_remaining = avg_time_per_step * remaining_steps
            progress["estimated_completion"] = datetime.now().timestamp() + estimated_remaining
    
    return progress

async def execute_workflow_with_monitoring(
    target_variable: str,
    eda_request: str,
    domain_context: str = "",
    hypothesis_limit: int = 10,
    config: Optional[Dict[str, Any]] = None,
    kb: Optional[Any] = None,
    simplified: bool = False,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Execute workflow with progress monitoring"""
    
    try:
        # Initialize workflow
        if simplified:
            workflow = create_simplified_workflow()
        else:
            workflow = create_eda_workflow()
        
        # Create initial state
        initial_state = EDAWorkflowState(
            target_variable=target_variable,
            eda_request=eda_request,
            domain_context=domain_context,
            hypothesis_limit=hypothesis_limit,
            univariate_results=None,
            generated_hypotheses=None,
            hypothesis_testing_results=None,
            final_summary=None,
            execution_status="initialized",
            error_messages=[],
            performance_metrics={},
            current_agent="",
            start_time=datetime.now(),
            end_time=None,
            config=config or {},
            kb=kb
        )
        
        # Execute workflow with monitoring
        final_state = await workflow.ainvoke(initial_state)
        
        # Call progress callback if provided
        if progress_callback:
            progress = monitor_workflow_progress(final_state)
            progress_callback(progress)
        
        # Dump hypotheses to file if they exist
        hypotheses_file = ""
        if final_state.get("generated_hypotheses"):
            workflow_type = "simplified" if simplified else "full"
            hypotheses_file = dump_hypotheses_to_file(
                final_state["generated_hypotheses"], 
                target_variable, 
                workflow_type
            )
        
        # Calculate execution metrics
        execution_time = None
        if final_state.get("start_time") and final_state.get("end_time"):
            execution_time = (final_state["end_time"] - final_state["start_time"]).total_seconds()
        
        # Prepare result
        result = {
            "status": "success" if final_state["execution_status"] == "completed" else "failed",
            "results": final_state.get("final_summary", {}),
            "execution_metrics": {
                "execution_time_seconds": execution_time,
                "workflow_path": final_state.get("current_agent", ""),
                "error_count": len(final_state.get("error_messages", [])),
                "hypotheses_generated": len(final_state.get("generated_hypotheses", [])),
                "performance_metrics": final_state.get("performance_metrics", {}),
                "hypotheses_file": hypotheses_file
            },
            "workflow_path": final_state.get("current_agent", ""),
            "error_messages": final_state.get("error_messages", [])
        }
        
        return result
        
    except Exception as e:
        logging.error(f"EDA workflow execution with monitoring failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "partial_results": {},
            "error_messages": [str(e)],
            "execution_metrics": {
                "execution_time_seconds": None,
                "workflow_path": "failed",
                "error_count": 1
            }
        }
