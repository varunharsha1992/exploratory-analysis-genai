"""
EDA Workflow State Definition

This module defines the state structure for the LangGraph EDA workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class EDAWorkflowState(TypedDict):
    """State structure for the EDA workflow"""
    
    # Input parameters
    target_variable: str
    eda_request: str
    domain_context: str
    hypothesis_limit: int
    data: Optional[pd.DataFrame]
    
    # Agent outputs
    univariate_results: Optional[Dict[str, Any]]
    generated_hypotheses: Optional[List[Dict[str, Any]]]
    eda_analysis_results: Optional[Dict[str, Any]]
    final_summary: Optional[Dict[str, Any]]
    eda_worker_results: Optional[Dict[str, Any]]
    hypothesis_testing_results: Optional[List[Dict[str, Any]]]
    # Workflow metadata
    execution_status: str
    error_messages: List[str]
    performance_metrics: Dict[str, Any]
    current_agent: str
    
    # Additional workflow state
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    config: Optional[Dict[str, Any]]
    kb: Optional[Any]  # Knowledge base instance

    #Intugle tools
    intugle_tools: Optional[Any]