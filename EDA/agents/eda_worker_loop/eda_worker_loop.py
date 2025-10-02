from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from EDA.LLMS.llms import get_llm
from EDA.workflow.eda_workflow_state import EDAWorkflowState
import json
import logging
import time
from typing import Dict, Any, Optional, List
from EDA.agents.eda_worker_loop.eda_worker_loop_prompt import prompt
from utils.config_loader import AgentConfigLoader
from datetime import datetime
from EDA.agents.eda_analysis.eda_analysis import EDAAnalysisAgent


class EDAWorkerLoopAgent:
    def __init__(self, max_workers: int = 5, timeout_per_hypothesis: int = 300, kb=None, intugle_tools=None):
        """
        Initialize the EDA Worker Loop Agent
        
        Args:
            max_workers: Maximum number of concurrent workers
            timeout_per_hypothesis: Timeout in seconds per hypothesis test
            kb: Knowledge base instance (Intugle integration)
        """
        self.max_workers = max_workers
        self.timeout_per_hypothesis = timeout_per_hypothesis
        self.kb = kb
        self.config_loader = AgentConfigLoader()
        self.intugle_tools = intugle_tools
        
        # Load model configuration
        model_config = self.config_loader.get_model_config("eda_worker_loop")
        self.llm = get_llm(model_config['provider'], model_config['model'])
        self.prompt = self.config_loader.load_prompt("eda_worker_loop")
        
        # Agent-specific configuration
        self.config = self.config_loader.get_agent_config("eda_worker_loop")
    
    def process_hypotheses(self, hypotheses: List[Dict[str, Any]], target_variable: str) -> Dict[str, Any]:
        """Process multiple hypotheses sequentially using EDA Analysis Agent"""
        
        start_time = time.time()
        successful_results = []
        failed_hypotheses = []

        eda_agent = EDAAnalysisAgent(
                    kb=self.kb,
                    target_variable=target_variable,
                    timeout=self.timeout_per_hypothesis,
                    intugle_tools=self.intugle_tools
                )       
        for hypothesis in hypotheses:
            try:
                # Analyze hypothesis
                result = eda_agent.analyze_hypothesis(hypothesis=hypothesis)
                
                if result.get("status") == "completed":
                    successful_results.append({
                        "hypothesis_id": hypothesis.get("hypothesis_id", "unknown"),
                        "status": "success",
                        "result": result
                    })
                else:
                    failed_hypotheses.append({
                        "hypothesis": hypothesis,
                        "error": result.get("error", "Analysis failed"),
                        "retry_count": 1
                    })
                    
            except Exception as e:
                logging.error(f"Hypothesis analysis failed: {str(e)}")
                failed_hypotheses.append({
                    "hypothesis": hypothesis,
                    "error": str(e),
                    "retry_count": 1
                })
        
        total_execution_time = time.time() - start_time
        
        return {
            "execution_summary": {
                "total_hypotheses": len(hypotheses),
                "successful_analyses": len(successful_results),
                "failed_analyses": len(failed_hypotheses),
                "success_rate": len(successful_results) / len(hypotheses) if hypotheses else 0,
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / len(hypotheses) if hypotheses else 0
            },
            "results": successful_results,
            "failed_hypotheses": failed_hypotheses
        }
    
    def process(self, state: EDAWorkflowState):
        """
        Process method for LangGraph integration
        
        Args:
            state: EDAWorkflowState containing workflow state
            
        Returns:
            Updated state with EDA worker loop results
        """
        try:
            # Extract parameters from state
            hypotheses = state.get("generated_hypotheses", [])
            target_variable = state.get("target_variable", "")
            
            if not hypotheses:
                logging.warning("No hypotheses provided for processing")
                results = {
                    "execution_summary": {
                        "total_hypotheses": 0,
                        "successful_analyses": 0,
                        "failed_analyses": 0,
                        "success_rate": 0,
                        "total_execution_time": 0,
                        "average_execution_time": 0
                    },
                    "results": [],
                    "failed_hypotheses": []
                }
            else:
                results = self.process_hypotheses(hypotheses, target_variable)
            
            # Update state with results
            updated_state = state.copy()
            updated_state["eda_worker_results"] = results
            updated_state["hypothesis_testing_results"] = results.get("results", [])
            updated_state["current_agent"] = "eda_worker_loop"
            updated_state["execution_status"] = "completed"
            updated_state["timestamp"] = datetime.now().isoformat()
            return updated_state
            
        except Exception as e:
            logging.error(f"EDA worker loop agent processing failed: {str(e)}")
            updated_state = state.copy()
            updated_state["error_messages"] = updated_state.get("error_messages", [])
            updated_state["error_messages"].append(f"EDA worker loop failed: {str(e)}")
            updated_state["execution_status"] = "failed"
            return updated_state

def eda_worker_loop_agent(state: EDAWorkflowState):
    """
    LangGraph node function for EDA worker loop
    
    Args:
        state: EDAWorkflowState containing workflow state
        
    Returns:
        Updated state with EDA worker loop results
    """
    # Extract configuration from state
    config = state.get("config", {})
    kb = state.get("kb")
    
    # Get agent-specific configuration
    worker_config = config.get("eda_worker_config", {})
    
    # Initialize agent
    agent = EDAWorkerLoopAgent(
        max_workers=worker_config.get("max_workers", 5),
        timeout_per_hypothesis=worker_config.get("timeout_per_hypothesis", 300),
        kb=kb,
        intugle_tools=state.get("intugle_tools")
    )
    
    # Process state
    return agent.process(state)
