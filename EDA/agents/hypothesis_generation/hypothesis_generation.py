from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from EDA.LLMS.llms import get_llm
from EDA.workflow.eda_workflow_state import EDAWorkflowState
import json
import logging
import os
from typing import Dict, Any, Optional, List
from EDA.agents.hypothesis_generation.hypothesis_generation_prompt import prompt
from utils.config_loader import AgentConfigLoader
# from utils.helper import clean_messages_for_agent, msg_to_dict  # Not available
from datetime import datetime
import requests

# Web search tool for domain research
@tool
def web_search(query: str) -> str:
    """
    Search the web for information related to the query.
    
    Args:
        query: The search query to look up
        
    Returns:
        Search results as a string
    """
    try:
        # Using DuckDuckGo search API (free alternative to Google)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Extract relevant information
        results = []
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
        
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:  # Limit to top 3
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"Related: {topic['Text']}")
        
        if data.get("Definition"):
            results.append(f"Definition: {data['Definition']}")
        
        return "\n".join(results) if results else "No relevant information found."
        
    except Exception as e:
        return f"Search failed: {str(e)}"

class HypothesisGenerationAgent:
    def __init__(self, target_variable: str, hypothesis_limit: int = 10, domain: str = "", intugle_tools=None):
        """
        Initialize the Hypothesis Generation Agent
        
        Args:
            target_variable: The variable to be analyzed
            hypothesis_limit: Maximum number of hypotheses to generate
            domain: Business domain context for hypothesis generation
        """
        self.target_variable = target_variable
        self.hypothesis_limit = hypothesis_limit
        self.domain = domain
        self.config_loader = AgentConfigLoader()
        self.intugle_tools = intugle_tools
        # Load model configuration
        model_config = self.config_loader.get_model_config("hypothesis_generation")
        self.llm = get_llm(model_config['provider'], model_config['model'])
        self.prompt = self.config_loader.load_prompt("hypothesis_generation")
        
        # Agent-specific configuration
        self.config = self.config_loader.get_agent_config("hypothesis_generation")
        self.transformation_mappings = self._load_transformation_mappings()
    
    def _load_transformation_mappings(self) -> Dict[str, str]:
        """Load relationship type to transformation mappings"""
        return {
            "log_log_relationship": "log_log",
            "log_linear_relationship": "log_linear", 
            "lagged_effect": "lag_1",
            "seasonal_effect": "seasonal_decomposition",
            "interaction_effect": "interaction",
            "polynomial_relationship": "polynomial_2",
            "exponential_relationship": "exponential"
        }
    
    def discover_related_variables(self, target_variable: str) -> Dict[str, Any]:
        """Use Intugle tools to find related variables"""
        if not self.intugle_tools.is_available():
            return {"semantic_matches": [], "kpi_matches": [], "predicted_links": []}
        
        try:
            # Search for variables semantically related to target
            related_vars_result = self.intugle_tools.search_variables(f"variables that influence {target_variable}")
            related_vars = related_vars_result.get("variables", [])
            
            # Search for domain-specific KPIs
            kpi_vars_result = self.intugle_tools.search_variables(f"KPIs related to {target_variable}")
            kpi_vars = kpi_vars_result.get("variables", [])
            
            # Get variable profiles for additional context
            variable_profiles = self.intugle_tools.get_variable_profiles()
            predicted_links = []
            
            # Extract predicted relationships from variable profiles
            if "profiles" in variable_profiles:
                for var_id, profile in variable_profiles["profiles"].items():
                    if target_variable.lower() in profile.get("description", "").lower():
                        predicted_links.append(profile)
            
            # Extract variable names for EDA
            variable_names = []
            for var in related_vars + kpi_vars:
                if "variable_id" in var:
                    variable_names.append(var["variable_id"])
            
            # Add variable names from profiles
            if "profiles" in variable_profiles:
                for var_id in variable_profiles["profiles"].keys():
                    if var_id not in variable_names:
                        variable_names.append(var_id)
            
            return {
                "semantic_matches": related_vars,
                "kpi_matches": kpi_vars,
                "predicted_links": predicted_links,
                "variable_names": variable_names
            }
            
        except Exception as e:
            logging.error(f"Variable discovery failed: {str(e)}")
            return {"semantic_matches": [], "kpi_matches": [], "predicted_links": [], "variable_names": []}
    
    def _augment_domain_context(self, research_context: str) -> str:
        """Use LangGraph ReAct agent with web search to augment domain context"""
        if not research_context:
            return ""
        
        # Create ReAct agent with web search tool
        tools = [web_search]
        react_agent = create_react_agent(self.llm, tools)
        
        # Create research prompt for the agent
        research_prompt = f"""
        You are a domain expert researching business context for hypothesis generation.
        
        Original Context: {research_context}
        Target Variable: {self.target_variable}
        Domain: {self.domain}
        
        Your task is to enhance this context by researching and identifying:
        1. Key business factors that typically influence {self.target_variable}
        2. Industry insights and best practices for {self.domain}
        3. Common data relationships and transformations in this domain
        4. Predictive analytics approaches for {self.target_variable}
        
        Use web search to find current information about:
        - "{self.target_variable} influencing factors {self.domain}"
        - "{self.domain} predictive analytics best practices"
        - "feature engineering for {self.target_variable} prediction"
        - "{self.domain} {self.target_variable} data relationships"
        
        After gathering research, provide an enhanced context that combines the original context with your research findings.
        """
        
        try:
            # Execute ReAct agent
            messages = [HumanMessage(content=research_prompt)]
            response = react_agent.invoke({"messages": messages})
            
            # Extract the final response
            if isinstance(response, dict) and "messages" in response:
                final_message = response["messages"][-1]
                enhanced_context = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                enhanced_context = str(response)
            
            # Combine original and enhanced context
            return f"{research_context}\n\nResearch-Enhanced Insights: {enhanced_context}"
            
        except Exception as e:
            logging.error(f"Domain context enhancement failed: {str(e)}")
            return research_context
    
    def generate_hypotheses(self, univariate_results: Dict = None, research_context: str = "") -> Dict[str, Any]:
        """
        Generate hypotheses about variables influencing the target
        
        Args:
            univariate_results: Results from univariate analysis
            research_context: Domain context for hypothesis generation
            
        Returns:
            Dictionary containing generated hypotheses and metadata
        """
        try:
            # Discover related variables
            variable_discovery = self.discover_related_variables(self.target_variable)
            
            # Augment domain context with LLM-powered research
            enhanced_context = self._augment_domain_context(research_context)
            
            # Generate hypotheses using LLM
            hypotheses = self._generate_hypotheses_with_llm(variable_discovery, univariate_results, enhanced_context)
            
            # Process and structure hypotheses
            processed_hypotheses = self._process_hypotheses(hypotheses)
            
            # Build results
            results = {
                "hypotheses": processed_hypotheses,
                "variable_names": variable_discovery.get("variable_names", []),
                "summary": {
                    "total_hypotheses": len(processed_hypotheses),
                    "high_confidence_hypotheses": len([h for h in processed_hypotheses if h.get("confidence", 0) > 0.7]),
                    "total_variables_discovered": len(variable_discovery.get("variable_names", []))
                },
                "recommendations": self._generate_recommendations(processed_hypotheses)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Hypothesis generation failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "hypotheses": [],
                "variable_names": [],
                "summary": {"total_hypotheses": 0, "total_variables_discovered": 0},
                "recommendations": []
            }
    
    def _generate_hypotheses_with_llm(self, variable_discovery: Dict, univariate_results: Dict, research_context: str) -> List[Dict]:
        """Generate hypotheses using LLM with discovered variables"""
        try:
            # Prepare context for LLM
            context = {
                "target_variable": self.target_variable,
                "domain": self.domain,
                "hypothesis_limit": self.hypothesis_limit,
                "variable_discovery": variable_discovery,
                "univariate_results": univariate_results,
                "research_context": research_context,
                "transformation_mappings": self.transformation_mappings
            }
            print(f"Number of hypotheses to generate: {self.hypothesis_limit}")
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ]
            
            # Get LLM response
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                return parsed_response.get("hypotheses", [])
            
            return self._generate_fallback_hypotheses(variable_discovery)
                
        except Exception as e:
            logging.error(f"LLM hypothesis generation failed: {str(e)}")
            return self._generate_fallback_hypotheses(variable_discovery)
    
    def _generate_fallback_hypotheses(self, variable_discovery: Dict) -> List[Dict]:
        """Generate basic hypotheses when LLM fails"""
        hypotheses = []
        
        # Create hypotheses from discovered variables
        for var in variable_discovery.get("semantic_matches", [])[:self.hypothesis_limit]:
            hypothesis = {
                "hypothesis_id": f"hyp_{len(hypotheses) + 1}",
                "hypothesis": f"{var.get('name', 'Unknown variable')} influences {self.target_variable}",
                "target_variable": {
                    "name": self.target_variable,
                    "transformation": "none",
                    "aggregate_by": "default"
                },
                "predictor_variable": {
                    "name": var.get('name', 'unknown'),
                    "transformation": "none",
                    "aggregate_by": "default"
                },
                "relationship_type": "linear_relationship",
                "expected_impact": "unknown",
                "confidence": 0.5,
                "research_support": ["Generated from semantic search"],
                "interaction_features": [],
                "test_priority": 0.5,
                "data_requirements": {
                    "required_tables": [self.target_variable, var.get('table', 'unknown')],
                    "required_columns": [f"{self.target_variable}.value", f"{var.get('name', 'unknown')}.value"],
                    "join_requirements": "standard_join"
                }
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _process_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """Process and validate generated hypotheses"""
        processed = []
        
        for i, hypothesis in enumerate(hypotheses):
            # Ensure required fields
            processed_hyp = {
                "hypothesis_id": hypothesis.get("hypothesis_id", f"hyp_{i+1}"),
                "hypothesis": hypothesis.get("hypothesis", f"Hypothesis {i+1}"),
                "target_variable": hypothesis.get("target_variable", {
                    "name": self.target_variable,
                    "transformation": "none",
                    "aggregate_by": "default"
                }),
                "predictor_variable": hypothesis.get("predictor_variable", {
                    "name": "unknown",
                    "transformation": "none", 
                    "aggregate_by": "default"
                }),
                "relationship_type": hypothesis.get("relationship_type", "linear_relationship"),
                "expected_impact": hypothesis.get("expected_impact", "unknown"),
                "confidence": float(hypothesis.get("confidence", 0.5)),
                "research_support": hypothesis.get("research_support", []),
                "interaction_features": hypothesis.get("interaction_features", []),
                "test_priority": float(hypothesis.get("test_priority", 0.5)),
                "aggregate_by": hypothesis.get("aggregate_by", {}),
                "data_requirements": hypothesis.get("data_requirements", {
                    "required_tables": [],
                    "required_columns": [],
                    "join_requirements": "standard_join"
                })
            }
            
            # Apply transformation mapping
            relationship_type = processed_hyp["relationship_type"]
            if relationship_type in self.transformation_mappings:
                transformation = self.transformation_mappings[relationship_type]
                processed_hyp["predictor_variable"]["transformation"] = transformation
            
            processed.append(processed_hyp)
        
        return processed
    
    def _generate_recommendations(self, hypotheses: List[Dict]) -> List[str]:
        """Generate testing recommendations based on hypotheses"""
        if not hypotheses:
            return ["No hypotheses generated - check variable discovery inputs"]
        
        recommendations = []
        
        # Sort by test priority
        sorted_hypotheses = sorted(hypotheses, key=lambda x: x.get("test_priority", 0), reverse=True)
        
        # Top priority recommendation
        top_hyp = sorted_hypotheses[0]
        recommendations.append(f"Start with {top_hyp.get('hypothesis', 'top priority hypothesis')} (highest priority)")
        
        # Transformation recommendations
        transformations = set([h.get("predictor_variable", {}).get("transformation", "none") for h in hypotheses])
        if "log" in transformations:
            recommendations.append("Test logarithmic transformations for price-related variables")
        
        # Data validation
        recommendations.append("Validate data availability for all required tables and columns")
        
        return recommendations
    
    def process(self, state: EDAWorkflowState):
        """
        Process method for LangGraph integration
        
        Args:
            state: EDAWorkflowState containing workflow state
            
        Returns:
            Updated state with hypothesis generation results
        """
        try:
            # Extract parameters from state
            target_variable = state.get("target_variable", self.target_variable)
            univariate_results = state.get("univariate_results", {})
            research_context = state.get("domain_context", self.domain)
            # Update agent parameters if different from state
            if target_variable != self.target_variable:
                self.target_variable = target_variable
            
            # Execute hypothesis generation
            hypothesis_results = self.generate_hypotheses(univariate_results, research_context)
            
            # Update state with results
            updated_state = state.copy()
            updated_state["hypothesis_results"] = hypothesis_results
            updated_state["generated_hypotheses"] = hypothesis_results.get("hypotheses", [])
            updated_state["discovered_variables"] = hypothesis_results.get("variable_names", [])
            updated_state["current_agent"] = "hypothesis_generation"
            updated_state["execution_status"] = "completed"
            updated_state["timestamp"] = datetime.now().isoformat()
            
            return updated_state
            
        except Exception as e:
            logging.error(f"Hypothesis generation agent processing failed: {str(e)}")
            updated_state = state.copy()
            updated_state["error_messages"] = updated_state.get("error_messages", [])
            updated_state["error_messages"].append(f"Hypothesis generation failed: {str(e)}")
            updated_state["execution_status"] = "failed"
            return updated_state

def hypothesis_generation_agent(state: EDAWorkflowState):
    """
    LangGraph node function for hypothesis generation
    
    Args:
        state: EDAWorkflowState containing workflow state
        
    Returns:
        Updated state with hypothesis generation results
    """
    # Extract configuration from state    
    # Get agent-specific configuration
    
    # Initialize agent
    agent = HypothesisGenerationAgent(
        target_variable=state.get("target_variable", ""),
        hypothesis_limit=state.get("hypothesis_limit", 1),
        domain=state.get("domain_context", ""),
        intugle_tools=state.get("intugle_tools")   
    )
    # Process state
    return agent.process(state)
