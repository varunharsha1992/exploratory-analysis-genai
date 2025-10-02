from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from EDA.LLMS.llms import get_llm
from EDA.workflow.eda_workflow_state import EDAWorkflowState
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from EDA.agents.univariate_analysis.univariate_analysis_prompt import prompt
from utils.config_loader import AgentConfigLoader
# from utils.helper import clean_messages_for_agent, msg_to_dict  # Not available due to moviepy dependency
from EDA.tools.univariate_analysis import (
    DataProfilingTool, 
    AnomalyDetectionTool, 
    TrendAnalysisTool,
    AnomalyMethod
)

class UnivariateAnalysisAgent:
    def __init__(self, target_variable: str = None, data: pd.DataFrame = None, kb=None, intugle_tools=None, config: Optional[Dict] = None):
        """
        Initialize the Univariate Analysis Agent with LLM agentic flow
        
        Args:
            target_variable: The variable to be analyzed
            data: Preloaded DataFrame containing the data to analyze
            kb: Knowledge base instance (Intugle integration)
            config: Configuration dictionary for analysis parameters
        """
        self.target_variable = target_variable
        self.data = data
        self.kb = kb
        self.intugle_tools = intugle_tools
        self.config = config or {}
        self.config_loader = AgentConfigLoader()
        
        # Load model configuration
        model_config = self.config_loader.get_model_config("univariate_analysis")
        self.llm = get_llm(model_config['provider'], model_config['model'])
        self.prompt = self.config_loader.load_prompt("univariate_analysis")
        
        # Analysis configuration
        self.anomaly_detection_method = self.config.get('anomaly_detection_method', 'iqr')
        self.trend_analysis = self.config.get('trend_analysis', True)
        self.visualization_enabled = self.config.get('visualization_enabled', True)
        
        # Initialize analysis tools
        self.data_profiling_tool = DataProfilingTool()
        self.anomaly_detection_tool = AnomalyDetectionTool()
        self.trend_analysis_tool = TrendAnalysisTool()
    
    def _create_analysis_tools(self) -> List:
        """Create tools for the LLM agent to use"""
        
        @tool
        def profile_variable(variable_id: str) -> Dict[str, Any]:
            """
            Generate comprehensive data profile for a variable including statistical summaries,
            data type analysis, distribution characteristics, completeness, and uniqueness metrics.
            
            Args:
                variable_id: The ID of the variable to profile
                
            Returns:
                Dictionary containing comprehensive variable profile
            """
            try:
                if self.data is None:
                    return {"error": "No data provided to agent. Please provide data during initialization."}
                return self.data_profiling_tool.profile_variable(variable_id, self.data)
            except Exception as e:
                return {"error": f"Data profiling failed: {str(e)}"}
        
        @tool
        def detect_anomalies(variable_id: str, methods: List[str] = None) -> Dict[str, Any]:
            """
            Detect anomalies and outliers in a variable using multiple statistical methods.
            
            Args:
                variable_id: The ID of the variable to analyze
                methods: List of anomaly detection methods to use (iqr, z_score, isolation_forest)
                
            Returns:
                Dictionary containing anomaly detection results
            """
            try:
                if self.data is None:
                    return {"error": "No data provided to agent. Please provide data during initialization."}
                
                if methods is None:
                    methods = [AnomalyMethod.IQR, AnomalyMethod.Z_SCORE, AnomalyMethod.ISOLATION_FOREST]
                else:
                    method_map = {
                        'iqr': AnomalyMethod.IQR,
                        'z_score': AnomalyMethod.Z_SCORE,
                        'isolation_forest': AnomalyMethod.ISOLATION_FOREST
                    }
                    methods = [method_map.get(m, AnomalyMethod.IQR) for m in methods]
                
                return self.anomaly_detection_tool.detect_anomalies(variable_id, self.data, methods)
            except Exception as e:
                return {"error": f"Anomaly detection failed: {str(e)}"}
        
        @tool
        def analyze_trends(variable_id: str) -> Dict[str, Any]:
            """
            Analyze temporal trends, seasonality, volatility, and cyclical patterns in a variable.
            
            Args:
                variable_id: The ID of the variable to analyze
                
            Returns:
                Dictionary containing trend analysis results
            """
            try:
                if self.data is None:
                    return {"error": "No data provided to agent. Please provide data during initialization."}
                return self.trend_analysis_tool.analyze_trends(variable_id, self.data)
            except Exception as e:
                return {"error": f"Trend analysis failed: {str(e)}"}
        
        @tool
        def discover_related_variables(target_variable: str, max_results: int = 10) -> Dict[str, Any]:
            """
            Discover variables related to the target variable using semantic search.
            
            Args:
                target_variable: The target variable to find related variables for
                max_results: Maximum number of related variables to return
                
            Returns:
                Dictionary containing related variables and relationship analysis
            """
            try:
                return self.intugle_tools.search_variables(target_variable, max_results)
            except Exception as e:
                return {"error": f"Related variables discovery failed: {str(e)}"}
        
        @tool
        def search_knowledge_base(query: str, max_results: int = 10) -> Dict[str, Any]:
            """
            Search the knowledge base for information related to the query.
            
            Args:
                query: The search query
                max_results: Maximum number of results to return
                
            Returns:
                Dictionary containing search results
            """
            try:
                return self.intugle_tools.search_variables(query, max_results)
            except Exception as e:
                return {"error": f"Knowledge base search failed: {str(e)}", "query": query}
        
        @tool
        def get_variable_metadata(variable_id: str) -> Dict[str, Any]:
            """
            Get metadata information about a variable from the knowledge base.
            
            Args:
                variable_id: The ID of the variable (format: "table.column")
                
            Returns:
                Dictionary containing variable metadata
            """
            try:
                # Extract table and variable from variable_id
                if '.' in variable_id:
                    table_name, variable_name = variable_id.split('.', 1)
                    return self.intugle_tools.get_variable_profiles(table_name, variable_name)
                else:
                    return {"error": "Invalid variable_id format. Expected 'table.column'", "variable_id": variable_id}
            except Exception as e:
                return {"error": f"Metadata retrieval failed: {str(e)}", "variable_id": variable_id}
        
        @tool
        def create_etl_schema(user_input: str) -> Dict[str, Any]:
            """
            Create ETL schema using LLM based on user input and available fields from KB.
            
            Args:
                user_input: User's natural language input describing what they want to analyze
                
            Returns:
                Dictionary containing generated ETL schema
            """
            try:
                return self.intugle_tools.create_etl_schema_with_llm(user_input)
            except Exception as e:
                return {"error": f"ETL schema creation failed: {str(e)}"}
        
        @tool
        def build_data_product(data_product_config: Dict[str, Any]) -> Dict[str, Any]:
            """
            Build DataProduct using the provided configuration.
            
            Args:
                data_product_config: DataProduct configuration following Intugle format
                
            Returns:
                Dictionary with build results
            """
            try:
                return self.intugle_tools.build_dataproduct(data_product_config)
            except Exception as e:
                return {"error": f"Data product build failed: {str(e)}"}
        
        return [
            profile_variable,
            detect_anomalies,
            analyze_trends,
            discover_related_variables,
            search_knowledge_base,
            get_variable_metadata,
            create_etl_schema,
            build_data_product
        ]
    
    def _prepare_analysis_request(self, target_variable: str, data: Optional[Any] = None) -> str:
        """Prepare the analysis request for the LLM agent"""
        
        # Check if Intugle tools are available
        intugle_available = self.intugle_tools.is_available()
        
        request = f"""
        Perform comprehensive univariate analysis on the target variable: {target_variable}
        
        Analysis Requirements:
        1. Generate a complete data profile including statistical summaries, data types, and distribution characteristics
        2. Detect anomalies and outliers using multiple methods
        3. Analyze trends, seasonality, and cyclical patterns
        4. Discover related variables through semantic search
        5. Assess data quality and provide actionable recommendations
        
        Available Tools:
        - Data profiling: profile_variable(variable_id) - uses preloaded data
        - Anomaly detection: detect_anomalies(variable_id, methods)
        - Trend analysis: analyze_trends(variable_id)
        - Related variables: discover_related_variables(target_variable, max_results)
        - Knowledge base search: search_knowledge_base(query, max_results)
        - Variable metadata: get_variable_metadata(variable_id)
        {"- ETL schema creation: create_etl_schema(user_input)" if intugle_available else ""}
        {"- Data product building: build_data_product(data_product_config)" if intugle_available else ""}
        
        Configuration:
        - Anomaly detection method: {self.anomaly_detection_method}
        - Trend analysis enabled: {self.trend_analysis}
        - Visualization enabled: {self.visualization_enabled}
        - Intugle tools available: {intugle_available}
        
        Please use the available tools to gather all necessary information and provide a comprehensive analysis.
        Focus on actionable insights that will inform feature engineering and model development decisions.
        
        Return your analysis in the specified JSON format with detailed findings and recommendations.
        """
        
        return request
    
    def _parse_agent_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM agent response and extract analysis results"""
        try:
            # Extract the final message from the agent response
            messages = response.get("messages", [])
            if not messages:
                return {"error": "No response from agent"}
            
            # Get the last AI message
            final_message = None
            for message in reversed(messages):
                if hasattr(message, 'content') and message.content:
                    final_message = message
                    break
            
            if not final_message:
                return {"error": "No valid response from agent"}
            
            content = final_message.content
            
            # Try to parse JSON response
            try:
                if content.startswith('{') and content.endswith('}'):
                    parsed_response = json.loads(content)
                    return parsed_response
                else:
                    # Look for JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed_response = json.loads(json_match.group())
                        return parsed_response
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, create a structured response from the text
            return {
                "target_variable": {
                    "name": self.target_variable,
                    "analysis_summary": content,
                    "analysis_type": "llm_generated"
                },
                "llm_response": content,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "status": "completed"
            }
            
        except Exception as e:
            logging.error(f"Failed to parse agent response: {str(e)}")
            return {
                "error": f"Response parsing failed: {str(e)}",
                "target_variable": self.target_variable,
                "status": "failed"
            }
    
    def process(self, state: EDAWorkflowState):
        """
        Process method for LangGraph integration with LLM agentic flow
        
        Args:
            state: EDAWorkflowState containing workflow state
            
        Returns:
            Updated state with univariate analysis results
        """
        try:
            # Extract target variable and data from state
            target_variable = state.get("target_variable")
            data = state.get("data")
            
            if not target_variable:
                raise ValueError("Missing target variable in state")
            
            # Update agent with target variable and data
            self.target_variable = target_variable
            if data is not None:
                self.data = data
            
            # Extract context and messages
            context = state.get("context", {})
            # messages = clean_messages_for_agent(state.get("messages", []))  # Not available due to moviepy dependency
            
            # Simplified message handling without helper functions
            messages = state.get("messages", [])
            if not messages or not any(getattr(m, "content", "").strip() for m in messages):
                # Create analysis request if no messages
                analysis_request = self._prepare_analysis_request(target_variable)
                messages = [AIMessage(content=analysis_request)]
            
            # messages_dict = [msg_to_dict(m) for m in messages]  # Not available due to moviepy dependency
            messages_dict = [{"role": "assistant" if hasattr(m, "type") and m.type == "ai" else "user", "content": getattr(m, "content", str(m))} for m in messages]
            
            # Create tools for the LLM agent
            tools = self._create_analysis_tools()
            
            # Create the LangGraph agent
            univariate_react_agent = create_react_agent(
                model=self.llm,
                tools=tools,
                prompt=self.prompt
            )
            
            # Execute the agentic analysis
            thread_id = context.get("thread_id")
            if thread_id:
                response = univariate_react_agent.invoke({
                    "messages": messages_dict
                }, config={"configurable": {"thread_id": thread_id}})
            else:
                response = univariate_react_agent.invoke({
                    "messages": messages_dict
                }, config={"configurable": {"thread_id": None}})
            
            # Parse the agent response
            analysis_results = self._parse_agent_response(response)
            
            # Update state with results
            updated_state = state.copy()
            updated_state["univariate_results"] = analysis_results
            updated_state["current_agent"] = "univariate_analysis"
            updated_state["execution_status"] = "completed"
            
            # Add agent response to messages
            updated_messages = messages.copy()
            agent_response = response["messages"][-1]
            
            content = agent_response.content
            if content.startswith('{') and content.endswith('}'):
                try:
                    result_json = json.loads(content)
                    content = result_json.get("text", content)
                except:
                    pass
            agent_response.content = content
            
            updated_messages.append(agent_response)
            updated_state["agent_results"] = updated_messages
            
            return updated_state
            
        except Exception as e:
            logging.error(f"Univariate analysis agent processing failed: {str(e)}")
            updated_state = state.copy()
            updated_state["error_messages"] = updated_state.get("error_messages", [])
            updated_state["error_messages"].append(f"Univariate analysis failed: {str(e)}")
            updated_state["execution_status"] = "failed"
            return updated_state

def univariate_analysis_agent(state: EDAWorkflowState):
    """
    LangGraph node function for univariate analysis with LLM agentic flow
    
    Args:
        state: EDAWorkflowState containing workflow state
        
    Returns:
        Updated state with univariate analysis results
    """
    # Extract configuration from state
    config = state.get("config", {})
    kb = state.get("kb")
    intugle_tools = state.get("intugle_tools")
    print(f"Intugle tools: {intugle_tools}")
    # Initialize agent
    agent = UnivariateAnalysisAgent(
        target_variable=state.get("target_variable", ""),
        intugle_tools=intugle_tools,
        data=state.get("data"),
        kb=kb,
        config=config.get("univariate_config", {})
    )
    
    # Process state
    return agent.process(state)