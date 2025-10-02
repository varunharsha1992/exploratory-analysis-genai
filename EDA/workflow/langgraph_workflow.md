# LangGraph EDA Workflow Specification

## Overview
This document specifies the complete LangGraph workflow for automated EDA (Exploratory Data Analysis) for predictive analytics. The workflow orchestrates 5 specialized agents to perform comprehensive data analysis and generate feature engineering recommendations.

## Workflow Architecture

### Graph Structure
```
Input: Target Variable + EDA Request
    ↓
[Univariate Analysis Agent]
    ↓
[Hypothesis Generation Agent]
    ↓
[EDA Worker Loop Agent] (Async Orchestrator)
    ↓
[EDA Analysis Agent] (Multiple Parallel Instances)
    ↓
Output: Feature Engineering Recommendations
```

### State Management
```python
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

class EDAWorkflowState(TypedDict):
    # Input parameters
    target_variable: str
    eda_request: str
    domain_context: str
    hypothesis_limit: int
    
    # Agent outputs
    univariate_results: Optional[Dict[str, Any]]
    generated_hypotheses: Optional[List[Dict[str, Any]]]
    hypothesis_testing_results: Optional[Dict[str, Any]]
    final_summary: Optional[Dict[str, Any]]
    
    # Workflow metadata
    execution_status: str
    error_messages: List[str]
    performance_metrics: Dict[str, Any]
    current_agent: str
```

## Node Definitions

### 1. Univariate Analysis Node
```python
def univariate_analysis_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Execute univariate analysis on target variable"""
    
    try:
        # Initialize agent
        univariate_agent = UnivariateAnalysisAgent(
            target_variable=state["target_variable"],
            kb=knowledge_builder,
            config=analysis_config
        )
        
        # Execute analysis
        results = univariate_agent.analyze()
        
        # Update state
        state["univariate_results"] = results
        state["current_agent"] = "univariate_analysis"
        state["execution_status"] = "completed"
        
        return state
        
    except Exception as e:
        state["error_messages"].append(f"Univariate analysis failed: {str(e)}")
        state["execution_status"] = "failed"
        return state
```

### 2. Hypothesis Generation Node
```python
def hypothesis_generation_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Generate hypotheses about influencing variables"""
    
    try:
        # Initialize agent
        hypothesis_agent = HypothesisGenerationAgent(
            target_variable=state["target_variable"],
            kb=knowledge_builder,
            hypothesis_limit=state["hypothesis_limit"],
            domain=state["domain_context"]
        )
        
        # Generate hypotheses
        hypotheses = hypothesis_agent.generate_hypotheses(
            univariate_results=state["univariate_results"],
            research_context=state["domain_context"]
        )
        
        # Update state
        state["generated_hypotheses"] = hypotheses
        state["current_agent"] = "hypothesis_generation"
        state["execution_status"] = "completed"
        
        return state
        
    except Exception as e:
        state["error_messages"].append(f"Hypothesis generation failed: {str(e)}")
        state["execution_status"] = "failed"
        return state
```

### 3. EDA Worker Loop Node
```python
async def eda_worker_loop_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Orchestrate parallel hypothesis testing"""
    
    try:
        # Initialize worker loop agent
        worker_agent = EDAWorkerLoopAgent(
            max_workers=5,
            timeout_per_hypothesis=300
        )
        
        # Process hypotheses in parallel
        results = await worker_agent.process_hypotheses(
            hypotheses=state["generated_hypotheses"],
            kb=knowledge_builder,
            target_variable=state["target_variable"]
        )
        
        # Update state
        state["hypothesis_testing_results"] = results
        state["current_agent"] = "eda_worker_loop"
        state["execution_status"] = "completed"
        
        return state
        
    except Exception as e:
        state["error_messages"].append(f"Hypothesis testing failed: {str(e)}")
        state["execution_status"] = "failed"
        return state
```

```

## Workflow Construction

### Graph Definition
```python
from langgraph.graph import StateGraph, END

def create_eda_workflow():
    """Create the complete EDA workflow graph"""
    
    # Initialize workflow
    workflow = StateGraph(EDAWorkflowState)
    
    # Add nodes
    workflow.add_node("univariate_analysis", univariate_analysis_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("eda_worker_loop", eda_worker_loop_node)
    
    # Define edges
    workflow.add_edge("univariate_analysis", "hypothesis_generation")
    workflow.add_edge("hypothesis_generation", "eda_worker_loop")
    workflow.add_edge("eda_worker_loop", "summarizer")
    
    # Set entry point
    workflow.set_entry_point("univariate_analysis")
    
    # Compile workflow
    return workflow.compile()
```

### Conditional Routing
```python
def should_continue_workflow(state: EDAWorkflowState) -> str:
    """Determine if workflow should continue based on execution status"""
    
    if state["execution_status"] == "failed":
        return "error_handler"
    elif state["current_agent"] == "summarizer":
        return END
    else:
        return "continue"

# Add conditional edges
workflow.add_conditional_edges(
    "univariate_analysis",
    should_continue_workflow,
    {
        "continue": "hypothesis_generation",
        "error_handler": "error_handler_node",
        END: END
    }
)
```

## Error Handling

### Error Handler Node
```python
def error_handler_node(state: EDAWorkflowState) -> EDAWorkflowState:
    """Handle workflow errors and provide recovery options"""
    
    error_summary = {
        "failed_agent": state["current_agent"],
        "error_messages": state["error_messages"],
        "recovery_options": [],
        "partial_results": {}
    }
    
    # Collect partial results
    if state["univariate_results"]:
        error_summary["partial_results"]["univariate_analysis"] = state["univariate_results"]
    
    if state["generated_hypotheses"]:
        error_summary["partial_results"]["hypotheses"] = state["generated_hypotheses"]
    
    # Provide recovery options
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
    
    state["final_summary"] = error_summary
    state["execution_status"] = "error_handled"
    
    return state
```

## Workflow Execution

### Main Execution Function
```python
async def execute_eda_workflow(
    target_variable: str,
    eda_request: str,
    domain_context: str = "",
    hypothesis_limit: int = 10
) -> Dict[str, Any]:
    """Execute the complete EDA workflow"""
    
    # Initialize workflow
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
        current_agent=""
    )
    
    # Execute workflow
    try:
        final_state = await workflow.ainvoke(initial_state)
        
        return {
            "status": "success",
            "results": final_state["final_summary"],
            "execution_metrics": final_state["performance_metrics"],
            "workflow_path": final_state["current_agent"]
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "partial_results": final_state.get("partial_results", {}),
            "error_messages": final_state.get("error_messages", [])
        }
```

### Workflow Monitoring
```python
def monitor_workflow_progress(state: EDAWorkflowState) -> Dict[str, Any]:
    """Monitor workflow execution progress"""
    
    progress = {
        "current_step": state["current_agent"],
        "completed_steps": [],
        "remaining_steps": [],
        "progress_percentage": 0,
        "estimated_completion": None
    }
    
    # Define workflow steps
    workflow_steps = [
        "univariate_analysis",
        "hypothesis_generation", 
        "eda_worker_loop",
    ]
    
    current_index = workflow_steps.index(state["current_agent"]) if state["current_agent"] in workflow_steps else 0
    
    progress["completed_steps"] = workflow_steps[:current_index]
    progress["remaining_steps"] = workflow_steps[current_index:]
    progress["progress_percentage"] = (current_index / len(workflow_steps)) * 100
    
    return progress
```

## Configuration and Customization

### Workflow Configuration
```python
class EDAWorkflowConfig:
    """Configuration for EDA workflow"""
    
```

### Customization Options
```python
def customize_workflow_for_domain(domain: str) -> EDAWorkflowConfig:
    """Customize workflow configuration based on domain"""
    
    config = EDAWorkflowConfig()
    

    return config
```

## Integration with Intugle

### Intugle Integration Layer
```python
class IntugleIntegration:
    """Integration layer for Intugle tools"""
    
    def __init__(self, project_base: str):
        self.knowledge_builder = KnowledgeBuilder(project_base)
        self.data_product_builder = DataProductBuilder(project_base)
        self.semantic_search = SemanticSearch(project_base)
    
    def initialize_semantic_layer(self):
        """Initialize Intugle semantic layer"""
        try:
            self.knowledge_builder.build()
            self.semantic_search.initialize()
            return True
        except Exception as e:
            print(f"Failed to initialize semantic layer: {e}")
            return False
    
    def get_available_variables(self) -> List[str]:
        """Get list of available variables from semantic layer"""
        return list(self.knowledge_builder.field_details.keys())
    
    def search_related_variables(self, query: str) -> List[Dict]:
        """Search for variables related to query"""
        return self.semantic_search.search(query)
```

## Usage Examples

### Basic Usage
```python
# Initialize and execute workflow
async def run_basic_eda():
    result = await execute_eda_workflow(
        target_variable="sales_revenue",
        eda_request="Analyze factors influencing sales revenue",
        domain_context="retail",
        hypothesis_limit=10
    )
    
    print(f"Workflow Status: {result['status']}")
    if result['status'] == 'success':
        recommendations = result['results']['feature_engineering_recommendations']
        print(f"Generated {len(recommendations['high_priority_features'])} high-priority features")
```

### Advanced Usage with Customization
```python
# Customize workflow for specific domain
async def run_customized_eda():
    config = customize_workflow_for_domain("healthcare")
    
    # Initialize workflow with custom config
    workflow = create_eda_workflow(config)
    
    result = await workflow.ainvoke({
        "target_variable": "patient_outcome",
        "eda_request": "Identify factors affecting patient outcomes",
        "domain_context": "healthcare",
        "hypothesis_limit": 15
    })
    
    return result
```

## Performance Optimization

### Parallel Processing
- EDA Worker Loop Agent processes multiple hypotheses concurrently
- Data prefetching reduces redundant queries
- Caching mechanisms for repeated analyses

### Resource Management
- Configurable worker limits and timeouts
- Memory usage monitoring and optimization
- Graceful handling of resource constraints

### Error Recovery
- Retry logic for failed hypothesis tests
- Partial result preservation
- Graceful degradation when components fail

This LangGraph workflow provides a robust, scalable framework for automated EDA that integrates seamlessly with Intugle's semantic layer and provides comprehensive feature engineering recommendations for predictive analytics.
