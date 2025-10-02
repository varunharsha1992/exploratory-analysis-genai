"""
Example usage of the HypothesisGenerationAgent

This file demonstrates how to use the HypothesisGenerationAgent
both as a standalone component and as part of the LangGraph workflow.
"""

import asyncio
from typing import Dict, Any
from EDA.agents.hypothesis_generation.hypothesis_generation import HypothesisGenerationAgent, hypothesis_generation_agent

# Mock dependencies for demonstration
class MockKnowledgeBase:
    """Mock Knowledge Base for testing purposes"""
    
    def search(self, query: str) -> list:
        """Mock semantic search functionality"""
        mock_results = [
            {"name": "price", "table": "products", "relationship": "negative", "correlation": -0.7},
            {"name": "promotion", "table": "marketing", "relationship": "positive", "correlation": 0.6},
            {"name": "customer_satisfaction", "table": "feedback", "relationship": "positive", "correlation": 0.5},
            {"name": "seasonality", "table": "calendar", "relationship": "seasonal", "correlation": 0.4}
        ]
        return mock_results
    
    @property
    def links(self):
        """Mock predicted relationships"""
        class MockLink:
            def __init__(self, source, target, strength):
                self.source_field_id = source
                self.target_field_id = target
                self.strength = strength
        
        return [
            MockLink("sales", "price", 0.8),
            MockLink("sales", "promotion", 0.7),
            MockLink("sales", "customer_satisfaction", 0.6)
        ]

def example_standalone_usage():
    """Example of using the agent as a standalone component"""
    print("=== Standalone HypothesisGenerationAgent Usage ===")
    
    # Initialize mock knowledge base
    mock_kb = MockKnowledgeBase()
    
    # Initialize agent
    agent = HypothesisGenerationAgent(
        target_variable="sales_revenue",
        hypothesis_limit=5,
        domain="retail"
    )
    
    # Mock univariate analysis results
    univariate_results = {
        "target_variable": "sales_revenue",
        "statistics": {
            "mean": 1000,
            "std": 200,
            "skewness": 0.5
        },
        "distribution": "right_skewed"
    }
    
    # Execute hypothesis generation
    results = agent.generate_hypotheses(
        univariate_results=univariate_results,
        research_context="Retail sales analysis for Q4 optimization"
    )
    
    # Display results
    print(f"Generated {results['summary']['total_hypotheses']} hypotheses")
    print(f"High confidence hypotheses: {results['summary']['high_confidence_hypotheses']}")
    
    for i, hypothesis in enumerate(results['hypotheses'], 1):
        print(f"\nHypothesis {i}: {hypothesis['hypothesis']}")
        print(f"  Confidence: {hypothesis['confidence']}")
        print(f"  Relationship: {hypothesis['relationship_type']}")
        print(f"  Predictor: {hypothesis['predictor_variable']['name']}")
        print(f"  Transformation: {hypothesis['predictor_variable']['transformation']}")
    
    print(f"\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    return results

def example_langgraph_usage():
    """Example of using the agent as a LangGraph node"""
    print("\n=== LangGraph Node Usage ===")
    
    # Create mock state
    state = {
        "target_variable": "sales_revenue",
        "domain": "retail",
        "hypothesis_limit": 3,
        "univariate_results": {
            "target_variable": "sales_revenue",
            "statistics": {"mean": 1000, "std": 200}
        },
        "domain_context": "Retail sales analysis",
        "config": {
            "hypothesis_config": {
                "max_hypotheses": 3,
                "domain": "retail"
            }
        }
    }
    
    # Execute as LangGraph node
    updated_state = hypothesis_generation_agent(state)
    
    # Display results
    print(f"Execution status: {updated_state['execution_status']}")
    print(f"Current agent: {updated_state['current_agent']}")
    
    if 'hypothesis_results' in updated_state:
        results = updated_state['hypothesis_results']
        print(f"Generated {results['summary']['total_hypotheses']} hypotheses")
        
        for i, hypothesis in enumerate(results['hypotheses'], 1):
            print(f"\nHypothesis {i}: {hypothesis['hypothesis']}")
            print(f"  Confidence: {hypothesis['confidence']}")
            print(f"  Test Priority: {hypothesis['test_priority']}")
    
    return updated_state

async def example_workflow_integration():
    """Example of integrating with the complete workflow"""
    print("\n=== Workflow Integration Example ===")
    
    # This would be used in the actual LangGraph workflow
    workflow_state = {
        "target_variable": "sales_revenue",
        "domain": "retail",
        "hypothesis_limit": 4,
        "univariate_results": {
            "target_variable": "sales_revenue",
            "statistics": {"mean": 1000, "std": 200, "skewness": 0.5},
            "distribution": "right_skewed",
            "outliers": [1500, 1600, 1700]
        },
        "domain_context": "Q4 retail sales optimization with focus on price elasticity",
        "config": {
            "hypothesis_config": {
                "max_hypotheses": 4,
                "domain": "retail",
                "min_confidence_threshold": 0.6
            }
        },
        "execution_status": "in_progress",
        "current_agent": "univariate_analysis"
    }
    
    # Execute agent node
    updated_state = hypothesis_generation_agent(workflow_state)
    
    # Display workflow results
    print(f"Workflow execution status: {updated_state['execution_status']}")
    print(f"Current agent: {updated_state['current_agent']}")
    print(f"Timestamp: {updated_state.get('timestamp', 'N/A')}")
    
    if 'hypothesis_results' in updated_state:
        results = updated_state['hypothesis_results']
        print(f"\nWorkflow Results:")
        print(f"  Total hypotheses: {results['summary']['total_hypotheses']}")
        print(f"  High confidence: {results['summary']['high_confidence_hypotheses']}")
        print(f"  Transformation types: {results['summary']['transformation_types']}")
        print(f"  Aggregation types: {results['summary']['aggregation_types']}")
        
        # Show top priority hypothesis
        if results['hypotheses']:
            top_hyp = max(results['hypotheses'], key=lambda x: x.get('test_priority', 0))
            print(f"\nTop Priority Hypothesis:")
            print(f"  {top_hyp['hypothesis']}")
            print(f"  Priority: {top_hyp['test_priority']}")
            print(f"  Confidence: {top_hyp['confidence']}")
    
    return updated_state

def example_error_handling():
    """Example of error handling scenarios"""
    print("\n=== Error Handling Examples ===")
    
    # Test with no knowledge base
    print("1. Testing without knowledge base:")
    agent_no_kb = HypothesisGenerationAgent(
        target_variable="sales_revenue",
        hypothesis_limit=2,
        domain="retail"
    )
    
    results_no_kb = agent_no_kb.generate_hypotheses()
    print(f"   Results without KB: {len(results_no_kb['hypotheses'])} hypotheses")
    
    # Test with empty variable discovery
    print("2. Testing with empty variable discovery:")
    agent_empty = HypothesisGenerationAgent(
        target_variable="unknown_variable",
        hypothesis_limit=2,
        domain="unknown_domain"
    )
    
    results_empty = agent_empty.generate_hypotheses()
    print(f"   Results with empty discovery: {len(results_empty['hypotheses'])} hypotheses")
    
    # Test error state
    print("3. Testing error state handling:")
    error_state = {
        "target_variable": None,  # Invalid target variable
        "config": {}
    }
    
    try:
        error_result = hypothesis_generation_agent(error_state)
        print(f"   Error handling result: {error_result['execution_status']}")
        if 'error_messages' in error_result:
            print(f"   Error messages: {error_result['error_messages']}")
    except Exception as e:
        print(f"   Exception caught: {str(e)}")

if __name__ == "__main__":
    # Run examples
    print("HypothesisGenerationAgent Examples")
    print("=" * 50)
    
    # Standalone usage
    standalone_results = example_standalone_usage()
    
    # LangGraph node usage
    langgraph_results = example_langgraph_usage()
    
    # Workflow integration
    asyncio.run(example_workflow_integration())
    
    # Error handling
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
