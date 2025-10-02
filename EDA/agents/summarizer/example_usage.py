"""
Example usage of the SummarizerAgent

This file demonstrates how to use the SummarizerAgent
both as a standalone component and as part of the LangGraph workflow.
"""

import asyncio
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.agents.summarizer.summarizer import SummarizerAgent, summarizer_agent

# Mock dependencies for demonstration
class MockUnivariateResults:
    """Mock univariate analysis results for testing purposes"""
    
    def get_mock_results(self) -> Dict[str, Any]:
        """Mock univariate analysis results"""
        return {
            "target_variable": {
                "profile": {
                    "missing_percentage": 2.1,
                    "mean": 150.5,
                    "std": 45.2,
                    "skewness": 0.3
                },
                "anomalies": {
                    "outliers_count": 12
                },
                "data_type": "numeric"
            },
            "data_quality_summary": {
                "overall_quality": "good",
                "issues_found": 1,
                "recommendations": ["Consider outlier treatment"]
            }
        }

class MockHypothesisResults:
    """Mock hypothesis testing results for testing purposes"""
    
    def get_mock_results(self) -> List[Dict[str, Any]]:
        """Mock hypothesis testing results"""
        return [
            {
                "hypothesis_id": "hyp_1",
                "status": "success",
                "hypothesis": {
                    "predictor_variable": {"name": "price", "transformation": "log"}
                },
                "result": {
                    "correlation_analysis": {
                        "correlations": {
                            "pearson": {
                                "correlation": -0.75,
                                "significance": "significant"
                            }
                        }
                    },
                    "transformations": {
                        "log_transformation": {
                            "improved_correlation": True
                        }
                    }
                }
            },
            {
                "hypothesis_id": "hyp_2", 
                "status": "success",
                "hypothesis": {
                    "predictor_variable": {"name": "marketing_spend", "transformation": "none"}
                },
                "result": {
                    "correlation_analysis": {
                        "correlations": {
                            "pearson": {
                                "correlation": 0.68,
                                "significance": "significant"
                            }
                        }
                    },
                    "transformations": {
                        "log_transformation": {
                            "improved_correlation": False
                        }
                    }
                }
            }
        ]

def example_standalone_usage():
    """Example of using the agent as a standalone component"""
    print("=== Standalone SummarizerAgent Usage ===")
    
    # Initialize agent
    agent = SummarizerAgent(
        target_variable="sales_revenue",
        domain_context="retail",
        modeling_objective="predictive_analytics"
    )
    
    # Mock input data
    mock_univariate = MockUnivariateResults()
    mock_hypothesis = MockHypothesisResults()
    
    univariate_results = mock_univariate.get_mock_results()
    hypothesis_results = mock_hypothesis.get_mock_results()
    
    # Generate comprehensive summary
    summary = agent.generate_comprehensive_summary(univariate_results, hypothesis_results)
    
    # Display results
    print(f"✅ Summary generated successfully")
    print(f"Executive summary overview: {summary.get('executive_summary', {}).get('overview', {})}")
    print(f"High priority features: {len(summary.get('feature_engineering_recommendations', {}).get('high_priority_features', []))}")
    print(f"Key findings: {len(summary.get('executive_summary', {}).get('key_findings', []))}")
    
    return summary

def example_langgraph_usage():
    """Example of using the agent as a LangGraph node"""
    print("\n=== LangGraph Node Usage ===")
    
    # Create mock state
    state = {
        "target_variable": "sales_revenue",
        "domain_context": "retail",
        "univariate_results": MockUnivariateResults().get_mock_results(),
        "hypothesis_testing_results": MockHypothesisResults().get_mock_results(),
        "config": {
            "summarizer_config": {
                "modeling_objective": "predictive_analytics"
            }
        },
        "error_messages": []
    }
    
    # Execute as LangGraph node
    updated_state = summarizer_agent(state)
    
    # Display results
    print(f"✅ LangGraph node executed successfully")
    print(f"Execution status: {updated_state.get('execution_status')}")
    print(f"Current agent: {updated_state.get('current_agent')}")
    print(f"Final summary available: {'final_summary' in updated_state}")
    
    return updated_state

async def example_workflow_integration():
    """Example of integrating with the complete workflow"""
    print("\n=== Workflow Integration Example ===")
    
    # This would be used in the actual LangGraph workflow
    workflow_state = {
        "target_variable": "sales_revenue",
        "domain_context": "retail",
        "eda_request": "Analyze factors influencing sales revenue",
        "univariate_results": MockUnivariateResults().get_mock_results(),
        "hypothesis_testing_results": MockHypothesisResults().get_mock_results(),
        "config": {
            "summarizer_config": {
                "modeling_objective": "predictive_analytics",
                "synthesis_depth": "comprehensive"
            }
        },
        "error_messages": []
    }
    
    # Execute agent node
    updated_state = summarizer_agent(workflow_state)
    
    # Display workflow results
    print(f"✅ Workflow integration completed")
    print(f"Final state keys: {list(updated_state.keys())}")
    if "final_summary" in updated_state:
        summary = updated_state["final_summary"]
        print(f"Summary sections: {list(summary.keys())}")
        print(f"Implementation roadmap available: {'implementation_roadmap' in summary}")
    
    return updated_state

def example_feature_engineering_analysis():
    """Example of detailed feature engineering analysis"""
    print("\n=== Feature Engineering Analysis Example ===")
    
    # Initialize agent with specific domain context
    agent = SummarizerAgent(
        target_variable="units_sold",
        domain_context="retail",
        modeling_objective="demand_forecasting"
    )
    
    # Mock comprehensive results
    univariate_results = {
        "target_variable": {
            "profile": {"missing_percentage": 1.5, "skewness": 0.8},
            "anomalies": {"outliers_count": 8},
            "data_type": "numeric"
        },
        "data_quality_summary": {
            "overall_quality": "excellent",
            "issues_found": 0
        }
    }
    
    hypothesis_results = [
        {
            "hypothesis_id": "price_elasticity",
            "status": "success",
            "hypothesis": {"predictor_variable": {"name": "price", "transformation": "log"}},
            "result": {
                "correlation_analysis": {
                    "correlations": {
                        "pearson": {"correlation": -0.82, "significance": "significant"}
                    }
                },
                "transformations": {
                    "log_transformation": {"improved_correlation": True}
                }
            }
        }
    ]
    
    # Generate detailed analysis
    summary = agent.generate_comprehensive_summary(univariate_results, hypothesis_results)
    
    # Display feature engineering insights
    if "feature_engineering_recommendations" in summary:
        recommendations = summary["feature_engineering_recommendations"]
        print(f"High priority features: {len(recommendations.get('high_priority_features', []))}")
        print(f"Transformation recommendations: {len(recommendations.get('transformation_recommendations', []))}")
        print(f"Interaction features: {len(recommendations.get('interaction_features', []))}")
    
    return summary

if __name__ == "__main__":
    # Run examples
    print("SummarizerAgent Examples")
    print("=" * 50)
    
    # Standalone usage
    standalone_results = example_standalone_usage()
    
    # LangGraph node usage
    langgraph_results = example_langgraph_usage()
    
    # Workflow integration
    asyncio.run(example_workflow_integration())
    
    # Feature engineering analysis
    feature_analysis = example_feature_engineering_analysis()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
