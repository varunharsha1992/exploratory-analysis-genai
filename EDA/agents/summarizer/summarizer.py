from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from EDA.LLMS.llms import get_llm
from EDA.workflow.eda_workflow_state import EDAWorkflowState
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from EDA.agents.summarizer.summarizer_prompt import prompt
from utils.config_loader import AgentConfigLoader
from utils.helper import clean_messages_for_agent, msg_to_dict

class SummarizerAgent:
    def __init__(self, target_variable: str, domain_context: str = "", modeling_objective: str = "predictive_analytics"):
        """
        Initialize the Summarizer Agent
        
        Args:
            target_variable: The main variable being analyzed
            domain_context: Business domain context
            modeling_objective: The goal of the predictive model
        """
        self.target_variable = target_variable
        self.domain_context = domain_context
        self.modeling_objective = modeling_objective
        self.config_loader = AgentConfigLoader()
        
        # Load model configuration
        model_config = self.config_loader.get_model_config("summarizer")
        self.llm = get_llm(model_config['provider'], model_config['model'])
        self.prompt = self.config_loader.load_prompt("summarizer")
        
        # Agent-specific configuration
        self.config = self.config_loader.get_agent_config("summarizer")
        
        self.agent = create_react_agent(self.llm, [])
    
    def synthesize_findings(self, univariate_results: Dict[str, Any], hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize findings from all EDA agents"""
        
        synthesis = {
            "data_quality_summary": self.analyze_data_quality(univariate_results),
            "variable_relationships": self.analyze_relationships(hypothesis_results),
            "transformation_insights": self.analyze_transformations(hypothesis_results),
            "feature_importance": self.assess_feature_importance(hypothesis_results),
            "interaction_insights": self.analyze_interactions(hypothesis_results),
            "modeling_implications": self.assess_modeling_implications(hypothesis_results)
        }
        
        return synthesis
    
    def analyze_data_quality(self, univariate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall data quality from univariate analysis"""
        
        target_profile = univariate_results.get("target_variable", {})
        data_quality = univariate_results.get("data_quality_summary", {})
        
        return {
            "overall_quality": data_quality.get("overall_quality", "unknown"),
            "target_variable_quality": {
                "missing_percentage": target_profile.get("profile", {}).get("missing_percentage", 0),
                "outliers_count": target_profile.get("anomalies", {}).get("outliers_count", 0),
                "distribution_type": self.classify_distribution(target_profile),
                "data_type": target_profile.get("data_type", "unknown")
            },
            "quality_issues": data_quality.get("issues_found", 0),
            "recommendations": data_quality.get("recommendations", [])
        }
    
    def analyze_relationships(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze variable relationships from hypothesis testing"""
        
        relationship_analysis = {
            "strong_relationships": [],
            "moderate_relationships": [],
            "weak_relationships": [],
            "non_significant": [],
            "relationship_patterns": {}
        }
        
        if not hypothesis_results:
            return relationship_analysis
        
        for result in hypothesis_results:
            if result.get("status") == "success":
                correlation_data = result.get("result", {}).get("correlation_analysis", {})
                correlations = correlation_data.get("correlations", {})
                
                # Use Pearson correlation as primary measure
                pearson_corr = correlations.get("pearson", {}).get("correlation", 0)
                significance = correlations.get("pearson", {}).get("significance", "not_significant")
                
                relationship_info = {
                    "hypothesis_id": result.get("hypothesis_id"),
                    "predictor_variable": self.extract_predictor_variable(result),
                    "correlation_strength": abs(pearson_corr),
                    "correlation_direction": "positive" if pearson_corr > 0 else "negative",
                    "significance": significance,
                    "transformation": self.extract_transformation(result)
                }
                
                # Categorize by strength
                if abs(pearson_corr) >= 0.7:
                    relationship_analysis["strong_relationships"].append(relationship_info)
                elif abs(pearson_corr) >= 0.4:
                    relationship_analysis["moderate_relationships"].append(relationship_info)
                elif abs(pearson_corr) >= 0.2:
                    relationship_analysis["weak_relationships"].append(relationship_info)
                else:
                    relationship_analysis["non_significant"].append(relationship_info)
        
        return relationship_analysis
    
    def analyze_transformations(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transformation effectiveness from hypothesis testing"""
        
        transformation_insights = {
            "log_transformations_effective": False,
            "log_effective_variables": [],
            "polynomial_relationships": False,
            "polynomial_variables": [],
            "standardization_benefits": False
        }
        
        if not hypothesis_results:
            return transformation_insights
        
        for result in hypothesis_results:
            if result.get("status") == "success":
                transformations = result.get("result", {}).get("transformations", {})
                
                # Check for effective log transformations
                if transformations.get("log_transformation", {}).get("improved_correlation", False):
                    transformation_insights["log_transformations_effective"] = True
                    transformation_insights["log_effective_variables"].append(
                        self.extract_predictor_variable(result)
                    )
                
                # Check for polynomial relationships
                if transformations.get("polynomial_transformation", {}).get("improved_correlation", False):
                    transformation_insights["polynomial_relationships"] = True
                    transformation_insights["polynomial_variables"].append(
                        self.extract_predictor_variable(result)
                    )
        
        return transformation_insights
    
    def analyze_interactions(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction effects from hypothesis testing"""
        
        interaction_insights = {
            "high_correlation_interactions": [],
            "interaction_patterns": {},
            "domain_specific_interactions": []
        }
        
        if not hypothesis_results:
            return interaction_insights
        
        # Analyze for interaction patterns
        for result in hypothesis_results:
            if result.get("status") == "success":
                correlation_data = result.get("result", {}).get("correlation_analysis", {})
                correlations = correlation_data.get("correlations", {})
                
                pearson_corr = correlations.get("pearson", {}).get("correlation", 0)
                
                if abs(pearson_corr) >= 0.6:
                    interaction_insights["high_correlation_interactions"].append({
                        "features": [self.target_variable, self.extract_predictor_variable(result)],
                        "correlation": pearson_corr,
                        "significance": correlations.get("pearson", {}).get("significance", "not_significant")
                    })
        
        return interaction_insights
    
    def assess_feature_importance(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess feature importance based on hypothesis testing results"""
        
        feature_importance = {
            "high_importance_features": [],
            "medium_importance_features": [],
            "low_importance_features": [],
            "importance_distribution": {}
        }
        
        if not hypothesis_results:
            return feature_importance
        
        for result in hypothesis_results:
            if result.get("status") == "success":
                correlation_data = result.get("result", {}).get("correlation_analysis", {})
                correlations = correlation_data.get("correlations", {})
                
                pearson_corr = correlations.get("pearson", {}).get("correlation", 0)
                predictor_var = self.extract_predictor_variable(result)
                
                importance_info = {
                    "feature_name": predictor_var,
                    "correlation_strength": abs(pearson_corr),
                    "correlation_direction": "positive" if pearson_corr > 0 else "negative",
                    "significance": correlations.get("pearson", {}).get("significance", "not_significant")
                }
                
                if abs(pearson_corr) >= 0.6:
                    feature_importance["high_importance_features"].append(importance_info)
                elif abs(pearson_corr) >= 0.3:
                    feature_importance["medium_importance_features"].append(importance_info)
                else:
                    feature_importance["low_importance_features"].append(importance_info)
        
        return feature_importance
    
    def assess_modeling_implications(self, hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess modeling implications from analysis results"""
        
        if not hypothesis_results:
            return {
                "algorithm_recommendation": "linear",
                "feature_selection_strategy": "correlation_based",
                "validation_strategy": "stratified",
                "complexity_assessment": "low",
                "expected_performance": "medium"
            }
        
        strong_relationships = sum(1 for result in hypothesis_results 
                                if result.get("status") == "success" and 
                                abs(result.get("result", {}).get("correlation_analysis", {})
                                    .get("correlations", {}).get("pearson", {}).get("correlation", 0)) >= 0.7)
        
        return {
            "algorithm_recommendation": "ensemble" if strong_relationships > 5 else "linear",
            "feature_selection_strategy": "correlation_based",
            "validation_strategy": "time_based" if self.domain_context in ["retail", "finance"] else "stratified",
            "complexity_assessment": "high" if strong_relationships > 5 else "medium",
            "expected_performance": "high" if strong_relationships > 3 else "medium"
        }
    
    def generate_feature_engineering_recommendations(self, synthesis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive feature engineering recommendations"""
        
        recommendations = {
            "high_priority_features": [],
            "transformation_recommendations": [],
            "interaction_features": [],
            "aggregation_features": [],
            "derived_features": [],
            "feature_selection_guidance": {},
            "modeling_considerations": []
        }
        
        # Analyze strong relationships for high-priority features
        strong_relationships = synthesis_results["variable_relationships"]["strong_relationships"]
        
        for relationship in strong_relationships:
            feature_rec = {
                "feature_name": relationship["predictor_variable"],
                "priority": "high",
                "reason": f"Strong {relationship['correlation_direction']} correlation ({relationship['correlation_strength']:.3f})",
                "transformation": relationship["transformation"],
                "expected_impact": "high",
                "implementation_notes": self.get_implementation_notes(relationship)
            }
            recommendations["high_priority_features"].append(feature_rec)
        
        # Generate transformation recommendations
        transformation_insights = synthesis_results["transformation_insights"]
        recommendations["transformation_recommendations"] = self.recommend_transformations(transformation_insights)
        
        # Generate interaction features
        interaction_insights = synthesis_results["interaction_insights"]
        recommendations["interaction_features"] = self.recommend_interactions(interaction_insights)
        
        # Generate aggregation features
        recommendations["aggregation_features"] = self.recommend_aggregations(synthesis_results)
        
        # Generate derived features
        recommendations["derived_features"] = self.recommend_derived_features(synthesis_results)
        
        return recommendations
    
    def recommend_transformations(self, transformation_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend specific transformations based on analysis"""
        
        recommendations = []
        
        # Log transformations
        if transformation_insights.get("log_transformations_effective", False):
            recommendations.append({
                "transformation_type": "log",
                "variables": transformation_insights.get("log_effective_variables", []),
                "reason": "Log transformation improved correlation strength",
                "implementation": "np.log1p(variable) to handle zeros",
                "expected_benefit": "Improved linear relationships and model performance"
            })
        
        # Polynomial transformations
        if transformation_insights.get("polynomial_relationships", False):
            recommendations.append({
                "transformation_type": "polynomial",
                "variables": transformation_insights.get("polynomial_variables", []),
                "reason": "Non-linear relationships detected",
                "implementation": "variable^2, variable^3 for polynomial features",
                "expected_benefit": "Capture non-linear relationships"
            })
        
        # Standardization
        recommendations.append({
            "transformation_type": "standardization",
            "variables": "all_numeric_features",
            "reason": "Improve model convergence and interpretability",
            "implementation": "StandardScaler from sklearn",
            "expected_benefit": "Better model performance and feature comparison"
        })
        
        return recommendations
    
    def recommend_interactions(self, interaction_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend interaction features based on analysis"""
        
        recommendations = []
        
        # High-correlation interactions
        high_corr_interactions = interaction_insights.get("high_correlation_interactions", [])
        for interaction in high_corr_interactions:
            recommendations.append({
                "interaction_type": "multiplicative",
                "features": interaction["features"],
                "reason": f"High correlation interaction detected ({interaction['correlation']:.3f})",
                "implementation": f"{interaction['features'][0]} * {interaction['features'][1]}",
                "expected_benefit": "Capture synergistic effects between variables"
            })
        
        # Domain-specific interactions
        domain_interactions = self.get_domain_specific_interactions()
        recommendations.extend(domain_interactions)
        
        return recommendations
    
    def recommend_aggregations(self, synthesis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend aggregation features based on analysis"""
        
        recommendations = []
        
        # Temporal aggregations for time-series data
        if self.domain_context in ["retail", "finance", "marketing"]:
            recommendations.append({
                "feature_type": "temporal_aggregation",
                "variables": [self.target_variable],
                "aggregation": "rolling_mean_7",
                "reason": "Capture weekly trends",
                "implementation": f"{self.target_variable}.rolling(7).mean()",
                "expected_benefit": "Smooth short-term fluctuations"
            })
        
        return recommendations
    
    def recommend_derived_features(self, synthesis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend derived features based on domain knowledge"""
        
        recommendations = []
        
        # Domain-specific derived features
        if self.domain_context == "retail":
            recommendations.extend([
                {
                    "feature_name": "price_elasticity",
                    "formula": "log(sales) / log(price)",
                    "reason": "Economic theory suggests price elasticity relationship",
                    "expected_benefit": "Capture price sensitivity"
                },
                {
                    "feature_name": "customer_lifetime_value",
                    "formula": "avg_order_value * purchase_frequency * customer_age",
                    "reason": "Domain knowledge suggests CLV importance",
                    "expected_benefit": "Capture customer value"
                }
            ])
        
        return recommendations
    
    def generate_executive_summary(self, synthesis_results: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of EDA findings"""
        
        summary = {
            "overview": {
                "target_variable": self.target_variable,
                "total_variables_analyzed": self.count_analyzed_variables(synthesis_results),
                "significant_relationships_found": len(synthesis_results["variable_relationships"]["strong_relationships"]),
                "data_quality_score": self.calculate_data_quality_score(synthesis_results),
                "analysis_confidence": self.assess_analysis_confidence(synthesis_results)
            },
            "key_findings": self.extract_key_findings(synthesis_results),
            "critical_insights": self.identify_critical_insights(synthesis_results),
            "feature_engineering_priority": self.prioritize_feature_engineering(recommendations),
            "modeling_recommendations": self.generate_modeling_recommendations(synthesis_results),
            "next_steps": self.recommend_next_steps(synthesis_results, recommendations)
        }
        
        return summary
    
    def generate_comprehensive_summary(self, univariate_results: Dict[str, Any], hypothesis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive summary of EDA findings and recommendations"""
        
        try:
            # Synthesize findings from all agents
            synthesis_results = self.synthesize_findings(univariate_results, hypothesis_results)
            
            # Generate feature engineering recommendations
            recommendations = self.generate_feature_engineering_recommendations(synthesis_results)
            
            # Generate executive summary
            executive_summary = self.generate_executive_summary(synthesis_results, recommendations)
            
            # Compile final comprehensive summary
            comprehensive_summary = {
                "executive_summary": executive_summary,
                "feature_engineering_recommendations": recommendations,
                "synthesis_results": synthesis_results,
                "implementation_roadmap": self.create_implementation_roadmap(recommendations),
                "success_metrics": self.define_success_metrics(synthesis_results),
                "next_steps": self.recommend_next_steps(synthesis_results, recommendations)
            }
            
            return comprehensive_summary
            
        except Exception as e:
            logging.error(f"Comprehensive summary generation failed: {str(e)}")
            return {
                "error": f"Summary generation failed: {str(e)}",
                "executive_summary": {"overview": {"analysis_confidence": "low"}},
                "feature_engineering_recommendations": {"high_priority_features": []},
                "synthesis_results": {}
            }
    
    def process(self, state: EDAWorkflowState) -> EDAWorkflowState:
        """
        Process method for LangGraph integration
        
        Args:
            state: EDAWorkflowState containing workflow state
            
        Returns:
            Updated state with final summary and recommendations
        """
        try:
            logging.info("Starting summarization and recommendation generation")
            
            # Update state with current agent
            state["current_agent"] = "summarizer"
            state["execution_status"] = "running"
            print(f"Input to summarizer: {state}")
            univariate_results = state.get("univariate_results", {})
            hypothesis_testing_results = state.get("hypothesis_testing_results", [])
            
            
            # Ensure hypothesis_testing_results is a list
            if hypothesis_testing_results is None:
                hypothesis_testing_results = []
            
            summary = self.generate_comprehensive_summary(univariate_results, hypothesis_testing_results)
            
            if "error" in summary:
                state["error_messages"] = state.get("error_messages", [])
                state["error_messages"].append(f"Summarizer agent failed: {summary['error']}")
                state["execution_status"] = "failed"
                print(f"Summarizer agent failed: {summary['error']}")
            else:
                state["final_summary"] = summary
                state["execution_status"] = "completed"
                logging.info("Summarization and recommendation generation completed successfully")
            
            state["timestamp"] = datetime.now().isoformat()
            return state
            
        except Exception as e:
            logging.error(f"Summarizer agent processing failed: {str(e)}")
            state["error_messages"] = state.get("error_messages", [])
            state["error_messages"].append(f"Summarizer agent failed: {str(e)}")
            state["execution_status"] = "failed"
            return state
    
    # Helper methods
    def classify_distribution(self, target_profile: Dict[str, Any]) -> str:
        """Classify the distribution type of the target variable"""
        profile = target_profile.get("profile", {})
        skewness = profile.get("skewness", 0)
        
        if abs(skewness) < 0.5:
            return "normal"
        elif abs(skewness) < 1.0:
            return "moderately_skewed"
        else:
            return "highly_skewed"
    
    def extract_predictor_variable(self, result: Dict[str, Any]) -> str:
        """Extract predictor variable name from hypothesis result"""
        return result.get("hypothesis", {}).get("predictor_variable", {}).get("name", "unknown")
    
    def extract_transformation(self, result: Dict[str, Any]) -> str:
        """Extract transformation type from hypothesis result"""
        return result.get("hypothesis", {}).get("predictor_variable", {}).get("transformation", "none")
    
    def get_implementation_notes(self, relationship: Dict[str, Any]) -> str:
        """Get implementation notes for a relationship"""
        return f"Apply {relationship['transformation']} transformation, consider interaction effects"
    
    def get_domain_specific_interactions(self) -> List[Dict[str, Any]]:
        """Get domain-specific interaction recommendations"""
        if self.domain_context == "retail":
            return [
                {
                    "interaction_type": "multiplicative",
                    "features": ["price", "promotion"],
                    "reason": "Retail domain knowledge suggests price-promotion interaction",
                    "implementation": "price * promotion",
                    "expected_benefit": "Capture promotional price effects"
                }
            ]
        return []
    
    def count_analyzed_variables(self, synthesis_results: Dict[str, Any]) -> int:
        """Count total variables analyzed"""
        relationships = synthesis_results["variable_relationships"]
        return (len(relationships["strong_relationships"]) + 
                len(relationships["moderate_relationships"]) + 
                len(relationships["weak_relationships"]))
    
    def calculate_data_quality_score(self, synthesis_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        data_quality = synthesis_results["data_quality_summary"]
        missing_pct = data_quality["target_variable_quality"]["missing_percentage"]
        outliers = data_quality["target_variable_quality"]["outliers_count"]
        
        # Simple scoring: lower missing data and outliers = higher score
        score = 1.0 - (missing_pct / 100) - min(outliers / 1000, 0.2)
        return max(score, 0.0)
    
    def assess_analysis_confidence(self, synthesis_results: Dict[str, Any]) -> str:
        """Assess confidence in analysis results"""
        strong_rels = len(synthesis_results["variable_relationships"]["strong_relationships"])
        data_quality_score = self.calculate_data_quality_score(synthesis_results)
        
        if strong_rels >= 3 and data_quality_score >= 0.8:
            return "high"
        elif strong_rels >= 1 and data_quality_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def extract_key_findings(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        findings = []
        
        # Data quality findings
        data_quality = synthesis_results["data_quality_summary"]
        if data_quality["target_variable_quality"]["missing_percentage"] > 5:
            findings.append(f"Target variable has {data_quality['target_variable_quality']['missing_percentage']:.1f}% missing values - requires imputation strategy")
        
        # Relationship findings
        relationships = synthesis_results["variable_relationships"]
        if relationships["strong_relationships"]:
            strongest = max(relationships["strong_relationships"], key=lambda x: x["correlation_strength"])
            findings.append(f"Strongest relationship found: {strongest['predictor_variable']} (correlation: {strongest['correlation_strength']:.3f})")
        
        # Transformation findings
        transformations = synthesis_results["transformation_insights"]
        if transformations.get("log_transformations_effective"):
            findings.append("Log transformations significantly improve variable relationships")
        
        return findings
    
    def identify_critical_insights(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Identify critical insights from analysis"""
        insights = []
        
        relationships = synthesis_results["variable_relationships"]
        if relationships["strong_relationships"]:
            insights.append("Multiple strong predictors identified - consider ensemble methods")
        
        data_quality = synthesis_results["data_quality_summary"]
        if data_quality["target_variable_quality"]["missing_percentage"] > 10:
            insights.append("High missing data requires robust imputation strategy")
        
        return insights
    
    def prioritize_feature_engineering(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize feature engineering recommendations"""
        return {
            "phase_1": {
                "description": "High-priority feature implementation",
                "features": [f["feature_name"] for f in recommendations["high_priority_features"]],
                "estimated_effort": "2-3 days",
                "expected_impact": "high"
            },
            "phase_2": {
                "description": "Interaction and aggregation features",
                "features": [f["features"] for f in recommendations["interaction_features"]],
                "estimated_effort": "1-2 days",
                "expected_impact": "medium"
            }
        }
    
    def generate_modeling_recommendations(self, synthesis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific modeling recommendations"""
        recommendations = []
        
        # Algorithm recommendations
        relationships = synthesis_results["variable_relationships"]
        if len(relationships["strong_relationships"]) > 5:
            recommendations.append({
                "category": "algorithm_selection",
                "recommendation": "Consider ensemble methods (Random Forest, XGBoost) due to multiple strong relationships",
                "rationale": "Multiple strong predictors suggest complex relationships that ensemble methods can capture"
            })
        else:
            recommendations.append({
                "category": "algorithm_selection", 
                "recommendation": "Start with linear models (Ridge, Lasso) for interpretability",
                "rationale": "Fewer strong relationships suggest linear models may be sufficient"
            })
        
        # Feature selection recommendations
        recommendations.append({
            "category": "feature_selection",
            "recommendation": "Use correlation-based feature selection with threshold 0.3",
            "rationale": "Focus on variables with moderate to strong correlations"
        })
        
        return recommendations
    
    def create_implementation_roadmap(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation roadmap for feature engineering"""
        return {
            "phase_1": {
                "description": "High-priority feature implementation",
                "features": [f["feature_name"] for f in recommendations["high_priority_features"]],
                "estimated_effort": "2-3 days",
                "expected_impact": "high"
            },
            "phase_2": {
                "description": "Interaction and aggregation features",
                "features": [f["features"] for f in recommendations["interaction_features"]],
                "estimated_effort": "1-2 days",
                "expected_impact": "medium"
            }
        }
    
    def define_success_metrics(self, synthesis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Define success metrics for the modeling project"""
        return {
            "model_performance_targets": {
                "rmse": "< 1000",
                "mae": "< 750", 
                "r2_score": "> 0.75"
            },
            "feature_importance_thresholds": {
                "top_5_features": "> 80% of total importance",
                "price_features": "> 30% of total importance"
            }
        }
    
    def recommend_next_steps(self, synthesis_results: Dict[str, Any], recommendations: Dict[str, Any]) -> List[str]:
        """Recommend next steps for the modeling project"""
        return [
            "Implement high-priority features",
            "Set up automated feature engineering pipeline",
            "Begin model training with recommended algorithms",
            "Validate feature importance matches EDA findings",
            "Monitor model performance against success metrics"
        ]

def summarizer_agent(state: EDAWorkflowState):
    """
    LangGraph node function for Summarizer agent
    
    Args:
        state: EDAWorkflowState containing workflow state
        
    Returns:
        Updated state with summarizer results
    """
    # Extract configuration from state
    config = state.get("config", {})
    
    # Get agent-specific configuration
    summarizer_config = config.get("summarizer_config", {})
    
    # Initialize agent
    agent = SummarizerAgent(
        target_variable=state.get("target_variable", ""),
        domain_context=state.get("domain_context", ""),
        modeling_objective=summarizer_config.get("modeling_objective", "predictive_analytics")
    )
    
    # Process state
    return agent.process(state)