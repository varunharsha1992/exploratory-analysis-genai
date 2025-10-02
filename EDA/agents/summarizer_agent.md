# Summarizer Agent

## Overview
The Summarizer Agent is the final component of the EDA workflow that synthesizes all findings from the previous agents and provides comprehensive feature engineering recommendations. It analyzes results from univariate analysis, hypothesis testing, and correlation analysis to generate actionable insights for predictive modeling.

## Agent Specification

### Purpose
- Synthesize findings from all previous EDA agents
- Generate comprehensive feature engineering recommendations
- Provide actionable insights for predictive modeling
- Create executive summary of EDA results
- Recommend specific transformations and feature combinations

### Input Requirements
- **Univariate Analysis Results**: Output from Univariate Analysis Agent
- **Hypothesis Testing Results**: Aggregated results from EDA Worker Loop Agent
- **Target Variable**: The variable being predicted
- **EDA Request**: Original analysis requirements
- **Domain Context**: Business domain and modeling objectives

### Core Capabilities

#### 1. Results Synthesis and Analysis
```python
class SummarizerAgent:
    def __init__(self, target_variable, domain_context, modeling_objective):
        self.target_variable = target_variable
        self.domain_context = domain_context
        self.modeling_objective = modeling_objective
        
    def synthesize_findings(self, univariate_results, hypothesis_results):
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
    
    def analyze_data_quality(self, univariate_results):
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
    
    def analyze_relationships(self, hypothesis_results):
        """Analyze variable relationships from hypothesis testing"""
        
        successful_results = hypothesis_results.get("successful_results", [])
        
        relationship_analysis = {
            "strong_relationships": [],
            "moderate_relationships": [],
            "weak_relationships": [],
            "non_significant": [],
            "relationship_patterns": {}
        }
        
        for result in successful_results:
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
```

#### 2. Feature Engineering Recommendations
```python
def generate_feature_engineering_recommendations(self, synthesis_results):
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

def recommend_transformations(self, transformation_insights):
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

def recommend_interactions(self, interaction_insights):
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
```

#### 3. Executive Summary Generation
```python
def generate_executive_summary(self, synthesis_results, recommendations):
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

def extract_key_findings(self, synthesis_results):
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

def generate_modeling_recommendations(self, synthesis_results):
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
    
    # Cross-validation recommendations
    recommendations.append({
        "category": "validation_strategy",
        "recommendation": "Use time-based split if temporal data, otherwise stratified k-fold",
        "rationale": "Ensure validation strategy matches data characteristics"
    })
    
    return recommendations
```

### Tools and Dependencies

#### Core Dependencies
- **Data Analysis**: Pandas, NumPy for data manipulation
- **Statistical Analysis**: SciPy for statistical computations
- **Visualization**: Matplotlib, Seaborn for summary charts
- **Report Generation**: Custom templates for executive summaries

#### External Tools
- **LLM Integration**: For generating natural language summaries
- **Chart Generation**: Plotly for executive dashboards
- **Report Export**: PDF/HTML generation for final reports

### Output Format

```python
{
    "executive_summary": {
        "overview": {
            "target_variable": "sales_revenue",
            "total_variables_analyzed": 15,
            "significant_relationships_found": 8,
            "data_quality_score": 0.85,
            "analysis_confidence": "high"
        },
        "key_findings": [
            "Strongest relationship: price (correlation: -0.75)",
            "Log transformations significantly improve relationships",
            "Target variable has 2.1% missing values",
            "8 variables show strong predictive potential"
        ],
        "critical_insights": [
            "Price elasticity is the dominant factor",
            "Seasonal patterns require temporal features",
            "Customer segmentation shows strong interaction effects"
        ]
    },
    "feature_engineering_recommendations": {
        "high_priority_features": [
            {
                "feature_name": "price",
                "priority": "high",
                "reason": "Strong negative correlation (-0.75)",
                "transformation": "log",
                "expected_impact": "high",
                "implementation_notes": "Apply log transformation, consider price elasticity features"
            },
            {
                "feature_name": "customer_satisfaction",
                "priority": "high", 
                "reason": "Strong positive correlation (0.68)",
                "transformation": "lag_1",
                "expected_impact": "high",
                "implementation_notes": "Use lagged values, create satisfaction trend features"
            }
        ],
        "transformation_recommendations": [
            {
                "transformation_type": "log",
                "variables": ["price", "sales_volume"],
                "reason": "Log transformation improved correlation strength",
                "implementation": "np.log1p(variable)",
                "expected_benefit": "Improved linear relationships"
            },
            {
                "transformation_type": "standardization",
                "variables": "all_numeric_features",
                "reason": "Improve model convergence",
                "implementation": "StandardScaler from sklearn",
                "expected_benefit": "Better model performance"
            }
        ],
        "interaction_features": [
            {
                "interaction_type": "multiplicative",
                "features": ["price", "promotion"],
                "reason": "High correlation interaction detected (0.65)",
                "implementation": "price * promotion",
                "expected_benefit": "Capture promotional price effects"
            }
        ],
        "aggregation_features": [
            {
                "feature_type": "temporal_aggregation",
                "variables": ["sales_revenue"],
                "aggregation": "rolling_mean_7",
                "reason": "Capture weekly trends",
                "implementation": "sales_revenue.rolling(7).mean()",
                "expected_benefit": "Smooth short-term fluctuations"
            }
        ],
        "derived_features": [
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
        ]
    },
    "modeling_recommendations": [
        {
            "category": "algorithm_selection",
            "recommendation": "Use XGBoost for initial modeling",
            "rationale": "Multiple strong relationships and potential non-linear effects",
            "implementation_notes": "Start with default parameters, tune learning rate and depth"
        },
        {
            "category": "feature_selection",
            "recommendation": "Use correlation-based selection with 0.3 threshold",
            "rationale": "Focus on variables with meaningful relationships",
            "implementation_notes": "Remove features with correlation < 0.3 to target"
        },
        {
            "category": "validation_strategy",
            "recommendation": "Use time-based split for temporal validation",
            "rationale": "Sales data shows temporal patterns",
            "implementation_notes": "Use last 20% of data for testing"
        }
    ],
    "implementation_roadmap": {
        "phase_1": {
            "description": "High-priority feature implementation",
            "features": ["price_log", "customer_satisfaction_lag", "price_elasticity"],
            "estimated_effort": "2-3 days",
            "expected_impact": "high"
        },
        {
            "description": "Interaction and aggregation features",
            "features": ["price_promotion_interaction", "rolling_sales_mean"],
            "estimated_effort": "1-2 days", 
            "expected_impact": "medium"
        },
        {
            "description": "Advanced derived features",
            "features": ["customer_lifetime_value", "seasonal_features"],
            "estimated_effort": "2-3 days",
            "expected_impact": "medium"
        }
    },
    "success_metrics": {
        "model_performance_targets": {
            "rmse": "< 1000",
            "mae": "< 750", 
            "r2_score": "> 0.75"
        },
        "feature_importance_thresholds": {
            "top_5_features": "> 80% of total importance",
            "price_features": "> 30% of total importance"
        }
    },
    "next_steps": [
        "Implement high-priority features (price, satisfaction)",
        "Set up automated feature engineering pipeline",
        "Begin model training with recommended algorithms",
        "Validate feature importance matches EDA findings",
        "Monitor model performance against success metrics"
    ]
}
```

### Agent Behavior

#### 1. Initialization
- Load all EDA results from previous agents
- Initialize synthesis and recommendation engines
- Set up report generation templates
- Configure domain-specific knowledge

#### 2. Results Synthesis
- Analyze data quality across all variables
- Synthesize relationship patterns and trends
- Identify transformation effectiveness
- Assess feature importance and interactions

#### 3. Recommendation Generation
- Generate feature engineering recommendations
- Prioritize features by expected impact
- Create implementation roadmaps
- Define success metrics and targets

#### 4. Report Compilation
- Generate executive summary
- Create detailed implementation guides
- Compile visualization dashboards
- Export comprehensive reports

### Error Handling
- Handle missing or incomplete EDA results
- Manage synthesis errors gracefully
- Provide fallback recommendations
- Ensure report generation completes successfully

### Performance Considerations
- Optimize synthesis algorithms for large result sets
- Cache frequently accessed analysis results
- Use efficient data structures for recommendations
- Implement parallel processing for report generation

### Integration Points
- **Input**: Receives aggregated results from EDA Worker Loop Agent
- **Output**: Provides comprehensive feature engineering recommendations
- **Tools**: Integrates with report generation and visualization tools
- **State**: Finalizes workflow state with complete analysis

### Example Usage
```python
# Initialize Summarizer Agent
summarizer = SummarizerAgent(
    target_variable="sales_revenue",
    domain_context="retail",
    modeling_objective="predictive_analytics"
)

# Generate comprehensive summary
summary = summarizer.generate_comprehensive_summary(
    univariate_results=univariate_analysis_results,
    hypothesis_results=hypothesis_testing_results
)

# Access recommendations
feature_recommendations = summary["feature_engineering_recommendations"]
modeling_recommendations = summary["modeling_recommendations"]
implementation_roadmap = summary["implementation_roadmap"]
```

### Success Criteria
- Successfully synthesize all EDA findings
- Generate actionable feature engineering recommendations
- Provide clear implementation guidance
- Create comprehensive executive summary
- Deliver specific modeling recommendations with rationale
