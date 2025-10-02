# Univariate Analysis Agent

## Overview
The Univariate Analysis Agent performs comprehensive univariate analysis on the target variable and related features. It serves as the foundation for the EDA workflow by providing data profiling, anomaly detection, and key trend analysis.

## Agent Specification

### Purpose
- Perform basic univariate analysis on target variable and key features
- Identify anomalies, outliers, and data quality issues
- Analyze key trends and patterns in the target variable
- Generate data profiles and statistical summaries
- Prepare foundation for hypothesis generation

### Input Requirements
- **Target Variable**: The variable to be predicted (e.g., "Sales", "Customer_Churn")
- **EDA Request**: High-level description of the analysis goal
- **Data Context**: Domain information and business context
- **Intugle Knowledge Base**: Pre-built semantic layer with `kb.links` and `kb.search()` capabilities

### Core Capabilities

#### 1. Data Profiling
```python
# Access Intugle tools
from intugle import KnowledgeBuilder, DataProductBuilder

# Generate comprehensive data profiles
def generate_data_profile(target_variable, kb):
    # Use kb.search() to find related variables
    related_vars = kb.search(f"variables related to {target_variable}")
    
    # Create data product for profiling
    dp_builder = DataProductBuilder()
    
    # Profile target variable
    target_profile = {
        "variable": target_variable,
        "data_type": "continuous/categorical",
        "missing_values": "count and percentage",
        "unique_values": "count and percentage",
        "statistical_summary": "mean, median, std, min, max, quartiles",
        "distribution": "histogram, box plot, density plot"
    }
    
    return target_profile
```

#### 2. Anomaly Detection
```python
def detect_anomalies(target_variable, data):
    anomalies = {
        "outliers": "statistical outliers using IQR method",
        "extreme_values": "values beyond 3 standard deviations",
        "missing_patterns": "patterns in missing data",
        "data_quality_issues": "inconsistencies and errors",
        "temporal_anomalies": "unusual patterns over time (if applicable)"
    }
    return anomalies
```

#### 3. Key Trend Analysis
```python
def analyze_trends(target_variable, data):
    trends = {
        "temporal_trends": "time series analysis if date columns exist",
        "seasonality": "seasonal patterns and cycles",
        "growth_patterns": "growth rates and trends",
        "volatility": "variance and stability measures",
        "correlation_with_time": "relationship with temporal features"
    }
    return trends
```

### Tools and Dependencies

#### Intugle Integration
- **`kb.search(query)`**: Semantic search for related variables and KPIs
- **`kb.links`**: Access predicted relationships between tables
- **`DataProductBuilder`**: Generate datasets for analysis
- **`SemanticSearch`**: Find relevant columns and tables

#### Additional Tools
- **Anomaly Detection**: Statistical methods (IQR, Z-score, Isolation Forest)
- **Statistical Analysis**: Pandas, NumPy, SciPy for statistical computations
- **Visualization**: Matplotlib, Seaborn, Plotly for trend visualization
- **Data Quality**: Custom functions for data quality assessment

### Output Format

```python
{
    "target_variable": {
        "name": "target_variable_name",
        "data_type": "continuous/categorical",
        "profile": {
            "count": 10000,
            "missing": 150,
            "missing_percentage": 1.5,
            "unique_values": 8500,
            "statistical_summary": {
                "mean": 125.5,
                "median": 120.0,
                "std": 45.2,
                "min": 10.0,
                "max": 500.0,
                "q25": 85.0,
                "q75": 165.0
            }
        },
        "anomalies": {
            "outliers_count": 25,
            "extreme_values": [600, 650, 700],
            "missing_patterns": "random",
            "data_quality_issues": []
        },
        "trends": {
            "temporal_trend": "increasing",
            "seasonality": "monthly_pattern",
            "volatility": "moderate",
            "growth_rate": 0.05
        }
    },
    "related_variables": [
        {
            "name": "related_var_1",
            "relationship": "strongly_correlated",
            "correlation": 0.85,
            "profile": {...}
        }
    ],
    "data_quality_summary": {
        "overall_quality": "good",
        "issues_found": 3,
        "recommendations": [
            "Handle 25 outliers in target variable",
            "Investigate missing values in feature X",
            "Consider log transformation for skewed distribution"
        ]
    }
}
```

### Agent Behavior

#### 1. Initialization
- Load Intugle knowledge base
- Initialize data profiling tools
- Set up anomaly detection parameters
- Configure visualization settings

#### 2. Analysis Execution
- Search for variables related to target using semantic search
- Generate comprehensive data profiles
- Detect anomalies and data quality issues
- Analyze trends and patterns
- Create visualizations for key findings

#### 3. Output Generation
- Compile statistical summaries
- Generate anomaly reports
- Create trend analysis
- Provide data quality recommendations
- Prepare findings for hypothesis generation

### Error Handling
- Handle missing or invalid target variables
- Manage data access issues
- Handle computational errors in statistical analysis
- Provide fallback options for visualization failures

### Performance Considerations
- Use efficient data sampling for large datasets
- Implement caching for repeated analyses
- Optimize visualization generation
- Handle memory constraints for large datasets

### Integration Points
- **Input**: Receives target variable and EDA request from workflow
- **Output**: Provides comprehensive univariate analysis to Hypothesis Generation Agent
- **Tools**: Integrates with Intugle semantic layer and custom analysis tools
- **State**: Updates workflow state with analysis results

### Example Usage
```python
# Initialize agent
univariate_agent = UnivariateAnalysisAgent(
    target_variable="sales_revenue",
    kb=knowledge_builder,
    config=analysis_config
)

# Execute analysis
results = univariate_agent.analyze()

# Access results
target_profile = results["target_variable"]
anomalies = results["anomalies"]
trends = results["trends"]
```

### Success Criteria
- Complete data profile generated for target variable
- All anomalies identified and documented
- Key trends analyzed and visualized
- Data quality issues identified with recommendations
- Results formatted for downstream agents
