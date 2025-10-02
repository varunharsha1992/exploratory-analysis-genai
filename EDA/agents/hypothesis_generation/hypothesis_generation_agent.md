# Hypothesis Generation Agent

## Overview
The Hypothesis Generation Agent is a RAG + Web Search powered agent that creates intelligent hypotheses about what variables might influence the target variable. It combines domain knowledge from the semantic layer with external research to generate data-driven hypotheses with specific transformation recommendations.

## Agent Specification

### Purpose
- Generate hypotheses about influencing variables for the target variable
- Provide specific transformation and aggregation recommendations
- Identify interaction features and complex relationships
- Limit hypotheses based on user requirements
- Create actionable hypotheses for downstream testing

### Input Requirements
- **Target Variable**: The variable to be predicted
- **Univariate Analysis Results**: Output from Univariate Analysis Agent
- **EDA Request**: High-level description of analysis goals
- **Hypothesis Limit**: Maximum number of hypotheses to generate (user-specified)
- **Domain Context**: Business domain and context information
- **Intugle Knowledge Base**: Semantic layer with `kb.links` and `kb.search()` capabilities

### Core Capabilities

#### 1. Semantic Variable Discovery
```python
def discover_related_variables(target_variable, kb):
    """Use Intugle semantic search to find related variables"""
    
    # Search for variables semantically related to target
    related_vars = kb.search(f"variables that influence {target_variable}")
    
    # Search for domain-specific KPIs
    kpi_vars = kb.search(f"KPIs related to {target_variable}")
    
    # Get predicted relationships
    relationships = kb.links
    
    # Find variables with strong predicted links
    linked_vars = []
    for link in relationships:
        if target_variable in [link.source_field_id, link.target_field_id]:
            linked_vars.append(link)
    
    return {
        "semantic_matches": related_vars,
        "kpi_matches": kpi_vars,
        "predicted_links": linked_vars
    }
```

#### 2. Web Search Integration
```python
def research_domain_knowledge(target_variable, domain):
    """Use web search to find domain-specific insights"""
    
    search_queries = [
        f"what factors influence {target_variable} in {domain}",
        f"{target_variable} predictive analytics {domain}",
        f"feature engineering for {target_variable} prediction",
        f"{domain} {target_variable} influencing variables"
    ]
    
    research_results = []
    for query in search_queries:
        results = web_search(query)
        research_results.extend(results)
    
    return research_results
```

#### 3. Hypothesis Generation Engine

#### 4. Transformation Recommendations
```python
def get_transformation_for_type(relationship_type):
    f"""Map relationship types to specific transformations. Loop through individual influencing variables and identify key transformations that impact the target variable. E.x. target variable sales and infleuncing variable price are typically through log-log. Use results of LLM + websearch tools + link prediction from search_variables. Use output search_profiles tool refer to and get exact table and column names
    
    example output:
    transformations = [
        {{"influencing variable": "price.price" [table_name.column_name]
        "target variable": "Sales"
        "relationship": "log_log""}},
        {{"influencing variable": "discount"
        "target variable": "Sales" [table_name.column_name]
        "relationship": "log_linear""}}
    """
    
    return transformation_map


### Tools and Dependencies

#### Intugle Integration
- **`tools.intugle_agent_tools.variable_search(query)`**: Semantic search for related variables and KPIs
- **`tools.intugle_agent_tools.filer_kb`**: Access tables, columns and 
- **tools.intugle_agent_tools.filer_kb.search_variables: get predicted relationships between tables containing target and influencing variables
-  **tools.get_variable_profiles: to get more information on the variables of interest

#### External Tools
- **Web Search**: Google Search API, Bing Search API, or similar
- **RAG System**: Vector database with domain knowledge embeddings. This needs to have a seperate utilitly that can take documents from a research database, chunk them and create embeddings in Qdrant Database. And a tool needs to be created to search through RAG system, similar to research_domain_knowledge

### Output Format

```python
{
    "hypotheses": [
        {
            "hypothesis_id": "hyp_1",
            "hypothesis": "Price has a log-log relationship with Sales",
            "target_variable": {
                "name": "sales",
                "transformation": "log",
                "aggregate_by": "product"
            },
            "predictor_variable": {
                "name": "price", 
                "transformation": "log",
                "aggregate_by": "product"
            },
            "relationship_type": "log_log_relationship",
            "expected_impact": "negative",
            "confidence": 0.85,
            "research_support": [
                "Price elasticity studies show log-log relationships",
                "Economic theory supports logarithmic price-sales relationships"
            ],
            "interaction_features": [
                "price * promotion",
                "price * seasonality",
                "price * customer_segment"
            ],
            "test_priority": 0.9,
            "data_requirements": {
                "required_tables": ["sales", "products", "pricing"],
                "required_columns": ["sales.amount", "products.price", "sales.date"],
                "join_requirements": "sales.product_id = products.id"
            }
        },
        {
            "hypothesis_id": "hyp_2", 
            "hypothesis": "Customer satisfaction has a lagged effect on Sales",
            "target_variable": {
                "name": "sales",
                "transformation": "none",
                "aggregate_by": "customer"
            },
            "predictor_variable": {
                "name": "customer_satisfaction",
                "transformation": "lag_1",
                "aggregate_by": "customer"
            },
            "relationship_type": "lagged_effect",
            "expected_impact": "positive",
            "confidence": 0.75,
            "research_support": [
                "Customer satisfaction leads to repeat purchases",
                "Satisfaction surveys predict future buying behavior"
            ],
            "interaction_features": [
                "satisfaction * customer_lifetime_value",
                "satisfaction * product_category"
            ],
            "test_priority": 0.8,
            "data_requirements": {
                "required_tables": ["sales", "customer_satisfaction"],
                "required_columns": ["sales.amount", "customer_satisfaction.score", "sales.date"],
                "join_requirements": "sales.customer_id = customer_satisfaction.customer_id"
            }
        }
    ],
    "summary": {
        "total_hypotheses": 2,
        "high_confidence_hypotheses": 2,
        "transformation_types": ["log", "lag_1"],
        "aggregation_types": ["product", "customer"],
        "interaction_features_count": 5,
        "data_coverage": {
            "tables_required": ["sales", "products", "pricing", "customer_satisfaction"],
            "columns_required": 8,
            "join_complexity": "medium"
        }
    },
    "recommendations": [
        "Start with log-log price-sales relationship (highest priority)",
        "Test lagged satisfaction effects with temporal aggregation",
        "Consider interaction features for more complex models",
        "Validate data availability for all required tables"
    ]
}
```

### Agent Behavior

#### 1. Initialization
- Load Intugle knowledge base
- Initialize web search and RAG systems
- Set up hypothesis generation parameters
- Configure transformation mappings

#### 2. Variable Discovery
- Use semantic search to find related variables
- Access predicted relationships from knowledge base
- Research domain-specific insights via web search
- Validate variable existence in data schema

#### 3. Hypothesis Generation
- Generate hypotheses for each discovered variable
- Apply different relationship types and transformations
- Calculate confidence scores based on semantic similarity
- Prioritize hypotheses based on research support

#### 4. Output Compilation
- Structure hypotheses with required transformations
- Generate interaction features
- Specify data requirements for each hypothesis
- Provide testing recommendations

### Error Handling
- Handle missing variables in semantic search results
- Manage web search API failures
- Handle invalid transformation combinations
- Provide fallback hypotheses when research fails

### Performance Considerations
- Cache semantic search results
- Limit web search queries to avoid rate limits
- Use efficient hypothesis generation algorithms
- Implement hypothesis deduplication

### Integration Points
- **Input**: Receives univariate analysis results and target variable
- **Output**: Provides structured hypotheses to EDA Worker Loop Agent
- **Tools**: Integrates with Intugle semantic layer and external research tools
- **State**: Updates workflow state with generated hypotheses

### Example Usage
```python
# Initialize agent
hypothesis_agent = HypothesisGenerationAgent(
    target_variable="sales_revenue",
    kb=knowledge_builder,
    hypothesis_limit=10,
    domain="retail"
)

# Generate hypotheses
hypotheses = hypothesis_agent.generate_hypotheses(
    univariate_results=univariate_analysis_results,
    research_context=domain_context
)

# Access results
high_priority_hypotheses = [h for h in hypotheses if h["test_priority"] > 0.8]
transformation_types = set([h["predictor_variable"]["transformation"] for h in hypotheses])
```

### Success Criteria
- Generate specified number of high-quality hypotheses
- Provide specific transformation and aggregation recommendations
- Include research-backed confidence scores
- Specify clear data requirements for each hypothesis
- Create actionable hypotheses for downstream testing
