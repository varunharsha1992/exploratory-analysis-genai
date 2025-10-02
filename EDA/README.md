# EDA (Exploratory Data Analysis) Workflow

This directory contains a comprehensive agentic workflow for performing automated EDA for predictive analytics using LangGraph and Intugle.

## Overview

The EDA workflow consists of 5 specialized agents that work together to perform comprehensive exploratory data analysis:

1. **Univariate Analysis Agent** - Performs basic data profiling and anomaly detection
2. **Hypothesis Generation Agent** - Generates hypotheses about influencing variables using RAG + Web Search
3. **EDA Worker Loop Agent** - Orchestrates hypothesis testing in parallel
4. **EDA Analysis Agent** - Tests individual hypotheses with correlation analysis and visualization
5. **Summarizer Agent** - Synthesizes findings and recommends feature engineering strategies

## Prerequisites

- Intugle semantic layer must be built using `KnowledgeBuilder`
- Target variable must be specified for predictive analytics
- Qdrant vector database running for semantic search
- OpenAI API key for embeddings and LLM operations

## Workflow Architecture

```
Input: Target Variable + EDA Request
    ↓
Univariate Analysis Agent
    ↓
Hypothesis Generation Agent
    ↓
EDA Worker Loop Agent (Async)
    ↓
EDA Analysis Agent (Multiple Instances)
    ↓
Summarizer Agent
    ↓
Output: Feature Engineering Recommendations
```

## File Structure

```
EDA/
├── README.md                           # This file
├── agents/
│   ├── univariate_analysis_agent.md    # Univariate analysis specifications
│   ├── hypothesis_generation_agent.md  # Hypothesis generation specifications
│   ├── eda_worker_loop_agent.md        # Worker loop orchestration
│   ├── eda_analysis_agent.md           # Individual hypothesis testing
│   └── summarizer_agent.md             # Results summarization
├── tools/
│   ├── intugle_tools.md                # Intugle integration tools
│   ├── anomaly_detection_tools.md      # Anomaly detection utilities
│   ├── correlation_tools.md            # Statistical correlation tools
│   └── visualization_tools.md          # Chart generation tools
├── workflow/
│   ├── langgraph_workflow.md           # Main workflow specification
│   ├── state_management.md             # State and data flow
│   └── error_handling.md               # Error handling strategies
└── examples/
    ├── sample_workflow.py              # Example implementation
    └── test_scenarios.md               # Test scenarios and expected outputs
```

## Quick Start

1. Ensure Intugle semantic layer is built
2. Configure environment variables (OpenAI API key, Qdrant URL)
3. Initialize the workflow with target variable
4. Execute the LangGraph workflow
5. Review generated feature engineering recommendations

## Key Features

- **Automated Hypothesis Generation**: Uses RAG + Web Search to generate data-driven hypotheses
- **Parallel Processing**: Tests multiple hypotheses concurrently for efficiency
- **Semantic Integration**: Leverages Intugle's semantic layer for intelligent data discovery
- **Comprehensive Analysis**: Covers univariate analysis, correlation testing, and visualization
- **Actionable Insights**: Provides specific feature engineering recommendations

## Integration with Intugle

The workflow heavily integrates with Intugle's capabilities:
- `kb.links` - Access predicted relationships between tables
- `kb.search()` - Semantic search for relevant variables and KPIs
- `DataProductBuilder` - Generate datasets for analysis using ETL schemas
- Semantic layer - Understand data structure and relationships

## Next Steps

1. Review individual agent specifications in the `agents/` directory
2. Understand tool integrations in the `tools/` directory
3. Study the workflow implementation in the `workflow/` directory
4. Run example scenarios from the `examples/` directory
