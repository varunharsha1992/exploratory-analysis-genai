"""
EDA Analysis Agent for Data Querying AI

This module contains the EDAAnalysisAgent class that performs comprehensive statistical analysis 
and hypothesis testing for individual hypotheses in the EDA workflow.

The agent is responsible for:
- Testing individual hypotheses with statistical analysis
- Fetching data using Intugle's DataProductBuilder and ETL schemas
- Performing correlation analysis and statistical tests
- Generating visualizations to illustrate relationships
- Providing detailed analysis results for each hypothesis
"""

from .eda_analysis import EDAAnalysisAgent, eda_analysis_agent

__all__ = ['EDAAnalysisAgent', 'eda_analysis_agent']
