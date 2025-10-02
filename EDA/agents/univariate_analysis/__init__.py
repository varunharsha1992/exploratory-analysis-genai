"""
Univariate Analysis Agent for EDA Workflow

This module contains the UnivariateAnalysisAgent class that performs comprehensive
univariate analysis on target variables for predictive analytics.

The agent integrates with LangGraph and provides:
- Data profiling and statistical summaries
- Anomaly detection and outlier identification
- Trend analysis and pattern recognition
- Data quality assessment and recommendations
- Related variable discovery through semantic search
"""

from .univariate_analysis import UnivariateAnalysisAgent, univariate_analysis_agent

__all__ = ['UnivariateAnalysisAgent', 'univariate_analysis_agent']
