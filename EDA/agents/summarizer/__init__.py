"""
Summarizer Agent for EDA Workflow

This module contains the SummarizerAgent class that synthesizes findings from all EDA agents
and provides comprehensive feature engineering recommendations for predictive modeling.
"""

from .summarizer import SummarizerAgent, summarizer_agent

__all__ = ['SummarizerAgent', 'summarizer_agent']

