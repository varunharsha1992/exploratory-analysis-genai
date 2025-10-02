"""
Hypothesis Generation Agent for EDA Workflow

This module contains the HypothesisGenerationAgent class that generates intelligent
hypotheses about variables that might influence the target variable. It combines
domain knowledge from the semantic layer with external research to generate
data-driven hypotheses with specific transformation recommendations.

The agent uses RAG + Web Search capabilities to create actionable hypotheses
for downstream testing in the EDA workflow.
"""

from .hypothesis_generation import HypothesisGenerationAgent, hypothesis_generation_agent

__all__ = ['HypothesisGenerationAgent', 'hypothesis_generation_agent']
