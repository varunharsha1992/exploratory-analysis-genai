"""
EDA Worker Loop Agent for EDA Workflow

This module contains the EDAWorkerLoopAgent class that orchestrates
parallel hypothesis testing using multiple EDA Analysis agents.
"""

from .eda_worker_loop import EDAWorkerLoopAgent, eda_worker_loop_agent

__all__ = ['EDAWorkerLoopAgent', 'eda_worker_loop_agent']
