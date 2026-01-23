"""
Multi-Agent Orchestrator for EEG Literature Research.
"""

from .orchestrator import (
    Orchestrator,
    OrchestratorResult,
    QueryType,
    ExecutionStrategy,
    QueryPlan,
    quick_search
)

__all__ = [
    "Orchestrator",
    "OrchestratorResult", 
    "QueryType",
    "ExecutionStrategy",
    "QueryPlan",
    "quick_search"
]
