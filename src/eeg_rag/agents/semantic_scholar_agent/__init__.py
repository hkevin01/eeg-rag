"""
Enhanced Semantic Scholar Agent Module

Provides comprehensive Semantic Scholar integration with:
- Paper search with field filtering
- Citation graph traversal
- Author expertise tracking
- Influential citation identification
- Paper recommendations
"""

from .s2_agent import SemanticScholarAgent, S2Paper
from .influence_scorer import InfluenceScorer

__all__ = [
    "SemanticScholarAgent",
    "S2Paper",
    "InfluenceScorer",
]
