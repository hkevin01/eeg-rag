"""
EEG-RAG Agent System

Multi-agent architecture for EEG research retrieval and synthesis.

Available Agents:
- BaseAgent: Abstract base class with resilience patterns
- PubMedAgent: Enhanced PubMed search with MeSH expansion and citation crawling
- SemanticScholarAgent: S2 integration with influence scoring
- SynthesisAgent: Evidence synthesis with gap detection

Supporting Classes:
- MeSHExpander: Medical Subject Heading expansion for PubMed
- CitationCrawler: PubMed citation network traversal
- InfluenceScorer: Research impact calculation
- EvidenceRanker: Evidence quality grading
- GapDetector: Research gap identification
"""

# Base agent infrastructure
from .base_agent import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    AgentQuery,
    AgentRegistry,
    CircuitBreaker,
    LoadBalancer,
    RetryManager,
)

# PubMed agent module
from .pubmed_agent import (
    PubMedAgent,
    PubMedPaper,
    MeSHExpander,
    CitationCrawler,
    PubMedQueryBuilder,
)

# Semantic Scholar agent module
from .semantic_scholar_agent import (
    SemanticScholarAgent,
    S2Paper,
    InfluenceScorer,
)

# Synthesis agent module
from .synthesis_agent import (
    SynthesisAgent,
    SynthesisResult,
    EvidenceRanker,
    EvidenceLevel,
    GapDetector,
    ResearchGap,
)

__all__ = [
    # Base infrastructure
    "BaseAgent",
    "AgentType",
    "AgentStatus",
    "AgentResult",
    "AgentQuery",
    "AgentRegistry",
    "CircuitBreaker",
    "LoadBalancer",
    "RetryManager",
    # PubMed agent
    "PubMedAgent",
    "PubMedPaper",
    "MeSHExpander",
    "CitationCrawler",
    "PubMedQueryBuilder",
    # Semantic Scholar agent
    "SemanticScholarAgent",
    "S2Paper",
    "InfluenceScorer",
    # Synthesis agent
    "SynthesisAgent",
    "SynthesisResult",
    "EvidenceRanker",
    "EvidenceLevel",
    "GapDetector",
    "ResearchGap",
]
