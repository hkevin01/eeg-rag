"""
RAG Module - Retrieval-Augmented Generation components
"""

from .corpus_builder import (
    EEGCorpusBuilder,
    Paper
)

from .embeddings import (
    PubMedBERTEmbedder,
    EmbeddingResult,
    BatchEmbeddingResult
)

from .agentic_rag import (
    AgenticRAGOrchestrator,
    AgenticRAGResult,
    AgenticStep,
    RetrievalDecision,
    RetrievalDecisionMaker,
    RetrievalNeed,
    ReformulationResult,
    ReformulationStrategy,
    QueryReformulator,
    SufficiencyCheck,
    SufficiencyEvaluator,
    SufficiencyStatus,
)

__all__ = [
    # Corpus / embeddings
    'EEGCorpusBuilder',
    'Paper',
    'PubMedBERTEmbedder',
    'EmbeddingResult',
    'BatchEmbeddingResult',
    # Agentic RAG
    'AgenticRAGOrchestrator',
    'AgenticRAGResult',
    'AgenticStep',
    'RetrievalDecision',
    'RetrievalDecisionMaker',
    'RetrievalNeed',
    'ReformulationResult',
    'ReformulationStrategy',
    'QueryReformulator',
    'SufficiencyCheck',
    'SufficiencyEvaluator',
    'SufficiencyStatus',
]
