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

__all__ = [
    'EEGCorpusBuilder',
    'Paper',
    'PubMedBERTEmbedder',
    'EmbeddingResult',
    'BatchEmbeddingResult'
]
