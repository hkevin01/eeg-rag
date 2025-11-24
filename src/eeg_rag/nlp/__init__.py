"""
NLP Module - Text processing and chunking
"""

from .chunking import (
    TextChunker,
    TextChunk,
    ChunkingResult
)

__all__ = [
    'TextChunker',
    'TextChunk',
    'ChunkingResult'
]
