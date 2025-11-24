"""
NLP Module - Text processing and chunking
"""

from .chunking import TextChunker, TextChunk, ChunkingResult
from .ner_eeg import EEGNER, Entity, EntityType, NERResult, EEGTerminologyDatabase

__all__ = [
    'TextChunker', 'TextChunk', 'ChunkingResult',
    'EEGNER', 'Entity', 'EntityType', 'NERResult', 'EEGTerminologyDatabase'
]
