"""
Memory Management Module for EEG-RAG

This module provides both short-term and long-term memory capabilities
for the EEG-RAG system.
"""

from .memory_manager import (
    MemoryType,
    MemoryEntry,
    ShortTermMemory,
    LongTermMemory,
    MemoryManager
)

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "ShortTermMemory", 
    "LongTermMemory",
    "MemoryManager"
]