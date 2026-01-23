"""Database module for EEG-RAG system."""

from eeg_rag.db.history_models import (
    SearchHistoryDB,
    SearchResult,
    SearchQuery,
    SearchSession
)
from eeg_rag.db.paper_store import (
    PaperStore,
    Paper,
    get_paper_store
)

__all__ = [
    'SearchHistoryDB',
    'SearchResult',
    'SearchQuery',
    'SearchSession',
    'PaperStore',
    'Paper',
    'get_paper_store'
]
