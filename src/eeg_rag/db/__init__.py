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
from eeg_rag.db.paper_resolver import (
    PaperResolver,
    ResolvedPaper,
    get_paper_resolver
)
from eeg_rag.db.metadata_index import (
    MetadataIndex,
    PaperReference
)

__all__ = [
    # History
    'SearchHistoryDB',
    'SearchResult',
    'SearchQuery',
    'SearchSession',
    # Paper storage
    'PaperStore',
    'Paper',
    'get_paper_store',
    # Paper resolution (multi-source)
    'PaperResolver',
    'ResolvedPaper', 
    'get_paper_resolver',
    # Metadata index
    'MetadataIndex',
    'PaperReference'
]
