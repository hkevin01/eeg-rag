"""
Utility modules for EEG-RAG.
"""

from .deduplication import PaperDeduplicator, deduplicate_papers
from .citations import CitationGenerator, generate_citations
from .quality_badges import (
    get_code_badge,
    get_data_badge,
    get_reproducibility_badge,
    get_citation_count_badge,
    get_all_badges,
    get_quality_score,
)

__all__ = [
    'PaperDeduplicator',
    'deduplicate_papers',
    'CitationGenerator',
    'generate_citations',
    'get_code_badge',
    'get_data_badge',
    'get_reproducibility_badge',
    'get_citation_count_badge',
    'get_all_badges',
    'get_quality_score',
]
