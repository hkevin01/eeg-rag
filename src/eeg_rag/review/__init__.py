"""
Systematic review automation module.

Provides tools for automated extraction, comparison, and reproducibility scoring
of deep learning EEG research papers.
"""

from eeg_rag.review.extractor import (
    SystematicReviewExtractor,
    ExtractionField,
    ExtractedData
)
from eeg_rag.review.comparator import (
    SystematicReviewComparator,
    ComparisonResults,
    ReproducibilityScorer
)

__all__ = [
    "SystematicReviewExtractor",
    "ExtractionField",
    "ExtractedData",
    "SystematicReviewComparator",
    "ComparisonResults",
    "ReproducibilityScorer"
]
