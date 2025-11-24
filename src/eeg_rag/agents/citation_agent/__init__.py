"""
Citation Agent Module - Agent 4
Citation validation and impact scoring
"""

from .citation_validator import (
    CitationValidator,
    CitationValidationResult,
    ImpactScore,
    ValidationStatus
)

__all__ = [
    'CitationValidator',
    'CitationValidationResult',
    'ImpactScore',
    'ValidationStatus'
]
