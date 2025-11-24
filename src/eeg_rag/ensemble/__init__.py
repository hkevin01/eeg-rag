"""
Ensemble Module - Integration and aggregation components
"""

from .context_aggregator import (
    ContextAggregator,
    Citation,
    Entity,
    AggregatedContext
)

from .generation_ensemble import (
    GenerationEnsemble,
    GenerationResult,
    EnsembleResponse,
    LLMProvider,
    MockLLMClient
)

from .final_aggregator import (
    FinalAggregator,
    FinalAnswer,
    HallucinationDetector,
    ResponseValidator,
    CitationFormatter
)

__all__ = [
    'ContextAggregator',
    'Citation',
    'Entity',
    'AggregatedContext',
    'GenerationEnsemble',
    'GenerationResult',
    'EnsembleResponse',
    'LLMProvider',
    'MockLLMClient',
    'FinalAggregator',
    'FinalAnswer',
    'HallucinationDetector',
    'ResponseValidator',
    'CitationFormatter'
]
