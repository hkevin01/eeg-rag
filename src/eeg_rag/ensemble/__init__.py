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

__all__ = [
    'ContextAggregator',
    'Citation',
    'Entity',
    'AggregatedContext',
    'GenerationEnsemble',
    'GenerationResult',
    'EnsembleResponse',
    'LLMProvider',
    'MockLLMClient'
]
