"""
Evaluation Module for EEG-RAG.

Provides RAGAS-style metrics, retrieval evaluation, and benchmarking tools.
"""

from .ragas_metrics import (
    RAGASEvaluator,
    RAGASInput,
    RAGASScores,
    ContextDocument,
    EvaluationMode,
    LLMProvider,
    FaithfulnessDetail,
    ContextChunkScore,
    HumanEvalRecord,
    export_for_human_eval,
)

from .retrieval_metrics import (
    RetrievalMetrics,
    RetrievalEvaluator,
)

__all__ = [
    # RAGAS
    "RAGASEvaluator",
    "RAGASInput",
    "RAGASScores",
    "ContextDocument",
    "EvaluationMode",
    "LLMProvider",
    "FaithfulnessDetail",
    "ContextChunkScore",
    "HumanEvalRecord",
    "export_for_human_eval",
    # Retrieval metrics
    "RetrievalMetrics",
    "RetrievalEvaluator",
]
