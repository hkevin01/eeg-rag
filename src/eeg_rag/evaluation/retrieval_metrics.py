"""
Retrieval Evaluation Metrics

Implements standard information retrieval metrics:
- Recall@K: What fraction of relevant docs are in top-K?
- Precision@K: What fraction of top-K are relevant?
- MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant doc
- NDCG@K (Normalized Discounted Cumulative Gain): Position-aware metric
- MAP (Mean Average Precision): Average precision across all recall levels

These metrics require ground truth relevance labels.
"""

import logging
from typing import List, Dict, Set, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : evaluation.retrieval_metrics.RetrievalMetrics
# Requirement  : `RetrievalMetrics` class shall be instantiable and expose the documented interface
# Purpose      : Container for retrieval metrics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RetrievalMetrics with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics"""
    recall_at_k: Dict[int, float]  # K -> recall value
    precision_at_k: Dict[int, float]  # K -> precision value
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # K -> NDCG value
    map_score: float  # Mean Average Precision
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalMetrics.__str__
    # Requirement  : `__str__` shall execute as specified
    # Purpose      :   str  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __str__(self) -> str:
        lines = ["Retrieval Metrics:"]
        lines.append(f"  MRR: {self.mrr:.4f}")
        lines.append(f"  MAP: {self.map_score:.4f}")
        
        for k in sorted(self.recall_at_k.keys()):
            lines.append(
                f"  Recall@{k}: {self.recall_at_k[k]:.4f}  "
                f"Precision@{k}: {self.precision_at_k[k]:.4f}  "
                f"NDCG@{k}: {self.ndcg_at_k[k]:.4f}"
            )
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ID           : evaluation.retrieval_metrics.RetrievalEvaluator
# Requirement  : `RetrievalEvaluator` class shall be instantiable and expose the documented interface
# Purpose      : Evaluate retrieval quality using standard IR metrics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RetrievalEvaluator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RetrievalEvaluator:
    """
    Evaluate retrieval quality using standard IR metrics
    
    Example:
        >>> evaluator = RetrievalEvaluator()
        >>> 
        >>> # Ground truth: query -> set of relevant doc IDs
        >>> ground_truth = {
        ...     "query1": {"doc1", "doc3", "doc5"},
        ...     "query2": {"doc2", "doc4"}
        ... }
        >>> 
        >>> # Retrieved results: query -> list of doc IDs (ranked)
        >>> retrieved = {
        ...     "query1": ["doc1", "doc2", "doc3", "doc7", "doc5"],
        ...     "query2": ["doc1", "doc2", "doc3", "doc4"]
        ... }
        >>> 
        >>> metrics = evaluator.evaluate(ground_truth, retrieved, k_values=[1, 3, 5])
        >>> print(metrics)
    """
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : logger: Optional[logging.Logger] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("eeg_rag.evaluation")
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.recall_at_k
    # Requirement  : `recall_at_k` shall calculate Recall@K
    # Purpose      : Calculate Recall@K
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : relevant: Set[str]; retrieved: List[str]; k: int
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def recall_at_k(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K
        
        Recall@K = |relevant ∩ top-K retrieved| / |relevant|
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            Recall value (0-1)
        """
        if not relevant:
            return 0.0
        
        top_k = set(retrieved[:k])
        hits = len(relevant.intersection(top_k))
        
        return hits / len(relevant)
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.precision_at_k
    # Requirement  : `precision_at_k` shall calculate Precision@K
    # Purpose      : Calculate Precision@K
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : relevant: Set[str]; retrieved: List[str]; k: int
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def precision_at_k(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K
        
        Precision@K = |relevant ∩ top-K retrieved| / K
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            Precision value (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        
        return hits / k
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.mean_reciprocal_rank
    # Requirement  : `mean_reciprocal_rank` shall calculate MRR (Mean Reciprocal Rank)
    # Purpose      : Calculate MRR (Mean Reciprocal Rank)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : relevant: Set[str]; retrieved: List[str]
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def mean_reciprocal_rank(
        self,
        relevant: Set[str],
        retrieved: List[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank)
        
        MRR = 1 / rank_of_first_relevant_doc
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: List of retrieved document IDs (ranked)
            
        Returns:
            MRR value (0-1)
        """
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.ndcg_at_k
    # Requirement  : `ndcg_at_k` shall calculate NDCG@K (Normalized Discounted Cumulative Gain)
    # Purpose      : Calculate NDCG@K (Normalized Discounted Cumulative Gain)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : relevant: Set[str]; retrieved: List[str]; k: int; relevance_scores: Optional[Dict[str, float]] (default=None)
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def ndcg_at_k(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        DCG = Σ (2^rel - 1) / log2(rank + 1)
        NDCG = DCG / IDCG (ideal DCG)
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: List of retrieved document IDs (ranked)
            k: Cut-off position
            relevance_scores: Optional dict mapping doc_id to graded relevance (0-3)
                            If not provided, binary relevance is assumed (0 or 1)
            
        Returns:
            NDCG value (0-1)
        """
        # ---------------------------------------------------------------------------
        # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.dcg
        # Requirement  : `dcg` shall compute DCG
        # Purpose      : Compute DCG
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : ranks: List[Tuple[int, float]]
        # Outputs      : float
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Synchronous — must not block event loop
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
        def dcg(ranks: List[Tuple[int, float]]) -> float:
            """Compute DCG"""
            score = 0.0
            for rank, rel in ranks:
                if rank <= k:
                    score += (2 ** rel - 1) / np.log2(rank + 1)
            return score
        
        # Compute DCG for retrieved docs
        retrieved_ranks = []
        for rank, doc_id in enumerate(retrieved[:k], 1):
            if relevance_scores and doc_id in relevance_scores:
                rel = relevance_scores[doc_id]
            elif doc_id in relevant:
                rel = 1.0
            else:
                rel = 0.0
            retrieved_ranks.append((rank, rel))
        
        dcg_score = dcg(retrieved_ranks)
        
        # Compute IDCG (ideal DCG)
        if relevance_scores:
            # Sort by relevance score
            ideal_scores = sorted(
                [relevance_scores.get(doc_id, 0.0) for doc_id in relevant],
                reverse=True
            )
        else:
            # Binary relevance
            ideal_scores = [1.0] * len(relevant)
        
        ideal_ranks = [(rank, rel) for rank, rel in enumerate(ideal_scores, 1)]
        idcg_score = dcg(ideal_ranks)
        
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.average_precision
    # Requirement  : `average_precision` shall calculate Average Precision (AP)
    # Purpose      : Calculate Average Precision (AP)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : relevant: Set[str]; retrieved: List[str]
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def average_precision(
        self,
        relevant: Set[str],
        retrieved: List[str]
    ) -> float:
        """
        Calculate Average Precision (AP)
        
        AP = (Σ P(k) × rel(k)) / |relevant|
        where P(k) is precision at position k, rel(k) is 1 if doc at k is relevant
        
        Args:
            relevant: Set of relevant document IDs
            retrieved: List of retrieved document IDs (ranked)
            
        Returns:
            AP value (0-1)
        """
        if not relevant:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_hits += 1
                precision = num_hits / rank
                score += precision
        
        return score / len(relevant)
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.evaluate
    # Requirement  : `evaluate` shall evaluate retrieval results against ground truth
    # Purpose      : Evaluate retrieval results against ground truth
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : ground_truth: Dict[str, Set[str]]; retrieved: Dict[str, List[str]]; k_values: List[int] (default=[1, 3, 5, 10, 20]); relevance_scores: Optional[Dict[str, Dict[str, float]]] (default=None)
    # Outputs      : RetrievalMetrics
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def evaluate(
        self,
        ground_truth: Dict[str, Set[str]],
        retrieved: Dict[str, List[str]],
        k_values: List[int] = [1, 3, 5, 10, 20],
        relevance_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval results against ground truth
        
        Args:
            ground_truth: Dict mapping query_id to set of relevant doc_ids
            retrieved: Dict mapping query_id to list of retrieved doc_ids (ranked)
            k_values: List of K values for Recall@K, Precision@K, NDCG@K
            relevance_scores: Optional nested dict: query_id -> {doc_id -> relevance}
                            For graded relevance NDCG calculation
            
        Returns:
            RetrievalMetrics object with aggregated metrics
        """
        self.logger.info(f"Evaluating {len(ground_truth)} queries...")
        
        # Initialize accumulators
        recall_at_k = {k: [] for k in k_values}
        precision_at_k = {k: [] for k in k_values}
        ndcg_at_k = {k: [] for k in k_values}
        mrr_scores = []
        ap_scores = []
        
        # Evaluate each query
        for query_id, relevant in ground_truth.items():
            if query_id not in retrieved:
                self.logger.warning(f"Query {query_id} not in retrieved results, skipping")
                continue
            
            retrieved_docs = retrieved[query_id]
            query_relevance = relevance_scores.get(query_id) if relevance_scores else None
            
            # Recall@K and Precision@K
            for k in k_values:
                recall_at_k[k].append(self.recall_at_k(relevant, retrieved_docs, k))
                precision_at_k[k].append(self.precision_at_k(relevant, retrieved_docs, k))
                ndcg_at_k[k].append(self.ndcg_at_k(relevant, retrieved_docs, k, query_relevance))
            
            # MRR
            mrr_scores.append(self.mean_reciprocal_rank(relevant, retrieved_docs))
            
            # AP
            ap_scores.append(self.average_precision(relevant, retrieved_docs))
        
        # Aggregate metrics
        metrics = RetrievalMetrics(
            recall_at_k={k: np.mean(scores) for k, scores in recall_at_k.items()},
            precision_at_k={k: np.mean(scores) for k, scores in precision_at_k.items()},
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            ndcg_at_k={k: np.mean(scores) for k, scores in ndcg_at_k.items()},
            map_score=np.mean(ap_scores) if ap_scores else 0.0
        )
        
        self.logger.info("Evaluation complete")
        return metrics
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.retrieval_metrics.RetrievalEvaluator.compare
    # Requirement  : `compare` shall compare two sets of metrics and compute improvements
    # Purpose      : Compare two sets of metrics and compute improvements
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : baseline_metrics: RetrievalMetrics; improved_metrics: RetrievalMetrics
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def compare(
        self,
        baseline_metrics: RetrievalMetrics,
        improved_metrics: RetrievalMetrics
    ) -> Dict[str, Any]:
        """
        Compare two sets of metrics and compute improvements
        
        Args:
            baseline_metrics: Baseline (e.g., without reranking)
            improved_metrics: Improved (e.g., with reranking)
            
        Returns:
            Dict with improvements and percentage changes
        """
        improvements = {
            'mrr': {
                'baseline': baseline_metrics.mrr,
                'improved': improved_metrics.mrr,
                'delta': improved_metrics.mrr - baseline_metrics.mrr,
                'pct_change': ((improved_metrics.mrr - baseline_metrics.mrr) / baseline_metrics.mrr * 100)
                    if baseline_metrics.mrr > 0 else 0.0
            },
            'map': {
                'baseline': baseline_metrics.map_score,
                'improved': improved_metrics.map_score,
                'delta': improved_metrics.map_score - baseline_metrics.map_score,
                'pct_change': ((improved_metrics.map_score - baseline_metrics.map_score) / baseline_metrics.map_score * 100)
                    if baseline_metrics.map_score > 0 else 0.0
            }
        }
        
        # Compare at each K
        for k in baseline_metrics.recall_at_k.keys():
            improvements[f'recall@{k}'] = {
                'baseline': baseline_metrics.recall_at_k[k],
                'improved': improved_metrics.recall_at_k[k],
                'delta': improved_metrics.recall_at_k[k] - baseline_metrics.recall_at_k[k],
                'pct_change': ((improved_metrics.recall_at_k[k] - baseline_metrics.recall_at_k[k]) / 
                              baseline_metrics.recall_at_k[k] * 100)
                    if baseline_metrics.recall_at_k[k] > 0 else 0.0
            }
            
            improvements[f'ndcg@{k}'] = {
                'baseline': baseline_metrics.ndcg_at_k[k],
                'improved': improved_metrics.ndcg_at_k[k],
                'delta': improved_metrics.ndcg_at_k[k] - baseline_metrics.ndcg_at_k[k],
                'pct_change': ((improved_metrics.ndcg_at_k[k] - baseline_metrics.ndcg_at_k[k]) / 
                              baseline_metrics.ndcg_at_k[k] * 100)
                    if baseline_metrics.ndcg_at_k[k] > 0 else 0.0
            }
        
        return improvements


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    evaluator = RetrievalEvaluator()
    
    # Sample data
    ground_truth = {
        "q1": {"doc1", "doc3", "doc5"},
        "q2": {"doc2", "doc4"}
    }
    
    retrieved = {
        "q1": ["doc1", "doc2", "doc3", "doc7", "doc5"],
        "q2": ["doc1", "doc2", "doc3", "doc4"]
    }
    
    metrics = evaluator.evaluate(ground_truth, retrieved, k_values=[1, 3, 5])
    print(metrics)
