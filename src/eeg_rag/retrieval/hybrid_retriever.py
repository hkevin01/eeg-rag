"""
# =============================================================================
# ID:             MOD-RETRIEVAL-001
# Requirement:    REQ-FUNC-010 — Hybrid search combining BM25 and dense retrieval;
#                 REQ-FUNC-011 — Reciprocal Rank Fusion (RRF) for result fusion;
#                 REQ-PERF-001 — Sub-100ms retrieval latency for 10K documents.
# Purpose:        Combine sparse (BM25) and dense (semantic embedding) retrieval
#                 using RRF to achieve higher recall and precision than either
#                 method alone for EEG domain queries.
# Rationale:      EEG literature contains both highly technical jargon (favoring
#                 exact-match BM25) and conceptually related terms (favoring dense
#                 semantic search). RRF robustly fuses rankings without requiring
#                 score normalization, tolerating distribution differences between
#                 sparse and dense similarity scores.
#                 Formula: score(d) = Σ 1/(k + rank(d)), k=60 (constant).
# Inputs:         query (str) — natural language EEG search query;
#                 top_k (int) — number of results to return (default 10);
#                 bm25_weight, dense_weight (float) — fusion weights (0–1);
#                 rrf_k (int) — RRF smoothing constant (default 60).
# Outputs:        List[HybridResult] sorted descending by rrf_score.
#                 Each result carries bm25_score, dense_score, ranks, and metadata.
# Preconditions:  BM25Retriever and DenseRetriever both initialized and indexed.
# Postconditions: Query expansion terms logged; reranking applied if configured.
# Assumptions:    Documents indexed in both BM25 and dense index with same doc_ids.
# Side Effects:   Query expansion may issue NLP inference calls (CPU-bound);
#                 cross-encoder reranking adds ~50ms per query if enabled.
# Failure Modes:  Either retriever returns empty → RRF uses only the available
#                 results; both empty → returns [].
# Error Handling: Individual retriever exceptions caught and logged; falls back
#                 to single-source results.
# Constraints:    Latency target: <100ms for 10K docs (REQ-PERF-001);
#                 adapt rrf_k=60 is standard; reducing it increases top-rank weight.
# Verification:   tests/test_hybrid_retriever.py; tests/test_retrieval_hybrid.py.
# References:     Cormack et al. 2009 "Reciprocal Rank Fusion outperforms Condorcet";
#                 REQ-FUNC-010, REQ-FUNC-011, REQ-PERF-001.
# =============================================================================
Hybrid Retriever for EEG-RAG.

This module combines BM25 (sparse) and dense (semantic) retrieval using
Reciprocal Rank Fusion (RRF) for optimal search results.

Requirements Implemented:
    - REQ-FUNC-010: Hybrid search combining BM25 and dense retrieval
    - REQ-FUNC-011: RRF fusion for result combination
    - REQ-PERF-001: Sub-100ms retrieval latency for 10K documents
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from eeg_rag.retrieval.bm25_retriever import BM25Retriever, BM25Result
from eeg_rag.retrieval.dense_retriever import DenseRetriever, DenseResult
from eeg_rag.retrieval.query_expander import EEGQueryExpander
from eeg_rag.retrieval.reranker import CrossEncoderReranker, NoOpReranker, RerankedResult
from eeg_rag.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : retrieval.hybrid_retriever.HybridResult
# Requirement  : `HybridResult` class shall be instantiable and expose the documented interface
# Purpose      : Result from hybrid search with both sparse and dense scores
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
# Verification : Instantiate HybridResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class HybridResult:
    """Result from hybrid search with both sparse and dense scores."""
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    rrf_score: float  # Final fused score
    bm25_score: float  # Raw BM25 score
    dense_score: float  # Raw dense score
    bm25_rank: Optional[int] = None  # Rank in BM25 results (1-indexed)
    dense_rank: Optional[int] = None  # Rank in dense results (1-indexed)


# ---------------------------------------------------------------------------
# ID           : retrieval.hybrid_retriever.HybridRetriever
# Requirement  : `HybridRetriever` class shall be instantiable and expose the documented interface
# Purpose      : Hybrid retrieval combining BM25 and dense search with RRF fusion
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
# Verification : Instantiate HybridRetriever with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and dense search with RRF fusion.

    Reciprocal Rank Fusion (RRF) is a simple but effective method to combine
    rankings from different retrieval systems. It's more robust than score
    normalization because it only uses rank information.

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where k is a constant (typically 60) and rank is 1-indexed.

    Example:
        >>> # Initialize both retrievers
        >>> bm25 = BM25Retriever()
        >>> dense = DenseRetriever()
        >>>
        >>> # Create hybrid retriever
        >>> hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        >>>
        >>> # Search with both methods
        >>> results = hybrid.search("epilepsy seizure detection", top_k=10)
        >>> for r in results:
        ...     print(f"{r.doc_id}: RRF={r.rrf_score:.3f}")
    """

    # ---------------------------------------------------------------------------
    # ID           : retrieval.hybrid_retriever.HybridRetriever.__init__
    # Requirement  : `__init__` shall initialize hybrid retriever
    # Purpose      : Initialize hybrid retriever
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : bm25_retriever: BM25Retriever; dense_retriever: DenseRetriever; bm25_weight: float (default=0.5); dense_weight: float (default=0.5); rrf_k: int (default=60); use_query_expansion: bool (default=True); use_reranking: bool (default=False); reranker_model: str (default='cross-encoder/ms-marco-MiniLM-L-6-v2'); adaptive_reranking: bool (default=False)
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
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        use_query_expansion: bool = True,
        use_reranking: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        adaptive_reranking: bool = False
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 sparse retriever
            dense_retriever: Dense semantic retriever
            bm25_weight: Weight for BM25 results (0-1)
            dense_weight: Weight for dense results (0-1)
            rrf_k: RRF constant (typically 60)
            use_query_expansion: Enable EEG domain query expansion
            use_reranking: Enable cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
            adaptive_reranking: Automatically decide whether to rerank based on query
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.use_query_expansion = use_query_expansion
        self.use_reranking = use_reranking
        self.adaptive_reranking = adaptive_reranking

        # Initialize query expander if enabled
        self.query_expander = EEGQueryExpander() if use_query_expansion else None

        # Initialize query analyzer for adaptive reranking
        if adaptive_reranking:
            self.query_analyzer = QueryAnalyzer()
            logger.info("  Adaptive reranking: enabled (query analyzer active)")
        else:
            self.query_analyzer = None

        # Initialize reranker if enabled
        if use_reranking or adaptive_reranking:
            try:
                self.reranker = CrossEncoderReranker(model_name=reranker_model)
                logger.info(f"  Reranking: {'adaptive' if adaptive_reranking else 'always'} ({reranker_model})")
            except ImportError:
                logger.warning("sentence-transformers not available, disabling reranking")
                self.reranker = NoOpReranker()
                self.use_reranking = False
                self.adaptive_reranking = False
        else:
            self.reranker = None

        logger.info(f"Initialized HybridRetriever")
        logger.info(f"  BM25 weight: {bm25_weight}")
        logger.info(f"  Dense weight: {dense_weight}")
        logger.info(f"  RRF k: {rrf_k}")
        logger.info(f"  Query expansion: {'enabled' if use_query_expansion else 'disabled'}")

    # ---------------------------------------------------------------------------
    # ID           : retrieval.hybrid_retriever.HybridRetriever._compute_rrf_scores
    # Requirement  : `_compute_rrf_scores` shall compute Reciprocal Rank Fusion scores
    # Purpose      : Compute Reciprocal Rank Fusion scores
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : bm25_results: List[BM25Result]; dense_results: List[DenseResult]
    # Outputs      : Dict[str, Tuple[float, Dict[str, Any]]]
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
    def _compute_rrf_scores(
        self,
        bm25_results: List[BM25Result],
        dense_results: List[DenseResult]
    ) -> Dict[str, Tuple[float, Dict[str, Any]]]:
        """
        Compute Reciprocal Rank Fusion scores.

        Args:
            bm25_results: Results from BM25 search
            dense_results: Results from dense search

        Returns:
            Dict mapping doc_id to (rrf_score, metadata)
        """
        rrf_scores = defaultdict(float)
        doc_metadata = {}  # Store text and metadata for each doc

        # BM25 rankings
        for rank, result in enumerate(bm25_results, 1):
            rrf_scores[result.doc_id] += (
                self.bm25_weight / (self.rrf_k + rank)
            )

            # Store metadata if not already present
            if result.doc_id not in doc_metadata:
                doc_metadata[result.doc_id] = {
                    "text": result.text,
                    "metadata": result.metadata,
                    "bm25_score": result.score,
                    "bm25_rank": rank,
                    "dense_score": 0.0,
                    "dense_rank": None
                }

        # Dense rankings
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.doc_id] += (
                self.dense_weight / (self.rrf_k + rank)
            )

            # Store or update metadata
            if result.doc_id not in doc_metadata:
                doc_metadata[result.doc_id] = {
                    "text": result.text,
                    "metadata": result.metadata,
                    "bm25_score": 0.0,
                    "bm25_rank": None,
                    "dense_score": result.score,
                    "dense_rank": rank
                }
            else:
                doc_metadata[result.doc_id]["dense_score"] = result.score
                doc_metadata[result.doc_id]["dense_rank"] = rank

        # Combine scores with metadata
        combined = {
            doc_id: (score, doc_metadata[doc_id])
            for doc_id, score in rrf_scores.items()
        }

        return combined

    # ---------------------------------------------------------------------------
    # ID           : retrieval.hybrid_retriever.HybridRetriever.search
    # Requirement  : `search` shall search using hybrid BM25 + dense retrieval with RRF
    # Purpose      : Search using hybrid BM25 + dense retrieval with RRF
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; top_k: int (default=10); retrieve_k: int (default=100); filters: Optional[Dict[str, Any]] (default=None)
    # Outputs      : List[HybridResult]
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
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieve_k: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[HybridResult]:
        """
        Search using hybrid BM25 + dense retrieval with RRF.

        Args:
            query: Search query
            top_k: Number of final results to return
            retrieve_k: Number of candidates to retrieve from each method
                       (should be >> top_k for effective fusion)
            filters: Optional filters for dense retrieval

        Returns:
            List of HybridResult objects sorted by RRF score (descending)

        Example:
            >>> results = hybrid.search(
            ...     "neural networks for seizure detection",
            ...     top_k=10,
            ...     retrieve_k=100
            ... )
        """
        logger.info(f"Hybrid search for: '{query}'")

        # Apply query expansion if enabled
        search_query = query
        if self.query_expander:
            search_query = self.query_expander.expand(query, max_expansions=2)
            if search_query != query.lower():
                logger.info(f"  Expanded: '{search_query}'")

        logger.info(f"  Retrieving {retrieve_k} candidates from each method")

        # Get results from both retrievers
        bm25_results = self.bm25.search(search_query, top_k=retrieve_k)
        dense_results = self.dense.search(search_query, top_k=retrieve_k, filters=filters)

        logger.info(f"  BM25: {len(bm25_results)} results")
        logger.info(f"  Dense: {len(dense_results)} results")

        # Compute RRF scores
        rrf_scores = self._compute_rrf_scores(bm25_results, dense_results)

        # Create HybridResult objects
        results = []
        for doc_id, (rrf_score, meta) in rrf_scores.items():
            results.append(HybridResult(
                doc_id=doc_id,
                text=meta["text"],
                metadata=meta["metadata"],
                rrf_score=rrf_score,
                bm25_score=meta["bm25_score"],
                dense_score=meta["dense_score"],
                bm25_rank=meta["bm25_rank"],
                dense_rank=meta["dense_rank"]
            ))

        # Sort by RRF score and take top_k
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        results = results[:top_k]

        # Determine if reranking should be applied
        should_rerank_query = self.use_reranking
        query_analysis = None

        if self.adaptive_reranking and self.query_analyzer:
            # Analyze query to decide if reranking is needed
            query_analysis = self.query_analyzer.analyze(query)
            should_rerank_query = query_analysis.should_rerank
            logger.info(f"  Query analysis: {query_analysis.complexity.value}, "
                       f"rerank={should_rerank_query} (conf: {query_analysis.confidence:.2f})")
            logger.info(f"  Reasoning: {query_analysis.reasoning}")

        # Apply reranking if enabled
        if should_rerank_query and self.reranker:
            logger.info(f"  Reranking {len(results)} results...")

            # Store original results for reference
            original_results = {r.doc_id: r for r in results}

            # Convert to reranker format
            candidates = []
            for r in results:
                candidates.append({
                    'doc_id': r.doc_id,
                    'text': r.text,
                    'score': r.rrf_score,
                    'metadata': r.metadata
                })

            # Rerank
            reranked = self.reranker.rerank(query, candidates)

            # Convert back to HybridResult
            results = []
            for rr in reranked:
                # Get original result to preserve BM25/Dense scores
                original = original_results.get(rr.doc_id)
                if original:
                    results.append(HybridResult(
                        doc_id=rr.doc_id,
                        text=rr.text,
                        metadata=rr.metadata,
                        rrf_score=rr.final_score,  # Use reranked score as RRF
                        bm25_score=original.bm25_score,
                        dense_score=original.dense_score,
                        bm25_rank=original.bm25_rank,
                        dense_rank=original.dense_rank
                    ))

            logger.info(f"  Reranking complete")
        elif self.adaptive_reranking:
            logger.info(f"  Skipping reranking for simple query")

        logger.info(f"✅ Hybrid search returned {len(results)} results")
        if results:
            logger.info(f"  Top result: {results[0].doc_id} (RRF: {results[0].rrf_score:.3f})")
            logger.info(f"    BM25 rank: {results[0].bm25_rank}, Dense rank: {results[0].dense_rank}")

        return results


if __name__ == "__main__":
    # Simple test
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

    logging.basicConfig(level=logging.INFO)

    # Test documents
    docs = [
        {
            "id": "1",
            "text": "Deep convolutional neural networks for epilepsy seizure detection from EEG signals",
            "metadata": {"year": 2019}
        },
        {
            "id": "2",
            "text": "Sleep staging using recurrent neural networks and EEG time series",
            "metadata": {"year": 2020}
        },
        {
            "id": "3",
            "text": "Motor imagery classification with convolutional neural networks for BCI",
            "metadata": {"year": 2021}
        },
        {
            "id": "4",
            "text": "Epileptic seizure prediction using machine learning and EEG features",
            "metadata": {"year": 2018}
        }
    ]

    # Initialize BM25
    bm25 = BM25Retriever(cache_dir="data/bm25_cache_test")
    bm25.index_documents(docs)

    # Initialize dense (requires Qdrant running - skip if not available)
    try:
        dense = DenseRetriever(url="http://localhost:6333", collection_name="eeg_papers")

        # Create hybrid retriever
        hybrid = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            bm25_weight=0.5,
            dense_weight=0.5
        )

        # Test search
        results = hybrid.search("epilepsy seizure detection", top_k=3, retrieve_k=10)

        print("\n🔍 Hybrid Search Results:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Doc {r.doc_id}:")
            print(f"   RRF Score: {r.rrf_score:.4f}")
            print(f"   BM25: score={r.bm25_score:.3f}, rank={r.bm25_rank}")
            print(f"   Dense: score={r.dense_score:.3f}, rank={r.dense_rank}")
            print(f"   Text: {r.text[:80]}...")

    except Exception as e:
        logger.error(f"Could not run hybrid test: {e}")
        logger.info("Make sure Qdrant is running with eeg_papers collection")
