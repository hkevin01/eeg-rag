"""
Hybrid Retriever for EEG-RAG.

This module combines BM25 (sparse) and dense (semantic) retrieval using
Reciprocal Rank Fusion (RRF) for optimal search results.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict

from eeg_rag.retrieval.bm25_retriever import BM25Retriever, BM25Result
from eeg_rag.retrieval.dense_retriever import DenseRetriever, DenseResult
from eeg_rag.retrieval.query_expander import EEGQueryExpander

logger = logging.getLogger(__name__)


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
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
        use_query_expansion: bool = True
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
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.use_query_expansion = use_query_expansion
        
        # Initialize query expander if enabled
        self.query_expander = EEGQueryExpander() if use_query_expansion else None
        
        logger.info(f"Initialized HybridRetriever")
        logger.info(f"  BM25 weight: {bm25_weight}")
        logger.info(f"  Dense weight: {dense_weight}")
        logger.info(f"  RRF k: {rrf_k}")
        logger.info(f"  Query expansion: {'enabled' if use_query_expansion else 'disabled'}")
        
        logger.info(f"Initialized HybridRetriever")
        logger.info(f"  BM25 weight: {bm25_weight}")
        logger.info(f"  Dense weight: {dense_weight}")
        logger.info(f"  RRF k: {rrf_k}")
    
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
