#!/usr/bin/env python3
"""
Cross-Encoder Reranking for EEG-RAG

Implements cross-encoder models for reranking retrieved documents
to improve relevance and reduce noise in RAG pipeline.
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging
import time

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

from ..utils.logging_utils import log_time
from ..utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Result from cross-encoder reranking."""
    document_id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    position_change: int
    metadata: Optional[Dict] = None


@dataclass
class RerankingMetrics:
    """Metrics from reranking operation."""
    total_documents: int
    reranked_documents: int
    processing_time_ms: float
    score_variance_reduction: float
    top_k_overlap: float
    average_position_change: float


class CrossEncoderReranker:
    """Cross-encoder based document reranker for EEG domain."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        enable_caching: bool = True
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of cross-encoder model to use.
            cache_dir: Directory to cache model.
            batch_size: Batch size for inference.
            max_length: Maximum sequence length.
            device: Device to run on (auto, cpu, cuda).
            enable_caching: Whether to enable score caching.
        """
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers not available for cross-encoder reranking")
            self.enabled = False
            return
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_caching = enable_caching
        self.enabled = True
        
        # Load model
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
                cache_folder=cache_dir
            )
            logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {str(e)}")
            self.enabled = False
            return
        
        # Score cache
        self._score_cache = {} if enable_caching else None
        
        # EEG-specific boost terms
        self.eeg_boost_terms = {
            'electrode', 'eeg', 'electroencephalography', 'seizure', 'epilepsy',
            'alpha', 'beta', 'theta', 'delta', 'gamma', 'p300', 'n400',
            'brain', 'neural', 'cognitive', 'clinical', 'biomarker'
        }
        
        # Metrics
        self.reranking_stats = {
            'total_queries': 0,
            'total_documents': 0,
            'average_score_improvement': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
        alpha: float = 0.7  # Weight for rerank score vs original score
    ) -> Tuple[List[RerankingResult], RerankingMetrics]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Search query.
            documents: List of documents with 'content' and 'score' keys.
            top_k: Number of top documents to return after reranking.
            score_threshold: Minimum score threshold for inclusion.
            alpha: Weight for combining original and rerank scores.
            
        Returns:
            Tuple of (reranked results, metrics).
        """
        if not self.enabled or not documents:
            return self._create_fallback_results(documents), self._create_empty_metrics()
        
        start_time = time.time()
        
        with log_time(logger, f"Cross-encoder reranking {len(documents)} documents"):
            # Prepare query-document pairs
            query_doc_pairs = self._prepare_pairs(query, documents)
            
            # Get rerank scores
            rerank_scores = await self._get_rerank_scores(query_doc_pairs)
            
            # Combine scores and create results
            results = self._combine_scores(query, documents, rerank_scores, alpha)
            
            # Filter and sort
            results = [r for r in results if r.final_score >= score_threshold]
            results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Limit to top_k
            if top_k:
                results = results[:top_k]
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        metrics = self._calculate_metrics(documents, results, processing_time)
        
        # Update stats
        self._update_stats(len(documents), rerank_scores)
        
        return results, metrics
    
    def _prepare_pairs(self, query: str, documents: List[Dict]) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for cross-encoder."""
        pairs = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Truncate content if too long
            if len(content) > 2000:  # Rough character limit
                content = content[:2000] + "..."
            
            pairs.append((query, content))
        
        return pairs
    
    async def _get_rerank_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Get reranking scores for query-document pairs."""
        if not pairs:
            return []
        
        # Check cache first
        if self.enable_caching:
            cached_scores = []
            uncached_pairs = []
            uncached_indices = []
            
            for i, pair in enumerate(pairs):
                cache_key = self._make_cache_key(pair)
                if cache_key in self._score_cache:
                    cached_scores.append((i, self._score_cache[cache_key]))
                    self.reranking_stats['cache_hits'] += 1
                else:
                    uncached_pairs.append(pair)
                    uncached_indices.append(i)
                    self.reranking_stats['cache_misses'] += 1
            
            # Process uncached pairs
            if uncached_pairs:
                uncached_scores = await self._compute_scores(uncached_pairs)
                
                # Cache new scores
                for pair, score in zip(uncached_pairs, uncached_scores):
                    cache_key = self._make_cache_key(pair)
                    self._score_cache[cache_key] = score
            else:
                uncached_scores = []
            
            # Combine cached and uncached scores
            all_scores = [0.0] * len(pairs)
            
            # Fill in cached scores
            for idx, score in cached_scores:
                all_scores[idx] = score
            
            # Fill in uncached scores
            for idx, score in zip(uncached_indices, uncached_scores):
                all_scores[idx] = score
            
            return all_scores
        
        else:
            # No caching - compute all scores
            return await self._compute_scores(pairs)
    
    async def _compute_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute cross-encoder scores for pairs."""
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                self._predict_batch,
                pairs
            )
            return scores.tolist()
            
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {str(e)}")
            return [0.5] * len(pairs)  # Fallback to neutral scores
    
    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Predict scores for batch of pairs."""
        all_scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs)
            all_scores.extend(batch_scores)
        
        return np.array(all_scores)
    
    def _combine_scores(
        self,
        query: str,
        documents: List[Dict],
        rerank_scores: List[float],
        alpha: float
    ) -> List[RerankingResult]:
        """Combine original and rerank scores."""
        results = []
        
        # Calculate original ranking
        original_ranking = {doc.get('id', i): i for i, doc in enumerate(documents)}
        
        for i, (doc, rerank_score) in enumerate(zip(documents, rerank_scores)):
            original_score = doc.get('score', 0.0)
            
            # Apply EEG domain boost
            domain_boost = self._calculate_domain_boost(query, doc.get('content', ''))
            adjusted_rerank_score = rerank_score + domain_boost
            
            # Combine scores
            final_score = alpha * adjusted_rerank_score + (1 - alpha) * original_score
            
            # Calculate position change after final ranking
            doc_id = doc.get('id', str(i))
            original_position = original_ranking.get(doc_id, i)
            
            result = RerankingResult(
                document_id=doc_id,
                content=doc.get('content', ''),
                original_score=original_score,
                rerank_score=adjusted_rerank_score,
                final_score=final_score,
                position_change=0,  # Will be calculated after sorting
                metadata=doc.get('metadata', {})
            )
            
            results.append(result)
        
        # Sort by final score and calculate position changes
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        for new_pos, result in enumerate(results):
            doc_id = result.document_id
            original_pos = original_ranking.get(doc_id, 0)
            result.position_change = original_pos - new_pos
        
        return results
    
    def _calculate_domain_boost(self, query: str, content: str) -> float:
        """Calculate EEG domain relevance boost."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        boost = 0.0
        
        # Boost for EEG-specific terms
        for term in self.eeg_boost_terms:
            if term in query_lower and term in content_lower:
                boost += 0.05  # Small boost per matching term
        
        # Extra boost for exact phrase matches
        query_phrases = [phrase.strip() for phrase in query_lower.split() if len(phrase) > 3]
        for phrase in query_phrases:
            if phrase in content_lower:
                boost += 0.02
        
        return min(boost, 0.2)  # Cap boost at 0.2
    
    def _make_cache_key(self, pair: Tuple[str, str]) -> str:
        """Create cache key for query-document pair."""
        query, content = pair
        # Use hash of combined content for efficient caching
        import hashlib
        combined = f"{query}||{content[:500]}"  # Use first 500 chars
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _calculate_metrics(
        self,
        original_docs: List[Dict],
        reranked_results: List[RerankingResult],
        processing_time: float
    ) -> RerankingMetrics:
        """Calculate reranking metrics."""
        if not original_docs or not reranked_results:
            return self._create_empty_metrics()
        
        # Calculate score variance reduction
        original_scores = [doc.get('score', 0.0) for doc in original_docs]
        reranked_scores = [r.final_score for r in reranked_results]
        
        orig_variance = np.var(original_scores) if len(original_scores) > 1 else 0
        rerank_variance = np.var(reranked_scores) if len(reranked_scores) > 1 else 0
        variance_reduction = max(0, orig_variance - rerank_variance)
        
        # Calculate top-k overlap
        top_k = min(5, len(original_docs), len(reranked_results))
        original_top_ids = {doc.get('id', i) for i, doc in enumerate(original_docs[:top_k])}
        reranked_top_ids = {r.document_id for r in reranked_results[:top_k]}
        top_k_overlap = len(original_top_ids & reranked_top_ids) / top_k if top_k > 0 else 0
        
        # Calculate average position change
        position_changes = [abs(r.position_change) for r in reranked_results]
        avg_position_change = np.mean(position_changes) if position_changes else 0
        
        return RerankingMetrics(
            total_documents=len(original_docs),
            reranked_documents=len(reranked_results),
            processing_time_ms=processing_time,
            score_variance_reduction=variance_reduction,
            top_k_overlap=top_k_overlap,
            average_position_change=avg_position_change
        )
    
    def _create_fallback_results(self, documents: List[Dict]) -> List[RerankingResult]:
        """Create fallback results when reranking is disabled."""
        results = []
        for i, doc in enumerate(documents):
            results.append(RerankingResult(
                document_id=doc.get('id', str(i)),
                content=doc.get('content', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                final_score=doc.get('score', 0.0),
                position_change=0,
                metadata=doc.get('metadata', {})
            ))
        return results
    
    def _create_empty_metrics(self) -> RerankingMetrics:
        """Create empty metrics object."""
        return RerankingMetrics(
            total_documents=0,
            reranked_documents=0,
            processing_time_ms=0.0,
            score_variance_reduction=0.0,
            top_k_overlap=0.0,
            average_position_change=0.0
        )
    
    def _update_stats(self, num_documents: int, rerank_scores: List[float]):
        """Update internal statistics."""
        self.reranking_stats['total_queries'] += 1
        self.reranking_stats['total_documents'] += num_documents
        
        if rerank_scores:
            avg_score = np.mean(rerank_scores)
            # Update running average
            current_avg = self.reranking_stats['average_score_improvement']
            query_count = self.reranking_stats['total_queries']
            self.reranking_stats['average_score_improvement'] = (
                (current_avg * (query_count - 1) + avg_score) / query_count
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranking statistics."""
        stats = self.reranking_stats.copy()
        stats['enabled'] = self.enabled
        stats['model_name'] = self.model_name if self.enabled else None
        
        if self.enable_caching and self._score_cache:
            stats['cache_size'] = len(self._score_cache)
            cache_total = stats['cache_hits'] + stats['cache_misses']
            stats['cache_hit_rate'] = stats['cache_hits'] / cache_total if cache_total > 0 else 0
        
        return stats
    
    def clear_cache(self):
        """Clear the score cache."""
        if self.enable_caching and self._score_cache:
            self._score_cache.clear()
            logger.info("Cross-encoder score cache cleared")


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(**kwargs) -> CrossEncoderReranker:
    """Get global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(**kwargs)
    return _reranker