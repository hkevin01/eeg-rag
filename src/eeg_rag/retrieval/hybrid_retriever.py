#!/usr/bin/env python3
"""
Hybrid Retrieval System (BM25 + Dense Retrieval)

Combines sparse (BM25) and dense retrieval for improved recall,
especially important for domain-specific terminology in EEG research.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import json
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    doc_id: Union[str, int]
    score: float
    content: str
    metadata: Dict[str, Any] = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'doc_id': self.doc_id,
            'score': self.score,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'metadata': self.metadata or {},
            'bm25_score': self.bm25_score,
            'dense_score': self.dense_score
        }


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense retrieval"""
    
    def __init__(self, dense_retriever=None, alpha: float = 0.5, 
                 fusion_method: str = 'weighted_sum'):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense retrieval system (FAISS, etc.)
            alpha: Weight for dense vs sparse (0.5 = equal weight)
            fusion_method: 'weighted_sum', 'reciprocal_rank', or 'max'
        """
        self.dense_retriever = dense_retriever
        self.bm25: Optional[BM25Okapi] = None
        self.alpha = alpha
        self.fusion_method = fusion_method
        
        # Document storage
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self.doc_ids: List[Union[str, int]] = []
        
        # Preprocessing for BM25
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 
            'between', 'among', 'under', 'over'
        ])
    
    def add_documents(self, documents: List[str], 
                     metadata: List[Dict[str, Any]] = None,
                     doc_ids: List[Union[str, int]] = None):
        """Add documents to the retrieval system"""
        self.documents = documents
        self.doc_metadata = metadata or [{} for _ in documents]
        self.doc_ids = doc_ids or list(range(len(documents)))
        
        # Build BM25 index
        self._build_bm25_index()
        
        # Add to dense retriever if available
        if self.dense_retriever and hasattr(self.dense_retriever, 'add_documents'):
            try:
                self.dense_retriever.add_documents(documents)
            except Exception as e:
                logger.warning(f"Failed to add documents to dense retriever: {e}")
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        try:
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in self.documents:
                tokens = self._tokenize_for_bm25(doc)
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"Built BM25 index for {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25 = None
    
    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 with EEG-specific preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace common EEG abbreviations to improve matching
        eeg_replacements = {
            'electroencephalogram': 'eeg',
            'electroencephalography': 'eeg',
            'event-related potential': 'erp',
            'event related potential': 'erp',
            'brain-computer interface': 'bci',
            'brain computer interface': 'bci',
            'motor imagery': 'motor_imagery',
            'sleep spindle': 'sleep_spindle',
            'k-complex': 'k_complex',
            'slow wave': 'slow_wave'
        }
        
        for full_term, abbrev in eeg_replacements.items():
            text = text.replace(full_term, abbrev)
        
        # Simple tokenization (could be improved with spaCy)
        import re
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords but keep EEG-specific terms
        eeg_terms = {
            'eeg', 'erp', 'bci', 'alpha', 'beta', 'gamma', 'delta', 'theta',
            'mu', 'c3', 'c4', 'cz', 'f3', 'f4', 'fz', 'o1', 'o2', 'oz',
            'p3', 'p4', 'pz', 't3', 't4', 't5', 't6', 'fp1', 'fp2',
            'seizure', 'epilepsy', 'spike', 'sharp', 'wave', 'rhythm'
        }
        
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stopwords or token in eeg_terms
        ]
        
        return filtered_tokens
    
    def search(self, query: str, top_k: int = 10, 
               dense_top_k: int = None, bm25_top_k: int = None) -> List[RetrievalResult]:
        """Hybrid search combining BM25 and dense retrieval"""
        if not self.documents:
            logger.warning("No documents indexed")
            return []
        
        # Set default top_k for individual retrievers
        dense_top_k = dense_top_k or min(top_k * 2, len(self.documents))
        bm25_top_k = bm25_top_k or min(top_k * 2, len(self.documents))
        
        # Get BM25 scores
        bm25_scores = self._get_bm25_scores(query, bm25_top_k)
        
        # Get dense scores
        dense_scores = self._get_dense_scores(query, dense_top_k)
        
        # Combine scores
        combined_results = self._combine_scores(bm25_scores, dense_scores, top_k)
        
        return combined_results
    
    def _get_bm25_scores(self, query: str, top_k: int) -> Dict[int, float]:
        """Get BM25 scores for query"""
        if not self.bm25:
            logger.warning("BM25 index not available")
            return {}
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_for_bm25(query)
            
            # Get scores for all documents
            scores = self.bm25.get_scores(query_tokens)
            
            # Normalize scores to 0-1 range
            if len(scores) > 0:
                max_score = max(scores)
                min_score = min(scores)
                if max_score > min_score:
                    scores = (scores - min_score) / (max_score - min_score)
                else:
                    scores = np.ones_like(scores)
            
            # Get top results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            return {idx: scores[idx] for idx in top_indices if scores[idx] > 0}
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return {}
    
    def _get_dense_scores(self, query: str, top_k: int) -> Dict[int, float]:
        """Get dense retrieval scores"""
        if not self.dense_retriever:
            logger.warning("Dense retriever not available")
            return {}
        
        try:
            # Try different methods to get dense scores
            if hasattr(self.dense_retriever, 'search'):
                results = self.dense_retriever.search(query, top_k=top_k)
                
                # Handle different result formats
                scores_dict = {}
                for i, result in enumerate(results):
                    if hasattr(result, 'doc_id') and hasattr(result, 'score'):
                        scores_dict[result.doc_id] = result.score
                    elif isinstance(result, dict):
                        doc_id = result.get('doc_id', result.get('id', i))
                        score = result.get('score', result.get('similarity', 0.8))
                        scores_dict[doc_id] = score
                    else:
                        scores_dict[i] = 0.8  # Default score
                
                return scores_dict
                
            elif hasattr(self.dense_retriever, 'similarity_search_with_score'):
                results = self.dense_retriever.similarity_search_with_score(query, k=top_k)
                return {i: score for i, (doc, score) in enumerate(results)}
            
            else:
                logger.warning("Dense retriever has no compatible search method")
                return {}
                
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")
            return {}
    
    def _combine_scores(self, bm25_scores: Dict[int, float], 
                       dense_scores: Dict[int, float], 
                       top_k: int) -> List[RetrievalResult]:
        """Combine BM25 and dense scores using specified fusion method"""
        # Get all document indices
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        if not all_indices:
            return []
        
        combined_results = []
        
        for idx in all_indices:
            if idx >= len(self.documents):
                continue
            
            bm25_score = bm25_scores.get(idx, 0.0)
            dense_score = dense_scores.get(idx, 0.0)
            
            # Apply fusion method
            if self.fusion_method == 'weighted_sum':
                final_score = self.alpha * dense_score + (1 - self.alpha) * bm25_score
            elif self.fusion_method == 'max':
                final_score = max(dense_score, bm25_score)
            elif self.fusion_method == 'reciprocal_rank':
                # Reciprocal rank fusion (simplified)
                dense_rank = 1 / (list(dense_scores.keys()).index(idx) + 1) if idx in dense_scores else 0
                bm25_rank = 1 / (list(bm25_scores.keys()).index(idx) + 1) if idx in bm25_scores else 0
                final_score = self.alpha * dense_rank + (1 - self.alpha) * bm25_rank
            else:
                final_score = (dense_score + bm25_score) / 2
            
            result = RetrievalResult(
                doc_id=self.doc_ids[idx],
                score=final_score,
                content=self.documents[idx],
                metadata=self.doc_metadata[idx],
                bm25_score=bm25_score,
                dense_score=dense_score
            )
            
            combined_results.append(result)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            'total_documents': len(self.documents),
            'bm25_available': self.bm25 is not None,
            'dense_retriever_available': self.dense_retriever is not None,
            'alpha': self.alpha,
            'fusion_method': self.fusion_method
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        config = {
            'alpha': self.alpha,
            'fusion_method': self.fusion_method,
            'total_documents': len(self.documents)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.alpha = config.get('alpha', 0.5)
        self.fusion_method = config.get('fusion_method', 'weighted_sum')
