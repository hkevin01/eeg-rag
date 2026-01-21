"""
BM25 Sparse Retriever for EEG-RAG.

This module implements BM25 (Best Match 25) sparse retrieval for EEG research papers.
BM25 is a probabilistic ranking function that provides strong keyword-based search.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pickle
import os

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 search."""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class BM25Retriever:
    """
    Sparse retrieval using BM25 algorithm.
    
    BM25 (Best Match 25) is a probabilistic ranking function that scores documents
    based on term frequency and document length normalization. It's particularly
    effective for keyword-based queries.
    
    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.index_documents([
        ...     {"id": "1", "text": "EEG epilepsy detection using CNN"},
        ...     {"id": "2", "text": "Sleep staging with deep learning"}
        ... ])
        >>> results = retriever.search("epilepsy seizure", top_k=5)
        >>> print(results[0].doc_id, results[0].score)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            cache_dir: Directory to cache BM25 index (optional)
        """
        self.cache_dir = cache_dir or "data/bm25_cache"
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        
        logger.info(f"Initialized BM25Retriever with cache_dir={self.cache_dir}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Simple whitespace tokenization with lowercasing.
        For production, consider using nltk or spacy for better tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with 'id', 'text', and optional metadata
            
        Example:
            >>> docs = [
            ...     {"id": "1", "text": "EEG signal processing", "title": "Paper 1"},
            ...     {"id": "2", "text": "Deep learning for EEG", "title": "Paper 2"}
            ... ]
            >>> retriever.index_documents(docs)
        """
        logger.info(f"Indexing {len(documents)} documents for BM25...")
        
        self.documents = documents
        self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"✅ Indexed {len(documents)} documents")
        
        # Cache the index
        self._save_cache()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[BM25Result]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_fn: Optional function to filter documents (receives doc dict)
            
        Returns:
            List of BM25Result objects sorted by score (descending)
            
        Example:
            >>> results = retriever.search("epilepsy detection", top_k=5)
            >>> for r in results:
            ...     print(f"{r.doc_id}: {r.score:.3f}")
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built, attempting to load from cache...")
            self._load_cache()
            
            if self.bm25 is None:
                raise ValueError("No BM25 index available. Call index_documents() first.")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create results
        results = []
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            
            # Apply filter if provided
            if filter_fn and not filter_fn(doc):
                continue
            
            results.append(BM25Result(
                doc_id=doc["id"],
                score=float(score),
                text=doc["text"],
                metadata=doc.get("metadata", {})
            ))
        
        # Sort by score descending and take top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        logger.info(f"BM25 search for '{query}' returned {len(results)} results")
        if results:
            logger.info(f"  Top result: {results[0].doc_id} (score: {results[0].score:.3f})")
        
        return results
    
    def _save_cache(self) -> None:
        """Save BM25 index to disk."""
        if not self.bm25:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "documents": self.documents,
                    "tokenized_corpus": self.tokenized_corpus
                }, f)
            logger.info(f"💾 Saved BM25 cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save BM25 cache: {e}")
    
    def _load_cache(self) -> None:
        """Load BM25 index from disk."""
        cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")
        
        if not os.path.exists(cache_path):
            logger.warning(f"No BM25 cache found at {cache_path}")
            return
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            
            logger.info(f"📂 Loaded BM25 cache from {cache_path}")
            logger.info(f"  Documents: {len(self.documents)}")
        except Exception as e:
            logger.error(f"Failed to load BM25 cache: {e}")


if __name__ == "__main__":
    # Simple test
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
        }
    ]
    
    # Create and test retriever
    retriever = BM25Retriever(cache_dir="data/bm25_cache_test")
    retriever.index_documents(docs)
    
    # Test search
    results = retriever.search("epilepsy seizure detection", top_k=3)
    
    print("\n🔍 BM25 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Doc {result.doc_id}: {result.score:.3f}")
        print(f"   {result.text[:80]}...")
