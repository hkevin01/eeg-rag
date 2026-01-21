"""
Dense Retriever for EEG-RAG.

This module provides a clean interface to the vector database for semantic search.
It wraps the VectorDB class with retrieval-specific functionality.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from eeg_rag.storage.vector_db import VectorDB, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class DenseResult:
    """Result from dense (semantic) search."""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


class DenseRetriever:
    """
    Dense retrieval using semantic embeddings.
    
    This retriever uses sentence-transformers to convert queries and documents
    into dense vectors, then performs similarity search in vector space.
    Particularly effective for semantic/conceptual queries.
    
    Example:
        >>> retriever = DenseRetriever(
        ...     url="http://localhost:6333",
        ...     collection_name="eeg_papers"
        ... )
        >>> results = retriever.search("neural networks for seizure prediction", top_k=5)
        >>> print(results[0].doc_id, results[0].score)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "eeg_papers",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize dense retriever.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of Qdrant collection
            model_name: Sentence-transformers model name
        """
        self.vector_db = VectorDB(
            qdrant_url=url,
            collection_name=collection_name,
            embedding_model=model_name
        )
        
        logger.info(f"Initialized DenseRetriever")
        logger.info(f"  URL: {url}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Model: {model_name}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DenseResult]:
        """
        Search documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional Qdrant filters (e.g., {"year": 2020})
            
        Returns:
            List of DenseResult objects sorted by similarity (descending)
            
        Example:
            >>> results = retriever.search("deep learning EEG classification", top_k=5)
            >>> for r in results:
            ...     print(f"{r.doc_id}: {r.score:.3f}")
        """
        # Use VectorDB search
        search_results: List[SearchResult] = self.vector_db.search(
            query=query,
            limit=top_k,
            filter_conditions=filters
        )
        
        # Convert to DenseResult
        results = []
        for sr in search_results:
            results.append(DenseResult(
                doc_id=sr.doc_id,
                score=sr.score,
                text=sr.payload.get("text", ""),
                metadata=sr.payload.get("metadata", {}),
                chunk_id=sr.chunk_id
            ))
        
        logger.info(f"Dense search for '{query}' returned {len(results)} results")
        if results:
            logger.info(f"  Top result: {results[0].doc_id} (score: {results[0].score:.3f})")
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        
        Returns:
            Dictionary with collection statistics
        """
        return self.vector_db.get_collection_info()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    # Create retriever (assumes Qdrant is running with eeg_papers collection)
    retriever = DenseRetriever(
        url="http://localhost:6333",
        collection_name="eeg_papers"
    )
    
    # Test search
    results = retriever.search("epilepsy seizure detection CNN", top_k=3)
    
    print("\n🔍 Dense Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Doc {result.doc_id}: {result.score:.3f}")
        print(f"   {result.text[:80]}...")
        
    # Show collection info
    info = retriever.get_collection_info()
    print(f"\n📊 Collection Info: {info}")
