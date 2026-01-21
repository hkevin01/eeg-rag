"""
Local Data Agent - Hybrid Retrieval System

This agent provides fast local retrieval from indexed EEG literature corpus using
hybrid search combining BM25 (sparse) and dense semantic search with RRF fusion.

Requirements Covered:
- REQ-AGT1-001: Hybrid retrieval integration (BM25 + Dense)
- REQ-AGT1-002: Fast retrieval (<100ms target)
- REQ-AGT1-003: Citation tracking
- REQ-AGT1-004: EEG-specific indexing
- REQ-AGT1-005: Relevance scoring with RRF fusion
- REQ-AGT1-006: Query expansion with EEG domain knowledge
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import pickle

# Import new hybrid retrieval system
try:
    from eeg_rag.retrieval import (
        HybridRetriever, HybridResult,
        BM25Retriever, DenseRetriever,
        EEGQueryExpander
    )
    HYBRID_RETRIEVAL_AVAILABLE = True
except ImportError:
    HYBRID_RETRIEVAL_AVAILABLE = False
    logging.warning("Hybrid retrieval not available, using fallback mode")

# Keep FAISS for backward compatibility
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, LocalDataAgent will use fallback mode")

from eeg_rag.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentResult,
    AgentQuery
)


@dataclass
class Citation:
    """
    Citation information for a document
    
    REQ-AGT1-006: Citation data structure
    """
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    url: Optional[str] = None
    abstract: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pmid": self.pmid,
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
            "abstract": self.abstract
        }
    
    def format_citation(self) -> str:
        """
        Format citation in standard format
        
        Returns:
            Formatted citation string
            
        REQ-AGT1-007: Citation formatting
        """
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        citation = f"{authors_str} ({self.year}). {self.title}. {self.journal}."
        
        if self.pmid:
            citation += f" PMID: {self.pmid}"
        if self.doi:
            citation += f" DOI: {self.doi}"
        
        return citation


@dataclass
class SearchResult:
    """
    Single search result with relevance score
    
    REQ-AGT1-008: Search result structure
    """
    document_id: str
    content: str
    citation: Citation
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "citation": self.citation.to_dict(),
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings
    
    REQ-AGT1-009: Vector store implementation
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "Flat",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
            logger: Logger instance
        """
        self.dimension = dimension
        self.index_type = index_type
        self.logger = logger or logging.getLogger("eeg_rag.faiss_store")
        
        # Initialize index
        if FAISS_AVAILABLE:
            self.index = self._create_index()
        else:
            self.index = None
            self.fallback_vectors = []
            self.logger.warning("Using fallback mode without FAISS")
        
        # Document metadata storage
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        self.logger.info(f"FAISSVectorStore initialized (dim={dimension}, type={index_type})")
    
    def _create_index(self) -> 'faiss.Index':
        """
        Create FAISS index based on type
        
        Returns:
            FAISS index
            
        REQ-AGT1-010: Index creation
        """
        if self.index_type == "Flat":
            # Exact search (L2 distance)
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        self.logger.debug(f"Created FAISS index: {self.index_type}")
        return index
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add documents to the vector store
        
        Args:
            embeddings: Document embeddings (N x dimension)
            documents: Document metadata
            
        Returns:
            List of assigned document IDs
            
        REQ-AGT1-011: Document indexing
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # Assign IDs
        doc_ids = list(range(self.next_id, self.next_id + len(documents)))
        self.next_id += len(documents)
        
        # Store metadata
        for doc_id, doc in zip(doc_ids, documents):
            self.documents[doc_id] = doc
        
        # Add to index
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings.astype('float32'))
        else:
            # Fallback mode
            self.fallback_vectors.extend(embeddings.tolist())
        
        self.logger.info(f"Added {len(documents)} documents to index")
        return doc_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document_id, distance) tuples
            
        REQ-AGT1-012: Vector search
        """
        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            query = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query, k)
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:  # Valid result
                    results.append((int(idx), float(dist)))
            
            return results
        else:
            # Fallback: simple cosine similarity
            return self._fallback_search(query_embedding, k)
    
    def _fallback_search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Tuple[int, float]]:
        """Fallback search without FAISS"""
        if not self.fallback_vectors:
            return []
        
        # Compute cosine similarities
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for idx, vec in enumerate(self.fallback_vectors):
            vec_array = np.array(vec)
            vec_norm = np.linalg.norm(vec_array)
            
            if query_norm > 0 and vec_norm > 0:
                similarity = np.dot(query_embedding, vec_array) / (query_norm * vec_norm)
                similarities.append((idx, -similarity))  # Negative for sorting
        
        # Sort by similarity (most similar first)
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:k]
    
    def save(self, path: Path) -> None:
        """
        Save index and metadata to disk
        
        Args:
            path: Directory to save to
            
        REQ-AGT1-013: Persistence
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        else:
            # Save fallback vectors
            with open(path / "fallback_vectors.pkl", "wb") as f:
                pickle.dump(self.fallback_vectors, f)
        
        # Save documents
        with open(path / "documents.json", "w") as f:
            json.dump({
                "documents": self.documents,
                "next_id": self.next_id,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f, indent=2)
        
        self.logger.info(f"Saved vector store to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load index and metadata from disk
        
        Args:
            path: Directory to load from
            
        REQ-AGT1-014: Loading
        """
        # Load documents
        with open(path / "documents.json", "r") as f:
            data = json.load(f)
            self.documents = {int(k): v for k, v in data["documents"].items()}
            self.next_id = data["next_id"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
        
        # Load index
        if FAISS_AVAILABLE:
            index_path = path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
        else:
            # Load fallback vectors
            fallback_path = path / "fallback_vectors.pkl"
            if fallback_path.exists():
                with open(fallback_path, "rb") as f:
                    self.fallback_vectors = pickle.load(f)
        
        self.logger.info(f"Loaded vector store from {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if FAISS_AVAILABLE and self.index is not None:
            return {
                "total_documents": len(self.documents),
                "dimension": self.dimension,
                "index_type": self.index_type,
                "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
                "using_faiss": True
            }
        else:
            return {
                "total_documents": len(self.documents),
                "dimension": self.dimension,
                "index_type": "fallback",
                "index_size": len(self.fallback_vectors),
                "using_faiss": False
            }


class LocalDataAgent(BaseAgent):
    """
    Local data agent for fast hybrid retrieval
    
    REQ-AGT1-015: Main agent implementation with hybrid search
    """
    
    def __init__(
        self,
        vector_store_path: Optional[Path] = None,
        embedding_dimension: int = 768,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        use_hybrid_retrieval: bool = True
    ):
        """
        Initialize local data agent
        
        Args:
            vector_store_path: Path to load/save vector store
            embedding_dimension: Dimension of embeddings
            config: Configuration options
            logger: Logger instance
            use_hybrid_retrieval: Use new hybrid retrieval system (recommended)
        """
        super().__init__(
            agent_type=AgentType.LOCAL_DATA,
            name="local_data_agent",
            config=config or {},
            logger=logger or logging.getLogger("eeg_rag.local_data_agent")
        )
        
        # Configuration
        self.top_k = config.get("top_k", 5) if config else 5
        self.min_relevance_score = config.get("min_relevance_score", 0.3) if config else 0.3
        self.use_hybrid_retrieval = use_hybrid_retrieval and HYBRID_RETRIEVAL_AVAILABLE
        
        # Initialize retrieval system
        if self.use_hybrid_retrieval:
            self.logger.info("Initializing hybrid retrieval system (BM25 + Dense + RRF)")
            self._init_hybrid_retrieval(config or {})
        else:
            self.logger.info("Initializing legacy FAISS retrieval")
            # Initialize legacy FAISS vector store
            self.vector_store = FAISSVectorStore(
                dimension=embedding_dimension,
                index_type=config.get("index_type", "Flat") if config else "Flat",
                logger=self.logger
            )
            
            # Load existing index if path provided
            if vector_store_path and vector_store_path.exists():
                self.vector_store.load(vector_store_path)
                self.logger.info(f"Loaded vector store from {vector_store_path}")
        
        self.logger.info(f"LocalDataAgent initialized (hybrid={self.use_hybrid_retrieval})")
    
    def _init_hybrid_retrieval(self, config: Dict[str, Any]) -> None:
        """
        Initialize hybrid retrieval system with BM25, Dense, and Query Expansion
        
        Args:
            config: Configuration dictionary with optional keys:
                - qdrant_url: Qdrant server URL (default: http://localhost:6333)
                - qdrant_collection: Collection name (default: eeg_papers)
                - bm25_cache_dir: BM25 index cache directory
                - bm25_weight: Weight for BM25 scores (default: 0.5)
                - dense_weight: Weight for dense scores (default: 0.5)
                - rrf_k: RRF rank constant (default: 60)
                - use_query_expansion: Enable EEG query expansion (default: True)
                - retrieve_k: Number of results to retrieve before reranking (default: 20)
        """
        # Get config values
        qdrant_url = config.get("qdrant_url", "http://localhost:6333")
        qdrant_collection = config.get("qdrant_collection", "eeg_papers")
        bm25_cache_dir = config.get("bm25_cache_dir", "data/embeddings/cache/bm25")
        
        # Initialize retrievers
        try:
            # BM25 sparse retriever
            self.bm25_retriever = BM25Retriever(cache_dir=bm25_cache_dir)
            self.logger.info(f"BM25 retriever initialized (cache: {bm25_cache_dir})")
            
            # Dense semantic retriever
            self.dense_retriever = DenseRetriever(
                url=qdrant_url,
                collection_name=qdrant_collection
            )
            self.logger.info(f"Dense retriever initialized (Qdrant: {qdrant_url}/{qdrant_collection})")
            
            # Hybrid retriever with RRF fusion
            self.hybrid_retriever = HybridRetriever(
                bm25_retriever=self.bm25_retriever,
                dense_retriever=self.dense_retriever,
                bm25_weight=config.get("bm25_weight", 0.5),
                dense_weight=config.get("dense_weight", 0.5),
                rrf_k=config.get("rrf_k", 60),
                use_query_expansion=config.get("use_query_expansion", True)
            )
            
            # Store config
            self.retrieve_k = config.get("retrieve_k", 20)
            
            self.logger.info("Hybrid retrieval system fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid retrieval: {e}")
            self.use_hybrid_retrieval = False
            raise
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute local search using hybrid retrieval or legacy FAISS
        
        Args:
            query: Search query
            
        Returns:
            AgentResult with search results
        """
        try:
            start_time = datetime.now()
            
            if self.use_hybrid_retrieval:
                # Use new hybrid retrieval system
                search_results = await self._hybrid_search(query.text)
            else:
                # Use legacy FAISS search
                search_results = await self._faiss_search(query.text)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"Local search completed: {len(search_results)} results "
                f"in {elapsed_time*1000:.1f}ms (hybrid={self.use_hybrid_retrieval})"
            )
            
            return AgentResult(
                success=True,
                data={
                    "results": [r.to_dict() for r in search_results],
                    "total_results": len(search_results),
                    "search_time_ms": elapsed_time * 1000,
                    "query": query.text
                },
                metadata={
                    "source": "hybrid_retrieval" if self.use_hybrid_retrieval else "local_faiss",
                    "top_k": self.top_k,
                    "min_relevance": self.min_relevance_score
                },
                agent_type=AgentType.LOCAL_DATA,
                elapsed_time=elapsed_time
            )
            
        except Exception as e:
            self.logger.exception(f"Local search failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.LOCAL_DATA
            )
    
    async def _hybrid_search(self, query_text: str) -> List[SearchResult]:
        """
        Perform hybrid search using BM25 + Dense + RRF fusion
        
        Args:
            query_text: Search query text
            
        Returns:
            List of SearchResult objects
        """
        # Perform hybrid search
        hybrid_results: List[HybridResult] = self.hybrid_retriever.search(
            query=query_text,
            top_k=self.top_k,
            retrieve_k=self.retrieve_k
        )
        
        # Convert HybridResult to SearchResult format
        search_results = []
        for result in hybrid_results:
            # Extract metadata
            metadata = result.metadata or {}
            
            # Create citation from metadata
            citation = Citation(
                pmid=metadata.get("pmid"),
                doi=metadata.get("doi"),
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                journal=metadata.get("journal", ""),
                year=metadata.get("year"),
                url=metadata.get("url"),
                abstract=metadata.get("abstract", "")
            )
            
            # Convert to SearchResult
            search_result = SearchResult(
                document_id=result.doc_id,
                content=result.text,
                citation=citation,
                relevance_score=result.rrf_score,  # Use RRF score as relevance
                metadata={
                    **metadata,
                    "bm25_score": result.bm25_score,
                    "dense_score": result.dense_score,
                    "bm25_rank": result.bm25_rank,
                    "dense_rank": result.dense_rank
                }
            )
            
            # Filter by relevance threshold
            if search_result.relevance_score >= self.min_relevance_score:
                search_results.append(search_result)
        
        return search_results
    
    async def _faiss_search(self, query_text: str) -> List[SearchResult]:
        """
        Legacy FAISS search (for backward compatibility)
        
        Args:
            query_text: Search query text
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self._generate_mock_embedding(query_text)
        
        # Search vector store
        raw_results = self.vector_store.search(
            query_embedding,
            k=self.top_k
        )
        
        # Convert to SearchResult objects
        search_results = []
        for doc_id, distance in raw_results:
            if doc_id in self.vector_store.documents:
                doc = self.vector_store.documents[doc_id]
                
                # Convert distance to relevance score (0-1)
                relevance_score = 1.0 / (1.0 + distance)
                
                if relevance_score >= self.min_relevance_score:
                    result = SearchResult(
                        document_id=str(doc_id),
                        content=doc.get("content", ""),
                        citation=Citation(**doc.get("citation", {})),
                        relevance_score=relevance_score,
                        metadata=doc.get("metadata", {})
                    )
                    search_results.append(result)
        
        return search_results
    
    async def execute_old(self, query: AgentQuery) -> AgentResult:
        """
        Execute local search
        
        Args:
            query: Search query
            
        Returns:
            AgentResult with search results
        """
        try:
            start_time = datetime.now()
            
            # For now, use a simple text-based fallback
            # In production, this would use actual embeddings
            query_embedding = self._generate_mock_embedding(query.text)
            
            # Search vector store
            raw_results = self.vector_store.search(
                query_embedding,
                k=self.top_k
            )
            
            # Convert to SearchResult objects
            search_results = []
            for doc_id, distance in raw_results:
                if doc_id in self.vector_store.documents:
                    doc = self.vector_store.documents[doc_id]
                    
                    # Convert distance to relevance score (0-1)
                    relevance_score = 1.0 / (1.0 + distance)
                    
                    if relevance_score >= self.min_relevance_score:
                        result = SearchResult(
                            document_id=str(doc_id),
                            content=doc.get("content", ""),
                            citation=Citation(**doc.get("citation", {})),
                            relevance_score=relevance_score,
                            metadata=doc.get("metadata", {})
                        )
                        search_results.append(result)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(
                f"Local search completed: {len(search_results)} results "
                f"in {elapsed_time*1000:.1f}ms"
            )
            
            return AgentResult(
                success=True,
                data={
                    "results": [r.to_dict() for r in search_results],
                    "total_results": len(search_results),
                    "search_time_ms": elapsed_time * 1000,
                    "query": query.text
                },
                metadata={
                    "source": "local_faiss",
                    "top_k": self.top_k,
                    "min_relevance": self.min_relevance_score
                },
                agent_type=AgentType.LOCAL_DATA,
                elapsed_time=elapsed_time
            )
            
        except Exception as e:
            self.logger.exception(f"Local search failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.LOCAL_DATA
            )
    
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate mock embedding for testing
        
        In production, this would use a real embedding model like:
        - sentence-transformers
        - OpenAI embeddings
        - Custom EEG-trained model
        
        Args:
            text: Input text
            
        Returns:
            Mock embedding vector
        """
        # Simple hash-based mock embedding
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.vector_store.dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Add documents to the local store
        
        Args:
            documents: List of document dictionaries
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of assigned document IDs
        """
        if embeddings is None:
            # Generate embeddings
            embeddings = np.array([
                self._generate_mock_embedding(doc.get("content", ""))
                for doc in documents
            ])
        
        doc_ids = self.vector_store.add_documents(embeddings, documents)
        self.logger.info(f"Added {len(documents)} documents")
        return doc_ids
    
    def save_index(self, path: Path) -> None:
        """Save vector store to disk"""
        self.vector_store.save(path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        base_stats = super().get_statistics()
        store_stats = self.vector_store.get_statistics()
        
        return {
            **base_stats,
            "vector_store": store_stats
        }


# Export public interface
__all__ = [
    "LocalDataAgent",
    "FAISSVectorStore",
    "Citation",
    "SearchResult"
]
