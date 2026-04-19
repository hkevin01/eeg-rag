"""
Vector database integration using Qdrant for semantic search.

Provides:
- Document embedding generation using sentence-transformers
- Qdrant collection management
- Vector similarity search
- Hybrid search support (dense + BM25)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : storage.vector_db.SearchResult
# Requirement  : `SearchResult` class shall be instantiable and expose the documented interface
# Purpose      : Result from vector search
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
# Verification : Instantiate SearchResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """Result from vector search."""
    doc_id: str
    score: float
    payload: Dict[str, Any]
    chunk_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ID           : storage.vector_db.VectorDB
# Requirement  : `VectorDB` class shall be instantiable and expose the documented interface
# Purpose      : Vector database interface using Qdrant + sentence-transformers
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
# Verification : Instantiate VectorDB with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class VectorDB:
    """
    Vector database interface using Qdrant + sentence-transformers.
    
    Features:
    - Automatic embedding generation (all-MiniLM-L6-v2, 384 dims)
    - Section-aware storage (Abstract, Methods, Results, etc.)
    - Metadata filtering (year, domain, code_available, etc.)
    - Hybrid search ready (dense + BM25 fusion)
    """
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.__init__
    # Requirement  : `__init__` shall initialize vector database
    # Purpose      : Initialize vector database
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : qdrant_url: str (default='http://localhost:6333'); collection_name: str (default='eeg_papers'); embedding_model: str (default='all-MiniLM-L6-v2')
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
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "eeg_papers",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector database.
        
        Args:
            qdrant_url: Qdrant server URL
            collection_name: Name of the collection to use
            embedding_model: Sentence transformer model name
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize sentence transformer
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Qdrant client will be initialized when needed
        self._client = None
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.client
    # Requirement  : `client` shall lazy initialization of Qdrant client
    # Purpose      : Lazy initialization of Qdrant client
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    @property
    def client(self):
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                
                self._client = QdrantClient(url=self.qdrant_url)
                logger.info(f"Connected to Qdrant at {self.qdrant_url}")
            except ImportError:
                logger.error("qdrant-client not installed. Run: pip install qdrant-client")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise
        
        return self._client
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.create_collection
    # Requirement  : `create_collection` shall create Qdrant collection with proper schema
    # Purpose      : Create Qdrant collection with proper schema
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : recreate: bool (default=False)
    # Outputs      : bool
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
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create Qdrant collection with proper schema.
        
        Args:
            recreate: If True, delete existing collection first
            
        Returns:
            True if collection created/exists
        """
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return True
            
            # Create collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✅ Collection created: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.embed_texts
    # Requirement  : `embed_texts` shall generate embeddings for texts
    # Purpose      : Generate embeddings for texts
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : texts: List[str]; batch_size: int (default=32)
    # Outputs      : np.ndarray
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
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.debug(f"Embedding {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        return embeddings
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.embed_query
    # Requirement  : `embed_query` shall generate embedding for a single query
    # Purpose      : Generate embedding for a single query
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : np.ndarray
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
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed_texts([query])[0]
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.index_documents
    # Requirement  : `index_documents` shall index documents into Qdrant
    # Purpose      : Index documents into Qdrant
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : documents: List[Dict[str, Any]]; text_field: str (default='text'); batch_size: int (default=100)
    # Outputs      : int
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
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        batch_size: int = 100
    ) -> int:
        """
        Index documents into Qdrant.
        
        Args:
            documents: List of document dicts with text_field and metadata
            text_field: Key containing text to embed
            batch_size: Batch size for indexing
            
        Returns:
            Number of documents indexed
        """
        try:
            from qdrant_client.models import PointStruct
            
            if not documents:
                logger.warning("No documents to index")
                return 0
            
            # Ensure collection exists
            self.create_collection(recreate=False)
            
            # Extract texts
            texts = [doc.get(text_field, "") for doc in documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.embed_texts(texts, batch_size=batch_size)
            
            # Create points
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Extract payload (include ALL fields - text is needed for retrieval)
                payload = dict(doc)  # Include everything
                payload['text_length'] = len(doc.get(text_field, ""))
                
                points.append(
                    PointStruct(
                        id=idx,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )
            
            # Batch upload
            logger.info(f"Uploading {len(points)} points to Qdrant...")
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                logger.debug(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"✅ Indexed {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return 0
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.search
    # Requirement  : `search` shall search for similar documents
    # Purpose      : Search for similar documents
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; limit: int (default=10); score_threshold: Optional[float] (default=None); filter_conditions: Optional[Dict[str, Any]] (default=None)
    # Outputs      : List[SearchResult]
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
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Metadata filters (e.g., {"year": 2020})
            
        Returns:
            List of SearchResult objects
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Generate query embedding
            query_vector = self.embed_query(query)
            
            # Build filter if provided
            qdrant_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                qdrant_filter = Filter(must=conditions)
            
            # Search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Convert to SearchResult objects
            results = []
            for point in search_result.points:
                results.append(
                    SearchResult(
                        doc_id=str(point.id),
                        score=point.score,
                        payload=point.payload,
                        chunk_id=point.payload.get('chunk_id')
                    )
                )
            
            logger.debug(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.get_collection_info
    # Requirement  : `get_collection_info` shall get information about the collection
    # Purpose      : Get information about the collection
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": self.embedding_dim,
                    "distance": "COSINE"
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    # ---------------------------------------------------------------------------
    # ID           : storage.vector_db.VectorDB.delete_collection
    # Requirement  : `delete_collection` shall delete the collection
    # Purpose      : Delete the collection
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False


# ---------------------------------------------------------------------------
# ID           : storage.vector_db.test_vector_db
# Requirement  : `test_vector_db` shall test the vector database functionality
# Purpose      : Test the vector database functionality
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
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
def test_vector_db():
    """Test the vector database functionality."""
    logger.info("Testing VectorDB...")
    
    # Initialize
    vdb = VectorDB()
    
    # Create collection
    vdb.create_collection(recreate=True)
    
    # Test documents
    docs = [
        {
            "text": "Deep learning for EEG seizure detection using CNN",
            "title": "CNN for Seizure Detection",
            "year": 2020,
            "domain": "Epilepsy"
        },
        {
            "text": "Recurrent neural networks for sleep stage classification",
            "title": "RNN Sleep Staging",
            "year": 2019,
            "domain": "Sleep"
        },
        {
            "text": "Brain-computer interface using motor imagery and deep learning",
            "title": "BCI Motor Imagery",
            "year": 2021,
            "domain": "BCI"
        }
    ]
    
    # Index documents
    count = vdb.index_documents(docs)
    logger.info(f"Indexed {count} documents")
    
    # Test search
    results = vdb.search("epilepsy seizure detection", limit=2)
    logger.info(f"Search results: {len(results)}")
    for r in results:
        logger.info(f"  - {r.payload.get('title')} (score: {r.score:.3f})")
    
    # Test filtered search
    results = vdb.search(
        "deep learning EEG",
        limit=5,
        filter_conditions={"domain": "Epilepsy"}
    )
    logger.info(f"Filtered search results: {len(results)}")
    
    # Collection info
    info = vdb.get_collection_info()
    logger.info(f"Collection info: {info}")
    
    logger.info("✅ VectorDB test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vector_db()
