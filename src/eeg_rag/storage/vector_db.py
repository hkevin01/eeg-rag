"""
Vector database integration using Qdrant for semantic search.

Provides:
- Document embedding generation using sentence-transformers
- Qdrant collection management
- Vector similarity search
- Hybrid search support (dense + BM25)
"""

import logging
import hashlib
import json
import threading
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from eeg_rag.retrieval.filters import to_qdrant_filter, validate_filters

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
    vector: Optional[List[float]] = None


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
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_backend: str = "torch",
        model_revision: Optional[str] = None,
        normalize_embeddings: bool = False,
        model_dtype: str = "float32",
        onnx_threads: Optional[int] = None,
        vector_datatype: str = "float32",
        allow_legacy_index: bool = False,
        api_key: Optional[str] = None,
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
        from qdrant_client.models import Datatype

        if embedding_backend not in {"torch", "onnx", "openvino"}:
            raise ValueError("Unsupported embedding backend")
        if model_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported inference dtype")
        self.model_revision = model_revision
        self.normalize_embeddings = normalize_embeddings
        self.vector_datatype = Datatype(vector_datatype)
        self.allow_legacy_index = allow_legacy_index
        self.api_key = api_key
        self._embedding_lock = threading.Lock()
        model_kwargs: Dict[str, Any] = {"torch_dtype": model_dtype}
        if embedding_backend == "onnx" and onnx_threads is not None:
            if onnx_threads < 1:
                raise ValueError("onnx_threads must be positive")
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = onnx_threads
            model_kwargs = {
                "session_options": options,
                "provider": "CPUExecutionProvider",
            }

        # Initialize sentence transformer
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            backend=embedding_backend,
            revision=model_revision,
            model_kwargs=model_kwargs,
            trust_remote_code=False,
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        if self.model_revision is None and hasattr(
            self.embedding_model, "_first_module"
        ):
            module = self.embedding_model._first_module()
            self.model_revision = getattr(
                getattr(getattr(module, "auto_model", None), "config", None),
                "_commit_hash",
                None,
            )
        modules = getattr(self.embedding_model, "_modules", {})
        self._pooling_signature = hashlib.sha256(
            json.dumps(
                [
                    (
                        name,
                        type(module).__name__,
                        (
                            module.get_config_dict()
                            if hasattr(module, "get_config_dict")
                            else {}
                        ),
                    )
                    for name, module in modules.items()
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
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

                self._client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)
                logger.info(f"Connected to Qdrant at {self.qdrant_url}")
            except ImportError:
                logger.error(
                    "qdrant-client not installed. Run: pip install qdrant-client"
                )
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
                    self.validate_collection()
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return True

            # Create collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                    datatype=self.vector_datatype,
                ),
                metadata={"eeg_rag_embedding": self.embedding_identity},
            )

            logger.info(f"✅ Collection created: {self.collection_name}")
            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    @property
    def embedding_identity(self) -> Dict[str, Any]:
        """Return the encoding contract stored with the corpus, independent of backend."""
        return {
            "model_name": self.embedding_model_name,
            "model_revision": self.model_revision,
            "dimensions": self.embedding_dim,
            "normalization": self.normalize_embeddings,
            "pooling": self._pooling_signature,
        }

    def validate_collection(self) -> None:
        """Reject incompatible or unproven embedding spaces before reading or writing.

        Raises:
            ValueError: The stored embedding identity or vector dimensions differ.
        """
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) or vectors.size != self.embedding_dim:
            raise ValueError("Collection embedding dimensions do not match")
        identity = (info.config.metadata or {}).get("eeg_rag_embedding")
        if identity is None and self.allow_legacy_index:
            logger.warning("Using legacy index without proven embedding provenance")
        elif identity != self.embedding_identity:
            raise ValueError(
                "Collection embedding identity does not match; rebuild a rollback-safe collection"
            )

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
        with self._embedding_lock:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

        return np.asarray(embeddings, dtype=np.float32)

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
        batch_size: int = 100,
        original_vectors_path: Optional[Path] = None,
    ) -> int:
        """
        Index documents into Qdrant.

        Args:
            documents: List of document dicts with text_field and metadata
            text_field: Key containing text to embed
            batch_size: Batch size for indexing
            original_vectors_path: Durable float32 archive required for compression.

        Returns:
            Number of documents indexed
        """
        try:
            from qdrant_client.models import PointStruct
            from qdrant_client.models import Datatype

            if not documents:
                logger.warning("No documents to index")
                return 0
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            compressed = self.vector_datatype != Datatype.FLOAT32
            if compressed and original_vectors_path is None:
                raise ValueError(
                    "Compressed indexing requires a full-precision archive"
                )

            # Ensure collection exists
            if not self.create_collection(recreate=False):
                raise RuntimeError("Cannot index without a compatible collection")

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
                payload["text_length"] = len(doc.get(text_field, ""))
                if not compressed:
                    payload["embedding_vector"] = embedding.tolist()
                payload["centroid_sketch"] = self._build_centroid_sketch(embedding)

                points.append(
                    PointStruct(
                        id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                str(
                                    doc.get("chunk_id")
                                    or doc.get("id")
                                    or hashlib.sha256(
                                        doc.get(text_field, "").encode()
                                    ).hexdigest()
                                ),
                            )
                        ),
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            if original_vectors_path is not None:
                archive_path = Path(original_vectors_path)
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with archive_path.open("a", encoding="utf-8") as archive:
                    for point in points:
                        archive.write(
                            json.dumps(
                                {
                                    "collection": self.collection_name,
                                    "point_id": point.id,
                                    "embedding": self.embedding_identity,
                                    "vector": point.vector,
                                    "payload": point.payload,
                                }
                            )
                            + "\n"
                        )
                    archive.flush()
                    os.fsync(archive.fileno())

            # Batch upload
            logger.info(f"Uploading {len(points)} points to Qdrant...")
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name, points=batch, wait=True
                )
                logger.debug(
                    f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}"
                )

            logger.info(f"✅ Indexed {len(documents)} documents")
            return len(documents)

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return 0

    @staticmethod
    def _build_centroid_sketch(
        embedding: np.ndarray, sketch_size: int = 16
    ) -> List[float]:
        """Build a compact centroid sketch from a dense embedding."""
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        if vector.size == 0:
            return []

        sketch_size = max(1, min(sketch_size, vector.size))
        chunks = np.array_split(vector, sketch_size)
        sketch = np.asarray([float(chunk.mean()) for chunk in chunks], dtype=np.float32)
        norm = float(np.linalg.norm(sketch))
        if norm > 0.0:
            sketch = sketch / norm
        return sketch.tolist()

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
        filter_conditions: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
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
        validate_filters(filter_conditions or {})
        self.validate_collection()
        try:

            # Generate query embedding
            query_vector = self.embed_query(query)

            # Build filter if provided
            qdrant_filter = to_qdrant_filter(filter_conditions or {})

            # Search
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=include_vectors,
            )

            # Convert to SearchResult objects
            results = []
            for point in search_result.points:
                results.append(
                    SearchResult(
                        doc_id=str(point.id),
                        score=point.score,
                        payload=point.payload,
                        chunk_id=point.payload.get("chunk_id"),
                        vector=(
                            list(point.vector)
                            if include_vectors
                            and getattr(point, "vector", None) is not None
                            else None
                        ),
                    )
                )

            logger.debug(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def configure_bm25(self, text_field: str = "text") -> None:
        """Build a scoring full-text index without replacing corpus vectors.

        Args:
            text_field: Existing payload field holding source chunk text.
        """
        from qdrant_client import models

        self.validate_collection()
        self.client.create_payload_index(
            self.collection_name,
            field_name=text_field,
            field_schema=models.TextIndexParams(
                type="text", tokenizer="word", lowercase=True, scoring=True
            ),
            wait=True,
        )

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        retrieve_k: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        text_field: str = "text",
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Fuse native BM25 and dense candidates in one Qdrant request.

        Args:
            query: EEG literature query.
            limit: Final result count.
            retrieve_k: Candidate count for each branch.
            filters: Shared eligibility constraints.
            text_field: Payload field configured for BM25 scoring.

        Returns:
            Server-side RRF results with their original payloads.

        Raises:
            ValueError: Invalid limits or incompatible embedding identity.
        """
        from qdrant_client import models

        if limit < 1 or retrieve_k < limit:
            raise ValueError("retrieve_k must be at least the positive result limit")
        self.validate_collection()
        query_filter = to_qdrant_filter(filters or {})
        response = self.client.query_points(
            self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.TextQuery(text=query),
                    using=text_field,
                    filter=query_filter,
                    limit=retrieve_k,
                ),
                models.Prefetch(
                    query=self.embed_query(query).tolist(),
                    filter=query_filter,
                    limit=retrieve_k,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k + 1, weights=weights)),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(
                str(point.id),
                float(point.score),
                point.payload or {},
                (point.payload or {}).get("chunk_id"),
            )
            for point in response.points
        ]

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
                "vectors_count": getattr(info, "indexed_vectors_count", None),
                "points_count": info.points_count,
                "status": info.status,
                "config": {"vector_size": self.embedding_dim, "distance": "COSINE"},
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
            "domain": "Epilepsy",
        },
        {
            "text": "Recurrent neural networks for sleep stage classification",
            "title": "RNN Sleep Staging",
            "year": 2019,
            "domain": "Sleep",
        },
        {
            "text": "Brain-computer interface using motor imagery and deep learning",
            "title": "BCI Motor Imagery",
            "year": 2021,
            "domain": "BCI",
        },
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
        "deep learning EEG", limit=5, filter_conditions={"domain": "Epilepsy"}
    )
    logger.info(f"Filtered search results: {len(results)}")

    # Collection info
    info = vdb.get_collection_info()
    logger.info(f"Collection info: {info}")

    logger.info("✅ VectorDB test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vector_db()
