"""
Embedding Generation with PubMedBERT
Generates vector embeddings for text chunks using biomedical language models

Requirements:
- REQ-EMB-001: Use PubMedBERT for biomedical text
- REQ-EMB-002: Batch processing for efficiency (32+ chunks at once)
- REQ-EMB-003: GPU acceleration when available
- REQ-EMB-004: Normalize embeddings for cosine similarity
- REQ-EMB-005: Cache embeddings to avoid recomputation
- REQ-EMB-006: Handle out-of-memory errors gracefully
- REQ-EMB-007: Progress tracking for large batches
- REQ-EMB-008: Export embeddings in numpy format
- REQ-EMB-009: Dimensionality: 768 (PubMedBERT-base)
- REQ-EMB-010: Processing speed: 100+ chunks/second on GPU
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import json
import logging
from datetime import datetime


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    chunk_id: str
    embedding: np.ndarray
    model_name: str
    embedding_dim: int
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without full embedding array)"""
        return {
            'chunk_id': self.chunk_id,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'embedding_norm': float(np.linalg.norm(self.embedding))
        }


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding generation"""
    embeddings: List[EmbeddingResult]
    total_chunks: int
    total_time: float
    average_time_per_chunk: float
    model_name: str
    batch_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_chunks': self.total_chunks,
            'total_time': self.total_time,
            'average_time_per_chunk': self.average_time_per_chunk,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'embeddings': [e.to_dict() for e in self.embeddings]
        }


class MockEmbeddingModel:
    """Mock embedding model for testing without downloading PubMedBERT"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.model_name = "mock-pubmedbert"

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Generate mock embeddings"""
        import time
        time.sleep(0.01 * len(texts))  # Simulate processing time

        # Generate random embeddings with consistent seed for reproducibility
        embeddings = []
        for text in texts:
            # Use text hash as seed for consistency
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)


class PubMedBERTEmbedder:
    """
    Embedding generator using PubMedBERT

    Uses sentence-transformers with PubMedBERT model optimized for
    biomedical text. Supports batch processing and GPU acceleration.
    """

    # Default model: PubMedBERT from Microsoft
    DEFAULT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    EMBEDDING_DIM = 768

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        use_gpu: bool = True,
        use_mock: bool = True,  # Use mock for testing
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize embedding generator

        Args:
            model_name: HuggingFace model name (default: PubMedBERT)
            batch_size: Batch size for processing
            use_gpu: Use GPU if available
            use_mock: Use mock model (for testing without downloading)
            cache_dir: Directory to cache model and embeddings
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.use_mock = use_mock
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embeddings/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        if use_mock:
            self.model = MockEmbeddingModel(self.EMBEDDING_DIM)
            self.model_name = "mock-pubmedbert"
        else:
            self._load_model()

        # Statistics
        self.stats = {
            'total_chunks_embedded': 0,
            'total_batches': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Load PubMedBERT model (real implementation)"""
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading model: {self.model_name}")

            # Load model with sentence-transformers
            device = 'cuda' if self.use_gpu else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)

            self.logger.info(f"Model loaded successfully on {device}")

        except ImportError:
            self.logger.warning(
                "sentence-transformers not installed. Using mock model. "
                "Install with: pip install sentence-transformers"
            )
            self.model = MockEmbeddingModel(self.EMBEDDING_DIM)
            self.use_mock = True

    def embed_texts(
        self,
        texts: List[str],
        chunk_ids: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            chunk_ids: Optional list of chunk identifiers
            show_progress: Show progress bar

        Returns:
            BatchEmbeddingResult with embeddings and statistics
        """
        import time
        start_time = time.time()

        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                total_chunks=0,
                total_time=0.0,
                average_time_per_chunk=0.0,
                model_name=self.model_name,
                batch_size=self.batch_size
            )

        # Generate chunk IDs if not provided
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(texts))]

        self.logger.info(f"Embedding {len(texts)} texts with batch_size={self.batch_size}")

        # Generate embeddings
        embeddings_array = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress and not self.use_mock
        )

        # Create EmbeddingResult objects
        results = []
        processing_time = time.time() - start_time
        time_per_chunk = processing_time / len(texts) if texts else 0

        for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, embeddings_array)):
            result = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                model_name=self.model_name,
                embedding_dim=len(embedding),
                processing_time=time_per_chunk,
                metadata={'batch_index': i}
            )
            results.append(result)

        # Update statistics
        self.stats['total_chunks_embedded'] += len(texts)
        self.stats['total_batches'] += (len(texts) + self.batch_size - 1) // self.batch_size
        self.stats['total_time'] += processing_time

        self.logger.info(
            f"Embedding complete: {len(texts)} texts in {processing_time:.2f}s "
            f"({len(texts)/processing_time:.1f} texts/sec)"
        )

        return BatchEmbeddingResult(
            embeddings=results,
            total_chunks=len(texts),
            total_time=processing_time,
            average_time_per_chunk=time_per_chunk,
            model_name=self.model_name,
            batch_size=self.batch_size
        )

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = 'text',
        id_field: str = 'chunk_id'
    ) -> BatchEmbeddingResult:
        """
        Embed text chunks from structured data

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text
            id_field: Field name containing chunk ID

        Returns:
            BatchEmbeddingResult with embeddings
        """
        texts = [chunk[text_field] for chunk in chunks]
        chunk_ids = [chunk[id_field] for chunk in chunks]

        return self.embed_texts(texts, chunk_ids)

    def save_embeddings(
        self,
        embeddings,  # Can be BatchEmbeddingResult or List[EmbeddingResult]
        output_path: Path
    ):
        """
        Save embeddings to disk

        Args:
            embeddings: BatchEmbeddingResult or List[EmbeddingResult] to save
            output_path: Path to save embeddings (numpy .npz format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract embeddings array and metadata
        if isinstance(embeddings, BatchEmbeddingResult):
            emb_list = embeddings.embeddings
            model_name = embeddings.model_name
        else:
            emb_list = embeddings
            model_name = emb_list[0].model_name if emb_list else "unknown"

        embeddings_array = np.array([e.embedding for e in emb_list])
        chunk_ids = [e.chunk_id for e in emb_list]

        # Save as numpy compressed format
        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            chunk_ids=chunk_ids,
            model_name=model_name,
            embedding_dim=emb_list[0].embedding_dim if emb_list else 0
        )

        # Save metadata as JSON
        metadata_path = output_path.with_suffix('.json')

        if isinstance(embeddings, BatchEmbeddingResult):
            metadata = embeddings.to_dict()
            # Remove embeddings array from metadata (too large)
            metadata['embeddings'] = [
                {k: v for k, v in e.items() if k != 'embedding'}
                for e in metadata['embeddings']
            ]
        else:
            metadata = {
                'total_chunks': len(emb_list),
                'model_name': model_name,
                'embeddings': [e.to_dict() for e in emb_list]
            }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Embeddings saved to {output_path}")

    def load_embeddings(self, input_path: Path) -> List[np.ndarray]:
        """
        Load embeddings from disk

        Args:
            input_path: Path to embeddings file

        Returns:
            List of numpy arrays (embeddings)
        """
        data = np.load(input_path, allow_pickle=True)

        # Return list of embedding arrays
        return [data['embeddings'][i] for i in range(len(data['embeddings']))]

    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.EMBEDDING_DIM,
            'batch_size': self.batch_size,
            'total_chunks_embedded': self.stats['total_chunks_embedded'],
            'total_batches': self.stats['total_batches'],
            'total_time': self.stats['total_time'],
            'average_speed': (
                self.stats['total_chunks_embedded'] / self.stats['total_time']
                if self.stats['total_time'] > 0 else 0
            ),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses']
        }
