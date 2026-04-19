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


# ---------------------------------------------------------------------------
# ID           : retrieval.dense_retriever.DenseResult
# Requirement  : `DenseResult` class shall be instantiable and expose the documented interface
# Purpose      : Result from dense (semantic) search
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
# Verification : Instantiate DenseResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class DenseResult:
    """Result from dense (semantic) search."""
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ID           : retrieval.dense_retriever.DenseRetriever
# Requirement  : `DenseRetriever` class shall be instantiable and expose the documented interface
# Purpose      : Dense retrieval using semantic embeddings
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
# Verification : Instantiate DenseRetriever with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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

    # Model presets for quick configuration
    MODEL_PRESETS: Dict[str, str] = {
        # General-purpose (fast, good quality)
        "general": "sentence-transformers/all-MiniLM-L6-v2",
        # Biomedical domain — best for EEG/neuroscience literature
        "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        # Biomedical, smaller / faster variant
        "biobert": "dmis-lab/biobert-base-cased-v1.2",
        # High-quality general (larger)
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        # Multi-lingual for non-English EEG literature
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    }

    # ---------------------------------------------------------------------------
    # ID           : retrieval.dense_retriever.DenseRetriever.__init__
    # Requirement  : `__init__` shall initialize dense retriever
    # Purpose      : Initialize dense retriever
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : url: str (default='http://localhost:6333'); collection_name: str (default='eeg_papers'); model_name: str (default='sentence-transformers/all-MiniLM-L6-v2'); model_preset: Optional[str] (default=None)
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
        url: str = "http://localhost:6333",
        collection_name: str = "eeg_papers",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_preset: Optional[str] = None,
    ):
        """
        Initialize dense retriever.

        Args:
            url: Qdrant server URL
            collection_name: Name of Qdrant collection
            model_name: Sentence-transformers model name or HuggingFace ID.
                Use ``model_preset`` for convenience aliases.
            model_preset: Shortcut key from ``MODEL_PRESETS`` dict.
                Overrides ``model_name`` when provided.
                Options: "general", "pubmedbert", "biobert", "mpnet",
                "multilingual".  Use "pubmedbert" for best EEG/biomedical
                retrieval quality.

        Example::

            # Use PubMedBERT for domain-optimal biomedical retrieval
            retriever = DenseRetriever(
                model_preset="pubmedbert",
                collection_name="eeg_papers",
            )
        """
        if model_preset:
            resolved_model = self.MODEL_PRESETS.get(model_preset, model_name)
            if model_preset not in self.MODEL_PRESETS:
                logger.warning(
                    "Unknown model_preset '%s'; valid options: %s. "
                    "Falling back to model_name='%s'.",
                    model_preset,
                    list(self.MODEL_PRESETS.keys()),
                    model_name,
                )
            model_name = resolved_model

        self.model_name = model_name
        self.model_preset = model_preset

        self.vector_db = VectorDB(
            qdrant_url=url,
            collection_name=collection_name,
            embedding_model=model_name
        )

        logger.info("Initialized DenseRetriever")
        logger.info("  URL: %s", url)
        logger.info("  Collection: %s", collection_name)
        logger.info("  Model: %s", model_name)

    # ---------------------------------------------------------------------------
    # ID           : retrieval.dense_retriever.DenseRetriever.search
    # Requirement  : `search` shall search documents using semantic similarity
    # Purpose      : Search documents using semantic similarity
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; top_k: int (default=10); filters: Optional[Dict[str, Any]] (default=None)
    # Outputs      : List[DenseResult]
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

    # ---------------------------------------------------------------------------
    # ID           : retrieval.dense_retriever.DenseRetriever.get_collection_info
    # Requirement  : `get_collection_info` shall get information about the vector collection
    # Purpose      : Get information about the vector collection
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
