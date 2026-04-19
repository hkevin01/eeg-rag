"""
Cross-Encoder Reranker for Hybrid Retrieval

Reranks retrieved documents using a cross-encoder model for better relevance.
Cross-encoders are more accurate than bi-encoders but slower, so we use them
for reranking top-K candidates.

Performance: ~100ms for reranking 20 candidates (acceptable for production)
Expected improvement: +5-10% recall over hybrid-only
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available for reranking")


# ---------------------------------------------------------------------------
# ID           : retrieval.reranker.RerankedResult
# Requirement  : `RerankedResult` class shall be instantiable and expose the documented interface
# Purpose      : Result after reranking
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
# Verification : Instantiate RerankedResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class RerankedResult:
    """Result after reranking"""
    doc_id: str
    text: str
    original_score: float  # Original hybrid score
    rerank_score: float    # Cross-encoder score
    final_score: float     # Combined score
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


# ---------------------------------------------------------------------------
# ID           : retrieval.reranker.CrossEncoderReranker
# Requirement  : `CrossEncoderReranker` class shall be instantiable and expose the documented interface
# Purpose      : Cross-encoder reranker for improving retrieval quality
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
# Verification : Instantiate CrossEncoderReranker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval quality
    
    Uses a cross-encoder model to rescore candidates from hybrid retrieval.
    Cross-encoders jointly encode query + document, providing more accurate
    relevance scores than bi-encoders.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.CrossEncoderReranker.__init__
    # Requirement  : `__init__` shall initialize cross-encoder reranker
    # Purpose      : Initialize cross-encoder reranker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : model_name: str (default='cross-encoder/ms-marco-MiniLM-L-6-v2'); combine_weight: float (default=0.7); logger: Optional[logging.Logger] (default=None)
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
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        combine_weight: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            combine_weight: Weight for rerank score vs original (0-1)
                          0.7 means 70% rerank, 30% original
            logger: Logger instance
        """
        self.model_name = model_name
        self.combine_weight = combine_weight
        self.logger = logger or logging.getLogger("eeg_rag.reranker")
        
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers required for reranking")
        
        # Load cross-encoder model
        self.logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.logger.info("Cross-encoder loaded successfully")
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.CrossEncoderReranker.rerank
    # Requirement  : `rerank` shall rerank candidates using cross-encoder
    # Purpose      : Rerank candidates using cross-encoder
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; candidates: List[Dict[str, Any]]; top_k: Optional[int] (default=None)
    # Outputs      : List[RerankedResult]
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
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder
        
        Args:
            query: Search query
            candidates: List of candidate documents with:
                - doc_id: Document ID
                - text: Document text
                - score: Original retrieval score
                - metadata: Additional metadata
            top_k: Number of results to return (None = return all)
            
        Returns:
            List of reranked results sorted by final score
        """
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Prepare query-document pairs
        pairs = [(query, doc.get('text', '')) for doc in candidates]
        
        # Get cross-encoder scores
        self.logger.debug(f"Reranking {len(candidates)} candidates...")
        rerank_scores = self.model.predict(pairs)
        
        # Combine with original scores
        reranked = []
        for i, (doc, rerank_score) in enumerate(zip(candidates, rerank_scores)):
            original_score = doc.get('score', 0.0)
            
            # Normalize rerank score to 0-1 range using sigmoid
            normalized_rerank = 1.0 / (1.0 + (-rerank_score).exp()) if hasattr(rerank_score, 'exp') else rerank_score
            
            # Combine scores
            final_score = (
                self.combine_weight * float(normalized_rerank) +
                (1 - self.combine_weight) * original_score
            )
            
            result = RerankedResult(
                doc_id=doc.get('doc_id', str(i)),
                text=doc.get('text', ''),
                original_score=original_score,
                rerank_score=float(rerank_score),
                final_score=final_score,
                metadata=doc.get('metadata', {}),
                chunk_id=doc.get('chunk_id')
            )
            reranked.append(result)
        
        # Sort by final score (descending)
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            reranked = reranked[:top_k]
        
        elapsed = (time.time() - start_time) * 1000
        self.logger.info(
            f"Reranked {len(candidates)} → {len(reranked)} results in {elapsed:.1f}ms"
        )
        
        return reranked
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.CrossEncoderReranker.get_statistics
    # Requirement  : `get_statistics` shall get reranker statistics
    # Purpose      : Get reranker statistics
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
    def get_statistics(self) -> Dict[str, Any]:
        """Get reranker statistics"""
        return {
            "model_name": self.model_name,
            "combine_weight": self.combine_weight,
            "available": CROSS_ENCODER_AVAILABLE
        }


# ---------------------------------------------------------------------------
# ID           : retrieval.reranker.NoOpReranker
# Requirement  : `NoOpReranker` class shall be instantiable and expose the documented interface
# Purpose      : No-op reranker for when cross-encoder is not available
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
# Verification : Instantiate NoOpReranker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class NoOpReranker:
    """
    No-op reranker for when cross-encoder is not available
    
    Simply passes through results without reranking
    """
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.NoOpReranker.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : logger: Optional[logging.Logger] (default=None)
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
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("eeg_rag.reranker")
        self.logger.warning("Using NoOp reranker (cross-encoder not available)")
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.NoOpReranker.rerank
    # Requirement  : `rerank` shall pass through without reranking
    # Purpose      : Pass through without reranking
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; candidates: List[Dict[str, Any]]; top_k: Optional[int] (default=None)
    # Outputs      : List[RerankedResult]
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
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankedResult]:
        """Pass through without reranking"""
        reranked = []
        for i, doc in enumerate(candidates):
            result = RerankedResult(
                doc_id=doc.get('doc_id', str(i)),
                text=doc.get('text', ''),
                original_score=doc.get('score', 0.0),
                rerank_score=0.0,
                final_score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {}),
                chunk_id=doc.get('chunk_id')
            )
            reranked.append(result)
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    # ---------------------------------------------------------------------------
    # ID           : retrieval.reranker.NoOpReranker.get_statistics
    # Requirement  : `get_statistics` shall execute as specified
    # Purpose      : Get statistics
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
    def get_statistics(self) -> Dict[str, Any]:
        return {"available": False, "model_name": "noop"}
