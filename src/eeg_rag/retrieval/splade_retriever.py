"""
SPLADE (Sparse Lexical and Expansion) Retriever

SPLADE is a learned sparse retrieval model that combines the efficiency of 
sparse retrieval (like BM25) with the quality of neural models. It learns
to expand queries and documents with important semantic terms.

Benefits over BM25:
- Learns term importance weights
- Automatic query/document expansion
- Better than BM25, faster than dense retrieval
- Expected: +10-15% recall over BM25 alone

Reference: https://arxiv.org/abs/2107.05720
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
from collections import defaultdict

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import torch
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False
    logging.warning("transformers/torch not available for SPLADE")


@dataclass
class SpladeResult:
    """Result from SPLADE search"""
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    sparse_vector: Optional[Dict[int, float]] = None  # term_id -> weight


class SpladeRetriever:
    """
    SPLADE retriever using learned sparse representations
    
    Uses a BERT-like model fine-tuned to predict term importance weights.
    Documents and queries are encoded as sparse vectors (most weights are 0).
    
    Example:
        >>> retriever = SpladeRetriever()
        >>> retriever.index_documents(docs)
        >>> results = retriever.search("epilepsy seizure detection", top_k=10)
    """
    
    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SPLADE retriever
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache index
            device: Device to run model on (cpu/cuda)
            logger: Logger instance
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.logger = logger or logging.getLogger("eeg_rag.splade")
        
        if not SPLADE_AVAILABLE:
            raise ImportError("transformers and torch required for SPLADE")
        
        # Load model and tokenizer
        self.logger.info(f"Loading SPLADE model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(f"SPLADE model loaded on {self.device}")
        
        # Document storage
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.doc_vectors: Dict[str, Dict[int, float]] = {}  # doc_id -> sparse vector
        
        # Load cache if exists
        if self.cache_dir and (self.cache_dir / "splade_index.pkl").exists():
            self.load_cache()
    
    def _encode(self, text: str, max_length: int = 512) -> Dict[int, float]:
        """
        Encode text to sparse vector using SPLADE
        
        Args:
            text: Input text
            max_length: Maximum token length
            
        Returns:
            Sparse vector as dict (term_id -> weight)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply log(1 + ReLU) to get importance scores
            # This is the SPLADE formulation
            scores = torch.log1p(torch.relu(logits))
            
            # Max pooling over tokens for each vocabulary term
            # Shape: [batch_size, vocab_size]
            vec = torch.max(scores, dim=1).values.squeeze()
        
        # Convert to sparse representation (only non-zero weights)
        sparse_vec = {}
        for idx, weight in enumerate(vec.cpu().numpy()):
            if weight > 0:
                sparse_vec[idx] = float(weight)
        
        return sparse_vec
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for search
        
        Args:
            documents: List of dicts with keys:
                - id: Document ID
                - text: Document text
                - metadata: Additional metadata (optional)
        """
        self.logger.info(f"Indexing {len(documents)} documents with SPLADE...")
        
        for i, doc in enumerate(documents):
            doc_id = doc["id"]
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            # Encode document
            sparse_vec = self._encode(text)
            
            # Store
            self.documents[doc_id] = {
                "text": text,
                "metadata": metadata
            }
            self.doc_vectors[doc_id] = sparse_vec
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"  Indexed {i + 1}/{len(documents)} documents")
        
        self.logger.info(f"✅ Indexed {len(documents)} documents")
        
        # Save cache
        if self.cache_dir:
            self.save_cache()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[SpladeResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum score threshold
            
        Returns:
            List of SpladeResult objects sorted by score
        """
        # Encode query
        query_vec = self._encode(query)
        
        # Compute scores for all documents (sparse dot product)
        scores = []
        for doc_id, doc_vec in self.doc_vectors.items():
            # Sparse dot product
            score = 0.0
            for term_id, query_weight in query_vec.items():
                if term_id in doc_vec:
                    score += query_weight * doc_vec[term_id]
            
            if score >= min_score:
                scores.append((doc_id, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_scores = scores[:top_k]
        
        # Create results
        results = []
        for doc_id, score in top_scores:
            doc_data = self.documents[doc_id]
            results.append(SpladeResult(
                doc_id=doc_id,
                text=doc_data["text"],
                score=score,
                metadata=doc_data["metadata"],
                sparse_vector=self.doc_vectors[doc_id]
            ))
        
        return results
    
    def save_cache(self) -> None:
        """Save index to disk"""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "documents": self.documents,
            "doc_vectors": self.doc_vectors,
            "model_name": self.model_name
        }
        
        cache_path = self.cache_dir / "splade_index.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        
        self.logger.info(f"Saved SPLADE index to {cache_path}")
    
    def load_cache(self) -> None:
        """Load index from disk"""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / "splade_index.pkl"
        if not cache_path.exists():
            return
        
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        
        self.documents = cache_data["documents"]
        self.doc_vectors = cache_data["doc_vectors"]
        
        self.logger.info(f"Loaded SPLADE index from {cache_path}")
        self.logger.info(f"  Documents: {len(self.documents)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        # Calculate average sparsity
        total_nonzero = sum(len(vec) for vec in self.doc_vectors.values())
        avg_nonzero = total_nonzero / len(self.doc_vectors) if self.doc_vectors else 0
        vocab_size = len(self.tokenizer)
        sparsity = 1 - (avg_nonzero / vocab_size) if vocab_size > 0 else 0
        
        return {
            "model_name": self.model_name,
            "num_documents": len(self.documents),
            "avg_nonzero_terms": avg_nonzero,
            "sparsity": sparsity,
            "vocab_size": vocab_size,
            "device": self.device
        }


if __name__ == "__main__":
    # Test SPLADE retriever
    import sys
    sys.path.insert(0, "/home/kevin/Projects/eeg-rag")
    
    logging.basicConfig(level=logging.INFO)
    
    # Sample documents
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
    
    # Initialize and index
    retriever = SpladeRetriever(
        cache_dir="data/splade_cache_test",
        device="cpu"
    )
    
    retriever.index_documents(docs)
    
    # Search
    results = retriever.search("epilepsy seizure detection", top_k=3)
    
    print("\n🔍 SPLADE Search Results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Doc {r.doc_id}:")
        print(f"   Score: {r.score:.4f}")
        print(f"   Text: {r.text[:80]}...")
        print(f"   Non-zero terms: {len(r.sparse_vector) if r.sparse_vector else 0}")
    
    # Statistics
    stats = retriever.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Avg non-zero terms: {stats['avg_nonzero_terms']:.1f}")
    print(f"   Sparsity: {stats['sparsity']:.2%}")
