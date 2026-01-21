"""
Demo: Hybrid Retrieval System

This script demonstrates the three retrieval methods:
1. BM25 (sparse keyword search)
2. Dense (semantic embedding search)  
3. Hybrid (RRF fusion of BM25 + Dense)

Shows how hybrid retrieval combines the best of both worlds.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eeg_rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

logging.basicConfig(level=logging.WARNING)  # Quiet logs for demo


def print_results(title: str, results: list, show_scores: bool = True):
    """Pretty print search results."""
    print(f"\n{'=' * 80}")
    print(f"🔍 {title}")
    print('=' * 80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Doc {r.doc_id}")
        
        if hasattr(r, 'rrf_score'):
            # Hybrid result
            print(f"   RRF Score: {r.rrf_score:.4f}")
            print(f"   ├─ BM25: score={r.bm25_score:.3f}, rank={r.bm25_rank or 'N/A'}")
            print(f"   └─ Dense: score={r.dense_score:.3f}, rank={r.dense_rank or 'N/A'}")
        elif show_scores:
            print(f"   Score: {r.score:.3f}")
        
        # Show text preview
        text = r.text[:120].replace('\n', ' ')
        print(f"   {text}...")


def main():
    """Run retrieval comparison demo."""
    print("\n" + "="*80)
    print("🚀 HYBRID RETRIEVAL DEMO")
    print("="*80)
    print("\nComparing three retrieval methods on EEG research papers:")
    print("  1. BM25 - Sparse keyword search (term frequency)")
    print("  2. Dense - Semantic embedding search (meaning)")
    print("  3. Hybrid - RRF fusion (best of both)")
    
    # Initialize retrievers
    print("\n📦 Loading retrievers...")
    bm25 = BM25Retriever(cache_dir="data/bm25_cache")
    bm25._load_cache()  # Load from disk
    
    dense = DenseRetriever(
        url="http://localhost:6333",
        collection_name="eeg_papers"
    )
    
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
        rrf_k=60
    )
    
    print("✅ All retrievers loaded")
    
    # Test queries
    queries = [
        ("epilepsy seizure detection", "Exact keywords - BM25 should excel"),
        ("predicting epileptic events using neural networks", "Semantic query - Dense should excel"),
        ("deep learning for EEG classification", "Mixed query - Hybrid should excel")
    ]
    
    for query, description in queries:
        print(f"\n\n{'#' * 80}")
        print(f"# QUERY: '{query}'")
        print(f"# {description}")
        print('#' * 80)
        
        # BM25 results
        bm25_results = bm25.search(query, top_k=5)
        print_results("BM25 Results (Keyword Match)", bm25_results[:3])
        
        # Dense results
        dense_results = dense.search(query, top_k=5)
        print_results("Dense Results (Semantic Search)", dense_results[:3])
        
        # Hybrid results
        hybrid_results = hybrid.search(query, top_k=5, retrieve_k=50)
        print_results("Hybrid Results (RRF Fusion)", hybrid_results[:3])
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("📊 SUMMARY")
    print('=' * 80)
    print("""
Key Observations:

1. **BM25 (Sparse)**:
   - Excellent for exact keyword matches
   - Fast and deterministic
   - Struggles with synonyms and paraphrases
   
2. **Dense (Semantic)**:
   - Understands meaning and concepts
   - Handles synonyms and related terms
   - Can miss exact keyword matches
   
3. **Hybrid (RRF Fusion)**:
   - Combines strengths of both methods
   - More robust across query types
   - Typically achieves 10-20% better recall

**Recommendation**: Use Hybrid retrieval for production systems.
    """)
    
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running (docker ps)")
        print("  2. Papers are indexed (python scripts/index_papers_to_qdrant.py)")
        print("  3. BM25 index is built (python scripts/build_bm25_index.py)")
