"""
Demo: Cross-Encoder Reranking for Improved Retrieval

This demo shows how cross-encoder reranking improves search results
by comparing hybrid retrieval with and without reranking.

Expected improvement: +5-10% recall on top-10 results
"""

import logging
import time
from pathlib import Path

# Setup paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_rag.retrieval.bm25_retriever import BM25Retriever
from eeg_rag.retrieval.dense_retriever import DenseRetriever
from eeg_rag.retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_results(results, title="Results"):
    """Pretty print search results"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Doc ID: {result.doc_id}")
        print(f"   RRF Score: {result.rrf_score:.4f}")
        print(f"   BM25 Score: {result.bm25_score:.3f} (Rank: {result.bm25_rank})")
        print(f"   Dense Score: {result.dense_score:.3f} (Rank: {result.dense_rank})")
        print(f"   Text: {result.text[:100]}...")


def main():
    """Run reranking demo"""
    
    print("\n" + "="*80)
    print("Cross-Encoder Reranking Demo")
    print("="*80)
    print("\nThis demo compares hybrid retrieval with and without reranking.")
    print("Expected: Reranking should improve relevance of top results.")
    
    # Initialize retrievers
    print("\n[1/5] Initializing BM25 retriever...")
    bm25_cache = "data/bm25_cache"
    bm25 = BM25Retriever(cache_dir=bm25_cache)
    print(f"  ✓ BM25 loaded from {bm25_cache}")
    
    print("\n[2/5] Initializing Dense retriever...")
    try:
        dense = DenseRetriever(
            url="http://localhost:6333",
            collection_name="eeg_papers"
        )
        print("  ✓ Connected to Qdrant")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        print("\n❌ Error: Qdrant not available.")
        print("Please start Qdrant with: docker-compose up -d")
        return
    
    # Create hybrid retriever WITHOUT reranking
    print("\n[3/5] Creating hybrid retriever (no reranking)...")
    hybrid_no_rerank = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranking=False
    )
    print("  ✓ Hybrid retriever ready (reranking disabled)")
    
    # Create hybrid retriever WITH reranking
    print("\n[4/5] Creating hybrid retriever (with reranking)...")
    hybrid_with_rerank = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranking=True,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("  ✓ Hybrid retriever ready (reranking enabled)")
    
    # Test queries
    queries = [
        "Convolutional neural networks for epileptic seizure detection",
        "Deep learning models for sleep stage classification from EEG",
        "Motor imagery BCI using deep learning techniques"
    ]
    
    print(f"\n[5/5] Running comparison on {len(queries)} queries...")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"{'#'*80}")
        
        # Search WITHOUT reranking
        print(f"\n🔍 Searching WITHOUT reranking...")
        start = time.time()
        results_no_rerank = hybrid_no_rerank.search(
            query,
            top_k=5,
            retrieve_k=20  # Retrieve more candidates for better reranking
        )
        time_no_rerank = (time.time() - start) * 1000
        print(f"   Search completed in {time_no_rerank:.1f}ms")
        
        # Search WITH reranking
        print(f"\n🔍 Searching WITH reranking...")
        start = time.time()
        results_with_rerank = hybrid_with_rerank.search(
            query,
            top_k=5,
            retrieve_k=20
        )
        time_with_rerank = (time.time() - start) * 1000
        print(f"   Search completed in {time_with_rerank:.1f}ms")
        
        # Compare results
        print(f"\n📊 Performance Comparison:")
        print(f"   No reranking:   {time_no_rerank:.1f}ms")
        print(f"   With reranking: {time_with_rerank:.1f}ms")
        print(f"   Overhead:       +{time_with_rerank - time_no_rerank:.1f}ms")
        
        # Show results
        print_results(results_no_rerank, "🔹 Results WITHOUT Reranking")
        print_results(results_with_rerank, "🔸 Results WITH Reranking")
        
        # Analyze differences
        print(f"\n📈 Analysis:")
        
        # Check if top result changed
        if results_no_rerank and results_with_rerank:
            if results_no_rerank[0].doc_id != results_with_rerank[0].doc_id:
                print(f"   ⚡ Top result changed!")
                print(f"      Before: {results_no_rerank[0].doc_id}")
                print(f"      After:  {results_with_rerank[0].doc_id}")
            else:
                print(f"   ✓ Top result unchanged: {results_no_rerank[0].doc_id}")
            
            # Check ranking changes
            ids_no_rerank = [r.doc_id for r in results_no_rerank]
            ids_with_rerank = [r.doc_id for r in results_with_rerank]
            
            if ids_no_rerank != ids_with_rerank:
                print(f"   📝 Ranking differences detected")
                
                # Find documents that moved up
                for idx, doc_id in enumerate(ids_with_rerank):
                    if doc_id in ids_no_rerank:
                        old_pos = ids_no_rerank.index(doc_id)
                        new_pos = idx
                        if new_pos < old_pos:
                            print(f"      ↑ Doc {doc_id} moved up: #{old_pos+1} → #{new_pos+1}")
            else:
                print(f"   ✓ Rankings identical")
        
        print(f"\n{'─'*80}")
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print("\n✅ Demo completed successfully!")
    print("\nKey Takeaways:")
    print("  • Cross-encoder reranking improves result relevance")
    print("  • Typical overhead: +50-150ms for 20 candidates")
    print("  • Best used when precision is more important than speed")
    print("  • Most effective with 20-50 candidates to rerank")
    print("\nNext steps:")
    print("  • Run evaluation with ground truth data")
    print("  • Measure Recall@K and MRR improvements")
    print("  • Tune combine_weight parameter (0.7 recommended)")


if __name__ == "__main__":
    main()
