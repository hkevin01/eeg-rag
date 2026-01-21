"""
Demo: Adaptive Reranking

Shows how the system intelligently decides when to use reranking based on query
complexity. Simple queries skip reranking (save latency), complex queries use it
(improve quality).

Expected behavior:
- Simple queries: No reranking (fast, ~60ms)
- Complex queries: With reranking (precise, ~160ms)
- Average latency: Better than always-rerank, quality better than never-rerank
"""

import logging
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_rag.retrieval.bm25_retriever import BM25Retriever
from eeg_rag.retrieval.dense_retriever import DenseRetriever
from eeg_rag.retrieval.hybrid_retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*80)
    print("ADAPTIVE RERANKING DEMO")
    print("="*80)
    print("\nShows intelligent query-based reranking decisions.")
    
    # Initialize
    print("\n[1/3] Initializing retrievers...")
    bm25 = BM25Retriever(cache_dir="data/bm25_cache")
    
    try:
        dense = DenseRetriever(url="http://localhost:6333", collection_name="eeg_papers")
    except Exception as e:
        print(f"\n❌ Qdrant not available: {e}")
        return
    
    # Create adaptive hybrid retriever
    print("\n[2/3] Creating adaptive hybrid retriever...")
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        adaptive_reranking=True  # Enable adaptive reranking
    )
    print("  ✓ Adaptive reranking enabled")
    
    # Test queries of varying complexity
    queries = [
        ("EEG", "Simple - 1 word"),
        ("seizure detection", "Simple - 2 words, specific"),
        ("alpha oscillations",  "Medium - technical but short"),
        ("convolutional neural networks for epileptic seizure detection", "Complex - long, multi-concept"),
        ("What are the best methods for sleep stage classification?", "Complex - question, vague terms"),
        ("BCI not using P300 but visual stimuli", "Complex - negation, boolean"),
    ]
    
    print(f"\n[3/3] Testing {len(queries)} queries...")
    print("="*80)
    
    total_time = 0
    reranked_count = 0
    
    for i, (query, description) in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print(f"Description: {description}")
        print(f"{'#'*80}\n")
        
        start = time.time()
        results = hybrid.search(query, top_k=5, retrieve_k=20)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        print(f"\n⏱️  Search completed in {elapsed:.1f}ms")
        print(f"📊 Found {len(results)} results\n")
        
        if results:
            print("Top 3 results:")
            for j, r in enumerate(results[:3], 1):
                print(f"  {j}. Doc {r.doc_id}: {r.text[:80]}...")
                print(f"     RRF: {r.rrf_score:.4f}, BM25: {r.bm25_score:.2f}, Dense: {r.dense_score:.2f}")
        
        print(f"\n{'─'*80}")
    
    # Summary
    avg_time = total_time / len(queries)
    
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"Total queries: {len(queries)}")
    print(f"Average latency: {avg_time:.1f}ms")
    print(f"Total time: {total_time:.1f}ms")
    
    print("\n✨ Benefits of Adaptive Reranking:")
    print("  • Simple queries skip reranking → Lower average latency")
    print("  • Complex queries use reranking → Better precision where it matters")
    print("  • Best of both worlds: Speed + Quality")
    print("  • No configuration needed - fully automatic")
    
    print("\n✅ Demo complete!")
    print("\nCompare with:")
    print("  • Always rerank: Higher average latency (~160ms)")
    print("  • Never rerank: Lower quality on complex queries")
    print("  • Adaptive: Optimal balance")


if __name__ == "__main__":
    main()
