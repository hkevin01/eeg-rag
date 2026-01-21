"""
Evaluate Reranking Improvements

This script measures the actual improvement from cross-encoder reranking
using standard IR metrics (Recall@K, MRR, NDCG, MAP).

It compares:
1. Baseline: Hybrid retrieval without reranking
2. Improved: Hybrid retrieval with cross-encoder reranking

Expected improvements: +5-10% on Recall@10, +10-15% on MRR
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Set, List, Any

# Setup paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_rag.retrieval.bm25_retriever import BM25Retriever
from eeg_rag.retrieval.dense_retriever import DenseRetriever
from eeg_rag.retrieval.hybrid_retriever import HybridRetriever
from eeg_rag.evaluation.retrieval_metrics import RetrievalEvaluator, RetrievalMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Ground truth data: EEG-specific queries with relevant documents
# In production, this would come from human annotations or click data
GROUND_TRUTH = {
    "seizure_detection_cnn": {
        "query": "Convolutional neural networks for epileptic seizure detection from EEG",
        "relevant_docs": {"15", "27", "42", "58", "73"}  # Example doc IDs
    },
    "sleep_staging_rnn": {
        "query": "Recurrent neural networks for automatic sleep stage classification",
        "relevant_docs": {"8", "19", "33", "51"}
    },
    "motor_imagery_bci": {
        "query": "Motor imagery classification for brain-computer interfaces",
        "relevant_docs": {"4", "22", "37", "49", "64"}
    },
    "erp_p300_detection": {
        "query": "P300 event-related potential detection and analysis methods",
        "relevant_docs": {"11", "28", "45", "62"}
    },
    "alpha_oscillations": {
        "query": "Alpha band oscillations during cognitive tasks in EEG",
        "relevant_docs": {"6", "18", "31", "47", "56"}
    }
}


def run_retrieval(
    hybrid_retriever: HybridRetriever,
    queries: Dict[str, Dict[str, Any]],
    top_k: int = 20
) -> Dict[str, List[str]]:
    """
    Run retrieval for all queries
    
    Args:
        hybrid_retriever: Configured hybrid retriever
        queries: Dict mapping query_id to query info
        top_k: Number of results to retrieve
        
    Returns:
        Dict mapping query_id to list of retrieved doc_ids
    """
    retrieved = {}
    
    for query_id, query_info in queries.items():
        query_text = query_info["query"]
        
        try:
            results = hybrid_retriever.search(
                query_text,
                top_k=top_k,
                retrieve_k=50  # Retrieve more candidates for better recall
            )
            
            # Extract doc IDs
            doc_ids = [r.doc_id for r in results]
            retrieved[query_id] = doc_ids
            
            logger.info(f"  {query_id}: Retrieved {len(doc_ids)} results")
            
        except Exception as e:
            logger.error(f"  Error retrieving {query_id}: {e}")
            retrieved[query_id] = []
    
    return retrieved


def main():
    """Run evaluation comparing reranking vs no reranking"""
    
    print("\n" + "="*80)
    print("Reranking Improvement Evaluation")
    print("="*80)
    print("\nThis script measures IR metrics for hybrid retrieval with/without reranking.")
    
    # Initialize retrievers
    print("\n[1/6] Initializing retrievers...")
    bm25_cache = "data/bm25_cache"
    bm25 = BM25Retriever(cache_dir=bm25_cache)
    print(f"  ✓ BM25 loaded from {bm25_cache}")
    
    try:
        dense = DenseRetriever(
            url="http://localhost:6333",
            collection_name="eeg_papers"
        )
        print("  ✓ Dense retriever connected to Qdrant")
    except Exception as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        print("\n❌ Error: Qdrant not available.")
        print("Please start Qdrant with: docker-compose up -d")
        return
    
    # Create baseline retriever (no reranking)
    print("\n[2/6] Creating baseline hybrid retriever (no reranking)...")
    hybrid_baseline = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranking=False
    )
    print("  ✓ Baseline retriever ready")
    
    # Create improved retriever (with reranking)
    print("\n[3/6] Creating improved hybrid retriever (with reranking)...")
    hybrid_reranked = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranking=True,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    print("  ✓ Improved retriever ready (reranking enabled)")
    
    # Extract ground truth in evaluator format
    ground_truth_relevance = {}
    for query_id, query_info in GROUND_TRUTH.items():
        ground_truth_relevance[query_id] = set(query_info["relevant_docs"])
    
    # Run baseline retrieval
    print(f"\n[4/6] Running baseline retrieval ({len(GROUND_TRUTH)} queries)...")
    start = time.time()
    baseline_results = run_retrieval(hybrid_baseline, GROUND_TRUTH, top_k=20)
    baseline_time = time.time() - start
    print(f"  ✓ Baseline completed in {baseline_time:.2f}s")
    
    # Run improved retrieval
    print(f"\n[5/6] Running improved retrieval (with reranking)...")
    start = time.time()
    reranked_results = run_retrieval(hybrid_reranked, GROUND_TRUTH, top_k=20)
    reranked_time = time.time() - start
    print(f"  ✓ Improved completed in {reranked_time:.2f}s")
    print(f"  Overhead: +{reranked_time - baseline_time:.2f}s ({(reranked_time/baseline_time - 1)*100:.1f}%)")
    
    # Evaluate metrics
    print(f"\n[6/6] Computing IR metrics...")
    evaluator = RetrievalEvaluator()
    
    baseline_metrics = evaluator.evaluate(
        ground_truth=ground_truth_relevance,
        retrieved=baseline_results,
        k_values=[1, 3, 5, 10, 20]
    )
    
    reranked_metrics = evaluator.evaluate(
        ground_truth=ground_truth_relevance,
        retrieved=reranked_results,
        k_values=[1, 3, 5, 10, 20]
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n📊 BASELINE (No Reranking):")
    print(baseline_metrics)
    
    print("\n📈 IMPROVED (With Reranking):")
    print(reranked_metrics)
    
    # Compute improvements
    print("\n" + "="*80)
    print("IMPROVEMENTS")
    print("="*80)
    
    improvements = evaluator.compare(baseline_metrics, reranked_metrics)
    
    print(f"\n✨ MRR:")
    print(f"   Baseline: {improvements['mrr']['baseline']:.4f}")
    print(f"   Improved: {improvements['mrr']['improved']:.4f}")
    print(f"   Change:   {improvements['mrr']['delta']:+.4f} ({improvements['mrr']['pct_change']:+.1f}%)")
    
    print(f"\n✨ MAP:")
    print(f"   Baseline: {improvements['map']['baseline']:.4f}")
    print(f"   Improved: {improvements['map']['improved']:.4f}")
    print(f"   Change:   {improvements['map']['delta']:+.4f} ({improvements['map']['pct_change']:+.1f}%)")
    
    for k in [1, 3, 5, 10, 20]:
        print(f"\n✨ Recall@{k}:")
        print(f"   Baseline: {improvements[f'recall@{k}']['baseline']:.4f}")
        print(f"   Improved: {improvements[f'recall@{k}']['improved']:.4f}")
        print(f"   Change:   {improvements[f'recall@{k}']['delta']:+.4f} ({improvements[f'recall@{k}']['pct_change']:+.1f}%)")
        
        print(f"\n✨ NDCG@{k}:")
        print(f"   Baseline: {improvements[f'ndcg@{k}']['baseline']:.4f}")
        print(f"   Improved: {improvements[f'ndcg@{k}']['improved']:.4f}")
        print(f"   Change:   {improvements[f'ndcg@{k}']['delta']:+.4f} ({improvements[f'ndcg@{k}']['pct_change']:+.1f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count improvements
    improved_metrics = []
    for metric, data in improvements.items():
        if data['delta'] > 0:
            improved_metrics.append((metric, data['pct_change']))
    
    improved_metrics.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ Metrics that improved ({len(improved_metrics)} / {len(improvements)}):")
    for metric, pct in improved_metrics[:5]:
        print(f"   • {metric}: +{pct:.1f}%")
    
    # Overall assessment
    avg_improvement = sum(imp[1] for imp in improved_metrics) / len(improved_metrics) if improved_metrics else 0
    print(f"\n📈 Average improvement: +{avg_improvement:.1f}%")
    
    # Cost-benefit analysis
    print(f"\n⏱️  Latency overhead: +{reranked_time - baseline_time:.2f}s (+{(reranked_time/baseline_time - 1)*100:.1f}%)")
    
    if avg_improvement > 5 and (reranked_time - baseline_time) < 1.0:
        print("\n✅ Recommendation: ENABLE reranking (good quality improvement with acceptable latency)")
    elif avg_improvement > 10:
        print("\n✅ Recommendation: ENABLE reranking (significant quality improvement)")
    elif avg_improvement > 0:
        print("\n⚠️  Recommendation: Consider enabling for high-precision use cases only")
    else:
        print("\n❌ Recommendation: DISABLE reranking (no significant improvement)")
    
    # Save results
    results_path = Path("data/evaluation/reranking_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_time": baseline_time,
            "reranked_time": reranked_time,
            "baseline_metrics": {
                "mrr": baseline_metrics.mrr,
                "map": baseline_metrics.map_score,
                "recall_at_k": baseline_metrics.recall_at_k,
                "ndcg_at_k": baseline_metrics.ndcg_at_k
            },
            "reranked_metrics": {
                "mrr": reranked_metrics.mrr,
                "map": reranked_metrics.map_score,
                "recall_at_k": reranked_metrics.recall_at_k,
                "ndcg_at_k": reranked_metrics.ndcg_at_k
            },
            "improvements": improvements
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
