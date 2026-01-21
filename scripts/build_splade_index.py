"""
Build SPLADE index from corpus

This script indexes documents using the SPLADE learned sparse retrieval model.
SPLADE provides better quality than BM25 while maintaining efficiency.
"""

import logging
import json
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_rag.retrieval.splade_retriever import SpladeRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path) -> list:
    """Load documents from corpus"""
    documents = []
    
    logger.info(f"Loading corpus from {corpus_path}")
    
    # Load from JSON files
    if corpus_path.is_dir():
        for json_file in corpus_path.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
    elif corpus_path.suffix == ".json":
        with open(corpus_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            else:
                documents = [data]
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def main():
    """Build SPLADE index"""
    
    print("\n" + "="*80)
    print("Building SPLADE Index")
    print("="*80)
    
    # Configuration
    corpus_path = Path("data/processed/corpus.json")
    cache_dir = Path("data/splade_cache")
    device = "cpu"  # Change to "cuda" if GPU available
    
    # Check if corpus exists
    if not corpus_path.exists():
        # Try alternative paths
        alternative_paths = [
            Path("data/test_corpus"),
            Path("data/demo_corpus"),
            Path("data/corpus_test")
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                corpus_path = alt_path
                break
        else:
            logger.error(f"Corpus not found at {corpus_path}")
            logger.info("Please provide corpus at one of:")
            for p in [corpus_path] + alternative_paths:
                logger.info(f"  - {p}")
            return
    
    # Load documents
    documents = load_corpus(corpus_path)
    
    if not documents:
        logger.error("No documents found in corpus")
        return
    
    # Initialize SPLADE retriever
    print(f"\n[1/2] Initializing SPLADE retriever...")
    print(f"  Model: naver/splade-cocondenser-ensembledistil")
    print(f"  Device: {device}")
    print(f"  Cache: {cache_dir}")
    
    try:
        retriever = SpladeRetriever(
            cache_dir=str(cache_dir),
            device=device
        )
    except ImportError as e:
        logger.error(f"Failed to initialize SPLADE: {e}")
        logger.info("Install dependencies: pip install transformers torch")
        return
    
    # Index documents
    print(f"\n[2/2] Indexing {len(documents)} documents...")
    retriever.index_documents(documents)
    
    # Display statistics
    print(f"\n" + "="*80)
    print("Indexing Complete")
    print("="*80)
    
    stats = retriever.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Documents indexed: {stats['num_documents']}")
    print(f"  Avg non-zero terms: {stats['avg_nonzero_terms']:.1f}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Vocab size: {stats['vocab_size']:,}")
    print(f"  Cache location: {cache_dir}")
    
    # Test search
    print(f"\n🔍 Testing search...")
    test_queries = [
        "epilepsy seizure detection",
        "sleep stage classification",
        "motor imagery BCI"
    ]
    
    for query in test_queries:
        results = retriever.search(query, top_k=3)
        print(f"\n  Query: '{query}'")
        print(f"  Results: {len(results)}")
        if results:
            print(f"  Top score: {results[0].score:.4f}")
    
    print(f"\n✅ SPLADE index built successfully!")
    print(f"\n💡 Next steps:")
    print(f"  • Compare with BM25: python3 examples/compare_retrievers.py")
    print(f"  • Use in hybrid retrieval: Update config to use SPLADE")


if __name__ == "__main__":
    main()
