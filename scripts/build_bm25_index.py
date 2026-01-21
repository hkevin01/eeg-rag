"""Build BM25 index from Qdrant collection."""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from eeg_rag.retrieval import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build BM25 index from Qdrant documents."""
    logger.info("Building BM25 index from Qdrant collection...")
    
    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "eeg_papers"
    
    # Get all documents from Qdrant
    logger.info(f"Fetching documents from {collection_name}...")
    
    # Scroll through all points
    offset = None
    documents = []
    batch_size = 100
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        for point in points:
            doc_id = point.payload.get("doc_id", str(point.id))
            text = point.payload.get("text", "")
            metadata = point.payload.get("metadata", {})
            
            documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata
            })
        
        offset = next_offset
        if offset is None:
            break
    
    logger.info(f"Fetched {len(documents)} documents from Qdrant")
    
    # Filter out empty documents
    documents = [d for d in documents if d["text"] and len(d["text"].strip()) > 0]
    logger.info(f"After filtering empty docs: {len(documents)} documents")
    
    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Retriever(cache_dir="data/bm25_cache")
    bm25.index_documents(documents)
    
    logger.info("✅ BM25 index built successfully!")
    logger.info(f"  Documents indexed: {len(documents)}")
    logger.info(f"  Cache location: data/bm25_cache/bm25_index.pkl")
    
    # Test search
    logger.info("\nTesting BM25 search...")
    results = bm25.search("epilepsy seizure detection", top_k=3)
    
    logger.info("Top 3 Results:")
    for i, r in enumerate(results, 1):
        logger.info(f"  {i}. Doc {r.doc_id}: {r.score:.3f}")
        logger.info(f"     {r.text[:80]}...")


if __name__ == "__main__":
    main()
