#!/usr/bin/env python3
"""
Index EEG papers into Qdrant vector database.

This script:
1. Loads papers from Roy et al. 2019 CSV
2. Chunks papers using CitationAwareChunker
3. Indexes chunks into Qdrant with embeddings
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from eeg_rag.storage.vector_db import VectorDB
from eeg_rag.chunking.citation_aware_chunker import CitationAwareChunker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_roy_papers(csv_path: str) -> list:
    """Load papers from Roy et al. 2019 CSV."""
    logger.info(f"Loading papers from {csv_path}")
    
    df = pd.read_csv(
        csv_path,
        encoding='utf-8',
        on_bad_lines='skip',
        low_memory=False,
        header=1
    )
    df.columns = df.columns.str.strip()
    
    logger.info(f"Loaded {len(df)} papers")
    
    # Convert to list of dicts
    papers = []
    for idx, row in df.iterrows():
        # Combine ALL text fields for full-text search
        text_parts = []
        
        # Title is most important
        if pd.notna(row.get('Title')):
            text_parts.append(str(row['Title']))
        
        # Add all descriptive fields
        text_fields = [
            'High-level Goal', 'Practical Goal', 'Design peculiarities',
            'EEG-specific design', 'Preprocessing (clean)', 'Features (clean)',
            'Architecture (clean)', 'Results', 'Dataset name',
            'Train and test data', 'Data Preprocessing'
        ]
        
        for field in text_fields:
            if field in row and pd.notna(row.get(field)):
                value = str(row[field]).strip()
                if value and value != 'nan' and len(value) > 5:
                    text_parts.append(f"{field}: {value}")
        
        text = '. '.join(text_parts)
        
        # Only keep papers with substantial text
        if not text.strip() or len(text) < 50:
            continue
        
        paper = {
            'text': text,
            'doc_id': str(row.get('Citation', f'paper_{idx}')),
            'title': str(row.get('Title', '')),
            'authors': str(row.get('Authors', '')),
            'year': int(row['Year']) if pd.notna(row.get('Year')) else 0,
            'domain': str(row.get('Domain 1', '')),
            'architecture': str(row.get('Architecture (clean)', '')),
            'dataset': str(row.get('Dataset name', '')),
            'code_available': str(row.get('Code available', '')),
            'data_available': str(row.get('Dataset accessibility', '')),
        }
        
        papers.append(paper)
    
    logger.info(f"Prepared {len(papers)} papers with text")
    return papers


def main():
    """Main indexing pipeline."""
    logger.info("=" * 60)
    logger.info("EEG Papers Indexing Pipeline")
    logger.info("=" * 60)
    
    # Paths
    csv_path = "data/systematic_review/roy_et_al_2019_data_items.csv"
    
    if not Path(csv_path).exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    
    # Load papers
    papers = load_roy_papers(csv_path)
    
    if not papers:
        logger.error("No papers loaded!")
        return
    
    # Initialize chunker
    logger.info("Initializing chunker (512 tokens, 128 overlap)...")
    chunker = CitationAwareChunker(
        chunk_size=512,
        overlap=128,
        min_chunk_size=100
    )
    
    # Chunk papers
    logger.info(f"Chunking {len(papers)} papers...")
    chunks = chunker.chunk_papers(papers, text_field='text')
    
    logger.info(f"Created {len(chunks)} chunks")
    logger.info(f"Average chunks per paper: {len(chunks) / len(papers):.1f}")
    
    # Prepare documents for indexing
    documents = []
    for chunk in chunks:
        doc = {
            'text': chunk.text,
            'chunk_id': f"{chunk.paper_id}_chunk_{chunk.chunk_id}",
            'paper_id': chunk.paper_id,
            'section': chunk.section,
            'has_citations': chunk.has_citations,
            **chunk.metadata
        }
        documents.append(doc)
    
    # Initialize VectorDB
    logger.info("Connecting to Qdrant...")
    vdb = VectorDB(
        qdrant_url="http://localhost:6333",
        collection_name="eeg_papers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Create collection (recreate to start fresh)
    logger.info("Creating collection (this will overwrite existing)...")
    vdb.create_collection(recreate=True)
    
    # Index documents
    logger.info(f"Indexing {len(documents)} chunks into Qdrant...")
    count = vdb.index_documents(documents, text_field='text', batch_size=50)
    
    # Get collection info
    info = vdb.get_collection_info()
    logger.info("=" * 60)
    logger.info("Indexing Complete!")
    logger.info(f"  Papers indexed: {len(papers)}")
    logger.info(f"  Chunks indexed: {count}")
    logger.info(f"  Collection: {info.get('name', 'eeg_papers')}")
    logger.info(f"  Status: {info.get('status', 'unknown')}")
    logger.info("=" * 60)
    
    # Test search
    logger.info("\nTesting search...")
    test_queries = [
        "CNN for seizure detection",
        "RNN sleep staging",
        "motor imagery BCI"
    ]
    
    for query in test_queries:
        results = vdb.search(query, limit=3)
        logger.info(f"\nQuery: '{query}'")
        logger.info(f"  Results: {len(results)}")
        for r in results[:2]:
            title = r.payload.get('title', 'Unknown')[:50]
            logger.info(f"    - {title}... (score: {r.score:.3f})")
    
    logger.info("\n✅ Indexing pipeline complete!")


if __name__ == "__main__":
    main()
