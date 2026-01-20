#!/usr/bin/env python3
"""
Main script to run the complete EEG data ingestion pipeline.

Usage:
    # Full ingestion from all sources
    python scripts/run_ingestion.py
    
    # Specific sources only
    python scripts/run_ingestion.py --sources pubmed arxiv
    
    # API keys are OPTIONAL - all sources work without them
    # Keys only provide faster rate limits
    python scripts/run_ingestion.py
"""

import asyncio
import argparse
import logging
from pathlib import Path
import os
import sys

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.ingestion.pipeline import IngestionPipeline
from eeg_rag.ingestion.chunker import EEGDocumentChunker


def setup_logging(log_file: str = "ingestion.log") -> logging.Logger:
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


async def main():
    """Main entry point for ingestion pipeline."""
    logger = setup_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="EEG-RAG Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ingestion from all sources
  python scripts/run_ingestion.py
  
  # PubMed and arXiv only
  python scripts/run_ingestion.py --sources pubmed arxiv
  
  # Last 5 years only
  python scripts/run_ingestion.py --years-back 5
  
Environment Variables (ALL OPTIONAL - system works without them):
  PUBMED_API_KEY          - Optional: NCBI API key for 10 req/sec (vs 3 req/sec)
  SEMANTIC_SCHOLAR_API_KEY - Optional: Semantic Scholar key for higher limits
  CONTACT_EMAIL           - Optional: Your email for polite API identification
        """
    )
    
    parser.add_argument(
        "--sources", 
        nargs="+", 
        default=["pubmed", "semantic_scholar", "arxiv", "openalex"],
        choices=["pubmed", "semantic_scholar", "arxiv", "openalex"],
        help="Data sources to use (default: all)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/raw",
        help="Output directory for raw data (default: data/raw)"
    )
    parser.add_argument(
        "--chunks-dir", 
        default="data/chunks",
        help="Output directory for chunked data (default: data/chunks)"
    )
    parser.add_argument(
        "--years-back", 
        type=int, 
        default=10,
        help="Years of literature to collect (default: 10)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=512,
        help="Target chunk size in tokens (default: 512)"
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip the chunking step after ingestion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Get API keys from environment
    pubmed_api_key = os.environ.get("PUBMED_API_KEY")
    s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    email = os.environ.get("CONTACT_EMAIL", "eeg-rag@example.com")
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("EEG-RAG Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Years back: {args.years_back}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"PubMed API key: {'set (10 req/s)' if pubmed_api_key else 'not set (3 req/s - OK)'}")
    logger.info(f"Semantic Scholar API key: {'set (faster)' if s2_api_key else 'not set (works fine)'}")
    logger.info(f"Contact email: {email}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("DRY RUN - No actual ingestion will be performed")
        return
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        output_dir=args.output_dir,
        pubmed_api_key=pubmed_api_key,
        semantic_scholar_api_key=s2_api_key,
        email=email
    )
    
    # Run ingestion
    logger.info("Starting data ingestion...")
    stats = await pipeline.run_full_ingestion(sources=args.sources)
    
    logger.info("=" * 60)
    logger.info("Ingestion Statistics:")
    for source, count in stats.items():
        if source != "output_file":
            logger.info(f"  {source}: {count}")
    logger.info(f"Output file: {stats.get('output_file')}")
    logger.info("=" * 60)
    
    # Chunk documents
    if not args.skip_chunking and stats.get("output_file"):
        logger.info("Starting document chunking...")
        
        import json
        
        chunker = EEGDocumentChunker(chunk_size=args.chunk_size)
        
        output_file = Path(stats["output_file"])
        chunks_dir = Path(args.chunks_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        chunks_file = chunks_dir / f"chunks_{output_file.stem}.jsonl"
        
        chunk_count = 0
        doc_count = 0
        
        with open(output_file) as infile, open(chunks_file, "w") as outfile:
            for line in infile:
                doc = json.loads(line)
                doc_count += 1
                
                for chunk in chunker.chunk_document(doc):
                    outfile.write(json.dumps({
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "chunk_type": chunk.chunk_type.value,
                        "chunk_index": chunk.chunk_index,
                        "pmid": chunk.pmid,
                        "doi": chunk.doi,
                        "title": chunk.title,
                        "authors": chunk.authors,
                        "publication_year": chunk.publication_year,
                        "journal": chunk.journal,
                        "token_count": chunk.token_count
                    }) + "\n")
                    chunk_count += 1
                
                if doc_count % 1000 == 0:
                    logger.info(f"Chunked {doc_count} documents ({chunk_count} chunks)")
        
        logger.info("=" * 60)
        logger.info("Chunking Statistics:")
        logger.info(f"  Documents processed: {doc_count}")
        logger.info(f"  Chunks created: {chunk_count}")
        logger.info(f"  Average chunks per doc: {chunk_count / max(doc_count, 1):.1f}")
        logger.info(f"  Output file: {chunks_file}")
        logger.info("=" * 60)
    
    logger.info("Pipeline complete!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total documents: {stats.get('total', 0)}")
    print(f"Output: {stats.get('output_file')}")
    if not args.skip_chunking:
        print(f"Chunks: {chunks_file}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
