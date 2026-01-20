#!/usr/bin/env python3
"""
Convenience script for bulk ingestion of 100K+ EEG research papers.

This script provides an easy way to run large-scale paper collection with:
- Automatic checkpointing (resume if interrupted)
- Progress tracking and ETA estimates
- Cross-source deduplication
- Adaptive rate limiting

Usage:
    # Default: 120K papers from all sources
    python scripts/run_bulk_ingestion.py
    
    # Resume interrupted ingestion  
    python scripts/run_bulk_ingestion.py --resume
    
    # Start fresh (ignore checkpoint)
    python scripts/run_bulk_ingestion.py --fresh
    
    # Custom targets
    python scripts/run_bulk_ingestion.py --pubmed 100000 --openalex 50000

Rate Limits (no API keys needed, but keys make it faster):
    - PubMed: 150 papers/min (500 with free key)
    - Semantic Scholar: 80 papers/min (300 with free key)
    - arXiv: 60 papers/min (no key available)
    - OpenAlex: 1000 papers/min (no key needed)

Get free API keys (optional):
    - PubMed: https://www.ncbi.nlm.nih.gov/account/settings/
    - Semantic Scholar: https://www.semanticscholar.org/product/api#api-key
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.ingestion import BulkIngestionManager, BulkIngestionConfig


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingestion of EEG research papers (100K+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        default=True,
        help="Resume from last checkpoint (default: True)"
    )
    parser.add_argument(
        "--fresh", 
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint"
    )
    parser.add_argument(
        "--pubmed", 
        type=int, 
        default=50000,
        help="Target number of PubMed papers (default: 50000)"
    )
    parser.add_argument(
        "--scholar", 
        type=int, 
        default=30000,
        help="Target number of Semantic Scholar papers (default: 30000)"
    )
    parser.add_argument(
        "--arxiv", 
        type=int, 
        default=10000,
        help="Target number of arXiv papers (default: 10000)"
    )
    parser.add_argument(
        "--openalex", 
        type=int, 
        default=30000,
        help="Target number of OpenAlex papers (default: 30000)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/bulk_ingestion",
        help="Output directory (default: data/bulk_ingestion)"
    )
    parser.add_argument(
        "--log-file",
        default="bulk_ingestion.log",
        help="Log file path (default: bulk_ingestion.log)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Calculate total
    total = args.pubmed + args.scholar + args.arxiv + args.openalex
    
    logger.info("=" * 70)
    logger.info("EEG-RAG BULK INGESTION")
    logger.info("=" * 70)
    logger.info(f"Target: {total:,} papers")
    logger.info(f"  - PubMed: {args.pubmed:,}")
    logger.info(f"  - Semantic Scholar: {args.scholar:,}")
    logger.info(f"  - arXiv: {args.arxiv:,}")
    logger.info(f"  - OpenAlex: {args.openalex:,}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Resume mode: {'Fresh start' if args.fresh else 'Resume from checkpoint'}")
    logger.info("=" * 70)
    
    # Create config
    config = BulkIngestionConfig(
        pubmed_target=args.pubmed,
        scholar_target=args.scholar,
        arxiv_target=args.arxiv,
        openalex_target=args.openalex,
        output_dir=args.output_dir
    )
    
    # Create manager and run
    manager = BulkIngestionManager(config=config)
    
    try:
        stats = asyncio.run(manager.run_bulk_ingestion(resume=not args.fresh))
        
        print("\n" + "=" * 70)
        print("BULK INGESTION COMPLETE!")
        print("=" * 70)
        print(f"Total papers collected: {stats['total_collected']:,}")
        print(f"Duplicates skipped: {stats['duplicates_skipped']:,}")
        print(f"Errors encountered: {stats['errors']}")
        print(f"Time elapsed: {stats['elapsed_time']}")
        print(f"\nOutput file: {stats['output_file']}")
        print(f"Checkpoint: {stats['checkpoint_file']}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        print("Your progress has been saved. Run with --resume to continue.")
        sys.exit(0)


if __name__ == "__main__":
    main()
