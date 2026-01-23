#!/usr/bin/env python3
"""
Production paper ingestion script for EEG-RAG.
Fetches papers from multiple sources and stores them in the paper database.

Usage:
    # Run with default settings (fetch from all sources)
    python scripts/ingest_papers.py

    # Specify targets per source
    python scripts/ingest_papers.py --pubmed 50000 --scholar 30000 --openalex 50000

    # Quick test run
    python scripts/ingest_papers.py --test

    # Resume interrupted ingestion
    python scripts/ingest_papers.py --resume
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.db.paper_store import PaperStore, Paper, get_paper_store
from eeg_rag.ingestion.bulk_ingestion import BulkIngestionManager, BulkIngestionConfig
from eeg_rag.ingestion.pipeline import UnifiedDocument

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def unified_doc_to_paper(doc: UnifiedDocument) -> Paper:
    """Convert UnifiedDocument to Paper for storage."""
    return Paper(
        paper_id=doc.doc_id,
        title=doc.title,
        abstract=doc.abstract,
        authors=doc.authors,
        year=doc.publication_year,
        source=doc.source,
        pmid=doc.pmid,
        doi=doc.doi,
        arxiv_id=doc.arxiv_id,
        s2_id=doc.semantic_scholar_id,
        openalex_id=doc.openalex_id,
        url=doc.pdf_url,
        journal=doc.journal,
        keywords=doc.keywords,
        mesh_terms=doc.mesh_terms,
        citation_count=doc.citation_count,
        full_text=doc.full_text
    )


class DatabaseIngestionManager:
    """Extended ingestion manager that stores papers in the database."""
    
    def __init__(
        self,
        config: BulkIngestionConfig = None,
        paper_store: PaperStore = None
    ):
        self.config = config or BulkIngestionConfig()
        self.paper_store = paper_store or get_paper_store()
        self.bulk_manager = BulkIngestionManager(config=self.config)
        
    async def ingest_to_database(
        self,
        pubmed_target: int = 50000,
        scholar_target: int = 30000,
        arxiv_target: int = 10000,
        openalex_target: int = 50000,
        resume: bool = False
    ) -> dict:
        """
        Ingest papers from all sources directly into the database.
        
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = datetime.now()
        
        # Update config
        self.config.pubmed_target = pubmed_target
        self.config.scholar_target = scholar_target
        self.config.arxiv_target = arxiv_target
        self.config.openalex_target = openalex_target
        
        logger.info("=" * 60)
        logger.info("EEG-RAG Production Paper Ingestion")
        logger.info("=" * 60)
        logger.info(f"Targets: PubMed={pubmed_target:,}, Scholar={scholar_target:,}, "
                   f"arXiv={arxiv_target:,}, OpenAlex={openalex_target:,}")
        logger.info(f"Total target: {pubmed_target + scholar_target + arxiv_target + openalex_target:,} papers")
        logger.info(f"Database: {self.paper_store.db_path}")
        logger.info("=" * 60)
        
        stats = {
            'started_at': start_time.isoformat(),
            'pubmed': {'added': 0, 'updated': 0, 'skipped': 0},
            'scholar': {'added': 0, 'updated': 0, 'skipped': 0},
            'arxiv': {'added': 0, 'updated': 0, 'skipped': 0},
            'openalex': {'added': 0, 'updated': 0, 'skipped': 0},
        }
        
        try:
            # Run bulk ingestion - this will collect papers to JSONL
            await self.bulk_manager.run_bulk_ingestion(resume=resume)
            
            # Now load the collected papers into the database
            output_file = Path(self.config.output_dir) / "bulk_corpus.jsonl"
            
            if output_file.exists():
                logger.info(f"Loading papers from {output_file} into database...")
                stats['file_load'] = await self._load_jsonl_to_db(output_file)
            else:
                logger.warning(f"No bulk corpus file found at {output_file}")
                
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            stats['error'] = str(e)
        
        # Get final database stats
        db_stats = self.paper_store.get_statistics()
        stats['final_db_stats'] = db_stats
        stats['completed_at'] = datetime.now().isoformat()
        stats['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("Ingestion Complete!")
        logger.info(f"Total papers in database: {db_stats['total_papers']:,}")
        logger.info(f"Duration: {stats['duration_seconds'] / 3600:.1f} hours")
        logger.info("=" * 60)
        
        return stats
    
    async def _load_jsonl_to_db(self, jsonl_path: Path) -> dict:
        """Load papers from JSONL file into database."""
        import json
        
        papers_batch = []
        batch_size = 1000
        total_added = 0
        total_updated = 0
        total_skipped = 0
        
        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = UnifiedDocument(**data)
                    paper = unified_doc_to_paper(doc)
                    papers_batch.append(paper)
                    
                    if len(papers_batch) >= batch_size:
                        added, updated, skipped = self.paper_store.add_papers_batch(
                            papers_batch, update_if_exists=True
                        )
                        total_added += added
                        total_updated += updated
                        total_skipped += skipped
                        papers_batch = []
                        
                        if line_num % 10000 == 0:
                            logger.info(f"Loaded {line_num:,} papers: "
                                       f"{total_added:,} added, {total_updated:,} updated")
                            
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {line_num}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        # Final batch
        if papers_batch:
            added, updated, skipped = self.paper_store.add_papers_batch(
                papers_batch, update_if_exists=True
            )
            total_added += added
            total_updated += updated
            total_skipped += skipped
        
        return {
            'added': total_added,
            'updated': total_updated,
            'skipped': total_skipped
        }


async def quick_ingest_openalex(target: int = 100000) -> dict:
    """
    Quick ingestion using OpenAlex (no API key needed, very generous limits).
    This is the fastest way to populate the database.
    """
    from eeg_rag.ingestion.openalex_client import OpenAlexClient, OpenAlexWork
    
    paper_store = get_paper_store()
    client = OpenAlexClient(email="eeg-rag@research.edu")
    
    logger.info(f"Quick ingestion from OpenAlex (target: {target:,} papers)")
    
    papers_collected = []
    
    try:
        # Use the built-in EEG corpus collection method
        async for work in client.collect_eeg_corpus(from_year=2010, max_results=target):
            # Extract publication year
            pub_year = None
            if work.publication_date:
                try:
                    pub_year = work.publication_date.year
                except:
                    pass
            
            # Extract author names
            author_names = []
            for a in (work.authors or []):
                name = a.get('name', '')
                if name:
                    author_names.append(name)
            
            # Extract concept names for keywords
            concept_names = []
            for c in (work.concepts or []):
                if isinstance(c, dict):
                    concept_names.append(c.get('name', ''))
                elif isinstance(c, str):
                    concept_names.append(c)
            
            paper = Paper(
                paper_id=work.openalex_id or f"openalex_{len(papers_collected)}",
                title=work.title or "",
                abstract=work.abstract or "",
                authors=author_names,
                year=pub_year,
                source="openalex",
                doi=work.doi,
                pmid=work.pmid,
                openalex_id=work.openalex_id,
                url=work.pdf_url,
                journal=work.journal or "",
                citation_count=work.citation_count or 0,
                keywords=concept_names[:10]  # Limit to 10 keywords
            )
            papers_collected.append(paper)
            
            if len(papers_collected) % 1000 == 0:
                logger.info(f"Collected {len(papers_collected):,} papers...")
            
            if len(papers_collected) >= target:
                break
                
    except Exception as e:
        logger.error(f"Error during OpenAlex ingestion: {e}")
    
    # Batch insert
    if papers_collected:
        logger.info(f"Inserting {len(papers_collected):,} papers into database...")
        added, updated, skipped = paper_store.add_papers_batch(
            papers_collected, update_if_exists=True
        )
        logger.info(f"Ingested {added:,} papers, updated {updated:,}, skipped {skipped:,}")
        return {'added': added, 'updated': updated, 'skipped': skipped}
    
    return {'added': 0, 'updated': 0, 'skipped': 0}


def main():
    parser = argparse.ArgumentParser(description="EEG-RAG Paper Ingestion")
    parser.add_argument("--pubmed", type=int, default=50000, help="Target PubMed papers")
    parser.add_argument("--scholar", type=int, default=30000, help="Target Semantic Scholar papers")
    parser.add_argument("--arxiv", type=int, default=10000, help="Target arXiv papers")
    parser.add_argument("--openalex", type=int, default=50000, help="Target OpenAlex papers")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted ingestion")
    parser.add_argument("--test", action="store_true", help="Quick test run with 1000 papers")
    parser.add_argument("--quick", action="store_true", help="Quick OpenAlex-only ingestion")
    parser.add_argument("--status", action="store_true", help="Show database status only")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Just show status
    if args.status:
        paper_store = get_paper_store()
        stats = paper_store.get_statistics()
        print("\n" + "=" * 60)
        print("EEG-RAG Paper Database Status")
        print("=" * 60)
        print(f"Total papers: {stats['total_papers']:,}")
        print(f"Database size: {stats['db_size_mb']:.1f} MB")
        print(f"\nBy source:")
        for source, count in stats.get('by_source', {}).items():
            print(f"  - {source}: {count:,}")
        print(f"\nYear range: {stats['year_range']['min']} - {stats['year_range']['max']}")
        print(f"PMID coverage: {stats['pmid_coverage']:.1f}%")
        print(f"DOI coverage: {stats['doi_coverage']:.1f}%")
        print("=" * 60 + "\n")
        return
    
    # Quick OpenAlex ingestion
    if args.quick:
        target = 1000 if args.test else 100000
        asyncio.run(quick_ingest_openalex(target))
        return
    
    # Test mode
    if args.test:
        args.pubmed = 100
        args.scholar = 100
        args.arxiv = 50
        args.openalex = 200
        logger.info("Running in test mode with reduced targets")
    
    # Full ingestion
    config = BulkIngestionConfig(
        pubmed_target=args.pubmed,
        scholar_target=args.scholar,
        arxiv_target=args.arxiv,
        openalex_target=args.openalex
    )
    
    manager = DatabaseIngestionManager(config=config)
    asyncio.run(manager.ingest_to_database(
        pubmed_target=args.pubmed,
        scholar_target=args.scholar,
        arxiv_target=args.arxiv,
        openalex_target=args.openalex,
        resume=args.resume
    ))


if __name__ == "__main__":
    main()
