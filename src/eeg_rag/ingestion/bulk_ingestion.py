"""
Bulk ingestion manager for large-scale paper collection (100K+ papers).

Features:
- Checkpointing: Resume from where you left off if interrupted
- Progress tracking: Real-time stats and ETA
- Adaptive rate limiting: Automatically uses faster rates with API keys
- Deduplication: Skips already-collected papers
- Error resilience: Continues on individual failures
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set

from .pipeline import IngestionPipeline, UnifiedDocument
from .pubmed_client import PubMedClient
from .scholar_client import SemanticScholarClient
from .arxiv_client import ArxivClient
from .openalex_client import OpenAlexClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionCheckpoint:
    """Tracks progress for resumable bulk ingestion."""
    
    # Run identification
    run_id: str
    started_at: str
    last_updated: str
    
    # Progress tracking per source
    pubmed_queries_completed: List[str] = field(default_factory=list)
    pubmed_ids_collected: Set[str] = field(default_factory=set)
    
    scholar_queries_completed: List[str] = field(default_factory=list)
    scholar_ids_collected: Set[str] = field(default_factory=set)
    
    arxiv_queries_completed: List[str] = field(default_factory=list)
    arxiv_ids_collected: Set[str] = field(default_factory=set)
    
    openalex_concepts_completed: List[str] = field(default_factory=list)
    openalex_ids_collected: Set[str] = field(default_factory=set)
    
    # Global deduplication
    all_dois: Set[str] = field(default_factory=set)
    all_pmids: Set[str] = field(default_factory=set)
    all_titles: Set[str] = field(default_factory=set)
    
    # Statistics
    total_collected: int = 0
    total_duplicates_skipped: int = 0
    total_errors: int = 0
    
    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        data = {
            'run_id': self.run_id,
            'started_at': self.started_at,
            'last_updated': datetime.now().isoformat(),
            'pubmed_queries_completed': self.pubmed_queries_completed,
            'pubmed_ids_collected': list(self.pubmed_ids_collected),
            'scholar_queries_completed': self.scholar_queries_completed,
            'scholar_ids_collected': list(self.scholar_ids_collected),
            'arxiv_queries_completed': self.arxiv_queries_completed,
            'arxiv_ids_collected': list(self.arxiv_ids_collected),
            'openalex_concepts_completed': self.openalex_concepts_completed,
            'openalex_ids_collected': list(self.openalex_ids_collected),
            'all_dois': list(self.all_dois),
            'all_pmids': list(self.all_pmids),
            'all_titles': list(self.all_titles),
            'total_collected': self.total_collected,
            'total_duplicates_skipped': self.total_duplicates_skipped,
            'total_errors': self.total_errors
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> Optional['IngestionCheckpoint']:
        """Load checkpoint from disk."""
        if not path.exists():
            return None
        
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            run_id=data['run_id'],
            started_at=data['started_at'],
            last_updated=data['last_updated'],
            pubmed_queries_completed=data.get('pubmed_queries_completed', []),
            pubmed_ids_collected=set(data.get('pubmed_ids_collected', [])),
            scholar_queries_completed=data.get('scholar_queries_completed', []),
            scholar_ids_collected=set(data.get('scholar_ids_collected', [])),
            arxiv_queries_completed=data.get('arxiv_queries_completed', []),
            arxiv_ids_collected=set(data.get('arxiv_ids_collected', [])),
            openalex_concepts_completed=data.get('openalex_concepts_completed', []),
            openalex_ids_collected=set(data.get('openalex_ids_collected', [])),
            all_dois=set(data.get('all_dois', [])),
            all_pmids=set(data.get('all_pmids', [])),
            all_titles=set(data.get('all_titles', [])),
            total_collected=data.get('total_collected', 0),
            total_duplicates_skipped=data.get('total_duplicates_skipped', 0),
            total_errors=data.get('total_errors', 0)
        )


@dataclass
class BulkIngestionConfig:
    """Configuration for bulk ingestion."""
    
    # Target counts per source
    pubmed_target: int = 50000
    scholar_target: int = 30000
    arxiv_target: int = 10000
    openalex_target: int = 30000
    
    # Rate limits (papers per minute, conservative defaults)
    pubmed_rate_no_key: int = 150  # 3 req/s * 50 papers/req
    pubmed_rate_with_key: int = 500  # 10 req/s * 50 papers/req
    scholar_rate_no_key: int = 80  # 100 req/5min
    scholar_rate_with_key: int = 300
    arxiv_rate: int = 60  # 3 sec delay between requests
    openalex_rate: int = 1000  # Very generous limits
    
    # Checkpointing
    checkpoint_interval: int = 100  # Save every N papers
    
    # Output
    output_dir: str = "data/bulk_ingestion"


class BulkIngestionManager:
    """
    Manages large-scale bulk ingestion of EEG research papers.
    
    Features:
    - Collects 100K+ papers from multiple sources
    - Checkpointing for resume capability
    - Real-time progress tracking with ETA
    - Automatic rate limit optimization
    - Cross-source deduplication
    
    Usage:
        manager = BulkIngestionManager()
        await manager.run_bulk_ingestion()
        
        # Resume interrupted ingestion
        manager = BulkIngestionManager()
        await manager.run_bulk_ingestion(resume=True)
    """
    
    def __init__(
        self,
        config: Optional[BulkIngestionConfig] = None,
        pubmed_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        email: str = "eeg-rag-bulk@research.edu"
    ):
        self.config = config or BulkIngestionConfig()
        self.pubmed_api_key = pubmed_api_key or os.environ.get("PUBMED_API_KEY")
        self.s2_api_key = semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.email = email
        
        # Initialize paths
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.output_file = self.output_dir / "bulk_corpus.jsonl"
        
        # Calculate effective rates
        self.pubmed_rate = (
            self.config.pubmed_rate_with_key if self.pubmed_api_key 
            else self.config.pubmed_rate_no_key
        )
        self.scholar_rate = (
            self.config.scholar_rate_with_key if self.s2_api_key
            else self.config.scholar_rate_no_key
        )
        
        # Progress tracking
        self.start_time: Optional[datetime] = None
        self.papers_per_minute: List[float] = []
    
    def _log_rate_limits(self) -> None:
        """Log effective rate limits based on API key availability."""
        logger.info("=" * 60)
        logger.info("Bulk Ingestion Rate Limits")
        logger.info("=" * 60)
        
        if self.pubmed_api_key:
            logger.info(f"✓ PubMed API key detected: {self.pubmed_rate} papers/min")
        else:
            logger.info(f"○ PubMed (no key): {self.pubmed_rate} papers/min")
            logger.info("  Get free key: https://www.ncbi.nlm.nih.gov/account/settings/")
        
        if self.s2_api_key:
            logger.info(f"✓ Semantic Scholar key detected: {self.scholar_rate} papers/min")
        else:
            logger.info(f"○ Semantic Scholar (no key): {self.scholar_rate} papers/min")
            logger.info("  Get free key: https://www.semanticscholar.org/product/api#api-key")
        
        logger.info(f"○ arXiv: {self.config.arxiv_rate} papers/min (no key needed)")
        logger.info(f"○ OpenAlex: {self.config.openalex_rate} papers/min (no key needed)")
        
        # Estimate total time
        total_target = (
            self.config.pubmed_target + 
            self.config.scholar_target + 
            self.config.arxiv_target + 
            self.config.openalex_target
        )
        
        # Rough estimate (papers collected in parallel)
        avg_rate = (self.pubmed_rate + self.scholar_rate + 
                   self.config.arxiv_rate + self.config.openalex_rate) / 4
        estimated_hours = total_target / avg_rate / 60
        
        logger.info("=" * 60)
        logger.info(f"Target: {total_target:,} papers")
        logger.info(f"Estimated time: {estimated_hours:.1f} hours")
        logger.info("=" * 60)
    
    def _estimate_eta(self, collected: int, target: int) -> str:
        """Calculate estimated time remaining."""
        if not self.papers_per_minute or collected == 0:
            return "calculating..."
        
        avg_rate = sum(self.papers_per_minute[-10:]) / len(self.papers_per_minute[-10:])
        if avg_rate <= 0:
            return "unknown"
        
        remaining = target - collected
        minutes_left = remaining / avg_rate
        
        if minutes_left < 60:
            return f"{int(minutes_left)} minutes"
        elif minutes_left < 1440:
            return f"{minutes_left / 60:.1f} hours"
        else:
            return f"{minutes_left / 1440:.1f} days"
    
    def _is_duplicate(self, doc: UnifiedDocument, checkpoint: IngestionCheckpoint) -> bool:
        """Check if document is a duplicate."""
        if doc.doi and doc.doi in checkpoint.all_dois:
            return True
        if doc.pmid and doc.pmid in checkpoint.all_pmids:
            return True
        
        # Normalize title for comparison
        normalized = doc.title.lower().strip()[:100] if doc.title else ""
        if normalized and normalized in checkpoint.all_titles:
            return True
        
        return False
    
    def _register_document(self, doc: UnifiedDocument, checkpoint: IngestionCheckpoint) -> None:
        """Register document for deduplication."""
        if doc.doi:
            checkpoint.all_dois.add(doc.doi)
        if doc.pmid:
            checkpoint.all_pmids.add(doc.pmid)
        if doc.title:
            normalized = doc.title.lower().strip()[:100]
            checkpoint.all_titles.add(normalized)
    
    async def _collect_pubmed(
        self, 
        checkpoint: IngestionCheckpoint,
        output_file
    ) -> int:
        """Collect papers from PubMed."""
        client = PubMedClient(api_key=self.pubmed_api_key, email=self.email)
        collected = 0
        
        # Extended EEG queries for bulk collection
        queries = [
            # Core EEG terms
            "electroencephalography[MeSH]",
            "electroencephalogram",
            "EEG recording",
            "EEG signal",
            "EEG analysis",
            "brain waves",
            "scalp EEG",
            "intracranial EEG",
            "iEEG",
            "ECoG electrocorticography",
            
            # Frequency bands
            "alpha rhythm[MeSH]",
            "beta rhythm[MeSH]", 
            "theta rhythm[MeSH]",
            "delta rhythm[MeSH]",
            "gamma oscillation",
            
            # Clinical applications
            "epilepsy EEG",
            "seizure detection EEG",
            "epileptiform discharge",
            "spike wave complex",
            "sleep EEG",
            "polysomnography[MeSH]",
            "sleep staging EEG",
            "anesthesia EEG monitoring",
            "coma EEG prognosis",
            "brain death EEG",
            
            # Event-related potentials
            "event-related potentials[MeSH]",
            "P300 ERP",
            "N400 component",
            "mismatch negativity",
            "contingent negative variation",
            "readiness potential",
            "error-related negativity",
            
            # Cognitive neuroscience
            "cognitive EEG",
            "working memory EEG",
            "attention EEG",
            "language processing EEG",
            "motor imagery EEG",
            "emotion EEG",
            
            # Brain-computer interface
            "brain-computer interface[MeSH]",
            "BCI EEG",
            "neurofeedback[MeSH]",
            "motor imagery BCI",
            "P300 speller",
            "SSVEP BCI",
            
            # Technical/methods
            "EEG artifact removal",
            "independent component analysis EEG",
            "EEG source localization",
            "EEG connectivity",
            "EEG coherence",
            "EEG microstate",
            "high-density EEG",
            "mobile EEG",
            "dry electrode EEG",
            
            # Disorders
            "ADHD EEG",
            "autism EEG",
            "depression EEG",
            "schizophrenia EEG",
            "Alzheimer EEG",
            "Parkinson EEG",
            "traumatic brain injury EEG",
            "stroke EEG",
            
            # Pediatric
            "neonatal EEG",
            "pediatric EEG",
            "developmental EEG"
        ]
        
        async with client:
            for query in queries:
                if query in checkpoint.pubmed_queries_completed:
                    logger.info(f"  Skipping completed query: {query}")
                    continue
                
                if collected >= self.config.pubmed_target:
                    break
                
                try:
                    logger.info(f"  PubMed query: {query}")
                    
                    # Search for IDs
                    ids = await client.search(query, max_results=2000)
                    
                    # Filter already collected
                    new_ids = [
                        id for id in ids 
                        if id not in checkpoint.pubmed_ids_collected
                    ]
                    
                    if not new_ids:
                        checkpoint.pubmed_queries_completed.append(query)
                        continue
                    
                    # Fetch articles in batches
                    for i in range(0, len(new_ids), 50):
                        batch_ids = new_ids[i:i+50]
                        articles = await client.fetch_articles(batch_ids)
                        
                        for article in articles:
                            # Convert to UnifiedDocument
                            doc = UnifiedDocument(
                                id=f"pubmed_{article.pmid}",
                                source="pubmed",
                                title=article.title,
                                abstract=article.abstract,
                                authors=article.authors,
                                publication_date=article.publication_date,
                                journal=article.journal,
                                doi=article.doi,
                                pmid=article.pmid,
                                url=f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/",
                                full_text=article.full_text,
                                mesh_terms=article.mesh_terms,
                                keywords=article.keywords,
                                eeg_entities=article.eeg_entities
                            )
                            
                            if self._is_duplicate(doc, checkpoint):
                                checkpoint.total_duplicates_skipped += 1
                                continue
                            
                            # Write to output
                            output_file.write(json.dumps(doc.__dict__) + "\n")
                            output_file.flush()
                            
                            self._register_document(doc, checkpoint)
                            checkpoint.pubmed_ids_collected.add(article.pmid)
                            checkpoint.total_collected += 1
                            collected += 1
                            
                            # Checkpoint periodically
                            if collected % self.config.checkpoint_interval == 0:
                                checkpoint.save(self.checkpoint_path)
                                logger.info(
                                    f"  Progress: {collected:,} papers "
                                    f"(+{self.config.checkpoint_interval} from PubMed)"
                                )
                        
                        # Rate limiting
                        await asyncio.sleep(0.1 if self.pubmed_api_key else 0.35)
                    
                    checkpoint.pubmed_queries_completed.append(query)
                    checkpoint.save(self.checkpoint_path)
                    
                except Exception as e:
                    logger.error(f"  Error in PubMed query '{query}': {e}")
                    checkpoint.total_errors += 1
        
        return collected
    
    async def _collect_semantic_scholar(
        self,
        checkpoint: IngestionCheckpoint,
        output_file
    ) -> int:
        """Collect papers from Semantic Scholar."""
        client = SemanticScholarClient(api_key=self.s2_api_key)
        collected = 0
        
        queries = [
            "electroencephalography",
            "EEG signal processing",
            "brain computer interface EEG",
            "epilepsy seizure detection",
            "sleep EEG analysis",
            "event related potentials",
            "EEG deep learning",
            "EEG classification",
            "motor imagery EEG",
            "cognitive neuroscience EEG",
            "EEG connectivity analysis",
            "EEG source localization",
            "neonatal EEG monitoring",
            "EEG artifact removal",
            "EEG microstate analysis",
            "alpha oscillations brain",
            "gamma oscillations cognition",
            "theta rhythm memory",
            "neurofeedback training",
            "P300 brain computer interface"
        ]
        
        async with client:
            for query in queries:
                if query in checkpoint.scholar_queries_completed:
                    continue
                
                if collected >= self.config.scholar_target:
                    break
                
                try:
                    logger.info(f"  Semantic Scholar query: {query}")
                    articles = await client.search_eeg_papers(query, limit=1000)
                    
                    for article in articles:
                        if article.paper_id in checkpoint.scholar_ids_collected:
                            continue
                        
                        doc = UnifiedDocument(
                            id=f"s2_{article.paper_id}",
                            source="semantic_scholar",
                            title=article.title,
                            abstract=article.abstract,
                            authors=article.authors,
                            publication_date=str(article.year) if article.year else None,
                            doi=article.doi,
                            url=article.url,
                            citation_count=article.citation_count
                        )
                        
                        if self._is_duplicate(doc, checkpoint):
                            checkpoint.total_duplicates_skipped += 1
                            continue
                        
                        output_file.write(json.dumps(doc.__dict__) + "\n")
                        output_file.flush()
                        
                        self._register_document(doc, checkpoint)
                        checkpoint.scholar_ids_collected.add(article.paper_id)
                        checkpoint.total_collected += 1
                        collected += 1
                        
                        if collected % self.config.checkpoint_interval == 0:
                            checkpoint.save(self.checkpoint_path)
                            logger.info(f"  Progress: {collected:,} papers from Semantic Scholar")
                    
                    checkpoint.scholar_queries_completed.append(query)
                    checkpoint.save(self.checkpoint_path)
                    
                    # Rate limiting
                    await asyncio.sleep(3.0 if not self.s2_api_key else 1.0)
                    
                except Exception as e:
                    logger.error(f"  Error in S2 query '{query}': {e}")
                    checkpoint.total_errors += 1
        
        return collected
    
    async def _collect_arxiv(
        self,
        checkpoint: IngestionCheckpoint,
        output_file
    ) -> int:
        """Collect papers from arXiv."""
        client = ArxivClient()
        collected = 0
        
        queries = [
            "electroencephalography",
            "EEG classification",
            "brain computer interface",
            "neural signal processing",
            "seizure detection",
            "sleep stage classification",
            "motor imagery",
            "EEG deep learning",
            "brain connectivity",
            "neural oscillations"
        ]
        
        async with client:
            for query in queries:
                if query in checkpoint.arxiv_queries_completed:
                    continue
                
                if collected >= self.config.arxiv_target:
                    break
                
                try:
                    logger.info(f"  arXiv query: {query}")
                    papers = await client.search_eeg_papers(query, max_results=500)
                    
                    for paper in papers:
                        if paper.arxiv_id in checkpoint.arxiv_ids_collected:
                            continue
                        
                        doc = UnifiedDocument(
                            id=f"arxiv_{paper.arxiv_id}",
                            source="arxiv",
                            title=paper.title,
                            abstract=paper.abstract,
                            authors=paper.authors,
                            publication_date=paper.published,
                            url=paper.pdf_url,
                            arxiv_id=paper.arxiv_id,
                            categories=paper.categories
                        )
                        
                        if self._is_duplicate(doc, checkpoint):
                            checkpoint.total_duplicates_skipped += 1
                            continue
                        
                        output_file.write(json.dumps(doc.__dict__) + "\n")
                        output_file.flush()
                        
                        self._register_document(doc, checkpoint)
                        checkpoint.arxiv_ids_collected.add(paper.arxiv_id)
                        checkpoint.total_collected += 1
                        collected += 1
                        
                        if collected % self.config.checkpoint_interval == 0:
                            checkpoint.save(self.checkpoint_path)
                            logger.info(f"  Progress: {collected:,} papers from arXiv")
                    
                    checkpoint.arxiv_queries_completed.append(query)
                    checkpoint.save(self.checkpoint_path)
                    
                    await asyncio.sleep(3.0)  # arXiv rate limit
                    
                except Exception as e:
                    logger.error(f"  Error in arXiv query '{query}': {e}")
                    checkpoint.total_errors += 1
        
        return collected
    
    async def _collect_openalex(
        self,
        checkpoint: IngestionCheckpoint,
        output_file
    ) -> int:
        """Collect papers from OpenAlex."""
        client = OpenAlexClient(email=self.email)
        collected = 0
        
        async with client:
            # Collect by concept
            for concept_id in client.EEG_CONCEPTS[:10]:
                if concept_id in checkpoint.openalex_concepts_completed:
                    continue
                
                if collected >= self.config.openalex_target:
                    break
                
                try:
                    logger.info(f"  OpenAlex concept: {concept_id}")
                    works = await client.get_works_by_concept(
                        concept_id, 
                        per_page=200,
                        max_pages=10
                    )
                    
                    for work in works:
                        if work.openalex_id in checkpoint.openalex_ids_collected:
                            continue
                        
                        doc = UnifiedDocument(
                            id=f"openalex_{work.openalex_id}",
                            source="openalex",
                            title=work.title,
                            abstract=work.abstract,
                            authors=work.authors,
                            publication_date=str(work.publication_year) if work.publication_year else None,
                            doi=work.doi,
                            url=work.url,
                            citation_count=work.cited_by_count,
                            openalex_id=work.openalex_id,
                            concepts=work.concepts
                        )
                        
                        if self._is_duplicate(doc, checkpoint):
                            checkpoint.total_duplicates_skipped += 1
                            continue
                        
                        output_file.write(json.dumps(doc.__dict__) + "\n")
                        output_file.flush()
                        
                        self._register_document(doc, checkpoint)
                        checkpoint.openalex_ids_collected.add(work.openalex_id)
                        checkpoint.total_collected += 1
                        collected += 1
                        
                        if collected % self.config.checkpoint_interval == 0:
                            checkpoint.save(self.checkpoint_path)
                            logger.info(f"  Progress: {collected:,} papers from OpenAlex")
                    
                    checkpoint.openalex_concepts_completed.append(concept_id)
                    checkpoint.save(self.checkpoint_path)
                    
                except Exception as e:
                    logger.error(f"  Error in OpenAlex concept '{concept_id}': {e}")
                    checkpoint.total_errors += 1
        
        return collected
    
    async def run_bulk_ingestion(self, resume: bool = True) -> Dict:
        """
        Run bulk ingestion of EEG research papers.
        
        Args:
            resume: If True, resume from checkpoint. If False, start fresh.
            
        Returns:
            Statistics dictionary
        """
        self.start_time = datetime.now()
        
        # Load or create checkpoint
        if resume and self.checkpoint_path.exists():
            checkpoint = IngestionCheckpoint.load(self.checkpoint_path)
            logger.info(f"Resuming from checkpoint: {checkpoint.total_collected:,} papers collected")
        else:
            checkpoint = IngestionCheckpoint(
                run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                started_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        
        self._log_rate_limits()
        
        # Open output file (append mode for resume)
        mode = "a" if resume and self.output_file.exists() else "w"
        
        with open(self.output_file, mode) as output_file:
            # Collect from each source
            logger.info("\n" + "=" * 60)
            logger.info("Starting PubMed collection...")
            logger.info("=" * 60)
            pubmed_count = await self._collect_pubmed(checkpoint, output_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("Starting Semantic Scholar collection...")
            logger.info("=" * 60)
            scholar_count = await self._collect_semantic_scholar(checkpoint, output_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("Starting arXiv collection...")
            logger.info("=" * 60)
            arxiv_count = await self._collect_arxiv(checkpoint, output_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("Starting OpenAlex collection...")
            logger.info("=" * 60)
            openalex_count = await self._collect_openalex(checkpoint, output_file)
        
        # Final checkpoint
        checkpoint.save(self.checkpoint_path)
        
        # Calculate stats
        elapsed = datetime.now() - self.start_time
        
        stats = {
            "total_collected": checkpoint.total_collected,
            "pubmed": len(checkpoint.pubmed_ids_collected),
            "semantic_scholar": len(checkpoint.scholar_ids_collected),
            "arxiv": len(checkpoint.arxiv_ids_collected),
            "openalex": len(checkpoint.openalex_ids_collected),
            "duplicates_skipped": checkpoint.total_duplicates_skipped,
            "errors": checkpoint.total_errors,
            "elapsed_time": str(elapsed),
            "output_file": str(self.output_file),
            "checkpoint_file": str(self.checkpoint_path)
        }
        
        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("BULK INGESTION COMPLETE")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        return stats


async def main():
    """CLI entry point for bulk ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bulk ingestion of EEG research papers (100K+)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh bulk ingestion
  python -m eeg_rag.ingestion.bulk_ingestion
  
  # Resume interrupted ingestion
  python -m eeg_rag.ingestion.bulk_ingestion --resume
  
  # Custom targets
  python -m eeg_rag.ingestion.bulk_ingestion --pubmed 100000 --scholar 50000

For faster ingestion, set optional API keys:
  export PUBMED_API_KEY="your-key"  # https://www.ncbi.nlm.nih.gov/account/settings/
  export SEMANTIC_SCHOLAR_API_KEY="your-key"  # https://www.semanticscholar.org/product/api#api-key
        """
    )
    
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoint)")
    parser.add_argument("--pubmed", type=int, default=50000, help="PubMed target count")
    parser.add_argument("--scholar", type=int, default=30000, help="Semantic Scholar target")
    parser.add_argument("--arxiv", type=int, default=10000, help="arXiv target count")
    parser.add_argument("--openalex", type=int, default=30000, help="OpenAlex target count")
    parser.add_argument("--output-dir", default="data/bulk_ingestion", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("bulk_ingestion.log"),
            logging.StreamHandler()
        ]
    )
    
    config = BulkIngestionConfig(
        pubmed_target=args.pubmed,
        scholar_target=args.scholar,
        arxiv_target=args.arxiv,
        openalex_target=args.openalex,
        output_dir=args.output_dir
    )
    
    manager = BulkIngestionManager(config=config)
    
    await manager.run_bulk_ingestion(resume=not args.fresh)


if __name__ == "__main__":
    asyncio.run(main())
