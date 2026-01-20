"""
Master ingestion pipeline that coordinates all data sources.
"""

import asyncio
from dataclasses import dataclass, asdict, field
from typing import Optional, AsyncIterator, Union
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

from .pubmed_client import PubMedClient, PubMedArticle
from .scholar_client import SemanticScholarClient, ScholarArticle
from .arxiv_client import ArxivClient, ArxivPaper
from .openalex_client import OpenAlexClient, OpenAlexWork

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDocument:
    """
    Unified document format for all sources.
    This is what gets stored and indexed in the RAG system.
    """
    # Core identifiers
    doc_id: str
    source: str  # pubmed, semantic_scholar, arxiv, openalex
    
    # External IDs (for deduplication and linking)
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    openalex_id: Optional[str] = None
    
    # Content
    title: str = ""
    abstract: str = ""
    full_text: Optional[str] = None
    
    # Metadata
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    publication_date: Optional[str] = None
    publication_year: Optional[int] = None
    
    # Academic metrics
    citation_count: int = 0
    reference_count: int = 0
    
    # Classification
    mesh_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    
    # EEG-specific metadata
    frequency_bands: list[str] = field(default_factory=list)
    erp_components: list[str] = field(default_factory=list)
    clinical_conditions: list[str] = field(default_factory=list)
    electrode_systems: list[str] = field(default_factory=list)
    sample_size: Optional[int] = None
    
    # Relationships
    references: list[str] = field(default_factory=list)  # IDs of cited papers
    citations: list[str] = field(default_factory=list)   # IDs of citing papers
    related_works: list[str] = field(default_factory=list)
    
    # Access info
    open_access: bool = False
    pdf_url: Optional[str] = None
    
    # Ingestion metadata
    ingested_at: str = ""
    
    def __post_init__(self):
        if not self.ingested_at:
            self.ingested_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def get_embedding_text(self) -> str:
        """Get text for embedding generation."""
        parts = [self.title]
        if self.abstract:
            parts.append(self.abstract)
        if self.mesh_terms:
            parts.append("Topics: " + ", ".join(self.mesh_terms[:10]))
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords[:10]))
        return " ".join(parts)


class IngestionPipeline:
    """
    Master pipeline for collecting data from all sources.
    Handles deduplication, normalization, and storage.
    """
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        pubmed_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        email: str = "your-email@example.com"
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            output_dir: Directory for storing raw ingested data
            pubmed_api_key: NCBI API key for higher rate limits
            semantic_scholar_api_key: Semantic Scholar API key
            email: Contact email for API identification
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.pubmed = PubMedClient(api_key=pubmed_api_key, email=email)
        self.semantic_scholar = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.arxiv = ArxivClient()
        self.openalex = OpenAlexClient(email=email)
        
        # Deduplication tracking
        self.seen_dois: set[str] = set()
        self.seen_pmids: set[str] = set()
        self.seen_titles: set[str] = set()  # Normalized titles for fuzzy matching
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        import re
        # Lowercase, remove punctuation, normalize whitespace
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        title = ' '.join(title.split())
        return title
    
    def _generate_doc_id(self, doc: UnifiedDocument) -> str:
        """Generate unique document ID."""
        # Prefer stable IDs
        if doc.doi:
            return f"doi:{doc.doi}"
        if doc.pmid:
            return f"pmid:{doc.pmid}"
        if doc.arxiv_id:
            return f"arxiv:{doc.arxiv_id}"
        if doc.semantic_scholar_id:
            return f"s2:{doc.semantic_scholar_id}"
        if doc.openalex_id:
            return f"oa:{doc.openalex_id}"
        
        # Fallback to title hash
        title_hash = hashlib.md5(doc.title.encode()).hexdigest()[:12]
        return f"hash:{title_hash}"
    
    def _is_duplicate(self, doc: UnifiedDocument) -> bool:
        """Check if document is a duplicate."""
        if doc.doi and doc.doi in self.seen_dois:
            return True
        if doc.pmid and doc.pmid in self.seen_pmids:
            return True
        
        norm_title = self._normalize_title(doc.title)
        if norm_title in self.seen_titles:
            return True
        
        return False
    
    def _mark_seen(self, doc: UnifiedDocument):
        """Mark document as seen for deduplication."""
        if doc.doi:
            self.seen_dois.add(doc.doi)
        if doc.pmid:
            self.seen_pmids.add(doc.pmid)
        self.seen_titles.add(self._normalize_title(doc.title))
    
    def _convert_pubmed(self, article: PubMedArticle) -> UnifiedDocument:
        """Convert PubMed article to unified format."""
        doc = UnifiedDocument(
            doc_id="",
            source="pubmed",
            pmid=article.pmid,
            doi=article.doi,
            pmc_id=article.pmc_id,
            title=article.title,
            abstract=article.abstract,
            full_text=article.full_text,
            authors=article.authors,
            journal=article.journal,
            publication_date=article.publication_date.isoformat() if article.publication_date else None,
            publication_year=article.publication_date.year if article.publication_date else None,
            mesh_terms=article.mesh_terms,
            keywords=article.keywords,
            reference_count=len(article.references),
            references=article.references,
            frequency_bands=article.frequency_bands_mentioned,
            erp_components=article.erp_components,
            clinical_conditions=article.clinical_conditions,
            electrode_systems=article.electrode_systems,
            sample_size=article.sample_size,
            open_access=article.full_text_available
        )
        doc.doc_id = self._generate_doc_id(doc)
        return doc
    
    def _convert_scholar(self, article: ScholarArticle) -> UnifiedDocument:
        """Convert Semantic Scholar article to unified format."""
        doc = UnifiedDocument(
            doc_id="",
            source="semantic_scholar",
            semantic_scholar_id=article.semantic_scholar_id,
            doi=article.doi,
            pmid=article.pmid,
            arxiv_id=article.arxiv_id,
            title=article.title,
            abstract=article.abstract or "",
            authors=article.authors,
            journal=article.venue,
            publication_year=article.year,
            citation_count=article.citation_count,
            reference_count=len(article.references),
            references=article.references,
            citations=article.citations,
            fields_of_study=article.fields_of_study or [],
            open_access=article.pdf_url is not None,
            pdf_url=article.pdf_url
        )
        doc.doc_id = self._generate_doc_id(doc)
        return doc
    
    def _convert_arxiv(self, paper: ArxivPaper) -> UnifiedDocument:
        """Convert arXiv paper to unified format."""
        doc = UnifiedDocument(
            doc_id="",
            source="arxiv",
            arxiv_id=paper.arxiv_id,
            doi=paper.doi,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            journal=paper.journal_ref or "",
            publication_date=paper.published.isoformat(),
            publication_year=paper.published.year,
            keywords=paper.categories,
            open_access=True,
            pdf_url=paper.pdf_url
        )
        doc.doc_id = self._generate_doc_id(doc)
        return doc
    
    def _convert_openalex(self, work: OpenAlexWork) -> UnifiedDocument:
        """Convert OpenAlex work to unified format."""
        doc = UnifiedDocument(
            doc_id="",
            source="openalex",
            openalex_id=work.openalex_id,
            doi=work.doi,
            pmid=work.pmid,
            title=work.title,
            abstract=work.abstract,
            authors=[a["name"] for a in work.authors],
            journal=work.journal,
            publication_date=work.publication_date.isoformat() if work.publication_date else None,
            publication_year=work.publication_date.year if work.publication_date else None,
            citation_count=work.citation_count,
            reference_count=len(work.referenced_works),
            concepts=[c["name"] for c in work.concepts if c.get("name")],
            references=work.referenced_works,
            related_works=work.related_works,
            open_access=work.open_access,
            pdf_url=work.pdf_url
        )
        doc.doc_id = self._generate_doc_id(doc)
        return doc

    async def ingest_pubmed(
        self,
        years_back: int = 10,
        max_per_query: int = 5000,
        include_full_text: bool = True
    ) -> AsyncIterator[UnifiedDocument]:
        """
        Ingest from PubMed.
        
        Args:
            years_back: Years of literature to collect
            max_per_query: Maximum articles per search query
            include_full_text: Whether to fetch PMC full text
            
        Yields:
            UnifiedDocument objects
        """
        logger.info("Starting PubMed ingestion...")
        
        async for article in self.pubmed.collect_eeg_corpus(
            years_back=years_back,
            max_per_query=max_per_query,
            include_full_text=include_full_text
        ):
            doc = self._convert_pubmed(article)
            
            if not self._is_duplicate(doc):
                self._mark_seen(doc)
                yield doc

    async def ingest_semantic_scholar(
        self,
        max_per_query: int = 1000,
        year_start: int = 2014
    ) -> AsyncIterator[UnifiedDocument]:
        """
        Ingest from Semantic Scholar.
        
        Args:
            max_per_query: Maximum papers per query
            year_start: Earliest publication year
            
        Yields:
            UnifiedDocument objects
        """
        logger.info("Starting Semantic Scholar ingestion...")
        
        async for article in self.semantic_scholar.collect_eeg_corpus(
            max_per_query=max_per_query,
            year_start=year_start
        ):
            doc = self._convert_scholar(article)
            
            if not self._is_duplicate(doc):
                self._mark_seen(doc)
                yield doc

    async def ingest_arxiv(
        self,
        max_results: int = 5000,
        years_back: int = 5
    ) -> AsyncIterator[UnifiedDocument]:
        """
        Ingest from arXiv.
        
        Args:
            max_results: Maximum total papers
            years_back: Years of papers to collect
            
        Yields:
            UnifiedDocument objects
        """
        logger.info("Starting arXiv ingestion...")
        
        async for paper in self.arxiv.collect_eeg_papers(
            max_results=max_results,
            years_back=years_back
        ):
            doc = self._convert_arxiv(paper)
            
            if not self._is_duplicate(doc):
                self._mark_seen(doc)
                yield doc

    async def ingest_openalex(
        self,
        from_year: int = 2014,
        max_results: int = 50000
    ) -> AsyncIterator[UnifiedDocument]:
        """
        Ingest from OpenAlex.
        
        Args:
            from_year: Earliest publication year
            max_results: Maximum works to collect
            
        Yields:
            UnifiedDocument objects
        """
        logger.info("Starting OpenAlex ingestion...")
        
        async for work in self.openalex.collect_eeg_corpus(
            from_year=from_year,
            max_results=max_results
        ):
            doc = self._convert_openalex(work)
            
            if not self._is_duplicate(doc):
                self._mark_seen(doc)
                yield doc

    async def run_full_ingestion(
        self,
        sources: list[str] = None,
        save_interval: int = 1000
    ) -> dict:
        """
        Run full ingestion from all sources.
        
        Args:
            sources: List of sources to use (default: all)
            save_interval: Save progress every N documents
            
        Returns:
            Statistics about ingestion
        """
        if sources is None:
            sources = ["pubmed", "semantic_scholar", "arxiv", "openalex"]
        
        stats = {source: 0 for source in sources}
        all_docs: list[UnifiedDocument] = []
        
        output_file = self.output_dir / f"eeg_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        async def process_source(source: str, iterator: AsyncIterator[UnifiedDocument]):
            nonlocal all_docs
            
            async for doc in iterator:
                all_docs.append(doc)
                stats[source] += 1
                
                # Periodic save
                if len(all_docs) % save_interval == 0:
                    self._save_documents(all_docs, output_file)
                    logger.info(f"Progress: {stats}")
        
        # Process sources sequentially to avoid rate limit issues
        if "pubmed" in sources:
            await process_source("pubmed", self.ingest_pubmed())
        
        if "semantic_scholar" in sources:
            await process_source("semantic_scholar", self.ingest_semantic_scholar())
        
        if "arxiv" in sources:
            await process_source("arxiv", self.ingest_arxiv())
        
        if "openalex" in sources:
            await process_source("openalex", self.ingest_openalex())
        
        # Final save
        self._save_documents(all_docs, output_file)
        
        stats["total"] = sum(v for k, v in stats.items() if k != "total")
        stats["output_file"] = str(output_file)
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def _save_documents(self, docs: list[UnifiedDocument], output_file: Path):
        """Save documents to JSONL file."""
        with open(output_file, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict()) + "\n")
        logger.info(f"Saved {len(docs)} documents to {output_file}")

    def reset_deduplication(self):
        """Reset deduplication tracking for a fresh run."""
        self.seen_dois.clear()
        self.seen_pmids.clear()
        self.seen_titles.clear()


# CLI interface
async def main():
    """CLI entry point for ingestion pipeline."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="EEG-RAG Data Ingestion")
    parser.add_argument("--sources", nargs="+", default=["pubmed", "semantic_scholar", "arxiv", "openalex"])
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--pubmed-api-key", default=os.environ.get("PUBMED_API_KEY"))
    parser.add_argument("--s2-api-key", default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    parser.add_argument("--email", default=os.environ.get("CONTACT_EMAIL", "your-email@example.com"))
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(
        output_dir=args.output_dir,
        pubmed_api_key=args.pubmed_api_key,
        semantic_scholar_api_key=args.s2_api_key,
        email=args.email
    )
    
    stats = await pipeline.run_full_ingestion(sources=args.sources)
    print(f"\nIngestion complete!\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
