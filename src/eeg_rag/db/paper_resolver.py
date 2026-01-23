# src/eeg_rag/db/paper_resolver.py
"""
On-demand paper resolver with local caching.

Resolves paper identifiers (PMID, DOI, OpenAlex ID, arXiv ID) to full content
by fetching from external APIs. Results are cached locally.

Supported Sources:
- PubMed (E-utilities API) - PMID lookup
- OpenAlex API - DOI and OpenAlex ID lookup  
- Semantic Scholar API - DOI, PMID, arXiv ID lookup
- arXiv API - arXiv ID lookup
- CrossRef API - DOI metadata lookup
- bioRxiv/medRxiv API - preprint DOI lookup
"""

import asyncio
import json
import logging
import re
import sqlite3
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)


@dataclass
class ResolvedPaper:
    """Full paper content resolved from external APIs."""
    pmid: Optional[str] = None
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None  # Semantic Scholar ID
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source: str = "unknown"
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0
    fetched_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_cache_row(cls, row: sqlite3.Row) -> "ResolvedPaper":
        """Create from cache database row."""
        def parse_json_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return val
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []
        
        return cls(
            pmid=row["pmid"],
            doi=row["doi"],
            openalex_id=row.get("openalex_id"),
            arxiv_id=row.get("arxiv_id"),
            s2_id=row.get("s2_id"),
            title=row["title"] or "",
            abstract=row["abstract"] or "",
            authors=parse_json_list(row["authors"]),
            year=row["year"],
            journal=row["journal"],
            mesh_terms=parse_json_list(row["mesh_terms"]),
            keywords=parse_json_list(row.get("keywords", "[]")),
            source=row["source"] or "unknown",
            url=row.get("url"),
            pdf_url=row.get("pdf_url"),
            citation_count=row.get("citation_count", 0),
            fetched_at=row["fetched_at"],
        )
    
    @property
    def primary_id(self) -> str:
        """Get the primary identifier."""
        if self.pmid:
            return f"pmid:{self.pmid}"
        elif self.doi:
            return f"doi:{self.doi}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        elif self.openalex_id:
            return f"openalex:{self.openalex_id}"
        elif self.s2_id:
            return f"s2:{self.s2_id}"
        return f"title:{self.title[:50]}"


class PaperResolver:
    """
    Resolves paper IDs to full content using external APIs.
    
    Supports multiple sources with fallback chain:
    - PubMed E-utilities (PMID) - Best for medical/life sciences
    - Semantic Scholar (DOI, PMID, arXiv) - Broad coverage, citation data
    - OpenAlex API (DOI, OpenAlex ID) - Open metadata
    - arXiv API (arXiv ID) - Preprints in physics, CS, etc.
    - CrossRef API (DOI) - Authoritative DOI metadata
    - bioRxiv/medRxiv API (DOI) - Life science preprints
    
    Results are cached locally in ~/.eeg_rag/cache/papers.db
    """
    
    CACHE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS papers (
        pmid TEXT PRIMARY KEY,
        doi TEXT,
        openalex_id TEXT,
        arxiv_id TEXT,
        s2_id TEXT,
        title TEXT,
        abstract TEXT,
        authors TEXT,  -- JSON array
        year INTEGER,
        journal TEXT,
        mesh_terms TEXT,  -- JSON array
        keywords TEXT,  -- JSON array
        source TEXT,
        url TEXT,
        pdf_url TEXT,
        citation_count INTEGER DEFAULT 0,
        fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_cache_doi ON papers(doi) WHERE doi IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_cache_openalex ON papers(openalex_id) WHERE openalex_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_cache_arxiv ON papers(arxiv_id) WHERE arxiv_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_cache_s2 ON papers(s2_id) WHERE s2_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_cache_fetched ON papers(fetched_at);
    
    -- Embeddings cache (optional, for vector search)
    CREATE TABLE IF NOT EXISTS embeddings (
        paper_id TEXT PRIMARY KEY,
        embedding BLOB,
        model TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Chunks cache (optional, for RAG)
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT,
        chunk_index INTEGER,
        text TEXT,
        embedding BLOB,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (paper_id) REFERENCES papers(pmid)
    );
    
    CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the paper resolver.
        
        Args:
            cache_dir: Directory for local cache. Defaults to ~/.eeg_rag/cache/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".eeg_rag" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_db = self.cache_dir / "papers.db"
        
        self._init_cache()
        
        # Rate limiting timestamps
        self._last_pubmed_request = datetime.min
        self._last_openalex_request = datetime.min
        self._last_s2_request = datetime.min
        self._last_arxiv_request = datetime.min
        self._last_crossref_request = datetime.min
        self._last_biorxiv_request = datetime.min
        
        # Rate limits (time between requests)
        self._pubmed_rate_limit = timedelta(seconds=0.34)  # 3/sec without API key
        self._openalex_rate_limit = timedelta(seconds=0.1)  # 10/sec
        self._s2_rate_limit = timedelta(seconds=1.0)  # 1/sec without API key (public)
        self._arxiv_rate_limit = timedelta(seconds=3.0)  # Polite: 3 sec between requests
        self._crossref_rate_limit = timedelta(seconds=0.05)  # ~20/sec with polite pool
        self._biorxiv_rate_limit = timedelta(seconds=0.5)  # ~2/sec
    
    def _init_cache(self):
        """Initialize the local cache database."""
        conn = sqlite3.connect(str(self.cache_db))
        conn.executescript(self.CACHE_SCHEMA)
        conn.close()
        logger.debug(f"Cache initialized at {self.cache_db}")
    
    @contextmanager
    def _get_cache_connection(self):
        """Context manager for cache database."""
        conn = sqlite3.connect(str(self.cache_db), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get_from_cache(self, pmid: Optional[str] = None, doi: Optional[str] = None) -> Optional[ResolvedPaper]:
        """Check if paper is in local cache."""
        with self._get_cache_connection() as conn:
            if pmid:
                cursor = conn.execute(
                    "SELECT * FROM papers WHERE pmid = ?", (pmid,)
                )
            elif doi:
                cursor = conn.execute(
                    "SELECT * FROM papers WHERE doi = ?", (doi,)
                )
            else:
                return None
            
            row = cursor.fetchone()
            if row:
                return ResolvedPaper.from_cache_row(row)
        return None
    
    def save_to_cache(self, paper: ResolvedPaper) -> bool:
        """Save paper to local cache."""
        with self._get_cache_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO papers
                    (pmid, doi, openalex_id, arxiv_id, s2_id, title, abstract, authors, year,
                     journal, mesh_terms, keywords, source, url, pdf_url, citation_count, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.pmid,
                    paper.doi,
                    paper.openalex_id,
                    paper.arxiv_id,
                    paper.s2_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    paper.year,
                    paper.journal,
                    json.dumps(paper.mesh_terms),
                    json.dumps(paper.keywords),
                    paper.source,
                    paper.url,
                    paper.pdf_url,
                    paper.citation_count,
                    datetime.now().isoformat(),
                ))
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to cache paper: {e}")
                return False
    
    async def resolve_pmid(self, pmid: str) -> Optional[ResolvedPaper]:
        """
        Resolve a PMID to full paper content.
        
        Checks cache first, then fetches from PubMed if needed.
        """
        # Check cache
        cached = self.get_from_cache(pmid=pmid)
        if cached:
            logger.debug(f"Cache hit for PMID {pmid}")
            return cached
        
        # Fetch from PubMed
        logger.debug(f"Fetching PMID {pmid} from PubMed")
        paper = await self._fetch_from_pubmed(pmid)
        
        if paper:
            self.save_to_cache(paper)
        
        return paper
    
    async def resolve_doi(self, doi: str) -> Optional[ResolvedPaper]:
        """
        Resolve a DOI to full paper content.
        
        Checks cache first, then fetches from OpenAlex if needed.
        """
        # Check cache
        cached = self.get_from_cache(doi=doi)
        if cached:
            logger.debug(f"Cache hit for DOI {doi}")
            return cached
        
        # Fetch from OpenAlex
        logger.debug(f"Fetching DOI {doi} from OpenAlex")
        paper = await self._fetch_from_openalex_doi(doi)
        
        if paper:
            self.save_to_cache(paper)
        
        return paper
    
    async def resolve_batch(
        self,
        pmids: Optional[List[str]] = None,
        dois: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[ResolvedPaper]:
        """
        Resolve multiple identifiers efficiently.
        
        Separates cached vs uncached, then batch-fetches uncached.
        """
        results: List[ResolvedPaper] = []
        pmids = pmids or []
        dois = dois or []
        
        # Check cache for PMIDs
        to_fetch_pmids: List[str] = []
        for pmid in pmids:
            cached = self.get_from_cache(pmid=pmid)
            if cached:
                results.append(cached)
            else:
                to_fetch_pmids.append(pmid)
        
        # Check cache for DOIs
        to_fetch_dois: List[str] = []
        for doi in dois:
            cached = self.get_from_cache(doi=doi)
            if cached:
                results.append(cached)
            else:
                to_fetch_dois.append(doi)
        
        logger.info(
            f"Cache: {len(pmids) + len(dois) - len(to_fetch_pmids) - len(to_fetch_dois)} hits, "
            f"Fetching: {len(to_fetch_pmids)} PMIDs, {len(to_fetch_dois)} DOIs"
        )
        
        # Batch fetch PMIDs from PubMed
        if to_fetch_pmids:
            fetched = await self._batch_fetch_pubmed(to_fetch_pmids, batch_size)
            for paper in fetched:
                self.save_to_cache(paper)
                results.append(paper)
        
        # Batch fetch DOIs from OpenAlex
        if to_fetch_dois:
            fetched = await self._batch_fetch_openalex(to_fetch_dois, batch_size)
            for paper in fetched:
                self.save_to_cache(paper)
                results.append(paper)
        
        return results
    
    async def _rate_limit_pubmed(self):
        """Enforce PubMed rate limiting."""
        elapsed = datetime.now() - self._last_pubmed_request
        if elapsed < self._pubmed_rate_limit:
            await asyncio.sleep((self._pubmed_rate_limit - elapsed).total_seconds())
        self._last_pubmed_request = datetime.now()
    
    async def _rate_limit_openalex(self):
        """Enforce OpenAlex rate limiting."""
        elapsed = datetime.now() - self._last_openalex_request
        if elapsed < self._openalex_rate_limit:
            await asyncio.sleep((self._openalex_rate_limit - elapsed).total_seconds())
        self._last_openalex_request = datetime.now()
    
    async def _fetch_from_pubmed(self, pmid: str) -> Optional[ResolvedPaper]:
        """Fetch a single paper from PubMed E-utilities."""
        try:
            import aiohttp
        except ImportError:
            # Fallback to sync
            return self._fetch_from_pubmed_sync(pmid)
        
        await self._rate_limit_pubmed()
        
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        xml_text = await resp.text()
                        papers = self._parse_pubmed_xml(xml_text)
                        return papers[0] if papers else None
        except Exception as e:
            logger.error(f"PubMed fetch failed for {pmid}: {e}")
        
        return None
    
    def _fetch_from_pubmed_sync(self, pmid: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for PubMed fetch."""
        import urllib.request
        
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&rettype=abstract"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_text = resp.read().decode('utf-8')
                papers = self._parse_pubmed_xml(xml_text)
                return papers[0] if papers else None
        except Exception as e:
            logger.error(f"PubMed sync fetch failed for {pmid}: {e}")
        
        return None
    
    async def _batch_fetch_pubmed(
        self,
        pmids: List[str],
        batch_size: int = 100
    ) -> List[ResolvedPaper]:
        """Batch fetch from PubMed."""
        results: List[ResolvedPaper] = []
        
        try:
            import aiohttp
        except ImportError:
            # Sync fallback
            for pmid in pmids:
                paper = self._fetch_from_pubmed_sync(pmid)
                if paper:
                    results.append(paper)
            return results
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            await self._rate_limit_pubmed()
            
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=60) as resp:
                        if resp.status == 200:
                            xml_text = await resp.text()
                            papers = self._parse_pubmed_xml(xml_text)
                            results.extend(papers)
                            logger.debug(f"Fetched {len(papers)} papers from PubMed batch")
            except Exception as e:
                logger.error(f"PubMed batch fetch failed: {e}")
        
        return results
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[ResolvedPaper]:
        """Parse PubMed XML response into ResolvedPaper objects."""
        papers: List[ResolvedPaper] = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    medline = article.find("MedlineCitation")
                    if medline is None:
                        continue
                    
                    # PMID
                    pmid_elem = medline.find("PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    # Article info
                    art = medline.find("Article")
                    if art is None:
                        continue
                    
                    # Title
                    title_elem = art.find("ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Abstract
                    abstract_parts = []
                    abstract_elem = art.find("Abstract")
                    if abstract_elem is not None:
                        for text in abstract_elem.findall("AbstractText"):
                            if text.text:
                                label = text.get("Label", "")
                                if label:
                                    abstract_parts.append(f"{label}: {text.text}")
                                else:
                                    abstract_parts.append(text.text)
                    abstract = " ".join(abstract_parts)
                    
                    # Authors
                    authors = []
                    author_list = art.find("AuthorList")
                    if author_list is not None:
                        for author in author_list.findall("Author"):
                            last = author.find("LastName")
                            first = author.find("ForeName")
                            if last is not None and first is not None:
                                authors.append(f"{last.text}, {first.text}")
                            elif last is not None:
                                authors.append(last.text)
                    
                    # Year
                    year = None
                    pub_date = art.find(".//PubDate")
                    if pub_date is not None:
                        year_elem = pub_date.find("Year")
                        if year_elem is not None and year_elem.text:
                            try:
                                year = int(year_elem.text)
                            except ValueError:
                                pass
                    
                    # Journal
                    journal = None
                    journal_elem = art.find(".//Journal/Title")
                    if journal_elem is not None:
                        journal = journal_elem.text
                    
                    # DOI
                    doi = None
                    for id_elem in article.findall(".//ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text
                            break
                    
                    # MeSH terms
                    mesh_terms = []
                    for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                        if mesh.text:
                            mesh_terms.append(mesh.text)
                    
                    # Keywords
                    keywords = []
                    for kw in medline.findall(".//Keyword"):
                        if kw.text:
                            keywords.append(kw.text)
                    
                    paper = ResolvedPaper(
                        pmid=pmid,
                        doi=doi,
                        title=title or "",
                        abstract=abstract,
                        authors=authors,
                        year=year,
                        journal=journal,
                        mesh_terms=mesh_terms,
                        keywords=keywords,
                        source="pubmed",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        fetched_at=datetime.now().isoformat(),
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
        
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        
        return papers
    
    async def _fetch_from_openalex_doi(self, doi: str) -> Optional[ResolvedPaper]:
        """Fetch paper from OpenAlex by DOI."""
        try:
            import aiohttp
        except ImportError:
            return self._fetch_from_openalex_sync(doi)
        
        await self._rate_limit_openalex()
        
        encoded_doi = quote(doi, safe='')
        url = f"https://api.openalex.org/works/https://doi.org/{encoded_doi}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_openalex_work(data)
        except Exception as e:
            logger.error(f"OpenAlex fetch failed for DOI {doi}: {e}")
        
        return None
    
    def _fetch_from_openalex_sync(self, doi: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for OpenAlex."""
        import urllib.request
        
        encoded_doi = quote(doi, safe='')
        url = f"https://api.openalex.org/works/https://doi.org/{encoded_doi}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return self._parse_openalex_work(data)
        except Exception as e:
            logger.error(f"OpenAlex sync fetch failed for DOI {doi}: {e}")
        
        return None
    
    async def _batch_fetch_openalex(
        self,
        dois: List[str],
        batch_size: int = 50
    ) -> List[ResolvedPaper]:
        """Batch fetch from OpenAlex."""
        results: List[ResolvedPaper] = []
        
        try:
            import aiohttp
        except ImportError:
            for doi in dois:
                paper = self._fetch_from_openalex_sync(doi)
                if paper:
                    results.append(paper)
            return results
        
        # OpenAlex supports filter queries for batch
        for i in range(0, len(dois), batch_size):
            batch = dois[i:i + batch_size]
            await self._rate_limit_openalex()
            
            # Build filter query
            doi_filter = "|".join(f"https://doi.org/{quote(d, safe='')}" for d in batch)
            url = f"https://api.openalex.org/works?filter=doi:{doi_filter}&per_page={batch_size}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=60) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for work in data.get("results", []):
                                paper = self._parse_openalex_work(work)
                                if paper:
                                    results.append(paper)
            except Exception as e:
                logger.error(f"OpenAlex batch fetch failed: {e}")
        
        return results
    
    def _parse_openalex_work(self, data: Dict[str, Any]) -> Optional[ResolvedPaper]:
        """Parse OpenAlex work into ResolvedPaper."""
        try:
            # Extract PMID from IDs
            pmid = None
            ids = data.get("ids", {})
            if "pmid" in ids:
                pmid = ids["pmid"].replace("https://pubmed.ncbi.nlm.nih.gov/", "")
            
            # DOI
            doi = data.get("doi", "").replace("https://doi.org/", "") if data.get("doi") else None
            
            # OpenAlex ID
            openalex_id = data.get("id", "").replace("https://openalex.org/", "")
            
            # Title
            title = data.get("title", "") or ""
            
            # Abstract
            abstract = ""
            if data.get("abstract_inverted_index"):
                # Reconstruct abstract from inverted index
                inv_idx = data["abstract_inverted_index"]
                if inv_idx:
                    word_positions = []
                    for word, positions in inv_idx.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract = " ".join(w for _, w in word_positions)
            
            # Authors
            authors = []
            for authorship in data.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors.append(name)
            
            # Year
            year = data.get("publication_year")
            
            # Journal/venue
            journal = None
            primary_loc = data.get("primary_location", {})
            if primary_loc:
                source = primary_loc.get("source", {})
                if source:
                    journal = source.get("display_name")
            
            # Keywords/concepts
            keywords = []
            for concept in data.get("concepts", [])[:10]:
                if concept.get("display_name"):
                    keywords.append(concept["display_name"])
            
            # Citation count
            citation_count = data.get("cited_by_count", 0)
            
            return ResolvedPaper(
                pmid=pmid,
                doi=doi,
                openalex_id=openalex_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                keywords=keywords,
                source="openalex",
                url=data.get("id"),
                citation_count=citation_count,
                fetched_at=datetime.now().isoformat(),
            )
        
        except Exception as e:
            logger.error(f"Failed to parse OpenAlex work: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_cache_connection() as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            stats["cached_papers"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            stats["cached_embeddings"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            stats["cached_chunks"] = cursor.fetchone()[0]
            
            # Cache size
            if self.cache_db.exists():
                stats["cache_size_mb"] = self.cache_db.stat().st_size / 1024 / 1024
            
            return stats
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear the cache, optionally only entries older than N days."""
        with self._get_cache_connection() as conn:
            if older_than_days:
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                conn.execute("DELETE FROM papers WHERE fetched_at < ?", (cutoff,))
                conn.execute("DELETE FROM chunks WHERE created_at < ?", (cutoff,))
            else:
                conn.execute("DELETE FROM papers")
                conn.execute("DELETE FROM embeddings")
                conn.execute("DELETE FROM chunks")
            conn.execute("VACUUM")


# Convenience function
def get_paper_resolver() -> PaperResolver:
    """Get the default paper resolver."""
    return PaperResolver()
