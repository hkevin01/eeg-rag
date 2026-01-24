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
        title TEXT,
        abstract TEXT,
        authors TEXT,  -- JSON array
        year INTEGER,
        journal TEXT,
        mesh_terms TEXT,  -- JSON array
        keywords TEXT,  -- JSON array
        source TEXT,
        url TEXT,
        citation_count INTEGER DEFAULT 0,
        fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_cache_doi ON papers(doi) WHERE doi IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_cache_openalex ON papers(openalex_id) WHERE openalex_id IS NOT NULL;
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
        
        # Migration: Add new columns if they don't exist
        cursor = conn.execute("PRAGMA table_info(papers)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        
        migrations = [
            ("arxiv_id", "ALTER TABLE papers ADD COLUMN arxiv_id TEXT"),
            ("s2_id", "ALTER TABLE papers ADD COLUMN s2_id TEXT"),
            ("pdf_url", "ALTER TABLE papers ADD COLUMN pdf_url TEXT"),
        ]
        
        for col_name, sql in migrations:
            if col_name not in existing_cols:
                try:
                    conn.execute(sql)
                    logger.info(f"Added column {col_name} to cache")
                except sqlite3.Error as e:
                    logger.debug(f"Column {col_name} migration: {e}")
        
        # Add new indexes if they don't exist
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_arxiv ON papers(arxiv_id) WHERE arxiv_id IS NOT NULL")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_s2 ON papers(s2_id) WHERE s2_id IS NOT NULL")
        except sqlite3.Error:
            pass
        
        conn.commit()
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
    
    def get_from_cache(self, pmid: Optional[str] = None, doi: Optional[str] = None, arxiv_id: Optional[str] = None) -> Optional[ResolvedPaper]:
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
            elif arxiv_id:
                cursor = conn.execute(
                    "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
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
    
    # ========================================================================
    # Semantic Scholar API
    # ========================================================================
    
    async def _rate_limit_s2(self):
        """Enforce Semantic Scholar rate limiting."""
        elapsed = datetime.now() - self._last_s2_request
        if elapsed < self._s2_rate_limit:
            await asyncio.sleep((self._s2_rate_limit - elapsed).total_seconds())
        self._last_s2_request = datetime.now()
    
    async def resolve_from_semantic_scholar(
        self, 
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None
    ) -> Optional[ResolvedPaper]:
        """
        Resolve paper from Semantic Scholar API.
        
        Supports DOI, PMID, arXiv ID, and S2 paper ID lookups.
        API docs: https://api.semanticscholar.org/api-docs/
        """
        # Build paper ID for S2 API
        if doi:
            paper_id = f"DOI:{doi}"
        elif pmid:
            paper_id = f"PMID:{pmid}"
        elif arxiv_id:
            paper_id = f"ARXIV:{arxiv_id}"
        else:
            return None
        
        try:
            import aiohttp
        except ImportError:
            return self._fetch_from_s2_sync(paper_id)
        
        await self._rate_limit_s2()
        
        # Request fields we need
        fields = "paperId,externalIds,title,abstract,authors,year,venue,citationCount,openAccessPdf,fieldsOfStudy"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id, safe='')}?fields={fields}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_s2_paper(data)
                    elif resp.status == 404:
                        logger.debug(f"Paper not found in S2: {paper_id}")
                    else:
                        logger.warning(f"S2 API returned {resp.status} for {paper_id}")
        except Exception as e:
            logger.error(f"Semantic Scholar fetch failed for {paper_id}: {e}")
        
        return None
    
    def _fetch_from_s2_sync(self, paper_id: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for Semantic Scholar."""
        import urllib.request
        
        fields = "paperId,externalIds,title,abstract,authors,year,venue,citationCount,openAccessPdf,fieldsOfStudy"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(paper_id, safe='')}?fields={fields}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'EEG-RAG/1.0 (https://github.com/eeg-rag)')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return self._parse_s2_paper(data)
        except Exception as e:
            logger.error(f"S2 sync fetch failed for {paper_id}: {e}")
        return None
    
    def _parse_s2_paper(self, data: Dict[str, Any]) -> Optional[ResolvedPaper]:
        """Parse Semantic Scholar API response."""
        try:
            external_ids = data.get("externalIds", {}) or {}
            
            # Extract IDs
            pmid = external_ids.get("PubMed")
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            s2_id = data.get("paperId")
            
            # Authors
            authors = []
            for author in data.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Keywords from fields of study
            keywords = data.get("fieldsOfStudy", []) or []
            
            # PDF URL
            pdf_url = None
            open_access = data.get("openAccessPdf")
            if open_access and isinstance(open_access, dict):
                pdf_url = open_access.get("url")
            
            return ResolvedPaper(
                pmid=pmid,
                doi=doi,
                arxiv_id=arxiv_id,
                s2_id=s2_id,
                title=data.get("title", "") or "",
                abstract=data.get("abstract", "") or "",
                authors=authors,
                year=data.get("year"),
                journal=data.get("venue", ""),
                keywords=keywords,
                source="semantic_scholar",
                url=f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id else None,
                pdf_url=pdf_url,
                citation_count=data.get("citationCount", 0) or 0,
                fetched_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to parse S2 paper: {e}")
            return None
    
    # ========================================================================
    # arXiv API
    # ========================================================================
    
    async def _rate_limit_arxiv(self):
        """Enforce arXiv rate limiting (3 seconds between requests)."""
        elapsed = datetime.now() - self._last_arxiv_request
        if elapsed < self._arxiv_rate_limit:
            await asyncio.sleep((self._arxiv_rate_limit - elapsed).total_seconds())
        self._last_arxiv_request = datetime.now()
    
    async def resolve_arxiv(self, arxiv_id: str) -> Optional[ResolvedPaper]:
        """
        Resolve paper from arXiv API.
        
        Args:
            arxiv_id: arXiv identifier (e.g., "2301.00001" or "hep-th/0601001")
        
        API docs: https://info.arxiv.org/help/api/basics.html
        """
        # Clean the ID (remove version if present)
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        
        try:
            import aiohttp
        except ImportError:
            return self._fetch_from_arxiv_sync(clean_id)
        
        await self._rate_limit_arxiv()
        
        url = f"http://export.arxiv.org/api/query?id_list={clean_id}&max_results=1"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 200:
                        xml_text = await resp.text()
                        return self._parse_arxiv_response(xml_text, clean_id)
        except Exception as e:
            logger.error(f"arXiv fetch failed for {arxiv_id}: {e}")
        
        return None
    
    def _fetch_from_arxiv_sync(self, arxiv_id: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for arXiv."""
        import urllib.request
        
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_text = resp.read().decode('utf-8')
                return self._parse_arxiv_response(xml_text, arxiv_id)
        except Exception as e:
            logger.error(f"arXiv sync fetch failed for {arxiv_id}: {e}")
        return None
    
    def _parse_arxiv_response(self, xml_text: str, arxiv_id: str) -> Optional[ResolvedPaper]:
        """Parse arXiv Atom API response."""
        try:
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(xml_text)
            entry = root.find('atom:entry', ns)
            
            if entry is None:
                logger.debug(f"No entry found for arXiv ID {arxiv_id}")
                return None
            
            # Title
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
            
            # Abstract/Summary
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            # Publication date (year)
            published_elem = entry.find('atom:published', ns)
            year = None
            if published_elem is not None and published_elem.text:
                try:
                    year = int(published_elem.text[:4])
                except (ValueError, IndexError):
                    pass
            
            # DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', ns)
            if doi_elem is not None:
                doi = doi_elem.text
            
            # Categories as keywords
            keywords = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term:
                    keywords.append(term)
            
            # Primary category as journal (sort of)
            journal = None
            primary_cat = entry.find('arxiv:primary_category', ns)
            if primary_cat is not None:
                journal = f"arXiv:{primary_cat.get('term', '')}"
            
            # PDF link
            pdf_url = None
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            
            return ResolvedPaper(
                doi=doi,
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                keywords=keywords,
                source="arxiv",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                fetched_at=datetime.now().isoformat(),
            )
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse arXiv response: {e}")
            return None
    
    # ========================================================================
    # CrossRef API
    # ========================================================================
    
    async def _rate_limit_crossref(self):
        """Enforce CrossRef rate limiting."""
        elapsed = datetime.now() - self._last_crossref_request
        if elapsed < self._crossref_rate_limit:
            await asyncio.sleep((self._crossref_rate_limit - elapsed).total_seconds())
        self._last_crossref_request = datetime.now()
    
    async def resolve_from_crossref(self, doi: str) -> Optional[ResolvedPaper]:
        """
        Resolve paper metadata from CrossRef API.
        
        CrossRef is the authoritative source for DOI metadata.
        API docs: https://api.crossref.org/swagger-ui/index.html
        """
        try:
            import aiohttp
        except ImportError:
            return self._fetch_from_crossref_sync(doi)
        
        await self._rate_limit_crossref()
        
        # Use polite pool with mailto
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        headers = {
            'User-Agent': 'EEG-RAG/1.0 (https://github.com/eeg-rag; mailto:eeg-rag@example.com)'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_crossref_work(data.get("message", {}))
                    elif resp.status == 404:
                        logger.debug(f"DOI not found in CrossRef: {doi}")
        except Exception as e:
            logger.error(f"CrossRef fetch failed for {doi}: {e}")
        
        return None
    
    def _fetch_from_crossref_sync(self, doi: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for CrossRef."""
        import urllib.request
        
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'EEG-RAG/1.0 (https://github.com/eeg-rag; mailto:eeg-rag@example.com)')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return self._parse_crossref_work(data.get("message", {}))
        except Exception as e:
            logger.error(f"CrossRef sync fetch failed for {doi}: {e}")
        return None
    
    def _parse_crossref_work(self, data: Dict[str, Any]) -> Optional[ResolvedPaper]:
        """Parse CrossRef API response."""
        try:
            doi = data.get("DOI", "")
            
            # Title (usually an array)
            title_list = data.get("title", [])
            title = title_list[0] if title_list else ""
            
            # Abstract (may be in HTML)
            abstract = data.get("abstract", "")
            if abstract:
                # Strip HTML tags
                abstract = re.sub(r'<[^>]+>', '', abstract)
            
            # Authors
            authors = []
            for author in data.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                if family:
                    name = f"{family}, {given}" if given else family
                    authors.append(name)
            
            # Year from published-print or published-online
            year = None
            for date_field in ["published-print", "published-online", "created"]:
                date_info = data.get(date_field)
                if date_info and "date-parts" in date_info:
                    parts = date_info["date-parts"]
                    if parts and parts[0] and len(parts[0]) > 0:
                        try:
                            year = int(parts[0][0])
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Journal/container title
            container = data.get("container-title", [])
            journal = container[0] if container else None
            
            # Subject/keywords
            keywords = data.get("subject", [])
            
            # URL
            url = data.get("URL") or f"https://doi.org/{doi}"
            
            return ResolvedPaper(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                keywords=keywords,
                source="crossref",
                url=url,
                citation_count=data.get("is-referenced-by-count", 0),
                fetched_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to parse CrossRef work: {e}")
            return None
    
    # ========================================================================
    # bioRxiv / medRxiv API
    # ========================================================================
    
    async def _rate_limit_biorxiv(self):
        """Enforce bioRxiv rate limiting."""
        elapsed = datetime.now() - self._last_biorxiv_request
        if elapsed < self._biorxiv_rate_limit:
            await asyncio.sleep((self._biorxiv_rate_limit - elapsed).total_seconds())
        self._last_biorxiv_request = datetime.now()
    
    async def resolve_from_biorxiv(self, doi: str) -> Optional[ResolvedPaper]:
        """
        Resolve paper from bioRxiv/medRxiv API.
        
        Works for preprints with DOI prefix 10.1101
        API docs: https://api.biorxiv.org/
        """
        # Determine server from DOI or try both
        servers = ["biorxiv", "medrxiv"]
        
        try:
            import aiohttp
        except ImportError:
            return self._fetch_from_biorxiv_sync(doi)
        
        await self._rate_limit_biorxiv()
        
        for server in servers:
            url = f"https://api.biorxiv.org/details/{server}/{doi}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=30) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            collection = data.get("collection", [])
                            if collection:
                                return self._parse_biorxiv_paper(collection[0], server)
            except Exception as e:
                logger.debug(f"bioRxiv fetch from {server} failed for {doi}: {e}")
                continue
        
        return None
    
    def _fetch_from_biorxiv_sync(self, doi: str) -> Optional[ResolvedPaper]:
        """Synchronous fallback for bioRxiv."""
        import urllib.request
        
        servers = ["biorxiv", "medrxiv"]
        
        for server in servers:
            url = f"https://api.biorxiv.org/details/{server}/{doi}"
            
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    collection = data.get("collection", [])
                    if collection:
                        return self._parse_biorxiv_paper(collection[0], server)
            except Exception:
                continue
        
        return None
    
    def _parse_biorxiv_paper(self, data: Dict[str, Any], server: str) -> Optional[ResolvedPaper]:
        """Parse bioRxiv/medRxiv API response."""
        try:
            doi = data.get("doi", "")
            
            # Authors (string, need to split)
            author_str = data.get("authors", "")
            authors = [a.strip() for a in author_str.split(";") if a.strip()] if author_str else []
            
            # Year from date
            year = None
            date_str = data.get("date", "")
            if date_str and len(date_str) >= 4:
                try:
                    year = int(date_str[:4])
                except ValueError:
                    pass
            
            # Category as keyword
            keywords = []
            category = data.get("category", "")
            if category:
                keywords.append(category)
            
            return ResolvedPaper(
                doi=doi,
                title=data.get("title", "") or "",
                abstract=data.get("abstract", "") or "",
                authors=authors,
                year=year,
                journal=f"{server} preprint",
                keywords=keywords,
                source=server,
                url=f"https://www.{server}.org/content/{doi}",
                pdf_url=f"https://www.{server}.org/content/{doi}.full.pdf",
                fetched_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to parse bioRxiv paper: {e}")
            return None
    
    # ========================================================================
    # Multi-Source Resolution with Fallback
    # ========================================================================
    
    async def resolve_with_fallback(
        self,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        prefer_source: Optional[str] = None
    ) -> Optional[ResolvedPaper]:
        """
        Resolve a paper using multiple sources with fallback.
        
        Tries sources in order based on identifier type and preference.
        Caches the result from the first successful source.
        
        Args:
            doi: DOI identifier
            pmid: PubMed ID
            arxiv_id: arXiv identifier
            prefer_source: Preferred source ('pubmed', 'openalex', 's2', 'crossref', 'arxiv', 'biorxiv')
        
        Returns:
            ResolvedPaper if found, None otherwise
        """
        # Check cache first
        if pmid:
            cached = self.get_from_cache(pmid=pmid)
            if cached:
                return cached
        if doi:
            cached = self.get_from_cache(doi=doi)
            if cached:
                return cached
        
        paper = None
        
        # PMID: Try PubMed first, then S2
        if pmid:
            paper = await self.resolve_pmid(pmid)
            if not paper:
                paper = await self.resolve_from_semantic_scholar(pmid=pmid)
        
        # arXiv: Try arXiv API first, then S2
        elif arxiv_id:
            paper = await self.resolve_arxiv(arxiv_id)
            if not paper:
                paper = await self.resolve_from_semantic_scholar(arxiv_id=arxiv_id)
        
        # DOI: Multiple fallback options
        elif doi:
            # Determine source order based on DOI prefix
            sources = self._get_doi_source_order(doi, prefer_source)
            
            best_paper = None
            for source in sources:
                try:
                    if source == "pubmed":
                        # Try to get PMID from other source first
                        continue
                    elif source == "openalex":
                        paper = await self.resolve_doi(doi)
                    elif source == "s2":
                        paper = await self.resolve_from_semantic_scholar(doi=doi)
                    elif source == "crossref":
                        paper = await self.resolve_from_crossref(doi)
                    elif source == "biorxiv":
                        paper = await self.resolve_from_biorxiv(doi)
                    else:
                        continue
                except Exception as e:
                    logger.debug(f"Source {source} failed for {doi}: {e}")
                    continue
                
                if paper:
                    if paper.abstract:  # Prefer sources with abstract
                        best_paper = paper
                        break
                    elif not best_paper:
                        best_paper = paper  # Keep first result as fallback
            
            paper = best_paper
        
        # Cache the result
        if paper:
            self.save_to_cache(paper)
        
        return paper
    
    def _get_doi_source_order(self, doi: str, prefer_source: Optional[str] = None) -> List[str]:
        """Determine source order based on DOI prefix."""
        # bioRxiv/medRxiv preprints
        if doi.startswith("10.1101/"):
            order = ["biorxiv", "s2", "crossref", "openalex"]
        # arXiv (rare to have DOI)
        elif "arxiv" in doi.lower():
            order = ["s2", "openalex", "crossref"]
        # Default: prefer S2 and OpenAlex for abstracts
        else:
            order = ["openalex", "s2", "crossref"]
        
        # Move preferred source to front
        if prefer_source and prefer_source in order:
            order.remove(prefer_source)
            order.insert(0, prefer_source)
        
        return order
    
    def resolve_with_fallback_sync(
        self,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None
    ) -> Optional[ResolvedPaper]:
        """Synchronous wrapper for resolve_with_fallback."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new event loop for sync call
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.resolve_with_fallback(doi=doi, pmid=pmid, arxiv_id=arxiv_id)
                    )
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(
                    self.resolve_with_fallback(doi=doi, pmid=pmid, arxiv_id=arxiv_id)
                )
        except RuntimeError:
            return asyncio.run(
                self.resolve_with_fallback(doi=doi, pmid=pmid, arxiv_id=arxiv_id)
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics including per-source breakdown."""
        with self._get_cache_connection() as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            stats["cached_papers"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            stats["cached_embeddings"] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            stats["cached_chunks"] = cursor.fetchone()[0]
            
            # Per-source breakdown
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count 
                FROM papers 
                GROUP BY source 
                ORDER BY count DESC
            """)
            stats["by_source"] = {row["source"]: row["count"] for row in cursor.fetchall()}
            
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
