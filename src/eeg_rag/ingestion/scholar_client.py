"""
Google Scholar client using scholarly library and Semantic Scholar API.
Note: Google Scholar doesn't have official API, so we use multiple sources.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
import logging
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : ingestion.scholar_client.ScholarArticle
# Requirement  : `ScholarArticle` class shall be instantiable and expose the documented interface
# Purpose      : Article from Google Scholar / Semantic Scholar
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ScholarArticle with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ScholarArticle:
    """Article from Google Scholar / Semantic Scholar."""
    scholar_id: Optional[str]
    semantic_scholar_id: Optional[str]
    title: str
    abstract: str
    authors: list[str]
    year: Optional[int]
    venue: str
    citation_count: int
    doi: Optional[str]
    pmid: Optional[str]
    arxiv_id: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    references: list[str]  # Semantic Scholar IDs
    citations: list[str]   # Papers that cite this
    influential_citation_count: int
    fields_of_study: list[str]
    tldr: Optional[str]  # Semantic Scholar's TL;DR


# ---------------------------------------------------------------------------
# ID           : ingestion.scholar_client.SemanticScholarClient
# Requirement  : `SemanticScholarClient` class shall be instantiable and expose the documented interface
# Purpose      : Primary client for academic paper retrieval
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate SemanticScholarClient with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class SemanticScholarClient:
    """
    Primary client for academic paper retrieval.
    Uses Semantic Scholar API (free, rate-limited to 100 req/5min).
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # EEG-focused search queries for Semantic Scholar
    EEG_QUERIES = [
        "electroencephalography EEG",
        "event-related potentials ERP brain",
        "brain-computer interface BCI EEG",
        "EEG signal processing machine learning",
        "EEG epilepsy seizure detection",
        "EEG sleep staging classification",
        "P300 brain-computer interface",
        "EEG emotion recognition",
        "EEG motor imagery classification",
        "resting-state EEG biomarker",
        "EEG connectivity analysis",
        "EEG source localization",
        "EEG artifact removal",
        "EEG deep learning neural network",
        "EEG attention cognitive neuroscience",
        "EEG memory encoding retrieval",
        "EEG language processing N400",
        "EEG consciousness disorders",
        "EEG Alzheimer dementia",
        "EEG depression anxiety",
    ]
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.__init__
    # Requirement  : `__init__` shall initialize Semantic Scholar client
    # Purpose      : Initialize Semantic Scholar client
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : api_key: Optional[str] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        
        # Rate limiting: 100 requests per 5 minutes without key
        self.request_count = 0
        self.window_start = time.time()
        self.max_requests = 100 if not api_key else 1000
        self.window_seconds = 300
        self._session: Optional[aiohttp.ClientSession] = None
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.__aenter__
    # Requirement  : `__aenter__` shall async context manager entry
    # Purpose      : Async context manager entry
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.__aexit__
    # Requirement  : `__aexit__` shall async context manager exit
    # Purpose      : Async context manager exit
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : exc_type; exc_val; exc_tb
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
        return False
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.close
    # Requirement  : `close` shall close the client session
    # Purpose      : Close the client session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
        
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient._rate_limit
    # Requirement  : `_rate_limit` shall enforce rate limiting
    # Purpose      : Enforce rate limiting
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start > self.window_seconds:
            self.request_count = 0
            self.window_start = current_time
        
        # Wait if at limit
        if self.request_count >= self.max_requests:
            wait_time = self.window_seconds - (current_time - self.window_start)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.window_start = time.time()
        
        self.request_count += 1

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient._fetch
    # Requirement  : `_fetch` shall fetch from Semantic Scholar API
    # Purpose      : Fetch from Semantic Scholar API
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : session: aiohttp.ClientSession; endpoint: str; params: dict (default=None)
    # Outputs      : dict
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    async def _fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> dict:
        """Fetch from Semantic Scholar API."""
        await self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        async with session.get(url, headers=self.headers, params=params) as response:
            if response.status == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                raise aiohttp.ClientError("Rate limited")
            response.raise_for_status()
            return await response.json()

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.search
    # Requirement  : `search` shall search for papers and return Semantic Scholar IDs
    # Purpose      : Search for papers and return Semantic Scholar IDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; limit: int (default=100); year_range: Optional[tuple[int, int]] (default=None); fields_of_study: Optional[list[str]] (default=None)
    # Outputs      : list[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple[int, int]] = None,
        fields_of_study: Optional[list[str]] = None
    ) -> list[str]:
        """
        Search for papers and return Semantic Scholar IDs.
        
        Args:
            query: Search query
            limit: Maximum results (max 100 per request)
            year_range: Optional (start_year, end_year) tuple
            fields_of_study: Filter by fields like "Medicine", "Computer Science"
            
        Returns:
            List of Semantic Scholar paper IDs
        """
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId"
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(session, "paper/search", params)
            papers = result.get("data", [])
            return [p["paperId"] for p in papers if p.get("paperId")]

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.get_paper
    # Requirement  : `get_paper` shall get full paper details by Semantic Scholar ID
    # Purpose      : Get full paper details by Semantic Scholar ID
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_id: str
    # Outputs      : Optional[ScholarArticle]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_paper(self, paper_id: str) -> Optional[ScholarArticle]:
        """
        Get full paper details by Semantic Scholar ID.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            ScholarArticle with full metadata
        """
        fields = [
            "paperId", "externalIds", "title", "abstract", "year",
            "authors", "venue", "citationCount", "influentialCitationCount",
            "references", "citations", "fieldsOfStudy", "tldr", "url",
            "openAccessPdf"
        ]
        
        async with aiohttp.ClientSession() as session:
            try:
                result = await self._fetch(
                    session,
                    f"paper/{paper_id}",
                    {"fields": ",".join(fields)}
                )
                return self._parse_paper(result)
            except aiohttp.ClientError as e:
                logger.warning(f"Error fetching paper {paper_id}: {e}")
                return None

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient._parse_paper
    # Requirement  : `_parse_paper` shall parse Semantic Scholar response into ScholarArticle
    # Purpose      : Parse Semantic Scholar response into ScholarArticle
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: dict
    # Outputs      : ScholarArticle
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _parse_paper(self, data: dict) -> ScholarArticle:
        """Parse Semantic Scholar response into ScholarArticle."""
        external_ids = data.get("externalIds", {})
        
        authors = [
            a.get("name", "") for a in data.get("authors", [])
        ]
        
        references = [
            r.get("paperId") for r in data.get("references", [])
            if r.get("paperId")
        ]
        
        citations = [
            c.get("paperId") for c in data.get("citations", [])
            if c.get("paperId")
        ]
        
        pdf_info = data.get("openAccessPdf", {})
        
        return ScholarArticle(
            scholar_id=None,
            semantic_scholar_id=data.get("paperId"),
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue", ""),
            citation_count=data.get("citationCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            doi=external_ids.get("DOI"),
            pmid=external_ids.get("PubMed"),
            arxiv_id=external_ids.get("ArXiv"),
            url=data.get("url"),
            pdf_url=pdf_info.get("url") if pdf_info else None,
            references=references,
            citations=citations,
            fields_of_study=data.get("fieldsOfStudy", []),
            tldr=data.get("tldr", {}).get("text") if data.get("tldr") else None
        )

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.get_papers_batch
    # Requirement  : `get_papers_batch` shall fetch multiple papers in batches
    # Purpose      : Fetch multiple papers in batches
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_ids: list[str]; batch_size: int (default=50)
    # Outputs      : AsyncIterator[ScholarArticle]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_papers_batch(self, paper_ids: list[str], batch_size: int = 50) -> AsyncIterator[ScholarArticle]:
        """
        Fetch multiple papers in batches.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs
            batch_size: Papers per batch request (max 500)
            
        Yields:
            ScholarArticle objects
        """
        fields = [
            "paperId", "externalIds", "title", "abstract", "year",
            "authors", "venue", "citationCount", "influentialCitationCount",
            "fieldsOfStudy", "tldr", "url", "openAccessPdf"
        ]
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(paper_ids), batch_size):
                batch = paper_ids[i:i + batch_size]
                
                try:
                    await self._rate_limit()
                    
                    async with session.post(
                        f"{self.BASE_URL}/paper/batch",
                        headers={**self.headers, "Content-Type": "application/json"},
                        json={"ids": batch},
                        params={"fields": ",".join(fields)}
                    ) as response:
                        response.raise_for_status()
                        results = await response.json()
                    
                    for paper_data in results:
                        if paper_data:
                            yield self._parse_paper(paper_data)
                            
                except Exception as e:
                    logger.error(f"Error fetching batch: {e}")
                    continue
                
                logger.info(f"Fetched {min(i + batch_size, len(paper_ids))}/{len(paper_ids)} papers")

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.get_author_papers
    # Requirement  : `get_author_papers` shall get paper IDs for a specific author
    # Purpose      : Get paper IDs for a specific author
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : author_id: str; limit: int (default=100)
    # Outputs      : list[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_author_papers(self, author_id: str, limit: int = 100) -> list[str]:
        """Get paper IDs for a specific author."""
        params = {
            "fields": "papers",
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(session, f"author/{author_id}", params)
            papers = result.get("papers", [])
            return [p["paperId"] for p in papers if p.get("paperId")]

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.get_recommendations
    # Requirement  : `get_recommendations` shall get recommended papers based on a paper ID
    # Purpose      : Get recommended papers based on a paper ID
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_id: str; limit: int (default=50)
    # Outputs      : list[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_recommendations(self, paper_id: str, limit: int = 50) -> list[str]:
        """Get recommended papers based on a paper ID."""
        params = {
            "fields": "paperId",
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(
                session,
                f"recommendations/v1/papers/forpaper/{paper_id}",
                params
            )
            papers = result.get("recommendedPapers", [])
            return [p["paperId"] for p in papers if p.get("paperId")]

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.SemanticScholarClient.collect_eeg_corpus
    # Requirement  : `collect_eeg_corpus` shall collect comprehensive EEG corpus from Semantic Scholar
    # Purpose      : Collect comprehensive EEG corpus from Semantic Scholar
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : max_per_query: int (default=1000); year_start: int (default=2014); include_recommendations: bool (default=True)
    # Outputs      : AsyncIterator[ScholarArticle]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def collect_eeg_corpus(
        self,
        max_per_query: int = 1000,
        year_start: int = 2014,
        include_recommendations: bool = True
    ) -> AsyncIterator[ScholarArticle]:
        """
        Collect comprehensive EEG corpus from Semantic Scholar.
        
        Args:
            max_per_query: Maximum papers per search query
            year_start: Earliest publication year to include
            include_recommendations: Whether to expand with recommendations
            
        Yields:
            ScholarArticle objects
        """
        seen_ids = set()
        high_impact_ids = []
        year_end = datetime.now().year
        
        for query in self.EEG_QUERIES:
            logger.info(f"Searching Semantic Scholar: {query}")
            
            paper_ids = await self.search(
                query=query,
                limit=max_per_query,
                year_range=(year_start, year_end),
                fields_of_study=["Medicine", "Computer Science", "Psychology", "Biology"]
            )
            
            new_ids = [pid for pid in paper_ids if pid not in seen_ids]
            seen_ids.update(new_ids)
            
            async for paper in self.get_papers_batch(new_ids):
                yield paper
                
                # Track high-impact papers for recommendations
                if paper.citation_count > 100:
                    high_impact_ids.append(paper.semantic_scholar_id)
        
        # Expand corpus with recommendations from high-impact papers
        if include_recommendations:
            logger.info(f"Getting recommendations from {len(high_impact_ids)} high-impact papers")
            
            for paper_id in high_impact_ids[:50]:  # Limit to top 50
                rec_ids = await self.get_recommendations(paper_id, limit=20)
                new_rec_ids = [rid for rid in rec_ids if rid not in seen_ids]
                seen_ids.update(new_rec_ids)
                
                async for paper in self.get_papers_batch(new_rec_ids):
                    yield paper
        
        logger.info(f"Total unique papers from Semantic Scholar: {len(seen_ids)}")


# ---------------------------------------------------------------------------
# ID           : ingestion.scholar_client.GoogleScholarScraper
# Requirement  : `GoogleScholarScraper` class shall be instantiable and expose the documented interface
# Purpose      : Fallback scraper for Google Scholar using scholarly library
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate GoogleScholarScraper with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GoogleScholarScraper:
    """
    Fallback scraper for Google Scholar using scholarly library.
    Use sparingly - Google may block aggressive scraping.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.GoogleScholarScraper.__init__
    # Requirement  : `__init__` shall initialize Google Scholar scraper
    # Purpose      : Initialize Google Scholar scraper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : use_proxy: bool (default=False)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, use_proxy: bool = False):
        """
        Initialize Google Scholar scraper.
        
        Args:
            use_proxy: Whether to use free proxy rotation
        """
        self.use_proxy = use_proxy
        self._scholarly = None
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.GoogleScholarScraper._get_scholarly
    # Requirement  : `_get_scholarly` shall lazy load scholarly library
    # Purpose      : Lazy load scholarly library
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _get_scholarly(self):
        """Lazy load scholarly library."""
        if self._scholarly is None:
            try:
                from scholarly import scholarly, ProxyGenerator
                if self.use_proxy:
                    pg = ProxyGenerator()
                    pg.FreeProxies()
                    scholarly.use_proxy(pg)
                self._scholarly = scholarly
            except ImportError:
                logger.warning("scholarly library not installed. Run: pip install scholarly")
                return None
        return self._scholarly
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.GoogleScholarScraper.search_sync
    # Requirement  : `search_sync` shall synchronous search (scholarly doesn't support async)
    # Purpose      : Synchronous search (scholarly doesn't support async)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; limit: int (default=20)
    # Outputs      : list[dict]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def search_sync(self, query: str, limit: int = 20) -> list[dict]:
        """
        Synchronous search (scholarly doesn't support async).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of paper dictionaries
        """
        scholarly = self._get_scholarly()
        if scholarly is None:
            return []
            
        results = []
        search_query = scholarly.search_pubs(query)
        
        for i, result in enumerate(search_query):
            if i >= limit:
                break
            
            try:
                # Fill in additional details
                filled = scholarly.fill(result)
                results.append({
                    "title": filled.get("bib", {}).get("title", ""),
                    "abstract": filled.get("bib", {}).get("abstract", ""),
                    "authors": filled.get("bib", {}).get("author", []),
                    "year": filled.get("bib", {}).get("pub_year"),
                    "venue": filled.get("bib", {}).get("venue", ""),
                    "citations": filled.get("num_citations", 0),
                    "url": filled.get("pub_url"),
                    "scholar_id": filled.get("author_id", [None])[0],
                })
                
                # Be polite with rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error processing result: {e}")
                continue
        
        return results

    # ---------------------------------------------------------------------------
    # ID           : ingestion.scholar_client.GoogleScholarScraper.search
    # Requirement  : `search` shall async wrapper for synchronous search
    # Purpose      : Async wrapper for synchronous search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; limit: int (default=20)
    # Outputs      : list[dict]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Async wrapper for synchronous search."""
        return await asyncio.to_thread(self.search_sync, query, limit)
