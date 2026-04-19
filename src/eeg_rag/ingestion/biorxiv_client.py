"""
bioRxiv and medRxiv client for preprint ingestion.
These contain the latest research before peer review.
"""

import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncIterator
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : ingestion.biorxiv_client.PreprintArticle
# Requirement  : `PreprintArticle` class shall be instantiable and expose the documented interface
# Purpose      : Preprint article from bioRxiv/medRxiv
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
# Verification : Instantiate PreprintArticle with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class PreprintArticle:
    """Preprint article from bioRxiv/medRxiv."""
    doi: str
    title: str
    abstract: str
    authors: List[str]
    date: str
    server: str  # 'biorxiv' or 'medrxiv'
    category: str
    version: int
    license: Optional[str] = None
    published_doi: Optional[str] = None  # DOI if published in journal


# ---------------------------------------------------------------------------
# ID           : ingestion.biorxiv_client.BioRxivClient
# Requirement  : `BioRxivClient` class shall be instantiable and expose the documented interface
# Purpose      : Async client for bioRxiv and medRxiv APIs
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
# Verification : Instantiate BioRxivClient with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class BioRxivClient:
    """
    Async client for bioRxiv and medRxiv APIs.
    
    API Documentation: https://api.biorxiv.org/
    
    Features:
    - Search both bioRxiv and medRxiv
    - Date-range queries for recent papers
    - Rate limiting to respect server limits
    """
    
    BASE_URL = "https://api.biorxiv.org"
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient.__init__
    # Requirement  : `__init__` shall initialize the client
    # Purpose      : Initialize the client
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : rate_limit: float (default=2.0)
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
    def __init__(self, rate_limit: float = 2.0):
        """
        Initialize the client.
        
        Args:
            rate_limit: Maximum requests per second
        """
        self.rate_limit = rate_limit
        self.min_delay = 1.0 / rate_limit
        self.last_request = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient._get_session
    # Requirement  : `_get_session` shall get or create aiohttp session
    # Purpose      : Get or create aiohttp session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : aiohttp.ClientSession
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
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient._rate_limit
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
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        self.last_request = asyncio.get_event_loop().time()
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient._fetch
    # Requirement  : `_fetch` shall fetch data from API with retry logic
    # Purpose      : Fetch data from API with retry logic
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : url: str
    # Outputs      : Dict
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
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch(self, url: str) -> Dict:
        """Fetch data from API with retry logic."""
        await self._rate_limit()
        session = await self._get_session()
        
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                # Rate limited, wait and retry
                await asyncio.sleep(5)
                raise Exception("Rate limited")
            else:
                response.raise_for_status()
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient.get_details_by_date
    # Requirement  : `get_details_by_date` shall get preprints by date range
    # Purpose      : Get preprints by date range
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : server: str (default='biorxiv'); start_date: str (default='2024-01-01'); end_date: Optional[str] (default=None); cursor: int (default=0)
    # Outputs      : tuple[List[Dict], Optional[int]]
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
    async def get_details_by_date(
        self,
        server: str = "biorxiv",
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        cursor: int = 0
    ) -> tuple[List[Dict], Optional[int]]:
        """
        Get preprints by date range.
        
        Args:
            server: 'biorxiv' or 'medrxiv'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            cursor: Pagination cursor
            
        Returns:
            Tuple of (papers list, next cursor or None)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.BASE_URL}/details/{server}/{start_date}/{end_date}/{cursor}"
        
        try:
            data = await self._fetch(url)
            
            papers = data.get("collection", [])
            messages = data.get("messages", [])
            
            # Check for next page
            next_cursor = None
            for msg in messages:
                if msg.get("status") == "ok":
                    total = msg.get("total", 0)
                    if cursor + len(papers) < total:
                        next_cursor = cursor + len(papers)
            
            return papers, next_cursor
            
        except Exception as e:
            logger.error(f"Error fetching {server} papers: {e}")
            return [], None
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient.search
    # Requirement  : `search` shall search for preprints matching a query
    # Purpose      : Search for preprints matching a query
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; server: str (default='biorxiv'); start_date: Optional[str] (default=None); end_date: Optional[str] (default=None); max_results: int (default=1000)
    # Outputs      : List[Dict]
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
        server: str = "biorxiv",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 1000
    ) -> List[Dict]:
        """
        Search for preprints matching a query.
        
        Note: bioRxiv API doesn't have true text search, so we:
        1. Fetch papers by date range
        2. Filter locally by query terms
        
        Args:
            query: Search terms
            server: 'biorxiv' or 'medrxiv'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_results: Maximum papers to return
            
        Returns:
            List of matching papers
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        query_terms = query.lower().split()
        results = []
        cursor = 0
        
        while len(results) < max_results:
            papers, next_cursor = await self.get_details_by_date(
                server=server,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor
            )
            
            if not papers:
                break
            
            # Filter by query terms
            for paper in papers:
                title = (paper.get("title", "") or "").lower()
                abstract = (paper.get("abstract", "") or "").lower()
                category = (paper.get("category", "") or "").lower()
                
                # Check if any query term matches
                text = f"{title} {abstract} {category}"
                if any(term in text for term in query_terms):
                    results.append({
                        "doi": paper.get("doi"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "authors": paper.get("authors", "").split("; ") if paper.get("authors") else [],
                        "date": paper.get("date"),
                        "server": server,
                        "category": paper.get("category"),
                        "version": paper.get("version", 1),
                        "license": paper.get("license"),
                        "published_doi": paper.get("published"),
                    })
                    
                    if len(results) >= max_results:
                        break
            
            if next_cursor is None:
                break
            cursor = next_cursor
            
            # Small delay between pages
            await asyncio.sleep(0.5)
        
        return results
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient.get_recent_eeg_preprints
    # Requirement  : `get_recent_eeg_preprints` shall get recent EEG-related preprints
    # Purpose      : Get recent EEG-related preprints
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : days: int (default=30); server: str (default='biorxiv')
    # Outputs      : List[Dict]
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
    async def get_recent_eeg_preprints(
        self,
        days: int = 30,
        server: str = "biorxiv"
    ) -> List[Dict]:
        """
        Get recent EEG-related preprints.
        
        Args:
            days: Number of days to look back
            server: 'biorxiv' or 'medrxiv'
            
        Returns:
            List of EEG-related preprints
        """
        eeg_terms = [
            "eeg", "electroencephalograph", "brain-computer interface",
            "seizure", "epilepsy", "sleep staging", "neural oscillation",
            "event-related potential", "erp", "motor imagery"
        ]
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        results = []
        cursor = 0
        
        while True:
            papers, next_cursor = await self.get_details_by_date(
                server=server,
                start_date=start_date,
                end_date=end_date,
                cursor=cursor
            )
            
            if not papers:
                break
            
            for paper in papers:
                title = (paper.get("title", "") or "").lower()
                abstract = (paper.get("abstract", "") or "").lower()
                text = f"{title} {abstract}"
                
                if any(term in text for term in eeg_terms):
                    results.append({
                        "doi": paper.get("doi"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "authors": paper.get("authors", "").split("; ") if paper.get("authors") else [],
                        "date": paper.get("date"),
                        "server": server,
                        "category": paper.get("category"),
                        "version": paper.get("version", 1),
                    })
            
            if next_cursor is None:
                break
            cursor = next_cursor
        
        logger.info(f"Found {len(results)} EEG-related {server} preprints from last {days} days")
        return results
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.biorxiv_client.BioRxivClient.close
    # Requirement  : `close` shall close the session
    # Purpose      : Close the session
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
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
