"""
# =============================================================================
# ID:             MOD-PUBMED-001
# Requirement:    REQ-PUBMED-010 — Full PubMed E-utilities integration;
#                 REQ-PUBMED-011 — Citation network analysis;
#                 REQ-PUBMED-012 — Intelligent MeSH-based query expansion;
#                 REQ-PUBMED-013 — Batch paper fetching with pagination.
# Purpose:        Retrieve EEG-relevant scientific papers from NCBI PubMed
#                 using E-utilities REST API with MeSH term expansion,
#                 citation crawling, and smart query construction.
# Rationale:      PubMed is the authoritative source for biomedical literature.
#                 MeSH expansion ensures consistent vocabulary mapping across
#                 EEG research queries (e.g., "seizure" → MeSH: C23.888.592.742).
#                 Batch fetching with rate limiting respects NCBI terms of service.
# Inputs:         AgentQuery.text (natural language); AgentQuery.parameters
#                 (max_results, use_mesh, date_range, article_types).
# Outputs:        AgentResult.data = {"papers": [PubMedPaper.to_dict()...],
#                 "total_count": int, "returned_count": int,
#                 "query_used": str, "mesh_suggestions": List[str]}.
# Preconditions:  NCBI E-utilities accessible; valid email provided for
#                 NCBI identification (required by NCBI TOS).
# Postconditions: Papers cached for cache_ttl (6 hours); rate limit state updated.
# Assumptions:    Up to 10,000 PMIDs per search result page; batch size 200.
# Side Effects:   HTTP GET to NCBI (rate-limited: 3/s no-key, 10/s with key);
#                 in-memory cache growth bounded by search history.
# Failure Modes:  HTTP 429 → wait and retry; timeout → return empty result;
#                 XML parse → log warning, skip malformed records.
# Error Handling: All exceptions caught in execute(); return AgentResult(success=False).
# Constraints:    Rate limit: 3 req/s (free), 10 req/s (API key);
#                 Timeout: 30s per request; batch: 200 PMIDs per efetch call.
# Verification:   tests/test_web_agent.py; tests/test_search_validation.py.
# References:     NCBI E-utilities documentation; MeSH vocabulary 2025;
#                 REQ-PUBMED-010–013.
# =============================================================================
Enhanced PubMed Agent

Comprehensive PubMed integration with MeSH expansion, citation crawling,
and smart query building.

Requirements Covered:
- REQ-PUBMED-010: Full PubMed E-utilities integration
- REQ-PUBMED-011: Citation network analysis
- REQ-PUBMED-012: Intelligent query expansion
- REQ-PUBMED-013: Batch paper fetching
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import aiohttp

from eeg_rag.agents.base_agent import (
    BaseAgent, AgentType, AgentResult, AgentQuery
)
from .mesh_expander import MeSHExpander
from .citation_crawler import CitationCrawler
from .query_builder import PubMedQueryBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : agents.pubmed_agent.pubmed_agent.PubMedPaper
# Requirement  : `PubMedPaper` class shall be instantiable and expose the documented interface
# Purpose      : Structured PubMed paper data
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
# Verification : Instantiate PubMedPaper with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class PubMedPaper:
    """Structured PubMed paper data."""
    pmid: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    pmc_id: Optional[str] = None

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedPaper.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "publication_types": self.publication_types,
            "affiliations": self.affiliations,
            "pmc_id": self.pmc_id,
            "source": "pubmed"
        }


# ---------------------------------------------------------------------------
# ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent
# Requirement  : `PubMedAgent` class shall be instantiable and expose the documented interface
# Purpose      : Enhanced PubMed agent with:
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
# Verification : Instantiate PubMedAgent with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class PubMedAgent(BaseAgent):
    """
    Enhanced PubMed agent with:
    - MeSH term expansion
    - Smart query building
    - Citation network traversal
    - Related articles discovery
    - Batch fetching with pagination
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.__init__
    # Requirement  : `__init__` shall initialize PubMed agent
    # Purpose      : Initialize PubMed agent
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str (default='PubMedAgent'); api_key: Optional[str] (default=None); email: str (default='researcher@example.com'); tool: str (default='eeg-rag'); config: Optional[Dict[str, Any]] (default=None)
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
    def __init__(
        self,
        name: str = "PubMedAgent",
        api_key: Optional[str] = None,
        email: str = "researcher@example.com",
        tool: str = "eeg-rag",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PubMed agent.

        Args:
            name: Agent name
            api_key: NCBI API key for higher rate limits
            email: Email for NCBI identification
            tool: Tool name for NCBI identification
            config: Optional configuration
        """
        super().__init__(
            agent_type=AgentType.WEB_SEARCH,
            name=name,
            config=config or {}
        )

        self.api_key = api_key
        self.email = email
        self.tool = tool

        # Rate limit: 3/sec without key, 10/sec with key
        self.requests_per_second = 10.0 if api_key else 3.0
        self.min_request_interval = 1.0 / self.requests_per_second
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()

        # Initialize components
        self.mesh_expander = MeSHExpander()
        self.query_builder = PubMedQueryBuilder(self.mesh_expander)

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False

        # Citation crawler (initialized lazily)
        self._citation_crawler: Optional[CitationCrawler] = None

        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(hours=6)

        # Statistics
        self.total_searches = 0
        self.total_papers_fetched = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"PubMedAgent initialized (api_key={'yes' if api_key else 'no'})")

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._get_session
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
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._get_citation_crawler
    # Requirement  : `_get_citation_crawler` shall get or create citation crawler
    # Purpose      : Get or create citation crawler
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : CitationCrawler
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
    async def _get_citation_crawler(self) -> CitationCrawler:
        """Get or create citation crawler."""
        if self._citation_crawler is None:
            session = await self._get_session()
            self._citation_crawler = CitationCrawler(
                session=session,
                api_key=self.api_key,
                email=self.email
            )
        return self._citation_crawler

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.close
    # Requirement  : `close` shall close HTTP session
    # Purpose      : Close HTTP session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
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
    async def close(self) -> None:
        """Close HTTP session."""
        if self._citation_crawler:
            await self._citation_crawler.close()
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._rate_limit
    # Requirement  : `_rate_limit` shall enforce rate limiting
    # Purpose      : Enforce rate limiting
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
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
    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._request_lock:
            import time
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self._last_request_time = time.time()

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._build_params
    # Requirement  : `_build_params` shall add common parameters to request
    # Purpose      : Add common parameters to request
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : params: Dict[str, Any]
    # Outputs      : Dict[str, Any]
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
    def _build_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add common parameters to request."""
        params["email"] = self.email
        params["tool"] = self.tool
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._get_cache
    # Requirement  : `_get_cache` shall get value from cache if not expired
    # Purpose      : Get value from cache if not expired
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str
    # Outputs      : Optional[Any]
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
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                self.cache_hits += 1
                return value
            del self._cache[key]
        self.cache_misses += 1
        return None

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._set_cache
    # Requirement  : `_set_cache` shall set value in cache
    # Purpose      : Set value in cache
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; value: Any
    # Outputs      : None
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
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (value, datetime.now())

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.execute
    # Requirement  : `execute` shall execute PubMed search
    # Purpose      : Execute PubMed search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: AgentQuery
    # Outputs      : AgentResult
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
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute PubMed search.

        Args:
            query: Search query

        Returns:
            AgentResult with papers
        """
        start_time = datetime.now()

        try:
            # Extract parameters
            params = query.parameters
            max_results = params.get("max_results", 50)
            use_mesh = params.get("use_mesh", True)
            date_range = params.get("date_range")
            article_types = params.get("article_types")

            # Check cache
            cache_key = f"search:{query.text}:{max_results}:{use_mesh}:{date_range}"
            cached = self._get_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for query: {query.text[:50]}")
                elapsed = (datetime.now() - start_time).total_seconds()
                return AgentResult(
                    success=True,
                    data=cached,
                    metadata={"source": "cache", "cache_hit": True},
                    agent_type=AgentType.WEB_SEARCH,
                    elapsed_time=elapsed
                )

            # Build optimized query
            pubmed_query = self.query_builder.build_query(
                query.text,
                use_mesh=use_mesh,
                date_range=date_range,
                article_types=article_types
            )

            logger.info(f"PubMed query: {pubmed_query[:100]}...")

            # Search for PMIDs
            search_result = await self._search(pubmed_query, max_results)
            pmids = search_result.get("pmids", [])
            total_count = search_result.get("total_count", 0)

            if not pmids:
                elapsed = (datetime.now() - start_time).total_seconds()
                return AgentResult(
                    success=True,
                    data={
                        "papers": [],
                        "total_count": 0,
                        "query_used": pubmed_query,
                        "mesh_suggestions": self.mesh_expander.get_mesh_suggestions(query.text)
                    },
                    agent_type=AgentType.WEB_SEARCH,
                    elapsed_time=elapsed
                )

            # Fetch paper details
            papers = await self._fetch_papers(pmids)

            # Build result
            result_data = {
                "papers": [p.to_dict() for p in papers],
                "total_count": total_count,
                "returned_count": len(papers),
                "query_used": pubmed_query,
                "mesh_suggestions": self.mesh_expander.get_mesh_suggestions(query.text)
            }

            # Cache result
            self._set_cache(cache_key, result_data)

            self.total_searches += 1
            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"PubMed search completed: {len(papers)} papers in {elapsed:.2f}s")

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "source": "pubmed",
                    "query_expanded": use_mesh
                },
                agent_type=AgentType.WEB_SEARCH,
                elapsed_time=elapsed
            )

        except Exception as e:
            logger.exception(f"PubMed search failed: {e}")
            elapsed = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.WEB_SEARCH,
                elapsed_time=elapsed
            )

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._search
    # Requirement  : `_search` shall execute ESearch to get PMIDs
    # Purpose      : Execute ESearch to get PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; max_results: int (default=50)
    # Outputs      : Dict[str, Any]
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
    async def _search(
        self,
        query: str,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Execute ESearch to get PMIDs.

        Args:
            query: PubMed query string
            max_results: Maximum results to return

        Returns:
            Dictionary with pmids and total_count
        """
        await self._rate_limit()
        session = await self._get_session()

        params = self._build_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "usehistory": "y"
        })

        try:
            async with session.get(
                f"{self.EUTILS_BASE}/esearch.fcgi",
                params=params
            ) as response:
                if response.status != 200:
                    logger.error(f"ESearch failed: {response.status}")
                    return {"pmids": [], "total_count": 0}

                data = await response.json()
                result = data.get("esearchresult", {})

                pmids = result.get("idlist", [])
                total_count = int(result.get("count", 0))

                logger.debug(f"ESearch found {total_count} results, returning {len(pmids)}")

                return {
                    "pmids": pmids,
                    "total_count": total_count,
                    "web_env": result.get("webenv"),
                    "query_key": result.get("querykey")
                }

        except Exception as e:
            logger.error(f"ESearch error: {e}")
            return {"pmids": [], "total_count": 0}

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._fetch_papers
    # Requirement  : `_fetch_papers` shall fetch paper details for PMIDs
    # Purpose      : Fetch paper details for PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmids: List[str]; batch_size: int (default=50)
    # Outputs      : List[PubMedPaper]
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
    async def _fetch_papers(
        self,
        pmids: List[str],
        batch_size: int = 50
    ) -> List[PubMedPaper]:
        """
        Fetch paper details for PMIDs.

        Args:
            pmids: List of PubMed IDs
            batch_size: Papers to fetch per request

        Returns:
            List of PubMedPaper objects
        """
        all_papers = []
        session = await self._get_session()

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]

            await self._rate_limit()

            params = self._build_params({
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml"
            })

            try:
                async with session.get(
                    f"{self.EUTILS_BASE}/efetch.fcgi",
                    params=params
                ) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        papers = self._parse_xml(xml_content)
                        all_papers.extend(papers)
                        self.total_papers_fetched += len(papers)
                    else:
                        logger.error(f"EFetch failed for batch: {response.status}")

            except Exception as e:
                logger.error(f"EFetch error for batch: {e}")

        return all_papers

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._parse_xml
    # Requirement  : `_parse_xml` shall parse PubMed XML response
    # Purpose      : Parse PubMed XML response
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : xml_content: str
    # Outputs      : List[PubMedPaper]
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
    def _parse_xml(self, xml_content: str) -> List[PubMedPaper]:
        """Parse PubMed XML response."""
        papers = []

        try:
            root = ElementTree.fromstring(xml_content)

            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")
                    continue

        except ElementTree.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return papers

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent._parse_article
    # Requirement  : `_parse_article` shall parse a single PubmedArticle element
    # Purpose      : Parse a single PubmedArticle element
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : article: ElementTree.Element
    # Outputs      : Optional[PubMedPaper]
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
    def _parse_article(self, article: ElementTree.Element) -> Optional[PubMedPaper]:
        """Parse a single PubmedArticle element."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""
        if not pmid:
            return None

        # Article info
        article_elem = medline.find(".//Article")
        if article_elem is None:
            return None

        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract
        abstract_parts = []
        for abstract_text in article_elem.findall(".//AbstractText"):
            if abstract_text.text:
                label = abstract_text.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {abstract_text.text}")
                else:
                    abstract_parts.append(abstract_text.text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last = author.find("LastName")
            first = author.find("ForeName")
            if last is not None:
                name = last.text
                if first is not None:
                    name = f"{last.text}, {first.text}"
                authors.append(name)

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Year
        year = None
        pub_date = article_elem.find(".//PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # PMC ID
        pmc_id = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                pmc_id = article_id.text
                break

        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        # Keywords
        keywords = []
        for keyword in medline.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

        # Publication types
        pub_types = []
        for pub_type in article_elem.findall(".//PublicationType"):
            if pub_type.text:
                pub_types.append(pub_type.text)

        # Affiliations
        affiliations = []
        for affiliation in article_elem.findall(".//AffiliationInfo/Affiliation"):
            if affiliation.text:
                affiliations.append(affiliation.text)

        return PubMedPaper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            doi=doi,
            mesh_terms=mesh_terms,
            keywords=keywords,
            publication_types=pub_types,
            affiliations=affiliations,
            pmc_id=pmc_id
        )

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.get_citation_network
    # Requirement  : `get_citation_network` shall get citation network for a paper
    # Purpose      : Get citation network for a paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; direction: str (default='both')
    # Outputs      : Dict[str, Any]
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
    async def get_citation_network(
        self,
        pmid: str,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Get citation network for a paper.

        Args:
            pmid: PubMed ID
            direction: "citing", "references", or "both"

        Returns:
            Citation network data
        """
        crawler = await self._get_citation_crawler()
        return await crawler.get_full_citation_network(pmid, direction=direction)

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.get_similar_papers
    # Requirement  : `get_similar_papers` shall get papers similar to a given paper
    # Purpose      : Get papers similar to a given paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; max_results: int (default=10)
    # Outputs      : List[PubMedPaper]
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
    async def get_similar_papers(
        self,
        pmid: str,
        max_results: int = 10
    ) -> List[PubMedPaper]:
        """
        Get papers similar to a given paper.

        Args:
            pmid: PubMed ID
            max_results: Maximum similar papers to return

        Returns:
            List of similar papers
        """
        crawler = await self._get_citation_crawler()
        similar_pmids = await crawler.get_similar_papers(pmid, max_results)

        if similar_pmids:
            return await self._fetch_papers(similar_pmids)
        return []

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.fetch_by_pmids
    # Requirement  : `fetch_by_pmids` shall fetch papers by their PMIDs
    # Purpose      : Fetch papers by their PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmids: List[str]
    # Outputs      : List[PubMedPaper]
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
    async def fetch_by_pmids(self, pmids: List[str]) -> List[PubMedPaper]:
        """
        Fetch papers by their PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of papers
        """
        return await self._fetch_papers(pmids)

    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.pubmed_agent.PubMedAgent.get_statistics
    # Requirement  : `get_statistics` shall get agent statistics
    # Purpose      : Get agent statistics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        base_stats = super().get_statistics()
        base_stats.update({
            "total_searches": self.total_searches,
            "total_papers_fetched": self.total_papers_fetched,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        })
        return base_stats
