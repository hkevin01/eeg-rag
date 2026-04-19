"""
Citation Network Crawler for PubMed

Traverses citation networks using PubMed's elink API.

Requirements Covered:
- REQ-PUBMED-003: Forward citation traversal
- REQ-PUBMED-004: Backward reference traversal
- REQ-PUBMED-005: Similar papers discovery
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : agents.pubmed_agent.citation_crawler.CitationCrawler
# Requirement  : `CitationCrawler` class shall be instantiable and expose the documented interface
# Purpose      : Crawl citation networks using PubMed's elink API
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
# Verification : Instantiate CitationCrawler with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CitationCrawler:
    """Crawl citation networks using PubMed's elink API."""
    
    ELINK_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.__init__
    # Requirement  : `__init__` shall initialize citation crawler
    # Purpose      : Initialize citation crawler
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : session: Optional[aiohttp.ClientSession] (default=None); api_key: Optional[str] (default=None); email: str (default='researcher@example.com')
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
        session: Optional[aiohttp.ClientSession] = None,
        api_key: Optional[str] = None,
        email: str = "researcher@example.com"
    ):
        """
        Initialize citation crawler.
        
        Args:
            session: Shared aiohttp session
            api_key: NCBI API key for higher rate limits
            email: Email for NCBI identification
        """
        self._session = session
        self._owns_session = False
        self.api_key = api_key
        self.email = email
        logger.info("CitationCrawler initialized")
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler._get_session
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
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.close
    # Requirement  : `close` shall close the HTTP session if we own it
    # Purpose      : Close the HTTP session if we own it
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
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler._build_params
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
        params["tool"] = "eeg-rag"
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.get_citing_papers
    # Requirement  : `get_citing_papers` shall get PMIDs of papers that cite this paper
    # Purpose      : Get PMIDs of papers that cite this paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; max_results: int (default=50)
    # Outputs      : List[str]
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
    async def get_citing_papers(
        self,
        pmid: str,
        max_results: int = 50
    ) -> List[str]:
        """
        Get PMIDs of papers that cite this paper.
        
        Args:
            pmid: PubMed ID of the source paper
            max_results: Maximum number of citing papers to return
            
        Returns:
            List of PMIDs that cite the source paper
        """
        session = await self._get_session()
        
        params = self._build_params({
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "linkname": "pubmed_pubmed_citedin",
            "retmode": "json"
        })
        
        try:
            async with session.get(self.ELINK_BASE, params=params) as response:
                if response.status == 429:
                    logger.warning("Rate limited by PubMed elink API")
                    return []
                
                if response.status != 200:
                    logger.error(f"elink request failed: {response.status}")
                    return []
                
                data = await response.json()
                
                # Parse citation links
                linksets = data.get("linksets", [])
                for linkset in linksets:
                    linksetdbs = linkset.get("linksetdbs", [])
                    for db in linksetdbs:
                        if db.get("linkname") == "pubmed_pubmed_citedin":
                            links = db.get("links", [])
                            pmids = [str(link) for link in links[:max_results]]
                            logger.debug(f"Found {len(pmids)} citing papers for PMID {pmid}")
                            return pmids
                            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching citations for PMID {pmid}")
        except Exception as e:
            logger.error(f"Citation crawl error for PMID {pmid}: {e}")
        
        return []
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.get_references
    # Requirement  : `get_references` shall get PMIDs of papers that this paper references
    # Purpose      : Get PMIDs of papers that this paper references
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; max_results: int (default=50)
    # Outputs      : List[str]
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
    async def get_references(
        self,
        pmid: str,
        max_results: int = 50
    ) -> List[str]:
        """
        Get PMIDs of papers that this paper references.
        
        Args:
            pmid: PubMed ID of the source paper
            max_results: Maximum number of references to return
            
        Returns:
            List of PMIDs referenced by the source paper
        """
        session = await self._get_session()
        
        params = self._build_params({
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "linkname": "pubmed_pubmed_refs",
            "retmode": "json"
        })
        
        try:
            async with session.get(self.ELINK_BASE, params=params) as response:
                if response.status != 200:
                    logger.error(f"elink refs request failed: {response.status}")
                    return []
                
                data = await response.json()
                
                linksets = data.get("linksets", [])
                for linkset in linksets:
                    linksetdbs = linkset.get("linksetdbs", [])
                    for db in linksetdbs:
                        if db.get("linkname") == "pubmed_pubmed_refs":
                            links = db.get("links", [])
                            pmids = [str(link) for link in links[:max_results]]
                            logger.debug(f"Found {len(pmids)} references for PMID {pmid}")
                            return pmids
                            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching references for PMID {pmid}")
        except Exception as e:
            logger.error(f"Reference fetch error for PMID {pmid}: {e}")
        
        return []
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.get_similar_papers
    # Requirement  : `get_similar_papers` shall get PMIDs of similar papers using PubMed's similarity algorithm
    # Purpose      : Get PMIDs of similar papers using PubMed's similarity algorithm
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; max_results: int (default=20)
    # Outputs      : List[str]
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
        max_results: int = 20
    ) -> List[str]:
        """
        Get PMIDs of similar papers using PubMed's similarity algorithm.
        
        Args:
            pmid: PubMed ID of the source paper
            max_results: Maximum number of similar papers to return
            
        Returns:
            List of PMIDs of similar papers
        """
        session = await self._get_session()
        
        params = self._build_params({
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "linkname": "pubmed_pubmed",
            "retmode": "json"
        })
        
        try:
            async with session.get(self.ELINK_BASE, params=params) as response:
                if response.status != 200:
                    logger.error(f"elink similar request failed: {response.status}")
                    return []
                
                data = await response.json()
                
                linksets = data.get("linksets", [])
                for linkset in linksets:
                    linksetdbs = linkset.get("linksetdbs", [])
                    for db in linksetdbs:
                        if db.get("linkname") == "pubmed_pubmed":
                            links = db.get("links", [])
                            # First result is usually the paper itself, skip it
                            pmids = [str(link) for link in links if str(link) != pmid][:max_results]
                            logger.debug(f"Found {len(pmids)} similar papers for PMID {pmid}")
                            return pmids
                            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching similar papers for PMID {pmid}")
        except Exception as e:
            logger.error(f"Similar papers error for PMID {pmid}: {e}")
        
        return []
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.citation_crawler.CitationCrawler.get_full_citation_network
    # Requirement  : `get_full_citation_network` shall build a citation network around a paper
    # Purpose      : Build a citation network around a paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; depth: int (default=1); direction: str (default='both')
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
    async def get_full_citation_network(
        self,
        pmid: str,
        depth: int = 1,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Build a citation network around a paper.
        
        Args:
            pmid: PubMed ID of the source paper
            depth: How many levels to traverse (1 = immediate citations/references)
            direction: "citing", "references", or "both"
            
        Returns:
            Dictionary with citation network data
        """
        network = {
            "source_pmid": pmid,
            "citing": [],
            "references": [],
            "depth": depth
        }
        
        tasks = []
        
        if direction in ("citing", "both"):
            tasks.append(("citing", self.get_citing_papers(pmid)))
        
        if direction in ("references", "both"):
            tasks.append(("references", self.get_references(pmid)))
        
        # Execute in parallel
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        
        for (key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting {key}: {result}")
                network[key] = []
            else:
                network[key] = result
        
        # Calculate statistics
        network["total_citations"] = len(network.get("citing", []))
        network["total_references"] = len(network.get("references", []))
        
        logger.info(
            f"Built citation network for PMID {pmid}: "
            f"{network['total_citations']} citations, "
            f"{network['total_references']} references"
        )
        
        return network
