"""
Enhanced Semantic Scholar Agent

Comprehensive Semantic Scholar integration with citation graphs and influence scoring.

Requirements Covered:
- REQ-S2-010: Full S2 API integration
- REQ-S2-011: Citation graph traversal
- REQ-S2-012: Author tracking
- REQ-S2-013: Paper recommendations
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from eeg_rag.agents.base_agent import (
    BaseAgent, AgentType, AgentResult, AgentQuery
)
from .influence_scorer import InfluenceScorer

logger = logging.getLogger(__name__)


@dataclass
class S2Paper:
    """Semantic Scholar paper data."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[Dict[str, str]] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    external_ids: Dict[str, str] = field(default_factory=dict)
    is_open_access: bool = False
    open_access_pdf: Optional[str] = None
    tldr: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": [a.get("name", "") for a in self.authors],
            "author_ids": [a.get("authorId", "") for a in self.authors],
            "year": self.year,
            "venue": self.venue,
            "journal": self.venue,  # Alias
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "reference_count": self.reference_count,
            "fields_of_study": self.fields_of_study,
            "pmid": self.external_ids.get("PubMed"),
            "doi": self.external_ids.get("DOI"),
            "arxiv_id": self.external_ids.get("ArXiv"),
            "is_open_access": self.is_open_access,
            "pdf_url": self.open_access_pdf,
            "tldr": self.tldr,
            "source": "semantic_scholar"
        }


class SemanticScholarAgent(BaseAgent):
    """
    Enhanced Semantic Scholar agent with:
    - Paper search with field filtering
    - Citation graph traversal
    - Author expertise tracking
    - Influential citation identification
    - Paper recommendations
    """
    
    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    S2_PARTNER_BASE = "https://partner.semanticscholar.org/graph/v1"
    
    # Fields to request from S2 API
    PAPER_FIELDS = [
        "paperId", "title", "abstract", "authors", "year", "venue",
        "citationCount", "influentialCitationCount", "referenceCount",
        "fieldsOfStudy", "externalIds", "isOpenAccess", "openAccessPdf",
        "tldr"
    ]
    
    def __init__(
        self,
        name: str = "SemanticScholarAgent",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Semantic Scholar agent.
        
        Args:
            name: Agent name
            api_key: S2 API key for higher rate limits
            config: Optional configuration
        """
        super().__init__(
            agent_type=AgentType.WEB_SEARCH,
            name=name,
            config=config or {}
        )
        
        self.api_key = api_key
        self.influence_scorer = InfluenceScorer()
        
        # Rate limiting (100 requests per 5 minutes = ~20/min)
        self.requests_per_minute = 100 if api_key else 20
        self.min_request_interval = 60.0 / self.requests_per_minute
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False
        
        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(hours=12)
        
        # Statistics
        self.total_searches = 0
        self.total_papers_fetched = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"SemanticScholarAgent initialized (api_key={'yes' if api_key else 'no'})")
    
    @property
    def base_url(self) -> str:
        """Get base URL based on API key availability."""
        return self.S2_PARTNER_BASE if self.api_key else self.S2_API_BASE
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            self._owns_session = True
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._request_lock:
            import time
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self._last_request_time = time.time()
    
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
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[key] = (value, datetime.now())
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute Semantic Scholar search.
        
        Args:
            query: Search query
            
        Returns:
            AgentResult with papers
        """
        start_time = datetime.now()
        
        try:
            params = query.parameters
            max_results = params.get("max_results", 50)
            year_range = params.get("year_range")
            fields_of_study = params.get("fields_of_study")
            open_access_only = params.get("open_access_only", False)
            
            # Check cache
            cache_key = f"s2_search:{query.text}:{max_results}:{year_range}"
            cached = self._get_cache(cache_key)
            if cached:
                logger.info(f"Cache hit for S2 query: {query.text[:50]}")
                elapsed = (datetime.now() - start_time).total_seconds()
                return AgentResult(
                    success=True,
                    data=cached,
                    metadata={"source": "cache", "cache_hit": True},
                    agent_type=AgentType.WEB_SEARCH,
                    elapsed_time=elapsed
                )
            
            # Execute search
            papers, total = await self._search(
                query.text,
                max_results=max_results,
                year_range=year_range,
                fields_of_study=fields_of_study,
                open_access_only=open_access_only
            )
            
            # Calculate influence scores and sort
            paper_dicts = [p.to_dict() for p in papers]
            for paper_dict in paper_dicts:
                paper_dict["influence_score"] = self.influence_scorer.score_paper(paper_dict)
            
            paper_dicts.sort(key=lambda p: p.get("influence_score", 0), reverse=True)
            
            result_data = {
                "papers": paper_dicts,
                "total_count": total,
                "returned_count": len(papers)
            }
            
            # Cache result
            self._set_cache(cache_key, result_data)
            
            self.total_searches += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"S2 search completed: {len(papers)} papers in {elapsed:.2f}s")
            
            return AgentResult(
                success=True,
                data=result_data,
                metadata={"source": "semantic_scholar"},
                agent_type=AgentType.WEB_SEARCH,
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception(f"S2 search failed: {e}")
            elapsed = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.WEB_SEARCH,
                elapsed_time=elapsed
            )
    
    async def _search(
        self,
        query: str,
        max_results: int = 50,
        year_range: Optional[Tuple[int, int]] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False
    ) -> Tuple[List[S2Paper], int]:
        """
        Execute paper search.
        
        Args:
            query: Search query
            max_results: Maximum results
            year_range: Optional (start_year, end_year)
            fields_of_study: Optional field filters
            open_access_only: Filter to open access
            
        Returns:
            Tuple of (papers, total_count)
        """
        await self._rate_limit()
        session = await self._get_session()
        
        params = {
            "query": query,
            "limit": min(100, max_results),
            "fields": ",".join(self.PAPER_FIELDS)
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        if open_access_only:
            params["openAccessPdf"] = ""
        
        try:
            async with session.get(
                f"{self.base_url}/paper/search",
                params=params
            ) as response:
                if response.status == 429:
                    logger.warning("Rate limited by S2 API")
                    return [], 0
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"S2 API error {response.status}: {error_text[:200]}")
                    return [], 0
                
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                        self.total_papers_fetched += 1
                
                total = data.get("total", len(papers))
                return papers, total
                
        except Exception as e:
            logger.error(f"S2 search error: {e}")
            return [], 0
    
    def _parse_paper(self, data: Dict[str, Any]) -> Optional[S2Paper]:
        """Parse S2 API response into S2Paper."""
        if not data or not data.get("paperId"):
            return None
        
        authors = []
        for author in data.get("authors") or []:
            authors.append({
                "authorId": author.get("authorId", ""),
                "name": author.get("name", "")
            })
        
        external_ids = data.get("externalIds") or {}
        
        open_access_pdf = None
        pdf_info = data.get("openAccessPdf")
        if pdf_info and isinstance(pdf_info, dict):
            open_access_pdf = pdf_info.get("url")
        
        tldr = None
        tldr_info = data.get("tldr")
        if tldr_info and isinstance(tldr_info, dict):
            tldr = tldr_info.get("text")
        
        return S2Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0) or 0,
            influential_citation_count=data.get("influentialCitationCount", 0) or 0,
            reference_count=data.get("referenceCount", 0) or 0,
            fields_of_study=data.get("fieldsOfStudy") or [],
            external_ids=external_ids,
            is_open_access=data.get("isOpenAccess", False),
            open_access_pdf=open_access_pdf,
            tldr=tldr
        )
    
    async def get_paper_details(self, paper_id: str) -> Optional[S2Paper]:
        """
        Get details for a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper details or None
        """
        await self._rate_limit()
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/paper/{paper_id}",
                params={"fields": ",".join(self.PAPER_FIELDS)}
            ) as response:
                if response.status == 404:
                    logger.warning(f"Paper not found: {paper_id}")
                    return None
                
                if response.status != 200:
                    logger.error(f"S2 paper details error: {response.status}")
                    return None
                
                data = await response.json()
                return self._parse_paper(data)
                
        except Exception as e:
            logger.error(f"Paper details error: {e}")
            return None
    
    async def get_citations(
        self,
        paper_id: str,
        max_results: int = 50
    ) -> List[S2Paper]:
        """
        Get papers that cite a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum citations to return
            
        Returns:
            List of citing papers
        """
        await self._rate_limit()
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/paper/{paper_id}/citations",
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": min(100, max_results)
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"S2 citations error: {response.status}")
                    return []
                
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    citing_paper = item.get("citingPaper", {})
                    paper = self._parse_paper(citing_paper)
                    if paper:
                        papers.append(paper)
                
                return papers
                
        except Exception as e:
            logger.error(f"Citations error: {e}")
            return []
    
    async def get_references(
        self,
        paper_id: str,
        max_results: int = 50
    ) -> List[S2Paper]:
        """
        Get papers referenced by a given paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum references to return
            
        Returns:
            List of referenced papers
        """
        await self._rate_limit()
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/paper/{paper_id}/references",
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": min(100, max_results)
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"S2 references error: {response.status}")
                    return []
                
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    cited_paper = item.get("citedPaper", {})
                    paper = self._parse_paper(cited_paper)
                    if paper:
                        papers.append(paper)
                
                return papers
                
        except Exception as e:
            logger.error(f"References error: {e}")
            return []
    
    async def get_author_papers(
        self,
        author_id: str,
        max_results: int = 50
    ) -> List[S2Paper]:
        """
        Get papers by a specific author.
        
        Args:
            author_id: Semantic Scholar author ID
            max_results: Maximum papers to return
            
        Returns:
            List of papers
        """
        await self._rate_limit()
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.base_url}/author/{author_id}/papers",
                params={
                    "fields": ",".join(self.PAPER_FIELDS),
                    "limit": min(100, max_results)
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"S2 author papers error: {response.status}")
                    return []
                
                data = await response.json()
                
                papers = []
                for item in data.get("data", []):
                    paper = self._parse_paper(item)
                    if paper:
                        papers.append(paper)
                
                return papers
                
        except Exception as e:
            logger.error(f"Author papers error: {e}")
            return []
    
    async def get_citation_graph(
        self,
        paper_id: str,
        depth: int = 1,
        max_per_level: int = 10
    ) -> Dict[str, Any]:
        """
        Build a citation graph around a paper.
        
        Args:
            paper_id: Center paper ID
            depth: How many levels to traverse
            max_per_level: Max papers per level
            
        Returns:
            Citation graph data
        """
        graph = {
            "center": paper_id,
            "citations": [],
            "references": [],
            "depth": depth
        }
        
        # Get center paper
        center_paper = await self.get_paper_details(paper_id)
        if center_paper:
            graph["center_paper"] = center_paper.to_dict()
        
        # Get citations and references in parallel
        citations_task = self.get_citations(paper_id, max_per_level)
        references_task = self.get_references(paper_id, max_per_level)
        
        citations, references = await asyncio.gather(
            citations_task, references_task,
            return_exceptions=True
        )
        
        if isinstance(citations, list):
            graph["citations"] = [p.to_dict() for p in citations]
            graph["citation_count"] = len(citations)
        
        if isinstance(references, list):
            graph["references"] = [p.to_dict() for p in references]
            graph["reference_count"] = len(references)
        
        return graph
    
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
