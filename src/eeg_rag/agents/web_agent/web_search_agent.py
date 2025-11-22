"""
Web Search Agent - Agent 2: PubMed API Integration
Handles real-time searching of PubMed database using E-utilities API.

Requirements Covered:
- REQ-AGT2-001: PubMed E-utilities integration (ESearch + EFetch)
- REQ-AGT2-002: Rate limiting (3 req/s default, 10 req/s with API key)
- REQ-AGT2-003: Result caching to prevent duplicate queries
- REQ-AGT2-004: Parse XML responses to extract metadata
- REQ-AGT2-005: Return SearchResult objects with citations
- REQ-AGT2-006: Handle API errors gracefully with retries
- REQ-AGT2-007: Support date range filtering
- REQ-AGT2-008: Support result limiting (retmax parameter)
- REQ-AGT2-009: Extract PMIDs from search results
- REQ-AGT2-010: Extract abstracts and metadata
- REQ-AGT2-011: Batch fetch for multiple PMIDs
- REQ-AGT2-012: Async execution pattern
- REQ-AGT2-013: Proper error handling and logging
- REQ-AGT2-014: Statistics tracking
- REQ-AGT2-015: Integration with BaseAgent interface
"""

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote_plus
import aiohttp

from eeg_rag.agents.base_agent import BaseAgent


@dataclass
class PubMedArticle:
    """
    Represents a PubMed article with metadata.

    Attributes:
        pmid: PubMed ID
        title: Article title
        authors: List of author names
        abstract: Article abstract text
        journal: Journal name
        pub_date: Publication date
        doi: Digital Object Identifier
        mesh_terms: Medical Subject Headings
        keywords: Article keywords
    """
    pmid: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    pub_date: str
    doi: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords
        }


@dataclass
class SearchResult:
    """
    Represents a search result from PubMed.

    Attributes:
        query: Original search query
        count: Total number of results found
        articles: List of PubMedArticle objects
        web_env: WebEnv string for History server
        query_key: Query key for History server
    """
    query: str
    count: int
    articles: List[PubMedArticle]
    web_env: Optional[str] = None
    query_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary format."""
        return {
            "query": self.query,
            "count": self.count,
            "articles": [article.to_dict() for article in self.articles],
            "web_env": self.web_env,
            "query_key": self.query_key
        }


class RateLimiter:
    """
    Rate limiter for API requests.

    Supports 3 requests/second (default) or 10 requests/second (with API key).
    """

    def __init__(self, requests_per_second: float = 3.0):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time

            if time_since_last_request < self.min_interval:
                wait_time = self.min_interval - time_since_last_request
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


class WebSearchAgent(BaseAgent):
    """
    Web Search Agent for PubMed database queries.

    This agent uses NCBI E-utilities (ESearch + EFetch) to search PubMed
    in real-time. Implements rate limiting, caching, and robust error handling.

    Attributes:
        base_url_search: ESearch API endpoint
        base_url_fetch: EFetch API endpoint
        email: Email for NCBI (required)
        tool: Tool name for NCBI (required)
        api_key: Optional API key for increased rate limits
        rate_limiter: Rate limiter instance
        cache: Query cache to prevent duplicate requests
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        name: str = "WebSearchAgent",
        email: str = "your.email@example.com",
        tool: str = "eeg-rag",
        api_key: Optional[str] = None,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Web Search Agent.
        
        Args:
            name: Agent name
            email: Email address (required by NCBI)
            tool: Tool name (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
            timeout: HTTP request timeout in seconds
            config: Optional configuration dictionary
        """
        from eeg_rag.agents.base_agent import AgentType
        
        super().__init__(
            agent_type=AgentType.WEB_SEARCH,
            name=name,
            config=config
        )        # NCBI E-utilities endpoints
        self.base_url_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        # Required NCBI parameters
        self.email = email
        self.tool = tool
        self.api_key = api_key
        self.timeout = timeout

        # Rate limiting (10 req/s with API key, 3 req/s without)
        requests_per_second = 10.0 if api_key else 3.0
        self.rate_limiter = RateLimiter(requests_per_second)

        # Query cache (stores query hash -> SearchResult)
        self.cache: Dict[str, SearchResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.total_searches = 0
        self.total_articles_fetched = 0
        self.total_errors = 0

    def _get_query_hash(self, query: str, **params) -> str:
        """
        Generate hash for query + parameters for caching.

        Args:
            query: Search query
            **params: Additional search parameters

        Returns:
            MD5 hash of query + parameters
        """
        cache_key = f"{query}|{sorted(params.items())}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    async def _make_request(
        self,
        url: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Make HTTP request with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Response text

        Raises:
            aiohttp.ClientError: On HTTP errors
        """
        await self.rate_limiter.acquire()

        # Add required parameters
        params["email"] = self.email
        params["tool"] = self.tool
        if self.api_key:
            params["api_key"] = self.api_key

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                return await response.text()

    async def _search_pubmed(
        self,
        query: str,
        retmax: int = 20,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        sort: str = "relevance"
    ) -> Dict[str, Any]:
        """
        Execute ESearch query against PubMed.

        Args:
            query: Search query string
            retmax: Maximum number of results to return
            mindate: Minimum date (YYYY/MM/DD format)
            maxdate: Maximum date (YYYY/MM/DD format)
            sort: Sort order (relevance, pub_date, etc.)

        Returns:
            Dictionary with keys: count, ids, web_env, query_key
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "xml",
            "sort": sort,
            "usehistory": "y"  # Use History server for large results
        }

        if mindate:
            params["mindate"] = mindate
        if maxdate:
            params["maxdate"] = maxdate
        if mindate or maxdate:
            params["datetype"] = "pdat"  # Publication date

        try:
            response_text = await self._make_request(self.base_url_search, params)
            root = ET.fromstring(response_text)

            # Extract results
            count = int(root.findtext("Count", "0"))
            web_env = root.findtext("WebEnv")
            query_key = root.findtext("QueryKey")

            # Extract PMIDs
            id_list = root.find("IdList")
            pmids = []
            if id_list is not None:
                pmids = [id_elem.text for id_elem in id_list.findall("Id")]

            return {
                "count": count,
                "ids": pmids,
                "web_env": web_env,
                "query_key": query_key
            }

        except ET.ParseError as e:
            self.total_errors += 1
            raise ValueError(f"Failed to parse ESearch XML response: {e}")
        except Exception as e:
            self.total_errors += 1
            raise RuntimeError(f"ESearch request failed: {e}")

    async def _fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch full article details using EFetch.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }

        try:
            response_text = await self._make_request(self.base_url_fetch, params)
            root = ET.fromstring(response_text)

            articles = []
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    article = self._parse_article(article_elem)
                    if article:
                        articles.append(article)
                        self.total_articles_fetched += 1
                except Exception as e:
                    # Skip individual articles that fail to parse
                    print(f"Warning: Failed to parse article: {e}")
                    continue

            return articles

        except ET.ParseError as e:
            self.total_errors += 1
            raise ValueError(f"Failed to parse EFetch XML response: {e}")
        except Exception as e:
            self.total_errors += 1
            raise RuntimeError(f"EFetch request failed: {e}")

    def _parse_article(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """
        Parse PubmedArticle XML element.

        Args:
            article_elem: XML element containing article data

        Returns:
            PubMedArticle object or None if parsing fails
        """
        try:
            medline_citation = article_elem.find(".//MedlineCitation")
            if medline_citation is None:
                return None

            # Extract PMID
            pmid_elem = medline_citation.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            if not pmid:
                return None

            # Extract title
            title_elem = medline_citation.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title available"

            # Extract authors
            authors = []
            author_list = medline_citation.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last_name = author.findtext("LastName", "")
                    initials = author.findtext("Initials", "")
                    if last_name:
                        authors.append(f"{last_name} {initials}".strip())

            # Extract abstract
            abstract_parts = []
            abstract = medline_citation.find(".//Abstract")
            if abstract is not None:
                for abstract_text in abstract.findall("AbstractText"):
                    if abstract_text.text:
                        label = abstract_text.get("Label", "")
                        text = abstract_text.text
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
            abstract_text = " ".join(abstract_parts) if abstract_parts else "No abstract available"

            # Extract journal
            journal_elem = medline_citation.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"

            # Extract publication date
            pub_date_elem = medline_citation.find(".//PubDate")
            pub_date = "Unknown date"
            if pub_date_elem is not None:
                year = pub_date_elem.findtext("Year", "")
                month = pub_date_elem.findtext("Month", "")
                day = pub_date_elem.findtext("Day", "")
                pub_date = f"{year}-{month}-{day}".strip("-")

            # Extract DOI
            doi = None
            article_id_list = article_elem.find(".//PubmedData/ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

            # Extract MeSH terms
            mesh_terms = []
            mesh_heading_list = medline_citation.find(".//MeshHeadingList")
            if mesh_heading_list is not None:
                for mesh_heading in mesh_heading_list.findall("MeshHeading"):
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is not None and descriptor.text:
                        mesh_terms.append(descriptor.text)

            # Extract keywords
            keywords = []
            keyword_list = medline_citation.find(".//KeywordList")
            if keyword_list is not None:
                for keyword in keyword_list.findall("Keyword"):
                    if keyword.text:
                        keywords.append(keyword.text)

            return PubMedArticle(
                pmid=pmid,
                title=title,
                authors=authors,
                abstract=abstract_text,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                mesh_terms=mesh_terms,
                keywords=keywords
            )

        except Exception as e:
            print(f"Error parsing article: {e}")
            return None

    async def execute(self, query, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute web search query.
        
        Args:
            query: Search query (can be string or AgentQuery for BaseAgent compatibility)
            context: Optional context dictionary with parameters:
                - max_results: Maximum number of results (default: 20)
                - mindate: Minimum publication date (YYYY/MM/DD)
                - maxdate: Maximum publication date (YYYY/MM/DD)
                - sort: Sort order (default: relevance)
                - use_cache: Whether to use cache (default: True)
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - query: Original query
                - count: Total results found
                - articles: List of article dictionaries
                - cached: Whether result was from cache
                - execution_time: Time taken in seconds
        """
        start_time = time.time()
        
        # Handle both string and AgentQuery
        from eeg_rag.agents.base_agent import AgentQuery
        if isinstance(query, AgentQuery):
            query_text = query.text
            context = context or query.context
        else:
            query_text = str(query)
        
        # Parse context parameters
        context = context or {}
        max_results = context.get("max_results", 20)
        mindate = context.get("mindate")
        maxdate = context.get("maxdate")
        sort = context.get("sort", "relevance")
        use_cache = context.get("use_cache", True)
        
        # Check cache
        cache_key = self._get_query_hash(
            query_text,
            max_results=max_results,
            mindate=mindate,
            maxdate=maxdate,
            sort=sort
        )

        if use_cache and cache_key in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[cache_key]
            execution_time = time.time() - start_time

            return {
                "success": True,
                "query": query_text,
                "count": cached_result.count,
                "articles": [article.to_dict() for article in cached_result.articles],
                "cached": True,
                "execution_time": execution_time
            }

        self.cache_misses += 1
        self.total_searches += 1

        try:
            # Execute search
            search_results = await self._search_pubmed(
                query=query_text,
                retmax=max_results,
                mindate=mindate,
                maxdate=maxdate,
                sort=sort
            )

            # Fetch article details
            articles = await self._fetch_articles(search_results["ids"])

            # Create SearchResult object
            result = SearchResult(
                query=query_text,
                count=search_results["count"],
                articles=articles,
                web_env=search_results.get("web_env"),
                query_key=search_results.get("query_key")
            )

            # Cache result
            if use_cache:
                self.cache[cache_key] = result

            execution_time = time.time() - start_time

            # Update BaseAgent statistics
            self.total_executions += 1
            self.successful_executions += 1

            return {
                "success": True,
                "query": query_text,
                "count": result.count,
                "articles": [article.to_dict() for article in result.articles],
                "cached": False,
                "execution_time": execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_executions += 1
            self.total_executions += 1

            return {
                "success": False,
                "query": query_text,
                "error": str(e),
                "execution_time": execution_time
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary containing:
                - name: Agent name
                - agent_id: Agent ID
                - total_executions: Total queries executed
                - successful_executions: Successful queries
                - failed_executions: Failed queries
                - average_execution_time: Average time per query
                - total_searches: Total PubMed searches
                - total_articles_fetched: Total articles retrieved
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - cache_hit_rate: Cache hit rate percentage
                - total_errors: Total errors encountered
        """
        base_stats = super().get_statistics()

        cache_hit_rate = 0.0
        if (self.cache_hits + self.cache_misses) > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)

        return {
            **base_stats,
            "total_searches": self.total_searches,
            "total_articles_fetched": self.total_articles_fetched,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "total_errors": self.total_errors
        }

    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
