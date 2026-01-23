"""
arXiv client for EEG/neuroscience preprints.
arXiv has many BCI and EEG ML papers before peer review.
"""

import asyncio
import aiohttp
import feedparser
from dataclasses import dataclass
from typing import Optional, AsyncIterator
from datetime import datetime
import logging
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Paper from arXiv."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    doi: Optional[str]
    journal_ref: Optional[str]
    pdf_url: str
    html_url: str
    comment: Optional[str]


class ArxivClient:
    """
    Async client for arXiv API.
    Excellent for EEG machine learning and BCI papers.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # arXiv categories relevant to EEG research
    RELEVANT_CATEGORIES = [
        "q-bio.NC",    # Neurons and Cognition
        "cs.LG",       # Machine Learning
        "cs.NE",       # Neural and Evolutionary Computing  
        "cs.HC",       # Human-Computer Interaction
        "eess.SP",     # Signal Processing
        "stat.ML",     # Machine Learning (Statistics)
        "cs.CV",       # Computer Vision (for some EEG viz papers)
    ]
    
    # EEG-specific search terms
    EEG_SEARCH_TERMS = [
        "EEG",
        "electroencephalography",
        "brain-computer interface",
        "BCI",
        "event-related potential",
        "motor imagery",
        "P300 speller",
        "SSVEP",  # Steady-state visual evoked potential
        "neural signal decoding",
        "brain signal classification",
        "EEG emotion recognition",
        "sleep staging EEG",
        "seizure detection",
        "EEG artifact",
        "EEG transformer",
        "EEG deep learning",
        "brain connectivity",
    ]
    
    def __init__(self, results_per_request: int = 100):
        """
        Initialize arXiv client.
        
        Args:
            results_per_request: Results per API call (max ~1000)
        """
        self.results_per_request = results_per_request
        self._last_request_time = 0.0
        # arXiv asks for 3 second delay between requests
        self.delay = 3.0
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
        return False
    
    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _rate_limit(self):
        """Enforce arXiv's rate limit."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _build_query(
        self,
        search_terms: list[str],
        categories: Optional[list[str]] = None
    ) -> str:
        """Build arXiv search query string."""
        # Search in title and abstract
        term_queries = [f'(ti:"{term}" OR abs:"{term}")' for term in search_terms]
        query = " OR ".join(term_queries)
        
        # Optionally restrict to categories
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({cat_query})"
        
        return query

    async def search(
        self,
        query: str,
        max_results: int = 1000,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> AsyncIterator[ArxivPaper]:
        """
        Search arXiv and yield papers.
        
        Args:
            query: arXiv query string
            max_results: Maximum papers to retrieve
            sort_by: relevance, lastUpdatedDate, or submittedDate
            sort_order: ascending or descending
            
        Yields:
            ArxivPaper objects
        """
        start = 0
        
        async with aiohttp.ClientSession() as session:
            while start < max_results:
                await self._rate_limit()
                
                params = {
                    "search_query": query,
                    "start": start,
                    "max_results": min(self.results_per_request, max_results - start),
                    "sortBy": sort_by,
                    "sortOrder": sort_order
                }
                
                async with session.get(self.BASE_URL, params=params) as response:
                    response.raise_for_status()
                    content = await response.text()
                
                feed = feedparser.parse(content)
                
                if not feed.entries:
                    break
                
                for entry in feed.entries:
                    paper = self._parse_entry(entry)
                    if paper:
                        yield paper
                
                start += len(feed.entries)
                logger.info(f"Fetched {start} papers from arXiv")

    def _parse_entry(self, entry) -> Optional[ArxivPaper]:
        """Parse arXiv feed entry into ArxivPaper."""
        try:
            # Extract arXiv ID
            arxiv_id = entry.id.split("/abs/")[-1]
            
            # Parse dates
            published = datetime.strptime(
                entry.published, "%Y-%m-%dT%H:%M:%SZ"
            )
            updated = datetime.strptime(
                entry.updated, "%Y-%m-%dT%H:%M:%SZ"
            )
            
            # Extract categories
            categories = [tag.term for tag in entry.tags]
            primary = entry.get("arxiv_primary_category", {}).get("term", categories[0] if categories else "")
            
            # Extract links
            pdf_url = ""
            html_url = entry.link
            for link in entry.links:
                if link.get("type") == "application/pdf":
                    pdf_url = link.href
            
            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=entry.title.replace("\n", " "),
                abstract=entry.summary.replace("\n", " "),
                authors=[a.name for a in entry.authors],
                categories=categories,
                primary_category=primary,
                published=published,
                updated=updated,
                doi=entry.get("arxiv_doi"),
                journal_ref=entry.get("arxiv_journal_ref"),
                pdf_url=pdf_url,
                html_url=html_url,
                comment=entry.get("arxiv_comment")
            )
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None

    async def collect_eeg_papers(
        self,
        max_results: int = 5000,
        years_back: int = 5
    ) -> AsyncIterator[ArxivPaper]:
        """
        Collect EEG-related papers from arXiv.
        
        Args:
            max_results: Maximum total papers
            years_back: How many years of papers to collect
            
        Yields:
            ArxivPaper objects
        """
        query = self._build_query(
            self.EEG_SEARCH_TERMS,
            self.RELEVANT_CATEGORIES
        )
        
        logger.info(f"Searching arXiv with query: {query[:100]}...")
        
        count = 0
        cutoff = datetime.now().year - years_back
        
        async for paper in self.search(query, max_results):
            # Filter by year
            if paper.published.year >= cutoff:
                yield paper
                count += 1
        
        logger.info(f"Collected {count} EEG papers from arXiv")

    async def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., '2301.12345')
            
        Returns:
            ArxivPaper or None if not found
        """
        # Normalize ID
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()
        
        query = f"id:{arxiv_id}"
        
        async for paper in self.search(query, max_results=1):
            return paper
        
        return None

    async def search_by_author(
        self,
        author_name: str,
        max_results: int = 100
    ) -> AsyncIterator[ArxivPaper]:
        """
        Search for papers by author name.
        
        Args:
            author_name: Author name to search
            max_results: Maximum results
            
        Yields:
            ArxivPaper objects
        """
        query = f'au:"{author_name}"'
        
        async for paper in self.search(query, max_results):
            yield paper
