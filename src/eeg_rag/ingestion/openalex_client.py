"""
OpenAlex client - free, open alternative to proprietary databases.
Contains 200M+ works with good coverage of EEG literature.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OpenAlexWork:
    """Work (paper) from OpenAlex."""
    openalex_id: str
    doi: Optional[str]
    pmid: Optional[str]
    title: str
    abstract: str
    authors: list[dict]  # {name, institution, orcid}
    publication_date: Optional[datetime]
    journal: str
    citation_count: int
    concepts: list[dict]  # {name, score}
    topics: list[dict]
    referenced_works: list[str]  # OpenAlex IDs
    related_works: list[str]
    open_access: bool
    pdf_url: Optional[str]


class OpenAlexClient:
    """
    Client for OpenAlex API.
    Completely free with 100K requests/day limit.
    """
    
    BASE_URL = "https://api.openalex.org"
    
    # OpenAlex concept IDs for EEG-related topics
    EEG_CONCEPTS = [
        "C150533007",  # Electroencephalography
        "C2776048018", # Event-related potential
        "C2779356033", # Brain–computer interface
        "C116915560",  # Evoked potential
        "C2779919020", # P300
        "C71291556",   # Epilepsy
        "C15708023",   # Cognitive neuroscience
        "C127413603",  # Computational neuroscience
        "C41008148",   # Machine learning (for EEG ML papers)
    ]
    
    def __init__(self, email: str = "your-email@example.com"):
        """
        Initialize OpenAlex client.
        
        Args:
            email: Email for polite pool (faster rate limits)
        """
        self.email = email
        self.headers = {"User-Agent": f"EEG-RAG/1.0 (mailto:{email})"}
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(headers=self.headers)
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
        
    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: dict = None
    ) -> dict:
        """Fetch from OpenAlex API."""
        url = f"{self.BASE_URL}/{endpoint}"
        if params is None:
            params = {}
        params["mailto"] = self.email
        
        async with session.get(url, headers=self.headers, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def search_works(
        self,
        query: Optional[str] = None,
        concept_ids: Optional[list[str]] = None,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        per_page: int = 200,
        max_results: int = 10000
    ) -> AsyncIterator[OpenAlexWork]:
        """
        Search for works with various filters.
        
        Args:
            query: Text search query
            concept_ids: Filter by OpenAlex concept IDs (optional, may be deprecated)
            from_year: Minimum publication year
            to_year: Maximum publication year
            per_page: Results per page (max 200)
            max_results: Maximum total results
            
        Yields:
            OpenAlexWork objects
        """
        filters = []
        
        # Skip concept_ids filtering as it's deprecated - use text search instead
        # if concept_ids:
        #     concept_filter = "|".join(concept_ids)
        #     filters.append(f"concepts.id:{concept_filter}")
        
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        
        # Only works with abstracts
        filters.append("has_abstract:true")
        
        params = {
            "per-page": per_page,
            "select": "id,doi,ids,title,abstract_inverted_index,authorships,publication_date,primary_location,cited_by_count,concepts,topics,referenced_works,related_works,open_access"
        }
        
        if query:
            params["search"] = query
        if filters:
            params["filter"] = ",".join(filters)
        
        cursor = "*"
        count = 0
        
        async with aiohttp.ClientSession() as session:
            while cursor and count < max_results:
                params["cursor"] = cursor
                
                result = await self._fetch(session, "works", params)
                
                works = result.get("results", [])
                if not works:
                    break
                
                for work_data in works:
                    work = self._parse_work(work_data)
                    if work:
                        yield work
                        count += 1
                        if count >= max_results:
                            break
                
                cursor = result.get("meta", {}).get("next_cursor")
                logger.info(f"Fetched {count} works from OpenAlex")
                
                # Small delay to be polite
                await asyncio.sleep(0.1)

    def _parse_work(self, data: dict) -> Optional[OpenAlexWork]:
        """Parse OpenAlex work data."""
        try:
            # Reconstruct abstract from inverted index
            abstract = ""
            abstract_inv = data.get("abstract_inverted_index", {})
            if abstract_inv:
                # Reconstruct from inverted index
                word_positions = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(word for _, word in word_positions)
            
            # Parse authors
            authors = []
            for authorship in data.get("authorships", []):
                author_info = authorship.get("author", {})
                institutions = authorship.get("institutions", [])
                authors.append({
                    "name": author_info.get("display_name", ""),
                    "orcid": author_info.get("orcid"),
                    "institution": institutions[0].get("display_name") if institutions else None
                })
            
            # Parse publication date
            pub_date = None
            date_str = data.get("publication_date")
            if date_str:
                try:
                    pub_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            
            # Get journal
            location = data.get("primary_location", {})
            source = location.get("source", {})
            journal = source.get("display_name", "") if source else ""
            
            # Parse IDs
            ids = data.get("ids", {})
            
            # Get PDF URL
            pdf_url = None
            oa = data.get("open_access", {})
            if oa.get("is_oa"):
                pdf_url = oa.get("oa_url")
            
            return OpenAlexWork(
                openalex_id=data.get("id", "").split("/")[-1],
                doi=ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None,
                pmid=ids.get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") if ids.get("pmid") else None,
                title=data.get("title", ""),
                abstract=abstract,
                authors=authors,
                publication_date=pub_date,
                journal=journal,
                citation_count=data.get("cited_by_count", 0),
                concepts=[
                    {"name": c.get("display_name"), "score": c.get("score")}
                    for c in data.get("concepts", [])[:10]
                ],
                topics=[
                    {"name": t.get("display_name")}
                    for t in data.get("topics", [])[:5]
                ],
                referenced_works=[
                    ref.split("/")[-1] for ref in data.get("referenced_works", [])[:100]
                ],
                related_works=[
                    ref.split("/")[-1] for ref in data.get("related_works", [])[:20]
                ],
                open_access=oa.get("is_oa", False) if oa else False,
                pdf_url=pdf_url
            )
            
        except Exception as e:
            logger.warning(f"Error parsing OpenAlex work: {e}")
            return None

    async def get_work_by_id(self, work_id: str) -> Optional[OpenAlexWork]:
        """
        Fetch a specific work by OpenAlex ID.
        
        Args:
            work_id: OpenAlex work ID (e.g., 'W2741809807')
            
        Returns:
            OpenAlexWork or None if not found
        """
        async with aiohttp.ClientSession() as session:
            try:
                params = {
                    "select": "id,doi,ids,title,abstract_inverted_index,authorships,publication_date,primary_location,cited_by_count,concepts,topics,referenced_works,related_works,open_access"
                }
                result = await self._fetch(session, f"works/{work_id}", params)
                return self._parse_work(result)
            except Exception as e:
                logger.warning(f"Error fetching work {work_id}: {e}")
                return None

    async def collect_eeg_corpus(
        self,
        from_year: int = 2014,
        max_results: int = 50000
    ) -> AsyncIterator[OpenAlexWork]:
        """
        Collect comprehensive EEG corpus from OpenAlex using keyword search.
        
        Args:
            from_year: Earliest publication year
            max_results: Maximum works to collect
            
        Yields:
            OpenAlexWork objects
        """
        logger.info("Collecting EEG corpus from OpenAlex using keyword search...")
        
        # Use text search for EEG-related terms
        eeg_search_query = "electroencephalography OR EEG OR brain-computer interface OR event-related potential"
        
        async for work in self.search_works(
            query=eeg_search_query,
            from_year=from_year,
            to_year=datetime.now().year,
            max_results=max_results
        ):
            yield work

    async def get_citations(self, work_id: str, max_results: int = 100) -> AsyncIterator[OpenAlexWork]:
        """
        Get works that cite a specific work.
        
        Args:
            work_id: OpenAlex work ID
            max_results: Maximum citing works to return
            
        Yields:
            OpenAlexWork objects (citing papers)
        """
        params = {
            "filter": f"cites:{work_id}",
            "per-page": min(max_results, 200),
            "select": "id,doi,ids,title,abstract_inverted_index,authorships,publication_date,primary_location,cited_by_count,concepts,topics,referenced_works,related_works,open_access"
        }
        
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(session, "works", params)
            
            for work_data in result.get("results", []):
                work = self._parse_work(work_data)
                if work:
                    yield work

    async def get_author_works(self, author_id: str, max_results: int = 100) -> AsyncIterator[OpenAlexWork]:
        """
        Get works by a specific author.
        
        Args:
            author_id: OpenAlex author ID
            max_results: Maximum works to return
            
        Yields:
            OpenAlexWork objects
        """
        params = {
            "filter": f"author.id:{author_id}",
            "per-page": min(max_results, 200),
            "select": "id,doi,ids,title,abstract_inverted_index,authorships,publication_date,primary_location,cited_by_count,concepts,topics,referenced_works,related_works,open_access"
        }
        
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(session, "works", params)
            
            for work_data in result.get("results", []):
                work = self._parse_work(work_data)
                if work:
                    yield work
