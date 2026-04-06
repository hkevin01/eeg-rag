"""
Europe PMC client for EEG open-access literature.

Europe PMC provides open-access full text for EU-funded research,
preprints indexed from Europe, and NIH-funded papers — complementing
PubMed with full-text availability and additional metadata.

API docs: https://europepmc.org/RestfulWebService
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class EuropePMCArticle:
    """Article record from Europe PMC."""

    pmid: Optional[str]
    pmcid: Optional[str]
    doi: Optional[str]
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: Optional[int]
    publication_date: Optional[str]
    is_open_access: bool
    full_text_available: bool
    full_text: Optional[str]           # populated if retrieved separately
    mesh_terms: List[str]
    keywords: List[str]
    grant_ids: List[str]               # funding reference IDs
    citation_count: int
    source: str                        # MED, PPR (preprint), etc.
    eeg_methods_mentioned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "publication_date": self.publication_date,
            "is_open_access": self.is_open_access,
            "full_text_available": self.full_text_available,
            "full_text": self.full_text,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "grant_ids": self.grant_ids,
            "citation_count": self.citation_count,
            "epmc_source": self.source,
            "eeg_methods_mentioned": self.eeg_methods_mentioned,
            "source": "europe_pmc",
        }


class EuropePMCClient:
    """
    Async client for the Europe PMC REST API.

    Yields EEG-relevant open-access articles including full-text
    when available via PMC. Supports preprints from bioRxiv/medRxiv
    indexed in Europe PMC.

    Rate limit: Polite use; no hard throttle on the public API.
    """

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    # EEG-focused search queries for Europe PMC query language
    EEG_QUERIES: List[str] = [
        'TITLE_ABS:"electroencephalography" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"EEG" AND TITLE_ABS:"epilepsy" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"brain-computer interface" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"EEG" AND TITLE_ABS:"sleep staging" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"event-related potential" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"neurofeedback" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"motor imagery" AND TITLE_ABS:"EEG" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"deep learning" AND TITLE_ABS:"EEG" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"seizure detection" AND TITLE_ABS:"EEG" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"EEG biomarker" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"cognitive EEG" AND OPEN_ACCESS:Y',
        'TITLE_ABS:"quantitative EEG" AND OPEN_ACCESS:Y',
    ]

    EEG_METHOD_KEYWORDS: List[str] = [
        "electroencephalograph",
        "eeg",
        "event-related potential",
        "erp",
        "brain-computer interface",
        "bci",
        "neurofeedback",
        "motor imagery",
        "p300",
        "n400",
        "ssvep",
        "brain oscillation",
        "sleep spindle",
        "seizure eeg",
        "source localization",
    ]

    def __init__(
        self,
        page_size: int = 100,
        timeout: float = 30.0,
        fetch_full_text: bool = False,
    ):
        """
        Args:
            page_size: Results per API request page.
            timeout: HTTP request timeout seconds.
            fetch_full_text: If True, attempt to retrieve full XML text
                             for open-access PMC articles (slower).
        """
        self.page_size = page_size
        self.fetch_full_text = fetch_full_text
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "EuropePMCClient":
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._session:
            await self._session.close()
            self._session = None
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search_eeg_articles(
        self,
        max_results: int = 1000,
        min_year: int = 2010,
    ) -> AsyncIterator[EuropePMCArticle]:
        """
        Yield unique EEG articles from Europe PMC.

        Args:
            max_results: Maximum articles to yield across all queries.
            min_year: Exclude articles published before this year.

        Yields:
            EuropePMCArticle records.
        """
        seen: set[str] = set()
        yielded = 0

        for query in self.EEG_QUERIES:
            if yielded >= max_results:
                break
            date_filtered = f"({query}) AND PUB_YEAR:[{min_year} TO 9999]"
            async for article in self._paginate(date_filtered):
                if yielded >= max_results:
                    break
                uid = article.pmid or article.doi or article.pmcid or article.title
                if uid in seen:
                    continue
                seen.add(uid)
                yield article
                yielded += 1

    async def fetch_full_text_xml(self, pmcid: str) -> Optional[str]:
        """Download full-text XML for a PMC article."""
        url = f"{self.BASE_URL}/article/{pmcid}/textyml"
        data = await self._get_text(url)
        return data

    async def get_citations(self, pmid: str, limit: int = 100) -> List[str]:
        """Return PMIDs that cite the given article."""
        url = f"{self.BASE_URL}/article/MED/{pmid}/citations"
        params = {"format": "json", "pageSize": str(min(limit, 1000))}
        data = await self._get_json(url, params)
        if not data:
            return []
        return [
            c.get("id", "")
            for c in data.get("citationList", {}).get("citation", [])
            if c.get("id")
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _paginate(
        self, query: str
    ) -> AsyncIterator[EuropePMCArticle]:
        cursor = "*"
        while True:
            params = {
                "query": query,
                "format": "json",
                "pageSize": str(self.page_size),
                "resultType": "core",
                "cursorMark": cursor,
                "sort": "CITED desc",
            }
            data = await self._get_json(f"{self.BASE_URL}/search", params)
            if not data:
                break

            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for item in results:
                article = self._parse_article(item)
                if article:
                    if self.fetch_full_text and article.pmcid and article.is_open_access:
                        article.full_text = await self.fetch_full_text_xml(
                            article.pmcid
                        )
                    yield article

            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            await asyncio.sleep(0.1)  # polite pacing

    def _parse_article(self, item: Dict[str, Any]) -> Optional[EuropePMCArticle]:
        import re

        try:
            pmid = item.get("pmid") or item.get("id") if item.get("source") == "MED" else None
            pmcid = item.get("pmcid")
            doi = item.get("doi")
            title = item.get("title", "")
            abstract = item.get("abstractText", "") or item.get("abstract", "")
            journal = item.get("journalTitle") or item.get("journal", {}).get("title", "")

            # year
            year_raw = item.get("pubYear") or item.get("firstPublicationDate", "")[:4]
            try:
                year = int(year_raw) if year_raw else None
            except ValueError:
                year = None

            # authors
            author_list = item.get("authorList", {}).get("author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            authors = [
                a.get("fullName") or f"{a.get('lastName', '')} {a.get('firstName', '')}".strip()
                for a in author_list
            ]

            # open access
            is_oa = str(item.get("isOpenAccess", "N")).upper() == "Y"
            ft_avail = bool(item.get("fullTextUrlList"))

            # MeSH
            mesh_list = item.get("meshHeadingList", {}).get("meshHeading", [])
            if isinstance(mesh_list, dict):
                mesh_list = [mesh_list]
            mesh_terms = [m.get("descriptorName", "") for m in mesh_list]

            # keywords
            kw_list = item.get("keywordList", {}).get("keyword", [])
            if isinstance(kw_list, str):
                kw_list = [kw_list]
            keywords = kw_list if isinstance(kw_list, list) else []

            # grants
            grant_list = item.get("grantsList", {}).get("grant", [])
            if isinstance(grant_list, dict):
                grant_list = [grant_list]
            grant_ids = [g.get("grantId", "") for g in grant_list]

            citation_count = int(item.get("citedByCount", 0) or 0)
            source = item.get("source", "MED")

            # EEG method detection
            full_text_for_scan = f"{title} {abstract}".lower()
            eeg_methods = [
                kw for kw in self.EEG_METHOD_KEYWORDS
                if re.search(kw, full_text_for_scan, re.IGNORECASE)
            ]

            return EuropePMCArticle(
                pmid=str(pmid) if pmid else None,
                pmcid=str(pmcid) if pmcid else None,
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                publication_date=item.get("firstPublicationDate"),
                is_open_access=is_oa,
                full_text_available=ft_avail,
                full_text=None,
                mesh_terms=[m for m in mesh_terms if m],
                keywords=[k for k in keywords if k],
                grant_ids=[g for g in grant_ids if g],
                citation_count=citation_count,
                source=source,
                eeg_methods_mentioned=eeg_methods,
            )

        except Exception as exc:
            logger.warning("Failed to parse Europe PMC article: %s", exc)
            return None

    async def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._session:
            raise RuntimeError("Must use as async context manager")
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                logger.warning("Europe PMC HTTP %d for %s", resp.status, url)
                return None
        except Exception as exc:
            logger.error("Europe PMC request error: %s", exc)
            return None

    async def _get_text(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        if not self._session:
            raise RuntimeError("Must use as async context manager")
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.text()
                return None
        except Exception as exc:
            logger.error("Europe PMC text request error: %s", exc)
            return None
