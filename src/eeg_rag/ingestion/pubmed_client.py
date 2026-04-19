"""
Comprehensive PubMed client for EEG literature retrieval.
Uses E-utilities API with full metadata extraction.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from datetime import datetime, timedelta
import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : ingestion.pubmed_client.PubMedArticle
# Requirement  : `PubMedArticle` class shall be instantiable and expose the documented interface
# Purpose      : Complete PubMed article with all metadata
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
# Verification : Instantiate PubMedArticle with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class PubMedArticle:
    """Complete PubMed article with all metadata."""
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    publication_date: Optional[datetime]
    doi: Optional[str]
    pmc_id: Optional[str]
    mesh_terms: list[str]
    keywords: list[str]
    publication_types: list[str]
    affiliations: list[str]
    references: list[str]  # List of cited PMIDs
    full_text_available: bool
    full_text: Optional[str] = None
    
    # EEG-specific extracted metadata
    eeg_methods: list[str] = field(default_factory=list)
    frequency_bands_mentioned: list[str] = field(default_factory=list)
    erp_components: list[str] = field(default_factory=list)
    electrode_systems: list[str] = field(default_factory=list)
    sample_size: Optional[int] = None
    clinical_conditions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ID           : ingestion.pubmed_client.PubMedClient
# Requirement  : `PubMedClient` class shall be instantiable and expose the documented interface
# Purpose      : Async PubMed client with comprehensive data extraction
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
# Verification : Instantiate PubMedClient with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class PubMedClient:
    """
    Async PubMed client with comprehensive data extraction.
    
    Features:
    - Full metadata extraction including MeSH terms
    - Reference extraction for citation networks
    - Full-text retrieval from PMC when available
    - EEG-specific entity extraction
    - Rate limiting and retry logic
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
    
    # Comprehensive EEG search queries
    EEG_SEARCH_QUERIES = [
        # Core EEG terms
        '"electroencephalography"[MeSH] OR "EEG"[Title/Abstract]',
        '"evoked potentials"[MeSH] OR "event-related potentials"[Title/Abstract]',
        '"brain-computer interface"[MeSH] OR "BCI"[Title/Abstract]',
        
        # Clinical EEG
        '"epilepsy"[MeSH] AND "electroencephalography"[MeSH]',
        '"sleep"[MeSH] AND "polysomnography"[MeSH]',
        '"consciousness disorders"[MeSH] AND "EEG"[Title/Abstract]',
        
        # Cognitive neuroscience
        '"cognition"[MeSH] AND "electroencephalography"[MeSH]',
        '"memory"[MeSH] AND "EEG"[Title/Abstract]',
        '"attention"[MeSH] AND "event-related potentials"[Title/Abstract]',
        
        # Specific ERPs
        '"P300"[Title/Abstract]',
        '"N400"[Title/Abstract]',
        '"mismatch negativity"[Title/Abstract]',
        '"error-related negativity"[Title/Abstract]',
        
        # Oscillations and rhythms
        '"theta rhythm"[Title/Abstract] OR "theta oscillations"[Title/Abstract]',
        '"alpha rhythm"[MeSH] OR "alpha oscillations"[Title/Abstract]',
        '"gamma oscillations"[Title/Abstract]',
        
        # Methods and analysis
        '"EEG signal processing"[Title/Abstract]',
        '"EEG machine learning"[Title/Abstract]',
        '"EEG deep learning"[Title/Abstract]',
        '"EEG connectivity"[Title/Abstract]',
        '"EEG source localization"[Title/Abstract]',
    ]
    
    # EEG entity patterns for extraction
    FREQUENCY_PATTERNS = {
        'delta': r'\bdelta\b.*?(?:rhythm|band|oscillation|wave|activity)',
        'theta': r'\btheta\b.*?(?:rhythm|band|oscillation|wave|activity)',
        'alpha': r'\balpha\b.*?(?:rhythm|band|oscillation|wave|activity)',
        'beta': r'\bbeta\b.*?(?:rhythm|band|oscillation|wave|activity)',
        'gamma': r'\bgamma\b.*?(?:rhythm|band|oscillation|wave|activity)',
    }
    
    ERP_PATTERNS = {
        'P100': r'\bP1(?:00)?\b',
        'N100': r'\bN1(?:00)?\b',
        'P200': r'\bP2(?:00)?\b',
        'N200': r'\bN2(?:00)?\b',
        'P300': r'\bP3(?:00)?(?:a|b)?\b',
        'N400': r'\bN400\b',
        'P600': r'\bP600\b',
        'MMN': r'\b(?:MMN|mismatch negativity)\b',
        'ERN': r'\b(?:ERN|error.related negativity)\b',
        'LPP': r'\b(?:LPP|late positive potential)\b',
        'CNV': r'\b(?:CNV|contingent negative variation)\b',
    }
    
    CLINICAL_PATTERNS = {
        'epilepsy': r'\bepilepsy|seizure|ictal|interictal\b',
        'alzheimers': r'\balzheimer|dementia|cognitive decline\b',
        'parkinsons': r'\bparkinson\b',
        'depression': r'\bdepression|MDD|major depressive\b',
        'schizophrenia': r'\bschizophrenia|psychosis\b',
        'adhd': r'\bADHD|attention.deficit\b',
        'autism': r'\bautism|ASD|autistic\b',
        'sleep_disorders': r'\binsomnia|sleep apnea|narcolepsy\b',
        'stroke': r'\bstroke|cerebrovascular\b',
        'tbi': r'\btraumatic brain injury|TBI|concussion\b',
    }

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.__init__
    # Requirement  : `__init__` shall initialize PubMed client
    # Purpose      : Initialize PubMed client
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : api_key: Optional[str] (default=None); email: str (default='your-email@example.com'); requests_per_second: float (default=3.0); max_concurrent: int (default=5)
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
        api_key: Optional[str] = None,
        email: str = "your-email@example.com",
        requests_per_second: float = 3.0,  # 10/sec with API key, 3/sec without
        max_concurrent: int = 5
    ):
        """
        Initialize PubMed client.
        
        Args:
            api_key: NCBI API key for higher rate limits
            email: Contact email (required by NCBI)
            requests_per_second: Rate limit (10/s with key, 3/s without)
            max_concurrent: Maximum concurrent requests
        """
        self.api_key = api_key
        self.email = email
        self.delay = 1.0 / requests_per_second
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.__aenter__
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
        self._session = aiohttp.ClientSession()
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.__aexit__
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
    # ID           : ingestion.pubmed_client.PubMedClient.close
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
    # ID           : ingestion.pubmed_client.PubMedClient._rate_limit
    # Requirement  : `_rate_limit` shall enforce rate limiting between requests
    # Purpose      : Enforce rate limiting between requests
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
        """Enforce rate limiting between requests."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._build_params
    # Requirement  : `_build_params` shall build request parameters with API key and email
    # Purpose      : Build request parameters with API key and email
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : **kwargs
    # Outputs      : dict
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
    def _build_params(self, **kwargs) -> dict:
        """Build request parameters with API key and email."""
        params = {"email": self.email, "tool": "eeg-rag"}
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(kwargs)
        return params

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._fetch
    # Requirement  : `_fetch` shall fetch from PubMed API with retry logic
    # Purpose      : Fetch from PubMed API with retry logic
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : session: aiohttp.ClientSession; endpoint: str; params: dict
    # Outputs      : str
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
    async def _fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict) -> str:
        """Fetch from PubMed API with retry logic."""
        async with self.semaphore:
            await self._rate_limit()
            url = f"{self.BASE_URL}/{endpoint}"
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.text()

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.search
    # Requirement  : `search` shall search PubMed and return list of PMIDs
    # Purpose      : Search PubMed and return list of PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; max_results: int (default=10000); min_date: Optional[str] (default=None); max_date: Optional[str] (default=None); sort: str (default='relevance')
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
        max_results: int = 10000,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        sort: str = "relevance"
    ) -> list[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            query: PubMed search query
            max_results: Maximum PMIDs to return
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            sort: Sort order ('relevance' or 'date')
            
        Returns:
            List of PMIDs matching the query
        """
        params = self._build_params(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort,
            retmode="json"
        )
        
        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"
        if max_date:
            params["maxdate"] = max_date
            
        async with aiohttp.ClientSession() as session:
            result = await self._fetch(session, "esearch.fcgi", params)
            import json
            data = json.loads(result)
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} articles for query: {query[:50]}...")
            return pmids

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.fetch_articles
    # Requirement  : `fetch_articles` shall fetch full article details for list of PMIDs
    # Purpose      : Fetch full article details for list of PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmids: list[str]; batch_size: int (default=200)
    # Outputs      : AsyncIterator[PubMedArticle]
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
    async def fetch_articles(
        self,
        pmids: list[str],
        batch_size: int = 200
    ) -> AsyncIterator[PubMedArticle]:
        """
        Fetch full article details for list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs to fetch
            batch_size: Number of articles per API request
            
        Yields:
            PubMedArticle objects with full metadata
        """
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i:i + batch_size]
                params = self._build_params(
                    db="pubmed",
                    id=",".join(batch),
                    retmode="xml",
                    rettype="full"
                )
                
                xml_data = await self._fetch(session, "efetch.fcgi", params)
                
                for article in self._parse_articles(xml_data):
                    yield article
                    
                logger.info(f"Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} articles")

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._parse_articles
    # Requirement  : `_parse_articles` shall parse PubMed XML response into article objects
    # Purpose      : Parse PubMed XML response into article objects
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : xml_data: str
    # Outputs      : list[PubMedArticle]
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
    def _parse_articles(self, xml_data: str) -> list[PubMedArticle]:
        """Parse PubMed XML response into article objects."""
        articles = []
        root = ET.fromstring(xml_data)
        
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                article = self._parse_single_article(article_elem)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
                
        return articles

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._parse_single_article
    # Requirement  : `_parse_single_article` shall parse a single PubMed article element
    # Purpose      : Parse a single PubMed article element
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : elem: ET.Element
    # Outputs      : Optional[PubMedArticle]
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
    def _parse_single_article(self, elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse a single PubMed article element."""
        # Extract PMID
        pmid_elem = elem.find(".//PMID")
        if pmid_elem is None:
            return None
        pmid = pmid_elem.text
        
        # Extract citation info
        citation = elem.find(".//MedlineCitation")
        article = citation.find(".//Article") if citation else None
        
        if article is None:
            return None
        
        # Title
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""
        
        # Abstract - handle structured abstracts
        abstract_parts = []
        abstract_elem = article.find(".//Abstract")
        if abstract_elem is not None:
            for text in abstract_elem.findall(".//AbstractText"):
                label = text.get("Label", "")
                content = "".join(text.itertext())
                if label:
                    abstract_parts.append(f"{label}: {content}")
                else:
                    abstract_parts.append(content)
        abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        for author in article.findall(".//Author"):
            lastname = author.findtext("LastName", "")
            forename = author.findtext("ForeName", "")
            if lastname:
                authors.append(f"{forename} {lastname}".strip())
        
        # Journal
        journal = article.findtext(".//Journal/Title", "")
        
        # Publication date
        pub_date = None
        date_elem = article.find(".//PubDate")
        if date_elem is not None:
            year = date_elem.findtext("Year")
            month = date_elem.findtext("Month", "01")
            day = date_elem.findtext("Day", "01")
            if year:
                try:
                    # Handle month names
                    month_map = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    month = month_map.get(month, month)
                    pub_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                except ValueError:
                    pub_date = datetime(int(year), 1, 1)
        
        # DOI
        doi = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break
        
        # PMC ID
        pmc_id = None
        for id_elem in elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "pmc":
                pmc_id = id_elem.text
                break
        
        # MeSH terms
        mesh_terms = []
        for mesh in citation.findall(".//MeshHeading"):
            descriptor = mesh.findtext("DescriptorName", "")
            if descriptor:
                mesh_terms.append(descriptor)
            for qualifier in mesh.findall("QualifierName"):
                if qualifier.text:
                    mesh_terms.append(f"{descriptor}/{qualifier.text}")
        
        # Keywords
        keywords = [kw.text for kw in citation.findall(".//Keyword") if kw.text]
        
        # Publication types
        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
        
        # Affiliations
        affiliations = []
        for aff in article.findall(".//Affiliation"):
            if aff.text:
                affiliations.append(aff.text)
        
        # References (cited PMIDs)
        references = []
        for ref in elem.findall(".//Reference"):
            for ref_id in ref.findall(".//ArticleId"):
                if ref_id.get("IdType") == "pubmed" and ref_id.text:
                    references.append(ref_id.text)
        
        # Create article object
        pub_article = PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            pmc_id=pmc_id,
            mesh_terms=mesh_terms,
            keywords=keywords,
            publication_types=pub_types,
            affiliations=affiliations,
            references=references,
            full_text_available=pmc_id is not None
        )
        
        # Extract EEG-specific entities
        self._extract_eeg_entities(pub_article)
        
        return pub_article

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._extract_eeg_entities
    # Requirement  : `_extract_eeg_entities` shall extract EEG-specific entities from article text
    # Purpose      : Extract EEG-specific entities from article text
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : article: PubMedArticle
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
    def _extract_eeg_entities(self, article: PubMedArticle):
        """Extract EEG-specific entities from article text."""
        text = f"{article.title} {article.abstract}".lower()
        
        # Extract frequency bands
        for band, pattern in self.FREQUENCY_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                article.frequency_bands_mentioned.append(band)
        
        # Extract ERP components
        for component, pattern in self.ERP_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                article.erp_components.append(component)
        
        # Extract clinical conditions
        for condition, pattern in self.CLINICAL_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                article.clinical_conditions.append(condition)
        
        # Extract sample size
        sample_match = re.search(r'(\d+)\s*(?:participants|subjects|patients|controls)', text)
        if sample_match:
            article.sample_size = int(sample_match.group(1))
        
        # Detect electrode systems
        if re.search(r'10-20|10-10|10-5', text):
            article.electrode_systems.append('10-20 system')
        if re.search(r'64.channel|64-channel|128.channel|256.channel', text):
            article.electrode_systems.append('high-density')
        if re.search(r'EGI|BioSemi|Neuroscan|Brain Products|g\.tec', text, re.IGNORECASE):
            article.electrode_systems.append('commercial system')

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.fetch_full_text_pmc
    # Requirement  : `fetch_full_text_pmc` shall fetch full text from PubMed Central
    # Purpose      : Fetch full text from PubMed Central
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmc_id: str
    # Outputs      : Optional[str]
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
    async def fetch_full_text_pmc(self, pmc_id: str) -> Optional[str]:
        """
        Fetch full text from PubMed Central.
        
        Args:
            pmc_id: PMC ID (e.g., 'PMC1234567')
            
        Returns:
            Full text content or None if not available
        """
        if not pmc_id:
            return None
            
        # Normalize PMC ID
        if not pmc_id.startswith("PMC"):
            pmc_id = f"PMC{pmc_id}"
        
        async with aiohttp.ClientSession() as session:
            # First, get the article XML URL from OA service
            params = {"id": pmc_id, "format": "xml"}
            
            try:
                async with session.get(self.PMC_OA_URL, params=params) as response:
                    if response.status != 200:
                        return None
                    oa_xml = await response.text()
                
                # Parse OA response to get download link
                root = ET.fromstring(oa_xml)
                link = root.find(".//link[@format='xml']")
                
                if link is None:
                    return None
                    
                xml_url = link.get("href")
                
                # Fetch full text XML
                async with session.get(xml_url) as response:
                    if response.status != 200:
                        return None
                    full_xml = await response.text()
                
                # Parse and extract text
                return self._extract_text_from_pmc_xml(full_xml)
                
            except Exception as e:
                logger.warning(f"Error fetching full text for {pmc_id}: {e}")
                return None

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient._extract_text_from_pmc_xml
    # Requirement  : `_extract_text_from_pmc_xml` shall extract readable text from PMC XML
    # Purpose      : Extract readable text from PMC XML
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : xml_data: str
    # Outputs      : str
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
    def _extract_text_from_pmc_xml(self, xml_data: str) -> str:
        """Extract readable text from PMC XML."""
        root = ET.fromstring(xml_data)
        
        sections = []
        
        # Extract body sections
        for sec in root.findall(".//sec"):
            title = sec.findtext("title", "")
            paragraphs = []
            
            for p in sec.findall(".//p"):
                text = "".join(p.itertext())
                if text.strip():
                    paragraphs.append(text.strip())
            
            if paragraphs:
                section_text = f"\n\n## {title}\n\n" if title else "\n\n"
                section_text += "\n\n".join(paragraphs)
                sections.append(section_text)
        
        return "".join(sections)

    # ---------------------------------------------------------------------------
    # ID           : ingestion.pubmed_client.PubMedClient.collect_eeg_corpus
    # Requirement  : `collect_eeg_corpus` shall collect comprehensive EEG corpus from PubMed
    # Purpose      : Collect comprehensive EEG corpus from PubMed
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : years_back: int (default=10); max_per_query: int (default=5000); include_full_text: bool (default=True)
    # Outputs      : AsyncIterator[PubMedArticle]
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
        years_back: int = 10,
        max_per_query: int = 5000,
        include_full_text: bool = True
    ) -> AsyncIterator[PubMedArticle]:
        """
        Collect comprehensive EEG corpus from PubMed.
        
        Args:
            years_back: Number of years of literature to collect
            max_per_query: Maximum articles per search query
            include_full_text: Whether to fetch full text from PMC
            
        Yields:
            PubMedArticle objects with all available data
        """
        min_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y/%m/%d")
        seen_pmids = set()
        
        for query in self.EEG_SEARCH_QUERIES:
            logger.info(f"Searching: {query}")
            
            pmids = await self.search(
                query=query,
                max_results=max_per_query,
                min_date=min_date
            )
            
            # Filter out already seen PMIDs
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)
            
            async for article in self.fetch_articles(new_pmids):
                # Optionally fetch full text
                if include_full_text and article.pmc_id:
                    article.full_text = await self.fetch_full_text_pmc(article.pmc_id)
                
                yield article
        
        logger.info(f"Total unique articles collected: {len(seen_pmids)}")
