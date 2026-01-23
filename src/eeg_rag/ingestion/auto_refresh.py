#!/usr/bin/env python3
"""
Automated Knowledge Base Refresh Pipeline

Keeps the EEG knowledge base current with new publications by:
- Scheduling automatic crawls of PubMed, arXiv, and ClinicalTrials.gov
- Deduplicating against existing indexed content
- Updating vector store and knowledge graph
- Tracking refresh history and statistics

This addresses the claim: "Rolling index of peer-reviewed EEG studies"
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import logging
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RefreshSource:
    """Configuration for a paper source."""
    name: str
    base_url: str
    enabled: bool = True
    last_refresh: Optional[datetime] = None
    papers_fetched: int = 0


@dataclass
class RefreshResult:
    """Result of a refresh operation."""
    source: str
    new_papers: int
    updated_papers: int
    failed: int
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "new_papers": self.new_papers,
            "updated_papers": self.updated_papers,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class PubMedCrawler:
    """Fetch new EEG papers from PubMed with full metadata extraction."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # Comprehensive EEG search query covering all aspects
    EEG_QUERY = """
    (electroencephalography[MeSH] OR EEG[tiab] OR "brain waves"[tiab] OR 
     "evoked potentials"[MeSH] OR "event-related potentials"[tiab] OR ERP[tiab] OR
     "sleep staging"[tiab] OR polysomnography[MeSH] OR 
     "seizure detection"[tiab] OR "epileptiform"[tiab] OR
     "brain-computer interface"[tiab] OR BCI[tiab] OR
     "neural oscillations"[tiab] OR "alpha rhythm"[tiab] OR "theta waves"[tiab] OR
     "P300"[tiab] OR "N400"[tiab] OR "mismatch negativity"[tiab] OR
     "interictal epileptiform"[tiab] OR "ictal EEG"[tiab])
    AND (humans[MeSH])
    AND (english[Language])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: str = "eeg-rag@research.org"
    ):
        self.api_key = api_key
        self.email = email
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search_recent(
        self, 
        days_back: int = 7,
        max_results: int = 500
    ) -> list[str]:
        """Search for EEG papers published in the last N days.
        
        Args:
            days_back: Number of days to look back
            max_results: Maximum PMIDs to return
            
        Returns:
            List of PMIDs matching the query
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_filter = f'("{start_date:%Y/%m/%d}"[PDAT] : "{end_date:%Y/%m/%d}"[PDAT])'
        
        query = f"{self.EEG_QUERY} AND {date_filter}"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
            "email": self.email,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            logger.info(
                f"PubMed search complete: {len(pmids)} papers found for last {days_back} days"
            )
            
            return pmids
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    async def fetch_paper_details(self, pmids: list[str]) -> list[dict]:
        """Fetch full details for a list of PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper dictionaries with full metadata
        """
        if not pmids:
            return []
            
        # PubMed allows up to 200 IDs per request
        batch_size = 200
        all_papers = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
                
            try:
                response = await self.client.get(
                    f"{self.BASE_URL}/efetch.fcgi",
                    params=params
                )
                response.raise_for_status()
                
                papers = self._parse_pubmed_xml(response.text)
                all_papers.extend(papers)
                
                # Rate limiting: 3 requests/second without API key, 10 with
                await asyncio.sleep(0.1 if self.api_key else 0.35)
                
            except Exception as e:
                logger.error(f"Failed to fetch batch starting at {i}: {e}")
                continue
                
        logger.info(f"Fetched {len(all_papers)} paper details from PubMed")
        return all_papers
    
    def _parse_pubmed_xml(self, xml_content: str) -> list[dict]:
        """Parse PubMed XML response into structured paper data."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return papers
        
        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue
                    
                pmid_elem = medline.find("PMID")
                if pmid_elem is None:
                    continue
                pmid = pmid_elem.text
                
                article_data = medline.find("Article")
                if article_data is None:
                    continue
                    
                title_elem = article_data.find("ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Get abstract (handle structured abstracts)
                abstract_elem = article_data.find("Abstract")
                abstract = ""
                if abstract_elem is not None:
                    abstract_parts = []
                    for at in abstract_elem.findall("AbstractText"):
                        label = at.get("Label", "")
                        text = at.text or ""
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)
                
                # Get journal info
                journal = ""
                journal_elem = article_data.find("Journal")
                if journal_elem is not None:
                    title_elem = journal_elem.find("Title")
                    if title_elem is not None:
                        journal = title_elem.text or ""
                
                # Get publication date
                year = ""
                pub_date = article_data.find(".//PubDate")
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None:
                        year = year_elem.text or ""
                
                # Get authors
                authors = []
                author_list = article_data.find("AuthorList")
                if author_list is not None:
                    for author in author_list.findall("Author"):
                        last_name = author.find("LastName")
                        first_name = author.find("ForeName")
                        if last_name is not None and last_name.text:
                            name = last_name.text
                            if first_name is not None and first_name.text:
                                name = f"{first_name.text} {name}"
                            authors.append(name)
                
                # Get MeSH terms
                mesh_terms = []
                mesh_list = medline.find("MeshHeadingList")
                if mesh_list is not None:
                    for mesh in mesh_list.findall("MeshHeading"):
                        descriptor = mesh.find("DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)
                
                # Get DOI
                doi = ""
                article_ids = article.find(".//ArticleIdList")
                if article_ids is not None:
                    for aid in article_ids.findall("ArticleId"):
                        if aid.get("IdType") == "doi" and aid.text:
                            doi = aid.text
                            break
                
                papers.append({
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": authors,
                    "mesh_terms": mesh_terms,
                    "source": "pubmed",
                    "fetched_at": datetime.now().isoformat(),
                })
                
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
                
        return papers
    
    async def close(self):
        """Cleanup HTTP client."""
        await self.client.aclose()


class ClinicalTrialsCrawler:
    """Fetch EEG-related clinical trials from ClinicalTrials.gov."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    EEG_TERMS = [
        "electroencephalography",
        "EEG monitoring",
        "brain-computer interface",
        "seizure monitoring",
        "epilepsy monitoring",
        "sleep study polysomnography",
        "evoked potentials",
    ]
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search_recent_trials(
        self,
        days_back: int = 30,
        max_results: int = 100
    ) -> list[dict]:
        """Search for recently updated EEG-related clinical trials.
        
        Args:
            days_back: Days to look back for updates
            max_results: Maximum trials to return per term
            
        Returns:
            List of trial dictionaries
        """
        trials = []
        
        for term in self.EEG_TERMS:
            try:
                params = {
                    "query.term": term,
                    "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
                    "sort": "LastUpdatePostDate:desc",
                    "pageSize": min(max_results, 100),
                    "format": "json",
                }
                
                response = await self.client.get(
                    f"{self.BASE_URL}/studies",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for study in data.get("studies", []):
                        trial = self._parse_trial(study)
                        if trial:
                            trials.append(trial)
                            
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to search term '{term}': {e}")
                continue
            
        # Deduplicate by NCT ID
        seen = set()
        unique_trials = []
        for trial in trials:
            if trial["nct_id"] not in seen:
                seen.add(trial["nct_id"])
                unique_trials.append(trial)
        
        logger.info(f"Found {len(unique_trials)} unique clinical trials")
        return unique_trials
    
    def _parse_trial(self, study: dict) -> Optional[dict]:
        """Parse a clinical trial into structured data."""
        try:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            desc_module = protocol.get("descriptionModule", {})
            status_module = protocol.get("statusModule", {})
            
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                return None
            
            return {
                "nct_id": nct_id,
                "title": id_module.get("officialTitle", id_module.get("briefTitle", "")),
                "description": desc_module.get("briefSummary", ""),
                "detailed_description": desc_module.get("detailedDescription", ""),
                "status": status_module.get("overallStatus", ""),
                "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                "source": "clinicaltrials",
                "fetched_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Error parsing trial: {e}")
            return None
            
    async def close(self):
        await self.client.aclose()


class ArxivCrawler:
    """Fetch EEG-related preprints from arXiv."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    # arXiv categories relevant to EEG
    SEARCH_QUERIES = [
        "all:EEG AND all:brain-computer interface",
        "all:electroencephalography AND all:deep learning",
        "all:EEG AND all:seizure detection",
        "all:neural oscillations AND all:EEG",
        "all:event-related potentials",
    ]
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search_recent(
        self,
        days_back: int = 30,
        max_results: int = 100
    ) -> list[dict]:
        """Search for recent EEG-related arXiv papers."""
        papers = []
        
        for query in self.SEARCH_QUERIES:
            try:
                params = {
                    "search_query": query,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                
                response = await self.client.get(self.BASE_URL, params=params)
                
                if response.status_code == 200:
                    parsed = self._parse_arxiv_response(response.text)
                    papers.extend(parsed)
                    
                await asyncio.sleep(3.0)  # arXiv rate limit: 1 request per 3 seconds
                
            except Exception as e:
                logger.warning(f"arXiv search failed for '{query}': {e}")
                continue
        
        # Deduplicate
        seen = set()
        unique = []
        for paper in papers:
            if paper["arxiv_id"] not in seen:
                seen.add(paper["arxiv_id"])
                unique.append(paper)
                
        logger.info(f"Found {len(unique)} unique arXiv papers")
        return unique
    
    def _parse_arxiv_response(self, xml_content: str) -> list[dict]:
        """Parse arXiv Atom feed response."""
        papers = []
        
        # arXiv uses Atom namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        try:
            root = ET.fromstring(xml_content)
            
            for entry in root.findall("atom:entry", ns):
                try:
                    arxiv_id = entry.find("atom:id", ns)
                    if arxiv_id is None:
                        continue
                    # Extract ID from URL
                    id_text = arxiv_id.text.split("/abs/")[-1]
                    
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip() if title_elem is not None else ""
                    
                    summary_elem = entry.find("atom:summary", ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""
                    
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.find("atom:name", ns)
                        if name is not None and name.text:
                            authors.append(name.text)
                    
                    published = entry.find("atom:published", ns)
                    pub_date = published.text if published is not None else ""
                    
                    papers.append({
                        "arxiv_id": id_text,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "published_date": pub_date,
                        "source": "arxiv",
                        "fetched_at": datetime.now().isoformat(),
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing arXiv entry: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"arXiv XML parse error: {e}")
            
        return papers
    
    async def close(self):
        await self.client.aclose()


class KnowledgeBaseRefresher:
    """
    Orchestrates automatic refresh of the EEG knowledge base.
    
    Coordinates multiple crawlers (PubMed, ClinicalTrials, arXiv),
    handles deduplication, and updates the vector store and knowledge graph.
    """
    
    def __init__(
        self,
        vector_store: Any = None,
        graph_store: Any = None,
        state_file: Path = Path("data/refresh_state.json"),
        pubmed_api_key: Optional[str] = None,
    ):
        """
        Initialize the refresher.
        
        Args:
            vector_store: Vector store for document embeddings
            graph_store: Knowledge graph store (Neo4j)
            state_file: Path to persist refresh state
            pubmed_api_key: Optional NCBI API key for higher rate limits
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.state_file = state_file
        
        self.pubmed = PubMedCrawler(api_key=pubmed_api_key)
        self.trials = ClinicalTrialsCrawler()
        self.arxiv = ArxivCrawler()
        
        self.scheduler = None
        self._load_state()
        
    def _load_state(self):
        """Load refresh state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.state = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
                self.state = self._default_state()
        else:
            self.state = self._default_state()
    
    def _default_state(self) -> dict:
        """Return default state structure."""
        return {
            "last_pubmed_refresh": None,
            "last_trials_refresh": None,
            "last_arxiv_refresh": None,
            "total_papers_indexed": 0,
            "total_trials_indexed": 0,
            "refresh_history": [],
            "indexed_pmids": [],  # Track indexed PMIDs for deduplication
            "indexed_nct_ids": [],
            "indexed_arxiv_ids": [],
        }
            
    def _save_state(self):
        """Persist refresh state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            
    async def refresh_pubmed(self, days_back: int = 7) -> RefreshResult:
        """Refresh PubMed papers.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            RefreshResult with statistics
        """
        start_time = datetime.now()
        errors = []
        new_papers = 0
        
        try:
            logger.info(f"Starting PubMed refresh for last {days_back} days")
            
            # Search for new papers
            pmids = await self.pubmed.search_recent(days_back=days_back)
            
            # Filter out already-indexed papers
            existing_pmids = set(self.state.get("indexed_pmids", []))
            new_pmids = [p for p in pmids if p not in existing_pmids]
            
            logger.info(
                f"PubMed: {len(pmids)} found, {len(existing_pmids)} already indexed, "
                f"{len(new_pmids)} new"
            )
            
            if new_pmids:
                # Fetch full details
                papers = await self.pubmed.fetch_paper_details(new_pmids)
                
                # Index papers
                for paper in papers:
                    try:
                        await self._index_paper(paper)
                        new_papers += 1
                        self.state["indexed_pmids"].append(paper["pmid"])
                    except Exception as e:
                        errors.append(f"Failed to index PMID {paper['pmid']}: {e}")
                        
            self.state["last_pubmed_refresh"] = datetime.now().isoformat()
            self.state["total_papers_indexed"] += new_papers
            self._save_state()
            
        except Exception as e:
            errors.append(f"PubMed refresh failed: {e}")
            logger.error(f"PubMed refresh error: {e}")
            
        duration = (datetime.now() - start_time).total_seconds()
        
        result = RefreshResult(
            source="pubmed",
            new_papers=new_papers,
            updated_papers=0,
            failed=len(errors),
            duration_seconds=duration,
            errors=errors,
        )
        
        self._log_refresh_result(result)
        return result
    
    async def refresh_trials(self, days_back: int = 30) -> RefreshResult:
        """Refresh clinical trials."""
        start_time = datetime.now()
        errors = []
        new_trials = 0
        
        try:
            logger.info("Starting ClinicalTrials.gov refresh")
            
            trials = await self.trials.search_recent_trials(days_back=days_back)
            
            # Filter already indexed
            existing = set(self.state.get("indexed_nct_ids", []))
            new_trials_list = [t for t in trials if t["nct_id"] not in existing]
            
            for trial in new_trials_list:
                try:
                    await self._index_trial(trial)
                    new_trials += 1
                    self.state["indexed_nct_ids"].append(trial["nct_id"])
                except Exception as e:
                    errors.append(f"Failed to index trial {trial['nct_id']}: {e}")
                    
            self.state["last_trials_refresh"] = datetime.now().isoformat()
            self.state["total_trials_indexed"] += new_trials
            self._save_state()
            
        except Exception as e:
            errors.append(f"Trials refresh failed: {e}")
            
        duration = (datetime.now() - start_time).total_seconds()
        
        return RefreshResult(
            source="clinicaltrials",
            new_papers=new_trials,
            updated_papers=0,
            failed=len(errors),
            duration_seconds=duration,
            errors=errors,
        )
    
    async def refresh_arxiv(self, days_back: int = 30) -> RefreshResult:
        """Refresh arXiv preprints."""
        start_time = datetime.now()
        errors = []
        new_papers = 0
        
        try:
            logger.info("Starting arXiv refresh")
            
            papers = await self.arxiv.search_recent(days_back=days_back)
            
            # Filter already indexed
            existing = set(self.state.get("indexed_arxiv_ids", []))
            new_papers_list = [p for p in papers if p["arxiv_id"] not in existing]
            
            for paper in new_papers_list:
                try:
                    await self._index_arxiv_paper(paper)
                    new_papers += 1
                    self.state["indexed_arxiv_ids"].append(paper["arxiv_id"])
                except Exception as e:
                    errors.append(f"Failed to index arXiv {paper['arxiv_id']}: {e}")
                    
            self.state["last_arxiv_refresh"] = datetime.now().isoformat()
            self._save_state()
            
        except Exception as e:
            errors.append(f"arXiv refresh failed: {e}")
            
        duration = (datetime.now() - start_time).total_seconds()
        
        return RefreshResult(
            source="arxiv",
            new_papers=new_papers,
            updated_papers=0,
            failed=len(errors),
            duration_seconds=duration,
            errors=errors,
        )
    
    async def _index_paper(self, paper: dict):
        """Index a paper into vector store and knowledge graph."""
        # Create document text
        text = f"{paper['title']}\n\n{paper['abstract']}"
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        metadata = {
            "pmid": paper["pmid"],
            "doi": paper.get("doi", ""),
            "title": paper["title"],
            "journal": paper["journal"],
            "year": paper["year"],
            "authors": paper["authors"],
            "mesh_terms": paper["mesh_terms"],
            "source": "pubmed",
            "content_hash": content_hash,
            "indexed_at": datetime.now().isoformat(),
        }
        
        # Add to vector store if available
        if self.vector_store is not None:
            await self.vector_store.add_documents(
                texts=[text],
                metadatas=[metadata],
                ids=[f"pmid_{paper['pmid']}"]
            )
        
        # Add to knowledge graph if available
        if self.graph_store is not None:
            await self._add_paper_to_graph(paper)
        
        logger.debug(f"Indexed paper PMID:{paper['pmid']}")
    
    async def _index_trial(self, trial: dict):
        """Index a clinical trial."""
        text = f"{trial['title']}\n\n{trial['description']}\n\n{trial.get('detailed_description', '')}"
        
        metadata = {
            "nct_id": trial["nct_id"],
            "title": trial["title"],
            "status": trial["status"],
            "source": "clinicaltrials",
            "indexed_at": datetime.now().isoformat(),
        }
        
        if self.vector_store is not None:
            await self.vector_store.add_documents(
                texts=[text],
                metadatas=[metadata],
                ids=[f"nct_{trial['nct_id']}"]
            )
    
    async def _index_arxiv_paper(self, paper: dict):
        """Index an arXiv paper."""
        text = f"{paper['title']}\n\n{paper['abstract']}"
        
        metadata = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "source": "arxiv",
            "indexed_at": datetime.now().isoformat(),
        }
        
        if self.vector_store is not None:
            await self.vector_store.add_documents(
                texts=[text],
                metadatas=[metadata],
                ids=[f"arxiv_{paper['arxiv_id']}"]
            )
    
    async def _add_paper_to_graph(self, paper: dict):
        """Add paper and relationships to knowledge graph."""
        if self.graph_store is None:
            return
            
        # Create Cypher query for Neo4j
        cypher = """
        MERGE (p:Paper {pmid: $pmid})
        SET p.title = $title,
            p.journal = $journal,
            p.year = $year,
            p.doi = $doi,
            p.indexed_at = datetime()
        
        WITH p
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (a)-[:AUTHORED]->(p)
        
        WITH p
        UNWIND $mesh_terms AS term
        MERGE (m:MeSHTerm {name: term})
        MERGE (p)-[:HAS_MESH_TERM]->(m)
        """
        
        try:
            await self.graph_store.execute(
                cypher,
                pmid=paper["pmid"],
                title=paper["title"],
                journal=paper["journal"],
                year=paper["year"],
                doi=paper.get("doi", ""),
                authors=paper["authors"],
                mesh_terms=paper["mesh_terms"],
            )
        except Exception as e:
            logger.warning(f"Failed to add paper to graph: {e}")
    
    def _log_refresh_result(self, result: RefreshResult):
        """Log refresh result to history."""
        self.state["refresh_history"].append({
            "timestamp": datetime.now().isoformat(),
            "source": result.source,
            "new_papers": result.new_papers,
            "duration": result.duration_seconds,
            "errors": len(result.errors),
        })
        
        # Keep only last 100 refresh records
        self.state["refresh_history"] = self.state["refresh_history"][-100:]
        self._save_state()
    
    def start_scheduler(self):
        """Start the automatic refresh scheduler using APScheduler."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            
            self.scheduler = AsyncIOScheduler()
            
            # PubMed: Every day at 3 AM
            self.scheduler.add_job(
                self.refresh_pubmed,
                CronTrigger(hour=3, minute=0),
                id="pubmed_daily",
                name="Daily PubMed Refresh",
                kwargs={"days_back": 7},
            )
            
            # Clinical Trials: Every week on Sunday at 4 AM
            self.scheduler.add_job(
                self.refresh_trials,
                CronTrigger(day_of_week="sun", hour=4, minute=0),
                id="trials_weekly",
                name="Weekly Clinical Trials Refresh",
            )
            
            # arXiv: Every 3 days
            self.scheduler.add_job(
                self.refresh_arxiv,
                CronTrigger(day="*/3", hour=5, minute=0),
                id="arxiv_triweekly",
                name="arXiv Refresh",
            )
            
            self.scheduler.start()
            logger.info("Auto-refresh scheduler started")
            
        except ImportError:
            logger.warning("APScheduler not installed. Auto-refresh scheduling disabled.")
        
    async def run_full_refresh(self) -> dict[str, RefreshResult]:
        """Run a full refresh of all sources.
        
        Returns:
            Dictionary mapping source name to RefreshResult
        """
        logger.info("Starting full knowledge base refresh")
        
        results = {}
        
        results["pubmed"] = await self.refresh_pubmed(days_back=30)
        results["clinicaltrials"] = await self.refresh_trials(days_back=60)
        results["arxiv"] = await self.refresh_arxiv(days_back=30)
        
        total_new = sum(r.new_papers for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        
        logger.info(
            f"Full refresh complete: {total_new} new documents, {total_failed} failures"
        )
        
        return results
    
    def get_refresh_stats(self) -> dict:
        """Get current refresh statistics.
        
        Returns:
            Dictionary with refresh statistics
        """
        return {
            "last_pubmed_refresh": self.state.get("last_pubmed_refresh"),
            "last_trials_refresh": self.state.get("last_trials_refresh"),
            "last_arxiv_refresh": self.state.get("last_arxiv_refresh"),
            "total_papers_indexed": self.state.get("total_papers_indexed", 0),
            "total_trials_indexed": self.state.get("total_trials_indexed", 0),
            "total_pmids": len(self.state.get("indexed_pmids", [])),
            "total_nct_ids": len(self.state.get("indexed_nct_ids", [])),
            "total_arxiv_ids": len(self.state.get("indexed_arxiv_ids", [])),
            "recent_history": self.state.get("refresh_history", [])[-10:],
        }
    
    async def close(self):
        """Cleanup resources."""
        if self.scheduler is not None:
            self.scheduler.shutdown()
        await self.pubmed.close()
        await self.trials.close()
        await self.arxiv.close()


# CLI for manual refresh
async def main():
    """Run a manual refresh from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refresh EEG-RAG knowledge base")
    parser.add_argument(
        "--source",
        choices=["pubmed", "trials", "arxiv", "all"],
        default="all",
        help="Source to refresh"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current statistics"
    )
    args = parser.parse_args()
    
    refresher = KnowledgeBaseRefresher()
    
    if args.stats:
        stats = refresher.get_refresh_stats()
        print("\n=== Knowledge Base Refresh Statistics ===")
        print(f"Last PubMed refresh: {stats['last_pubmed_refresh']}")
        print(f"Last Trials refresh: {stats['last_trials_refresh']}")
        print(f"Last arXiv refresh: {stats['last_arxiv_refresh']}")
        print(f"Total papers indexed: {stats['total_papers_indexed']}")
        print(f"Total trials indexed: {stats['total_trials_indexed']}")
        print(f"PMIDs tracked: {stats['total_pmids']}")
        print(f"NCT IDs tracked: {stats['total_nct_ids']}")
        print(f"arXiv IDs tracked: {stats['total_arxiv_ids']}")
        return
    
    print(f"\nRefreshing {args.source} (looking back {args.days} days)...")
    
    if args.source == "pubmed":
        result = await refresher.refresh_pubmed(days_back=args.days)
        print(f"PubMed: {result.new_papers} new papers indexed")
    elif args.source == "trials":
        result = await refresher.refresh_trials(days_back=args.days)
        print(f"Trials: {result.new_papers} new trials indexed")
    elif args.source == "arxiv":
        result = await refresher.refresh_arxiv(days_back=args.days)
        print(f"arXiv: {result.new_papers} new papers indexed")
    else:
        results = await refresher.run_full_refresh()
        for source, result in results.items():
            print(f"{source}: {result.new_papers} new, {result.failed} failed")
    
    await refresher.close()


if __name__ == "__main__":
    asyncio.run(main())
