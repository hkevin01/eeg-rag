#!/usr/bin/env python3
# scripts/build_metadata_index.py
"""
Build the lightweight metadata index for distribution.

This script is run by maintainers to create the compressed index
that ships with the repository. Users don't need to run this.

The index contains only:
- PMIDs, DOIs, OpenAlex IDs, arXiv IDs (identifiers)
- Titles (for search)
- Years, keywords (for filtering)

Full abstracts/content are NOT stored - they're fetched on-demand.

Multi-Source Strategy:
- OpenAlex: Primary source (~300K EEG papers)
- PubMed: Clinical/medical focus (~400K papers)
- Europe PMC: European sources + preprints (~100K unique)
- arXiv: Machine learning/BCI papers (~10K)

Total target: 500K+ unique EEG-related papers

Usage:
    python scripts/build_metadata_index.py --target 500000
    python scripts/build_metadata_index.py --quick  # 50K for testing
    python scripts/build_metadata_index.py --source pubmed --target 100000
"""

import argparse
import json
import logging
import sys
import time
import asyncio
import aiohttp
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.db.metadata_index import MetadataIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# EEG-focused search queries for comprehensive coverage
# Expanded list for 500K+ paper coverage
EEG_QUERIES = [
    # Core EEG terms (high yield)
    "electroencephalography",
    "electroencephalogram",
    "EEG signal processing",
    "EEG analysis",
    "brain waves",
    "neural oscillations",
    "brain electrical activity",
    "scalp recording",
    "electrophysiology brain",
    
    # Frequency bands
    "alpha rhythm EEG",
    "alpha oscillation",
    "beta oscillation brain",
    "theta oscillation brain", 
    "theta rhythm",
    "delta waves sleep",
    "gamma oscillation cognition",
    "gamma band activity",
    "mu rhythm",
    "sensorimotor rhythm",
    
    # Clinical applications (high yield)
    "epilepsy EEG",
    "seizure detection EEG",
    "epileptiform discharge",
    "interictal epileptiform",
    "ictal EEG",
    "status epilepticus",
    "temporal lobe epilepsy",
    "absence seizure",
    "generalized epilepsy",
    "focal epilepsy",
    "sleep disorder EEG",
    "sleep staging polysomnography",
    "insomnia EEG",
    "sleep apnea EEG",
    "REM sleep",
    "slow wave sleep",
    
    # ERPs (Event-Related Potentials)
    "event-related potential",
    "evoked potential",
    "P300 EEG",
    "P300 component",
    "N400 language",
    "N400 semantic",
    "mismatch negativity",
    "MMN auditory",
    "error-related negativity",
    "ERN brain",
    "N170 face",
    "P100 visual",
    "N100 auditory",
    "contingent negative variation",
    "readiness potential",
    "lateralized readiness potential",
    
    # BCI (Brain-Computer Interface)
    "brain-computer interface",
    "brain machine interface",
    "motor imagery EEG",
    "motor imagery classification",
    "SSVEP BCI",
    "steady state visual evoked",
    "P300 speller",
    "EEG-based control",
    "neurofeedback",
    "biofeedback EEG",
    
    # Cognitive neuroscience
    "working memory EEG",
    "attention EEG",
    "selective attention brain",
    "cognitive load EEG",
    "mental workload EEG",
    "emotion recognition EEG",
    "affective computing EEG",
    "decision making EEG",
    "executive function EEG",
    "memory encoding EEG",
    "learning brain activity",
    
    # Methods and processing
    "independent component analysis EEG",
    "ICA EEG",
    "artifact removal EEG",
    "EEG artifact",
    "eye movement artifact",
    "muscle artifact EEG",
    "source localization EEG",
    "dipole source",
    "EEG preprocessing",
    "EEG filtering",
    "connectivity analysis EEG",
    "coherence EEG",
    "phase synchronization EEG",
    "Granger causality EEG",
    "time-frequency analysis EEG",
    "wavelet EEG",
    "spectral analysis EEG",
    "power spectral density EEG",
    "EEG microstate",
    
    # Disorders and clinical conditions
    "ADHD EEG",
    "attention deficit EEG",
    "autism spectrum EEG",
    "ASD brain",
    "depression EEG",
    "major depressive disorder EEG",
    "anxiety EEG",
    "schizophrenia EEG",
    "psychosis EEG",
    "Alzheimer EEG",
    "dementia EEG",
    "mild cognitive impairment EEG",
    "Parkinson EEG",
    "stroke EEG",
    "traumatic brain injury EEG",
    "TBI brain",
    "coma EEG",
    "disorders of consciousness",
    "vegetative state brain",
    "migraine EEG",
    "headache brain",
    "chronic pain EEG",
    "addiction EEG",
    "substance use disorder brain",
    
    # Technical/hardware
    "high-density EEG",
    "hdEEG",
    "dry electrode EEG",
    "mobile EEG",
    "portable EEG",
    "wireless EEG",
    "wearable EEG",
    "real-time EEG",
    "EEG amplifier",
    "10-20 system electrode",
    "EEG montage",
    "reference electrode EEG",
    
    # Machine learning / AI
    "deep learning EEG",
    "CNN EEG classification",
    "convolutional neural network EEG",
    "recurrent neural network EEG",
    "LSTM EEG",
    "transformer EEG",
    "EEG decoding",
    "neural network EEG",
    "machine learning EEG",
    "automatic EEG",
    "EEG classification",
    "EEG feature extraction",
    "transfer learning EEG",
    "domain adaptation EEG",
    
    # Specific paradigms
    "resting state EEG",
    "eyes closed EEG",
    "eyes open EEG",
    "auditory evoked potential",
    "AEP brain",
    "visual evoked potential",
    "VEP brain",
    "somatosensory evoked",
    "SEP brain",
    "oddball paradigm",
    "go/nogo task EEG",
    "flanker task EEG",
    "stroop task EEG",
    "n-back task EEG",
    
    # Combined modalities
    "EEG fMRI",
    "simultaneous EEG fMRI",
    "EEG MEG",
    "MEG magnetoencephalography",
    "EEG NIRS",
    "near infrared spectroscopy brain",
    "multimodal neuroimaging",
    
    # Development and aging
    "neonatal EEG",
    "newborn EEG",
    "pediatric EEG",
    "infant EEG",
    "child EEG",
    "adolescent brain",
    "aging brain EEG",
    "elderly EEG",
    "brain development",
    "neurodevelopment",
    
    # Anesthesia and consciousness
    "anesthesia EEG",
    "depth of anesthesia",
    "bispectral index",
    "consciousness EEG",
    "awareness monitoring",
    "sedation EEG",
    
    # Neurology specific
    "encephalopathy EEG",
    "metabolic encephalopathy",
    "hepatic encephalopathy EEG",
    "triphasic waves",
    "periodic lateralized epileptiform",
    "PLEDS",
    "burst suppression",
    "isoelectric EEG",
    "brain death EEG",
    
    # Advanced analysis
    "EEG complexity",
    "entropy EEG",
    "fractal analysis EEG",
    "nonlinear dynamics EEG",
    "chaos EEG",
    "network analysis EEG",
    "graph theory brain",
    "small world network brain",
    "functional connectivity EEG",
    "effective connectivity brain",
    
    # Language and speech
    "language processing EEG",
    "speech perception EEG",
    "reading EEG",
    "semantic processing brain",
    "syntactic processing brain",
    "bilingual brain",
    
    # Motor and movement
    "movement related potential",
    "motor cortex EEG",
    "movement execution brain",
    "gait EEG",
    "tremor EEG",
    "voluntary movement brain",
    
    # Sensory processing
    "auditory processing brain",
    "visual processing EEG",
    "somatosensory processing",
    "multisensory integration brain",
    "pain processing EEG",
]

# Additional queries for PubMed (MeSH-focused)
PUBMED_MESH_QUERIES = [
    "electroencephalography[MeSH]",
    "evoked potentials[MeSH]",
    "brain waves[MeSH]",
    "epilepsy[MeSH] AND EEG",
    "sleep stages[MeSH]",
    "brain-computer interfaces[MeSH]",
    "event-related potentials, P300[MeSH]",
    "alpha rhythm[MeSH]",
    "theta rhythm[MeSH]",
    "cortical synchronization[MeSH]",
]


def fetch_openalex_works(query: str, per_page: int = 200, max_results: int = 10000) -> Iterator[Dict[str, Any]]:
    """
    Fetch works from OpenAlex API (metadata only).
    
    Only extracts identifiers and titles - no abstracts.
    """
    import urllib.request
    import urllib.parse
    from time import sleep
    
    base_url = "https://api.openalex.org/works"
    cursor = "*"
    fetched = 0
    
    while fetched < max_results:
        params = {
            "search": query,
            "filter": "type:article",
            "select": "id,doi,ids,title,publication_year,concepts",
            "per_page": min(per_page, max_results - fetched),
            "cursor": cursor,
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                
                results = data.get("results", [])
                if not results:
                    break
                
                for work in results:
                    yield work
                    fetched += 1
                
                # Get next cursor
                meta = data.get("meta", {})
                cursor = meta.get("next_cursor")
                if not cursor:
                    break
                
                # Rate limiting
                sleep(0.1)
                
        except Exception as e:
            logger.warning(f"OpenAlex fetch failed for '{query}': {e}")
            break
    
    logger.info(f"Fetched {fetched} works for query: {query}")


def extract_reference(work: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Extract minimal reference data from OpenAlex work."""
    try:
        ids = work.get("ids", {})
        
        # Extract PMID
        pmid = None
        pmid_url = ids.get("pmid", "")
        if pmid_url:
            pmid = pmid_url.replace("https://pubmed.ncbi.nlm.nih.gov/", "")
        
        # Extract DOI
        doi = None
        if work.get("doi"):
            doi = work["doi"].replace("https://doi.org/", "")
        
        # Extract OpenAlex ID
        openalex_id = None
        if work.get("id"):
            openalex_id = work["id"].replace("https://openalex.org/", "")
        
        # Must have at least one ID
        if not (pmid or doi or openalex_id):
            return None
        
        # Title
        title = work.get("title", "") or ""
        if not title:
            return None
        
        # Year
        year = work.get("publication_year")
        
        # Keywords from concepts (top 5)
        keywords = []
        for concept in work.get("concepts", [])[:5]:
            name = concept.get("display_name", "")
            if name:
                keywords.append(name)
        
        # Add query term as keyword
        if query not in keywords:
            keywords.append(query)
        
        return {
            "pmid": pmid,
            "doi": doi,
            "openalex_id": openalex_id,
            "title": title,
            "year": year,
            "source": "openalex",
            "keywords": keywords,
        }
        
    except Exception as e:
        logger.debug(f"Failed to extract reference: {e}")
        return None


# =============================================================================
# PubMed Fetching (E-utilities API)
# =============================================================================

def fetch_pubmed_ids(query: str, max_results: int = 10000, api_key: Optional[str] = None) -> List[str]:
    """
    Search PubMed and return list of PMIDs.
    
    Rate limit: 3/sec without key, 10/sec with key
    """
    import os
    from time import sleep
    
    api_key = api_key or os.environ.get("NCBI_API_KEY")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    pmids = []
    retstart = 0
    batch_size = 10000  # PubMed max per request
    
    while len(pmids) < max_results:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(batch_size, max_results - len(pmids)),
            "retstart": retstart,
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                result = data.get("esearchresult", {})
                ids = result.get("idlist", [])
                
                if not ids:
                    break
                
                pmids.extend(ids)
                retstart += len(ids)
                
                # Check if we've gotten all results
                total = int(result.get("count", 0))
                if retstart >= total:
                    break
                
                sleep(0.35 if api_key else 0.5)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            break
    
    return pmids[:max_results]


def fetch_pubmed_details(pmids: List[str], api_key: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """
    Fetch paper details from PubMed for a list of PMIDs.
    
    Yields reference dicts with pmid, title, year, keywords.
    """
    import os
    from time import sleep
    
    api_key = api_key or os.environ.get("NCBI_API_KEY")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    batch_size = 500  # Fetch 500 at a time
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                result = data.get("result", {})
                
                for pmid in batch:
                    if pmid not in result:
                        continue
                    
                    doc = result[pmid]
                    if not isinstance(doc, dict):
                        continue
                    
                    # Extract DOI from articleids
                    doi = None
                    for aid in doc.get("articleids", []):
                        if aid.get("idtype") == "doi":
                            doi = aid.get("value")
                            break
                    
                    # Extract year from pubdate
                    year = None
                    pubdate = doc.get("pubdate", "")
                    if pubdate:
                        try:
                            year = int(pubdate[:4])
                        except (ValueError, IndexError):
                            pass
                    
                    title = doc.get("title", "")
                    if not title:
                        continue
                    
                    yield {
                        "pmid": pmid,
                        "doi": doi,
                        "openalex_id": None,
                        "title": title,
                        "year": year,
                        "source": "pubmed",
                        "keywords": [],  # esummary doesn't return keywords
                    }
                
                sleep(0.35 if api_key else 0.5)
                
        except Exception as e:
            logger.warning(f"PubMed fetch batch failed: {e}")


def fetch_pubmed_works(query: str, max_results: int = 10000) -> Iterator[Dict[str, Any]]:
    """
    Fetch works from PubMed by query.
    
    Two-step process:
    1. Search for PMIDs
    2. Fetch details for each PMID
    """
    logger.info(f"Searching PubMed for: {query}")
    pmids = fetch_pubmed_ids(query, max_results=max_results)
    logger.info(f"Found {len(pmids)} PMIDs")
    
    fetched = 0
    for ref in fetch_pubmed_details(pmids):
        ref["keywords"].append(query)  # Add query as keyword
        yield ref
        fetched += 1
    
    logger.info(f"Fetched {fetched} papers from PubMed for: {query}")


# =============================================================================
# Europe PMC Fetching
# =============================================================================

def fetch_europe_pmc_works(query: str, max_results: int = 10000) -> Iterator[Dict[str, Any]]:
    """
    Fetch works from Europe PMC API.
    
    Includes PubMed, PMC, preprints, and European sources.
    Rate limit: ~10 requests/second (generous)
    """
    from time import sleep
    
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    cursor_mark = "*"
    fetched = 0
    page_size = 1000
    
    while fetched < max_results:
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(page_size, max_results - fetched),
            "cursorMark": cursor_mark,
            "resultType": "lite",  # Faster, includes basic metadata
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                results = data.get("resultList", {}).get("result", [])
                
                if not results:
                    break
                
                for doc in results:
                    ref = extract_europe_pmc_reference(doc, query)
                    if ref:
                        yield ref
                        fetched += 1
                
                # Get next cursor
                cursor_mark = data.get("nextCursorMark")
                if not cursor_mark:
                    break
                
                sleep(0.1)  # Rate limiting
                
        except Exception as e:
            logger.warning(f"Europe PMC fetch failed: {e}")
            break
    
    logger.info(f"Fetched {fetched} papers from Europe PMC for: {query}")


def extract_europe_pmc_reference(doc: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Extract reference from Europe PMC result."""
    try:
        pmid = doc.get("pmid")
        doi = doc.get("doi")
        pmc_id = doc.get("pmcid")
        
        # Must have at least one ID
        if not (pmid or doi or pmc_id):
            return None
        
        title = doc.get("title", "")
        if not title:
            return None
        
        # Year
        year = None
        pub_year = doc.get("pubYear")
        if pub_year:
            try:
                year = int(pub_year)
            except ValueError:
                pass
        
        # Keywords
        keywords = [query]
        
        return {
            "pmid": pmid,
            "doi": doi,
            "openalex_id": None,
            "title": title,
            "year": year,
            "source": "europe_pmc",
            "keywords": keywords,
        }
        
    except Exception as e:
        logger.debug(f"Failed to extract Europe PMC reference: {e}")
        return None


# =============================================================================
# arXiv Fetching (for ML/BCI papers)
# =============================================================================

def fetch_arxiv_works(query: str, max_results: int = 1000) -> Iterator[Dict[str, Any]]:
    """
    Fetch works from arXiv API.
    
    Good for ML/BCI papers not in PubMed.
    Rate limit: 1 request per 3 seconds
    """
    from time import sleep
    
    base_url = "http://export.arxiv.org/api/query"
    start = 0
    batch_size = 100  # arXiv recommends smaller batches
    fetched = 0
    
    while fetched < max_results:
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": min(batch_size, max_results - fetched),
            "sortBy": "relevance",
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_text = resp.read().decode('utf-8')
                
                # Parse Atom feed
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(xml_text)
                entries = root.findall("atom:entry", ns)
                
                if not entries:
                    break
                
                for entry in entries:
                    ref = extract_arxiv_reference(entry, ns, query)
                    if ref:
                        yield ref
                        fetched += 1
                
                start += len(entries)
                sleep(3)  # arXiv rate limit
                
        except Exception as e:
            logger.warning(f"arXiv fetch failed: {e}")
            break
    
    logger.info(f"Fetched {fetched} papers from arXiv for: {query}")


def extract_arxiv_reference(entry: ET.Element, ns: Dict, query: str) -> Optional[Dict[str, Any]]:
    """Extract reference from arXiv entry."""
    try:
        # arXiv ID
        id_elem = entry.find("atom:id", ns)
        if id_elem is None:
            return None
        
        arxiv_url = id_elem.text
        arxiv_id = arxiv_url.split("/abs/")[-1] if arxiv_url else None
        
        # Title
        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
        if not title:
            return None
        
        # Year
        year = None
        published = entry.find("atom:published", ns)
        if published is not None and published.text:
            try:
                year = int(published.text[:4])
            except (ValueError, IndexError):
                pass
        
        # DOI (if available)
        doi = None
        # arXiv namespace for DOI
        arxiv_ns = {"arxiv": "http://arxiv.org/schemas/atom"}
        doi_elem = entry.find("arxiv:doi", arxiv_ns)
        if doi_elem is not None:
            doi = doi_elem.text
        
        return {
            "pmid": None,
            "doi": doi,
            "openalex_id": None,
            "arxiv_id": arxiv_id,
            "title": title,
            "year": year,
            "source": "arxiv",
            "keywords": [query],
        }
        
    except Exception as e:
        logger.debug(f"Failed to extract arXiv reference: {e}")
        return None


# =============================================================================
# Multi-Source Build Function
# =============================================================================

def build_index(
    target: int = 500000,
    output_dir: Optional[Path] = None,
    sources: Optional[List[str]] = None,
    resume: bool = False
):
    """
    Build the metadata index by fetching from multiple sources.
    
    Sources used (in order):
    1. OpenAlex - Primary source, good coverage, fast API
    2. PubMed - Clinical focus, MeSH terms, authoritative
    3. Europe PMC - European sources, preprints
    4. arXiv - ML/BCI papers
    
    Args:
        target: Target number of papers (default 500K)
        output_dir: Output directory for index
        sources: List of sources to use (default: all)
        resume: Resume from existing index instead of rebuilding
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "metadata"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "index.db"
    
    # Handle resume vs fresh build
    if not resume and db_path.exists():
        db_path.unlink()
    
    # Initialize index
    index = MetadataIndex(db_path=db_path, read_only=False)
    index.initialize()
    
    # Get current count if resuming
    initial_count = 0
    if resume:
        stats = index.get_stats()
        initial_count = stats.get("total_papers", 0)
    
    # Default sources
    if sources is None:
        sources = ["openalex", "pubmed", "europe_pmc", "arxiv"]
    
    print(f"\n{'='*70}")
    print("  EEG-RAG Multi-Source Metadata Index Builder")
    print(f"{'='*70}")
    print(f"  Target: {target:,} paper references")
    print(f"  Output: {db_path}")
    print(f"  Sources: {', '.join(sources)}")
    print(f"  Queries: {len(EEG_QUERIES)} general + {len(PUBMED_MESH_QUERIES)} MeSH")
    if resume:
        print(f"  Resuming from: {initial_count:,} papers")
    print(f"{'='*70}\n")
    
    total_added = initial_count
    start_time = time.time()
    
    # Calculate papers per source
    remaining = target - total_added
    papers_per_source = {
        "openalex": int(remaining * 0.45),    # 45% from OpenAlex
        "pubmed": int(remaining * 0.35),      # 35% from PubMed
        "europe_pmc": int(remaining * 0.15),  # 15% from Europe PMC
        "arxiv": int(remaining * 0.05),       # 5% from arXiv
    }
    
    # ==========================================================================
    # Phase 1: OpenAlex (fast, comprehensive)
    # ==========================================================================
    if "openalex" in sources and total_added < target:
        print(f"\n📚 Phase 1: OpenAlex (target: {papers_per_source['openalex']:,})")
        print("-" * 60)
        
        openalex_target = papers_per_source["openalex"]
        per_query = openalex_target // len(EEG_QUERIES) + 1
        
        for i, query in enumerate(EEG_QUERIES, 1):
            if total_added >= target:
                break
            
            print(f"  [{i}/{len(EEG_QUERIES)}] {query[:50]}...", end=" ", flush=True)
            
            refs = []
            for work in fetch_openalex_works(query, max_results=per_query):
                ref = extract_reference(work, query)
                if ref:
                    refs.append(ref)
            
            if refs:
                added = index.add_references_batch(refs)
                total_added += added
                print(f"✓ +{added:,} (total: {total_added:,})")
            else:
                print("○ 0")
    
    # ==========================================================================
    # Phase 2: PubMed (clinical focus)
    # ==========================================================================
    if "pubmed" in sources and total_added < target:
        print(f"\n📖 Phase 2: PubMed (target: {papers_per_source['pubmed']:,})")
        print("-" * 60)
        
        # Use MeSH queries for PubMed
        all_pubmed_queries = PUBMED_MESH_QUERIES + [
            f"{q} AND (electroencephalography[MeSH] OR EEG)" 
            for q in EEG_QUERIES[:30]  # First 30 queries
        ]
        
        pubmed_target = papers_per_source["pubmed"]
        per_query = pubmed_target // len(all_pubmed_queries) + 1
        
        for i, query in enumerate(all_pubmed_queries, 1):
            if total_added >= target:
                break
            
            print(f"  [{i}/{len(all_pubmed_queries)}] {query[:50]}...", end=" ", flush=True)
            
            refs = list(fetch_pubmed_works(query, max_results=per_query))
            
            if refs:
                added = index.add_references_batch(refs)
                total_added += added
                print(f"✓ +{added:,} (total: {total_added:,})")
            else:
                print("○ 0")
    
    # ==========================================================================
    # Phase 3: Europe PMC (preprints, European sources)
    # ==========================================================================
    if "europe_pmc" in sources and total_added < target:
        print(f"\n🌍 Phase 3: Europe PMC (target: {papers_per_source['europe_pmc']:,})")
        print("-" * 60)
        
        # Focus on unique sources not in PubMed
        epmc_queries = [
            "SRC:PPR AND EEG",  # Preprints
            "SRC:PMC AND EEG brain",  # PMC full text
            "electroencephalography AND OPEN_ACCESS:Y",
            "brain computer interface",
            "seizure detection machine learning",
            "sleep EEG classification",
            "emotion recognition EEG deep learning",
        ] + [q for q in EEG_QUERIES[:20]]
        
        epmc_target = papers_per_source["europe_pmc"]
        per_query = epmc_target // len(epmc_queries) + 1
        
        for i, query in enumerate(epmc_queries, 1):
            if total_added >= target:
                break
            
            print(f"  [{i}/{len(epmc_queries)}] {query[:50]}...", end=" ", flush=True)
            
            refs = list(fetch_europe_pmc_works(query, max_results=per_query))
            
            if refs:
                added = index.add_references_batch(refs)
                total_added += added
                print(f"✓ +{added:,} (total: {total_added:,})")
            else:
                print("○ 0")
    
    # ==========================================================================
    # Phase 4: arXiv (ML/BCI papers)
    # ==========================================================================
    if "arxiv" in sources and total_added < target:
        print(f"\n📄 Phase 4: arXiv (target: {papers_per_source['arxiv']:,})")
        print("-" * 60)
        
        arxiv_queries = [
            "EEG classification",
            "brain computer interface",
            "motor imagery decoding",
            "seizure prediction deep learning",
            "EEG signal processing neural network",
            "emotion recognition EEG",
            "sleep staging deep learning",
            "EEG transformer",
            "SSVEP brain",
            "P300 speller",
        ]
        
        arxiv_target = papers_per_source["arxiv"]
        per_query = arxiv_target // len(arxiv_queries) + 1
        
        for i, query in enumerate(arxiv_queries, 1):
            if total_added >= target:
                break
            
            print(f"  [{i}/{len(arxiv_queries)}] {query[:40]}...", end=" ", flush=True)
            
            refs = list(fetch_arxiv_works(query, max_results=per_query))
            
            if refs:
                added = index.add_references_batch(refs)
                total_added += added
                print(f"✓ +{added:,} (total: {total_added:,})")
            else:
                print("○ 0")
    
    # ==========================================================================
    # Final Stats
    # ==========================================================================
    stats = index.get_stats()
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("  ✅ Index Complete!")
    print(f"{'='*70}")
    print(f"  Total papers: {stats['total_papers']:,}")
    print(f"  By source: {stats.get('by_source', {})}")
    print(f"  With PMID: {stats['with_pmid']:,} ({100*stats['with_pmid']/max(stats['total_papers'],1):.1f}%)")
    print(f"  With DOI: {stats['with_doi']:,} ({100*stats['with_doi']/max(stats['total_papers'],1):.1f}%)")
    print(f"  Year range: {stats['year_range']['min']} - {stats['year_range']['max']}")
    print(f"  Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Build time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {(stats['total_papers'] - initial_count) / max(elapsed, 1):.0f} papers/sec")
    
    # Compress for distribution
    print(f"\n  Compressing for distribution...")
    compressed = index.compress()
    print(f"  Compressed size: {compressed.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {compressed}")
    
    print(f"\n{'='*70}")
    
    return db_path


def main():
    parser = argparse.ArgumentParser(
        description="Build EEG-RAG metadata index from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build full 500K index (takes ~2-3 hours)
  python scripts/build_metadata_index.py --target 500000

  # Quick test build with 50K papers
  python scripts/build_metadata_index.py --quick

  # Resume an interrupted build
  python scripts/build_metadata_index.py --resume --target 500000

  # Build from specific source only
  python scripts/build_metadata_index.py --source openalex --target 100000

  # Build from PubMed only with custom output
  python scripts/build_metadata_index.py --source pubmed --target 50000 --output /tmp/test_index

Sources available:
  openalex   - OpenAlex (250M+ works, fast API)
  pubmed     - PubMed/MEDLINE (35M+ papers, clinical focus)
  europe_pmc - Europe PMC (includes preprints)
  arxiv      - arXiv (ML/BCI papers)
"""
    )
    parser.add_argument("--target", type=int, default=500000,
                       help="Target number of papers (default: 500000)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick build with 50K papers for testing")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for index")
    parser.add_argument("--source", type=str, default=None,
                       help="Use only this source (openalex, pubmed, europe_pmc, arxiv)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing index instead of rebuilding")
    
    args = parser.parse_args()
    
    target = 50000 if args.quick else args.target
    output_dir = Path(args.output) if args.output else None
    sources = [args.source] if args.source else None
    
    build_index(target=target, output_dir=output_dir, sources=sources, resume=args.resume)


if __name__ == "__main__":
    main()

