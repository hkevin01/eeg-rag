#!/usr/bin/env python3
# scripts/build_metadata_index.py
"""
Build the lightweight metadata index for distribution.

This script is run by maintainers to create the compressed index
that ships with the repository. Users don't need to run this.

The index contains only:
- PMIDs, DOIs, OpenAlex IDs (identifiers)
- Titles (for search)
- Years, keywords (for filtering)

Full abstracts/content are NOT stored - they're fetched on-demand.

Usage:
    python scripts/build_metadata_index.py --target 500000
    python scripts/build_metadata_index.py --quick  # 50K for testing
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.db.metadata_index import MetadataIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# EEG-focused search queries for comprehensive coverage
EEG_QUERIES = [
    # Core EEG terms
    "electroencephalography",
    "EEG signal processing",
    "EEG analysis",
    "brain waves",
    "neural oscillations",
    
    # Frequency bands
    "alpha rhythm EEG",
    "beta oscillation brain",
    "theta oscillation brain",
    "delta waves sleep",
    "gamma oscillation cognition",
    
    # Clinical applications
    "epilepsy EEG",
    "seizure detection EEG",
    "epileptiform discharge",
    "sleep disorder EEG",
    "sleep staging polysomnography",
    
    # ERPs
    "event-related potential",
    "P300 EEG",
    "N400 language",
    "mismatch negativity",
    "error-related negativity",
    
    # BCI
    "brain-computer interface",
    "motor imagery EEG",
    "SSVEP BCI",
    "P300 speller",
    
    # Cognitive neuroscience
    "working memory EEG",
    "attention EEG",
    "cognitive load EEG",
    "emotion recognition EEG",
    
    # Methods
    "independent component analysis EEG",
    "artifact removal EEG",
    "source localization EEG",
    "EEG preprocessing",
    "connectivity analysis EEG",
    
    # Disorders
    "ADHD EEG",
    "autism spectrum EEG",
    "depression EEG",
    "schizophrenia EEG",
    "Alzheimer EEG",
    "Parkinson EEG",
    
    # Technical
    "high-density EEG",
    "dry electrode EEG",
    "mobile EEG",
    "real-time EEG",
    
    # Machine learning
    "deep learning EEG",
    "CNN EEG classification",
    "EEG decoding",
    "neural network EEG",
    
    # Specific paradigms
    "resting state EEG",
    "auditory evoked potential",
    "visual evoked potential",
    "somatosensory evoked",
    
    # Combined modalities
    "EEG fMRI",
    "EEG MEG",
    "simultaneous EEG",
    
    # Development
    "neonatal EEG",
    "pediatric EEG",
    "infant EEG",
    "aging brain EEG",
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


def build_index(target: int = 500000, output_dir: Optional[Path] = None):
    """Build the metadata index by fetching from OpenAlex."""
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "metadata"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "index.db"
    
    # Remove existing index
    if db_path.exists():
        db_path.unlink()
    
    # Initialize index
    index = MetadataIndex(db_path=db_path, read_only=False)
    index.initialize()
    
    print(f"\n{'='*60}")
    print("  EEG-RAG Metadata Index Builder")
    print(f"{'='*60}")
    print(f"  Target: {target:,} paper references")
    print(f"  Output: {db_path}")
    print(f"  Queries: {len(EEG_QUERIES)}")
    print(f"{'='*60}\n")
    
    total_added = 0
    per_query = target // len(EEG_QUERIES) + 1
    start_time = time.time()
    
    for i, query in enumerate(EEG_QUERIES, 1):
        if total_added >= target:
            break
        
        print(f"[{i}/{len(EEG_QUERIES)}] Fetching: {query}")
        
        refs = []
        for work in fetch_openalex_works(query, max_results=per_query):
            ref = extract_reference(work, query)
            if ref:
                refs.append(ref)
        
        if refs:
            added = index.add_references_batch(refs)
            total_added += added
            print(f"         Added: {added} new references (total: {total_added:,})")
        
        # Progress
        elapsed = time.time() - start_time
        rate = total_added / elapsed if elapsed > 0 else 0
        eta = (target - total_added) / rate if rate > 0 else 0
        print(f"         Rate: {rate:.0f}/sec, ETA: {eta/60:.1f} min")
        print()
    
    # Get final stats
    stats = index.get_stats()
    print(f"\n{'='*60}")
    print("  Index Complete!")
    print(f"{'='*60}")
    print(f"  Total papers: {stats['total_papers']:,}")
    print(f"  With PMID: {stats['with_pmid']:,} ({100*stats['with_pmid']/stats['total_papers']:.1f}%)")
    print(f"  With DOI: {stats['with_doi']:,} ({100*stats['with_doi']/stats['total_papers']:.1f}%)")
    print(f"  Year range: {stats['year_range']['min']} - {stats['year_range']['max']}")
    print(f"  Database size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Compress for distribution
    print(f"\nCompressing for distribution...")
    compressed = index.compress()
    print(f"  Compressed size: {compressed.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {compressed}")
    
    print(f"\n✅ Done! Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    return db_path


def main():
    parser = argparse.ArgumentParser(description="Build EEG-RAG metadata index")
    parser.add_argument("--target", type=int, default=500000,
                       help="Target number of papers (default: 500000)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick build with 50K papers for testing")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for index")
    
    args = parser.parse_args()
    
    target = 50000 if args.quick else args.target
    output_dir = Path(args.output) if args.output else None
    
    build_index(target=target, output_dir=output_dir)


if __name__ == "__main__":
    main()
