# src/eeg_rag/search/hybrid_pipeline.py
"""
Hybrid search pipeline using metadata index + on-demand resolution.

This is the main search interface that:
1. Searches the local metadata index (instant)
2. Resolves top results to full content (cached or API)
3. Returns complete paper objects for RAG

The approach keeps the repo lightweight while providing
access to 500K+ papers.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from eeg_rag.db.metadata_index import MetadataIndex, PaperReference
from eeg_rag.db.paper_resolver import PaperResolver, ResolvedPaper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : search.hybrid_pipeline.SearchResult
# Requirement  : `SearchResult` class shall be instantiable and expose the documented interface
# Purpose      : Combined search result with reference and resolved content
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
# Verification : Instantiate SearchResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """Combined search result with reference and resolved content."""
    reference: PaperReference
    paper: Optional[ResolvedPaper] = None
    relevance_score: float = 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.SearchResult.has_full_content
    # Requirement  : `has_full_content` shall check if full content was resolved
    # Purpose      : Check if full content was resolved
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    @property
    def has_full_content(self) -> bool:
        """Check if full content was resolved."""
        return self.paper is not None and bool(self.paper.abstract)
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.SearchResult.title
    # Requirement  : `title` shall get title from paper or reference
    # Purpose      : Get title from paper or reference
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    @property
    def title(self) -> str:
        """Get title from paper or reference."""
        if self.paper:
            return self.paper.title
        return self.reference.title
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.SearchResult.abstract
    # Requirement  : `abstract` shall get abstract (empty if not resolved)
    # Purpose      : Get abstract (empty if not resolved)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    @property
    def abstract(self) -> str:
        """Get abstract (empty if not resolved)."""
        if self.paper:
            return self.paper.abstract
        return ""
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.SearchResult.pmid
    # Requirement  : `pmid` shall get PMID
    # Purpose      : Get PMID
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Optional[str]
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
    @property
    def pmid(self) -> Optional[str]:
        """Get PMID."""
        if self.paper and self.paper.pmid:
            return self.paper.pmid
        return self.reference.pmid
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.SearchResult.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "abstract": self.abstract,
            "pmid": self.pmid,
            "doi": self.paper.doi if self.paper else self.reference.doi,
            "year": self.paper.year if self.paper else self.reference.year,
            "authors": self.paper.authors if self.paper else [],
            "journal": self.paper.journal if self.paper else None,
            "source": self.paper.source if self.paper else self.reference.source,
            "relevance_score": self.relevance_score,
            "has_full_content": self.has_full_content,
        }


# ---------------------------------------------------------------------------
# ID           : search.hybrid_pipeline.HybridSearchPipeline
# Requirement  : `HybridSearchPipeline` class shall be instantiable and expose the documented interface
# Purpose      : Two-stage search pipeline:
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
# Verification : Instantiate HybridSearchPipeline with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HybridSearchPipeline:
    """
    Two-stage search pipeline:
    
    Stage 1: Fast metadata search (instant, local)
        - Searches the lightweight index that ships with repo
        - Returns paper references with IDs and titles
    
    Stage 2: On-demand content resolution (cached or API)
        - Resolves top N references to full paper content
        - Fetches from PubMed/OpenAlex if not cached
        - Caches results for future use
    
    This approach provides:
    - Instant search results (metadata)
    - Full content for top results
    - Minimal repo size (~10-50MB)
    - No upfront ingestion required for users
    """
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.__init__
    # Requirement  : `__init__` shall initialize the hybrid search pipeline
    # Purpose      : Initialize the hybrid search pipeline
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : metadata_index: Optional[MetadataIndex] (default=None); paper_resolver: Optional[PaperResolver] (default=None)
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
        metadata_index: Optional[MetadataIndex] = None,
        paper_resolver: Optional[PaperResolver] = None,
    ):
        """
        Initialize the hybrid search pipeline.
        
        Args:
            metadata_index: Custom metadata index (default: repo's index.db)
            paper_resolver: Custom paper resolver (default: ~/.eeg_rag/cache/)
        """
        self.metadata_index = metadata_index or MetadataIndex()
        self.paper_resolver = paper_resolver or PaperResolver()
        
        # Check if index exists
        if not self.metadata_index.exists():
            logger.warning(
                "Metadata index not found. Run 'python scripts/setup_user.py' or "
                "'python scripts/build_metadata_index.py --quick'"
            )
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.search
    # Requirement  : `search` shall search for papers with optional content resolution
    # Purpose      : Search for papers with optional content resolution
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; max_results: int (default=10); resolve_content: bool (default=True); year_min: Optional[int] (default=None); year_max: Optional[int] (default=None)
    # Outputs      : List[SearchResult]
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
        max_results: int = 10,
        resolve_content: bool = True,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for papers with optional content resolution.
        
        Args:
            query: Search query (keywords, phrases)
            max_results: Maximum number of results to return
            resolve_content: Whether to fetch full paper content
            year_min: Minimum publication year
            year_max: Maximum publication year
        
        Returns:
            List of SearchResult objects with optional full content
        """
        if not query or not query.strip():
            return []
        
        logger.info(f"Searching for: {query}")
        
        # Stage 1: Fast metadata search
        # Get extra candidates for filtering
        candidates = self.metadata_index.search(
            query,
            limit=max_results * 3,  # Over-fetch for filtering
            year_min=year_min,
            year_max=year_max,
        )
        
        if not candidates:
            logger.info("No matches found in metadata index")
            return []
        
        logger.info(f"Found {len(candidates)} candidates in metadata index")
        
        # Create initial results
        results = [
            SearchResult(reference=ref, relevance_score=1.0 - (i * 0.01))
            for i, ref in enumerate(candidates[:max_results * 2])
        ]
        
        # Stage 2: Resolve content for top results
        if resolve_content:
            # Collect IDs to resolve
            pmids = [r.reference.pmid for r in results if r.reference.pmid]
            dois = [
                r.reference.doi for r in results 
                if r.reference.doi and not r.reference.pmid
            ]
            
            logger.info(f"Resolving {len(pmids)} PMIDs, {len(dois)} DOIs")
            
            # Batch resolve
            resolved = await self.paper_resolver.resolve_batch(
                pmids=pmids,
                dois=dois,
            )
            
            # Create lookup maps
            pmid_map = {p.pmid: p for p in resolved if p.pmid}
            doi_map = {p.doi: p for p in resolved if p.doi and not p.pmid}
            
            # Attach resolved papers to results
            for result in results:
                if result.reference.pmid and result.reference.pmid in pmid_map:
                    result.paper = pmid_map[result.reference.pmid]
                elif result.reference.doi and result.reference.doi in doi_map:
                    result.paper = doi_map[result.reference.doi]
            
            logger.info(f"Resolved {len(resolved)} papers from cache/API")
        
        # Filter to only those with content if resolving
        if resolve_content:
            results = [r for r in results if r.has_full_content]
        
        return results[:max_results]
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.search_sync
    # Requirement  : `search_sync` shall synchronous wrapper for search
    # Purpose      : Synchronous wrapper for search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; max_results: int (default=10); resolve_content: bool (default=True); year_min: Optional[int] (default=None); year_max: Optional[int] (default=None)
    # Outputs      : List[SearchResult]
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
    def search_sync(
        self,
        query: str,
        max_results: int = 10,
        resolve_content: bool = True,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Synchronous wrapper for search.
        
        Use this in non-async contexts like Streamlit.
        """
        return asyncio.run(self.search(
            query=query,
            max_results=max_results,
            resolve_content=resolve_content,
            year_min=year_min,
            year_max=year_max,
        ))
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.search_references_only
    # Requirement  : `search_references_only` shall fast search that returns only references (no content resolution)
    # Purpose      : Fast search that returns only references (no content resolution)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; limit: int (default=100); year_min: Optional[int] (default=None); year_max: Optional[int] (default=None)
    # Outputs      : List[PaperReference]
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
    def search_references_only(
        self,
        query: str,
        limit: int = 100,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[PaperReference]:
        """
        Fast search that returns only references (no content resolution).
        
        Use this for quick searches or when you only need IDs/titles.
        """
        return self.metadata_index.search(
            query,
            limit=limit,
            year_min=year_min,
            year_max=year_max,
        )
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.get_paper
    # Requirement  : `get_paper` shall get a single paper by PMID or DOI
    # Purpose      : Get a single paper by PMID or DOI
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: Optional[str] (default=None); doi: Optional[str] (default=None)
    # Outputs      : Optional[ResolvedPaper]
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
    async def get_paper(self, pmid: Optional[str] = None, doi: Optional[str] = None) -> Optional[ResolvedPaper]:
        """
        Get a single paper by PMID or DOI.
        
        Uses cache if available, otherwise fetches from API.
        """
        if pmid:
            return await self.paper_resolver.resolve_pmid(pmid)
        elif doi:
            return await self.paper_resolver.resolve_doi(doi)
        return None
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.get_paper_sync
    # Requirement  : `get_paper_sync` shall synchronous wrapper for get_paper
    # Purpose      : Synchronous wrapper for get_paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: Optional[str] (default=None); doi: Optional[str] (default=None)
    # Outputs      : Optional[ResolvedPaper]
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
    def get_paper_sync(self, pmid: Optional[str] = None, doi: Optional[str] = None) -> Optional[ResolvedPaper]:
        """Synchronous wrapper for get_paper."""
        return asyncio.run(self.get_paper(pmid=pmid, doi=doi))
    
    # ---------------------------------------------------------------------------
    # ID           : search.hybrid_pipeline.HybridSearchPipeline.get_stats
    # Requirement  : `get_stats` shall get statistics about the search pipeline
    # Purpose      : Get statistics about the search pipeline
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search pipeline."""
        stats = {
            "metadata_index": {},
            "cache": {},
        }
        
        # Metadata index stats
        if self.metadata_index.exists():
            try:
                stats["metadata_index"] = self.metadata_index.get_stats()
            except Exception as e:
                stats["metadata_index"]["error"] = str(e)
        else:
            stats["metadata_index"]["exists"] = False
        
        # Cache stats
        try:
            stats["cache"] = self.paper_resolver.get_cache_stats()
        except Exception as e:
            stats["cache"]["error"] = str(e)
        
        return stats


# Convenience functions
# ---------------------------------------------------------------------------
# ID           : search.hybrid_pipeline.get_search_pipeline
# Requirement  : `get_search_pipeline` shall get the default search pipeline
# Purpose      : Get the default search pipeline
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : HybridSearchPipeline
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
def get_search_pipeline() -> HybridSearchPipeline:
    """Get the default search pipeline."""
    return HybridSearchPipeline()


# ---------------------------------------------------------------------------
# ID           : search.hybrid_pipeline.search_papers
# Requirement  : `search_papers` shall simple synchronous search function
# Purpose      : Simple synchronous search function
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : query: str; max_results: int (default=10); resolve_content: bool (default=True)
# Outputs      : List[SearchResult]
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
def search_papers(
    query: str,
    max_results: int = 10,
    resolve_content: bool = True,
) -> List[SearchResult]:
    """
    Simple synchronous search function.
    
    Example:
        results = search_papers("P300 epilepsy")
        for r in results:
            print(f"{r.title} - PMID:{r.pmid}")
            print(f"  {r.abstract[:200]}...")
    """
    pipeline = HybridSearchPipeline()
    return pipeline.search_sync(
        query=query,
        max_results=max_results,
        resolve_content=resolve_content,
    )
