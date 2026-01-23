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


@dataclass
class SearchResult:
    """Combined search result with reference and resolved content."""
    reference: PaperReference
    paper: Optional[ResolvedPaper] = None
    relevance_score: float = 0.0
    
    @property
    def has_full_content(self) -> bool:
        """Check if full content was resolved."""
        return self.paper is not None and bool(self.paper.abstract)
    
    @property
    def title(self) -> str:
        """Get title from paper or reference."""
        if self.paper:
            return self.paper.title
        return self.reference.title
    
    @property
    def abstract(self) -> str:
        """Get abstract (empty if not resolved)."""
        if self.paper:
            return self.paper.abstract
        return ""
    
    @property
    def pmid(self) -> Optional[str]:
        """Get PMID."""
        if self.paper and self.paper.pmid:
            return self.paper.pmid
        return self.reference.pmid
    
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
    
    def get_paper_sync(self, pmid: Optional[str] = None, doi: Optional[str] = None) -> Optional[ResolvedPaper]:
        """Synchronous wrapper for get_paper."""
        return asyncio.run(self.get_paper(pmid=pmid, doi=doi))
    
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
def get_search_pipeline() -> HybridSearchPipeline:
    """Get the default search pipeline."""
    return HybridSearchPipeline()


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
