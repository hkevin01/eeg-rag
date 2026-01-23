# src/eeg_rag/search/search_with_history.py
"""
Search wrapper that integrates history tracking.
Wraps the existing search pipeline to automatically record all searches.
"""

import time
from typing import List, Optional, Dict, Any, Callable
import logging

from eeg_rag.services.history_manager import HistoryManager
from eeg_rag.db.history_models import SearchResult, SearchQuery


logger = logging.getLogger(__name__)


class SearchWithHistory:
    """
    Wrapper for search functions that automatically records history.
    
    Example usage:
        # Wrap your existing search function
        search = SearchWithHistory(your_search_function)
        results = await search.search("EEG epilepsy detection", query_type="hybrid")
        
        # Results are automatically recorded in history
    """
    
    def __init__(
        self,
        search_fn: Callable = None,
        history_manager: Optional[HistoryManager] = None
    ):
        """
        Initialize search wrapper.
        
        Args:
            search_fn: The underlying search function to wrap. Should accept
                       query_text and return list of results.
            history_manager: Optional custom history manager instance
        """
        self.search_fn = search_fn
        self.history_manager = history_manager or HistoryManager.get_instance()
        self._last_query_id: Optional[str] = None
    
    async def search(
        self,
        query_text: str,
        query_type: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        record_history: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute search and optionally record in history.
        
        Args:
            query_text: The search query
            query_type: Type of search (natural, semantic, hybrid, etc.)
            filters: Optional filters for the search
            record_history: Whether to record this search in history
            **kwargs: Additional arguments passed to the search function
        
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Execute the actual search
        if self.search_fn:
            raw_results = await self.search_fn(query_text, **kwargs)
        else:
            # Return empty if no search function configured
            raw_results = []
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Convert raw results to SearchResult objects
        results = self._convert_results(raw_results)
        
        # Record in history
        if record_history and results:
            try:
                saved_query = self.history_manager.record_search(
                    query_text=query_text,
                    query_type=query_type,
                    results=results,
                    execution_time_ms=execution_time_ms,
                    filters=filters
                )
                self._last_query_id = saved_query.id
                logger.debug(f"Recorded search in history: {saved_query.id}")
            except Exception as e:
                logger.warning(f"Failed to record search history: {e}")
        
        return results
    
    def _convert_results(self, raw_results: List[Any]) -> List[SearchResult]:
        """Convert raw search results to SearchResult objects."""
        results = []
        
        for idx, r in enumerate(raw_results):
            # Handle different result formats
            if isinstance(r, SearchResult):
                results.append(r)
            elif isinstance(r, dict):
                try:
                    result = SearchResult(
                        paper_id=r.get('paper_id', r.get('id', f"result_{idx}")),
                        title=r.get('title', 'Untitled'),
                        authors=r.get('authors', []),
                        year=r.get('year'),
                        abstract=r.get('abstract', ''),
                        source=r.get('source', 'unknown'),
                        relevance_score=r.get('relevance_score', r.get('score', 0.0)),
                        doi=r.get('doi'),
                        pmid=r.get('pmid'),
                        url=r.get('url'),
                        snippet=r.get('snippet', r.get('highlight', ''))
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to convert result: {e}")
            else:
                # Try to extract attributes from object
                try:
                    result = SearchResult(
                        paper_id=getattr(r, 'paper_id', getattr(r, 'id', f"result_{idx}")),
                        title=getattr(r, 'title', 'Untitled'),
                        authors=getattr(r, 'authors', []),
                        year=getattr(r, 'year', None),
                        abstract=getattr(r, 'abstract', ''),
                        source=getattr(r, 'source', 'unknown'),
                        relevance_score=getattr(r, 'relevance_score', getattr(r, 'score', 0.0)),
                        doi=getattr(r, 'doi', None),
                        pmid=getattr(r, 'pmid', None),
                        url=getattr(r, 'url', None),
                        snippet=getattr(r, 'snippet', '')
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to convert result object: {e}")
        
        return results
    
    def record_click(self, paper_id: str):
        """Record that user clicked on a search result."""
        if self._last_query_id:
            self.history_manager.record_result_click(self._last_query_id, paper_id)
    
    def record_feedback(self, feedback: str):
        """Record user feedback for the last search."""
        if self._last_query_id:
            self.history_manager.record_feedback(self._last_query_id, feedback)
    
    def save_paper(
        self,
        result: SearchResult,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Save a paper to the reading list."""
        self.history_manager.save_paper(result, notes=notes, tags=tags)
    
    @property
    def last_query_id(self) -> Optional[str]:
        """Get the ID of the last recorded search."""
        return self._last_query_id
    
    def get_recent_searches(self, limit: int = 20) -> List[SearchQuery]:
        """Get recent search history."""
        return self.history_manager.get_recent(limit=limit, include_results=False)
    
    def get_starred_searches(self, limit: int = 50) -> List[SearchQuery]:
        """Get starred searches."""
        return self.history_manager.get_recent(limit=limit, starred_only=True)
    
    def search_in_history(self, search_text: str) -> List[SearchQuery]:
        """Search through past queries."""
        return self.history_manager.search_in_history(search_text)


# Convenience function to create a wrapped search
def create_search_with_history(search_fn: Callable) -> SearchWithHistory:
    """
    Create a SearchWithHistory wrapper for an existing search function.
    
    Args:
        search_fn: Async function that takes query_text and returns results
    
    Returns:
        SearchWithHistory instance wrapping the function
    
    Example:
        async def my_search(query_text, **kwargs):
            # Your search logic here
            return results
        
        search = create_search_with_history(my_search)
        results = await search.search("EEG patterns")
    """
    return SearchWithHistory(search_fn=search_fn)
