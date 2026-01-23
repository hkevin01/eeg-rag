# src/eeg_rag/services/history_manager.py
"""
High-level search history management service.
Provides a clean interface for the application to interact with search history.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from eeg_rag.db.history_models import (
    SearchHistoryDB, SearchResult, SearchQuery, SearchSession
)


logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages search history operations.
    Wraps the database layer and provides high-level operations.
    """
    
    _instance: Optional["HistoryManager"] = None
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the history manager.
        
        Args:
            db_path: Optional custom database path
        """
        self.db = SearchHistoryDB(db_path)
        self._current_session: Optional[SearchSession] = None
        logger.info(f"History manager initialized with db: {self.db.db_path}")
    
    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "HistoryManager":
        """Get singleton instance of HistoryManager."""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance
    
    # ==================== Session Management ====================
    
    def start_session(self, name: Optional[str] = None, topic: Optional[str] = None) -> SearchSession:
        """Start a new research session."""
        self._current_session = self.db.create_session(name=name, topic=topic)
        logger.info(f"Started session: {self._current_session.id}")
        return self._current_session
    
    def get_current_session(self) -> Optional[SearchSession]:
        """Get the current active session."""
        return self._current_session
    
    def end_session(self):
        """End the current session."""
        self._current_session = None
    
    # ==================== Search Recording ====================
    
    def record_search(
        self,
        query_text: str,
        query_type: str,
        results: List[SearchResult],
        execution_time_ms: float,
        filters: Optional[Dict] = None
    ) -> SearchQuery:
        """
        Record a search and its results.
        
        Args:
            query_text: The search query string
            query_type: Type of search (natural, semantic, hybrid, etc.)
            results: List of search results
            execution_time_ms: Time taken for the search
            filters: Optional filters applied to the search
        
        Returns:
            The saved SearchQuery object
        """
        session_id = self._current_session.id if self._current_session else None
        
        saved_query = self.db.save_search(
            query_text=query_text,
            query_type=query_type,
            results=results,
            execution_time_ms=execution_time_ms,
            filters=filters,
            session_id=session_id
        )
        
        logger.debug(f"Recorded search: {query_text[:50]}... ({len(results)} results)")
        return saved_query
    
    def record_result_click(self, query_id: str, paper_id: str):
        """Record that the user clicked on a search result."""
        self.db.record_click(query_id, paper_id)
        logger.debug(f"Recorded click: {paper_id} from query {query_id}")
    
    def record_feedback(self, query_id: str, feedback: str):
        """
        Record user feedback for a search.
        
        Args:
            query_id: The search query ID
            feedback: 'helpful', 'not_helpful', or None
        """
        self.db.set_feedback(query_id, feedback)
        logger.debug(f"Recorded feedback: {feedback} for query {query_id}")
    
    # ==================== History Retrieval ====================
    
    def get_recent(
        self, limit: int = 50, starred_only: bool = False,
        include_results: bool = True
    ) -> List[SearchQuery]:
        """Get recent searches."""
        return self.db.get_recent_searches(
            limit=limit,
            starred_only=starred_only,
            include_results=include_results
        )
    
    def get_search(self, query_id: str) -> Optional[SearchQuery]:
        """Get a specific search by ID."""
        return self.db.get_search_by_id(query_id)
    
    def search_in_history(self, search_text: str, limit: int = 20) -> List[SearchQuery]:
        """Search through past queries."""
        return self.db.search_history(search_text, limit)
    
    def get_sessions(self, limit: int = 20) -> List[SearchSession]:
        """Get recent search sessions."""
        return self.db.get_sessions(limit)
    
    # ==================== User Actions ====================
    
    def toggle_star(self, query_id: str) -> bool:
        """Toggle star status for a search."""
        return self.db.toggle_star(query_id)
    
    def add_note(self, query_id: str, note: str):
        """Add a note to a search."""
        self.db.add_note(query_id, note)
    
    def delete_search(self, query_id: str):
        """Delete a search from history."""
        self.db.delete_search(query_id)
        logger.info(f"Deleted search: {query_id}")
    
    def clear_old_history(self, days: int = 30, keep_starred: bool = True) -> int:
        """
        Clear history older than specified days.
        
        Args:
            days: Clear entries older than this many days
            keep_starred: If True, keep starred searches
        
        Returns:
            Number of deleted entries
        """
        before_date = datetime.now() - timedelta(days=days)
        count = self.db.clear_history(before_date=before_date, keep_starred=keep_starred)
        logger.info(f"Cleared {count} old searches")
        return count
    
    # ==================== Paper Management ====================
    
    def save_paper(
        self, result: SearchResult,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Save a paper to the reading list."""
        self.db.save_paper(result, notes=notes, tags=tags)
        logger.info(f"Saved paper: {result.paper_id}")
    
    def get_saved_papers(
        self,
        read_status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get saved papers with optional filtering."""
        return self.db.get_saved_papers(
            read_status=read_status,
            tag=tag,
            limit=limit
        )
    
    def update_paper_status(self, paper_id: str, status: str):
        """Update read status of a saved paper."""
        self.db.update_paper_status(paper_id, status)
    
    def remove_saved_paper(self, paper_id: str):
        """Remove a paper from the saved list."""
        self.db.remove_saved_paper(paper_id)
        logger.info(f"Removed paper: {paper_id}")
    
    # ==================== Analytics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search history statistics."""
        return self.db.get_search_stats()
    
    def export_history(self, filepath: Path, format: str = "json"):
        """
        Export search history to a file.
        
        Args:
            filepath: Destination file path
            format: 'json' or 'csv'
        """
        self.db.export_history(filepath, format)
        logger.info(f"Exported history to: {filepath}")
    
    # ==================== Utility ====================
    
    def create_search_result(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        year: Optional[int],
        abstract: str,
        source: str,
        relevance_score: float,
        **kwargs
    ) -> SearchResult:
        """Helper to create a SearchResult object."""
        return SearchResult(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            source=source,
            relevance_score=relevance_score,
            doi=kwargs.get('doi'),
            pmid=kwargs.get('pmid'),
            url=kwargs.get('url'),
            snippet=kwargs.get('snippet')
        )
