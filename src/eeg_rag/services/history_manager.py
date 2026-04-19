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


# ---------------------------------------------------------------------------
# ID           : services.history_manager.HistoryManager
# Requirement  : `HistoryManager` class shall be instantiable and expose the documented interface
# Purpose      : Manages search history operations
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
# Verification : Instantiate HistoryManager with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HistoryManager:
    """
    Manages search history operations.
    Wraps the database layer and provides high-level operations.
    """
    
    _instance: Optional["HistoryManager"] = None
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.__init__
    # Requirement  : `__init__` shall initialize the history manager
    # Purpose      : Initialize the history manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : db_path: Optional[Path] (default=None)
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
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the history manager.
        
        Args:
            db_path: Optional custom database path
        """
        self.db = SearchHistoryDB(db_path)
        self._current_session: Optional[SearchSession] = None
        logger.info(f"History manager initialized with db: {self.db.db_path}")
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_instance
    # Requirement  : `get_instance` shall get singleton instance of HistoryManager
    # Purpose      : Get singleton instance of HistoryManager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : db_path: Optional[Path] (default=None)
    # Outputs      : 'HistoryManager'
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
    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "HistoryManager":
        """Get singleton instance of HistoryManager."""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance
    
    # ==================== Session Management ====================
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.start_session
    # Requirement  : `start_session` shall start a new research session
    # Purpose      : Start a new research session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: Optional[str] (default=None); topic: Optional[str] (default=None)
    # Outputs      : SearchSession
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
    def start_session(self, name: Optional[str] = None, topic: Optional[str] = None) -> SearchSession:
        """Start a new research session."""
        self._current_session = self.db.create_session(name=name, topic=topic)
        logger.info(f"Started session: {self._current_session.id}")
        return self._current_session
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_current_session
    # Requirement  : `get_current_session` shall get the current active session
    # Purpose      : Get the current active session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Optional[SearchSession]
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
    def get_current_session(self) -> Optional[SearchSession]:
        """Get the current active session."""
        return self._current_session
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.end_session
    # Requirement  : `end_session` shall end the current session
    # Purpose      : End the current session
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def end_session(self):
        """End the current session."""
        self._current_session = None
    
    # ==================== Search Recording ====================
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.record_search
    # Requirement  : `record_search` shall record a search and its results
    # Purpose      : Record a search and its results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_text: str; query_type: str; results: List[SearchResult]; execution_time_ms: float; filters: Optional[Dict] (default=None)
    # Outputs      : SearchQuery
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.record_result_click
    # Requirement  : `record_result_click` shall record that the user clicked on a search result
    # Purpose      : Record that the user clicked on a search result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str; paper_id: str
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
    def record_result_click(self, query_id: str, paper_id: str):
        """Record that the user clicked on a search result."""
        self.db.record_click(query_id, paper_id)
        logger.debug(f"Recorded click: {paper_id} from query {query_id}")
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.record_feedback
    # Requirement  : `record_feedback` shall record user feedback for a search
    # Purpose      : Record user feedback for a search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str; feedback: str
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_recent
    # Requirement  : `get_recent` shall get recent searches
    # Purpose      : Get recent searches
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : limit: int (default=50); starred_only: bool (default=False); include_results: bool (default=True)
    # Outputs      : List[SearchQuery]
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_search
    # Requirement  : `get_search` shall get a specific search by ID
    # Purpose      : Get a specific search by ID
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str
    # Outputs      : Optional[SearchQuery]
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
    def get_search(self, query_id: str) -> Optional[SearchQuery]:
        """Get a specific search by ID."""
        return self.db.get_search_by_id(query_id)
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.search_in_history
    # Requirement  : `search_in_history` shall search through past queries
    # Purpose      : Search through past queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : search_text: str; limit: int (default=20)
    # Outputs      : List[SearchQuery]
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
    def search_in_history(self, search_text: str, limit: int = 20) -> List[SearchQuery]:
        """Search through past queries."""
        return self.db.search_history(search_text, limit)
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_sessions
    # Requirement  : `get_sessions` shall get recent search sessions
    # Purpose      : Get recent search sessions
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : limit: int (default=20)
    # Outputs      : List[SearchSession]
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
    def get_sessions(self, limit: int = 20) -> List[SearchSession]:
        """Get recent search sessions."""
        return self.db.get_sessions(limit)
    
    # ==================== User Actions ====================
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.toggle_star
    # Requirement  : `toggle_star` shall toggle star status for a search
    # Purpose      : Toggle star status for a search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str
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
    def toggle_star(self, query_id: str) -> bool:
        """Toggle star status for a search."""
        return self.db.toggle_star(query_id)
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.add_note
    # Requirement  : `add_note` shall add a note to a search
    # Purpose      : Add a note to a search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str; note: str
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
    def add_note(self, query_id: str, note: str):
        """Add a note to a search."""
        self.db.add_note(query_id, note)
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.delete_search
    # Requirement  : `delete_search` shall delete a search from history
    # Purpose      : Delete a search from history
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_id: str
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
    def delete_search(self, query_id: str):
        """Delete a search from history."""
        self.db.delete_search(query_id)
        logger.info(f"Deleted search: {query_id}")
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.clear_old_history
    # Requirement  : `clear_old_history` shall clear history older than specified days
    # Purpose      : Clear history older than specified days
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : days: int (default=30); keep_starred: bool (default=True)
    # Outputs      : int
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.save_paper
    # Requirement  : `save_paper` shall save a paper to the reading list
    # Purpose      : Save a paper to the reading list
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : result: SearchResult; notes: Optional[str] (default=None); tags: Optional[List[str]] (default=None)
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
    def save_paper(
        self, result: SearchResult,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Save a paper to the reading list."""
        self.db.save_paper(result, notes=notes, tags=tags)
        logger.info(f"Saved paper: {result.paper_id}")
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_saved_papers
    # Requirement  : `get_saved_papers` shall get saved papers with optional filtering
    # Purpose      : Get saved papers with optional filtering
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : read_status: Optional[str] (default=None); tag: Optional[str] (default=None); limit: int (default=100)
    # Outputs      : List[Dict]
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.update_paper_status
    # Requirement  : `update_paper_status` shall update read status of a saved paper
    # Purpose      : Update read status of a saved paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_id: str; status: str
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
    def update_paper_status(self, paper_id: str, status: str):
        """Update read status of a saved paper."""
        self.db.update_paper_status(paper_id, status)
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.remove_saved_paper
    # Requirement  : `remove_saved_paper` shall remove a paper from the saved list
    # Purpose      : Remove a paper from the saved list
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_id: str
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
    def remove_saved_paper(self, paper_id: str):
        """Remove a paper from the saved list."""
        self.db.remove_saved_paper(paper_id)
        logger.info(f"Removed paper: {paper_id}")
    
    # ==================== Analytics ====================
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.get_stats
    # Requirement  : `get_stats` shall get search history statistics
    # Purpose      : Get search history statistics
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
        """Get search history statistics."""
        return self.db.get_search_stats()
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.export_history
    # Requirement  : `export_history` shall export search history to a file
    # Purpose      : Export search history to a file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : filepath: Path; format: str (default='json')
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
    
    # ---------------------------------------------------------------------------
    # ID           : services.history_manager.HistoryManager.create_search_result
    # Requirement  : `create_search_result` shall helper to create a SearchResult object
    # Purpose      : Helper to create a SearchResult object
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper_id: str; title: str; authors: List[str]; year: Optional[int]; abstract: str; source: str; relevance_score: float; **kwargs
    # Outputs      : SearchResult
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
