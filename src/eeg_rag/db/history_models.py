# src/eeg_rag/db/history_models.py
"""
Search history models using SQLite for local storage.
Stores queries, results, and user interactions.
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import hashlib
import uuid


# ---------------------------------------------------------------------------
# ID           : db.history_models.SearchResult
# Requirement  : `SearchResult` class shall be instantiable and expose the documented interface
# Purpose      : Individual search result item
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
    """Individual search result item."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: str
    source: str  # pubmed, arxiv, etc.
    relevance_score: float
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None  # Highlighted matching text
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchResult.to_dict
    # Requirement  : `to_dict` shall execute as specified
    # Purpose      : To dict
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
        return asdict(self)
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchResult.from_dict
    # Requirement  : `from_dict` shall execute as specified
    # Purpose      : From dict
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: Dict[str, Any]
    # Outputs      : 'SearchResult'
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
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(**data)


# ---------------------------------------------------------------------------
# ID           : db.history_models.SearchQuery
# Requirement  : `SearchQuery` class shall be instantiable and expose the documented interface
# Purpose      : Represents a single search query and its results
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
# Verification : Instantiate SearchQuery with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SearchQuery:
    """Represents a single search query and its results."""
    id: str
    query_text: str
    query_type: str  # "natural", "structured", "semantic", "hybrid"
    timestamp: datetime
    results: List[SearchResult]
    result_count: int
    execution_time_ms: float
    filters: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    starred: bool = False
    notes: Optional[str] = None
    
    # Metadata for analytics
    user_feedback: Optional[str] = None  # "helpful", "not_helpful", None
    clicked_results: List[str] = field(default_factory=list)  # paper_ids clicked
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchQuery.to_dict
    # Requirement  : `to_dict` shall execute as specified
    # Purpose      : To dict
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
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['results'] = [r.to_dict() if isinstance(r, SearchResult) else r for r in self.results]
        return data
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchQuery.from_dict
    # Requirement  : `from_dict` shall execute as specified
    # Purpose      : From dict
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: Dict[str, Any]
    # Outputs      : 'SearchQuery'
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
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['results'] = [
            SearchResult.from_dict(r) if isinstance(r, dict) else r 
            for r in data.get('results', [])
        ]
        return cls(**data)


# ---------------------------------------------------------------------------
# ID           : db.history_models.SearchSession
# Requirement  : `SearchSession` class shall be instantiable and expose the documented interface
# Purpose      : Groups related searches into a session (like a research session)
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
# Verification : Instantiate SearchSession with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SearchSession:
    """Groups related searches into a session (like a research session)."""
    id: str
    name: Optional[str]
    created_at: datetime
    updated_at: datetime
    query_ids: List[str]
    topic: Optional[str] = None
    notes: Optional[str] = None
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchSession.to_dict
    # Requirement  : `to_dict` shall execute as specified
    # Purpose      : To dict
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
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data


# ---------------------------------------------------------------------------
# ID           : db.history_models.SearchHistoryDB
# Requirement  : `SearchHistoryDB` class shall be instantiable and expose the documented interface
# Purpose      : SQLite-based search history storage
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
# Verification : Instantiate SearchHistoryDB with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class SearchHistoryDB:
    """
    SQLite-based search history storage.
    Provides local persistence for search queries and results.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.__init__
    # Requirement  : `__init__` shall initialize the search history database
    # Purpose      : Initialize the search history database
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
        Initialize the search history database.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.eeg_rag/history.db
        """
        if db_path is None:
            db_path = Path.home() / ".eeg_rag" / "history.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB._get_connection
    # Requirement  : `_get_connection` shall context manager for database connections
    # Purpose      : Context manager for database connections
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
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB._init_db
    # Requirement  : `_init_db` shall initialize database schema
    # Purpose      : Initialize database schema
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
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Search queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_queries (
                    id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    result_count INTEGER,
                    execution_time_ms REAL,
                    filters TEXT,
                    session_id TEXT,
                    starred INTEGER DEFAULT 0,
                    notes TEXT,
                    user_feedback TEXT,
                    clicked_results TEXT,
                    query_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Search results table (normalized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    abstract TEXT,
                    source TEXT,
                    relevance_score REAL,
                    doi TEXT,
                    pmid TEXT,
                    url TEXT,
                    snippet TEXT,
                    result_rank INTEGER,
                    FOREIGN KEY (query_id) REFERENCES search_queries(id) ON DELETE CASCADE
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    topic TEXT,
                    notes TEXT
                )
            """)
            
            # Session-query relationship
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_queries (
                    session_id TEXT,
                    query_id TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, query_id),
                    FOREIGN KEY (session_id) REFERENCES search_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (query_id) REFERENCES search_queries(id) ON DELETE CASCADE
                )
            """)
            
            # Saved papers / reading list
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    abstract TEXT,
                    source TEXT,
                    doi TEXT,
                    pmid TEXT,
                    url TEXT,
                    saved_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    tags TEXT,
                    read_status TEXT DEFAULT 'unread'
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON search_queries(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_hash ON search_queries(query_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_starred ON search_queries(starred)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_query ON search_results(query_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_paper ON search_results(paper_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_paper_id ON saved_papers(paper_id)")
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB._hash_query
    # Requirement  : `_hash_query` shall create a hash for query deduplication
    # Purpose      : Create a hash for query deduplication
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_text: str; filters: Optional[Dict] (default=None)
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
    def _hash_query(self, query_text: str, filters: Optional[Dict] = None) -> str:
        """Create a hash for query deduplication."""
        content = query_text.lower().strip()
        if filters:
            content += json.dumps(filters, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    # ==================== Query Operations ====================
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.save_search
    # Requirement  : `save_search` shall save a search query and its results
    # Purpose      : Save a search query and its results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_text: str; query_type: str; results: List[SearchResult]; execution_time_ms: float; filters: Optional[Dict] (default=None); session_id: Optional[str] (default=None)
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
    def save_search(
        self,
        query_text: str,
        query_type: str,
        results: List[SearchResult],
        execution_time_ms: float,
        filters: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> SearchQuery:
        """Save a search query and its results."""
        query_id = str(uuid.uuid4())
        timestamp = datetime.now()
        query_hash = self._hash_query(query_text, filters)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO search_queries 
                (id, query_text, query_type, timestamp, result_count, execution_time_ms,
                 filters, session_id, query_hash, clicked_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, query_text, query_type, timestamp.isoformat(),
                len(results), execution_time_ms,
                json.dumps(filters) if filters else None,
                session_id, query_hash, json.dumps([])
            ))
            
            for rank, result in enumerate(results):
                cursor.execute("""
                    INSERT INTO search_results
                    (query_id, paper_id, title, authors, year, abstract, source,
                     relevance_score, doi, pmid, url, snippet, result_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, result.paper_id, result.title,
                    json.dumps(result.authors), result.year, result.abstract,
                    result.source, result.relevance_score, result.doi,
                    result.pmid, result.url, result.snippet, rank
                ))
            
            if session_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO session_queries (session_id, query_id)
                    VALUES (?, ?)
                """, (session_id, query_id))
                
                cursor.execute("""
                    UPDATE search_sessions SET updated_at = ? WHERE id = ?
                """, (timestamp.isoformat(), session_id))
        
        return SearchQuery(
            id=query_id, query_text=query_text, query_type=query_type,
            timestamp=timestamp, results=results, result_count=len(results),
            execution_time_ms=execution_time_ms, filters=filters, session_id=session_id
        )
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.get_recent_searches
    # Requirement  : `get_recent_searches` shall get recent search queries
    # Purpose      : Get recent search queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : limit: int (default=50); offset: int (default=0); include_results: bool (default=True); starred_only: bool (default=False)
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
    def get_recent_searches(
        self, limit: int = 50, offset: int = 0,
        include_results: bool = True, starred_only: bool = False
    ) -> List[SearchQuery]:
        """Get recent search queries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            where_clause = "WHERE starred = 1" if starred_only else ""
            cursor.execute(f"""
                SELECT * FROM search_queries {where_clause}
                ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """, (limit, offset))
            
            queries = []
            for row in cursor.fetchall():
                results = []
                if include_results:
                    cursor.execute("""
                        SELECT * FROM search_results WHERE query_id = ? ORDER BY result_rank
                    """, (row['id'],))
                    
                    for r in cursor.fetchall():
                        results.append(SearchResult(
                            paper_id=r['paper_id'], title=r['title'],
                            authors=json.loads(r['authors']) if r['authors'] else [],
                            year=r['year'], abstract=r['abstract'] or "",
                            source=r['source'], relevance_score=r['relevance_score'],
                            doi=r['doi'], pmid=r['pmid'], url=r['url'], snippet=r['snippet']
                        ))
                
                queries.append(SearchQuery(
                    id=row['id'], query_text=row['query_text'],
                    query_type=row['query_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    results=results, result_count=row['result_count'],
                    execution_time_ms=row['execution_time_ms'],
                    filters=json.loads(row['filters']) if row['filters'] else None,
                    session_id=row['session_id'], starred=bool(row['starred']),
                    notes=row['notes'], user_feedback=row['user_feedback'],
                    clicked_results=json.loads(row['clicked_results']) if row['clicked_results'] else []
                ))
            
            return queries
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.get_search_by_id
    # Requirement  : `get_search_by_id` shall get a specific search by ID
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
    def get_search_by_id(self, query_id: str) -> Optional[SearchQuery]:
        """Get a specific search by ID."""
        searches = self._get_searches_by_ids([query_id])
        return searches[0] if searches else None
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB._get_searches_by_ids
    # Requirement  : `_get_searches_by_ids` shall get multiple searches by their IDs
    # Purpose      : Get multiple searches by their IDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_ids: List[str]
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
    def _get_searches_by_ids(self, query_ids: List[str]) -> List[SearchQuery]:
        """Get multiple searches by their IDs."""
        if not query_ids:
            return []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(query_ids))
            cursor.execute(f"SELECT * FROM search_queries WHERE id IN ({placeholders})", query_ids)
            
            queries = []
            for row in cursor.fetchall():
                cursor.execute("""
                    SELECT * FROM search_results WHERE query_id = ? ORDER BY result_rank
                """, (row['id'],))
                
                results = []
                for r in cursor.fetchall():
                    results.append(SearchResult(
                        paper_id=r['paper_id'], title=r['title'],
                        authors=json.loads(r['authors']) if r['authors'] else [],
                        year=r['year'], abstract=r['abstract'] or "",
                        source=r['source'], relevance_score=r['relevance_score'],
                        doi=r['doi'], pmid=r['pmid'], url=r['url'], snippet=r['snippet']
                    ))
                
                queries.append(SearchQuery(
                    id=row['id'], query_text=row['query_text'],
                    query_type=row['query_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    results=results, result_count=row['result_count'],
                    execution_time_ms=row['execution_time_ms'],
                    filters=json.loads(row['filters']) if row['filters'] else None,
                    session_id=row['session_id'], starred=bool(row['starred']),
                    notes=row['notes'], user_feedback=row['user_feedback'],
                    clicked_results=json.loads(row['clicked_results']) if row['clicked_results'] else []
                ))
            
            return queries
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.search_history
    # Requirement  : `search_history` shall search through past queries
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
    def search_history(self, search_text: str, limit: int = 20) -> List[SearchQuery]:
        """Search through past queries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM search_queries WHERE query_text LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (f"%{search_text}%", limit))
            
            query_ids = [row['id'] for row in cursor.fetchall()]
            return self._get_searches_by_ids(query_ids)
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.toggle_star
    # Requirement  : `toggle_star` shall toggle starred status. Returns new status
    # Purpose      : Toggle starred status. Returns new status
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
        """Toggle starred status. Returns new status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT starred FROM search_queries WHERE id = ?", (query_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            new_status = 0 if row['starred'] else 1
            cursor.execute("UPDATE search_queries SET starred = ? WHERE id = ?", (new_status, query_id))
            return bool(new_status)
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.add_note
    # Requirement  : `add_note` shall add or update note for a search
    # Purpose      : Add or update note for a search
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
        """Add or update note for a search."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE search_queries SET notes = ? WHERE id = ?", (note, query_id))
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.record_click
    # Requirement  : `record_click` shall record that user clicked on a result
    # Purpose      : Record that user clicked on a result
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
    def record_click(self, query_id: str, paper_id: str):
        """Record that user clicked on a result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT clicked_results FROM search_queries WHERE id = ?", (query_id,))
            row = cursor.fetchone()
            if row:
                clicked = json.loads(row['clicked_results']) if row['clicked_results'] else []
                if paper_id not in clicked:
                    clicked.append(paper_id)
                    cursor.execute(
                        "UPDATE search_queries SET clicked_results = ? WHERE id = ?",
                        (json.dumps(clicked), query_id)
                    )
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.set_feedback
    # Requirement  : `set_feedback` shall set user feedback for a search
    # Purpose      : Set user feedback for a search
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
    def set_feedback(self, query_id: str, feedback: str):
        """Set user feedback for a search."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE search_queries SET user_feedback = ? WHERE id = ?", (feedback, query_id))
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.delete_search
    # Requirement  : `delete_search` shall delete a search and its results
    # Purpose      : Delete a search and its results
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
        """Delete a search and its results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_results WHERE query_id = ?", (query_id,))
            cursor.execute("DELETE FROM search_queries WHERE id = ?", (query_id,))
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.clear_history
    # Requirement  : `clear_history` shall clear search history
    # Purpose      : Clear search history
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : before_date: Optional[datetime] (default=None); keep_starred: bool (default=True)
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
    def clear_history(self, before_date: Optional[datetime] = None, keep_starred: bool = True) -> int:
        """Clear search history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if before_date:
                conditions.append("timestamp < ?")
                params.append(before_date.isoformat())
            
            if keep_starred:
                conditions.append("starred = 0")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor.execute(f"SELECT id FROM search_queries WHERE {where_clause}", params)
            query_ids = [row['id'] for row in cursor.fetchall()]
            
            if query_ids:
                placeholders = ','.join('?' * len(query_ids))
                cursor.execute(f"DELETE FROM search_results WHERE query_id IN ({placeholders})", query_ids)
                cursor.execute(f"DELETE FROM search_queries WHERE id IN ({placeholders})", query_ids)
            
            return len(query_ids)
    
    # ==================== Session Operations ====================
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.create_session
    # Requirement  : `create_session` shall create a new search session
    # Purpose      : Create a new search session
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
    def create_session(self, name: Optional[str] = None, topic: Optional[str] = None) -> SearchSession:
        """Create a new search session."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_sessions (id, name, created_at, updated_at, topic)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, name, now.isoformat(), now.isoformat(), topic))
        
        return SearchSession(
            id=session_id, name=name, created_at=now,
            updated_at=now, query_ids=[], topic=topic
        )
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.get_sessions
    # Requirement  : `get_sessions` shall get recent sessions
    # Purpose      : Get recent sessions
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
        """Get recent sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, GROUP_CONCAT(sq.query_id) as query_ids
                FROM search_sessions s
                LEFT JOIN session_queries sq ON s.id = sq.session_id
                GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append(SearchSession(
                    id=row['id'], name=row['name'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    query_ids=row['query_ids'].split(',') if row['query_ids'] else [],
                    topic=row['topic'], notes=row['notes']
                ))
            
            return sessions
    
    # ==================== Saved Papers ====================
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.save_paper
    # Requirement  : `save_paper` shall save a paper to reading list
    # Purpose      : Save a paper to reading list
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
    def save_paper(self, result: SearchResult, notes: Optional[str] = None, tags: Optional[List[str]] = None):
        """Save a paper to reading list."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO saved_papers
                (paper_id, title, authors, year, abstract, source, doi, pmid, url, notes, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.paper_id, result.title, json.dumps(result.authors),
                result.year, result.abstract, result.source, result.doi,
                result.pmid, result.url, notes, json.dumps(tags) if tags else None
            ))
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.get_saved_papers
    # Requirement  : `get_saved_papers` shall get saved papers with optional filters
    # Purpose      : Get saved papers with optional filters
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
    def get_saved_papers(self, read_status: Optional[str] = None, tag: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get saved papers with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM saved_papers WHERE 1=1"
            params = []
            
            if read_status:
                query += " AND read_status = ?"
                params.append(read_status)
            
            if tag:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
            
            query += " ORDER BY saved_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            papers = []
            for row in cursor.fetchall():
                papers.append({
                    "paper_id": row['paper_id'], "title": row['title'],
                    "authors": json.loads(row['authors']) if row['authors'] else [],
                    "year": row['year'], "abstract": row['abstract'],
                    "source": row['source'], "doi": row['doi'],
                    "pmid": row['pmid'], "url": row['url'],
                    "saved_at": row['saved_at'], "notes": row['notes'],
                    "tags": json.loads(row['tags']) if row['tags'] else [],
                    "read_status": row['read_status']
                })
            
            return papers
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.update_paper_status
    # Requirement  : `update_paper_status` shall update read status of saved paper
    # Purpose      : Update read status of saved paper
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
        """Update read status of saved paper."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE saved_papers SET read_status = ? WHERE paper_id = ?", (status, paper_id))
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.remove_saved_paper
    # Requirement  : `remove_saved_paper` shall remove paper from saved list
    # Purpose      : Remove paper from saved list
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
        """Remove paper from saved list."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM saved_papers WHERE paper_id = ?", (paper_id,))
    
    # ==================== Analytics ====================
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.get_search_stats
    # Requirement  : `get_search_stats` shall get search history statistics
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
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search history statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM search_queries")
            stats['total_searches'] = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT query_type, COUNT(*) as count FROM search_queries GROUP BY query_type
            """)
            stats['by_type'] = {row['query_type']: row['count'] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM search_queries WHERE timestamp > datetime('now', '-30 days')
                GROUP BY DATE(timestamp) ORDER BY date
            """)
            stats['daily_searches'] = [{"date": row['date'], "count": row['count']} for row in cursor.fetchall()]
            
            cursor.execute("""
                SELECT query_text, COUNT(*) as count FROM search_queries
                GROUP BY query_hash ORDER BY count DESC LIMIT 10
            """)
            stats['top_queries'] = [{"query": row['query_text'], "count": row['count']} for row in cursor.fetchall()]
            
            cursor.execute("SELECT AVG(result_count) as avg FROM search_queries")
            stats['avg_results'] = cursor.fetchone()['avg'] or 0
            
            cursor.execute("SELECT COUNT(*) as count FROM search_queries WHERE starred = 1")
            stats['starred_count'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM saved_papers")
            stats['saved_papers'] = cursor.fetchone()['count']
            
            return stats
    
    # ---------------------------------------------------------------------------
    # ID           : db.history_models.SearchHistoryDB.export_history
    # Requirement  : `export_history` shall export search history to file
    # Purpose      : Export search history to file
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
        """Export search history to file."""
        searches = self.get_recent_searches(limit=10000, include_results=True)
        
        if format == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "total_searches": len(searches),
                "searches": [s.to_dict() for s in searches]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'query', 'type', 'result_count', 'execution_time_ms', 'starred', 'feedback'])
                for s in searches:
                    writer.writerow([
                        s.timestamp.isoformat(), s.query_text, s.query_type,
                        s.result_count, s.execution_time_ms, s.starred, s.user_feedback
                    ])
