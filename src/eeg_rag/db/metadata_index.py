# src/eeg_rag/db/metadata_index.py
"""
Lightweight paper metadata index that ships with the repo.
Contains only identifiers - full content fetched on-demand.

This keeps the repository small (~10-50MB) while providing
access to 500K+ EEG research papers.
"""

import gzip
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class PaperReference:
    """
    Minimal paper reference (~100 bytes vs ~1.2KB for full paper).
    Contains only identifiers needed to fetch full content on-demand.
    """
    pmid: Optional[str] = None
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    title: str = ""
    year: Optional[int] = None
    source: str = "unknown"
    keywords: List[str] = field(default_factory=list)
    
    @property
    def primary_id(self) -> str:
        """Get the best available identifier."""
        if self.pmid:
            return f"pmid:{self.pmid}"
        elif self.doi:
            return f"doi:{self.doi}"
        elif self.openalex_id:
            return f"openalex:{self.openalex_id}"
        return f"title:{self.title[:50]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "title": self.title,
            "year": self.year,
            "source": self.source,
            "keywords": self.keywords,
        }
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PaperReference":
        keywords = row["keywords"] or "[]"
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except json.JSONDecodeError:
                keywords = []
        
        return cls(
            pmid=row["pmid"],
            doi=row["doi"],
            openalex_id=row["openalex_id"],
            title=row["title"] or "",
            year=row["year"],
            source=row["source"] or "unknown",
            keywords=keywords,
        )


class MetadataIndex:
    """
    Minimal index containing ~500K paper identifiers.
    
    Compressed size: ~10-30MB (just PMIDs, DOIs, titles)
    
    This index ships with the repository and provides:
    - Fast keyword/title search using FTS5
    - PMID/DOI lookup for batch fetching
    - Year and source filtering
    
    Full paper content is fetched on-demand from PubMed/OpenAlex.
    """
    
    SCHEMA = """
    -- Minimal papers table (< 200 bytes per row)
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pmid TEXT UNIQUE,
        doi TEXT,
        openalex_id TEXT,
        arxiv_id TEXT,
        title TEXT NOT NULL,
        year INTEGER,
        source TEXT DEFAULT 'unknown',
        keywords TEXT,  -- JSON array for search
        indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Indexes for fast lookup
    CREATE INDEX IF NOT EXISTS idx_pmid ON papers(pmid) WHERE pmid IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi) WHERE doi IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_openalex ON papers(openalex_id) WHERE openalex_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_arxiv ON papers(arxiv_id) WHERE arxiv_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_year ON papers(year);
    CREATE INDEX IF NOT EXISTS idx_source ON papers(source);
    
    -- FTS5 for fast full-text search on titles/keywords
    CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
        title,
        keywords,
        content='papers',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    
    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
        INSERT INTO papers_fts(rowid, title, keywords)
        VALUES (new.id, new.title, new.keywords);
    END;
    
    CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, keywords)
        VALUES ('delete', old.id, old.title, old.keywords);
    END;
    
    CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, title, keywords)
        VALUES ('delete', old.id, old.title, old.keywords);
        INSERT INTO papers_fts(rowid, title, keywords)
        VALUES (new.id, new.title, new.keywords);
    END;
    
    -- Statistics table
    CREATE TABLE IF NOT EXISTS index_stats (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """
    
    def __init__(self, db_path: Optional[Path] = None, read_only: bool = True):
        """
        Initialize the metadata index.
        
        Args:
            db_path: Path to the SQLite database. Defaults to data/metadata/index.db
            read_only: If True, opens database in read-only mode (for distribution)
        """
        if db_path is None:
            # Default: ships with repo
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "metadata" / "index.db"
        
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None
        
        # Try to extract if only compressed version exists
        self._maybe_extract_compressed()
    
    def _maybe_extract_compressed(self):
        """Extract compressed index if needed."""
        compressed = self.db_path.with_suffix('.db.gz')
        if compressed.exists() and not self.db_path.exists():
            logger.info(f"Extracting compressed index from {compressed}")
            with gzip.open(compressed, 'rb') as f_in:
                with open(self.db_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            logger.info(f"Extracted to {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        if self.read_only and self.db_path.exists():
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30)
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
        
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        finally:
            conn.close()
    
    def initialize(self):
        """Initialize the database schema (for building new index)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
            # Run migrations for existing databases
            self._migrate(conn)
        logger.info(f"Initialized metadata index at {self.db_path}")
    
    def _migrate(self, conn: sqlite3.Connection):
        """Run database migrations for schema updates."""
        # Check existing columns
        cursor = conn.execute("PRAGMA table_info(papers)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Add arxiv_id column if missing
        if "arxiv_id" not in columns:
            try:
                conn.execute("ALTER TABLE papers ADD COLUMN arxiv_id TEXT")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_arxiv ON papers(arxiv_id) WHERE arxiv_id IS NOT NULL")
                logger.info("Added arxiv_id column to metadata index")
            except sqlite3.OperationalError:
                pass  # Column already exists
    
    def add_reference(
        self,
        title: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        openalex_id: Optional[str] = None,
        year: Optional[int] = None,
        source: str = "unknown",
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """Add a paper reference to the index."""
        if self.read_only:
            raise RuntimeError("Cannot add to read-only index")
        
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO papers 
                    (pmid, doi, openalex_id, title, year, source, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pmid, doi, openalex_id, title, year, source,
                    json.dumps(keywords or [])
                ))
                return True
            except sqlite3.IntegrityError:
                return False
    
    def add_references_batch(self, refs: List[Dict[str, Any]]) -> int:
        """Add multiple references efficiently."""
        if self.read_only:
            raise RuntimeError("Cannot add to read-only index")
        
        added = 0
        with self._get_connection() as conn:
            for ref in refs:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO papers 
                        (pmid, doi, openalex_id, arxiv_id, title, year, source, keywords)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ref.get("pmid"),
                        ref.get("doi"),
                        ref.get("openalex_id"),
                        ref.get("arxiv_id"),
                        ref.get("title", ""),
                        ref.get("year"),
                        ref.get("source", "unknown"),
                        json.dumps(ref.get("keywords", []))
                    ))
                    added += 1
                except sqlite3.IntegrityError:
                    pass
        return added
    
    def search(
        self,
        query: str,
        limit: int = 100,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        source: Optional[str] = None,
    ) -> List[PaperReference]:
        """
        Fast FTS5 search on titles and keywords.
        
        Args:
            query: Search terms
            limit: Maximum results to return
            year_min: Filter by minimum year
            year_max: Filter by maximum year
            source: Filter by source (pubmed, openalex, etc.)
        
        Returns:
            List of PaperReference objects
        """
        if not query or not query.strip():
            return []
        
        # Sanitize query for FTS5
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        
        with self._get_connection() as conn:
            # Build query with optional filters
            sql = """
                SELECT p.pmid, p.doi, p.openalex_id, p.title, p.year, p.source, p.keywords
                FROM papers_fts fts
                JOIN papers p ON fts.rowid = p.id
                WHERE papers_fts MATCH ?
            """
            params: List[Any] = [sanitized]
            
            if year_min:
                sql += " AND p.year >= ?"
                params.append(year_min)
            if year_max:
                sql += " AND p.year <= ?"
                params.append(year_max)
            if source:
                sql += " AND p.source = ?"
                params.append(source)
            
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            
            try:
                cursor = conn.execute(sql, params)
                return [PaperReference.from_row(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError as e:
                logger.warning(f"FTS search failed: {e}")
                return []
    
    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize query for FTS5 safety."""
        # Remove special FTS5 characters
        special_chars = ['*', '"', "'", '-', '+', '(', ')', ':', '^', '~']
        sanitized = query
        for char in special_chars:
            sanitized = sanitized.replace(char, ' ')
        
        # Split into words and filter
        words = [w.strip() for w in sanitized.split() if len(w.strip()) >= 2]
        if not words:
            return ""
        
        return ' '.join(words)
    
    def get_pmids(self, limit: Optional[int] = None) -> Iterator[str]:
        """Iterate over all PMIDs in the index."""
        with self._get_connection() as conn:
            sql = "SELECT pmid FROM papers WHERE pmid IS NOT NULL"
            if limit:
                sql += f" LIMIT {limit}"
            cursor = conn.execute(sql)
            for row in cursor:
                yield row["pmid"]
    
    def get_dois(self, limit: Optional[int] = None) -> Iterator[str]:
        """Iterate over all DOIs in the index."""
        with self._get_connection() as conn:
            sql = "SELECT doi FROM papers WHERE doi IS NOT NULL"
            if limit:
                sql += f" LIMIT {limit}"
            cursor = conn.execute(sql)
            for row in cursor:
                yield row["doi"]
    
    def get_references_for_topic(
        self,
        topic: str,
        limit: int = 1000
    ) -> List[PaperReference]:
        """Get paper references related to a topic."""
        return self.search(topic, limit=limit)
    
    def get_pmids_for_topic(self, topic: str, limit: int = 1000) -> List[str]:
        """Get PMIDs for papers related to a topic."""
        refs = self.search(topic, limit=limit)
        return [r.pmid for r in refs if r.pmid]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._get_connection() as conn:
            stats = {}
            
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            stats["total_papers"] = cursor.fetchone()[0]
            
            # By source
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count 
                FROM papers 
                GROUP BY source 
                ORDER BY count DESC
            """)
            stats["by_source"] = {row["source"]: row["count"] for row in cursor}
            
            # PMID coverage
            cursor = conn.execute("SELECT COUNT(*) FROM papers WHERE pmid IS NOT NULL")
            stats["with_pmid"] = cursor.fetchone()[0]
            
            # DOI coverage
            cursor = conn.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL")
            stats["with_doi"] = cursor.fetchone()[0]
            
            # Year range
            cursor = conn.execute("SELECT MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL")
            row = cursor.fetchone()
            stats["year_range"] = {"min": row[0], "max": row[1]}
            
            return stats
    
    def compress(self, output_path: Optional[Path] = None) -> Path:
        """Compress the index for distribution."""
        if output_path is None:
            output_path = self.db_path.with_suffix('.db.gz')
        
        # Vacuum and optimize first
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        
        # Compress
        with open(self.db_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
                f_out.write(f_in.read())
        
        original_size = self.db_path.stat().st_size
        compressed_size = output_path.stat().st_size
        ratio = compressed_size / original_size * 100
        
        logger.info(
            f"Compressed index: {original_size/1024/1024:.1f}MB -> "
            f"{compressed_size/1024/1024:.1f}MB ({ratio:.1f}%)"
        )
        
        return output_path
    
    def exists(self) -> bool:
        """Check if the index exists."""
        return self.db_path.exists() or self.db_path.with_suffix('.db.gz').exists()


# Convenience function
def get_metadata_index() -> MetadataIndex:
    """Get the default metadata index (read-only)."""
    return MetadataIndex(read_only=True)
