# src/eeg_rag/db/paper_store.py
"""
Production-grade paper storage for 500K+ EEG research papers.
Uses SQLite with full-text search and efficient indexing.
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
from contextlib import contextmanager
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a research paper with full metadata."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int] = None
    source: str = "unknown"  # pubmed, arxiv, semantic_scholar, openalex
    pmid: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None  # Semantic Scholar ID
    openalex_id: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    citation_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # Full-text content if available
    full_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        # Handle authors as string or list
        authors = data.get('authors', [])
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except:
                authors = [a.strip() for a in authors.split(',') if a.strip()]
        
        # Handle keywords and mesh_terms as string or list
        keywords = data.get('keywords', [])
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except:
                keywords = []
        
        mesh_terms = data.get('mesh_terms', [])
        if isinstance(mesh_terms, str):
            try:
                mesh_terms = json.loads(mesh_terms)
            except:
                mesh_terms = []
        
        return cls(
            paper_id=data.get('paper_id', ''),
            title=data.get('title', ''),
            abstract=data.get('abstract', ''),
            authors=authors,
            year=data.get('year'),
            source=data.get('source', 'unknown'),
            pmid=data.get('pmid'),
            doi=data.get('doi'),
            arxiv_id=data.get('arxiv_id'),
            s2_id=data.get('s2_id'),
            openalex_id=data.get('openalex_id'),
            url=data.get('url'),
            journal=data.get('journal'),
            keywords=keywords,
            mesh_terms=mesh_terms,
            citation_count=data.get('citation_count', 0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            full_text=data.get('full_text')
        )
    
    def get_primary_id(self) -> str:
        """Get the most authoritative ID for this paper."""
        if self.pmid:
            return f"pmid:{self.pmid}"
        elif self.doi:
            return f"doi:{self.doi}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        elif self.s2_id:
            return f"s2:{self.s2_id}"
        elif self.openalex_id:
            return f"openalex:{self.openalex_id}"
        return self.paper_id


class PaperStore:
    """
    SQLite-based paper storage optimized for 500K+ papers.
    
    Features:
    - Full-text search using FTS5
    - Efficient deduplication across sources
    - Batch insertion for bulk loading
    - Source-specific indexing
    - Statistics and analytics
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the paper store.
        
        Args:
            db_path: Path to SQLite database. Defaults to ~/.eeg_rag/papers.db
        """
        if db_path is None:
            db_path = Path.home() / ".eeg_rag" / "papers.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"PaperStore initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema with FTS5 for full-text search."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors TEXT,
                    year INTEGER,
                    source TEXT,
                    pmid TEXT,
                    doi TEXT,
                    arxiv_id TEXT,
                    s2_id TEXT,
                    openalex_id TEXT,
                    url TEXT,
                    journal TEXT,
                    keywords TEXT,
                    mesh_terms TEXT,
                    citation_count INTEGER DEFAULT 0,
                    full_text TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT
                )
            """)
            
            # FTS5 virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    title,
                    abstract,
                    authors,
                    keywords,
                    mesh_terms,
                    content=papers,
                    content_rowid=id
                )
            """)
            
            # Triggers to keep FTS in sync
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                    INSERT INTO papers_fts(rowid, title, abstract, authors, keywords, mesh_terms)
                    VALUES (new.id, new.title, new.abstract, new.authors, new.keywords, new.mesh_terms);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, keywords, mesh_terms)
                    VALUES ('delete', old.id, old.title, old.abstract, old.authors, old.keywords, old.mesh_terms);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors, keywords, mesh_terms)
                    VALUES ('delete', old.id, old.title, old.abstract, old.authors, old.keywords, old.mesh_terms);
                    INSERT INTO papers_fts(rowid, title, abstract, authors, keywords, mesh_terms)
                    VALUES (new.id, new.title, new.abstract, new.authors, new.keywords, new.mesh_terms);
                END
            """)
            
            # Indexes for efficient querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_pmid ON papers(pmid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_s2 ON papers(s2_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_openalex ON papers(openalex_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(content_hash)")
            
            # Statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    papers_added INTEGER DEFAULT 0,
                    papers_updated INTEGER DEFAULT 0,
                    papers_skipped INTEGER DEFAULT 0,
                    last_ingestion TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source)
                )
            """)
    
    def _compute_hash(self, title: str, abstract: str) -> str:
        """Compute a content hash for deduplication."""
        content = f"{title.lower().strip()[:200]}|{abstract.lower().strip()[:500]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_paper(self, paper: Paper, update_if_exists: bool = False) -> bool:
        """
        Add a single paper to the store.
        
        Args:
            paper: Paper object to add
            update_if_exists: If True, update existing paper with new data
            
        Returns:
            True if paper was added/updated, False if skipped
        """
        content_hash = self._compute_hash(paper.title, paper.abstract)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check for existing paper
            cursor.execute("""
                SELECT id FROM papers 
                WHERE paper_id = ? OR content_hash = ? 
                   OR (pmid IS NOT NULL AND pmid = ?)
                   OR (doi IS NOT NULL AND doi = ?)
            """, (paper.paper_id, content_hash, paper.pmid, paper.doi))
            
            existing = cursor.fetchone()
            
            if existing:
                if update_if_exists:
                    cursor.execute("""
                        UPDATE papers SET
                            title = ?, abstract = ?, authors = ?, year = ?,
                            source = ?, pmid = COALESCE(?, pmid), doi = COALESCE(?, doi),
                            arxiv_id = COALESCE(?, arxiv_id), s2_id = COALESCE(?, s2_id),
                            openalex_id = COALESCE(?, openalex_id), url = COALESCE(?, url),
                            journal = COALESCE(?, journal), keywords = ?, mesh_terms = ?,
                            citation_count = CASE WHEN ? > citation_count THEN ? ELSE citation_count END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        paper.title, paper.abstract, json.dumps(paper.authors), paper.year,
                        paper.source, paper.pmid, paper.doi, paper.arxiv_id, paper.s2_id,
                        paper.openalex_id, paper.url, paper.journal,
                        json.dumps(paper.keywords), json.dumps(paper.mesh_terms),
                        paper.citation_count, paper.citation_count, existing['id']
                    ))
                    return True
                return False
            
            # Insert new paper
            cursor.execute("""
                INSERT INTO papers (
                    paper_id, title, abstract, authors, year, source,
                    pmid, doi, arxiv_id, s2_id, openalex_id, url, journal,
                    keywords, mesh_terms, citation_count, full_text, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.paper_id, paper.title, paper.abstract,
                json.dumps(paper.authors), paper.year, paper.source,
                paper.pmid, paper.doi, paper.arxiv_id, paper.s2_id,
                paper.openalex_id, paper.url, paper.journal,
                json.dumps(paper.keywords), json.dumps(paper.mesh_terms),
                paper.citation_count, paper.full_text, content_hash
            ))
            
            return True
    
    def add_papers_batch(
        self, 
        papers: List[Paper], 
        batch_size: int = 1000,
        update_if_exists: bool = False
    ) -> Tuple[int, int, int]:
        """
        Add multiple papers efficiently in batches.
        
        Args:
            papers: List of Paper objects
            batch_size: Number of papers per transaction
            update_if_exists: Update existing papers with new data
            
        Returns:
            (added, updated, skipped) counts
        """
        added = updated = skipped = 0
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for paper in batch:
                    content_hash = self._compute_hash(paper.title, paper.abstract)
                    
                    # Check for existing
                    cursor.execute("""
                        SELECT id FROM papers 
                        WHERE paper_id = ? OR content_hash = ? 
                           OR (pmid IS NOT NULL AND pmid = ?)
                           OR (doi IS NOT NULL AND doi = ?)
                    """, (paper.paper_id, content_hash, paper.pmid, paper.doi))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        if update_if_exists:
                            cursor.execute("""
                                UPDATE papers SET
                                    title = ?, abstract = ?, authors = ?, year = ?,
                                    source = ?, pmid = COALESCE(?, pmid), doi = COALESCE(?, doi),
                                    arxiv_id = COALESCE(?, arxiv_id), s2_id = COALESCE(?, s2_id),
                                    openalex_id = COALESCE(?, openalex_id), url = COALESCE(?, url),
                                    journal = COALESCE(?, journal), keywords = ?, mesh_terms = ?,
                                    citation_count = CASE WHEN ? > citation_count THEN ? ELSE citation_count END,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            """, (
                                paper.title, paper.abstract, json.dumps(paper.authors), paper.year,
                                paper.source, paper.pmid, paper.doi, paper.arxiv_id, paper.s2_id,
                                paper.openalex_id, paper.url, paper.journal,
                                json.dumps(paper.keywords), json.dumps(paper.mesh_terms),
                                paper.citation_count, paper.citation_count, existing['id']
                            ))
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        try:
                            cursor.execute("""
                                INSERT INTO papers (
                                    paper_id, title, abstract, authors, year, source,
                                    pmid, doi, arxiv_id, s2_id, openalex_id, url, journal,
                                    keywords, mesh_terms, citation_count, full_text, content_hash
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                paper.paper_id, paper.title, paper.abstract,
                                json.dumps(paper.authors), paper.year, paper.source,
                                paper.pmid, paper.doi, paper.arxiv_id, paper.s2_id,
                                paper.openalex_id, paper.url, paper.journal,
                                json.dumps(paper.keywords), json.dumps(paper.mesh_terms),
                                paper.citation_count, paper.full_text, content_hash
                            ))
                            added += 1
                        except sqlite3.IntegrityError:
                            skipped += 1
            
            if (i + batch_size) % 10000 == 0:
                logger.info(f"Processed {i + batch_size} papers: {added} added, {updated} updated, {skipped} skipped")
        
        # Update stats
        self._update_stats(papers[0].source if papers else "unknown", added, updated, skipped)
        
        return added, updated, skipped
    
    def _update_stats(self, source: str, added: int, updated: int, skipped: int):
        """Update ingestion statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ingestion_stats (source, papers_added, papers_updated, papers_skipped)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    papers_added = papers_added + excluded.papers_added,
                    papers_updated = papers_updated + excluded.papers_updated,
                    papers_skipped = papers_skipped + excluded.papers_skipped,
                    last_ingestion = CURRENT_TIMESTAMP
            """, (source, added, updated, skipped))
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by its unique ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
            row = cursor.fetchone()
            return Paper.from_dict(dict(row)) if row else None
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Paper]:
        """Get a paper by its PubMed ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE pmid = ?", (pmid,))
            row = cursor.fetchone()
            return Paper.from_dict(dict(row)) if row else None
    
    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """Get a paper by its DOI."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE doi = ?", (doi,))
            row = cursor.fetchone()
            return Paper.from_dict(dict(row)) if row else None
    
    def search_papers(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Full-text search across papers.
        
        Args:
            query: Search query (supports FTS5 syntax)
            limit: Maximum results
            offset: Skip first N results
            year_from: Filter by minimum year
            year_to: Filter by maximum year
            sources: Filter by sources (pubmed, arxiv, etc.)
            
        Returns:
            List of matching papers
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build the query
            sql = """
                SELECT papers.* FROM papers
                JOIN papers_fts ON papers.id = papers_fts.rowid
                WHERE papers_fts MATCH ?
            """
            params: List[Any] = [query]
            
            if year_from:
                sql += " AND papers.year >= ?"
                params.append(year_from)
            
            if year_to:
                sql += " AND papers.year <= ?"
                params.append(year_to)
            
            if sources:
                placeholders = ','.join('?' * len(sources))
                sql += f" AND papers.source IN ({placeholders})"
                params.extend(sources)
            
            sql += " ORDER BY bm25(papers_fts) LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(sql, params)
            return [Paper.from_dict(dict(row)) for row in cursor.fetchall()]
    
    def get_total_count(self) -> int:
        """Get total number of papers in the store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            return cursor.fetchone()[0]
    
    def get_counts_by_source(self) -> Dict[str, int]:
        """Get paper counts grouped by source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source, COUNT(*) as count FROM papers GROUP BY source
            """)
            return {row['source']: row['count'] for row in cursor.fetchall()}
    
    def get_counts_by_year(self, year_from: int = 1990, year_to: int = 2030) -> Dict[int, int]:
        """Get paper counts grouped by year."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT year, COUNT(*) as count FROM papers 
                WHERE year >= ? AND year <= ?
                GROUP BY year ORDER BY year
            """, (year_from, year_to))
            return {row['year']: row['count'] for row in cursor.fetchall()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive store statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM papers")
            total = cursor.fetchone()[0]
            
            # By source
            cursor.execute("SELECT source, COUNT(*) as count FROM papers GROUP BY source")
            by_source = {row['source']: row['count'] for row in cursor.fetchall()}
            
            # By year range
            cursor.execute("""
                SELECT MIN(year) as min_year, MAX(year) as max_year, AVG(year) as avg_year
                FROM papers WHERE year IS NOT NULL
            """)
            year_row = cursor.fetchone()
            
            # With PMIDs
            cursor.execute("SELECT COUNT(*) FROM papers WHERE pmid IS NOT NULL")
            with_pmid = cursor.fetchone()[0]
            
            # With DOIs
            cursor.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL")
            with_doi = cursor.fetchone()[0]
            
            # Ingestion stats
            cursor.execute("SELECT * FROM ingestion_stats")
            ingestion = {row['source']: {
                'added': row['papers_added'],
                'updated': row['papers_updated'],
                'skipped': row['papers_skipped'],
                'last_ingestion': row['last_ingestion']
            } for row in cursor.fetchall()}
            
            return {
                'total_papers': total,
                'by_source': by_source,
                'year_range': {
                    'min': year_row['min_year'],
                    'max': year_row['max_year'],
                    'avg': round(year_row['avg_year']) if year_row['avg_year'] else None
                },
                'with_pmid': with_pmid,
                'with_doi': with_doi,
                'pmid_coverage': round(with_pmid / total * 100, 1) if total else 0,
                'doi_coverage': round(with_doi / total * 100, 1) if total else 0,
                'ingestion_stats': ingestion,
                'db_size_mb': round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0
            }
    
    def iter_papers(
        self, 
        batch_size: int = 1000,
        source: Optional[str] = None
    ) -> Iterator[Paper]:
        """
        Iterate over all papers efficiently.
        
        Args:
            batch_size: Number of papers to fetch per query
            source: Optional source filter
            
        Yields:
            Paper objects
        """
        offset = 0
        
        while True:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if source:
                    cursor.execute(
                        "SELECT * FROM papers WHERE source = ? LIMIT ? OFFSET ?",
                        (source, batch_size, offset)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM papers LIMIT ? OFFSET ?",
                        (batch_size, offset)
                    )
                
                rows = cursor.fetchall()
                
                if not rows:
                    break
                
                for row in rows:
                    yield Paper.from_dict(dict(row))
                
                offset += batch_size
    
    def export_to_jsonl(self, output_path: Path, source: Optional[str] = None) -> int:
        """Export papers to JSONL format for embedding generation."""
        count = 0
        with open(output_path, 'w') as f:
            for paper in self.iter_papers(source=source):
                f.write(json.dumps(paper.to_dict()) + '\n')
                count += 1
        logger.info(f"Exported {count} papers to {output_path}")
        return count


# Singleton instance
_paper_store: Optional[PaperStore] = None


def get_paper_store(db_path: Optional[Path] = None) -> PaperStore:
    """Get or create the singleton PaperStore instance."""
    global _paper_store
    if _paper_store is None:
        _paper_store = PaperStore(db_path)
    return _paper_store
