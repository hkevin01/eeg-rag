"""
Memory Management System for Agentic RAG

This module implements both short-term (working) and long-term (persistent) memory
for the EEG-RAG system, enabling context retention across queries and sessions.

Requirements Covered:
- REQ-MEM-001: Short-term memory for current session (last N queries)
- REQ-MEM-002: Long-term memory for persistent knowledge
- REQ-MEM-003: Memory retrieval with relevance scoring
- REQ-MEM-004: Memory consolidation and cleanup
- REQ-MEM-005: Redis/SQLite integration for storage
"""

import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import hashlib

# REQ-MEM-006: Memory entry types
from enum import Enum


class MemoryType(Enum):
    """Type of memory entry"""
    QUERY = "query"
    RESPONSE = "response"
    CONTEXT = "context"
    ENTITY = "entity"
    FACT = "fact"
    USER_PREFERENCE = "user_preference"


@dataclass
class MemoryEntry:
    """
    Single memory entry
    
    REQ-MEM-007: Memory entries must track:
    - Content
    - Type
    - Timestamp
    - Relevance score
    - Metadata
    """
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique ID if not provided"""
        if not self.entry_id:
            # REQ-MEM-008: Generate unique IDs based on content hash
            content_hash = hashlib.md5(
                f"{self.content}{self.timestamp.isoformat()}".encode()
            ).hexdigest()
            self.entry_id = f"{self.memory_type.value}_{content_hash[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary"""
        return cls(
            entry_id=data["entry_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            relevance_score=data["relevance_score"],
            metadata=data["metadata"]
        )
    
    def is_expired(self, ttl_hours: float) -> bool:
        """
        Check if entry is expired
        
        Args:
            ttl_hours: Time to live in hours
            
        Returns:
            True if expired
            
        REQ-MEM-009: Memory expiration logic
        """
        age = (datetime.now() - self.timestamp).total_seconds() / 3600.0
        return age > ttl_hours


class ShortTermMemory:
    """
    Short-term (working) memory for current session
    
    REQ-MEM-010: Implements:
    - Fixed-size buffer (FIFO)
    - Recent queries and responses
    - Fast in-memory access
    - Automatic expiration
    """
    
    def __init__(
        self,
        max_entries: int = 50,
        ttl_hours: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize short-term memory
        
        Args:
            max_entries: Maximum number of entries to keep
            ttl_hours: Time to live in hours
            logger: Logger instance
        """
        self.max_entries = max_entries
        self.ttl_hours = ttl_hours
        self.logger = logger or logging.getLogger("eeg_rag.memory.short_term")
        
        # REQ-MEM-011: Use deque for efficient FIFO operations
        self._memory: deque = deque(maxlen=max_entries)
        
        # REQ-MEM-012: Index for fast lookup by ID
        self._index: Dict[str, MemoryEntry] = {}
        
        self.logger.info(
            f"Initialized ShortTermMemory "
            f"(max_entries={max_entries}, ttl={ttl_hours}h)"
        )
    
    def add(self, entry: MemoryEntry) -> None:
        """
        Add entry to short-term memory
        
        Args:
            entry: Memory entry to add
            
        REQ-MEM-013: FIFO with automatic eviction
        """
        # Check if we're about to evict an entry
        if len(self._memory) == self.max_entries:
            evicted = self._memory[0]
            del self._index[evicted.entry_id]
            self.logger.debug(f"Evicted entry: {evicted.entry_id}")
        
        # Add new entry
        self._memory.append(entry)
        self._index[entry.entry_id] = entry
        
        self.logger.debug(
            f"Added entry {entry.entry_id} "
            f"(type={entry.memory_type.value}, size={len(self._memory)})"
        )
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get entry by ID
        
        Args:
            entry_id: Entry ID
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        entry = self._index.get(entry_id)
        
        # Check expiration
        if entry and entry.is_expired(self.ttl_hours):
            self.logger.debug(f"Entry {entry_id} expired, removing")
            self.remove(entry_id)
            return None
        
        return entry
    
    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """
        Get N most recent entries
        
        Args:
            n: Number of recent entries
            
        Returns:
            List of recent memory entries
            
        REQ-MEM-014: Retrieve recent context
        """
        # Return last N entries (most recent)
        recent = list(self._memory)[-n:]
        
        # Filter out expired entries
        valid_recent = [
            entry for entry in recent
            if not entry.is_expired(self.ttl_hours)
        ]
        
        self.logger.debug(f"Retrieved {len(valid_recent)} recent entries")
        return valid_recent
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memory by content similarity
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            top_k: Number of results
            
        Returns:
            List of (entry, similarity_score) tuples
            
        REQ-MEM-015: Memory search with relevance scoring
        """
        results = []
        query_lower = query.lower()
        
        for entry in self._memory:
            # Skip expired
            if entry.is_expired(self.ttl_hours):
                continue
            
            # Filter by type if specified
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # Simple similarity: word overlap (can be enhanced with embeddings)
            entry_lower = entry.content.lower()
            query_words = set(query_lower.split())
            entry_words = set(entry_lower.split())
            
            if query_words and entry_words:
                overlap = len(query_words & entry_words)
                similarity = overlap / len(query_words | entry_words)
                
                if similarity > 0.1:  # Threshold
                    results.append((entry, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.debug(
            f"Search found {len(results)} matches, returning top {top_k}"
        )
        return results[:top_k]
    
    def remove(self, entry_id: str) -> bool:
        """
        Remove entry by ID
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if removed, False if not found
        """
        if entry_id in self._index:
            entry = self._index[entry_id]
            self._memory.remove(entry)
            del self._index[entry_id]
            self.logger.debug(f"Removed entry: {entry_id}")
            return True
        return False
    
    def clear(self) -> None:
        """
        Clear all entries
        
        REQ-MEM-016: Memory reset functionality
        """
        self._memory.clear()
        self._index.clear()
        self.logger.info("Cleared short-term memory")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
            
        REQ-MEM-017: Automatic cleanup
        """
        expired_ids = [
            entry.entry_id for entry in self._memory
            if entry.is_expired(self.ttl_hours)
        ]
        
        for entry_id in expired_ids:
            self.remove(entry_id)
        
        self.logger.info(f"Cleaned up {len(expired_ids)} expired entries")
        return len(expired_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        type_counts = {}
        for entry in self._memory:
            type_name = entry.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_entries": len(self._memory),
            "max_entries": self.max_entries,
            "utilization": len(self._memory) / self.max_entries,
            "type_counts": type_counts,
            "oldest_entry_age_hours": (
                (datetime.now() - self._memory[0].timestamp).total_seconds() / 3600.0
                if self._memory else 0
            )
        }


class LongTermMemory:
    """
    Long-term (persistent) memory using SQLite
    
    REQ-MEM-018: Implements:
    - Persistent storage across sessions
    - Query history
    - User preferences
    - Validated facts
    - Entity knowledge
    """
    
    def __init__(
        self,
        db_path: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize long-term memory
        
        Args:
            db_path: Path to SQLite database
            logger: Logger instance
        """
        self.db_path = db_path
        self.logger = logger or logging.getLogger("eeg_rag.memory.long_term")
        
        # Create database directory if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Initialized LongTermMemory (db={db_path})")
    
    def _init_database(self) -> None:
        """
        Initialize SQLite database schema
        
        REQ-MEM-019: Database schema for long-term storage
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Memory entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type
                ON memory_entries(memory_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON memory_entries(timestamp DESC)
            """)
            
            conn.commit()
        
        self.logger.debug("Database schema initialized")
    
    def add(self, entry: MemoryEntry) -> None:
        """
        Add entry to long-term memory
        
        Args:
            entry: Memory entry to persist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO memory_entries
                (entry_id, content, memory_type, timestamp, relevance_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.content,
                entry.memory_type.value,
                entry.timestamp.isoformat(),
                entry.relevance_score,
                json.dumps(entry.metadata)
            ))
            
            conn.commit()
        
        self.logger.debug(f"Persisted entry: {entry.entry_id}")
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get entry by ID
        
        Args:
            entry_id: Entry ID
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT entry_id, content, memory_type, timestamp, 
                       relevance_score, metadata
                FROM memory_entries
                WHERE entry_id = ?
            """, (entry_id,))
            
            row = cursor.fetchone()
            
            if row:
                return MemoryEntry(
                    entry_id=row[0],
                    content=row[1],
                    memory_type=MemoryType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    relevance_score=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                )
        
        return None
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search long-term memory
        
        Args:
            query: Search query
            memory_type: Filter by type
            limit: Maximum results
            
        Returns:
            List of matching entries
            
        REQ-MEM-020: Long-term memory search
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            sql = """
                SELECT entry_id, content, memory_type, timestamp,
                       relevance_score, metadata
                FROM memory_entries
                WHERE content LIKE ?
            """
            params = [f"%{query}%"]
            
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type.value)
            
            sql += " ORDER BY relevance_score DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            
            entries = []
            for row in cursor.fetchall():
                entries.append(MemoryEntry(
                    entry_id=row[0],
                    content=row[1],
                    memory_type=MemoryType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    relevance_score=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                ))
        
        self.logger.debug(f"Search found {len(entries)} long-term entries")
        return entries
    
    def get_all_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get all entries of a specific type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT entry_id, content, memory_type, timestamp,
                       relevance_score, metadata
                FROM memory_entries
                WHERE memory_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (memory_type.value, limit))
            
            entries = []
            for row in cursor.fetchall():
                entries.append(MemoryEntry(
                    entry_id=row[0],
                    content=row[1],
                    memory_type=MemoryType(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    relevance_score=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                ))
        
        return entries
    
    def delete_old_entries(self, days: int = 90) -> int:
        """
        Delete entries older than specified days
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of entries deleted
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM memory_entries
                WHERE timestamp < ?
            """, (cutoff,))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        self.logger.info(f"Deleted {deleted_count} entries older than {days} days")
        return deleted_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total count
            cursor.execute("SELECT COUNT(*) FROM memory_entries")
            total = cursor.fetchone()[0]
            
            # Count by type
            cursor.execute("""
                SELECT memory_type, COUNT(*)
                FROM memory_entries
                GROUP BY memory_type
            """)
            type_counts = dict(cursor.fetchall())
            
            # Oldest entry
            cursor.execute("""
                SELECT MIN(timestamp) FROM memory_entries
            """)
            oldest = cursor.fetchone()[0]
            
        return {
            "total_entries": total,
            "type_counts": type_counts,
            "oldest_entry": oldest,
            "database_size_mb": self.db_path.stat().st_size / (1024 * 1024)
            if self.db_path.exists() else 0
        }


class MemoryManager:
    """
    Unified memory manager combining short-term and long-term memory
    
    REQ-MEM-021: Orchestrates both memory systems
    """
    
    def __init__(
        self,
        db_path: Path,
        short_term_max_entries: int = 50,
        short_term_ttl_hours: float = 1.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize memory manager
        
        Args:
            db_path: Path to SQLite database for long-term memory
            short_term_max_entries: Max entries in short-term memory
            short_term_ttl_hours: TTL for short-term entries
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("eeg_rag.memory.manager")
        
        # Initialize both memory systems
        self.short_term = ShortTermMemory(
            max_entries=short_term_max_entries,
            ttl_hours=short_term_ttl_hours,
            logger=self.logger
        )
        
        self.long_term = LongTermMemory(
            db_path=db_path,
            logger=self.logger
        )
        
        self.logger.info("MemoryManager initialized with short-term + long-term")
    
    def add_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add query to both memory systems"""
        entry = MemoryEntry(
            content=query,
            memory_type=MemoryType.QUERY,
            metadata=metadata or {}
        )
        
        # Add to short-term (fast access)
        self.short_term.add(entry)
        
        # Persist to long-term
        self.long_term.add(entry)
        
        self.logger.debug(f"Added query to memory: {entry.entry_id}")
    
    def add_response(
        self,
        response: str,
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add response to memory"""
        metadata = metadata or {}
        if query_id:
            metadata["query_id"] = query_id
        
        entry = MemoryEntry(
            content=response,
            memory_type=MemoryType.RESPONSE,
            metadata=metadata
        )
        
        self.short_term.add(entry)
        self.long_term.add(entry)
        
        self.logger.debug(f"Added response to memory: {entry.entry_id}")
    
    def get_recent_context(self, n: int = 5) -> Dict[str, Any]:
        """
        Get recent conversation context
        
        Args:
            n: Number of recent exchanges
            
        Returns:
            Dictionary with recent queries and responses
        """
        recent = self.short_term.get_recent(n * 2)  # Queries + responses
        
        queries = [e for e in recent if e.memory_type == MemoryType.QUERY]
        responses = [e for e in recent if e.memory_type == MemoryType.RESPONSE]
        
        return {
            "recent_queries": [q.content for q in queries],
            "recent_responses": [r.content for r in responses],
            "context_size": len(recent)
        }
    
    def consolidate(self) -> None:
        """
        Consolidate important short-term memories to long-term
        
        REQ-MEM-022: Memory consolidation logic
        """
        # This is already happening in real-time, but could be enhanced
        # to promote frequently accessed entries, increase relevance scores, etc.
        self.logger.debug("Memory consolidation (already real-time)")
    
    def cleanup(self) -> Dict[str, int]:
        """
        Cleanup both memory systems
        
        Returns:
            Statistics on cleanup operations
        """
        short_term_cleaned = self.short_term.cleanup_expired()
        long_term_cleaned = self.long_term.delete_old_entries(days=90)
        
        stats = {
            "short_term_cleaned": short_term_cleaned,
            "long_term_cleaned": long_term_cleaned
        }
        
        self.logger.info(f"Memory cleanup complete: {stats}")
        return stats
    
    def get_full_statistics(self) -> Dict[str, Any]:
        """Get statistics from both memory systems"""
        return {
            "short_term": self.short_term.get_statistics(),
            "long_term": self.long_term.get_statistics()
        }


# REQ-MEM-023: Export public interface
__all__ = [
    "MemoryType",
    "MemoryEntry",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryManager"
]
