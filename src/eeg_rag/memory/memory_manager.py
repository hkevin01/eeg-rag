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

Enhancements:
- REQ-MEM-025: Improved validation with standardized error messages
- REQ-MEM-026: Time unit standardization (all times in seconds)
- REQ-MEM-027: Robust error handling with retry mechanisms
- REQ-MEM-028: Boundary condition validation
- REQ-MEM-029: Memory usage monitoring and optimization
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

# Import common utilities for standardized operations
from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    validate_range,
    standardize_time_unit,
    safe_divide,
    compute_content_hash,
    retry_with_backoff,
    handle_database_operation,
    ensure_directory_exists,
    format_error_message,
    SECOND, MINUTE, HOUR
)

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
    
    REQ-MEM-025: Enhanced validation for all fields
    """
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique ID if not provided and validate all fields"""
        # REQ-MEM-025: Validate content
        self.content = validate_non_empty_string(
            self.content, 
            "content",
            allow_none=False
        )
        
        # REQ-MEM-025: Validate relevance score
        self.relevance_score = validate_range(
            self.relevance_score,
            0.0, 1.0,
            "relevance_score"
        )
        
        # REQ-MEM-025: Ensure metadata is a dict
        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata).__name__}")
        
        # REQ-MEM-008: Generate unique ID if not provided
        if not self.entry_id:
            self.entry_id = compute_content_hash(
                f"{self.content}{self.timestamp.isoformat()}",
                prefix=self.memory_type.value
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        try:
            return {
                "entry_id": self.entry_id,
                "content": self.content,
                "memory_type": self.memory_type.value,
                "timestamp": self.timestamp.isoformat(),
                "relevance_score": self.relevance_score,
                "metadata": self.metadata
            }
        except Exception as e:
            raise ValueError(f"Failed to serialize MemoryEntry: {str(e)}") from e
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary with validation"""
        try:
            # REQ-MEM-025: Validate required fields
            if not isinstance(data, dict):
                raise ValueError(f"Expected dictionary, got {type(data).__name__}")
            
            required_fields = ["content", "memory_type", "timestamp"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return cls(
                entry_id=data.get("entry_id"),
                content=data["content"],
                memory_type=MemoryType(data["memory_type"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                relevance_score=data.get("relevance_score", 1.0),
                metadata=data.get("metadata", {})
            )
        except Exception as e:
            raise ValueError(f"Failed to deserialize MemoryEntry: {str(e)}") from e
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """
        Check if entry is expired
        
        Args:
            ttl_seconds: Time to live in seconds (REQ-MEM-026: standardized to seconds)
            
        Returns:
            True if expired
            
        REQ-MEM-009: Memory expiration logic
        REQ-MEM-026: Time measurements in seconds
        """
        # REQ-MEM-025: Validate TTL
        ttl_seconds = validate_positive_number(ttl_seconds, "ttl_seconds", allow_zero=True)
        
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        return age_seconds > ttl_seconds
    
    def update_relevance_score(self, new_score: float, reason: Optional[str] = None) -> None:
        """
        Update relevance score with validation
        
        Args:
            new_score: New relevance score (0.0 to 1.0)
            reason: Optional reason for the update
            
        REQ-MEM-025: Validated score updates
        """
        old_score = self.relevance_score
        self.relevance_score = validate_range(new_score, 0.0, 1.0, "new_score")
        
        # Track the update in metadata
        if "score_history" not in self.metadata:
            self.metadata["score_history"] = []
        
        self.metadata["score_history"].append({
            "old_score": old_score,
            "new_score": self.relevance_score,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        })
    
    def get_age_seconds(self) -> float:
        """
        Get entry age in seconds
        
        Returns:
            Age in seconds
            
        REQ-MEM-026: Standardized time units
        """
        return (datetime.now() - self.timestamp).total_seconds()


class ShortTermMemory:
    """
    Short-term (working) memory for current session
    
    REQ-MEM-010: Implements:
    - Fixed-size buffer (FIFO)
    - Recent queries and responses
    - Fast in-memory access
    - Automatic expiration
    
    REQ-MEM-025: Enhanced validation and error handling
    REQ-MEM-026: Time measurements standardized to seconds
    """
    
    def __init__(
        self,
        max_entries: int = 50,
        ttl_seconds: float = 3600.0,  # REQ-MEM-026: Default 1 hour in seconds
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize short-term memory
        
        Args:
            max_entries: Maximum number of entries to keep
            ttl_seconds: Time to live in seconds (REQ-MEM-026: standardized)
            logger: Logger instance
            
        Raises:
            ValueError: If parameters are invalid
        """
        # REQ-MEM-025: Validate parameters
        self.max_entries = validate_positive_number(
            max_entries, 
            "max_entries", 
            min_value=1
        )
        self.ttl_seconds = validate_positive_number(
            ttl_seconds, 
            "ttl_seconds", 
            allow_zero=True
        )
        
        self.logger = logger or logging.getLogger("eeg_rag.memory.short_term")
        
        # REQ-MEM-011: Use deque for efficient FIFO operations
        self._memory: deque = deque(maxlen=self.max_entries)
        
        # REQ-MEM-012: Index for fast lookup by ID
        self._index: Dict[str, MemoryEntry] = {}
        
        # REQ-MEM-028: Statistics for monitoring
        self._stats = {
            "total_entries_added": 0,
            "entries_evicted": 0,
            "expired_entries_removed": 0,
            "search_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info(
            f"Initialized ShortTermMemory "
            f"(max_entries={self.max_entries}, ttl={self.ttl_seconds}s)"
        )
    
    def add(self, entry: MemoryEntry) -> None:
        """
        Add entry to short-term memory with validation
        
        Args:
            entry: Memory entry to add
            
        Raises:
            ValueError: If entry is invalid
            
        REQ-MEM-013: FIFO with automatic eviction
        REQ-MEM-025: Entry validation
        """
        # REQ-MEM-025: Validate entry
        if not isinstance(entry, MemoryEntry):
            raise ValueError(f"Expected MemoryEntry, got {type(entry).__name__}")
        
        try:
            # Check if we're about to evict an entry
            if len(self._memory) == self.max_entries:
                evicted = self._memory[0]
                del self._index[evicted.entry_id]
                self._stats["entries_evicted"] += 1
                self.logger.debug(f"Evicted entry: {evicted.entry_id}")
            
            # Add new entry
            self._memory.append(entry)
            self._index[entry.entry_id] = entry
            self._stats["total_entries_added"] += 1
            
            self.logger.debug(
                f"Added entry {entry.entry_id} "
                f"(type={entry.memory_type.value}, size={len(self._memory)})"
            )
            
        except Exception as e:
            error_msg = format_error_message(
                "add entry to short-term memory",
                e,
                {"entry_id": getattr(entry, 'entry_id', 'unknown')}
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get entry by ID with expiration checking
        
        Args:
            entry_id: Entry ID
            
        Returns:
            MemoryEntry if found and not expired, None otherwise
            
        REQ-MEM-025: Input validation and error handling
        """
        # REQ-MEM-025: Validate entry_id
        entry_id = validate_non_empty_string(entry_id, "entry_id")
        
        entry = self._index.get(entry_id)
        
        if entry is None:
            self._stats["cache_misses"] += 1
            return None
        
        # Check expiration
        if entry.is_expired(self.ttl_seconds):
            self.logger.debug(f"Entry {entry_id} expired, removing")
            self.remove(entry_id)
            self._stats["expired_entries_removed"] += 1
            self._stats["cache_misses"] += 1
            return None
        
        self._stats["cache_hits"] += 1
        return entry
    
    def get_recent(self, n: int = 5) -> List[MemoryEntry]:
        """
        Get N most recent entries
        
        Args:
            n: Number of recent entries
            
        Returns:
            List of recent memory entries
            
        Raises:
            ValueError: If n is invalid
            
        REQ-MEM-014: Retrieve recent context
        REQ-MEM-025: Parameter validation
        """
        # REQ-MEM-025: Validate parameter
        n = validate_positive_number(n, "n", min_value=1)
        
        try:
            # Return last N entries (most recent)
            recent = list(self._memory)[-n:]
            
            # Filter out expired entries
            valid_recent = [
                entry for entry in recent
                if not entry.is_expired(self.ttl_seconds)
            ]
            
            self.logger.debug(
                f"Retrieved {len(valid_recent)} recent entries (requested {n})"
            )
            return valid_recent
            
        except Exception as e:
            error_msg = format_error_message(
                "get recent entries",
                e,
                {"requested_count": n, "memory_size": len(self._memory)}
            )
            self.logger.error(error_msg)
            return []  # Return empty list on error rather than crashing
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 5,
        min_similarity: float = 0.1
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memory by content similarity with enhanced validation
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            top_k: Number of results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of (entry, similarity_score) tuples sorted by score
            
        Raises:
            ValueError: If parameters are invalid
            
        REQ-MEM-015: Memory search with relevance scoring
        REQ-MEM-025: Enhanced parameter validation
        """
        # REQ-MEM-025: Validate parameters
        query = validate_non_empty_string(query, "query")
        top_k = validate_positive_number(top_k, "top_k", min_value=1)
        min_similarity = validate_range(min_similarity, 0.0, 1.0, "min_similarity")
        
        self._stats["search_operations"] += 1
        
        try:
            results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            if not query_words:
                self.logger.warning(f"Search query '{query}' contains no valid words")
                return []
            
            for entry in self._memory:
                # Skip expired entries
                if entry.is_expired(self.ttl_seconds):
                    continue
                
                # Filter by type if specified
                if memory_type and entry.memory_type != memory_type:
                    continue
                
                # Compute similarity using word overlap
                entry_lower = entry.content.lower()
                entry_words = set(entry_lower.split())
                
                if not entry_words:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(query_words & entry_words)
                union = len(query_words | entry_words)
                similarity = safe_divide(intersection, union, default=0.0)
                
                # Apply relevance score boost
                boosted_similarity = similarity * entry.relevance_score
                
                if boosted_similarity >= min_similarity:
                    results.append((entry, boosted_similarity))
            
            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            final_results = results[:top_k]
            
            self.logger.debug(
                f"Search '{query}' found {len(results)} matches, "
                f"returning top {len(final_results)}"
            )
            
            return final_results
            
        except Exception as e:
            error_msg = format_error_message(
                "search memory",
                e,
                {
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "memory_type": memory_type.value if memory_type else None,
                    "top_k": top_k
                }
            )
            self.logger.error(error_msg)
            return []  # Return empty list on error
    
    def remove(self, entry_id: str) -> bool:
        """
        Remove entry by ID with validation
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if removed, False if not found
            
        Raises:
            ValueError: If entry_id is invalid
            
        REQ-MEM-025: Input validation
        """
        # REQ-MEM-025: Validate entry_id
        entry_id = validate_non_empty_string(entry_id, "entry_id")
        
        try:
            if entry_id in self._index:
                entry = self._index[entry_id]
                self._memory.remove(entry)
                del self._index[entry_id]
                self.logger.debug(f"Removed entry: {entry_id}")
                return True
            
            self.logger.debug(f"Entry not found for removal: {entry_id}")
            return False
            
        except Exception as e:
            error_msg = format_error_message(
                "remove memory entry",
                e,
                {"entry_id": entry_id}
            )
            self.logger.error(error_msg)
            return False
    
    def clear(self) -> None:
        """
        Clear all entries with statistics tracking
        
        REQ-MEM-016: Memory reset functionality
        REQ-MEM-028: Statistics tracking
        """
        try:
            entries_cleared = len(self._memory)
            self._memory.clear()
            self._index.clear()
            
            # Reset relevant statistics
            self._stats["entries_evicted"] += entries_cleared
            
            self.logger.info(f"Cleared short-term memory ({entries_cleared} entries)")
            
        except Exception as e:
            error_msg = format_error_message("clear short-term memory", e)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries with enhanced error handling
        
        Returns:
            Number of entries removed
            
        REQ-MEM-017: Automatic cleanup
        REQ-MEM-025: Error handling
        """
        try:
            expired_ids = []
            
            # Collect expired entry IDs
            for entry in self._memory:
                if entry.is_expired(self.ttl_seconds):
                    expired_ids.append(entry.entry_id)
            
            # Remove expired entries
            removed_count = 0
            for entry_id in expired_ids:
                if self.remove(entry_id):
                    removed_count += 1
                    self._stats["expired_entries_removed"] += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired entries")
            else:
                self.logger.debug("No expired entries found during cleanup")
            
            return removed_count
            
        except Exception as e:
            error_msg = format_error_message("cleanup expired entries", e)
            self.logger.error(error_msg)
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        
        Returns:
            Dictionary with memory statistics
            
        REQ-MEM-028: Enhanced statistics for monitoring
        """
        try:
            # Count entries by type
            type_counts = {}
            total_age_seconds = 0.0
            
            for entry in self._memory:
                type_name = entry.memory_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                total_age_seconds += entry.get_age_seconds()
            
            # Calculate utilization and averages
            current_size = len(self._memory)
            utilization = safe_divide(current_size, self.max_entries, default=0.0)
            avg_age_seconds = safe_divide(total_age_seconds, current_size, default=0.0)
            
            # Get oldest entry age
            oldest_age_seconds = (
                self._memory[0].get_age_seconds() if self._memory else 0.0
            )
            
            # Cache hit rate
            total_cache_ops = self._stats["cache_hits"] + self._stats["cache_misses"]
            cache_hit_rate = safe_divide(
                self._stats["cache_hits"], 
                total_cache_ops,
                default=0.0
            )
            
            base_stats = {
                "total_entries": current_size,
                "max_entries": self.max_entries,
                "utilization": utilization,
                "ttl_seconds": self.ttl_seconds,
                "type_counts": type_counts,
                "oldest_entry_age_seconds": oldest_age_seconds,
                "average_entry_age_seconds": avg_age_seconds,
                "cache_hit_rate": cache_hit_rate
            }
            
            # Add operational statistics
            base_stats.update(self._stats)
            
            return base_stats
            
        except Exception as e:
            error_msg = format_error_message("get memory statistics", e)
            self.logger.error(error_msg)
            return {
                "error": str(e),
                "total_entries": 0,
                "max_entries": self.max_entries
            }
    
    def get_memory_usage_info(self) -> Dict[str, Any]:
        """
        Get detailed memory usage information for monitoring
        
        Returns:
            Dictionary with memory usage details
            
        REQ-MEM-029: Memory usage monitoring
        """
        try:
            import sys
            
            # Calculate approximate memory usage
            entry_sizes = []
            for entry in self._memory:
                # Rough estimate of entry size
                entry_size = (
                    sys.getsizeof(entry.content) +
                    sys.getsizeof(entry.metadata) +
                    sys.getsizeof(entry.entry_id) +
                    100  # overhead estimate
                )
                entry_sizes.append(entry_size)
            
            total_memory_bytes = sum(entry_sizes)
            avg_entry_size = safe_divide(total_memory_bytes, len(entry_sizes), default=0.0)
            
            return {
                "total_memory_bytes": total_memory_bytes,
                "total_memory_mb": total_memory_bytes / (1024 * 1024),
                "average_entry_size_bytes": avg_entry_size,
                "largest_entry_size_bytes": max(entry_sizes) if entry_sizes else 0,
                "smallest_entry_size_bytes": min(entry_sizes) if entry_sizes else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Could not calculate memory usage: {str(e)}")
            return {"error": "Memory usage calculation failed"}


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
    REQ-MEM-025: Enhanced validation and error handling
    """
    
    def __init__(
        self,
        db_path: Path,
        short_term_max_entries: int = 50,
        short_term_ttl_hours: float = 1.0,  # Keep for backward compatibility
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize memory manager
        
        Args:
            db_path: Path to SQLite database for long-term memory
            short_term_max_entries: Max entries in short-term memory
            short_term_ttl_hours: TTL for short-term entries (converted to seconds internally)
            logger: Logger instance
            
        REQ-MEM-025: Parameter validation
        REQ-MEM-026: Time unit standardization
        """
        # REQ-MEM-025: Validate parameters
        if not isinstance(db_path, Path):
            db_path = Path(db_path)
        
        short_term_max_entries = validate_positive_number(
            short_term_max_entries,
            "short_term_max_entries",
            min_value=1
        )
        
        short_term_ttl_hours = validate_positive_number(
            short_term_ttl_hours,
            "short_term_ttl_hours",
            allow_zero=True
        )
        
        self.logger = logger or logging.getLogger("eeg_rag.memory.manager")
        
        try:
            # REQ-MEM-026: Convert hours to seconds for internal consistency
            ttl_seconds = standardize_time_unit(short_term_ttl_hours, "hours")
            
            # Initialize both memory systems
            self.short_term = ShortTermMemory(
                max_entries=short_term_max_entries,
                ttl_seconds=ttl_seconds,
                logger=self.logger
            )
            
            self.long_term = LongTermMemory(
                db_path=db_path,
                logger=self.logger
            )
            
            self.logger.info(
                f"MemoryManager initialized: short-term ({short_term_max_entries} entries, "
                f"{ttl_seconds}s TTL) + long-term ({db_path})"
            )
            
        except Exception as e:
            error_msg = format_error_message(
                "initialize memory manager",
                e,
                {
                    "db_path": str(db_path),
                    "max_entries": short_term_max_entries,
                    "ttl_hours": short_term_ttl_hours
                }
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
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
