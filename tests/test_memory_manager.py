"""
Unit tests for Memory Management System

Tests cover:
- Short-term memory operations
- Long-term memory persistence
- Memory search and retrieval
- Cleanup operations
"""

import unittest
import tempfile
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.memory.memory_manager import (
    MemoryType,
    MemoryEntry,
    ShortTermMemory,
    LongTermMemory,
    MemoryManager
)


class TestMemoryEntry(unittest.TestCase):
    """Test MemoryEntry dataclass"""

    def test_entry_creation(self):
        """Test creating a memory entry"""
        entry = MemoryEntry(
            content="What is EEG?",
            memory_type=MemoryType.QUERY
        )

        self.assertIsNotNone(entry.entry_id)
        self.assertEqual(entry.content, "What is EEG?")
        self.assertEqual(entry.memory_type, MemoryType.QUERY)
        self.assertEqual(entry.relevance_score, 1.0)

    def test_entry_id_generation(self):
        """Test unique ID generation"""
        entry1 = MemoryEntry(content="Test 1", memory_type=MemoryType.QUERY)
        entry2 = MemoryEntry(content="Test 2", memory_type=MemoryType.QUERY)

        self.assertNotEqual(entry1.entry_id, entry2.entry_id)
        self.assertTrue(entry1.entry_id.startswith("query_"))

    def test_entry_serialization(self):
        """Test to_dict and from_dict"""
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.FACT,
            metadata={"source": "test"}
        )

        data = entry.to_dict()
        restored = MemoryEntry.from_dict(data)

        self.assertEqual(restored.content, entry.content)
        self.assertEqual(restored.memory_type, entry.memory_type)
        self.assertEqual(restored.metadata, entry.metadata)

    def test_entry_expiration(self):
        """Test expiration logic"""
        # Create entry with past timestamp
        entry = MemoryEntry(
            content="Old entry",
            memory_type=MemoryType.QUERY,
            timestamp=datetime.now() - timedelta(hours=2)
        )

        # TTL in seconds: 1 hour = 3600 seconds, 3 hours = 10800 seconds
        self.assertTrue(entry.is_expired(3600.0))  # TTL 1 hour in seconds
        self.assertFalse(entry.is_expired(10800.0))  # TTL 3 hours in seconds


class TestShortTermMemory(unittest.TestCase):
    """Test ShortTermMemory class"""

    def setUp(self):
        """Create fresh short-term memory for each test"""
        # Use ttl_seconds instead of ttl_hours (1 hour = 3600 seconds)
        self.memory = ShortTermMemory(max_entries=10, ttl_seconds=3600.0)

    def test_add_and_get(self):
        """Test adding and retrieving entries"""
        entry = MemoryEntry(
            content="Test query",
            memory_type=MemoryType.QUERY
        )

        self.memory.add(entry)
        retrieved = self.memory.get(entry.entry_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Test query")

    def test_fifo_eviction(self):
        """Test FIFO eviction when max_entries reached"""
        entries = []
        for i in range(15):  # More than max_entries (10)
            entry = MemoryEntry(
                content=f"Query {i}",
                memory_type=MemoryType.QUERY
            )
            entries.append(entry)
            self.memory.add(entry)

        # First 5 should be evicted
        self.assertIsNone(self.memory.get(entries[0].entry_id))
        self.assertIsNone(self.memory.get(entries[4].entry_id))

        # Last 10 should remain
        self.assertIsNotNone(self.memory.get(entries[14].entry_id))
        self.assertIsNotNone(self.memory.get(entries[5].entry_id))

    def test_get_recent(self):
        """Test retrieving recent entries"""
        for i in range(5):
            entry = MemoryEntry(
                content=f"Query {i}",
                memory_type=MemoryType.QUERY
            )
            self.memory.add(entry)

        recent = self.memory.get_recent(n=3)

        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].content, "Query 4")  # Most recent

    def test_search(self):
        """Test searching memory"""
        self.memory.add(MemoryEntry("EEG alpha waves", MemoryType.QUERY))
        self.memory.add(MemoryEntry("ERP P300 component", MemoryType.QUERY))
        self.memory.add(MemoryEntry("Sleep stage classification", MemoryType.QUERY))

        results = self.memory.search("EEG alpha")

        self.assertGreater(len(results), 0)
        self.assertIn("alpha", results[0][0].content.lower())

    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        # Add entry with past timestamp
        old_entry = MemoryEntry(
            content="Old query",
            memory_type=MemoryType.QUERY,
            timestamp=datetime.now() - timedelta(hours=2)
        )
        self.memory.add(old_entry)

        # Add fresh entry
        new_entry = MemoryEntry(
            content="New query",
            memory_type=MemoryType.QUERY
        )
        self.memory.add(new_entry)

        # Cleanup expired entries (TTL is 3600 seconds = 1 hour)
        cleaned = self.memory.cleanup_expired()

        self.assertEqual(cleaned, 1)  # One old entry removed
        self.assertIsNone(self.memory.get(old_entry.entry_id))
        self.assertIsNotNone(self.memory.get(new_entry.entry_id))

    def test_statistics(self):
        """Test memory statistics"""
        self.memory.add(MemoryEntry("Query 1", MemoryType.QUERY))
        self.memory.add(MemoryEntry("Response 1", MemoryType.RESPONSE))
        self.memory.add(MemoryEntry("Query 2", MemoryType.QUERY))

        stats = self.memory.get_statistics()

        self.assertEqual(stats["total_entries"], 3)
        self.assertEqual(stats["type_counts"]["query"], 2)
        self.assertEqual(stats["type_counts"]["response"], 1)


class TestLongTermMemory(unittest.TestCase):
    """Test LongTermMemory class"""

    def setUp(self):
        """Create temporary database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.memory = LongTermMemory(db_path=self.db_path)

    def test_add_and_get(self):
        """Test persisting and retrieving entries"""
        entry = MemoryEntry(
            content="Persistent fact",
            memory_type=MemoryType.FACT,
            metadata={"verified": True}
        )

        self.memory.add(entry)
        retrieved = self.memory.get(entry.entry_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Persistent fact")
        self.assertEqual(retrieved.metadata["verified"], True)

    def test_search(self):
        """Test searching long-term memory"""
        self.memory.add(MemoryEntry("EEG measures brain activity", MemoryType.FACT))
        self.memory.add(MemoryEntry("Alpha waves indicate relaxation", MemoryType.FACT))
        self.memory.add(MemoryEntry("P300 is an ERP component", MemoryType.FACT))

        results = self.memory.search("EEG")

        self.assertGreater(len(results), 0)
        self.assertIn("EEG", results[0].content)

    def test_get_all_by_type(self):
        """Test retrieving entries by type"""
        self.memory.add(MemoryEntry("Query 1", MemoryType.QUERY))
        self.memory.add(MemoryEntry("Fact 1", MemoryType.FACT))
        self.memory.add(MemoryEntry("Query 2", MemoryType.QUERY))

        queries = self.memory.get_all_by_type(MemoryType.QUERY)
        facts = self.memory.get_all_by_type(MemoryType.FACT)

        self.assertEqual(len(queries), 2)
        self.assertEqual(len(facts), 1)

    def test_delete_old_entries(self):
        """Test deleting old entries"""
        # Add old entry
        old_entry = MemoryEntry(
            content="Old fact",
            memory_type=MemoryType.FACT,
            timestamp=datetime.now() - timedelta(days=100)
        )
        self.memory.add(old_entry)

        # Add recent entry
        new_entry = MemoryEntry(
            content="New fact",
            memory_type=MemoryType.FACT
        )
        self.memory.add(new_entry)

        # Delete entries older than 90 days
        deleted = self.memory.delete_old_entries(days=90)

        self.assertEqual(deleted, 1)
        self.assertIsNone(self.memory.get(old_entry.entry_id))
        self.assertIsNotNone(self.memory.get(new_entry.entry_id))

    def test_statistics(self):
        """Test database statistics"""
        self.memory.add(MemoryEntry("Query 1", MemoryType.QUERY))
        self.memory.add(MemoryEntry("Fact 1", MemoryType.FACT))

        stats = self.memory.get_statistics()

        self.assertEqual(stats["total_entries"], 2)
        self.assertIn("query", stats["type_counts"])
        self.assertIn("fact", stats["type_counts"])


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager integration"""

    def setUp(self):
        """Create memory manager with temporary database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        self.manager = MemoryManager(
            db_path=self.db_path,
            short_term_max_entries=20,
            short_term_ttl_hours=1.0
        )

    def test_add_query_and_response(self):
        """Test adding query-response pairs"""
        self.manager.add_query("What is EEG?")
        self.manager.add_response(
            "EEG measures electrical brain activity",
            metadata={"confidence": 0.95}
        )

        # Check both memory systems
        short_stats = self.manager.short_term.get_statistics()
        long_stats = self.manager.long_term.get_statistics()

        self.assertEqual(short_stats["total_entries"], 2)
        self.assertEqual(long_stats["total_entries"], 2)

    def test_get_recent_context(self):
        """Test retrieving recent conversation context"""
        self.manager.add_query("Query 1")
        self.manager.add_response("Response 1")
        self.manager.add_query("Query 2")
        self.manager.add_response("Response 2")

        context = self.manager.get_recent_context(n=2)

        self.assertEqual(len(context["recent_queries"]), 2)
        self.assertEqual(len(context["recent_responses"]), 2)

    def test_cleanup(self):
        """Test memory cleanup"""
        # Add some entries
        self.manager.add_query("Test query")
        self.manager.add_response("Test response")

        # Run cleanup
        stats = self.manager.cleanup()

        self.assertIn("short_term_cleaned", stats)
        self.assertIn("long_term_cleaned", stats)

    def test_full_statistics(self):
        """Test combined statistics"""
        self.manager.add_query("Query 1")
        self.manager.add_response("Response 1")

        stats = self.manager.get_full_statistics()

        self.assertIn("short_term", stats)
        self.assertIn("long_term", stats)
        self.assertEqual(stats["short_term"]["total_entries"], 2)


if __name__ == "__main__":
    unittest.main()
