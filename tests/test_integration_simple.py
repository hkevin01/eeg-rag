"""
Simple Integration Tests for EEG-RAG Components

These tests verify basic integration between core components without 
complex mocking or external dependencies.
"""

import unittest
import tempfile
import logging
from pathlib import Path

from eeg_rag.memory.memory_manager import MemoryManager, MemoryType
from eeg_rag.utils.common_utils import check_system_health


class TestBasicIntegration(unittest.TestCase):
    """Basic integration tests for core functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            db_path=self.temp_path / "test_memory.db",
            short_term_max_entries=10,
            short_term_ttl_hours=1.0
        )

    def tearDown(self):
        """Clean up test environment"""
        self.memory_manager.cleanup()
        self.temp_dir.cleanup()

    def test_memory_manager_basic_operations(self):
        """Test basic memory manager operations"""
        # Test adding queries and responses
        self.memory_manager.add_query("What is alpha frequency?")
        self.memory_manager.add_response("Alpha frequency is 8-13 Hz")
        
        # Test getting recent context
        context = self.memory_manager.get_recent_context(5)
        self.assertIn("recent_queries", context)
        self.assertIn("recent_responses", context)
        self.assertEqual(len(context["recent_queries"]), 1)
        self.assertEqual(len(context["recent_responses"]), 1)
        
        # Test statistics
        stats = self.memory_manager.get_full_statistics()
        self.assertIn("short_term", stats)
        self.assertIn("long_term", stats)

    def test_system_health_monitoring(self):
        """Test system health monitoring functionality"""
        health = check_system_health()
        
        # Verify health object structure
        self.assertIsNotNone(health.status)
        self.assertIsInstance(health.cpu_percent, (int, float))
        self.assertIsInstance(health.memory_percent, (int, float))
        self.assertIsInstance(health.warnings, list)
        
        # Test health dictionary conversion
        health_dict = health.to_dict()
        self.assertIn("status", health_dict)
        self.assertIn("cpu_percent", health_dict)
        self.assertIn("memory_percent", health_dict)

    def test_memory_persistence(self):
        """Test that memory persists to database"""
        # Add some test data
        self.memory_manager.add_query("Test query")
        self.memory_manager.add_response("Test response")
        
        # Create new memory manager with same database
        new_memory_manager = MemoryManager(
            db_path=self.temp_path / "test_memory.db",
            short_term_max_entries=10,
            short_term_ttl_hours=1.0
        )
        
        # Check that long-term memory persisted
        long_term_stats = new_memory_manager.get_full_statistics()["long_term"]
        self.assertGreater(long_term_stats["total_entries"], 0)
        
        new_memory_manager.cleanup()

    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        # Add test data
        self.memory_manager.add_query("Test query for cleanup")
        
        # Test cleanup
        cleanup_stats = self.memory_manager.cleanup()
        self.assertIn("short_term_cleaned", cleanup_stats)
        self.assertIn("long_term_cleaned", cleanup_stats)
        self.assertIsInstance(cleanup_stats["short_term_cleaned"], int)
        self.assertIsInstance(cleanup_stats["long_term_cleaned"], int)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
