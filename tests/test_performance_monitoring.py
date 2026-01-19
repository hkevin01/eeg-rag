"""
Tests for Performance Monitoring System

Tests the comprehensive performance monitoring, benchmarking,
and optimization capabilities.
"""

import unittest
import tempfile
import time
import asyncio
from pathlib import Path

from eeg_rag.monitoring import (
    PerformanceMonitor,
    PerformanceMetrics, 
    BenchmarkResult,
    SystemOptimizer,
    monitor_performance
)
from eeg_rag.memory.memory_manager import MemoryManager


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = PerformanceMonitor(
            max_metrics_history=100,
            performance_threshold_ms=50.0,
            memory_threshold_mb=100.0
        )
    
    def test_basic_operation_monitoring(self):
        """Test basic operation monitoring"""
        # Start and end an operation
        op_id = self.monitor.start_operation("test_operation")
        self.assertIsNotNone(op_id)
        self.assertIn("test_operation", op_id)
        
        # Simulate some work
        time.sleep(0.01)  # 10ms
        
        # End operation
        metrics = self.monitor.end_operation(op_id, success=True)
        
        # Verify metrics
        self.assertEqual(metrics.operation_name, "test_operation")
        self.assertTrue(metrics.success)
        self.assertGreater(metrics.duration_ms, 0)
        self.assertIsNone(metrics.error_message)
    
    def test_performance_context_manager(self):
        """Test performance monitoring context manager"""
        with monitor_performance(self.monitor, "context_test") as pm:
            time.sleep(0.005)  # 5ms
        
        # Check that metrics were recorded
        stats = self.monitor.get_operation_statistics("context_test")
        self.assertIn("context_test", stats)
        self.assertEqual(stats["context_test"]["total_operations"], 1)
    
    def test_operation_statistics(self):
        """Test operation statistics collection"""
        # Run multiple operations
        for i in range(5):
            op_id = self.monitor.start_operation("batch_test")
            time.sleep(0.001)  # 1ms
            self.monitor.end_operation(op_id, success=True)
        
        # Get statistics
        stats = self.monitor.get_operation_statistics("batch_test")
        batch_stats = stats["batch_test"]
        
        self.assertEqual(batch_stats["total_operations"], 5)
        self.assertEqual(batch_stats["successful_operations"], 5)
        self.assertEqual(batch_stats["failed_operations"], 0)
        self.assertGreater(batch_stats["avg_duration_ms"], 0)
    
    def test_benchmark_operation(self):
        """Test operation benchmarking"""
        def test_function():
            time.sleep(0.001)  # 1ms simulation
            return "test_result"
        
        # Run benchmark
        result = self.monitor.benchmark_operation(
            operation_func=test_function,
            iterations=10,
            benchmark_name="test_benchmark"
        )
        
        # Verify benchmark result
        self.assertEqual(result.benchmark_name, "test_benchmark")
        self.assertEqual(result.operation_count, 10)
        self.assertGreater(result.avg_duration_ms, 0)
        self.assertEqual(result.success_rate, 1.0)
        self.assertGreater(result.throughput_ops_per_sec, 0)
    
    def test_error_handling_in_monitoring(self):
        """Test error handling during monitoring"""
        def failing_function():
            raise ValueError("Test error")
        
        # Test benchmark with failing function
        result = self.monitor.benchmark_operation(
            operation_func=failing_function,
            iterations=5,
            benchmark_name="error_benchmark"
        )
        
        self.assertEqual(result.success_rate, 0.0)  # All failed
        self.assertEqual(result.operation_count, 5)
    
    def test_metrics_export(self):
        """Test metrics export functionality"""
        # Generate some metrics
        for i in range(3):
            with monitor_performance(self.monitor, f"export_test_{i}"):
                time.sleep(0.001)
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            self.monitor.export_metrics(export_path, format="json")
            self.assertTrue(export_path.exists())
            self.assertGreater(export_path.stat().st_size, 0)
        finally:
            if export_path.exists():
                export_path.unlink()


class TestAsyncBenchmarking(unittest.IsolatedAsyncioTestCase):
    """Test async benchmarking capabilities"""
    
    async def asyncSetUp(self):
        """Set up async test environment"""
        self.monitor = PerformanceMonitor()
    
    async def test_async_benchmark(self):
        """Test async operation benchmarking"""
        async def async_test_function():
            await asyncio.sleep(0.001)  # 1ms simulation
            return "async_result"
        
        # Run async benchmark
        result = await self.monitor.benchmark_async_operation(
            async_operation_func=async_test_function,
            iterations=5,
            benchmark_name="async_test",
            concurrency=2
        )
        
        # Verify results
        self.assertEqual(result.benchmark_name, "async_test")
        self.assertEqual(result.operation_count, 5)
        self.assertEqual(result.success_rate, 1.0)
        self.assertGreater(result.throughput_ops_per_sec, 0)


class TestSystemOptimizer(unittest.TestCase):
    """Test system optimization recommendations"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = PerformanceMonitor(
            performance_threshold_ms=10.0,  # Low threshold for testing
            memory_threshold_mb=50.0
        )
        self.optimizer = SystemOptimizer(self.monitor)
    
    def test_performance_analysis(self):
        """Test performance analysis and recommendations"""
        # Create some slow operations to trigger recommendations
        for i in range(3):
            op_id = self.monitor.start_operation("slow_operation")
            time.sleep(0.020)  # 20ms - above threshold
            self.monitor.end_operation(op_id, success=True)
        
        # Analyze performance
        analysis = self.optimizer.analyze_performance()
        
        # Verify analysis structure
        self.assertIn("performance_summary", analysis)
        self.assertIn("bottlenecks", analysis)
        self.assertIn("resource_usage", analysis)
        self.assertIn("recommendations", analysis)
        
        # Check for bottleneck detection
        bottlenecks = analysis["bottlenecks"]
        self.assertGreater(len(bottlenecks), 0)
        self.assertEqual(bottlenecks[0]["type"], "slow_operation")


class TestRealWorldIntegration(unittest.TestCase):
    """Test monitoring with real EEG-RAG components"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.monitor = PerformanceMonitor()
    
    def tearDown(self):
        """Clean up test environment"""
        self.temp_dir.cleanup()
    
    def test_memory_manager_monitoring(self):
        """Test monitoring MemoryManager operations"""
        # Create memory manager
        memory_manager = MemoryManager(
            db_path=self.temp_path / "test_perf.db",
            short_term_max_entries=10,
            short_term_ttl_hours=1.0
        )
        
        # Monitor memory operations
        with monitor_performance(self.monitor, "memory_add_query"):
            memory_manager.add_query("What is the alpha frequency range?")
        
        with monitor_performance(self.monitor, "memory_add_response"):
            memory_manager.add_response("Alpha frequency is typically 8-13 Hz")
        
        with monitor_performance(self.monitor, "memory_get_context"):
            context = memory_manager.get_recent_context(5)
        
        # Verify monitoring data
        stats = self.monitor.get_operation_statistics()
        self.assertIn("memory_add_query", stats)
        self.assertIn("memory_add_response", stats)
        self.assertIn("memory_get_context", stats)
        
        # All operations should be successful
        for op_name in ["memory_add_query", "memory_add_response", "memory_get_context"]:
            op_stats = stats[op_name]
            self.assertEqual(op_stats["success_rate"], 1.0)
        
        # Cleanup
        memory_manager.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
