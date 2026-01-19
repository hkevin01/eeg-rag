"""
System resilience and performance testing for EEG-RAG.

These tests verify that the system can handle stress conditions,
recover from failures, and maintain performance under load.

REQ-RES-001: System must handle resource constraints gracefully
REQ-RES-002: Circuit breakers must protect against external failures
REQ-RES-003: Performance must meet acceptable thresholds
REQ-RES-004: System must recover from transient failures
"""

import asyncio
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.utils.common_utils import (
    check_system_health,
    SystemStatus,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    retry_with_backoff
)
from eeg_rag.agents.base_agent import BaseAgent, AgentType, AgentQuery, AgentResult


class MockStressAgent(BaseAgent):
    """Mock agent for stress testing"""
    
    def __init__(self, execution_time: float = 0.1, failure_rate: float = 0.0):
        super().__init__(AgentType.LOCAL_DATA, "stress_agent")
        self.execution_time = execution_time
        self.failure_rate = failure_rate
        self.call_count = 0
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """Execute with configurable timing and failure rate"""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(self.execution_time)
        
        # Simulate failures
        import random
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated failure {self.call_count}")
        
        return AgentResult(
            success=True,
            data=f"Result {self.call_count}",
            metadata={"execution_time": self.execution_time},
            agent_type=self.agent_type
        )


class TestSystemResilience(unittest.IsolatedAsyncioTestCase):
    """Test system resilience under stress conditions"""

    def setUp(self):
        """Set up test environment"""
        # Configure logging for tests
        logging.basicConfig(level=logging.WARNING)
        
    async def test_concurrent_agent_execution(self):
        """Test system with multiple concurrent agents"""
        agents = [MockStressAgent(execution_time=0.1) for _ in range(5)]
        query = AgentQuery(text="concurrent test query")
        
        # Execute all agents concurrently
        start_time = time.time()
        tasks = [agent.run(query) for agent in agents]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        
        # Verify all succeeded
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertTrue(result.success)
        
        # Should complete in roughly parallel time, not sequential
        self.assertLess(elapsed_time, 0.5)  # Much less than 5 * 0.1 = 0.5s

    async def test_agent_failure_recovery(self):
        """Test agent recovery from failures"""
        agent = MockStressAgent(execution_time=0.01, failure_rate=0.7)
        query = AgentQuery(text="failure recovery test")
        
        # Execute multiple times to test failure handling
        success_count = 0
        failure_count = 0
        
        for i in range(10):
            try:
                result = await agent.run(query)
                if result.success:
                    success_count += 1
                else:
                    failure_count += 1
            except Exception:
                failure_count += 1
        
        # Should have both successes and failures
        self.assertGreater(success_count, 0, "Should have some successes")
        self.assertGreater(failure_count, 0, "Should have some failures")
        
        # Agent statistics should be updated
        stats = agent.get_statistics()
        self.assertEqual(stats["total_executions"], 10)
        self.assertEqual(stats["successful_executions"], success_count)
        self.assertEqual(stats["failed_executions"], failure_count)

    async def test_circuit_breaker_protection(self):
        """Test circuit breaker protects against cascading failures"""
        cb = CircuitBreaker("test_service", failure_threshold=3, timeout_seconds=0.5)
        
        call_count = 0
        
        async def failing_service():
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"Service failure {call_count}")
        
        # Trigger circuit breaker opening
        for i in range(3):
            with self.assertRaises(RuntimeError):
                await cb.call(failing_service)
        
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        self.assertEqual(call_count, 3)
        
        # Further calls should be blocked without calling service
        for i in range(3):
            with self.assertRaises(CircuitBreakerOpenError):
                await cb.call(failing_service)
        
        # Service should not have been called additional times
        self.assertEqual(call_count, 3)

    async def test_retry_mechanism_resilience(self):
        """Test retry mechanism with transient failures"""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=5,
            initial_delay=0.01,  # Fast for testing
            max_delay=0.1
        )
        async def intermittent_service(data: str) -> str:
            nonlocal call_count
            call_count += 1
            
            # Fail first 2 attempts, succeed on 3rd
            if call_count < 3:
                raise ConnectionError(f"Transient failure {call_count}")
            return f"Success after {call_count} attempts: {data}"
        
        # Should succeed after retries
        result = await intermittent_service("test_data")
        
        self.assertEqual(result, "Success after 3 attempts: test_data")
        self.assertEqual(call_count, 3)

    async def test_performance_under_load(self):
        """Test performance metrics under load"""
        # Create agent with known performance characteristics
        agent = MockStressAgent(execution_time=0.05)  # 50ms per execution
        
        # Warm up
        query = AgentQuery(text="warmup")
        await agent.run(query)
        
        # Performance test
        num_requests = 20
        start_time = time.time()
        
        tasks = [agent.run(AgentQuery(text=f"request_{i}")) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Verify all succeeded
        self.assertEqual(len(results), num_requests)
        for result in results:
            self.assertTrue(result.success)
        
        # Check performance metrics
        stats = agent.get_statistics()
        avg_time = stats["average_execution_time_seconds"]
        
        # Average execution time should be reasonable
        self.assertLess(avg_time, 0.1)  # Should be under 100ms
        self.assertGreaterEqual(avg_time, 0.0)  # Should be non-negative


class TestResourceMonitoring(unittest.TestCase):
    """Test resource monitoring and health checks"""

    def test_system_health_monitoring(self):
        """Test basic system health monitoring"""
        health = check_system_health()
        
        # Should return valid health data
        self.assertIsNotNone(health)
        
        # Status should be one of the valid enum values
        valid_statuses = {SystemStatus.HEALTHY, SystemStatus.WARNING, SystemStatus.CRITICAL, SystemStatus.UNKNOWN}
        self.assertIn(health.status, valid_statuses)
        
        # Metrics should be non-negative (unless system monitoring failed)
        if health.status != SystemStatus.UNKNOWN:
            self.assertGreaterEqual(health.cpu_percent, 0)
            self.assertGreaterEqual(health.memory_percent, 0)
            self.assertGreaterEqual(health.disk_percent, 0)

    def test_health_thresholds(self):
        """Test health monitoring with different thresholds"""
        # Test with very permissive thresholds (should be healthy)
        health_permissive = check_system_health(
            cpu_warning_threshold=99.0,
            memory_warning_threshold=99.0,
            disk_warning_threshold=99.0
        )
        
        # Test with strict thresholds (likely to trigger warnings)
        health_strict = check_system_health(
            cpu_warning_threshold=1.0,
            memory_warning_threshold=1.0,
            disk_warning_threshold=1.0
        )
        
        # Permissive should be healthier or equal
        if (health_permissive.status != SystemStatus.UNKNOWN and 
            health_strict.status != SystemStatus.UNKNOWN):
            # Health status ordering: HEALTHY < WARNING < CRITICAL
            status_order = {
                SystemStatus.HEALTHY: 0,
                SystemStatus.WARNING: 1,
                SystemStatus.CRITICAL: 2
            }
            
            permissive_level = status_order.get(health_permissive.status, 999)
            strict_level = status_order.get(health_strict.status, 999)
            
            self.assertLessEqual(permissive_level, strict_level)


class TestFailureSimulation(unittest.IsolatedAsyncioTestCase):
    """Test system behavior under simulated failures"""

    async def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure"""
        # Simulate memory pressure by checking health with low thresholds
        health = check_system_health(memory_warning_threshold=10.0, memory_critical_threshold=50.0)
        
        # System should still function even with memory warnings
        self.assertIsNotNone(health)
        
        # If memory usage is high, warnings should be present
        if health.memory_percent > 10.0:
            warning_messages = " ".join(health.warnings)
            if health.memory_percent > 50.0:
                # Should be critical status
                self.assertEqual(health.status, SystemStatus.CRITICAL)
            else:
                # Should be at least warning status
                self.assertIn(health.status, [SystemStatus.WARNING, SystemStatus.CRITICAL])

    async def test_network_failure_simulation(self):
        """Test circuit breaker with simulated network failures"""
        cb = CircuitBreaker("network_service", failure_threshold=2, timeout_seconds=0.1)
        
        async def simulate_network_call():
            # Simulate network timeout
            await asyncio.sleep(0.01)
            raise ConnectionError("Network unavailable")
        
        # Trigger circuit breaker
        for _ in range(2):
            with self.assertRaises(ConnectionError):
                await cb.call(simulate_network_call)
        
        # Circuit should be open
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        
        # Wait for timeout and test recovery
        await asyncio.sleep(0.15)
        
        # Create a successful service for recovery test
        async def working_service():
            return "network_restored"
        
        # Should recover and work
        result = await cb.call(working_service)
        self.assertEqual(result, "network_restored")
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)


class TestPerformanceBenchmarks(unittest.IsolatedAsyncioTestCase):
    """Performance benchmarking tests"""

    async def test_agent_throughput_benchmark(self):
        """Benchmark agent throughput"""
        agent = MockStressAgent(execution_time=0.01)  # 10ms per request
        
        # Measure throughput over time period
        duration = 1.0  # 1 second test
        start_time = time.time()
        completed_requests = 0
        
        while time.time() - start_time < duration:
            query = AgentQuery(text=f"benchmark_request_{completed_requests}")
            result = await agent.run(query)
            if result.success:
                completed_requests += 1
        
        actual_duration = time.time() - start_time
        requests_per_second = completed_requests / actual_duration
        
        # Should achieve reasonable throughput (at least 50 req/sec with 10ms requests)
        self.assertGreater(requests_per_second, 50)
        
        # Log performance metrics
        stats = agent.get_statistics()
        print(f"\nPerformance Benchmark Results:")
        print(f"  Requests per second: {requests_per_second:.1f}")
        print(f"  Total requests: {completed_requests}")
        print(f"  Average execution time: {stats['average_execution_time_seconds']:.3f}s")
        print(f"  Min execution time: {stats['min_execution_time_seconds']:.3f}s")
        print(f"  Max execution time: {stats['max_execution_time_seconds']:.3f}s")

    async def test_memory_efficiency_benchmark(self):
        """Test memory efficiency under sustained load"""
        initial_health = check_system_health()
        initial_memory = initial_health.memory_percent
        
        # Run sustained load
        agent = MockStressAgent(execution_time=0.001)  # Fast execution
        
        for i in range(100):
            query = AgentQuery(text=f"memory_test_{i}")
            result = await agent.run(query)
            self.assertTrue(result.success)
        
        # Check memory after load
        final_health = check_system_health()
        final_memory = final_health.memory_percent
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 5%)
        self.assertLess(memory_increase, 5.0, 
                       f"Memory increased by {memory_increase:.1f}% after load test")


if __name__ == "__main__":
    # Configure detailed logging for resilience tests
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run tests
    unittest.main(verbosity=2)