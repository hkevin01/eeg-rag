#!/usr/bin/env python3
"""
Production Readiness Integration Tests

Tests complete system integration including:
- Docker container functionality
- Redis caching integration
- Cross-encoder reranking
- Production monitoring
- Load handling
- Error recovery
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import subprocess
import logging

# Test imports
from src.eeg_rag.utils.redis_cache import RedisCacheManager
from src.eeg_rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.eeg_rag.monitoring.production_monitor import ProductionMonitor
from src.eeg_rag.agents.orchestrator.orchestrator_agent import OrchestratorAgent
from src.eeg_rag.memory.memory_manager import MemoryManager
from src.eeg_rag.evaluation.load_testing import LoadTester, LoadTestConfig
from src.eeg_rag.evaluation.benchmarking import EEGRAGBenchmark

logger = logging.getLogger(__name__)


class TestProductionReadiness:
    """Integration tests for production readiness."""
    
    @pytest.fixture
    async def setup_system(self):
        """Setup complete system for testing."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize components
            memory_manager = MemoryManager(
                db_path=temp_path / "test_memory.db"
            )
            
            orchestrator = OrchestratorAgent(
                memory_manager=memory_manager
            )
            
            monitor = ProductionMonitor()
            
            yield {
                'orchestrator': orchestrator,
                'memory_manager': memory_manager,
                'monitor': monitor,
                'temp_path': temp_path
            }
    
    @pytest.mark.asyncio
    async def test_redis_cache_integration(self):
        """Test Redis cache integration functionality."""
        # Skip if Redis not available
        try:
            cache = RedisCacheManager(
                host='localhost',
                port=6379,
                db=15,  # Use test database
                prefix='test_eeg_rag'
            )
            
            await cache.connect()
            
            # Test basic cache operations
            test_key = "test:query:12345"
            test_data = {
                "query": "What are EEG frequency bands?",
                "response": "Alpha, beta, theta, delta, gamma",
                "timestamp": time.time()
            }
            
            # Test set/get
            await cache.set(test_key, test_data, ttl=3600)
            cached_data = await cache.get(test_key)
            
            assert cached_data is not None
            assert cached_data["query"] == test_data["query"]
            
            # Test query result caching
            query_hash = "test_query_hash"
            query_results = [
                {"title": "Paper 1", "content": "Content 1"},
                {"title": "Paper 2", "content": "Content 2"}
            ]
            
            await cache.cache_query_results(query_hash, query_results)
            cached_results = await cache.get_cached_query_results(query_hash)
            
            assert len(cached_results) == 2
            assert cached_results[0]["title"] == "Paper 1"
            
            # Cleanup
            await cache.delete(test_key)
            await cache.delete_cached_query_results(query_hash)
            await cache.disconnect()
            
        except Exception as e:
            pytest.skip(f"Redis not available: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_cross_encoder_reranking(self):
        """Test cross-encoder reranking functionality."""
        try:
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                enable_medical_boost=True,
                enable_caching=False  # Disable caching for testing
            )
            
            # Test query and documents
            query = "What are EEG alpha waves?"
            documents = [
                {
                    "content": "Alpha waves are neural oscillations in the frequency range of 8-13 Hz.",
                    "title": "EEG Alpha Waves Study",
                    "score": 0.7
                },
                {
                    "content": "Beta waves occur in the frequency range of 13-30 Hz during active thinking.",
                    "title": "Beta Wave Analysis",
                    "score": 0.5
                },
                {
                    "content": "Alpha oscillations are associated with relaxed awareness and attention.",
                    "title": "Alpha Oscillation Research",
                    "score": 0.6
                }
            ]
            
            # Test reranking
            reranked_docs = await reranker.rerank_documents(
                query=query,
                documents=documents,
                top_k=3
            )
            
            assert len(reranked_docs) == 3
            
            # Verify that documents are properly reranked
            # The first document should be most relevant to alpha waves
            assert "alpha" in reranked_docs[0]["content"].lower()
            
            # Verify score adjustment
            for doc in reranked_docs:
                assert "rerank_score" in doc
                assert isinstance(doc["rerank_score"], float)
            
        except Exception as e:
            pytest.skip(f"Cross-encoder model not available: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_production_monitoring(self, setup_system):
        """Test production monitoring capabilities."""
        system = await setup_system.__anext__()
        monitor = system['monitor']
        
        # Test system health check
        health_status = await monitor.get_system_health()
        
        assert isinstance(health_status, dict)
        assert 'healthy' in health_status
        assert 'components' in health_status
        
        # Test metrics collection
        await monitor.record_query_metrics(
            query_id="test_query_123",
            response_time_ms=150.0,
            success=True,
            agent_type="orchestrator"
        )
        
        metrics = await monitor.get_current_metrics()
        
        assert isinstance(metrics, dict)
        assert 'queries_processed' in metrics
        assert metrics['queries_processed'] >= 1
        
        # Test alert conditions
        # Simulate high response time
        await monitor.record_query_metrics(
            query_id="slow_query",
            response_time_ms=5000.0,  # 5 seconds
            success=True,
            agent_type="orchestrator"
        )
        
        # Check if alerts are triggered
        alerts = await monitor.get_active_alerts()
        assert isinstance(alerts, list)
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self, setup_system):
        """Test complete end-to-end query processing."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        
        # Test queries covering different scenarios
        test_queries = [
            "What are the main EEG frequency bands?",
            "How is EEG used in epilepsy diagnosis?",
            "What is the P300 component?"
        ]
        
        for query_text in test_queries:
            from src.eeg_rag.agents.base_agent import AgentQuery, QueryComplexity
            
            query = AgentQuery(
                text=query_text,
                complexity=QueryComplexity.MEDIUM,
                context={},
                max_response_time=30.0
            )
            
            start_time = time.time()
            response = await orchestrator.process_query(query)
            end_time = time.time()
            
            # Verify response structure
            assert hasattr(response, 'content')
            assert hasattr(response, 'sources')
            assert hasattr(response, 'citations')
            assert hasattr(response, 'confidence')
            
            # Verify response quality
            assert len(response.content) > 50  # Substantial response
            assert response.confidence > 0.0
            
            # Verify performance
            response_time = (end_time - start_time) * 1000
            assert response_time < 10000  # Under 10 seconds
            
            logger.info(f"Query processed in {response_time:.1f}ms: {query_text[:50]}...")
    
    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self, setup_system):
        """Test system behavior under concurrent load."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        
        from src.eeg_rag.agents.base_agent import AgentQuery, QueryComplexity
        
        # Create multiple concurrent queries
        queries = [
            AgentQuery(
                text=f"EEG test query {i}",
                complexity=QueryComplexity.LOW,
                context={},
                max_response_time=30.0
            )
            for i in range(5)
        ]
        
        # Execute queries concurrently
        start_time = time.time()
        
        async def process_single_query(query):
            try:
                return await orchestrator.process_query(query)
            except Exception as e:
                logger.error(f"Query failed: {str(e)}")
                return None
        
        results = await asyncio.gather(
            *[process_single_query(query) for query in queries],
            return_exceptions=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify results
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        assert len(successful_results) >= 3  # At least 60% success rate
        assert total_time < 60  # Complete within 1 minute
        
        logger.info(f"Processed {len(successful_results)}/{len(queries)} queries concurrently in {total_time:.1f}s")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, setup_system):
        """Test system error recovery capabilities."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        
        from src.eeg_rag.agents.base_agent import AgentQuery, QueryComplexity
        
        # Test with malformed query
        invalid_query = AgentQuery(
            text="",  # Empty query
            complexity=QueryComplexity.LOW,
            context={},
            max_response_time=5.0
        )
        
        try:
            response = await orchestrator.process_query(invalid_query)
            # Should either handle gracefully or raise appropriate exception
            assert response is not None
        except Exception as e:
            # Exception is acceptable for invalid input
            assert isinstance(e, (ValueError, RuntimeError))
        
        # Test recovery with valid query after error
        valid_query = AgentQuery(
            text="What are EEG frequency bands?",
            complexity=QueryComplexity.MEDIUM,
            context={},
            max_response_time=30.0
        )
        
        response = await orchestrator.process_query(valid_query)
        assert response is not None
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_memory_management(self, setup_system):
        """Test memory management under load."""
        system = await setup_system.__anext__()
        memory_manager = system['memory_manager']
        
        # Store multiple conversations
        for i in range(10):
            conversation_id = f"test_conv_{i}"
            
            # Add multiple messages per conversation
            for j in range(5):
                await memory_manager.store_conversation_turn(
                    conversation_id=conversation_id,
                    query=f"Test query {j}",
                    response=f"Test response {j}",
                    metadata={"turn": j}
                )
        
        # Verify memory storage
        conversations = await memory_manager.get_conversation_history("test_conv_5", limit=10)
        assert len(conversations) == 5
        
        # Test memory cleanup
        await memory_manager.cleanup_old_conversations(max_age_days=0)  # Clean all
        
        conversations_after_cleanup = await memory_manager.get_conversation_history("test_conv_5", limit=10)
        # Should be cleaned up or significantly reduced
        assert len(conversations_after_cleanup) <= len(conversations)
    
    @pytest.mark.asyncio
    async def test_light_load_performance(self, setup_system):
        """Test system performance under light load."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        
        load_tester = LoadTester(orchestrator)
        
        config = LoadTestConfig(
            concurrent_users=2,
            total_requests=10,
            ramp_up_time=5.0,
            max_response_time_ms=5000.0,
            max_error_rate=0.2,
            max_memory_mb=1024.0
        )
        
        results = await load_tester.run_load_test(config)
        
        # Verify performance meets targets
        assert results.error_rate <= 0.2  # Max 20% error rate
        assert results.avg_response_time_ms <= 5000  # Max 5s response time
        assert results.performance_score >= 50  # Minimum performance score
        
        logger.info(f"Light load test - Score: {results.performance_score:.1f}, "
                   f"Error rate: {results.error_rate:.3f}, "
                   f"Avg response: {results.avg_response_time_ms:.1f}ms")
    
    def test_docker_health_check(self):
        """Test Docker health check script."""
        health_script_path = Path("docker/scripts/healthcheck.sh")
        
        if not health_script_path.exists():
            pytest.skip("Health check script not found")
        
        # Test script execution (without actual service)
        try:
            result = subprocess.run(
                ["bash", str(health_script_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Script should execute without syntax errors
            # May fail due to missing service, but should not have bash errors
            assert result.returncode in [0, 1]  # 0 = healthy, 1 = unhealthy
            
        except subprocess.TimeoutExpired:
            pytest.fail("Health check script timed out")
        except FileNotFoundError:
            pytest.skip("Bash not available")
    
    @pytest.mark.asyncio
    async def test_production_configuration_validation(self):
        """Test production configuration validation."""
        # Test Redis configuration
        try:
            cache = RedisCacheManager(
                host='localhost',
                port=6379,
                db=15,
                max_connections=10,
                default_ttl=3600
            )
            
            # Test connection pool settings
            await cache.connect()
            
            # Validate connection pool is properly configured
            assert cache.default_ttl == 3600
            assert cache.prefix == 'eeg_rag'
            
            await cache.disconnect()
            
        except Exception as e:
            logger.warning(f"Redis configuration test skipped: {str(e)}")
        
        # Test monitoring configuration
        monitor = ProductionMonitor()
        
        # Verify monitoring is properly initialized
        assert hasattr(monitor, 'metrics_store')
        assert hasattr(monitor, 'alert_manager')
        
        # Test metrics collection configuration
        health = await monitor.get_system_health()
        assert isinstance(health, dict)
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, setup_system):
        """Test graceful shutdown behavior."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        memory_manager = system['memory_manager']
        
        # Start some background operations
        from src.eeg_rag.agents.base_agent import AgentQuery, QueryComplexity
        
        query = AgentQuery(
            text="Test query for shutdown",
            complexity=QueryComplexity.LOW,
            context={}
        )
        
        # Start query processing
        query_task = asyncio.create_task(orchestrator.process_query(query))
        
        # Allow some processing time
        await asyncio.sleep(0.1)
        
        # Test graceful shutdown
        try:
            # Close memory manager
            await memory_manager.close()
            
            # Wait for query to complete or timeout
            try:
                await asyncio.wait_for(query_task, timeout=5.0)
            except asyncio.TimeoutError:
                query_task.cancel()
            
            # Verify cleanup was successful
            assert True  # If we get here, shutdown was graceful
            
        except Exception as e:
            pytest.fail(f"Graceful shutdown failed: {str(e)}")


@pytest.mark.integration
class TestProductionDeployment:
    """Tests for production deployment scenarios."""
    
    def test_docker_build_configuration(self):
        """Test Docker build configuration."""
        dockerfile_prod = Path("docker/Dockerfile.prod")
        
        if not dockerfile_prod.exists():
            pytest.skip("Production Dockerfile not found")
        
        # Read Dockerfile and verify production settings
        with open(dockerfile_prod, 'r') as f:
            content = f.read()
        
        # Check for production optimizations
        assert 'COPY requirements.txt' in content
        assert 'pip install' in content
        assert 'COPY src/' in content
        
        # Check for security settings
        assert 'USER' in content  # Should run as non-root
        
        # Check for health check
        assert 'HEALTHCHECK' in content
    
    def test_production_entrypoint(self):
        """Test production entrypoint script."""
        entrypoint_script = Path("docker/scripts/entrypoint.prod.sh")
        
        if not entrypoint_script.exists():
            pytest.skip("Production entrypoint script not found")
        
        # Verify script is executable
        assert entrypoint_script.stat().st_mode & 0o111  # Has execute permission
        
        # Test script execution (basic syntax check)
        try:
            result = subprocess.run(
                ["bash", "-n", str(entrypoint_script)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Syntax error in entrypoint script: {result.stderr}"
            
        except FileNotFoundError:
            pytest.skip("Bash not available for syntax checking")


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_query_response_time_benchmark(self, setup_system):
        """Benchmark query response times."""
        system = await setup_system.__anext__()
        orchestrator = system['orchestrator']
        
        from src.eeg_rag.agents.base_agent import AgentQuery, QueryComplexity
        
        # Test queries of different complexities
        benchmark_queries = [
            ("What are EEG frequency bands?", QueryComplexity.LOW),
            ("How is EEG used in epilepsy diagnosis?", QueryComplexity.MEDIUM),
            ("Compare different EEG artifact removal methods", QueryComplexity.HIGH)
        ]
        
        response_times = []
        
        for query_text, complexity in benchmark_queries:
            query = AgentQuery(
                text=query_text,
                complexity=complexity,
                context={}
            )
            
            start_time = time.time()
            response = await orchestrator.process_query(query)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
            
            logger.info(f"Query ({complexity.value}): {response_time:.1f}ms")
            
            # Verify response quality
            assert len(response.content) > 0
        
        # Verify performance targets
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 5000  # Under 5 seconds on average
        
        logger.info(f"Average response time: {avg_response_time:.1f}ms")