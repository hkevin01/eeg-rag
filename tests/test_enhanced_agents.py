"""
Tests for Enhanced Multi-Agent System

This module tests the advanced coordination, planning, and execution
capabilities of the enhanced agent architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.eeg_rag.agents.base_agent import (
    AgentType, AgentStatus, AgentResult, AgentQuery,
    AgentCoordinator, CircuitBreaker, LoadBalancer, RetryManager
)
from src.eeg_rag.agents.orchestrator.orchestrator_agent import (
    EnhancedOrchestratorAgent, ExecutionNode, ExecutionPlan
)
from src.eeg_rag.planning.enhanced_planner import EnhancedQueryPlanner
from src.eeg_rag.planning.query_planner import QueryPlan, QueryIntent, QueryComplexity


class MockAgent:
    """Mock agent for testing"""
    
    def __init__(self, name: str, agent_type: AgentType):
        self.name = name
        self.agent_type = agent_type
        self.total_executions = 0
        self.successful_executions = 0
        
    async def execute_with_monitoring(self, query: AgentQuery) -> AgentResult:
        """Mock execution"""
        self.total_executions += 1
        
        # Simulate some execution time
        await asyncio.sleep(0.01)
        
        # Simulate 80% success rate
        import random
        success = random.random() > 0.2
        
        if success:
            self.successful_executions += 1
            return AgentResult(
                success=True,
                data=f"Mock result for: {query.text}",
                agent_type=self.agent_type,
                confidence_score=0.85
            )
        else:
            return AgentResult(
                success=False,
                data=None,
                error="Mock execution failed",
                agent_type=self.agent_type
            )


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=30.0)
        
        assert cb.failure_threshold == 3
        assert cb.timeout_seconds == 30.0
        assert cb.can_execute() == True
        assert cb.metrics.total_requests == 0
    
    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking"""
        cb = CircuitBreaker(failure_threshold=2)
        
        # Should allow execution initially
        assert cb.can_execute() == True
        
        # Record failures
        cb.record_failure()
        assert cb.can_execute() == True
        
        cb.record_failure()
        assert cb.can_execute() == False  # Should be open now
        assert cb.metrics.consecutive_failures == 2
    
    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker success reset"""
        cb = CircuitBreaker(failure_threshold=2)
        
        # Record failures to open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() == False
        
        # Force to half-open for testing
        cb.state = cb.state.HALF_OPEN
        
        # Record success should reset
        cb.record_success()
        cb.record_success()
        cb.record_success()
        
        assert cb.metrics.consecutive_failures == 0


class TestLoadBalancer:
    """Test load balancer functionality"""
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization"""
        lb = LoadBalancer(strategy="round_robin")
        
        assert lb.strategy == "round_robin"
        assert len(lb.nodes) == 0
    
    def test_load_balancer_node_management(self):
        """Test adding/removing nodes"""
        lb = LoadBalancer()
        
        # Create mock agents
        agent1 = MockAgent("agent1", AgentType.LOCAL_DATA)
        agent2 = MockAgent("agent2", AgentType.LOCAL_DATA)
        
        # Add nodes
        lb.add_node(agent1, weight=1.0)
        lb.add_node(agent2, weight=2.0)
        
        assert len(lb.nodes) == 2
        assert lb.nodes["agent1"].weight == 1.0
        assert lb.nodes["agent2"].weight == 2.0
        
        # Remove node
        lb.remove_node("agent1")
        assert len(lb.nodes) == 1
        assert "agent1" not in lb.nodes
    
    def test_load_balancer_selection(self):
        """Test agent selection strategies"""
        lb = LoadBalancer(strategy="round_robin")
        
        # Add mock agents
        agent1 = MockAgent("agent1", AgentType.LOCAL_DATA)
        agent2 = MockAgent("agent2", AgentType.LOCAL_DATA)
        
        lb.add_node(agent1)
        lb.add_node(agent2)
        
        # Test round robin selection
        selected1 = lb.select_agent()
        selected2 = lb.select_agent()
        
        assert selected1 != selected2  # Should alternate
    
    def test_load_balancer_health_scoring(self):
        """Test health scoring updates"""
        lb = LoadBalancer()
        
        agent = MockAgent("agent1", AgentType.LOCAL_DATA)
        lb.add_node(agent)
        
        # Test successful update
        lb.update_node_stats("agent1", response_time=0.5, success=True)
        
        node = lb.nodes["agent1"]
        assert node.avg_response_time == 0.5
        assert node.health_score > 0.5  # Should improve
        
        # Test failed update
        initial_health = node.health_score
        lb.update_node_stats("agent1", response_time=2.0, success=False)
        
        assert node.health_score < initial_health  # Should decrease


class TestRetryManager:
    """Test retry manager functionality"""
    
    @pytest.mark.asyncio
    async def test_retry_manager_success(self):
        """Test retry manager with successful operation"""
        retry_manager = RetryManager(max_retries=3)
        
        async def mock_operation():
            return "success"
        
        result = await retry_manager.execute_with_retry(mock_operation)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_manager_eventual_success(self):
        """Test retry manager with eventual success"""
        retry_manager = RetryManager(max_retries=3, base_delay=0.01)
        
        attempt_count = 0
        async def mock_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await retry_manager.execute_with_retry(mock_operation)
        assert result == "success"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_manager_permanent_failure(self):
        """Test retry manager with permanent failure"""
        retry_manager = RetryManager(max_retries=2, base_delay=0.01)
        
        async def mock_operation():
            raise ValueError("Non-retryable error")
        
        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(mock_operation)


class TestAgentCoordinator:
    """Test agent coordinator functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_coordinator_initialization(self):
        """Test agent coordinator initialization"""
        coordinator = AgentCoordinator()
        
        assert len(coordinator.circuit_breakers) == 0
        assert len(coordinator.load_balancers) == 0
        assert coordinator.retry_manager is not None
    
    @pytest.mark.asyncio
    async def test_agent_coordinator_circuit_breaker_integration(self):
        """Test circuit breaker integration"""
        coordinator = AgentCoordinator()
        agent = MockAgent("test_agent", AgentType.LOCAL_DATA)
        
        # Add circuit breaker
        coordinator.add_circuit_breaker("test_agent", failure_threshold=1)
        
        query = AgentQuery(text="test query")
        
        # First execution should work
        result = await coordinator.execute_with_coordination(
            agent, query, use_circuit_breaker=True, use_retry=False
        )
        assert result.success  # Assuming mock agent succeeds
        
        # Force circuit breaker open
        coordinator.circuit_breakers["test_agent"].record_failure()
        
        # Next execution should be blocked
        result = await coordinator.execute_with_coordination(
            agent, query, use_circuit_breaker=True, use_retry=False
        )
        assert not result.success
        assert result.circuit_breaker_triggered


class TestEnhancedQueryPlanner:
    """Test enhanced query planner functionality"""
    
    @pytest.fixture
    def enhanced_planner(self):
        """Create enhanced query planner for testing"""
        return EnhancedQueryPlanner()
    
    @pytest.mark.asyncio
    async def test_enhanced_intent_classification(self, enhanced_planner):
        """Test enhanced intent classification"""
        # Test factual query
        intent = enhanced_planner._classify_intent_enhanced(
            "What are EEG biomarkers?", {}
        )
        assert intent == QueryIntent.FACTUAL
        
        # Test comparison query
        intent = enhanced_planner._classify_intent_enhanced(
            "Compare alpha waves versus beta waves", {}
        )
        assert intent == QueryIntent.COMPARISON
        
        # Test analysis query - "correlation between X and Y" is classified as MULTI_PART
        intent = enhanced_planner._classify_intent_enhanced(
            "Analyze the correlation between EEG patterns and epilepsy", {}
        )
        assert intent == QueryIntent.MULTI_PART  # "between X and Y" triggers multi-part classification
    
    def test_enhanced_complexity_assessment(self, enhanced_planner):
        """Test enhanced complexity assessment"""
        # Simple query
        complexity = enhanced_planner._assess_complexity_enhanced(
            "What is EEG?", {}
        )
        assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        
        # Complex query
        complexity = enhanced_planner._assess_complexity_enhanced(
            "Provide a comprehensive systematic analysis and comparison of "
            "multiple EEG biomarker methodologies for epilepsy detection "
            "including their correlation patterns and clinical effectiveness?", 
            {'multi_modal': True}
        )
        assert complexity == QueryComplexity.VERY_COMPLEX
    
    @pytest.mark.asyncio
    async def test_enhanced_query_planning(self, enhanced_planner):
        """Test enhanced query planning workflow"""
        query_text = "Compare EEG biomarkers for epilepsy vs sleep disorders"
        context = {'research_mode': True}
        user_preferences = {'preferred_agents': ['local_data', 'web_search']}
        
        plan = await enhanced_planner.plan_query_enhanced(
            query_text, context, user_preferences
        )
        
        assert isinstance(plan, QueryPlan)
        assert plan.intent in [QueryIntent.COMPARISON, QueryIntent.REVIEW]
        assert len(plan.actions) > 0
        assert plan.metadata['enhanced_planning'] == True
        assert plan.estimated_time > 0
    
    def test_pattern_caching(self, enhanced_planner):
        """Test pattern caching functionality"""
        # Simulate adding to cache
        cache_key = "comparison_complex_123"
        pattern = {
            'structure': ['Info about A', 'Info about B', 'Compare A vs B'],
            'agents': [['local_data'], ['web_search'], ['knowledge_graph']],
            'dependencies': [[], [], [0, 1]]
        }
        enhanced_planner.pattern_cache[cache_key] = pattern
        
        # Test cache application
        cached_queries = enhanced_planner._apply_cached_pattern(
            "New comparison query", pattern
        )
        
        assert len(cached_queries) == 3
        assert "New comparison query" in cached_queries[0].text
    
    def test_execution_history_recording(self, enhanced_planner):
        """Test execution history recording"""
        mock_plan = QueryPlan(
            original_query="test query",
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            sub_queries=[],
            chain_of_thought=[],
            actions=[]
        )
        
        enhanced_planner.record_execution_result(
            "test query", mock_plan, True, 2.5, ['local_data']
        )
        
        assert len(enhanced_planner.execution_history) == 1
        record = enhanced_planner.execution_history[0]
        assert record['success'] == True
        assert record['execution_time'] == 2.5
        assert record['successful_agents'] == ['local_data']


class TestEnhancedOrchestrator:
    """Test enhanced orchestrator functionality"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for orchestrator testing"""
        from src.eeg_rag.agents.base_agent import AgentRegistry
        from src.eeg_rag.memory.memory_manager import MemoryManager
        
        # Mock registry
        registry = Mock(spec=AgentRegistry)
        registry.get_all.return_value = [
            MockAgent("local_agent", AgentType.LOCAL_DATA),
            MockAgent("web_agent", AgentType.WEB_SEARCH)
        ]
        registry.get_by_type.return_value = [MockAgent("local_agent", AgentType.LOCAL_DATA)]
        registry.get.return_value = MockAgent("local_agent", AgentType.LOCAL_DATA)
        
        # Mock memory manager
        memory_manager = Mock(spec=MemoryManager)
        
        # Mock planner
        planner = Mock(spec=EnhancedQueryPlanner)
        
        return registry, memory_manager, planner
    
    @pytest.mark.asyncio
    async def test_enhanced_orchestrator_initialization(self, mock_components):
        """Test enhanced orchestrator initialization"""
        registry, memory_manager, planner = mock_components
        
        # Mock performance monitor
        from src.eeg_rag.monitoring import PerformanceMonitor
        performance_monitor = Mock(spec=PerformanceMonitor)
        
        orchestrator = EnhancedOrchestratorAgent(
            registry=registry,
            memory_manager=memory_manager,
            planner=planner,
            performance_monitor=performance_monitor,
            enable_coordination=True
        )
        
        assert orchestrator.coordinator is not None
        assert orchestrator.performance_monitor is not None
        assert orchestrator.adaptation_enabled == True
    
    @pytest.mark.asyncio
    async def test_coordination_setup(self, mock_components):
        """Test coordination setup"""
        registry, memory_manager, planner = mock_components
        
        orchestrator = EnhancedOrchestratorAgent(
            registry=registry,
            memory_manager=memory_manager,
            planner=planner,
            enable_coordination=True
        )
        
        # Check that coordination was set up
        assert len(orchestrator.coordinator.circuit_breakers) > 0
    
    def test_coordination_statistics(self, mock_components):
        """Test coordination statistics"""
        registry, memory_manager, planner = mock_components
        
        orchestrator = EnhancedOrchestratorAgent(
            registry=registry,
            memory_manager=memory_manager,
            planner=planner,
            enable_coordination=True
        )
        
        stats = orchestrator.get_coordination_statistics()
        
        assert 'coordination_enabled' in stats
        assert 'performance_monitoring' in stats
        assert 'adaptation_enabled' in stats
        assert stats['coordination_enabled'] == True


class TestAgentIntegration:
    """Integration tests for enhanced agent system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_enhanced_workflow(self):
        """Test complete enhanced workflow"""
        # Create enhanced query planner
        planner = EnhancedQueryPlanner()
        
        # Test planning
        plan = await planner.plan_query_enhanced(
            "Compare EEG alpha waves and beta waves for epilepsy detection",
            context={'research_mode': True},
            user_preferences={'preferred_agents': ['local_data', 'web_search']}
        )
        
        # Verify plan structure
        assert isinstance(plan, QueryPlan)
        assert len(plan.actions) > 0
        assert plan.intent != QueryIntent.UNKNOWN
        assert plan.estimated_time > 0
        
        # Test coordination components
        coordinator = AgentCoordinator()
        coordinator.add_circuit_breaker("test_agent", failure_threshold=3)
        
        assert len(coordinator.circuit_breakers) == 1
        assert coordinator.circuit_breakers["test_agent"].can_execute()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration"""
        from src.eeg_rag.monitoring import PerformanceMonitor
        
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Test with enhanced planner
        planner = EnhancedQueryPlanner()
        
        # Record some execution history
        mock_plan = QueryPlan(
            original_query="test query",
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            sub_queries=[],
            chain_of_thought=[],
            actions=[]
        )
        
        planner.record_execution_result(
            "test query", mock_plan, True, 1.5, ['local_data']
        )
        
        assert len(planner.execution_history) == 1
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization features"""
        planner = EnhancedQueryPlanner()
        
        # Add some execution history
        for i in range(5):
            mock_plan = QueryPlan(
                original_query=f"test query {i}",
                intent=QueryIntent.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                sub_queries=[],
                chain_of_thought=[],
                actions=[]
            )
            
            planner.record_execution_result(
                f"test query {i}", mock_plan, True, 2.0, ['local_data', 'web_search']
            )
        
        # Test history-based optimization
        assert len(planner.execution_history) == 5
        assert planner.adaptive_optimization == True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
