"""
Unit tests for Orchestrator Agent

Tests cover:
- Execution plan creation
- Dependency resolution
- Parallel group execution
- Result aggregation
"""

import unittest
import asyncio
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    AgentQuery,
    AgentRegistry
)
from eeg_rag.agents.orchestrator.orchestrator_agent import (
    OrchestratorAgent,
    ExecutionNode,
    ExecutionPlan
)
from eeg_rag.planning.query_planner import (
    QueryPlanner,
    QueryPlan,
    ReActAction,
    QueryIntent,
    QueryComplexity
)
from eeg_rag.memory.memory_manager import MemoryManager


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    def __init__(self, agent_type, name, delay=0.1):
        super().__init__(agent_type=agent_type, name=name)
        self.delay = delay
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """Mock execution with delay"""
        await asyncio.sleep(self.delay)
        return AgentResult(
            success=True,
            data={"response": f"Result from {self.name}"},
            metadata={"processed": True},
            agent_type=self.agent_type
        )


class TestExecutionNode(unittest.TestCase):
    """Test ExecutionNode dataclass"""
    
    def test_node_creation(self):
        """Test creating an execution node"""
        action = ReActAction(
            action_type="search",
            reasoning="Need to search",
            parameters={"query": "test"},
            expected_outcome="results",
            agent_name="test_agent",
            parallel_group=0
        )
        
        node = ExecutionNode(
            action=action,
            agent_name="test_agent",
            parallel_group=0
        )
        
        self.assertEqual(node.agent_name, "test_agent")
        self.assertEqual(node.status, AgentStatus.IDLE)
        self.assertIsNone(node.result)
    
    def test_node_completion_status(self):
        """Test node completion checking"""
        action = ReActAction(
            action_type="search",
            reasoning="test",
            parameters={},
            expected_outcome="results",
            agent_name="test_agent",
            parallel_group=0
        )
        
        node = ExecutionNode(action=action, agent_name="test_agent")
        
        # Initially not complete
        self.assertFalse(node.is_complete)
        
        # Mark as completed
        node.status = AgentStatus.COMPLETED
        self.assertTrue(node.is_complete)
        
        # Failed also counts as complete
        node.status = AgentStatus.FAILED
        self.assertTrue(node.is_complete)
    
    def test_node_elapsed_time(self):
        """Test elapsed time calculation"""
        action = ReActAction(
            action_type="search",
            reasoning="test",
            parameters={},
            expected_outcome="results",
            agent_name="test_agent",
            parallel_group=0
        )
        
        node = ExecutionNode(action=action, agent_name="test_agent")
        
        # No time yet
        self.assertEqual(node.elapsed_time, 0.0)
        
        # Set times
        node.start_time = datetime.now()
        node.end_time = datetime.now()
        
        # Should have some time (though very small)
        self.assertGreaterEqual(node.elapsed_time, 0.0)


class TestExecutionPlan(unittest.TestCase):
    """Test ExecutionPlan dataclass"""
    
    def setUp(self):
        """Create test query plan and nodes"""
        self.query_plan = QueryPlan(
            original_query="test query",
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            cot_reasoning=[],
            sub_queries=[],
            actions=[],
            required_agents=set(),
            estimated_latency=1.0,
            confidence=0.9
        )
        
        self.nodes = [
            ExecutionNode(
                action=ReActAction("search", "test", {}, "results", "agent1", 0),
                agent_name="agent1",
                parallel_group=0
            ),
            ExecutionNode(
                action=ReActAction("search", "test", {}, "results", "agent2", 0),
                agent_name="agent2",
                parallel_group=0
            ),
            ExecutionNode(
                action=ReActAction("process", "test", {}, "results", "agent3", 1),
                agent_name="agent3",
                parallel_group=1
            )
        ]
    
    def test_plan_creation(self):
        """Test creating an execution plan"""
        plan = ExecutionPlan(
            query_plan=self.query_plan,
            nodes=self.nodes,
            parallel_groups={}
        )
        
        self.assertEqual(len(plan.nodes), 3)
        self.assertEqual(len(plan.parallel_groups), 2)  # Groups 0 and 1
    
    def test_get_ready_nodes(self):
        """Test getting ready nodes"""
        plan = ExecutionPlan(
            query_plan=self.query_plan,
            nodes=self.nodes,
            parallel_groups={}
        )
        
        # All nodes should be ready (no dependencies)
        ready = plan.get_ready_nodes()
        self.assertEqual(len(ready), 3)
        
        # Mark one as executing
        self.nodes[0].status = AgentStatus.EXECUTING
        ready = plan.get_ready_nodes()
        self.assertEqual(len(ready), 2)
    
    def test_get_next_parallel_group(self):
        """Test getting next parallel group"""
        plan = ExecutionPlan(
            query_plan=self.query_plan,
            nodes=self.nodes,
            parallel_groups={}
        )
        
        # Should get group 0 first
        group = plan.get_next_parallel_group()
        self.assertIsNotNone(group)
        self.assertEqual(len(group), 2)  # agent1 and agent2
        
        # Mark group 0 as complete
        for node in group:
            node.status = AgentStatus.COMPLETED
        
        # Should now get group 1
        group = plan.get_next_parallel_group()
        self.assertIsNotNone(group)
        self.assertEqual(len(group), 1)  # agent3
    
    def test_all_complete(self):
        """Test checking if all nodes are complete"""
        plan = ExecutionPlan(
            query_plan=self.query_plan,
            nodes=self.nodes,
            parallel_groups={}
        )
        
        # Initially not complete
        self.assertFalse(plan.all_complete())
        
        # Mark all as complete
        for node in self.nodes:
            node.status = AgentStatus.COMPLETED
        
        self.assertTrue(plan.all_complete())


class TestOrchestratorAgent(unittest.IsolatedAsyncioTestCase):
    """Test OrchestratorAgent class"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.db"
        
        # Create memory manager
        self.memory = MemoryManager(db_path=self.db_path)
        
        # Create agent registry with mock agents
        self.registry = AgentRegistry()
        self.registry.register(MockAgent(AgentType.LOCAL_DATA, "agent1"))
        self.registry.register(MockAgent(AgentType.WEB_SEARCH, "agent2"))
        
        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            memory_manager=self.memory,
            agent_registry=self.registry
        )
    
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        self.assertEqual(self.orchestrator.agent_type, AgentType.ORCHESTRATOR)
        self.assertEqual(self.orchestrator.name, "orchestrator")
        self.assertIsNotNone(self.orchestrator.memory)
        self.assertIsNotNone(self.orchestrator.registry)
        self.assertIsNotNone(self.orchestrator.planner)
    
    async def test_plan_creation(self):
        """Test execution plan creation from query plan"""
        query_plan = QueryPlan(
            original_query="test query",
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            cot_reasoning=[],
            sub_queries=[],
            actions=[
                ReActAction("search", "test", {}, "results", "agent1", 0),
                ReActAction("search", "test", {}, "results", "agent2", 0),
            ],
            required_agents=set(),
            estimated_latency=1.0,
            confidence=0.9
        )
        
        exec_plan = self.orchestrator._create_execution_plan(query_plan)
        
        self.assertEqual(len(exec_plan.nodes), 2)
        self.assertEqual(len(exec_plan.parallel_groups), 1)
    
    async def test_result_aggregation(self):
        """Test aggregating results from multiple agents"""
        results = [
            AgentResult(
                success=True,
                data={"result": "data1"},
                agent_type=AgentType.LOCAL_DATA
            ),
            AgentResult(
                success=True,
                data={"result": "data2"},
                agent_type=AgentType.WEB_SEARCH
            ),
            AgentResult(
                success=False,
                data={},
                error="Failed",
                agent_type=AgentType.CLOUD_KB
            )
        ]
        
        query_plan = QueryPlan(
            original_query="test",
            intent=QueryIntent.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            cot_reasoning=[],
            sub_queries=[],
            actions=[],
            required_agents=set(),
            estimated_latency=1.0,
            confidence=0.9
        )
        
        aggregated = self.orchestrator._aggregate_results(results, query_plan)
        
        self.assertEqual(aggregated["successful_agents"], 2)
        self.assertEqual(aggregated["failed_agents"], 1)
        self.assertEqual(aggregated["total_agents"], 3)
        self.assertAlmostEqual(aggregated["success_rate"], 2/3)


if __name__ == "__main__":
    unittest.main()
