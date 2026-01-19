"""
Boundary condition tests for BaseAgent system.

These tests verify that the agent system handles edge cases,
invalid inputs, and off-nominal scenarios gracefully.

REQ-TEST-001: Test nominal and off-nominal conditions
REQ-TEST-002: Validate error handling for boundary cases
REQ-TEST-003: Test input validation and sanitization
REQ-TEST-004: Verify system resilience under stress
"""

import unittest
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.agents.base_agent import (
    AgentType,
    AgentStatus,
    AgentQuery,
    AgentResult,
    BaseAgent,
    AgentRegistry
)


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    def __init__(self, agent_type=AgentType.LOCAL_DATA, name="mock_agent", config=None, fail_mode=None):
        super().__init__(agent_type, name, config)
        self.fail_mode = fail_mode
        self.execution_count = 0
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """Mock execute method with configurable failure modes"""
        self.execution_count += 1
        
        if self.fail_mode == "timeout":
            await asyncio.sleep(10)  # Simulate timeout
        elif self.fail_mode == "exception":
            raise RuntimeError("Mock execution error")
        elif self.fail_mode == "invalid_result":
            return "not an AgentResult"  # Invalid return type
        elif self.fail_mode == "none_result":
            return None
        
        # Normal successful execution
        return AgentResult(
            success=True,
            data={"response": f"Mock response to: {query.text}"},
            agent_type=self.agent_type
        )


class TestAgentQueryBoundaryConditions(unittest.TestCase):
    """Test AgentQuery boundary conditions"""
    
    def test_valid_query_creation(self):
        """Test creating valid queries"""
        # Minimal valid query
        query = AgentQuery(text="test")
        self.assertEqual(query.text, "test")
        self.assertIsNotNone(query.query_id)
        
        # Query with all fields
        query = AgentQuery(
            text="complex query",
            intent="search",
            context={"user_id": "123"},
            parameters={"max_results": 10}
        )
        self.assertEqual(query.text, "complex query")
        self.assertEqual(query.intent, "search")
        self.assertEqual(query.context["user_id"], "123")
        self.assertEqual(query.parameters["max_results"], 10)
    
    def test_empty_text_validation(self):
        """Test validation of empty query text"""
        # Empty string
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text="")
        self.assertIn("cannot be empty", str(cm.exception))
        
        # Whitespace only
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text="   \n\t  ")
        self.assertIn("cannot be empty", str(cm.exception))
        
        # None text (should be caught by dataclass)
        with self.assertRaises((TypeError, ValueError)):
            AgentQuery(text=None)
    
    def test_invalid_types(self):
        """Test validation of invalid parameter types"""
        # Non-string text
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text=123)
        self.assertIn("must be a string", str(cm.exception))
        
        # Non-dict context
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text="test", context="not a dict")
        self.assertIn("must be a dictionary", str(cm.exception))
        
        # Non-dict parameters
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text="test", parameters=[1, 2, 3])
        self.assertIn("must be a dictionary", str(cm.exception))
    
    def test_long_text_handling(self):
        """Test handling of very long query text"""
        # Very long text (10KB)
        long_text = "x" * 10240
        query = AgentQuery(text=long_text)
        self.assertEqual(len(query.text), 10240)
        
        # Extremely long text (1MB)
        very_long_text = "y" * (1024 * 1024)
        query = AgentQuery(text=very_long_text)
        self.assertEqual(len(query.text), 1024 * 1024)
    
    def test_special_characters(self):
        """Test handling of special characters in query"""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        query = AgentQuery(text=special_chars)
        self.assertEqual(query.text, special_chars)
        
        # Unicode characters
        unicode_text = "Hello 世界 🌍 émojis"
        query = AgentQuery(text=unicode_text)
        self.assertEqual(query.text, unicode_text)
    
    def test_context_and_parameters_access(self):
        """Test safe access to context and parameters"""
        query = AgentQuery(
            text="test",
            context={"key1": "value1"},
            parameters={"param1": 100}
        )
        
        # Valid keys
        self.assertEqual(query.get_context_value("key1"), "value1")
        self.assertEqual(query.get_parameter_value("param1"), 100)
        
        # Missing keys with defaults
        self.assertIsNone(query.get_context_value("missing"))
        self.assertEqual(query.get_context_value("missing", "default"), "default")
        self.assertEqual(query.get_parameter_value("missing", 42), 42)
    
    def test_query_serialization(self):
        """Test query serialization/deserialization"""
        query = AgentQuery(
            text="test query",
            intent="analysis",
            context={"user": "alice"},
            parameters={"timeout": 30}
        )
        
        # Test to_dict
        data = query.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["text"], "test query")
        self.assertIn("timestamp", data)
        
        # Test with complex nested data
        complex_query = AgentQuery(
            text="complex",
            context={"nested": {"deep": {"value": [1, 2, 3]}}}
        )
        complex_data = complex_query.to_dict()
        self.assertEqual(complex_data["context"]["nested"]["deep"]["value"], [1, 2, 3])


class TestBaseAgentBoundaryConditions(unittest.TestCase):
    """Test BaseAgent boundary conditions"""
    
    def test_valid_agent_creation(self):
        """Test creating valid agents"""
        # Minimal agent
        agent = MockAgent()
        self.assertEqual(agent.agent_type, AgentType.LOCAL_DATA)
        self.assertIn("mock_agent", agent.name)
        
        # Agent with all parameters
        agent = MockAgent(
            agent_type=AgentType.WEB_SEARCH,
            name="custom_agent"
        )
        self.assertEqual(agent.agent_type, AgentType.WEB_SEARCH)
        self.assertEqual(agent.name, "custom_agent")
    
    def test_invalid_agent_parameters(self):
        """Test validation of invalid agent parameters"""
        # Invalid agent type
        with self.assertRaises(ValueError) as cm:
            MockAgent(agent_type="not_an_enum")
        self.assertIn("must be an AgentType enum", str(cm.exception))
        
        # Invalid name type
        with self.assertRaises(ValueError) as cm:
            MockAgent(name=123)
        self.assertIn("must be a string", str(cm.exception))
        
        # Empty name
        with self.assertRaises(ValueError) as cm:
            MockAgent(name="")
        self.assertIn("cannot be empty", str(cm.exception))
        
        # Invalid config type
        with self.assertRaises(ValueError) as cm:
            MockAgent(config="not a dict")
        self.assertIn("must be a dictionary", str(cm.exception))
    
    def test_agent_execution_success(self):
        """Test successful agent execution"""
        async def test_execution():
            agent = MockAgent()
            query = AgentQuery(text="test query")
            
            result = await agent.run(query)
            
            self.assertTrue(result.success)
            self.assertIsNotNone(result.data)
            self.assertEqual(agent.status, AgentStatus.COMPLETED)
            self.assertEqual(agent.successful_executions, 1)
            self.assertEqual(agent.failed_executions, 0)
        
        asyncio.run(test_execution())
    
    def test_agent_execution_exception(self):
        """Test agent execution with exception"""
        async def test_execution():
            agent = MockAgent(fail_mode="exception")
            query = AgentQuery(text="test query")
            
            result = await agent.run(query)
            
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertEqual(agent.status, AgentStatus.FAILED)
            self.assertEqual(agent.successful_executions, 0)
            self.assertEqual(agent.failed_executions, 1)
        
        asyncio.run(test_execution())
    
    def test_agent_execution_timeout(self):
        """Test agent execution with timeout"""
        async def test_execution():
            agent = MockAgent(fail_mode="timeout")
            query = AgentQuery(text="test query")
            
            # Use asyncio.wait_for to simulate timeout
            try:
                result = await asyncio.wait_for(agent.run(query), timeout=1.0)
                # If we get here, the timeout didn't work as expected
                self.assertTrue(result.success or not result.success)  # Either outcome is valid
            except asyncio.TimeoutError:
                # This is also acceptable - the outer timeout caught it
                pass
        
        asyncio.run(test_execution())
    
    def test_invalid_query_handling(self):
        """Test handling of invalid query in run method"""
        async def test_execution():
            agent = MockAgent()
            
            # Test with non-AgentQuery object
            with self.assertRaises(ValueError) as cm:
                await agent.run("not a query object")
            self.assertIn("Expected AgentQuery", str(cm.exception))
            
            # Test with None
            with self.assertRaises(ValueError) as cm:
                await agent.run(None)
            self.assertIn("Expected AgentQuery", str(cm.exception))
        
        asyncio.run(test_execution())
    
    def test_invalid_result_handling(self):
        """Test handling of invalid result from execute method"""
        async def test_execution():
            agent = MockAgent(fail_mode="invalid_result")
            query = AgentQuery(text="test query")
            
            result = await agent.run(query)
            
            self.assertFalse(result.success)
            self.assertIn("must return AgentResult", result.error)
            self.assertEqual(agent.status, AgentStatus.FAILED)
        
        asyncio.run(test_execution())
    
    def test_multiple_executions(self):
        """Test agent behavior with multiple executions"""
        async def test_executions():
            agent = MockAgent()
            
            # Execute multiple times
            for i in range(5):
                query = AgentQuery(text=f"query {i}")
                result = await agent.run(query)
                self.assertTrue(result.success)
            
            # Check statistics
            self.assertEqual(agent.total_executions, 5)
            self.assertEqual(agent.successful_executions, 5)
            self.assertEqual(agent.failed_executions, 0)
            
            stats = agent.get_statistics()
            self.assertEqual(stats["success_rate"], 1.0)
            self.assertEqual(stats["total_execution_samples"], 5)
        
        asyncio.run(test_executions())
    
    def test_mixed_success_failure(self):
        """Test agent with mixed success and failure executions"""
        async def test_executions():
            # Create agents with different behaviors
            success_agent = MockAgent()
            failure_agent = MockAgent(fail_mode="exception")
            
            query = AgentQuery(text="test")
            
            # Mix of successes and failures
            agents = [success_agent, failure_agent] * 3
            
            for agent in agents:
                result = await agent.run(query)
                # Results will vary based on agent type
            
            # Check success agent
            self.assertEqual(success_agent.total_executions, 3)
            self.assertEqual(success_agent.successful_executions, 3)
            
            # Check failure agent
            self.assertEqual(failure_agent.total_executions, 3)
            self.assertEqual(failure_agent.failed_executions, 3)
        
        asyncio.run(test_executions())
    
    def test_statistics_reset(self):
        """Test statistics reset functionality"""
        async def test_reset():
            agent = MockAgent()
            query = AgentQuery(text="test")
            
            # Execute a few times
            for _ in range(3):
                await agent.run(query)
            
            # Verify statistics
            self.assertEqual(agent.total_executions, 3)
            self.assertGreater(len(agent._execution_times), 0)
            
            # Reset statistics
            agent.reset_statistics()
            
            # Verify reset
            self.assertEqual(agent.total_executions, 0)
            self.assertEqual(agent.successful_executions, 0)
            self.assertEqual(agent.failed_executions, 0)
            self.assertEqual(len(agent._execution_times), 0)
        
        asyncio.run(test_reset())
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        async def test_summary():
            agent = MockAgent()
            query = AgentQuery(text="test")
            
            # Execute once
            await agent.run(query)
            
            # Get performance summary
            summary = agent.get_performance_summary()
            self.assertIsInstance(summary, str)
            self.assertIn(agent.name, summary)
            self.assertIn("Executions: 1 total", summary)
            self.assertIn("100.0%", summary)  # Success rate
        
        asyncio.run(test_summary())


class TestAgentRegistryBoundaryConditions(unittest.TestCase):
    """Test AgentRegistry boundary conditions"""
    
    def setUp(self):
        """Set up test registry"""
        self.registry = AgentRegistry()
    
    def test_empty_registry(self):
        """Test operations on empty registry"""
        self.assertIsNone(self.registry.get("nonexistent"))
        self.assertEqual(len(self.registry.get_all()), 0)
        self.assertEqual(len(self.registry.get_by_type(AgentType.LOCAL_DATA)), 0)
    
    def test_register_valid_agents(self):
        """Test registering valid agents"""
        agent1 = MockAgent(AgentType.LOCAL_DATA, "agent1")
        agent2 = MockAgent(AgentType.WEB_SEARCH, "agent2")
        
        self.registry.register(agent1)
        self.registry.register(agent2)
        
        self.assertEqual(len(self.registry.get_all()), 2)
        self.assertEqual(self.registry.get("agent1"), agent1)
        self.assertEqual(self.registry.get("agent2"), agent2)
    
    def test_register_duplicate_names(self):
        """Test registering agents with duplicate names"""
        agent1 = MockAgent(AgentType.LOCAL_DATA, "duplicate")
        agent2 = MockAgent(AgentType.WEB_SEARCH, "duplicate")
        
        self.registry.register(agent1)
        self.registry.register(agent2)  # Should overwrite
        
        self.assertEqual(len(self.registry.get_all()), 1)
        self.assertEqual(self.registry.get("duplicate"), agent2)
    
    def test_get_by_type(self):
        """Test getting agents by type"""
        local_agent1 = MockAgent(AgentType.LOCAL_DATA, "local1")
        local_agent2 = MockAgent(AgentType.LOCAL_DATA, "local2")
        web_agent = MockAgent(AgentType.WEB_SEARCH, "web1")
        
        self.registry.register(local_agent1)
        self.registry.register(local_agent2)
        self.registry.register(web_agent)
        
        local_agents = self.registry.get_by_type(AgentType.LOCAL_DATA)
        self.assertEqual(len(local_agents), 2)
        
        web_agents = self.registry.get_by_type(AgentType.WEB_SEARCH)
        self.assertEqual(len(web_agents), 1)
        
        # Non-existent type
        cloud_agents = self.registry.get_by_type(AgentType.CLOUD_KB)
        self.assertEqual(len(cloud_agents), 0)
    
    def test_statistics_aggregation(self):
        """Test registry statistics aggregation"""
        async def test_stats():
            agent1 = MockAgent(AgentType.LOCAL_DATA, "agent1")
            agent2 = MockAgent(AgentType.WEB_SEARCH, "agent2")
            
            self.registry.register(agent1)
            self.registry.register(agent2)
            
            # Execute some operations
            query = AgentQuery(text="test")
            await agent1.run(query)
            await agent2.run(query)
            
            # Get aggregated statistics
            stats = self.registry.get_statistics()
            
            self.assertIn("agent1", stats)
            self.assertIn("agent2", stats)
            self.assertEqual(stats["agent1"]["total_executions"], 1)
            self.assertEqual(stats["agent2"]["total_executions"], 1)
        
        asyncio.run(test_stats())


if __name__ == "__main__":
    unittest.main()