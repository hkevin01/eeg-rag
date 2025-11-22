"""
Test Suite for Agent 4: MCP Server Agent
Tests Model Context Protocol integration, tool execution, and resource access
"""

import pytest
import asyncio
from eeg_rag.agents.mcp_agent import (
    MCPAgent,
    Tool,
    Resource,
    ExecutionResult,
    ToolType,
    ResourceType,
    ExecutionStatus,
    MockMCPServer
)


class TestTool:
    """Tests for Tool dataclass"""
    
    def test_tool_creation(self):
        """Test creating a tool"""
        tool = Tool(
            tool_id="test_001",
            name="test_tool",
            tool_type=ToolType.CODE_EXECUTION,
            description="A test tool",
            parameters={"code": "string"},
            capabilities=["python3"],
            timeout=30.0
        )
        
        assert tool.tool_id == "test_001"
        assert tool.name == "test_tool"
        assert tool.tool_type == ToolType.CODE_EXECUTION
        assert "python3" in tool.capabilities
    
    def test_tool_to_dict(self):
        """Test tool serialization"""
        tool = Tool(
            tool_id="test_001",
            name="test_tool",
            tool_type=ToolType.FILE_ACCESS,
            description="File access tool",
            parameters={"path": "string"}
        )
        
        result = tool.to_dict()
        assert result['tool_id'] == 'test_001'
        assert result['type'] == 'file_access'
        assert 'parameters' in result


class TestResource:
    """Tests for Resource dataclass"""
    
    def test_resource_creation(self):
        """Test creating a resource"""
        resource = Resource(
            resource_id="res_001",
            name="test_dataset",
            resource_type=ResourceType.DATASET,
            uri="file:///data/test.csv",
            metadata={"size": "10MB"},
            access_permissions=["read"]
        )
        
        assert resource.resource_id == "res_001"
        assert resource.resource_type == ResourceType.DATASET
        assert "read" in resource.access_permissions
    
    def test_resource_to_dict(self):
        """Test resource serialization"""
        resource = Resource(
            resource_id="res_001",
            name="api_endpoint",
            resource_type=ResourceType.API_ENDPOINT,
            uri="https://api.example.com"
        )
        
        result = resource.to_dict()
        assert result['resource_id'] == 'res_001'
        assert result['type'] == 'api_endpoint'
        assert result['uri'] == 'https://api.example.com'


class TestExecutionResult:
    """Tests for ExecutionResult dataclass"""
    
    def test_execution_result_success(self):
        """Test successful execution result"""
        result = ExecutionResult(
            execution_id="exec_001",
            tool_id="tool_001",
            status=ExecutionStatus.SUCCESS,
            output={"result": 42},
            execution_time=0.5
        )
        
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output['result'] == 42
        assert result.error is None
    
    def test_execution_result_failure(self):
        """Test failed execution result"""
        result = ExecutionResult(
            execution_id="exec_002",
            tool_id="tool_002",
            status=ExecutionStatus.FAILED,
            output=None,
            error="Execution failed",
            execution_time=0.1
        )
        
        assert result.status == ExecutionStatus.FAILED
        assert result.error == "Execution failed"
        assert result.output is None
    
    def test_execution_result_to_dict(self):
        """Test execution result serialization"""
        result = ExecutionResult(
            execution_id="exec_001",
            tool_id="tool_001",
            status=ExecutionStatus.SUCCESS,
            output={"value": 100}
        )
        
        result_dict = result.to_dict()
        assert result_dict['execution_id'] == 'exec_001'
        assert result_dict['status'] == 'success'
        assert result_dict['output']['value'] == 100


class TestMockMCPServer:
    """Tests for mock MCP server"""
    
    def test_server_initialization(self):
        """Test mock server has tools and resources"""
        server = MockMCPServer()
        
        assert len(server.tools) > 0
        assert len(server.resources) > 0
        assert any(t.tool_type == ToolType.CODE_EXECUTION for t in server.tools)
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing available tools"""
        server = MockMCPServer()
        tools = await server.list_tools()
        
        assert len(tools) >= 4
        assert all(isinstance(t, Tool) for t in tools)
    
    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing available resources"""
        server = MockMCPServer()
        resources = await server.list_resources()
        
        assert len(resources) >= 2
        assert all(isinstance(r, Resource) for r in resources)
    
    @pytest.mark.asyncio
    async def test_execute_code_tool(self):
        """Test executing code execution tool"""
        server = MockMCPServer()
        
        result = await server.execute_tool(
            "tool_001",
            {"code": "print('hello')", "timeout": 30}
        )
        
        assert result.status == ExecutionStatus.SUCCESS
        assert 'output' in result.output
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool"""
        server = MockMCPServer()
        
        result = await server.execute_tool("invalid_tool", {})
        
        assert result.status == ExecutionStatus.FAILED
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_access_resource(self):
        """Test accessing a resource"""
        server = MockMCPServer()
        
        result = await server.access_resource("res_001", "read")
        
        assert result['resource_id'] == 'res_001'
        assert result['operation'] == 'read'
        assert 'data' in result
    
    @pytest.mark.asyncio
    async def test_access_nonexistent_resource(self):
        """Test accessing non-existent resource"""
        server = MockMCPServer()
        
        with pytest.raises(ValueError):
            await server.access_resource("invalid_resource", "read")


class TestMCPAgent:
    """Tests for MCPAgent class"""
    
    def test_agent_initialization(self):
        """Test creating MCP agent"""
        agent = MCPAgent(
            name="TestMCPAgent",
            agent_type="mcp",
            use_mock=True
        )
        
        assert agent.name == "TestMCPAgent"
        assert agent.agent_type == "mcp"
        assert "tool_execution" in agent.capabilities
        assert agent.max_concurrent_executions == 5
    
    @pytest.mark.asyncio
    async def test_agent_initialization_discovers_tools(self):
        """Test agent discovers tools on initialization"""
        agent = MCPAgent(use_mock=True)
        
        await agent.initialize()
        
        assert len(agent.tools_cache) > 0
        assert len(agent.resources_cache) > 0
        assert agent.cache_valid is True
    
    @pytest.mark.asyncio
    async def test_execute_with_tool_id(self):
        """Test executing a specific tool by ID"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.execute(
            query="test",
            tool_id="tool_001",
            parameters={"code": "print('test')", "timeout": 30}
        )
        
        assert result.status == ExecutionStatus.SUCCESS
        assert result.tool_id == "tool_001"
    
    @pytest.mark.asyncio
    async def test_execute_infers_code_tool(self):
        """Test agent infers code execution tool from query"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.execute("Execute python code: print('hello')")
        
        assert result.status == ExecutionStatus.SUCCESS
        # Should have inferred code execution tool
    
    @pytest.mark.asyncio
    async def test_execute_infers_file_tool(self):
        """Test agent infers file access tool from query"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.execute("Read file /path/to/file.txt")
        
        assert result.status == ExecutionStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_execute_infers_data_tool(self):
        """Test agent infers data processing tool from query"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.execute("Analyze data and compute statistics")
        
        assert result.status == ExecutionStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_tool(self):
        """Test executing with invalid tool ID"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.execute(
            query="test",
            tool_id="invalid_tool",
            parameters={}
        )
        
        assert result.status == ExecutionStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_updates_statistics(self):
        """Test execution updates statistics"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        # Execute multiple times
        await agent.execute("Execute code", tool_id="tool_001", parameters={"code": "test", "timeout": 30})
        await agent.execute("Execute code", tool_id="tool_001", parameters={"code": "test2", "timeout": 30})
        
        stats = agent.get_statistics()
        assert stats['total_executions'] == 2
        assert stats['successful_executions'] >= 1
    
    @pytest.mark.asyncio
    async def test_execution_history(self):
        """Test execution history is maintained"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        await agent.execute("test query", tool_id="tool_001", parameters={"code": "test", "timeout": 30})
        
        history = agent.get_execution_history()
        assert len(history) > 0
        assert isinstance(history[0], ExecutionResult)
    
    @pytest.mark.asyncio
    async def test_execution_history_limit(self):
        """Test execution history respects limit"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        # Execute multiple times
        for i in range(15):
            await agent.execute(f"query {i}", tool_id="tool_001", parameters={"code": f"test{i}", "timeout": 30})
        
        history = agent.get_execution_history(limit=5)
        assert len(history) <= 5
    
    @pytest.mark.asyncio
    async def test_access_resource(self):
        """Test accessing a resource through agent"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        result = await agent.access_resource("res_001", "read")
        
        assert result['resource_id'] == 'res_001'
        assert agent.stats['resources_accessed'] == 1
    
    @pytest.mark.asyncio
    async def test_access_invalid_resource(self):
        """Test accessing invalid resource raises error"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        with pytest.raises(ValueError):
            await agent.access_resource("invalid_resource")
    
    def test_get_available_tools(self):
        """Test getting list of available tools"""
        agent = MCPAgent(use_mock=True)
        
        # Manually populate cache for this test
        agent.tools_cache = {
            "tool_001": Tool("tool_001", "test", ToolType.CODE_EXECUTION, "desc", {})
        }
        
        tools = agent.get_available_tools()
        assert len(tools) == 1
        assert tools[0].tool_id == "tool_001"
    
    def test_get_available_resources(self):
        """Test getting list of available resources"""
        agent = MCPAgent(use_mock=True)
        
        # Manually populate cache
        agent.resources_cache = {
            "res_001": Resource("res_001", "test", ResourceType.DATASET, "uri://test")
        }
        
        resources = agent.get_available_resources()
        assert len(resources) == 1
        assert resources[0].resource_id == "res_001"
    
    def test_statistics_tracking(self):
        """Test agent statistics"""
        agent = MCPAgent(use_mock=True)
        
        stats = agent.get_statistics()
        
        assert 'name' in stats
        assert 'total_executions' in stats
        assert 'success_rate' in stats
        assert 'average_execution_time' in stats
    
    def test_clear_cache(self):
        """Test clearing agent caches"""
        agent = MCPAgent(use_mock=True)
        agent.tools_cache = {"tool": "data"}
        agent.cache_valid = True
        
        agent.clear_cache()
        
        assert len(agent.tools_cache) == 0
        assert agent.cache_valid is False
    
    def test_clear_history(self):
        """Test clearing execution history"""
        agent = MCPAgent(use_mock=True)
        agent.execution_history = [ExecutionResult("id", "tool", ExecutionStatus.SUCCESS, {})]
        
        agent.clear_history()
        
        assert len(agent.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation before execution"""
        agent = MCPAgent(use_mock=True)
        await agent.initialize()
        
        # Try to execute without required parameters
        result = await agent.execute(
            query="test",
            tool_id="tool_001",
            parameters={}  # Missing required params
        )
        
        # Should fail validation
        assert result.status == ExecutionStatus.FAILED
        assert "parameter" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_auto_initialize_on_execute(self):
        """Test agent auto-initializes if cache invalid"""
        agent = MCPAgent(use_mock=True)
        
        # Cache should be invalid initially
        assert agent.cache_valid is False
        
        # Execute should trigger initialization
        result = await agent.execute("test query", tool_id="tool_001", parameters={"code": "test", "timeout": 30})
        
        # Cache should now be valid
        assert agent.cache_valid is True
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_limit(self):
        """Test max concurrent executions setting"""
        agent = MCPAgent(use_mock=True, max_concurrent_executions=3)
        
        assert agent.max_concurrent_executions == 3
