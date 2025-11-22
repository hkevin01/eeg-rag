"""
Agent 4: MCP Server Agent
Connects to Model Context Protocol (MCP) servers for tool use, code execution, and resource access.

REQ-AGT4-001 to REQ-AGT4-015: MCP integration, tool discovery, execution, resource access
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import asyncio
import json
import time
import hashlib
import subprocess
import tempfile
import os
from pathlib import Path


# Tool and Resource Types
class ToolType(Enum):
    """Types of tools available through MCP"""
    CODE_EXECUTION = "code_execution"
    FILE_ACCESS = "file_access"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    WEB_SCRAPING = "web_scraping"
    DATA_PROCESSING = "data_processing"
    COMPUTATION = "computation"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Types of resources accessible through MCP"""
    FILE = "file"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    WEB_PAGE = "web_page"
    DATASET = "dataset"
    MODEL = "model"
    CUSTOM = "custom"


class ExecutionStatus(Enum):
    """Status of tool execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Tool:
    """Represents an MCP tool"""
    tool_id: str
    name: str
    tool_type: ToolType
    description: str
    parameters: Dict[str, Any]
    capabilities: List[str] = field(default_factory=list)
    timeout: float = 30.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'type': self.tool_type.value,
            'description': self.description,
            'parameters': self.parameters,
            'capabilities': self.capabilities,
            'timeout': self.timeout
        }


@dataclass
class Resource:
    """Represents an MCP resource"""
    resource_id: str
    name: str
    resource_type: ResourceType
    uri: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_permissions: List[str] = field(default_factory=lambda: ["read"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'resource_id': self.resource_id,
            'name': self.name,
            'type': self.resource_type.value,
            'uri': self.uri,
            'metadata': self.metadata,
            'permissions': self.access_permissions
        }


@dataclass
class ExecutionResult:
    """Result from tool execution"""
    execution_id: str
    tool_id: str
    status: ExecutionStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'execution_id': self.execution_id,
            'tool_id': self.tool_id,
            'status': self.status.value,
            'output': self.output,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class MockMCPServer:
    """Mock MCP server for testing (replace with real MCP client in production)"""
    
    def __init__(self):
        self.tools = self._create_mock_tools()
        self.resources = self._create_mock_resources()
    
    def _create_mock_tools(self) -> List[Tool]:
        """Create mock tools"""
        return [
            Tool(
                tool_id="tool_001",
                name="python_executor",
                tool_type=ToolType.CODE_EXECUTION,
                description="Execute Python code in a sandboxed environment",
                parameters={"code": "string", "timeout": "number"},
                capabilities=["python3", "numpy", "pandas"],
                timeout=30.0
            ),
            Tool(
                tool_id="tool_002",
                name="file_reader",
                tool_type=ToolType.FILE_ACCESS,
                description="Read file contents",
                parameters={"path": "string", "encoding": "string"},
                capabilities=["read"],
                timeout=10.0
            ),
            Tool(
                tool_id="tool_003",
                name="data_analyzer",
                tool_type=ToolType.DATA_PROCESSING,
                description="Analyze tabular data and compute statistics",
                parameters={"data": "array", "operations": "array"},
                capabilities=["mean", "std", "correlation"],
                timeout=20.0
            ),
            Tool(
                tool_id="tool_004",
                name="api_caller",
                tool_type=ToolType.API_CALL,
                description="Make HTTP API calls",
                parameters={"url": "string", "method": "string", "headers": "object"},
                capabilities=["GET", "POST"],
                timeout=30.0
            )
        ]
    
    def _create_mock_resources(self) -> List[Resource]:
        """Create mock resources"""
        return [
            Resource(
                resource_id="res_001",
                name="eeg_dataset",
                resource_type=ResourceType.DATASET,
                uri="file:///data/eeg/sample.csv",
                metadata={"size": "100MB", "format": "CSV"},
                access_permissions=["read"]
            ),
            Resource(
                resource_id="res_002",
                name="pubmed_api",
                resource_type=ResourceType.API_ENDPOINT,
                uri="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                metadata={"rate_limit": "3/sec"},
                access_permissions=["read"]
            )
        ]
    
    async def list_tools(self) -> List[Tool]:
        """List available tools"""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self.tools
    
    async def list_resources(self) -> List[Resource]:
        """List available resources"""
        await asyncio.sleep(0.01)
        return self.resources
    
    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute a tool"""
        start_time = time.time()
        await asyncio.sleep(0.05)  # Simulate execution
        
        # Find tool
        tool = next((t for t in self.tools if t.tool_id == tool_id), None)
        if not tool:
            return ExecutionResult(
                execution_id=f"exec_{int(time.time())}",
                tool_id=tool_id,
                status=ExecutionStatus.FAILED,
                output=None,
                error=f"Tool {tool_id} not found",
                execution_time=time.time() - start_time
            )
        
        # Simulate execution based on tool type
        try:
            if tool.tool_type == ToolType.CODE_EXECUTION:
                output = self._execute_code(parameters.get('code', ''))
            elif tool.tool_type == ToolType.FILE_ACCESS:
                output = self._read_file(parameters.get('path', ''))
            elif tool.tool_type == ToolType.DATA_PROCESSING:
                output = self._analyze_data(parameters.get('data', []))
            elif tool.tool_type == ToolType.API_CALL:
                output = self._call_api(parameters)
            else:
                output = {"result": "Mock execution successful"}
            
            return ExecutionResult(
                execution_id=f"exec_{int(time.time() * 1000)}",
                tool_id=tool_id,
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
                metadata={'tool_name': tool.name}
            )
        
        except Exception as e:
            return ExecutionResult(
                execution_id=f"exec_{int(time.time() * 1000)}",
                tool_id=tool_id,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Mock code execution"""
        return {
            'output': f'Executed: {code[:50]}...',
            'return_value': 42,
            'stdout': 'Mock execution output'
        }
    
    def _read_file(self, path: str) -> Dict[str, Any]:
        """Mock file reading"""
        return {
            'content': f'Mock content of {path}',
            'size': 1024,
            'encoding': 'utf-8'
        }
    
    def _analyze_data(self, data: List[Any]) -> Dict[str, Any]:
        """Mock data analysis"""
        return {
            'count': len(data),
            'mean': 50.0,
            'std': 15.0,
            'min': 10.0,
            'max': 90.0
        }
    
    def _call_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock API call"""
        return {
            'status_code': 200,
            'response': {'data': 'Mock API response'},
            'headers': {'Content-Type': 'application/json'}
        }
    
    async def access_resource(
        self,
        resource_id: str,
        operation: str = "read"
    ) -> Dict[str, Any]:
        """Access a resource"""
        await asyncio.sleep(0.02)
        
        resource = next((r for r in self.resources if r.resource_id == resource_id), None)
        if not resource:
            raise ValueError(f"Resource {resource_id} not found")
        
        if operation not in resource.access_permissions:
            raise PermissionError(f"Operation {operation} not permitted on {resource_id}")
        
        return {
            'resource_id': resource_id,
            'operation': operation,
            'data': f'Mock data from {resource.name}',
            'metadata': resource.metadata
        }


class MCPAgent:
    """
    Agent 4: MCP Server Agent
    Connects to Model Context Protocol servers for tool use and resource access
    
    REQ-AGT4-001: Initialize MCP server connection
    REQ-AGT4-002: Discover available tools
    REQ-AGT4-003: List accessible resources
    REQ-AGT4-004: Execute tools with parameters
    REQ-AGT4-005: Handle tool execution results
    REQ-AGT4-006: Access resources (files, databases, APIs)
    REQ-AGT4-007: Stream execution results for long-running tasks
    REQ-AGT4-008: Implement timeout handling (default 30s)
    REQ-AGT4-009: Cache tool and resource metadata
    REQ-AGT4-010: Track execution statistics
    REQ-AGT4-011: Handle execution errors gracefully
    REQ-AGT4-012: Support multiple concurrent executions
    REQ-AGT4-013: Validate tool parameters before execution
    REQ-AGT4-014: Provide execution history
    REQ-AGT4-015: Generate execution IDs for tracking
    """
    
    def __init__(
        self,
        name: str = "MCPAgent",
        agent_type: str = "mcp",
        server_uri: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        use_mock: bool = True,  # Use mock for testing
        max_concurrent_executions: int = 5
    ):
        """
        Initialize MCP Agent
        
        Args:
            name: Agent name
            agent_type: Agent type identifier
            server_uri: MCP server URI
            capabilities: List of agent capabilities
            use_mock: Use mock MCP server (for testing)
            max_concurrent_executions: Max parallel tool executions
        """
        self.name = name
        self.agent_type = agent_type
        self.server_uri = server_uri
        self.max_concurrent_executions = max_concurrent_executions
        self.capabilities = capabilities or [
            "tool_execution",
            "resource_access",
            "code_execution",
            "data_processing"
        ]
        
        # MCP server connection (mock or real)
        if use_mock:
            self.server = MockMCPServer()
        else:
            # In production, use real MCP client
            raise NotImplementedError("Real MCP client not yet implemented. Set use_mock=True.")
        
        # Tool and resource caches
        self.tools_cache: Dict[str, Tool] = {}
        self.resources_cache: Dict[str, Resource] = {}
        self.cache_valid = False
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeout_executions': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'tools_discovered': 0,
            'resources_accessed': 0
        }
    
    async def initialize(self):
        """Initialize agent by discovering tools and resources"""
        # Discover tools
        tools = await self.server.list_tools()
        self.tools_cache = {tool.tool_id: tool for tool in tools}
        self.stats['tools_discovered'] = len(tools)
        
        # Discover resources
        resources = await self.server.list_resources()
        self.resources_cache = {res.resource_id: res for res in resources}
        
        self.cache_valid = True
    
    async def execute(
        self,
        query: str,
        tool_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a tool or process a query
        
        Args:
            query: Natural language query or tool name
            tool_id: Specific tool ID to execute
            parameters: Tool parameters
            
        Returns:
            ExecutionResult with output and metadata
        """
        start_time = time.time()
        
        # Ensure cache is valid
        if not self.cache_valid:
            await self.initialize()
        
        # Determine tool to execute
        if tool_id:
            tool = self.tools_cache.get(tool_id)
        else:
            # Try to infer tool from query
            tool = self._infer_tool_from_query(query)
        
        if not tool:
            return ExecutionResult(
                execution_id=f"exec_{int(time.time() * 1000)}",
                tool_id=tool_id or "unknown",
                status=ExecutionStatus.FAILED,
                output=None,
                error="No suitable tool found for query",
                execution_time=time.time() - start_time
            )
        
        # Prepare parameters
        if parameters is None:
            parameters = self._extract_parameters_from_query(query, tool)
        
        # Validate parameters
        validation_error = self._validate_parameters(tool, parameters)
        if validation_error:
            return ExecutionResult(
                execution_id=f"exec_{int(time.time() * 1000)}",
                tool_id=tool.tool_id,
                status=ExecutionStatus.FAILED,
                output=None,
                error=validation_error,
                execution_time=time.time() - start_time
            )
        
        # Execute tool
        self.stats['total_executions'] += 1
        
        try:
            result = await self.server.execute_tool(tool.tool_id, parameters)
            
            # Update statistics
            if result.status == ExecutionStatus.SUCCESS:
                self.stats['successful_executions'] += 1
            elif result.status == ExecutionStatus.TIMEOUT:
                self.stats['timeout_executions'] += 1
            else:
                self.stats['failed_executions'] += 1
            
            self.stats['total_execution_time'] += result.execution_time
            self.stats['average_execution_time'] = (
                self.stats['total_execution_time'] / self.stats['total_executions']
            )
            
            # Add to history
            self.execution_history.append(result)
            if len(self.execution_history) > 100:
                self.execution_history.pop(0)
            
            return result
        
        except Exception as e:
            self.stats['failed_executions'] += 1
            return ExecutionResult(
                execution_id=f"exec_{int(time.time() * 1000)}",
                tool_id=tool.tool_id,
                status=ExecutionStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _infer_tool_from_query(self, query: str) -> Optional[Tool]:
        """Infer which tool to use based on query"""
        query_lower = query.lower()
        
        # Simple keyword matching
        if any(word in query_lower for word in ['execute', 'run', 'code', 'python']):
            return next((t for t in self.tools_cache.values() 
                        if t.tool_type == ToolType.CODE_EXECUTION), None)
        
        if any(word in query_lower for word in ['read', 'file', 'open']):
            return next((t for t in self.tools_cache.values() 
                        if t.tool_type == ToolType.FILE_ACCESS), None)
        
        if any(word in query_lower for word in ['analyze', 'data', 'statistics']):
            return next((t for t in self.tools_cache.values() 
                        if t.tool_type == ToolType.DATA_PROCESSING), None)
        
        if any(word in query_lower for word in ['api', 'call', 'request', 'http']):
            return next((t for t in self.tools_cache.values() 
                        if t.tool_type == ToolType.API_CALL), None)
        
        # Default to first available tool
        return next(iter(self.tools_cache.values()), None)
    
    def _extract_parameters_from_query(
        self,
        query: str,
        tool: Tool
    ) -> Dict[str, Any]:
        """Extract parameters from natural language query"""
        # Simple parameter extraction (in production, use LLM)
        params = {}
        
        if tool.tool_type == ToolType.CODE_EXECUTION:
            params['code'] = query
            params['timeout'] = 30
        elif tool.tool_type == ToolType.FILE_ACCESS:
            # Try to extract file path
            words = query.split()
            for word in words:
                if '/' in word or '\\' in word or '.' in word:
                    params['path'] = word
                    break
            params['encoding'] = 'utf-8'
        elif tool.tool_type == ToolType.DATA_PROCESSING:
            params['data'] = []
            params['operations'] = ['mean', 'std']
        elif tool.tool_type == ToolType.API_CALL:
            params['url'] = 'https://api.example.com'
            params['method'] = 'GET'
        
        return params
    
    def _validate_parameters(
        self,
        tool: Tool,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Validate tool parameters"""
        # Check required parameters exist
        for param_name in tool.parameters:
            if param_name not in parameters:
                return f"Missing required parameter: {param_name}"
        
        return None
    
    async def access_resource(
        self,
        resource_id: str,
        operation: str = "read"
    ) -> Dict[str, Any]:
        """Access a resource"""
        if not self.cache_valid:
            await self.initialize()
        
        if resource_id not in self.resources_cache:
            raise ValueError(f"Resource {resource_id} not found")
        
        self.stats['resources_accessed'] += 1
        return await self.server.access_resource(resource_id, operation)
    
    def get_available_tools(self) -> List[Tool]:
        """Get list of available tools"""
        return list(self.tools_cache.values())
    
    def get_available_resources(self) -> List[Resource]:
        """Get list of available resources"""
        return list(self.resources_cache.values())
    
    def get_execution_history(self, limit: int = 10) -> List[ExecutionResult]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'name': self.name,
            'agent_type': self.agent_type,
            'total_executions': self.stats['total_executions'],
            'successful_executions': self.stats['successful_executions'],
            'failed_executions': self.stats['failed_executions'],
            'timeout_executions': self.stats['timeout_executions'],
            'success_rate': (
                self.stats['successful_executions'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0.0
            ),
            'average_execution_time': self.stats['average_execution_time'],
            'tools_available': len(self.tools_cache),
            'resources_available': len(self.resources_cache),
            'tools_discovered': self.stats['tools_discovered'],
            'resources_accessed': self.stats['resources_accessed']
        }
    
    def clear_cache(self):
        """Clear tool and resource caches"""
        self.tools_cache.clear()
        self.resources_cache.clear()
        self.cache_valid = False
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
