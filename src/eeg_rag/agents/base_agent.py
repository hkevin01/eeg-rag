"""
Base Agent Class for Agentic RAG System

This module provides the foundation for all specialized agents in the EEG-RAG system.
Implements robust error handling, time measurement, and logging as per requirements.

Requirements Covered:
- REQ-AGT-001: Base agent interface with standardized methods
- REQ-AGT-002: Error handling with graceful degradation
- REQ-AGT-003: Performance monitoring for all agent operations
- REQ-AGT-004: Logging with context tracking
- REQ-AGT-005: Async support for parallel execution
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from eeg_rag.utils.logging_utils import PerformanceTimer, log_exception, timed


# REQ-AGT-006: Define agent types for routing and identification
class AgentType(Enum):
    """Enumeration of agent types in the system"""
    ORCHESTRATOR = "orchestrator"
    LOCAL_DATA = "local_data"
    WEB_SEARCH = "web_search"
    CLOUD_KB = "cloud_kb"
    MCP_SERVER = "mcp_server"
    AGGREGATOR = "aggregator"


# REQ-AGT-007: Agent status tracking for monitoring
class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentResult:
    """
    Standardized result from agent execution
    
    REQ-AGT-008: All agent results must include:
    - Success status
    - Data payload
    - Metadata (sources, confidence, timing)
    - Error information (if applicable)
    """
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    agent_type: Optional[AgentType] = None
    elapsed_time: float = 0.0  # REQ-AGT-009: Time in seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging/serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "error": self.error,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "elapsed_time": self.elapsed_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentQuery:
    """
    Standardized query structure for agents
    
    REQ-AGT-010: Query must include:
    - Original text
    - Intent/purpose
    - Context (memory, conversation)
    - Configuration/parameters
    """
    text: str
    intent: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    query_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary"""
        return {
            "text": self.text,
            "intent": self.intent,
            "context": self.context,
            "parameters": self.parameters,
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat()
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the EEG-RAG system
    
    REQ-AGT-011: All agents must:
    - Implement execute() method
    - Support async execution
    - Provide error handling
    - Track performance metrics
    - Log all operations
    
    REQ-AGT-012: Time measurement standardization
    - All times in seconds (SECOND = 1.0)
    - Human-readable formatting available
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize base agent
        
        Args:
            agent_type: Type of agent (from AgentType enum)
            name: Human-readable agent name
            config: Configuration dictionary
            logger: Logger instance (creates new if None)
        
        REQ-AGT-013: All agents must be identifiable
        """
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_agent"
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"eeg_rag.agents.{self.name}")
        
        # REQ-AGT-014: Track agent state
        self.status = AgentStatus.IDLE
        self.last_execution_time: Optional[float] = None
        self.total_executions: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        
        self.logger.info(f"Initialized {self.name} (type: {agent_type.value})")
    
    @abstractmethod
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute agent-specific logic (must be implemented by subclasses)
        
        Args:
            query: Standardized query object
            
        Returns:
            AgentResult with data and metadata
            
        REQ-AGT-015: All agents must implement execute() method
        REQ-AGT-016: Execute must be async for parallel execution
        """
        pass
    
    @timed
    async def run(self, query: AgentQuery) -> AgentResult:
        """
        Main entry point for agent execution with error handling and timing
        
        Args:
            query: Standardized query object
            
        Returns:
            AgentResult with success/failure information
            
        REQ-AGT-017: Wrapper for execute() with:
        - Status tracking
        - Error handling
        - Performance measurement
        - Logging
        """
        self.logger.info(f"[{self.name}] Starting execution for query: {query.query_id}")
        self.status = AgentStatus.EXECUTING
        self.total_executions += 1
        
        start_time = datetime.now()
        result = None
        
        try:
            # REQ-AGT-018: Use PerformanceTimer for accurate timing
            with PerformanceTimer(f"{self.name}.execute", self.logger):
                result = await self.execute(query)
            
            # REQ-AGT-019: Track successful execution
            self.successful_executions += 1
            self.status = AgentStatus.COMPLETED
            self.logger.info(
                f"[{self.name}] Execution completed successfully "
                f"(time: {result.elapsed_time:.2f}s)"
            )
            
        except asyncio.TimeoutError as e:
            # REQ-AGT-020: Handle timeout errors
            self.status = AgentStatus.TIMEOUT
            self.failed_executions += 1
            error_msg = f"Agent execution timeout: {str(e)}"
            self.logger.error(f"[{self.name}] {error_msg}")
            result = AgentResult(
                success=False,
                data=None,
                error=error_msg,
                agent_type=self.agent_type
            )
            
        except Exception as e:
            # REQ-AGT-021: Handle all other errors with logging
            self.status = AgentStatus.FAILED
            self.failed_executions += 1
            error_msg = f"Agent execution failed: {str(e)}"
            self.logger.exception(f"[{self.name}] {error_msg}")
            result = AgentResult(
                success=False,
                data=None,
                error=error_msg,
                agent_type=self.agent_type
            )
        
        finally:
            # REQ-AGT-022: Always compute elapsed time
            elapsed = (datetime.now() - start_time).total_seconds()
            self.last_execution_time = elapsed
            
            if result:
                result.elapsed_time = elapsed
                result.agent_type = self.agent_type
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent performance statistics
        
        Returns:
            Dictionary with execution statistics
            
        REQ-AGT-023: Provide performance metrics for monitoring
        """
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        return {
            "agent_name": self.name,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "last_execution_time": self.last_execution_time
        }
    
    def reset_statistics(self) -> None:
        """
        Reset agent statistics
        
        REQ-AGT-024: Allow statistics reset for testing/monitoring
        """
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.last_execution_time = None
        self.logger.info(f"[{self.name}] Statistics reset")
    
    def __repr__(self) -> str:
        """String representation of agent"""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type={self.agent_type.value}, status={self.status.value})"
        )


# REQ-AGT-025: Agent registry for managing multiple agents
class AgentRegistry:
    """
    Registry for managing and accessing agents
    
    REQ-AGT-026: Centralized agent management
    """
    
    def __init__(self):
        """Initialize empty agent registry"""
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("eeg_rag.agents.registry")
    
    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent
        
        Args:
            agent: Agent instance to register
            
        REQ-AGT-027: Track all active agents
        """
        if agent.name in self._agents:
            self.logger.warning(
                f"Agent '{agent.name}' already registered, overwriting"
            )
        
        self._agents[agent.name] = agent
        self.logger.info(
            f"Registered agent: {agent.name} (type: {agent.agent_type.value})"
        )
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent by name
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(name)
    
    def get_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """
        Get all agents of a specific type
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents matching the type
            
        REQ-AGT-028: Query agents by type
        """
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]
    
    def get_all(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self._agents.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all agents
        
        Returns:
            Dictionary with aggregated statistics
            
        REQ-AGT-029: System-wide performance monitoring
        """
        return {
            agent.name: agent.get_statistics()
            for agent in self._agents.values()
        }


# REQ-AGT-030: Export public interface
__all__ = [
    "AgentType",
    "AgentStatus",
    "AgentResult",
    "AgentQuery",
    "BaseAgent",
    "AgentRegistry"
]
