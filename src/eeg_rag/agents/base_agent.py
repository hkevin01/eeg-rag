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

Enhancements:
- REQ-AGT-031: Enhanced validation using common utilities
- REQ-AGT-032: Improved error handling and resilience
- REQ-AGT-033: Standardized time measurements in seconds
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from eeg_rag.utils.logging_utils import PerformanceTimer, log_exception, timed
from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    format_error_message,
    SECOND
)


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
    
    REQ-AGT-031: Enhanced validation for all fields
    """
    text: str
    intent: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    query_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate query fields after initialization"""
        # REQ-AGT-031: Validate query text
        self.text = validate_non_empty_string(self.text, "query text")
        
        # Validate optional fields
        if self.intent is not None:
            self.intent = validate_non_empty_string(self.intent, "intent", allow_none=True)
        
        # Ensure context and parameters are dictionaries
        if self.context is None:
            self.context = {}
        elif not isinstance(self.context, dict):
            raise ValueError(f"context must be a dictionary, got {type(self.context).__name__}")
        
        if self.parameters is None:
            self.parameters = {}
        elif not isinstance(self.parameters, dict):
            raise ValueError(f"parameters must be a dictionary, got {type(self.parameters).__name__}")
        
        # Generate query_id if not provided
        if not self.query_id:
            from eeg_rag.utils.common_utils import compute_content_hash
            self.query_id = compute_content_hash(
                f"{self.text}{self.timestamp.isoformat()}",
                prefix="query"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary with validation"""
        try:
            return {
                "text": self.text,
                "intent": self.intent,
                "context": self.context,
                "parameters": self.parameters,
                "query_id": self.query_id,
                "timestamp": self.timestamp.isoformat()
            }
        except Exception as e:
            raise ValueError(f"Failed to serialize AgentQuery: {str(e)}") from e
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """
        Safely get value from context dictionary
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        return self.context.get(key, default)
    
    def get_parameter_value(self, key: str, default: Any = None) -> Any:
        """
        Safely get value from parameters dictionary
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        return self.parameters.get(key, default)


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
    
    REQ-AGT-031: Enhanced validation and error handling
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
            
        Raises:
            ValueError: If agent_type is invalid or name is invalid
        
        REQ-AGT-013: All agents must be identifiable
        REQ-AGT-031: Input validation
        """
        # REQ-AGT-031: Validate agent_type
        if not isinstance(agent_type, AgentType):
            raise ValueError(f"agent_type must be an AgentType enum, got {type(agent_type).__name__}")
        
        self.agent_type = agent_type
        
        # REQ-AGT-031: Validate and set name
        if name is not None:
            self.name = validate_non_empty_string(name, "agent name")
        else:
            self.name = f"{agent_type.value}_agent"
        
        # Validate config
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config.copy()  # Defensive copy
        else:
            raise ValueError(f"config must be a dictionary, got {type(config).__name__}")
        
        # Set up logger
        self.logger = logger or logging.getLogger(f"eeg_rag.agents.{self.name}")
        
        # REQ-AGT-014: Track agent state
        self.status = AgentStatus.IDLE
        self.last_execution_time: Optional[float] = None
        self.total_executions: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        
        # REQ-AGT-033: Track timing statistics in seconds
        self._execution_times: List[float] = []
        self._max_execution_time: float = 0.0
        self._min_execution_time: float = float('inf')
        
        self.logger.info(
            f"Initialized {self.name} (type: {agent_type.value})"
        )
    
    @abstractmethod
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute agent-specific logic (must be implemented by subclasses)
        
        Args:
            query: Standardized query object (pre-validated)
            
        Returns:
            AgentResult with data and metadata
            
        Raises:
            NotImplementedError: If not implemented by subclass
            
        REQ-AGT-015: All agents must implement execute() method
        REQ-AGT-016: Execute must be async for parallel execution
        REQ-AGT-031: Input validation handled at BaseAgent level
        """
        pass
    
    async def run(self, query: AgentQuery) -> AgentResult:
        """
        Main entry point for agent execution with error handling and timing
        
        Args:
            query: Standardized query object
            
        Returns:
            AgentResult with success/failure information
            
        Raises:
            ValueError: If query is invalid
            
        REQ-AGT-017: Wrapper for execute() with:
        - Status tracking
        - Error handling
        - Performance measurement
        - Logging
        
        REQ-AGT-031: Input validation
        REQ-AGT-032: Enhanced error handling
        """
        # REQ-AGT-031: Validate input
        if not isinstance(query, AgentQuery):
            raise ValueError(f"Expected AgentQuery, got {type(query).__name__}")
        
        execution_context = {
            "agent_name": self.name,
            "agent_type": self.agent_type.value,
            "query_id": query.query_id,
            "query_text": query.text[:50] + "..." if len(query.text) > 50 else query.text
        }
        
        self.logger.info(f"[{self.name}] Starting execution for query: {query.query_id}")
        self.status = AgentStatus.EXECUTING
        self.total_executions += 1
        
        start_time = datetime.now()
        result = None
        
        try:
            # REQ-AGT-018: Use PerformanceTimer for accurate timing
            with PerformanceTimer(f"{self.name}.execute", self.logger):
                result = await self.execute(query)
            
            # REQ-AGT-031: Validate result
            if not isinstance(result, AgentResult):
                raise ValueError(
                    f"Agent execute() must return AgentResult, got {type(result).__name__}"
                )
            
            # REQ-AGT-019: Track successful execution
            self.successful_executions += 1
            self.status = AgentStatus.COMPLETED
            
            # REQ-AGT-033: Track execution time statistics
            elapsed = result.elapsed_time
            self._execution_times.append(elapsed)
            self._max_execution_time = max(self._max_execution_time, elapsed)
            self._min_execution_time = min(self._min_execution_time, elapsed)
            
            self.logger.info(
                f"[{self.name}] Execution completed successfully "
                f"(time: {result.elapsed_time:.3f}s)"
            )
            
        except asyncio.TimeoutError as e:
            # REQ-AGT-020: Handle timeout errors
            self.status = AgentStatus.TIMEOUT
            self.failed_executions += 1
            error_msg = f"Agent execution timeout after {(datetime.now() - start_time).total_seconds():.1f}s: {str(e)}"
            
            formatted_error = format_error_message(
                f"{self.name} execution",
                e,
                execution_context
            )
            self.logger.error(formatted_error)
            
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
            
            formatted_error = format_error_message(
                f"{self.name} execution",
                e,
                execution_context
            )
            self.logger.exception(formatted_error)
            
            result = AgentResult(
                success=False,
                data=None,
                error=str(e),
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
        Get comprehensive agent performance statistics
        
        Returns:
            Dictionary with execution statistics
            
        REQ-AGT-023: Provide performance metrics for monitoring
        REQ-AGT-033: Include detailed timing statistics
        """
        from eeg_rag.utils.common_utils import safe_divide
        
        # Basic success rate
        success_rate = safe_divide(
            self.successful_executions, 
            self.total_executions,
            default=0.0
        )
        
        failure_rate = safe_divide(
            self.failed_executions,
            self.total_executions, 
            default=0.0
        )
        
        # Timing statistics
        avg_execution_time = safe_divide(
            sum(self._execution_times),
            len(self._execution_times),
            default=0.0
        )
        
        # Median execution time
        median_time = 0.0
        if self._execution_times:
            sorted_times = sorted(self._execution_times)
            n = len(sorted_times)
            if n % 2 == 0:
                median_time = (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
            else:
                median_time = sorted_times[n//2]
        
        base_stats = {
            "agent_name": self.name,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
        }
        
        # Add timing statistics
        timing_stats = {
            "last_execution_time_seconds": self.last_execution_time,
            "average_execution_time_seconds": avg_execution_time,
            "median_execution_time_seconds": median_time,
            "min_execution_time_seconds": self._min_execution_time if self._min_execution_time != float('inf') else 0.0,
            "max_execution_time_seconds": self._max_execution_time,
            "total_execution_samples": len(self._execution_times)
        }
        
        base_stats.update(timing_stats)
        return base_stats
    
    def get_performance_summary(self) -> str:
        """
        Get human-readable performance summary
        
        Returns:
            Formatted performance summary string
            
        REQ-AGT-033: Human-readable performance reporting
        """
        stats = self.get_statistics()
        
        from eeg_rag.utils.common_utils import format_duration_human_readable
        
        summary_parts = [
            f"Agent: {stats['agent_name']} ({stats['agent_type']})",
            f"Status: {stats['status']}",
            f"Executions: {stats['total_executions']} total, {stats['successful_executions']} successful",
            f"Success Rate: {stats['success_rate']:.1%}"
        ]
        
        if stats['average_execution_time_seconds'] > 0:
            avg_time = format_duration_human_readable(stats['average_execution_time_seconds'])
            summary_parts.append(f"Avg Time: {avg_time}")
        
        return " | ".join(summary_parts)
    
    def reset_statistics(self) -> None:
        """
        Reset agent statistics including timing data
        
        REQ-AGT-024: Allow statistics reset for testing/monitoring
        REQ-AGT-033: Reset timing statistics
        """
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.last_execution_time = None
        
        # Reset timing statistics
        self._execution_times.clear()
        self._max_execution_time = 0.0
        self._min_execution_time = float('inf')
        
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
