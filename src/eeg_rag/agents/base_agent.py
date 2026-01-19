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
import time
import random
from collections import defaultdict, deque

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
    PAUSED = "paused"
    RECOVERING = "recovering"
    CIRCUIT_OPEN = "circuit_open"
    THROTTLED = "throttled"


@dataclass
class AgentResult:
    """
    Standardized result from agent execution
    
    REQ-AGT-008: All agent results must include:
    - Success status
    - Data payload
    - Metadata (sources, confidence, timing)
    - Error information (if applicable)
    - Performance metrics
    - Coordination data
    """
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    agent_type: Optional[AgentType] = None
    elapsed_time: float = 0.0  # REQ-AGT-009: Time in seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Enhanced performance metrics
    cpu_time: float = 0.0
    memory_peak: float = 0.0  # MB
    network_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Coordination metrics
    retry_count: int = 0
    circuit_breaker_triggered: bool = False
    load_balancer_node: Optional[str] = None
    confidence_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging/serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "metadata": self.metadata,
            "error": self.error,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "elapsed_time": self.elapsed_time,
            "timestamp": self.timestamp.isoformat(),
            "cpu_time": self.cpu_time,
            "memory_peak": self.memory_peak,
            "network_calls": self.network_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retry_count": self.retry_count,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "load_balancer_node": self.load_balancer_node,
            "confidence_score": self.confidence_score
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
    "AgentRegistry",
    "CircuitBreaker",
    "LoadBalancer",
    "RetryManager",
    "AgentCoordinator"
]


# Advanced Coordination Infrastructure
class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern for agent resilience
    
    REQ-AGT-034: Prevent cascade failures across agents
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        test_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.test_requests = test_requests
        
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.test_request_count = 0
        
        self.logger = logging.getLogger("eeg_rag.agents.circuit_breaker")
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if (self.metrics.last_failure_time and
                (datetime.now() - self.metrics.last_failure_time).total_seconds() > self.timeout_seconds):
                self.state = CircuitBreakerState.HALF_OPEN
                self.test_request_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return self.test_request_count < self.test_requests
        
        return False
    
    def record_success(self) -> None:
        """Record successful execution"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.consecutive_failures = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.test_request_count += 1
            if self.test_request_count >= self.test_requests:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info("Circuit breaker reset to CLOSED")
    
    def record_failure(self) -> None:
        """Record failed execution"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker reopened due to test failure")
        elif (self.state == CircuitBreakerState.CLOSED and
              self.metrics.consecutive_failures >= self.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker opened after {self.metrics.consecutive_failures} failures")


@dataclass
class LoadBalancerNode:
    """Node in load balancer"""
    agent: BaseAgent
    weight: float = 1.0
    active_requests: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    health_score: float = 1.0  # 0.0 to 1.0


class LoadBalancer:
    """
    Intelligent load balancer for agent instances
    
    REQ-AGT-035: Distribute load across agent replicas
    """
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.nodes: Dict[str, LoadBalancerNode] = {}
        self.current_index = 0
        self.logger = logging.getLogger("eeg_rag.agents.load_balancer")
    
    def add_node(self, agent: BaseAgent, weight: float = 1.0) -> None:
        """Add agent to load balancer"""
        node = LoadBalancerNode(agent=agent, weight=weight)
        self.nodes[agent.name] = node
        self.logger.info(f"Added node {agent.name} with weight {weight}")
    
    def remove_node(self, agent_name: str) -> None:
        """Remove agent from load balancer"""
        if agent_name in self.nodes:
            del self.nodes[agent_name]
            self.logger.info(f"Removed node {agent_name}")
    
    def select_agent(self) -> Optional[BaseAgent]:
        """Select agent based on load balancing strategy"""
        if not self.nodes:
            return None
        
        healthy_nodes = [
            node for node in self.nodes.values()
            if node.health_score > 0.1  # Minimum health threshold
        ]
        
        if not healthy_nodes:
            self.logger.warning("No healthy nodes available")
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(healthy_nodes)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(healthy_nodes)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(healthy_nodes)
        elif self.strategy == "response_time":
            return self._response_time_selection(healthy_nodes)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_nodes)
    
    def _round_robin_selection(self, nodes: List[LoadBalancerNode]) -> BaseAgent:
        """Simple round robin selection"""
        if self.current_index >= len(nodes):
            self.current_index = 0
        
        selected = nodes[self.current_index]
        self.current_index += 1
        return selected.agent
    
    def _weighted_round_robin_selection(self, nodes: List[LoadBalancerNode]) -> BaseAgent:
        """Weighted round robin based on health and capacity"""
        total_weight = sum(node.weight * node.health_score for node in nodes)
        if total_weight <= 0:
            return self._round_robin_selection(nodes)
        
        # Select based on weighted probability
        import random
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight * node.health_score
            if current_weight >= rand_val:
                return node.agent
        
        # Fallback to last node
        return nodes[-1].agent
    
    def _least_connections_selection(self, nodes: List[LoadBalancerNode]) -> BaseAgent:
        """Select node with least active connections"""
        return min(nodes, key=lambda n: n.active_requests).agent
    
    def _response_time_selection(self, nodes: List[LoadBalancerNode]) -> BaseAgent:
        """Select node with best response time"""
        return min(nodes, key=lambda n: n.avg_response_time or float('inf')).agent
    
    def update_node_stats(self, agent_name: str, response_time: float, success: bool) -> None:
        """Update node statistics after request"""
        if agent_name not in self.nodes:
            return
        
        node = self.nodes[agent_name]
        node.total_requests += 1
        
        # Update average response time with exponential moving average
        alpha = 0.1  # Smoothing factor
        if node.avg_response_time == 0:
            node.avg_response_time = response_time
        else:
            node.avg_response_time = (alpha * response_time + 
                                    (1 - alpha) * node.avg_response_time)
        
        # Update health score based on success rate and response time
        if success:
            node.health_score = min(1.0, node.health_score + 0.1)
        else:
            node.health_score = max(0.0, node.health_score - 0.2)


class RetryManager:
    """
    Intelligent retry management with exponential backoff
    
    REQ-AGT-036: Smart retry logic for failed operations
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        self.logger = logging.getLogger("eeg_rag.agents.retry_manager")
    
    async def execute_with_retry(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with intelligent retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Retrying operation (attempt {attempt + 1}) after {delay:.2f}s delay")
                    await asyncio.sleep(delay)
                
                result = await operation(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation failed on attempt {attempt + 1}: {str(e)}")
                
                if attempt >= self.max_retries:
                    break
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    self.logger.info(f"Non-retryable error: {str(e)}")
                    break
        
        # All retries exhausted
        self.logger.error(f"Operation failed after {self.max_retries + 1} attempts")
        if last_exception:
            raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 25% jitter to prevent thundering herd
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if error is retryable"""
        retryable_types = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        
        if isinstance(error, retryable_types):
            return True
        
        # Check for specific error messages
        error_str = str(error).lower()
        retryable_messages = [
            "connection reset",
            "connection timeout",
            "temporary failure",
            "service unavailable",
            "too many requests"
        ]
        
        return any(msg in error_str for msg in retryable_messages)


class AgentCoordinator:
    """
    Advanced agent coordination with circuit breakers, load balancing, and retry logic
    
    REQ-AGT-037: Comprehensive agent orchestration
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.retry_manager = RetryManager()
        
        self.logger = logging.getLogger("eeg_rag.agents.coordinator")
    
    def add_circuit_breaker(self, agent_name: str, **kwargs) -> None:
        """Add circuit breaker for agent"""
        self.circuit_breakers[agent_name] = CircuitBreaker(**kwargs)
    
    def add_load_balancer(self, agent_type: str, strategy: str = "weighted_round_robin") -> None:
        """Add load balancer for agent type"""
        self.load_balancers[agent_type] = LoadBalancer(strategy=strategy)
    
    async def execute_with_coordination(
        self,
        agent: BaseAgent,
        query: AgentQuery,
        use_circuit_breaker: bool = True,
        use_retry: bool = True
    ) -> AgentResult:
        """Execute agent with full coordination features"""
        
        # Check circuit breaker
        if use_circuit_breaker and agent.name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[agent.name]
            if not circuit_breaker.can_execute():
                return AgentResult(
                    success=False,
                    data=None,
                    error="Circuit breaker is open",
                    agent_type=agent.agent_type,
                    circuit_breaker_triggered=True
                )
        
        # Execute with retry if enabled
        try:
            if use_retry:
                result = await self.retry_manager.execute_with_retry(
                    agent.execute_with_monitoring,
                    query
                )
            else:
                result = await agent.execute_with_monitoring(query)
            
            # Record success in circuit breaker
            if use_circuit_breaker and agent.name in self.circuit_breakers:
                self.circuit_breakers[agent.name].record_success()
            
            return result
            
        except Exception as e:
            # Record failure in circuit breaker
            if use_circuit_breaker and agent.name in self.circuit_breakers:
                self.circuit_breakers[agent.name].record_failure()
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e),
                agent_type=agent.agent_type
            )
