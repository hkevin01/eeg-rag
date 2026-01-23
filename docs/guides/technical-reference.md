# EEG-RAG Technical Reference

> **Comprehensive technical documentation for system architecture, resilience features, and development guidelines**

This document provides detailed technical information for developers, system administrators, and advanced users working with the EEG-RAG system.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [System Health & Monitoring](#system-health--monitoring)
3. [Resilience Features](#resilience-features)
4. [Performance Characteristics](#performance-characteristics)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Development Guidelines](#development-guidelines)
7. [Testing Strategy](#testing-strategy)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Architecture

### Core Components

#### BaseAgent Class
- **Location**: `src/eeg_rag/agents/base_agent.py`
- **Purpose**: Abstract base class for all agents in the system
- **Key Features**:
  - Standardized execution interface
  - Comprehensive error handling
  - Performance metrics tracking
  - Status monitoring
  - Async support for parallel operations

```python
# Example agent implementation
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CUSTOM, "my_agent")
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        # Agent-specific logic here
        return AgentResult(
            success=True,
            data="processed_result",
            metadata={"source": "custom_processing"}
        )
```

#### Common Utilities
- **Location**: `src/eeg_rag/utils/common_utils.py`
- **Purpose**: Centralized validation, error handling, and system utilities
- **Key Features**:
  - Input validation with consistent error messages
  - Time unit standardization (all measurements in seconds)
  - Retry mechanisms with exponential backoff
  - System health monitoring
  - Circuit breaker protection

#### Memory Management
- **Location**: `src/eeg_rag/memory/memory_manager.py`
- **Purpose**: Dual memory system for context retention
- **Features**:
  - Short-term memory for conversation state
  - Long-term memory for knowledge persistence
  - Configurable TTL and size limits
  - Statistics tracking

---

## System Health & Monitoring

### Health Monitoring System

The system includes comprehensive health monitoring capabilities:

```python
from eeg_rag.utils.common_utils import check_system_health

# Get current system health
health = check_system_health()
print(f"Status: {health.status.value}")
print(f"CPU: {health.cpu_percent:.1f}%")
print(f"Memory: {health.memory_percent:.1f}%")
print(f"Disk: {health.disk_percent:.1f}%")
```

#### Health Status Levels

| Status     | Description                           | Thresholds                             | Actions                              |
| ---------- | ------------------------------------- | -------------------------------------- | ------------------------------------ |
| `HEALTHY`  | All systems operating normally        | CPU <80%, Memory <85%, Disk <90%       | Continue normal operation            |
| `WARNING`  | Some resources under stress           | CPU 80-95%, Memory 85-95%, Disk 90-98% | Monitor closely, prepare for scaling |
| `CRITICAL` | System resources severely constrained | CPU >95%, Memory >95%, Disk >98%       | Immediate intervention required      |
| `UNKNOWN`  | Unable to collect metrics             | System monitoring failed               | Check system monitoring tools        |

#### Configurable Thresholds

```python
# Custom health check with different thresholds
health = check_system_health(
    cpu_warning_threshold=70.0,    # Warning at 70% CPU
    cpu_critical_threshold=90.0,   # Critical at 90% CPU
    memory_warning_threshold=80.0, # Warning at 80% memory
    memory_critical_threshold=95.0 # Critical at 95% memory
)
```

#### Health Metrics

The system tracks detailed metrics:

```python
health = check_system_health()
metrics = health.metrics

print(f"CPU cores: {metrics['cpu_count']}")
print(f"Total memory: {metrics['memory_total_gb']:.2f} GB")
print(f"Available memory: {metrics['memory_available_gb']:.2f} GB")
print(f"Total disk space: {metrics['disk_total_gb']:.2f} GB")
print(f"Free disk space: {metrics['disk_free_gb']:.2f} GB")
print(f"Load average: {metrics['load_average']}")
```

---

## Resilience Features

### Circuit Breaker Pattern

The system implements circuit breakers to protect against cascading failures:

```python
from eeg_rag.utils.common_utils import CircuitBreaker

# Create circuit breaker for external service
cb = CircuitBreaker(
    name="pubmed_api",
    failure_threshold=5,      # Open after 5 failures
    timeout_seconds=60.0      # Try again after 60 seconds
)

# Use circuit breaker
async def call_external_api():
    try:
        result = await cb.call(external_service_function)
        return result
    except CircuitBreakerOpenError:
        # Service is down, use fallback
        return fallback_response()
```

#### Circuit Breaker States

| State       | Description        | Behavior                              |
| ----------- | ------------------ | ------------------------------------- |
| `CLOSED`    | Normal operation   | Calls pass through normally           |
| `OPEN`      | Service is failing | Calls are blocked, return immediately |
| `HALF_OPEN` | Testing recovery   | Single call allowed to test service   |

### Retry Mechanisms

Automated retry with exponential backoff for transient failures:

```python
from eeg_rag.utils.common_utils import retry_with_backoff

@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=10.0
)
async def unreliable_operation():
    # Operation that might fail transiently
    return await external_api_call()
```

### Performance Monitoring

Agents automatically track performance metrics:

```python
agent = LocalDataAgent()

# Execute queries
for i in range(10):
    query = AgentQuery(text=f"Query {i}")
    result = await agent.run(query)

# Get performance statistics
stats = agent.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average execution time: {stats['average_execution_time_seconds']:.3f}s")
print(f"Total executions: {stats['total_executions']}")
```

---

## Performance Characteristics

### Benchmarks

Based on system testing with MockAgent:

| Metric                | Value                   | Notes                                     |
| --------------------- | ----------------------- | ----------------------------------------- |
| **Throughput**        | ~95 requests/second     | Single agent, concurrent execution        |
| **Latency**           | <0.1 seconds            | Per agent execution (excluding LLM calls) |
| **Concurrency**       | 5+ agents in parallel   | Limited by system resources               |
| **Memory Efficiency** | <5% increase under load | Sustained 100-request test                |

### Scaling Characteristics

- **CPU**: Linear scaling up to CPU core count
- **Memory**: Bounded growth with memory management
- **I/O**: Async operations prevent blocking
- **Network**: Circuit breakers prevent cascading failures

### Performance Tuning

```python
# Optimize for throughput
agent = LocalDataAgent(config={
    "batch_size": 100,           # Process in batches
    "cache_enabled": True,       # Enable result caching
    "async_workers": 4           # Parallel processing workers
})

# Optimize for latency
agent = LocalDataAgent(config={
    "batch_size": 1,            # Process immediately
    "preload_models": True,     # Keep models in memory
    "connection_pool_size": 10  # Maintain connections
})
```

---

## Error Handling Patterns

### Standardized Error Messages

All errors use consistent formatting:

```python
from eeg_rag.utils.common_utils import format_error_message

try:
    risky_operation()
except Exception as e:
    formatted_error = format_error_message(
        "operation_name",
        e,
        {"context": "additional_info"}
    )
    logger.error(formatted_error)
```

### Error Categories

| Category          | HTTP Code | Description        | Recovery Action                 |
| ----------------- | --------- | ------------------ | ------------------------------- |
| **Validation**    | 400       | Invalid input data | Fix input and retry             |
| **Authorization** | 401/403   | Access denied      | Check credentials               |
| **Resource**      | 404       | Resource not found | Verify resource exists          |
| **Rate Limit**    | 429       | Too many requests  | Wait and retry                  |
| **Server**        | 500       | Internal error     | Check logs, retry if transient  |
| **Timeout**       | 504       | Request timeout    | Increase timeout, check network |

### Exception Hierarchy

```python
class EEGRAGError(Exception):
    """Base exception for EEG-RAG system"""
    pass

class ValidationError(EEGRAGError):
    """Input validation failed"""
    pass

class ResourceError(EEGRAGError):
    """Resource access failed"""
    pass

class CircuitBreakerOpenError(EEGRAGError):
    """Circuit breaker is open"""
    pass
```

---

## Development Guidelines

### Code Style

- **Formatting**: Black formatter (line length 88)
- **Imports**: isort for consistent import ordering
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all classes and methods

### Validation Standards

All inputs must be validated using common utilities:

```python
from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    validate_range
)

def process_query(text: str, max_results: int, confidence_threshold: float):
    # Validate all inputs
    text = validate_non_empty_string(text, "query text")
    max_results = validate_positive_number(max_results, "max_results")
    confidence_threshold = validate_range(
        confidence_threshold, 0.0, 1.0, "confidence_threshold"
    )
    
    # Process with validated inputs
    return process_validated_query(text, max_results, confidence_threshold)
```

### Time Measurement Standards

All time measurements must use seconds as the base unit:

```python
from eeg_rag.utils.common_utils import SECOND, MINUTE, HOUR, format_duration_human_readable

# Store durations in seconds
execution_time = 3.5 * SECOND  # 3.5 seconds
cache_ttl = 10 * MINUTE        # 10 minutes
backup_interval = 24 * HOUR    # 24 hours

# Display human-readable times
print(f"Execution took: {format_duration_human_readable(execution_time)}")
```

### Error Handling Standards

```python
async def robust_operation():
    try:
        return await risky_operation()
    except ValidationError:
        # Validation errors should not be retried
        raise
    except (ConnectionError, TimeoutError) as e:
        # Transient errors can be retried
        logger.warning(f"Transient error: {e}")
        raise
    except Exception as e:
        # Unexpected errors need investigation
        logger.exception(f"Unexpected error in robust_operation: {e}")
        raise
```

---

## Testing Strategy

### Test Categories

#### Unit Tests
- **Location**: `tests/unit/`
- **Scope**: Individual functions and classes
- **Mock**: External dependencies
- **Coverage**: >90% for core modules

#### Integration Tests
- **Location**: `tests/integration/`
- **Scope**: Component interactions
- **Dependencies**: Real services (Redis, FAISS)
- **Coverage**: Happy path and error scenarios

#### Boundary Tests
- **Location**: `tests/test_*_boundary_conditions.py`
- **Scope**: Edge cases and limits
- **Focus**: Input validation, resource constraints
- **Coverage**: All validation functions

#### Resilience Tests
- **Location**: `tests/test_system_resilience.py`
- **Scope**: System behavior under stress
- **Focus**: Failure simulation, recovery testing
- **Coverage**: Circuit breakers, retry mechanisms

### Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests
pytest tests/test_system_resilience.py   # Resilience tests

# Run with coverage
pytest tests/ --cov=eeg_rag --cov-report=html

# Run performance benchmarks
pytest tests/test_system_resilience.py::TestPerformanceBenchmarks -v -s
```

### Current Test Status

| Module                | Unit Tests | Integration Tests | Boundary Tests | Resilience Tests |
| --------------------- | ---------- | ----------------- | -------------- | ---------------- |
| **Common Utils**      | ✅ 45 tests | ✅                 | ✅              | ✅                |
| **Base Agent**        | ✅          | ✅                 | ✅ 23 tests     | ✅                |
| **Memory Manager**    | ✅ 19 tests | ✅                 | ✅              | ✅                |
| **Local Agent**       | ✅ 20 tests | ✅                 | ✅              | ✅                |
| **System Resilience** | N/A        | N/A               | N/A            | ✅ 11 tests       |

**Total: 324 tests passing**

---

## Troubleshooting Guide

### Common Issues

#### High Memory Usage

**Symptoms**: Memory warnings in health check, slow performance
**Causes**: Large document cache, memory leaks, inefficient data structures
**Solutions**:
```python
# Check memory usage
health = check_system_health()
if health.memory_percent > 85:
    # Clear caches
    agent.clear_cache()
    # Reduce batch sizes
    agent.config["batch_size"] = 10
    # Force garbage collection
    import gc; gc.collect()
```

#### Circuit Breaker Open

**Symptoms**: `CircuitBreakerOpenError` exceptions
**Causes**: External service failures, network issues, service overload
**Solutions**:
```python
# Check circuit breaker status
if cb.state == CircuitBreakerState.OPEN:
    # Wait for timeout or reset manually
    cb.reset()
    # Use fallback service
    result = await fallback_service()
```

#### Performance Degradation

**Symptoms**: High execution times, low throughput
**Causes**: Resource contention, inefficient queries, blocking operations
**Solutions**:
```python
# Check agent statistics
stats = agent.get_statistics()
if stats['average_execution_time_seconds'] > 1.0:
    # Profile slow operations
    # Optimize queries
    # Increase concurrency
    agent.config["max_workers"] = 8
```

### Diagnostic Commands

```python
# System health overview
health = check_system_health()
print(health.to_dict())

# Agent performance summary
agent = get_agent("local_data")
print(agent.get_performance_summary())

# Memory usage breakdown
import tracemalloc
tracemalloc.start()
# ... run operations ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')[:10]
```

### Log Analysis

```bash
# Find error patterns
grep "ERROR" logs/eeg-rag.log | tail -20

# Monitor performance
grep "execution completed" logs/eeg-rag.log | awk '{print $NF}' | stats

# Check circuit breaker events
grep "circuit breaker" logs/eeg-rag.log
```

### Performance Monitoring

```python
# Real-time performance monitoring
import time
import asyncio

async def monitor_performance():
    while True:
        health = check_system_health()
        if health.status != SystemStatus.HEALTHY:
            print(f"⚠️  System status: {health.status.value}")
            for warning in health.warnings:
                print(f"   {warning}")
        
        await asyncio.sleep(30)  # Check every 30 seconds

# Run in background
asyncio.create_task(monitor_performance())
```

---

## API Reference

### Core Classes

#### SystemHealth
```python
@dataclass
class SystemHealth:
    status: SystemStatus
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    timestamp: datetime
    warnings: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]: ...
```

#### CircuitBreaker
```python
@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any: ...
    def reset(self) -> None: ...
```

#### AgentResult
```python
@dataclass
class AgentResult:
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    agent_type: Optional[AgentType] = None
    elapsed_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
```

### Utility Functions

```python
# Validation functions
validate_non_empty_string(value: str, field_name: str, allow_none: bool = False) -> str
validate_positive_number(value: Union[int, float], field_name: str) -> Union[int, float]
validate_range(value: float, min_val: float, max_val: float, field_name: str) -> float

# System monitoring
check_system_health(**thresholds) -> SystemHealth
create_circuit_breaker(name: str, **kwargs) -> CircuitBreaker

# Time utilities
standardize_time_unit(value: float, unit: str) -> float
format_duration_human_readable(seconds: float, precision: int = 2) -> str

# Error handling
format_error_message(operation: str, exception: Exception, context: Dict[str, Any]) -> str
retry_with_backoff(max_retries: int = 3, **kwargs) -> Callable  # Decorator
```

---

This technical reference provides comprehensive information for working with the EEG-RAG system. For additional details, see:

- [Project Status](PROJECT_STATUS.md) - Current implementation status
- [Architecture Decisions](architecture-decisions/) - Design rationales
- [API Documentation](api/) - Generated API docs
- [Contributing Guide](../CONTRIBUTING.md) - Development workflow