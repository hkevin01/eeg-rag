"""
Performance Monitoring System for EEG-RAG

This module provides comprehensive performance monitoring, benchmarking,
and optimization capabilities for the EEG-RAG system.

Features:
- Real-time performance metrics collection
- Automated benchmarking and profiling
- Resource usage optimization
- Performance regression detection
- System health alerting
"""

import time
import psutil
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import statistics
import json

from eeg_rag.utils.common_utils import (
    check_system_health, 
    SystemHealth,
    validate_positive_number,
    format_error_message
)


# ---------------------------------------------------------------------------
# ID           : monitoring.performance_monitor.PerformanceMetrics
# Requirement  : `PerformanceMetrics` class shall be instantiable and expose the documented interface
# Purpose      : Performance metrics for monitoring system performance
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate PerformanceMetrics with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system performance"""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMetrics.duration_seconds
    # Requirement  : `duration_seconds` shall get duration in seconds
    # Purpose      : Get duration in seconds
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        return self.duration_ms / 1000.0
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMetrics.to_dict
    # Requirement  : `to_dict` shall convert to dictionary for serialization
    # Purpose      : Convert to dictionary for serialization
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time, 
            "duration_ms": self.duration_ms,
            "duration_seconds": self.duration_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat()
        }


# ---------------------------------------------------------------------------
# ID           : monitoring.performance_monitor.BenchmarkResult
# Requirement  : `BenchmarkResult` class shall be instantiable and expose the documented interface
# Purpose      : Result of benchmark testing
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate BenchmarkResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass 
class BenchmarkResult:
    """Result of benchmark testing"""
    benchmark_name: str
    operation_count: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_dev_ms: float
    throughput_ops_per_sec: float
    success_rate: float
    system_health: SystemHealth
    timestamp: datetime = field(default_factory=datetime.now)
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.BenchmarkResult.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "benchmark_name": self.benchmark_name,
            "operation_count": self.operation_count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "std_dev_ms": self.std_dev_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "success_rate": self.success_rate,
            "system_health": self.system_health.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }


# ---------------------------------------------------------------------------
# ID           : monitoring.performance_monitor.PerformanceMonitor
# Requirement  : `PerformanceMonitor` class shall be instantiable and expose the documented interface
# Purpose      : Advanced performance monitoring system with real-time metrics collection,
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate PerformanceMonitor with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class PerformanceMonitor:
    """
    Advanced performance monitoring system with real-time metrics collection,
    benchmarking capabilities, and optimization recommendations.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.__init__
    # Requirement  : `__init__` shall initialize performance monitor
    # Purpose      : Initialize performance monitor
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : max_metrics_history: int (default=1000); performance_threshold_ms: float (default=1000.0); memory_threshold_mb: float (default=500.0); logger: Optional[logging.Logger] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        max_metrics_history: int = 1000,
        performance_threshold_ms: float = 1000.0,
        memory_threshold_mb: float = 500.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize performance monitor
        
        Args:
            max_metrics_history: Maximum metrics to keep in memory
            performance_threshold_ms: Alert threshold for slow operations
            memory_threshold_mb: Alert threshold for high memory usage
            logger: Logger instance
        """
        self.max_metrics_history = validate_positive_number(
            max_metrics_history, "max_metrics_history", min_value=10
        )
        self.performance_threshold_ms = validate_positive_number(
            performance_threshold_ms, "performance_threshold_ms"
        )
        self.memory_threshold_mb = validate_positive_number(
            memory_threshold_mb, "memory_threshold_mb"
        )
        
        self.logger = logger or logging.getLogger("eeg_rag.monitoring.performance")
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=self.max_metrics_history)
        self.active_operations: Dict[str, float] = {}
        
        # Performance statistics
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Benchmark results storage
        self.benchmark_results: List[BenchmarkResult] = []
        
        self.logger.info(
            f"PerformanceMonitor initialized: "
            f"history={self.max_metrics_history}, "
            f"perf_threshold={self.performance_threshold_ms}ms, "
            f"mem_threshold={self.memory_threshold_mb}MB"
        )
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.start_operation
    # Requirement  : `start_operation` shall start monitoring an operation
    # Purpose      : Start monitoring an operation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : operation_name: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def start_operation(self, operation_name: str) -> str:
        """
        Start monitoring an operation
        
        Args:
            operation_name: Name of the operation to monitor
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        self.active_operations[operation_id] = time.time()
        
        self.logger.debug(f"Started monitoring operation: {operation_name}")
        return operation_id
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.end_operation
    # Requirement  : `end_operation` shall end monitoring an operation and record metrics
    # Purpose      : End monitoring an operation and record metrics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : operation_id: str; success: bool (default=True); error_message: Optional[str] (default=None); metadata: Optional[Dict[str, Any]] (default=None)
    # Outputs      : PerformanceMetrics
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def end_operation(
        self, 
        operation_id: str, 
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """
        End monitoring an operation and record metrics
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            error_message: Error message if operation failed
            metadata: Additional metadata
            
        Returns:
            PerformanceMetrics object
        """
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation ID {operation_id} not found")
        
        end_time = time.time()
        start_time = self.active_operations.pop(operation_id)
        
        # Extract operation name from ID
        operation_name = operation_id.rsplit("_", 1)[0]
        
        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        
        # Get current system metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self._update_operation_stats(metrics)
        
        # Check for performance alerts
        self._check_performance_alerts(metrics)
        
        self.logger.debug(
            f"Operation {operation_name} completed: "
            f"{duration_ms:.2f}ms, {memory_usage_mb:.2f}MB, "
            f"success={success}"
        )
        
        return metrics
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor._update_operation_stats
    # Requirement  : `_update_operation_stats` shall update aggregated statistics for operation type
    # Purpose      : Update aggregated statistics for operation type
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : metrics: PerformanceMetrics
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _update_operation_stats(self, metrics: PerformanceMetrics) -> None:
        """Update aggregated statistics for operation type"""
        op_name = metrics.operation_name
        
        if op_name not in self.operation_stats:
            self.operation_stats[op_name] = {
                "total_operations": 0,
                "total_duration_ms": 0.0,
                "successful_operations": 0,
                "failed_operations": 0,
                "durations": []
            }
        
        stats = self.operation_stats[op_name]
        stats["total_operations"] += 1
        stats["total_duration_ms"] += metrics.duration_ms
        stats["durations"].append(metrics.duration_ms)
        
        if metrics.success:
            stats["successful_operations"] += 1
        else:
            stats["failed_operations"] += 1
        
        # Keep only last 100 durations for statistics
        if len(stats["durations"]) > 100:
            stats["durations"] = stats["durations"][-100:]
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor._check_performance_alerts
    # Requirement  : `_check_performance_alerts` shall check metrics against thresholds and generate alerts
    # Purpose      : Check metrics against thresholds and generate alerts
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : metrics: PerformanceMetrics
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        # Check duration threshold
        if metrics.duration_ms > self.performance_threshold_ms:
            alerts.append(
                f"Slow operation detected: {metrics.operation_name} "
                f"took {metrics.duration_ms:.2f}ms (threshold: {self.performance_threshold_ms}ms)"
            )
        
        # Check memory threshold
        if metrics.memory_usage_mb > self.memory_threshold_mb:
            alerts.append(
                f"High memory usage detected: {metrics.operation_name} "
                f"used {metrics.memory_usage_mb:.2f}MB (threshold: {self.memory_threshold_mb}MB)"
            )
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(alert)
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.get_operation_statistics
    # Requirement  : `get_operation_statistics` shall get performance statistics for operations
    # Purpose      : Get performance statistics for operations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : operation_name: Optional[str] (default=None)
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_operation_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for operations
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Dictionary with operation statistics
        """
        if operation_name:
            if operation_name not in self.operation_stats:
                return {"error": f"No statistics found for operation: {operation_name}"}
            
            stats = self.operation_stats[operation_name].copy()
            durations = stats.pop("durations", [])
            
            if durations:
                stats.update({
                    "avg_duration_ms": statistics.mean(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    "success_rate": stats["successful_operations"] / stats["total_operations"]
                })
            
            return {operation_name: stats}
        else:
            # Return all operations
            all_stats = {}
            for op_name in self.operation_stats:
                all_stats.update(self.get_operation_statistics(op_name))
            return all_stats
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.get_recent_metrics
    # Requirement  : `get_recent_metrics` shall get recent performance metrics
    # Purpose      : Get recent performance metrics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : count: int (default=50)
    # Outputs      : List[Dict[str, Any]]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_recent_metrics(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        count = validate_positive_number(count, "count", min_value=1)
        recent = list(self.metrics_history)[-count:]
        return [m.to_dict() for m in recent]
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.benchmark_operation
    # Requirement  : `benchmark_operation` shall benchmark an operation with multiple iterations
    # Purpose      : Benchmark an operation with multiple iterations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : operation_func: Callable; operation_args: tuple (default=()); operation_kwargs: Optional[Dict[str, Any]] (default=None); iterations: int (default=100); benchmark_name: str (default='benchmark')
    # Outputs      : BenchmarkResult
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def benchmark_operation(
        self,
        operation_func: Callable,
        operation_args: tuple = (),
        operation_kwargs: Optional[Dict[str, Any]] = None,
        iterations: int = 100,
        benchmark_name: str = "benchmark"
    ) -> BenchmarkResult:
        """
        Benchmark an operation with multiple iterations
        
        Args:
            operation_func: Function to benchmark
            operation_args: Arguments to pass to function
            operation_kwargs: Keyword arguments to pass to function
            iterations: Number of iterations to run
            benchmark_name: Name for the benchmark
            
        Returns:
            BenchmarkResult with statistics
        """
        operation_kwargs = operation_kwargs or {}
        iterations = validate_positive_number(iterations, "iterations", min_value=1)
        
        self.logger.info(f"Starting benchmark '{benchmark_name}' with {iterations} iterations")
        
        # Record system health before benchmark
        system_health_before = check_system_health()
        
        # Run benchmark
        durations = []
        successes = 0
        
        start_benchmark = time.time()
        
        for i in range(iterations):
            op_id = self.start_operation(f"{benchmark_name}_iter_{i}")
            
            try:
                start_iter = time.time()
                operation_func(*operation_args, **operation_kwargs)
                end_iter = time.time()
                
                duration_ms = (end_iter - start_iter) * 1000
                durations.append(duration_ms)
                successes += 1
                
                self.end_operation(op_id, success=True)
                
            except Exception as e:
                duration_ms = (time.time() - start_iter) * 1000
                durations.append(duration_ms)
                
                self.end_operation(op_id, success=False, error_message=str(e))
                self.logger.debug(f"Benchmark iteration {i} failed: {e}")
        
        end_benchmark = time.time()
        total_duration_ms = (end_benchmark - start_benchmark) * 1000
        
        # Calculate statistics
        avg_duration = statistics.mean(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
        throughput = iterations / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0
        success_rate = successes / iterations if iterations > 0 else 0.0
        
        # Record system health after benchmark
        system_health_after = check_system_health()
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            operation_count=iterations,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            system_health=system_health_after
        )
        
        self.benchmark_results.append(result)
        
        self.logger.info(
            f"Benchmark '{benchmark_name}' completed: "
            f"avg={avg_duration:.2f}ms, throughput={throughput:.2f}ops/s, "
            f"success_rate={success_rate:.1%}"
        )
        
        return result
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.benchmark_async_operation
    # Requirement  : `benchmark_async_operation` shall benchmark an async operation
    # Purpose      : Benchmark an async operation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : async_operation_func: Callable; operation_args: tuple (default=()); operation_kwargs: Optional[Dict[str, Any]] (default=None); iterations: int (default=100); benchmark_name: str (default='async_benchmark'); concurrency: int (default=1)
    # Outputs      : BenchmarkResult
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def benchmark_async_operation(
        self,
        async_operation_func: Callable,
        operation_args: tuple = (),
        operation_kwargs: Optional[Dict[str, Any]] = None,
        iterations: int = 100,
        benchmark_name: str = "async_benchmark",
        concurrency: int = 1
    ) -> BenchmarkResult:
        """
        Benchmark an async operation
        
        Args:
            async_operation_func: Async function to benchmark
            operation_args: Arguments to pass to function
            operation_kwargs: Keyword arguments to pass to function
            iterations: Number of iterations to run
            benchmark_name: Name for the benchmark
            concurrency: Number of concurrent operations
            
        Returns:
            BenchmarkResult with statistics
        """
        operation_kwargs = operation_kwargs or {}
        iterations = validate_positive_number(iterations, "iterations", min_value=1)
        concurrency = validate_positive_number(concurrency, "concurrency", min_value=1)
        
        self.logger.info(
            f"Starting async benchmark '{benchmark_name}' with {iterations} iterations, "
            f"concurrency={concurrency}"
        )
        
        system_health_before = check_system_health()
        
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # ---------------------------------------------------------------------------
        # ID           : monitoring.performance_monitor.PerformanceMonitor.run_iteration
        # Requirement  : `run_iteration` shall execute as specified
        # Purpose      : Run iteration
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : i: int
        # Outputs      : Implicitly None or see body
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Must be awaited (async)
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
        async def run_iteration(i: int):
            async with semaphore:
                op_id = self.start_operation(f"{benchmark_name}_async_iter_{i}")
                
                try:
                    start_iter = time.time()
                    await async_operation_func(*operation_args, **operation_kwargs)
                    end_iter = time.time()
                    
                    duration_ms = (end_iter - start_iter) * 1000
                    self.end_operation(op_id, success=True)
                    return duration_ms, True
                    
                except Exception as e:
                    duration_ms = (time.time() - start_iter) * 1000
                    self.end_operation(op_id, success=False, error_message=str(e))
                    self.logger.debug(f"Async benchmark iteration {i} failed: {e}")
                    return duration_ms, False
        
        # Run all iterations
        start_benchmark = time.time()
        results = await asyncio.gather(*[run_iteration(i) for i in range(iterations)])
        end_benchmark = time.time()
        
        total_duration_ms = (end_benchmark - start_benchmark) * 1000
        
        # Process results
        durations = [r[0] for r in results]
        successes = sum(1 for r in results if r[1])
        
        # Calculate statistics
        avg_duration = statistics.mean(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0
        throughput = iterations / (total_duration_ms / 1000) if total_duration_ms > 0 else 0.0
        success_rate = successes / iterations if iterations > 0 else 0.0
        
        system_health_after = check_system_health()
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            operation_count=iterations,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            std_dev_ms=std_dev,
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            system_health=system_health_after
        )
        
        self.benchmark_results.append(result)
        
        self.logger.info(
            f"Async benchmark '{benchmark_name}' completed: "
            f"avg={avg_duration:.2f}ms, throughput={throughput:.2f}ops/s, "
            f"success_rate={success_rate:.1%}"
        )
        
        return result
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.PerformanceMonitor.export_metrics
    # Requirement  : `export_metrics` shall export collected metrics to file
    # Purpose      : Export collected metrics to file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : filepath: Path; format: str (default='json')
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def export_metrics(self, filepath: Path, format: str = "json") -> None:
        """
        Export collected metrics to file
        
        Args:
            filepath: Output file path
            format: Export format ("json" or "csv")
        """
        if format.lower() == "json":
            data = {
                "metrics_history": [m.to_dict() for m in self.metrics_history],
                "operation_stats": self.get_operation_statistics(),
                "benchmark_results": [b.to_dict() for b in self.benchmark_results],
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format.lower() == "csv":
            import csv
            
            with open(filepath, 'w', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].to_dict().keys())
                    writer.writeheader()
                    for metrics in self.metrics_history:
                        writer.writerow(metrics.to_dict())
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {filepath} in {format} format")


# ---------------------------------------------------------------------------
# ID           : monitoring.performance_monitor.SystemOptimizer
# Requirement  : `SystemOptimizer` class shall be instantiable and expose the documented interface
# Purpose      : System optimization recommendations based on performance metrics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate SystemOptimizer with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class SystemOptimizer:
    """
    System optimization recommendations based on performance metrics
    """
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer.__init__
    # Requirement  : `__init__` shall initialize optimizer with performance monitor
    # Purpose      : Initialize optimizer with performance monitor
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : performance_monitor: PerformanceMonitor
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize optimizer with performance monitor
        
        Args:
            performance_monitor: PerformanceMonitor instance
        """
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger("eeg_rag.monitoring.optimizer")
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer.analyze_performance
    # Requirement  : `analyze_performance` shall analyze performance metrics and generate optimization recommendations
    # Purpose      : Analyze performance metrics and generate optimization recommendations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance metrics and generate optimization recommendations
        
        Returns:
            Dictionary with analysis and recommendations
        """
        stats = self.performance_monitor.get_operation_statistics()
        recent_metrics = self.performance_monitor.get_recent_metrics(100)
        
        analysis = {
            "performance_summary": self._summarize_performance(stats),
            "bottlenecks": self._identify_bottlenecks(stats),
            "resource_usage": self._analyze_resource_usage(recent_metrics),
            "recommendations": self._generate_recommendations(stats, recent_metrics),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer._summarize_performance
    # Requirement  : `_summarize_performance` shall generate performance summary
    # Purpose      : Generate performance summary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : stats: Dict[str, Any]
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _summarize_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {
            "total_operations": sum(s.get("total_operations", 0) for s in stats.values()),
            "operation_types": len(stats),
            "avg_success_rate": 0.0,
            "slowest_operations": []
        }
        
        if stats:
            success_rates = [s.get("success_rate", 0) for s in stats.values()]
            summary["avg_success_rate"] = statistics.mean(success_rates)
            
            # Find slowest operations
            slowest = sorted(
                stats.items(),
                key=lambda x: x[1].get("avg_duration_ms", 0),
                reverse=True
            )[:5]
            
            summary["slowest_operations"] = [
                {"operation": op, "avg_duration_ms": data.get("avg_duration_ms", 0)}
                for op, data in slowest
            ]
        
        return summary
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer._identify_bottlenecks
    # Requirement  : `_identify_bottlenecks` shall identify performance bottlenecks
    # Purpose      : Identify performance bottlenecks
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : stats: Dict[str, Any]
    # Outputs      : List[Dict[str, Any]]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _identify_bottlenecks(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        for op_name, op_stats in stats.items():
            avg_duration = op_stats.get("avg_duration_ms", 0)
            success_rate = op_stats.get("success_rate", 1.0)
            
            # Check for slow operations
            if avg_duration > self.performance_monitor.performance_threshold_ms:
                bottlenecks.append({
                    "type": "slow_operation",
                    "operation": op_name,
                    "avg_duration_ms": avg_duration,
                    "threshold_ms": self.performance_monitor.performance_threshold_ms,
                    "severity": "high" if avg_duration > 2 * self.performance_monitor.performance_threshold_ms else "medium"
                })
            
            # Check for low success rates
            if success_rate < 0.95:
                bottlenecks.append({
                    "type": "low_success_rate",
                    "operation": op_name,
                    "success_rate": success_rate,
                    "severity": "high" if success_rate < 0.8 else "medium"
                })
        
        return bottlenecks
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer._analyze_resource_usage
    # Requirement  : `_analyze_resource_usage` shall analyze resource usage patterns
    # Purpose      : Analyze resource usage patterns
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : recent_metrics: List[Dict[str, Any]]
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _analyze_resource_usage(self, recent_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        memory_usage = [m["memory_usage_mb"] for m in recent_metrics]
        cpu_usage = [m["cpu_percent"] for m in recent_metrics]
        durations = [m["duration_ms"] for m in recent_metrics]
        
        return {
            "memory": {
                "avg_mb": statistics.mean(memory_usage),
                "max_mb": max(memory_usage),
                "std_dev_mb": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
            },
            "cpu": {
                "avg_percent": statistics.mean(cpu_usage),
                "max_percent": max(cpu_usage),
                "std_dev_percent": statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0,
            },
            "duration": {
                "avg_ms": statistics.mean(durations),
                "max_ms": max(durations),
                "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            }
        }
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.SystemOptimizer._generate_recommendations
    # Requirement  : `_generate_recommendations` shall generate optimization recommendations
    # Purpose      : Generate optimization recommendations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : stats: Dict[str, Any]; recent_metrics: List[Dict[str, Any]]
    # Outputs      : List[Dict[str, Any]]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _generate_recommendations(
        self, 
        stats: Dict[str, Any], 
        recent_metrics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze memory usage
        if recent_metrics:
            avg_memory = statistics.mean(m["memory_usage_mb"] for m in recent_metrics)
            if avg_memory > self.performance_monitor.memory_threshold_mb:
                recommendations.append({
                    "category": "memory_optimization",
                    "priority": "high",
                    "description": f"High memory usage detected (avg: {avg_memory:.1f}MB)",
                    "suggestions": [
                        "Implement memory cleanup in long-running operations",
                        "Reduce batch sizes for processing",
                        "Add memory monitoring to detect leaks"
                    ]
                })
        
        # Analyze operation performance
        for op_name, op_stats in stats.items():
            avg_duration = op_stats.get("avg_duration_ms", 0)
            if avg_duration > self.performance_monitor.performance_threshold_ms:
                recommendations.append({
                    "category": "performance_optimization", 
                    "priority": "medium",
                    "description": f"Slow operation: {op_name} ({avg_duration:.1f}ms)",
                    "suggestions": [
                        "Profile operation to identify bottlenecks",
                        "Consider caching frequently accessed data",
                        "Optimize database queries or API calls"
                    ]
                })
        
        return recommendations


# Performance monitoring context manager
# ---------------------------------------------------------------------------
# ID           : monitoring.performance_monitor.monitor_performance
# Requirement  : `monitor_performance` class shall be instantiable and expose the documented interface
# Purpose      : Context manager for easy performance monitoring
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate monitor_performance with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class monitor_performance:
    """Context manager for easy performance monitoring"""
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.monitor_performance.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : monitor: PerformanceMonitor; operation_name: str
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.operation_id = None
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.monitor_performance.__enter__
    # Requirement  : `__enter__` shall execute as specified
    # Purpose      :   enter  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __enter__(self):
        self.operation_id = self.monitor.start_operation(self.operation_name)
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : monitoring.performance_monitor.monitor_performance.__exit__
    # Requirement  : `__exit__` shall execute as specified
    # Purpose      :   exit  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : exc_type; exc_val; exc_tb
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None
        self.monitor.end_operation(self.operation_id, success, error_message)
