#!/usr/bin/env python3
"""
Memory Management Utilities for EEG-RAG

Provides memory monitoring, profiling, and optimization utilities
for managing memory in production environments.

Requirements Implemented:
    - REQ-MEM-001: Memory usage monitoring
    - REQ-MEM-002: Memory leak detection
    - REQ-MEM-003: Garbage collection optimization
    - REQ-PERF-003: Resource efficiency metrics

Example Usage:
    >>> from eeg_rag.utils.memory_utils import (
    ...     get_memory_usage, MemoryMonitor, memory_efficient
    ... )
    >>> 
    >>> # Get current memory usage
    >>> usage = get_memory_usage()
    >>> print(f"RSS: {usage.rss_mb:.2f} MB")
    >>> 
    >>> # Monitor memory in a context
    >>> with MemoryMonitor("embedding_operation") as monitor:
    ...     process_embeddings()
    >>> print(f"Peak memory: {monitor.peak_mb:.2f} MB")
"""

import gc
import sys
import psutil
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from .logging_utils import get_logger

logger = get_logger(__name__)

# Type variable for generic return types
T = TypeVar('T')


@dataclass
class MemoryUsage:
    """
    Current memory usage snapshot.
    
    REQ-MEM-001: Memory usage data structure.
    
    Attributes:
        rss_mb: Resident Set Size in megabytes
        vms_mb: Virtual Memory Size in megabytes
        percent: Memory usage percentage
        available_mb: Available system memory in MB
        timestamp: When the snapshot was taken
    """
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rss_mb': round(self.rss_mb, 2),
            'vms_mb': round(self.vms_mb, 2),
            'percent': round(self.percent, 2),
            'available_mb': round(self.available_mb, 2),
            'timestamp': self.timestamp,
        }
    
    def __str__(self) -> str:
        return f"Memory(RSS={self.rss_mb:.1f}MB, VMS={self.vms_mb:.1f}MB, {self.percent:.1f}%)"


@dataclass
class MemoryProfile:
    """
    Memory profiling result for an operation.
    
    REQ-MEM-001: Memory profiling data structure.
    
    Attributes:
        operation: Name of the profiled operation
        start_mb: Memory at start (RSS)
        end_mb: Memory at end (RSS)
        peak_mb: Peak memory during operation
        delta_mb: Memory change (end - start)
        duration_seconds: Time taken
        gc_collections: Number of garbage collections triggered
    """
    operation: str
    start_mb: float
    end_mb: float
    peak_mb: float
    delta_mb: float
    duration_seconds: float
    gc_collections: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def leaked(self) -> bool:
        """Check if there's a potential memory leak (significant positive delta)."""
        return self.delta_mb > 10.0  # More than 10MB retained
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'start_mb': round(self.start_mb, 2),
            'end_mb': round(self.end_mb, 2),
            'peak_mb': round(self.peak_mb, 2),
            'delta_mb': round(self.delta_mb, 2),
            'duration_seconds': round(self.duration_seconds, 4),
            'gc_collections': self.gc_collections,
            'leaked': self.leaked,
            'timestamp': self.timestamp,
        }


def get_memory_usage() -> MemoryUsage:
    """
    Get current memory usage.
    
    REQ-MEM-001: Real-time memory monitoring.
    
    Returns:
        MemoryUsage object with current memory stats
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return MemoryUsage(
        rss_mb=mem_info.rss / (1024 * 1024),
        vms_mb=mem_info.vms / (1024 * 1024),
        percent=process.memory_percent(),
        available_mb=virtual_mem.available / (1024 * 1024),
    )


def get_system_memory() -> Dict[str, float]:
    """
    Get system-wide memory information.
    
    REQ-MEM-001: System memory monitoring.
    
    Returns:
        Dictionary with system memory stats in MB
    """
    mem = psutil.virtual_memory()
    return {
        'total_mb': mem.total / (1024 * 1024),
        'available_mb': mem.available / (1024 * 1024),
        'used_mb': mem.used / (1024 * 1024),
        'percent': mem.percent,
        'free_mb': mem.free / (1024 * 1024),
    }


def force_gc() -> Dict[str, int]:
    """
    Force garbage collection and return statistics.
    
    REQ-MEM-003: Garbage collection optimization.
    
    Returns:
        Dictionary with GC stats per generation
    """
    before = get_memory_usage()
    
    # Collect all generations
    collected = {
        'gen0': gc.collect(0),
        'gen1': gc.collect(1),
        'gen2': gc.collect(2),
    }
    
    after = get_memory_usage()
    collected['freed_mb'] = before.rss_mb - after.rss_mb
    
    logger.debug(f"GC freed {collected['freed_mb']:.2f} MB")
    return collected


def get_object_count() -> Dict[str, int]:
    """
    Get count of objects by type.
    
    REQ-MEM-002: Memory leak detection helper.
    
    Returns:
        Dictionary with top object types and counts
    """
    gc.collect()
    type_counts: Dict[str, int] = {}
    
    for obj in gc.get_objects():
        type_name = type(obj).__name__
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    # Return top 20 types
    sorted_counts = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_counts[:20])


class MemoryMonitor:
    """
    Context manager for memory monitoring.
    
    REQ-MEM-001: Memory monitoring during operations.
    
    Example:
        >>> with MemoryMonitor("process_documents") as monitor:
        ...     process_documents()
        >>> print(f"Used {monitor.peak_mb:.2f} MB peak")
    """
    
    def __init__(
        self,
        operation: str,
        log_result: bool = True,
        gc_before: bool = False,
        gc_after: bool = False,
        track_peak: bool = True,
        peak_interval: float = 0.1
    ):
        """
        Initialize memory monitor.
        
        Args:
            operation: Name of the operation being monitored
            log_result: Whether to log the result
            gc_before: Run GC before monitoring
            gc_after: Run GC after monitoring
            track_peak: Track peak memory usage
            peak_interval: Interval for peak tracking in seconds
        """
        self.operation = operation
        self.log_result = log_result
        self.gc_before = gc_before
        self.gc_after = gc_after
        self.track_peak = track_peak
        self.peak_interval = peak_interval
        
        self.start_mb: float = 0.0
        self.end_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.start_time: float = 0.0
        self.gc_collections: int = 0
        
        self._stop_tracking = threading.Event()
        self._tracker_thread: Optional[threading.Thread] = None
    
    def __enter__(self) -> 'MemoryMonitor':
        """Start monitoring."""
        if self.gc_before:
            gc.collect()
        
        self.start_mb = get_memory_usage().rss_mb
        self.peak_mb = self.start_mb
        self.start_time = time.time()
        
        if self.track_peak:
            self._start_peak_tracking()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log results."""
        if self.track_peak:
            self._stop_peak_tracking()
        
        if self.gc_after:
            gc.collect()
        
        self.end_mb = get_memory_usage().rss_mb
        duration = time.time() - self.start_time
        
        if self.log_result:
            self._log_result(duration)
        
        return False
    
    def _start_peak_tracking(self):
        """Start background peak memory tracking."""
        self._stop_tracking.clear()
        
        def track():
            while not self._stop_tracking.is_set():
                current = get_memory_usage().rss_mb
                self.peak_mb = max(self.peak_mb, current)
                time.sleep(self.peak_interval)
        
        self._tracker_thread = threading.Thread(target=track, daemon=True)
        self._tracker_thread.start()
    
    def _stop_peak_tracking(self):
        """Stop background peak memory tracking."""
        self._stop_tracking.set()
        if self._tracker_thread:
            self._tracker_thread.join(timeout=1.0)
    
    def _log_result(self, duration: float):
        """Log monitoring results."""
        delta = self.end_mb - self.start_mb
        logger.info(
            f"Memory[{self.operation}]: "
            f"Start={self.start_mb:.1f}MB, End={self.end_mb:.1f}MB, "
            f"Peak={self.peak_mb:.1f}MB, Delta={delta:+.1f}MB, "
            f"Duration={duration:.2f}s"
        )
    
    def get_profile(self) -> MemoryProfile:
        """Get memory profile for this operation."""
        return MemoryProfile(
            operation=self.operation,
            start_mb=self.start_mb,
            end_mb=self.end_mb,
            peak_mb=self.peak_mb,
            delta_mb=self.end_mb - self.start_mb,
            duration_seconds=time.time() - self.start_time,
            gc_collections=self.gc_collections,
        )


class MemoryLeakDetector:
    """
    Detects potential memory leaks by tracking object references.
    
    REQ-MEM-002: Memory leak detection.
    
    Example:
        >>> detector = MemoryLeakDetector()
        >>> detector.snapshot("before")
        >>> # ... operations ...
        >>> detector.snapshot("after")
        >>> leaks = detector.compare("before", "after")
    """
    
    def __init__(self):
        """Initialize leak detector."""
        self._snapshots: Dict[str, Dict[str, int]] = {}
    
    def snapshot(self, name: str):
        """
        Take a snapshot of current object counts.
        
        Args:
            name: Name for this snapshot
        """
        gc.collect()
        self._snapshots[name] = get_object_count()
        logger.debug(f"Memory snapshot '{name}' taken")
    
    def compare(
        self,
        before: str,
        after: str,
        threshold: int = 100
    ) -> Dict[str, int]:
        """
        Compare two snapshots to detect leaks.
        
        Args:
            before: Name of the 'before' snapshot
            after: Name of the 'after' snapshot
            threshold: Minimum difference to report
            
        Returns:
            Dictionary of type -> count increase for potential leaks
        """
        if before not in self._snapshots or after not in self._snapshots:
            raise ValueError(f"Snapshots not found: {before}, {after}")
        
        before_counts = self._snapshots[before]
        after_counts = self._snapshots[after]
        
        leaks = {}
        for type_name, after_count in after_counts.items():
            before_count = before_counts.get(type_name, 0)
            diff = after_count - before_count
            if diff >= threshold:
                leaks[type_name] = diff
        
        return leaks
    
    def clear(self):
        """Clear all snapshots."""
        self._snapshots.clear()


def memory_efficient(
    max_mb: Optional[float] = None,
    gc_after: bool = True,
    log_usage: bool = True
) -> Callable:
    """
    Decorator for memory-efficient function execution.
    
    REQ-MEM-003: Memory optimization.
    
    Args:
        max_mb: Maximum allowed memory increase (logs warning if exceeded)
        gc_after: Run GC after function execution
        log_usage: Log memory usage
        
    Returns:
        Decorated function
        
    Example:
        >>> @memory_efficient(max_mb=100)
        ... def process_large_data():
        ...     ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_usage = get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if gc_after:
                    gc.collect()
                
                end_usage = get_memory_usage()
                delta = end_usage.rss_mb - start_usage.rss_mb
                
                if log_usage:
                    logger.debug(
                        f"Memory[{func.__name__}]: "
                        f"Delta={delta:+.1f}MB, Current={end_usage.rss_mb:.1f}MB"
                    )
                
                if max_mb and delta > max_mb:
                    logger.warning(
                        f"Memory limit exceeded in {func.__name__}: "
                        f"{delta:.1f}MB > {max_mb}MB allowed"
                    )
        
        return wrapper
    return decorator


@contextmanager
def low_memory_mode():
    """
    Context manager for low-memory operations.
    
    REQ-MEM-003: Memory optimization for constrained environments.
    
    Enables aggressive garbage collection and reduces memory usage.
    """
    # Store original thresholds
    original_thresholds = gc.get_threshold()
    
    try:
        # Set more aggressive GC thresholds
        gc.set_threshold(100, 5, 5)
        
        # Force initial collection
        gc.collect()
        
        yield
        
    finally:
        # Restore thresholds
        gc.set_threshold(*original_thresholds)
        
        # Final cleanup
        gc.collect()


class MemoryPool:
    """
    Simple object pool for reusing expensive objects.
    
    REQ-MEM-003: Memory optimization through object reuse.
    
    Example:
        >>> pool = MemoryPool(create_fn=lambda: np.zeros(1000000))
        >>> arr = pool.acquire()
        >>> # use arr
        >>> pool.release(arr)
    """
    
    def __init__(
        self,
        create_fn: Callable[[], T],
        max_size: int = 10,
        reset_fn: Optional[Callable[[T], None]] = None
    ):
        """
        Initialize memory pool.
        
        Args:
            create_fn: Function to create new objects
            max_size: Maximum pool size
            reset_fn: Optional function to reset objects before reuse
        """
        self.create_fn = create_fn
        self.max_size = max_size
        self.reset_fn = reset_fn
        self._pool: List[T] = []
        self._lock = threading.Lock()
        self._total_created = 0
        self._total_reused = 0
    
    def acquire(self) -> T:
        """
        Acquire an object from the pool or create a new one.
        
        Returns:
            Object from pool or newly created
        """
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._total_reused += 1
                return obj
            
            self._total_created += 1
            return self.create_fn()
    
    def release(self, obj: T):
        """
        Return an object to the pool.
        
        Args:
            obj: Object to return
        """
        with self._lock:
            if len(self._pool) < self.max_size:
                if self.reset_fn:
                    self.reset_fn(obj)
                self._pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'total_created': self._total_created,
                'total_reused': self._total_reused,
                'reuse_ratio': (
                    self._total_reused / max(1, self._total_created + self._total_reused)
                ),
            }


# Memory thresholds for monitoring
MEMORY_THRESHOLDS = {
    'warning_percent': 70.0,
    'critical_percent': 85.0,
    'oom_percent': 95.0,
}


def check_memory_health() -> Dict[str, Any]:
    """
    Check overall memory health.
    
    REQ-MEM-001: Memory health monitoring.
    
    Returns:
        Dictionary with health status and recommendations
    """
    usage = get_memory_usage()
    sys_mem = get_system_memory()
    
    status = "healthy"
    recommendations = []
    
    if usage.percent >= MEMORY_THRESHOLDS['oom_percent']:
        status = "critical"
        recommendations.append("URGENT: Memory nearly exhausted. Reduce workload immediately.")
    elif usage.percent >= MEMORY_THRESHOLDS['critical_percent']:
        status = "critical"
        recommendations.append("Memory usage critical. Consider reducing batch sizes.")
    elif usage.percent >= MEMORY_THRESHOLDS['warning_percent']:
        status = "warning"
        recommendations.append("Memory usage elevated. Monitor closely.")
    
    if sys_mem['available_mb'] < 1024:
        recommendations.append("Less than 1GB system memory available.")
    
    return {
        'status': status,
        'process_usage': usage.to_dict(),
        'system_memory': sys_mem,
        'recommendations': recommendations,
        'thresholds': MEMORY_THRESHOLDS,
    }


# Global memory monitor for application-wide tracking
_global_profiles: List[MemoryProfile] = []
_global_profiles_lock = threading.Lock()


def record_memory_profile(profile: MemoryProfile):
    """Record a memory profile globally."""
    with _global_profiles_lock:
        _global_profiles.append(profile)
        # Keep only last 1000 profiles
        if len(_global_profiles) > 1000:
            _global_profiles.pop(0)


def get_memory_profiles() -> List[Dict[str, Any]]:
    """Get all recorded memory profiles."""
    with _global_profiles_lock:
        return [p.to_dict() for p in _global_profiles]


def clear_memory_profiles():
    """Clear all recorded memory profiles."""
    with _global_profiles_lock:
        _global_profiles.clear()
