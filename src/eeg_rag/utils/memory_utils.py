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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.MemoryUsage
# Requirement  : `MemoryUsage` class shall be instantiable and expose the documented interface
# Purpose      : Current memory usage snapshot
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
# Verification : Instantiate MemoryUsage with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryUsage.to_dict
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
        """Convert to dictionary for serialization."""
        return {
            'rss_mb': round(self.rss_mb, 2),
            'vms_mb': round(self.vms_mb, 2),
            'percent': round(self.percent, 2),
            'available_mb': round(self.available_mb, 2),
            'timestamp': self.timestamp,
        }
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryUsage.__str__
    # Requirement  : `__str__` shall execute as specified
    # Purpose      :   str  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def __str__(self) -> str:
        return f"Memory(RSS={self.rss_mb:.1f}MB, VMS={self.vms_mb:.1f}MB, {self.percent:.1f}%)"


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.MemoryProfile
# Requirement  : `MemoryProfile` class shall be instantiable and expose the documented interface
# Purpose      : Memory profiling result for an operation
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
# Verification : Instantiate MemoryProfile with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryProfile.leaked
    # Requirement  : `leaked` shall check if there's a potential memory leak (significant positive delta)
    # Purpose      : Check if there's a potential memory leak (significant positive delta)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    def leaked(self) -> bool:
        """Check if there's a potential memory leak (significant positive delta)."""
        return self.delta_mb > 10.0  # More than 10MB retained
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryProfile.to_dict
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.get_memory_usage
# Requirement  : `get_memory_usage` shall get current memory usage
# Purpose      : Get current memory usage
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : MemoryUsage
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.get_system_memory
# Requirement  : `get_system_memory` shall get system-wide memory information
# Purpose      : Get system-wide memory information
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Dict[str, float]
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.force_gc
# Requirement  : `force_gc` shall force garbage collection and return statistics
# Purpose      : Force garbage collection and return statistics
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Dict[str, int]
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.get_object_count
# Requirement  : `get_object_count` shall get count of objects by type
# Purpose      : Get count of objects by type
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Dict[str, int]
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.MemoryMonitor
# Requirement  : `MemoryMonitor` class shall be instantiable and expose the documented interface
# Purpose      : Context manager for memory monitoring
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
# Verification : Instantiate MemoryMonitor with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class MemoryMonitor:
    """
    Context manager for memory monitoring.
    
    REQ-MEM-001: Memory monitoring during operations.
    
    Example:
        >>> with MemoryMonitor("process_documents") as monitor:
        ...     process_documents()
        >>> print(f"Used {monitor.peak_mb:.2f} MB peak")
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor.__init__
    # Requirement  : `__init__` shall initialize memory monitor
    # Purpose      : Initialize memory monitor
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : operation: str; log_result: bool (default=True); gc_before: bool (default=False); gc_after: bool (default=False); track_peak: bool (default=True); peak_interval: float (default=0.1)
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor.__enter__
    # Requirement  : `__enter__` shall start monitoring
    # Purpose      : Start monitoring
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : 'MemoryMonitor'
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor.__exit__
    # Requirement  : `__exit__` shall stop monitoring and log results
    # Purpose      : Stop monitoring and log results
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor._start_peak_tracking
    # Requirement  : `_start_peak_tracking` shall start background peak memory tracking
    # Purpose      : Start background peak memory tracking
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
    def _start_peak_tracking(self):
        """Start background peak memory tracking."""
        self._stop_tracking.clear()
        
        # ---------------------------------------------------------------------------
        # ID           : utils.memory_utils.MemoryMonitor.track
        # Requirement  : `track` shall execute as specified
        # Purpose      : Track
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
        def track():
            while not self._stop_tracking.is_set():
                current = get_memory_usage().rss_mb
                self.peak_mb = max(self.peak_mb, current)
                time.sleep(self.peak_interval)
        
        self._tracker_thread = threading.Thread(target=track, daemon=True)
        self._tracker_thread.start()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor._stop_peak_tracking
    # Requirement  : `_stop_peak_tracking` shall stop background peak memory tracking
    # Purpose      : Stop background peak memory tracking
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
    def _stop_peak_tracking(self):
        """Stop background peak memory tracking."""
        self._stop_tracking.set()
        if self._tracker_thread:
            self._tracker_thread.join(timeout=1.0)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor._log_result
    # Requirement  : `_log_result` shall log monitoring results
    # Purpose      : Log monitoring results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : duration: float
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
    def _log_result(self, duration: float):
        """Log monitoring results."""
        delta = self.end_mb - self.start_mb
        logger.info(
            f"Memory[{self.operation}]: "
            f"Start={self.start_mb:.1f}MB, End={self.end_mb:.1f}MB, "
            f"Peak={self.peak_mb:.1f}MB, Delta={delta:+.1f}MB, "
            f"Duration={duration:.2f}s"
        )
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryMonitor.get_profile
    # Requirement  : `get_profile` shall get memory profile for this operation
    # Purpose      : Get memory profile for this operation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : MemoryProfile
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.MemoryLeakDetector
# Requirement  : `MemoryLeakDetector` class shall be instantiable and expose the documented interface
# Purpose      : Detects potential memory leaks by tracking object references
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
# Verification : Instantiate MemoryLeakDetector with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryLeakDetector.__init__
    # Requirement  : `__init__` shall initialize leak detector
    # Purpose      : Initialize leak detector
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
    def __init__(self):
        """Initialize leak detector."""
        self._snapshots: Dict[str, Dict[str, int]] = {}
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryLeakDetector.snapshot
    # Requirement  : `snapshot` shall take a snapshot of current object counts
    # Purpose      : Take a snapshot of current object counts
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str
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
    def snapshot(self, name: str):
        """
        Take a snapshot of current object counts.
        
        Args:
            name: Name for this snapshot
        """
        gc.collect()
        self._snapshots[name] = get_object_count()
        logger.debug(f"Memory snapshot '{name}' taken")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryLeakDetector.compare
    # Requirement  : `compare` shall compare two snapshots to detect leaks
    # Purpose      : Compare two snapshots to detect leaks
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : before: str; after: str; threshold: int (default=100)
    # Outputs      : Dict[str, int]
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryLeakDetector.clear
    # Requirement  : `clear` shall clear all snapshots
    # Purpose      : Clear all snapshots
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
    def clear(self):
        """Clear all snapshots."""
        self._snapshots.clear()


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.memory_efficient
# Requirement  : `memory_efficient` shall decorator for memory-efficient function execution
# Purpose      : Decorator for memory-efficient function execution
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : max_mb: Optional[float] (default=None); gc_after: bool (default=True); log_usage: bool (default=True)
# Outputs      : Callable
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
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable[..., T]
    # Outputs      : Callable[..., T]
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
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # ---------------------------------------------------------------------------
        # ID           : utils.memory_utils.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : T
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.low_memory_mode
# Requirement  : `low_memory_mode` shall context manager for low-memory operations
# Purpose      : Context manager for low-memory operations
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.MemoryPool
# Requirement  : `MemoryPool` class shall be instantiable and expose the documented interface
# Purpose      : Simple object pool for reusing expensive objects
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
# Verification : Instantiate MemoryPool with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryPool.__init__
    # Requirement  : `__init__` shall initialize memory pool
    # Purpose      : Initialize memory pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : create_fn: Callable[[], T]; max_size: int (default=10); reset_fn: Optional[Callable[[T], None]] (default=None)
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryPool.acquire
    # Requirement  : `acquire` shall acquire an object from the pool or create a new one
    # Purpose      : Acquire an object from the pool or create a new one
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : T
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryPool.release
    # Requirement  : `release` shall return an object to the pool
    # Purpose      : Return an object to the pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : obj: T
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryPool.clear
    # Requirement  : `clear` shall clear the pool
    # Purpose      : Clear the pool
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
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.memory_utils.MemoryPool.stats
    # Requirement  : `stats` shall get pool statistics
    # Purpose      : Get pool statistics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, int]
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.check_memory_health
# Requirement  : `check_memory_health` shall check overall memory health
# Purpose      : Check overall memory health
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


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.record_memory_profile
# Requirement  : `record_memory_profile` shall record a memory profile globally
# Purpose      : Record a memory profile globally
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : profile: MemoryProfile
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
def record_memory_profile(profile: MemoryProfile):
    """Record a memory profile globally."""
    with _global_profiles_lock:
        _global_profiles.append(profile)
        # Keep only last 1000 profiles
        if len(_global_profiles) > 1000:
            _global_profiles.pop(0)


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.get_memory_profiles
# Requirement  : `get_memory_profiles` shall get all recorded memory profiles
# Purpose      : Get all recorded memory profiles
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
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
def get_memory_profiles() -> List[Dict[str, Any]]:
    """Get all recorded memory profiles."""
    with _global_profiles_lock:
        return [p.to_dict() for p in _global_profiles]


# ---------------------------------------------------------------------------
# ID           : utils.memory_utils.clear_memory_profiles
# Requirement  : `clear_memory_profiles` shall clear all recorded memory profiles
# Purpose      : Clear all recorded memory profiles
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
def clear_memory_profiles():
    """Clear all recorded memory profiles."""
    with _global_profiles_lock:
        _global_profiles.clear()
