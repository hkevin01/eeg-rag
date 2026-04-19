#!/usr/bin/env python3
"""
Time Utilities Module for EEG-RAG

Provides standardized time measurement, conversion, and formatting utilities
to ensure consistent time handling throughout the application.

Requirements Implemented:
    - REQ-TIME-001: Consistent time units across the system
    - REQ-TIME-002: Standard time measurement for performance monitoring
    - REQ-PERF-001: Performance measurement with sub-millisecond precision

Time Unit Standards:
    - Short durations (< 1 second): MILLISECONDS
    - Medium durations (1 second - 1 hour): SECONDS
    - Long durations (> 1 hour): MINUTES or HOURS
    - Timestamps: ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)

Example Usage:
    >>> from eeg_rag.utils.time_utils import Timer, TimeUnits, convert_time
    >>> 
    >>> # High-precision timing
    >>> with Timer() as t:
    ...     result = expensive_operation()
    >>> print(f"Took {t.elapsed_ms:.2f}ms")
    >>> 
    >>> # Time conversion
    >>> ms = convert_time(2.5, TimeUnits.SECONDS, TimeUnits.MILLISECONDS)
    >>> print(f"{ms}ms")  # 2500.0ms
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .logging_utils import get_logger

# REQ-TIME-001: Standard time units for the application
logger = get_logger(__name__)

# Type variable for generic function decorators
F = TypeVar('F', bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.TimeUnits
# Requirement  : `TimeUnits` class shall be instantiable and expose the documented interface
# Purpose      : Standardized time units for consistent measurement across the application
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
# Verification : Instantiate TimeUnits with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class TimeUnits(Enum):
    """
    Standardized time units for consistent measurement across the application.
    
    REQ-TIME-001: All time values in the system use these standard units.
    
    Usage:
        >>> unit = TimeUnits.MILLISECONDS
        >>> print(f"Short operation: 50{unit.symbol}")  # "Short operation: 50ms"
    """
    NANOSECONDS = ("ns", 1e-9)
    MICROSECONDS = ("μs", 1e-6)
    MILLISECONDS = ("ms", 1e-3)
    SECONDS = ("s", 1.0)
    MINUTES = ("min", 60.0)
    HOURS = ("hr", 3600.0)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimeUnits.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : symbol: str; seconds_multiplier: float
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
    def __init__(self, symbol: str, seconds_multiplier: float):
        self.symbol = symbol
        self.seconds_multiplier = seconds_multiplier
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimeUnits.from_string
    # Requirement  : `from_string` shall parse a time unit from string representation
    # Purpose      : Parse a time unit from string representation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : unit_string: str
    # Outputs      : 'TimeUnits'
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
    @classmethod
    def from_string(cls, unit_string: str) -> "TimeUnits":
        """
        Parse a time unit from string representation.
        
        Args:
            unit_string: String like 'ms', 's', 'min', etc.
            
        Returns:
            Corresponding TimeUnits enum value
            
        Raises:
            ValueError: If unit string is not recognized
            
        Example:
            >>> TimeUnits.from_string('ms')
            TimeUnits.MILLISECONDS
        """
        unit_map = {
            'ns': cls.NANOSECONDS,
            'nanoseconds': cls.NANOSECONDS,
            'nanosecond': cls.NANOSECONDS,
            'μs': cls.MICROSECONDS,
            'us': cls.MICROSECONDS,
            'microseconds': cls.MICROSECONDS,
            'microsecond': cls.MICROSECONDS,
            'ms': cls.MILLISECONDS,
            'milliseconds': cls.MILLISECONDS,
            'millisecond': cls.MILLISECONDS,
            's': cls.SECONDS,
            'sec': cls.SECONDS,
            'seconds': cls.SECONDS,
            'second': cls.SECONDS,
            'min': cls.MINUTES,
            'minutes': cls.MINUTES,
            'minute': cls.MINUTES,
            'hr': cls.HOURS,
            'hour': cls.HOURS,
            'hours': cls.HOURS,
        }
        
        normalized = unit_string.lower().strip()
        if normalized not in unit_map:
            valid_units = ', '.join(sorted(set(unit_map.keys())))
            raise ValueError(
                f"Unknown time unit: '{unit_string}'. "
                f"Valid units: {valid_units}"
            )
        return unit_map[normalized]


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.convert_time
# Requirement  : `convert_time` shall convert a time value between different units
# Purpose      : Convert a time value between different units
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : value: float; from_unit: TimeUnits; to_unit: TimeUnits
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
def convert_time(
    value: float,
    from_unit: TimeUnits,
    to_unit: TimeUnits
) -> float:
    """
    Convert a time value between different units.
    
    REQ-TIME-001: Provides consistent time unit conversion.
    
    Args:
        value: The time value to convert
        from_unit: Source time unit
        to_unit: Target time unit
        
    Returns:
        Converted time value
        
    Raises:
        ValueError: If value is negative
        
    Examples:
        >>> convert_time(2.5, TimeUnits.SECONDS, TimeUnits.MILLISECONDS)
        2500.0
        >>> convert_time(1500, TimeUnits.MILLISECONDS, TimeUnits.SECONDS)
        1.5
    """
    # REQ-FUNC-002: Input validation
    if value < 0:
        raise ValueError(f"Time value cannot be negative: {value}")
    
    # Convert to seconds, then to target unit
    seconds = value * from_unit.seconds_multiplier
    return seconds / to_unit.seconds_multiplier


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.format_duration
# Requirement  : `format_duration` shall format a duration in seconds to a human-readable string
# Purpose      : Format a duration in seconds to a human-readable string
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : duration_seconds: float; precision: int (default=2); auto_scale: bool (default=True)
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
def format_duration(
    duration_seconds: float,
    precision: int = 2,
    auto_scale: bool = True
) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    REQ-TIME-001: Consistent time formatting for user display.
    
    Args:
        duration_seconds: Duration in seconds
        precision: Decimal places for the formatted value
        auto_scale: Automatically choose appropriate unit
        
    Returns:
        Formatted duration string (e.g., "150.25ms", "2.50s", "1.25min")
        
    Examples:
        >>> format_duration(0.150)
        "150.00ms"
        >>> format_duration(2.5)
        "2.50s"
        >>> format_duration(150.0)
        "2.50min"
    """
    # REQ-FUNC-002: Handle edge cases
    if duration_seconds < 0:
        return f"-{format_duration(abs(duration_seconds), precision, auto_scale)}"
    
    if duration_seconds == 0:
        return f"0.{'0' * precision}ms"
    
    if not auto_scale:
        return f"{duration_seconds:.{precision}f}s"
    
    # REQ-TIME-001: Auto-scale to appropriate unit
    if duration_seconds < 0.001:  # < 1ms
        us = duration_seconds * 1_000_000
        return f"{us:.{precision}f}μs"
    elif duration_seconds < 1.0:  # < 1s
        ms = duration_seconds * 1000
        return f"{ms:.{precision}f}ms"
    elif duration_seconds < 60:  # < 1min
        return f"{duration_seconds:.{precision}f}s"
    elif duration_seconds < 3600:  # < 1hr
        minutes = duration_seconds / 60
        return f"{minutes:.{precision}f}min"
    else:
        hours = duration_seconds / 3600
        return f"{hours:.{precision}f}hr"


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.TimingStats
# Requirement  : `TimingStats` class shall be instantiable and expose the documented interface
# Purpose      : Statistics for a series of timed operations
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
# Verification : Instantiate TimingStats with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class TimingStats:
    """
    Statistics for a series of timed operations.
    
    REQ-PERF-002: Track latency statistics for performance monitoring.
    
    Attributes:
        name: Identifier for the timed operation
        samples: List of duration measurements (in seconds)
        unit: Display unit for statistics
    """
    name: str
    samples: List[float] = field(default_factory=list)
    unit: TimeUnits = TimeUnits.MILLISECONDS
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.add_sample
    # Requirement  : `add_sample` shall add a timing sample
    # Purpose      : Add a timing sample
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : duration_seconds: float
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
    def add_sample(self, duration_seconds: float) -> None:
        """Add a timing sample."""
        self.samples.append(duration_seconds)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.count
    # Requirement  : `count` shall number of samples collected
    # Purpose      : Number of samples collected
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : int
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
    def count(self) -> int:
        """Number of samples collected."""
        return len(self.samples)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.total_seconds
    # Requirement  : `total_seconds` shall total time across all samples
    # Purpose      : Total time across all samples
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
    def total_seconds(self) -> float:
        """Total time across all samples."""
        return sum(self.samples) if self.samples else 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.mean_seconds
    # Requirement  : `mean_seconds` shall mean duration in seconds
    # Purpose      : Mean duration in seconds
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
    def mean_seconds(self) -> float:
        """Mean duration in seconds."""
        return self.total_seconds / self.count if self.samples else 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.min_seconds
    # Requirement  : `min_seconds` shall minimum duration in seconds
    # Purpose      : Minimum duration in seconds
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
    def min_seconds(self) -> float:
        """Minimum duration in seconds."""
        return min(self.samples) if self.samples else 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.max_seconds
    # Requirement  : `max_seconds` shall maximum duration in seconds
    # Purpose      : Maximum duration in seconds
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
    def max_seconds(self) -> float:
        """Maximum duration in seconds."""
        return max(self.samples) if self.samples else 0.0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.percentile
    # Requirement  : `percentile` shall calculate the p-th percentile of samples
    # Purpose      : Calculate the p-th percentile of samples
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : p: float
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
    def percentile(self, p: float) -> float:
        """
        Calculate the p-th percentile of samples.
        
        Args:
            p: Percentile (0-100)
            
        Returns:
            Duration at the given percentile in seconds
        """
        if not self.samples:
            return 0.0
        
        if p < 0 or p > 100:
            raise ValueError(f"Percentile must be 0-100, got {p}")
        
        sorted_samples = sorted(self.samples)
        index = (len(sorted_samples) - 1) * p / 100
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_samples):
            return sorted_samples[-1]
        
        weight = index - lower
        return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.p50
    # Requirement  : `p50` shall 50th percentile (median) in seconds
    # Purpose      : 50th percentile (median) in seconds
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
    def p50(self) -> float:
        """50th percentile (median) in seconds."""
        return self.percentile(50)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.p95
    # Requirement  : `p95` shall 95th percentile in seconds
    # Purpose      : 95th percentile in seconds
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
    def p95(self) -> float:
        """95th percentile in seconds."""
        return self.percentile(95)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.p99
    # Requirement  : `p99` shall 99th percentile in seconds
    # Purpose      : 99th percentile in seconds
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
    def p99(self) -> float:
        """99th percentile in seconds."""
        return self.percentile(99)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.to_dict
    # Requirement  : `to_dict` shall convert statistics to dictionary format
    # Purpose      : Convert statistics to dictionary format
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
        """
        Convert statistics to dictionary format.
        
        REQ-PERF-002: Serializable statistics for monitoring.
        """
        multiplier = 1 / self.unit.seconds_multiplier
        return {
            'name': self.name,
            'count': self.count,
            'unit': self.unit.symbol,
            'total': round(self.total_seconds * multiplier, 3),
            'mean': round(self.mean_seconds * multiplier, 3),
            'min': round(self.min_seconds * multiplier, 3),
            'max': round(self.max_seconds * multiplier, 3),
            'p50': round(self.p50 * multiplier, 3),
            'p95': round(self.p95 * multiplier, 3),
            'p99': round(self.p99 * multiplier, 3),
        }
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.TimingStats.__str__
    # Requirement  : `__str__` shall human-readable summary
    # Purpose      : Human-readable summary
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
        """Human-readable summary."""
        if not self.samples:
            return f"{self.name}: no samples"
        
        stats = self.to_dict()
        return (
            f"{self.name}: n={stats['count']}, "
            f"mean={stats['mean']}{stats['unit']}, "
            f"p95={stats['p95']}{stats['unit']}, "
            f"min={stats['min']}{stats['unit']}, "
            f"max={stats['max']}{stats['unit']}"
        )


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.Timer
# Requirement  : `Timer` class shall be instantiable and expose the documented interface
# Purpose      : High-precision context manager for timing code blocks
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
# Verification : Instantiate Timer with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class Timer:
    """
    High-precision context manager for timing code blocks.
    
    REQ-TIME-002: Standard time measurement for performance monitoring.
    REQ-PERF-001: Sub-millisecond precision timing.
    
    Uses time.perf_counter() for highest precision timing available.
    
    Attributes:
        elapsed: Elapsed time in seconds
        elapsed_ms: Elapsed time in milliseconds
        elapsed_us: Elapsed time in microseconds
        
    Examples:
        >>> # Context manager usage
        >>> with Timer() as t:
        ...     result = expensive_operation()
        >>> print(f"Took {t.elapsed_ms:.2f}ms")
        
        >>> # Manual start/stop
        >>> timer = Timer()
        >>> timer.start()
        >>> result = expensive_operation()
        >>> timer.stop()
        >>> print(f"Took {timer.elapsed_ms:.2f}ms")
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.__init__
    # Requirement  : `__init__` shall initialize timer
    # Purpose      : Initialize timer
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: Optional[str] (default=None)
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
    def __init__(self, name: Optional[str] = None):
        """
        Initialize timer.
        
        Args:
            name: Optional name for logging/identification
        """
        self.name = name
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._is_running: bool = False
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.start
    # Requirement  : `start` shall start the timer
    # Purpose      : Start the timer
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : 'Timer'
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
    def start(self) -> "Timer":
        """
        Start the timer.
        
        Returns:
            Self for method chaining
        """
        self._start_time = time.perf_counter()
        self._end_time = None
        self._is_running = True
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.stop
    # Requirement  : `stop` shall stop the timer
    # Purpose      : Stop the timer
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
    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns:
            Elapsed time in seconds
            
        Raises:
            RuntimeError: If timer was not started
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        
        self._end_time = time.perf_counter()
        self._is_running = False
        return self.elapsed
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.elapsed
    # Requirement  : `elapsed` shall get elapsed time in seconds
    # Purpose      : Get elapsed time in seconds
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
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        If timer is still running, returns time since start.
        """
        if self._start_time is None:
            return 0.0
        
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.elapsed_ms
    # Requirement  : `elapsed_ms` shall elapsed time in milliseconds
    # Purpose      : Elapsed time in milliseconds
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
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.elapsed_us
    # Requirement  : `elapsed_us` shall elapsed time in microseconds
    # Purpose      : Elapsed time in microseconds
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
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self.elapsed * 1_000_000
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.elapsed_ns
    # Requirement  : `elapsed_ns` shall elapsed time in nanoseconds
    # Purpose      : Elapsed time in nanoseconds
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
    def elapsed_ns(self) -> float:
        """Elapsed time in nanoseconds."""
        return self.elapsed * 1_000_000_000
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.is_running
    # Requirement  : `is_running` shall check if timer is currently running
    # Purpose      : Check if timer is currently running
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
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        return self._is_running
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.__enter__
    # Requirement  : `__enter__` shall enter context manager
    # Purpose      : Enter context manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : 'Timer'
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
    def __enter__(self) -> "Timer":
        """Enter context manager."""
        self.start()
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.__exit__
    # Requirement  : `__exit__` shall exit context manager
    # Purpose      : Exit context manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : *args
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
    def __exit__(self, *args) -> None:
        """Exit context manager."""
        self.stop()
        if self.name:
            logger.debug(f"Timer '{self.name}': {format_duration(self.elapsed)}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.Timer.__str__
    # Requirement  : `__str__` shall human-readable representation
    # Purpose      : Human-readable representation
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
        """Human-readable representation."""
        status = "running" if self._is_running else "stopped"
        return f"Timer({self.name or 'unnamed'}, {status}, {format_duration(self.elapsed)})"


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.timed
# Requirement  : `timed` shall decorator for timing function execution
# Purpose      : Decorator for timing function execution
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: Optional[str] (default=None); log_level: str (default='DEBUG'); threshold_ms: Optional[float] (default=None)
# Outputs      : Callable[[F], F]
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
def timed(
    name: Optional[str] = None,
    log_level: str = "DEBUG",
    threshold_ms: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator for timing function execution.
    
    REQ-TIME-002: Standard timing decorator for performance monitoring.
    
    Args:
        name: Optional name for the timer (defaults to function name)
        log_level: Log level for timing messages
        threshold_ms: Only log if execution exceeds this threshold (ms)
        
    Returns:
        Decorated function
        
    Examples:
        >>> @timed()
        ... def my_function():
        ...     pass
        
        >>> @timed(name="expensive_op", threshold_ms=100)
        ... def slow_function():
        ...     pass
    """
    # ---------------------------------------------------------------------------
    # ID           : utils.time_utils.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: F
    # Outputs      : F
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
    def decorator(func: F) -> F:
        timer_name = name or func.__name__
        
        # ---------------------------------------------------------------------------
        # ID           : utils.time_utils.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(timer_name) as timer:
                result = func(*args, **kwargs)
            
            elapsed_ms = timer.elapsed_ms
            
            # Only log if above threshold (if specified)
            if threshold_ms is None or elapsed_ms >= threshold_ms:
                log_func = getattr(logger, log_level.lower(), logger.debug)
                log_func(f"{timer_name} completed in {format_duration(timer.elapsed)}")
            
            return result
        
        # ---------------------------------------------------------------------------
        # ID           : utils.time_utils.async_wrapper
        # Requirement  : `async_wrapper` shall execute as specified
        # Purpose      : Async wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
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
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with Timer(timer_name) as timer:
                result = await func(*args, **kwargs)
            
            elapsed_ms = timer.elapsed_ms
            
            # Only log if above threshold (if specified)
            if threshold_ms is None or elapsed_ms >= threshold_ms:
                log_func = getattr(logger, log_level.lower(), logger.debug)
                log_func(f"{timer_name} completed in {format_duration(timer.elapsed)}")
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.get_utc_timestamp
# Requirement  : `get_utc_timestamp` shall get current UTC timestamp in ISO 8601 format
# Purpose      : Get current UTC timestamp in ISO 8601 format
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
def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.
    
    REQ-TIME-001: Standard timestamp format.
    
    Returns:
        ISO 8601 formatted timestamp string
        
    Example:
        >>> get_utc_timestamp()
        '2026-01-25T10:30:45.123Z'
    """
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.get_unix_timestamp
# Requirement  : `get_unix_timestamp` shall get current Unix timestamp with millisecond precision
# Purpose      : Get current Unix timestamp with millisecond precision
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
def get_unix_timestamp() -> float:
    """
    Get current Unix timestamp with millisecond precision.
    
    Returns:
        Unix timestamp in seconds
    """
    return time.time()


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.get_unix_timestamp_ms
# Requirement  : `get_unix_timestamp_ms` shall get current Unix timestamp in milliseconds
# Purpose      : Get current Unix timestamp in milliseconds
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : int
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
def get_unix_timestamp_ms() -> int:
    """
    Get current Unix timestamp in milliseconds.
    
    Returns:
        Unix timestamp in milliseconds
    """
    return int(time.time() * 1000)


# Module-level timing statistics registry
_timing_registry: Dict[str, TimingStats] = {}


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.record_timing
# Requirement  : `record_timing` shall record a timing sample in the global registry
# Purpose      : Record a timing sample in the global registry
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: str; duration_seconds: float
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
def record_timing(name: str, duration_seconds: float) -> None:
    """
    Record a timing sample in the global registry.
    
    REQ-PERF-002: Aggregate timing statistics.
    
    Args:
        name: Operation name
        duration_seconds: Duration in seconds
    """
    if name not in _timing_registry:
        _timing_registry[name] = TimingStats(name=name)
    _timing_registry[name].add_sample(duration_seconds)


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.get_timing_stats
# Requirement  : `get_timing_stats` shall get timing statistics for an operation
# Purpose      : Get timing statistics for an operation
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: str
# Outputs      : Optional[TimingStats]
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
def get_timing_stats(name: str) -> Optional[TimingStats]:
    """
    Get timing statistics for an operation.
    
    Args:
        name: Operation name
        
    Returns:
        TimingStats or None if not found
    """
    return _timing_registry.get(name)


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.get_all_timing_stats
# Requirement  : `get_all_timing_stats` shall get all timing statistics as a dictionary
# Purpose      : Get all timing statistics as a dictionary
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Dict[str, Dict[str, Any]]
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
def get_all_timing_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get all timing statistics as a dictionary.
    
    Returns:
        Dictionary of operation names to statistics
    """
    return {name: stats.to_dict() for name, stats in _timing_registry.items()}


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.clear_timing_stats
# Requirement  : `clear_timing_stats` shall clear all timing statistics
# Purpose      : Clear all timing statistics
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
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
def clear_timing_stats() -> None:
    """Clear all timing statistics."""
    _timing_registry.clear()


# REQ-PERF-001: Performance threshold constants (in seconds)
LATENCY_THRESHOLDS = {
    'local_retrieval': 0.100,      # 100ms for local operations
    'external_api': 5.0,           # 5s for external APIs
    'end_to_end_query': 2.0,       # 2s for complete query
    'embedding_generation': 0.050,  # 50ms for embedding
    'cache_operation': 0.010,      # 10ms for cache access
}


# ---------------------------------------------------------------------------
# ID           : utils.time_utils.check_latency_threshold
# Requirement  : `check_latency_threshold` shall check if an operation exceeded its latency threshold
# Purpose      : Check if an operation exceeded its latency threshold
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : operation: str; duration_seconds: float; warn: bool (default=True)
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
def check_latency_threshold(
    operation: str,
    duration_seconds: float,
    warn: bool = True
) -> bool:
    """
    Check if an operation exceeded its latency threshold.
    
    REQ-PERF-001: Latency monitoring and alerting.
    
    Args:
        operation: Operation name (must be in LATENCY_THRESHOLDS)
        duration_seconds: Actual duration
        warn: Whether to log a warning if threshold exceeded
        
    Returns:
        True if within threshold, False if exceeded
    """
    threshold = LATENCY_THRESHOLDS.get(operation)
    
    if threshold is None:
        logger.warning(f"Unknown operation for threshold check: {operation}")
        return True
    
    exceeded = duration_seconds > threshold
    
    if exceeded and warn:
        logger.warning(
            f"Latency threshold exceeded for {operation}: "
            f"{format_duration(duration_seconds)} > {format_duration(threshold)}"
        )
    
    return not exceeded
