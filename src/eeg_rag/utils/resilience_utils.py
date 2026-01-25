"""
Resilience Utilities for EEG-RAG

Provides health monitoring, circuit breakers, rate limiting, graceful
degradation, and crash recovery mechanisms for production resilience.

Requirements Implemented:
- REQ-REL-001: Circuit breaker pattern
- REQ-REL-002: Health monitoring
- REQ-REL-004: Rate limiting
- REQ-REL-005: Graceful degradation
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# REQ-REL-001: Circuit breaker states
class CircuitState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """
    REQ-REL-001: Statistics for circuit breaker.
    
    Attributes:
        total_calls: Total number of calls
        failures: Number of failed calls
        successes: Number of successful calls
        rejected: Number of rejected calls (when open)
        last_failure: Timestamp of last failure
        last_success: Timestamp of last success
    """
    total_calls: int = 0
    failures: int = 0
    successes: int = 0
    rejected: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    state_changes: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_calls": self.total_calls,
            "failures": self.failures,
            "successes": self.successes,
            "rejected": self.rejected,
            "failure_rate": self.failure_rate,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "state_changes": self.state_changes,
        }


class CircuitBreaker:
    """
    REQ-REL-001: Implements the circuit breaker pattern.
    
    Prevents cascading failures by failing fast when a service is unhealthy.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        monitored_exceptions: tuple = (Exception,),
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before half-open
            half_open_max_calls: Max test calls in half-open state
            monitored_exceptions: Exceptions that count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.monitored_exceptions = monitored_exceptions
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._stats = CircuitStats()
        self._lock = threading.Lock()
        self._state_change_callbacks: List[Callable] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
        
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")
        
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        current_state = self.state  # Triggers state check
        
        with self._lock:
            self._stats.total_calls += 1
            
            if current_state == CircuitState.OPEN:
                self._stats.rejected += 1
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
            
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected += 1
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.monitored_exceptions as e:
            self._on_failure(e)
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._stats.successes += 1
            self._stats.last_success = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.CLOSED)
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._stats.failures += 1
            self._stats.last_failure = datetime.now()
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._last_failure_time = None
    
    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        return self._stats


class CircuitOpenError(Exception):
    """Raised when trying to call through an open circuit."""
    pass


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable:
    """
    REQ-REL-001: Decorator to apply circuit breaker pattern.
    
    Args:
        name: Circuit breaker identifier
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery attempt
        
    Returns:
        Decorated function
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    _circuit_breakers[name] = cb
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get a circuit breaker by name."""
    return _circuit_breakers.get(name)


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


# REQ-REL-002: Health check system
class HealthStatus(Enum):
    """Health check statuses."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """
    REQ-REL-002: Result of a health check.
    
    Attributes:
        name: Name of the health check
        status: Health status
        message: Optional status message
        duration_ms: Time taken for check
        details: Additional details
    """
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class HealthChecker:
    """
    REQ-REL-002: Manages health checks for system components.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.Lock()
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    def register(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Unique name for the check
            check_fn: Function that returns HealthCheckResult
        """
        with self._lock:
            self._checks[name] = check_fn
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            name: Name of check to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False
    
    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Name of check to run
            
        Returns:
            HealthCheckResult
        """
        with self._lock:
            check_fn = self._checks.get(name)
        
        if check_fn is None:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check '{name}' not found",
            )
        
        start_time = time.time()
        try:
            result = check_fn()
            result.duration_ms = (time.time() - start_time) * 1000
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        
        with self._lock:
            self._last_results[name] = result
        
        return result
    
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check names to results
        """
        with self._lock:
            check_names = list(self._checks.keys())
        
        results = {}
        for name in check_names:
            results[name] = self.check(name)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            HealthStatus based on all checks
        """
        results = self.check_all()
        
        if not results:
            return HealthStatus.HEALTHY
        
        statuses = [r.status for r in results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get cached results from last check."""
        with self._lock:
            return self._last_results.copy()


# Global health checker instance
_health_checker = HealthChecker()


def register_health_check(
    name: str,
    check_fn: Callable[[], HealthCheckResult],
) -> None:
    """Register a health check with the global checker."""
    _health_checker.register(name, check_fn)


def check_health(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Check system health.
    
    Args:
        name: Optional specific check to run
        
    Returns:
        Health check results
    """
    if name:
        result = _health_checker.check(name)
        return result.to_dict()
    
    results = _health_checker.check_all()
    overall = _health_checker.get_overall_status()
    
    return {
        "status": overall.value,
        "checks": {k: v.to_dict() for k, v in results.items()},
        "timestamp": datetime.now().isoformat(),
    }


# REQ-REL-004: Rate limiting
class RateLimiter:
    """
    REQ-REL-004: Token bucket rate limiter.
    
    Controls the rate of operations to prevent overload.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        name: str = "default",
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity (defaults to rate)
            name: Identifier for this rate limiter
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.name = name
        
        self._tokens = self.capacity
        self._last_update = time.time()
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._accepted = 0
        self._rejected = 0
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_update = now
    
    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        with self._lock:
            self._refill()
            self._total_requests += 1
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._accepted += 1
                return True
            else:
                self._rejected += 1
                return False
    
    def wait(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait
            
        Returns:
            True if tokens acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Calculate wait time
            with self._lock:
                wait_time = (tokens - self._tokens) / self.rate
            
            time.sleep(min(wait_time, 0.1))
    
    async def async_wait(
        self,
        tokens: float = 1.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Async version of wait.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait
            
        Returns:
            True if tokens acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            with self._lock:
                wait_time = (tokens - self._tokens) / self.rate
            
            await asyncio.sleep(min(wait_time, 0.1))
    
    def stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "name": self.name,
                "rate": self.rate,
                "capacity": self.capacity,
                "tokens_available": self._tokens,
                "total_requests": self._total_requests,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "rejection_rate": self._rejected / max(1, self._total_requests),
            }


def rate_limit(
    rate: float,
    capacity: Optional[float] = None,
    block: bool = True,
    timeout: Optional[float] = None,
) -> Callable:
    """
    REQ-REL-004: Decorator to apply rate limiting.
    
    Args:
        rate: Tokens per second
        capacity: Token bucket capacity
        block: Whether to block or fail immediately
        timeout: Max wait time if blocking
        
    Returns:
        Decorated function
    """
    limiter = RateLimiter(rate=rate, capacity=capacity)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if block:
                if not limiter.wait(timeout=timeout):
                    raise RateLimitExceeded(f"Rate limit timeout for {func.__name__}")
            else:
                if not limiter.acquire():
                    raise RateLimitExceeded(f"Rate limit exceeded for {func.__name__}")
            
            return func(*args, **kwargs)
        
        wrapper.rate_limiter = limiter
        return wrapper
    
    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


# REQ-REL-005: Graceful degradation
@dataclass
class DegradationLevel:
    """
    REQ-REL-005: Degradation level configuration.
    
    Attributes:
        name: Level name (e.g., "normal", "reduced", "minimal")
        priority: Priority threshold (lower = more critical)
        enabled_features: Features enabled at this level
    """
    name: str
    priority: int
    enabled_features: Set[str] = field(default_factory=set)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at this level."""
        return feature in self.enabled_features


class GracefulDegradation:
    """
    REQ-REL-005: Manages graceful degradation of services.
    
    Allows the system to continue operating with reduced functionality
    when under stress or when components fail.
    """
    
    # Standard degradation levels
    NORMAL = DegradationLevel("normal", 0, set())
    REDUCED = DegradationLevel("reduced", 1, set())
    MINIMAL = DegradationLevel("minimal", 2, set())
    EMERGENCY = DegradationLevel("emergency", 3, set())
    
    def __init__(self):
        """Initialize degradation manager."""
        self._current_level = self.NORMAL
        self._feature_priorities: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._level_change_callbacks: List[Callable] = []
    
    def register_feature(self, name: str, priority: int) -> None:
        """
        Register a feature with its priority.
        
        Args:
            name: Feature name
            priority: Priority (0 = critical, higher = less critical)
        """
        with self._lock:
            self._feature_priorities[name] = priority
    
    def set_level(self, level: DegradationLevel) -> None:
        """
        Set the degradation level.
        
        Args:
            level: New degradation level
        """
        with self._lock:
            old_level = self._current_level
            self._current_level = level
            
            # Update enabled features based on priority
            level.enabled_features = {
                name for name, priority in self._feature_priorities.items()
                if priority <= level.priority
            }
            
            logger.warning(
                f"Degradation level changed: {old_level.name} -> {level.name}"
            )
            
            for callback in self._level_change_callbacks:
                try:
                    callback(old_level, level)
                except Exception as e:
                    logger.error(f"Degradation callback error: {e}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled at current level.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled
        """
        with self._lock:
            priority = self._feature_priorities.get(feature, 0)
            return priority <= self._current_level.priority
    
    @property
    def current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        with self._lock:
            return self._current_level
    
    def on_level_change(self, callback: Callable) -> None:
        """Register a callback for level changes."""
        self._level_change_callbacks.append(callback)
    
    def status(self) -> Dict[str, Any]:
        """Get degradation status."""
        with self._lock:
            return {
                "level": self._current_level.name,
                "priority": self._current_level.priority,
                "enabled_features": list(self._current_level.enabled_features),
                "all_features": {
                    name: {
                        "priority": priority,
                        "enabled": priority <= self._current_level.priority,
                    }
                    for name, priority in self._feature_priorities.items()
                },
            }


# Global degradation manager
_degradation = GracefulDegradation()


def degrade_to(level: DegradationLevel) -> None:
    """Set the global degradation level."""
    _degradation.set_level(level)


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled globally."""
    return _degradation.is_feature_enabled(feature)


def feature_gate(feature: str) -> Callable:
    """
    REQ-REL-005: Decorator for feature-gated functions.
    
    Args:
        feature: Feature name to check
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature):
                raise FeatureDisabledError(
                    f"Feature '{feature}' is disabled at current degradation level"
                )
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class FeatureDisabledError(Exception):
    """Raised when a disabled feature is accessed."""
    pass


# Convenience exports
__all__ = [
    "CircuitState",
    "CircuitStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",
    "register_health_check",
    "check_health",
    "RateLimiter",
    "rate_limit",
    "RateLimitExceeded",
    "DegradationLevel",
    "GracefulDegradation",
    "degrade_to",
    "is_feature_enabled",
    "feature_gate",
    "FeatureDisabledError",
]
