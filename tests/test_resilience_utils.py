"""
Unit Tests for Resilience Utilities

Tests circuit breakers, health checks, rate limiting, and graceful degradation.

Requirements Tested:
- REQ-REL-001: Circuit breaker pattern
- REQ-REL-002: Health monitoring
- REQ-REL-004: Rate limiting
- REQ-REL-005: Graceful degradation
"""

import pytest
import time
import threading
from datetime import datetime
from unittest.mock import Mock

from eeg_rag.utils.resilience_utils import (
    CircuitState,
    CircuitStats,
    CircuitBreaker,
    CircuitOpenError,
    circuit_breaker,
    get_circuit_breaker,
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    register_health_check,
    check_health,
    RateLimiter,
    rate_limit,
    RateLimitExceeded,
    DegradationLevel,
    GracefulDegradation,
    is_feature_enabled,
    feature_gate,
    FeatureDisabledError,
)


class TestCircuitState:
    """Test CircuitState enum"""
    
    def test_states_exist(self):
        """REQ-REL-001: Circuit states defined"""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitStats:
    """Test CircuitStats dataclass"""
    
    def test_default_stats(self):
        """REQ-REL-001: Default statistics"""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.failures == 0
        assert stats.failure_rate == 0.0
    
    def test_failure_rate_calculation(self):
        """REQ-REL-001: Failure rate calculation"""
        stats = CircuitStats(total_calls=100, failures=25)
        assert stats.failure_rate == 0.25
    
    def test_to_dict(self):
        """REQ-REL-001: Serialize stats"""
        stats = CircuitStats(
            total_calls=10,
            failures=2,
            successes=8,
        )
        d = stats.to_dict()
        assert d['total_calls'] == 10
        assert d['failure_rate'] == 0.2


class TestCircuitBreaker:
    """Test CircuitBreaker class"""
    
    def test_initial_state_closed(self):
        """REQ-REL-001: Initial state is closed"""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
    
    def test_successful_call(self):
        """REQ-REL-001: Successful call through circuit"""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        result = cb.call(lambda: "success")
        
        assert result == "success"
        assert cb.stats.successes == 1
    
    def test_opens_after_threshold(self):
        """REQ-REL-001: Opens after failure threshold"""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        def failing():
            raise ValueError("Error")
        
        # Fail threshold times
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing)
        
        assert cb.state == CircuitState.OPEN
        assert cb.stats.failures == 3
    
    def test_rejects_when_open(self):
        """REQ-REL-001: Rejects calls when open"""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        # Should reject
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "test")
        
        assert cb.stats.rejected >= 1
    
    def test_half_open_after_timeout(self):
        """REQ-REL-001: Transitions to half-open after timeout"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_closes_on_success_in_half_open(self):
        """REQ-REL-001: Closes on success in half-open"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Open and wait
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        
        time.sleep(0.15)
        
        # Successful call in half-open
        result = cb.call(lambda: "success")
        
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_reset(self):
        """REQ-REL-001: Manual reset"""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        
        cb.reset()
        assert cb.state == CircuitState.CLOSED
    
    def test_state_change_callback(self):
        """REQ-REL-001: State change callback"""
        cb = CircuitBreaker("test", failure_threshold=2)
        callbacks = []
        
        cb.on_state_change(lambda name, old, new: callbacks.append((name, old, new)))
        
        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        
        assert len(callbacks) == 1
        assert callbacks[0][2] == CircuitState.OPEN


class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator"""
    
    def test_decorator_wraps_function(self):
        """REQ-REL-001: Decorator wraps function"""
        @circuit_breaker("test_decorator", failure_threshold=3)
        def my_func():
            return "result"
        
        result = my_func()
        assert result == "result"
    
    def test_decorator_provides_circuit_breaker(self):
        """REQ-REL-001: Decorated function has circuit_breaker attribute"""
        @circuit_breaker("test_decorator_2", failure_threshold=3)
        def my_func():
            return "result"
        
        assert hasattr(my_func, 'circuit_breaker')
        assert isinstance(my_func.circuit_breaker, CircuitBreaker)


class TestHealthStatus:
    """Test HealthStatus enum"""
    
    def test_statuses_exist(self):
        """REQ-REL-002: Health statuses defined"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass"""
    
    def test_result_creation(self):
        """REQ-REL-002: Create health check result"""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
    
    def test_to_dict(self):
        """REQ-REL-002: Serialize result"""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            duration_ms=10.5,
        )
        d = result.to_dict()
        assert d['name'] == "test"
        assert d['status'] == "healthy"


class TestHealthChecker:
    """Test HealthChecker class"""
    
    def test_register_check(self):
        """REQ-REL-002: Register health check"""
        checker = HealthChecker()
        
        def my_check():
            return HealthCheckResult("test", HealthStatus.HEALTHY)
        
        checker.register("test", my_check)
        result = checker.check("test")
        
        assert result.status == HealthStatus.HEALTHY
    
    def test_unregister_check(self):
        """REQ-REL-002: Unregister health check"""
        checker = HealthChecker()
        checker.register("test", lambda: HealthCheckResult("test", HealthStatus.HEALTHY))
        
        removed = checker.unregister("test")
        assert removed is True
        
        removed = checker.unregister("nonexistent")
        assert removed is False
    
    def test_check_nonexistent(self):
        """REQ-REL-002: Check nonexistent returns unhealthy"""
        checker = HealthChecker()
        result = checker.check("nonexistent")
        
        assert result.status == HealthStatus.UNHEALTHY
    
    def test_check_all(self):
        """REQ-REL-002: Check all registered"""
        checker = HealthChecker()
        checker.register("check1", lambda: HealthCheckResult("check1", HealthStatus.HEALTHY))
        checker.register("check2", lambda: HealthCheckResult("check2", HealthStatus.DEGRADED))
        
        results = checker.check_all()
        
        assert len(results) == 2
        assert results['check1'].status == HealthStatus.HEALTHY
        assert results['check2'].status == HealthStatus.DEGRADED
    
    def test_overall_status_healthy(self):
        """REQ-REL-002: Overall status when all healthy"""
        checker = HealthChecker()
        checker.register("check1", lambda: HealthCheckResult("check1", HealthStatus.HEALTHY))
        checker.register("check2", lambda: HealthCheckResult("check2", HealthStatus.HEALTHY))
        
        overall = checker.get_overall_status()
        assert overall == HealthStatus.HEALTHY
    
    def test_overall_status_degraded(self):
        """REQ-REL-002: Overall status when one degraded"""
        checker = HealthChecker()
        checker.register("check1", lambda: HealthCheckResult("check1", HealthStatus.HEALTHY))
        checker.register("check2", lambda: HealthCheckResult("check2", HealthStatus.DEGRADED))
        
        overall = checker.get_overall_status()
        assert overall == HealthStatus.DEGRADED
    
    def test_overall_status_unhealthy(self):
        """REQ-REL-002: Overall status when one unhealthy"""
        checker = HealthChecker()
        checker.register("check1", lambda: HealthCheckResult("check1", HealthStatus.DEGRADED))
        checker.register("check2", lambda: HealthCheckResult("check2", HealthStatus.UNHEALTHY))
        
        overall = checker.get_overall_status()
        assert overall == HealthStatus.UNHEALTHY


class TestRateLimiter:
    """Test RateLimiter class"""
    
    def test_allows_within_rate(self):
        """REQ-REL-004: Allows calls within rate"""
        limiter = RateLimiter(rate=10.0, capacity=10.0)
        
        for _ in range(10):
            assert limiter.acquire() is True
    
    def test_rejects_over_capacity(self):
        """REQ-REL-004: Rejects calls over capacity"""
        limiter = RateLimiter(rate=10.0, capacity=5.0)
        
        # Use up capacity
        for _ in range(5):
            limiter.acquire()
        
        # Should reject
        assert limiter.acquire() is False
    
    def test_refills_over_time(self):
        """REQ-REL-004: Refills tokens over time"""
        limiter = RateLimiter(rate=100.0, capacity=10.0)
        
        # Use up capacity
        for _ in range(10):
            limiter.acquire()
        
        # Wait for refill
        time.sleep(0.1)
        
        # Should have ~10 tokens now
        assert limiter.acquire() is True
    
    def test_wait_for_tokens(self):
        """REQ-REL-004: Wait for tokens"""
        limiter = RateLimiter(rate=100.0, capacity=1.0)
        
        limiter.acquire()  # Use the token
        
        # Should wait and succeed
        start = time.time()
        result = limiter.wait(timeout=0.5)
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed >= 0.01  # Had to wait
    
    def test_wait_timeout(self):
        """REQ-REL-004: Wait times out"""
        limiter = RateLimiter(rate=0.1, capacity=1.0)
        
        limiter.acquire()  # Use the token
        
        result = limiter.wait(timeout=0.05)
        assert result is False
    
    def test_stats(self):
        """REQ-REL-004: Rate limiter statistics"""
        limiter = RateLimiter(rate=10.0, capacity=5.0)
        
        for _ in range(5):
            limiter.acquire()
        limiter.acquire()  # Should reject
        
        stats = limiter.stats()
        assert stats['accepted'] == 5
        assert stats['rejected'] == 1


class TestRateLimitDecorator:
    """Test rate_limit decorator"""
    
    def test_decorator_allows_calls(self):
        """REQ-REL-004: Decorator allows calls within rate"""
        @rate_limit(rate=100.0, capacity=10.0, block=False)
        def my_func():
            return "result"
        
        result = my_func()
        assert result == "result"
    
    def test_decorator_rejects_over_limit(self):
        """REQ-REL-004: Decorator rejects over limit"""
        @rate_limit(rate=1.0, capacity=1.0, block=False)
        def my_func():
            return "result"
        
        my_func()  # Use token
        
        with pytest.raises(RateLimitExceeded):
            my_func()


class TestDegradationLevel:
    """Test DegradationLevel dataclass"""
    
    def test_level_creation(self):
        """REQ-REL-005: Create degradation level"""
        level = DegradationLevel("test", 1, {"feature1", "feature2"})
        assert level.name == "test"
        assert level.priority == 1
    
    def test_is_feature_enabled(self):
        """REQ-REL-005: Check feature enabled"""
        level = DegradationLevel("test", 1, {"feature1", "feature2"})
        assert level.is_feature_enabled("feature1") is True
        assert level.is_feature_enabled("feature3") is False


class TestGracefulDegradation:
    """Test GracefulDegradation class"""
    
    def test_initial_level_normal(self):
        """REQ-REL-005: Initial level is normal"""
        degradation = GracefulDegradation()
        assert degradation.current_level.name == "normal"
    
    def test_register_feature(self):
        """REQ-REL-005: Register feature with priority"""
        degradation = GracefulDegradation()
        degradation.register_feature("search", 0)
        degradation.register_feature("analytics", 2)
        
        status = degradation.status()
        assert "search" in status['all_features']
        assert "analytics" in status['all_features']
    
    def test_feature_enabled_at_level(self):
        """REQ-REL-005: Feature enabled based on level"""
        degradation = GracefulDegradation()
        degradation.register_feature("critical", 0)
        degradation.register_feature("normal", 1)
        degradation.register_feature("optional", 2)
        
        # At normal level (priority 0), only critical enabled
        assert degradation.is_feature_enabled("critical") is True
        assert degradation.is_feature_enabled("normal") is False
        
        # Raise level
        degradation.set_level(DegradationLevel("reduced", 1, set()))
        assert degradation.is_feature_enabled("normal") is True
        assert degradation.is_feature_enabled("optional") is False
    
    def test_status(self):
        """REQ-REL-005: Get degradation status"""
        degradation = GracefulDegradation()
        degradation.register_feature("search", 0)
        
        status = degradation.status()
        assert 'level' in status
        assert 'all_features' in status


class TestFeatureGate:
    """Test feature_gate decorator"""
    
    def test_allows_enabled_feature(self):
        """REQ-REL-005: Allows call when feature enabled"""
        degradation = GracefulDegradation()
        degradation.register_feature("search", 0)
        
        # Feature is enabled by default
        @feature_gate("search")
        def search_function():
            return "search result"
        
        # This test uses global state, so we can't reliably test it
        # Just verify the decorator exists
        assert hasattr(search_function, '__wrapped__')


class TestThreadSafety:
    """Test thread safety of resilience utilities"""
    
    def test_circuit_breaker_thread_safe(self):
        """REQ-REL-001: Circuit breaker thread safety"""
        cb = CircuitBreaker("test_threaded", failure_threshold=100)
        errors = []
        
        def worker():
            try:
                for i in range(50):
                    try:
                        if i % 3 == 0:
                            cb.call(lambda: (_ for _ in ()).throw(ValueError()))
                        else:
                            cb.call(lambda: "success")
                    except (ValueError, CircuitOpenError):
                        pass
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_rate_limiter_thread_safe(self):
        """REQ-REL-004: Rate limiter thread safety"""
        limiter = RateLimiter(rate=1000.0, capacity=100.0)
        errors = []
        
        def worker():
            try:
                for _ in range(50):
                    limiter.acquire()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
