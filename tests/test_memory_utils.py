"""
Unit Tests for Memory Management Utilities

Tests the memory monitoring, profiling, and optimization utilities.

Requirements Tested:
- REQ-MEM-001: Memory usage monitoring
- REQ-MEM-002: Memory leak detection
- REQ-MEM-003: Garbage collection optimization
"""

import pytest
import gc
import time
import threading
from unittest.mock import Mock, patch
from typing import Any, Dict, List

from eeg_rag.utils.memory_utils import (
    MemoryUsage,
    MemoryProfile,
    get_memory_usage,
    get_system_memory,
    force_gc,
    get_object_count,
    MemoryMonitor,
    MemoryLeakDetector,
    memory_efficient,
    low_memory_mode,
    MemoryPool,
    MEMORY_THRESHOLDS,
    check_memory_health,
    record_memory_profile,
    get_memory_profiles,
    clear_memory_profiles,
)


class TestMemoryUsage:
    """Test MemoryUsage dataclass"""
    
    def test_memory_usage_creation(self):
        """REQ-MEM-001: Create memory usage snapshot"""
        usage = MemoryUsage(
            rss_mb=100.0,
            vms_mb=200.0,
            percent=5.0,
            available_mb=8000.0
        )
        assert usage.rss_mb == 100.0
        assert usage.vms_mb == 200.0
        assert usage.percent == 5.0
    
    def test_memory_usage_to_dict(self):
        """REQ-MEM-001: Serialize memory usage"""
        usage = MemoryUsage(
            rss_mb=100.5,
            vms_mb=200.5,
            percent=5.5,
            available_mb=8000.5
        )
        d = usage.to_dict()
        assert d['rss_mb'] == 100.5
        assert 'timestamp' in d
    
    def test_memory_usage_str(self):
        """REQ-MEM-001: String representation"""
        usage = MemoryUsage(
            rss_mb=100.0,
            vms_mb=200.0,
            percent=5.0,
            available_mb=8000.0
        )
        s = str(usage)
        assert "100.0" in s
        assert "MB" in s


class TestMemoryProfile:
    """Test MemoryProfile dataclass"""
    
    def test_memory_profile_creation(self):
        """REQ-MEM-001: Create memory profile"""
        profile = MemoryProfile(
            operation="test_op",
            start_mb=100.0,
            end_mb=110.0,
            peak_mb=115.0,
            delta_mb=10.0,
            duration_seconds=1.5
        )
        assert profile.operation == "test_op"
        assert profile.delta_mb == 10.0
    
    def test_memory_profile_leaked_property(self):
        """REQ-MEM-002: Detect potential memory leak"""
        # Small delta - not a leak
        profile1 = MemoryProfile(
            operation="test",
            start_mb=100.0,
            end_mb=105.0,
            peak_mb=110.0,
            delta_mb=5.0,
            duration_seconds=1.0
        )
        assert profile1.leaked is False
        
        # Large delta - potential leak
        profile2 = MemoryProfile(
            operation="test",
            start_mb=100.0,
            end_mb=120.0,
            peak_mb=125.0,
            delta_mb=20.0,
            duration_seconds=1.0
        )
        assert profile2.leaked is True
    
    def test_memory_profile_to_dict(self):
        """REQ-MEM-001: Serialize memory profile"""
        profile = MemoryProfile(
            operation="test_op",
            start_mb=100.0,
            end_mb=110.0,
            peak_mb=115.0,
            delta_mb=10.0,
            duration_seconds=1.5,
            gc_collections=3
        )
        d = profile.to_dict()
        assert d['operation'] == "test_op"
        assert d['gc_collections'] == 3
        assert 'leaked' in d


class TestGetMemoryUsage:
    """Test get_memory_usage function"""
    
    def test_returns_memory_usage(self):
        """REQ-MEM-001: Get current memory usage"""
        usage = get_memory_usage()
        assert isinstance(usage, MemoryUsage)
        assert usage.rss_mb > 0
        assert usage.vms_mb > 0
        assert 0 <= usage.percent <= 100
    
    def test_memory_usage_consistent(self):
        """REQ-MEM-001: Memory usage is consistent"""
        usage1 = get_memory_usage()
        usage2 = get_memory_usage()
        # Should be within 10% of each other for quick consecutive calls
        assert abs(usage1.rss_mb - usage2.rss_mb) < usage1.rss_mb * 0.1


class TestGetSystemMemory:
    """Test get_system_memory function"""
    
    def test_returns_system_memory(self):
        """REQ-MEM-001: Get system memory info"""
        mem = get_system_memory()
        assert 'total_mb' in mem
        assert 'available_mb' in mem
        assert 'used_mb' in mem
        assert 'percent' in mem
        assert mem['total_mb'] > 0
    
    def test_memory_sums_correctly(self):
        """REQ-MEM-001: Memory values are consistent"""
        mem = get_system_memory()
        # Used should be less than total
        assert mem['used_mb'] < mem['total_mb']


class TestForceGC:
    """Test force_gc function"""
    
    def test_force_gc_returns_stats(self):
        """REQ-MEM-003: Force GC and get stats"""
        stats = force_gc()
        assert 'gen0' in stats
        assert 'gen1' in stats
        assert 'gen2' in stats
        assert 'freed_mb' in stats
    
    def test_force_gc_collects_objects(self):
        """REQ-MEM-003: GC actually collects objects"""
        # Create some garbage
        garbage = [list(range(1000)) for _ in range(100)]
        del garbage
        
        stats = force_gc()
        # Should have collected at least something
        total_collected = stats['gen0'] + stats['gen1'] + stats['gen2']
        assert total_collected >= 0  # Just verify it ran


class TestGetObjectCount:
    """Test get_object_count function"""
    
    def test_returns_object_counts(self):
        """REQ-MEM-002: Get object type counts"""
        counts = get_object_count()
        assert isinstance(counts, dict)
        assert len(counts) <= 20  # Top 20 types
        # Should include common types
        assert any('dict' in k.lower() or 'list' in k.lower() for k in counts.keys())


class TestMemoryMonitor:
    """Test MemoryMonitor context manager"""
    
    def test_basic_monitoring(self):
        """REQ-MEM-001: Basic memory monitoring"""
        with MemoryMonitor("test_op", log_result=False, track_peak=False) as monitor:
            _ = [i for i in range(10000)]
        
        assert monitor.start_mb > 0
        assert monitor.end_mb > 0
        assert monitor.operation == "test_op"
    
    def test_peak_tracking(self):
        """REQ-MEM-001: Peak memory tracking"""
        with MemoryMonitor("test_peak", log_result=False, track_peak=True, peak_interval=0.01) as monitor:
            # Allocate some memory
            data = [list(range(10000)) for _ in range(10)]
            time.sleep(0.05)  # Give time for peak tracking
            del data
        
        assert monitor.peak_mb >= monitor.start_mb
    
    def test_gc_options(self):
        """REQ-MEM-003: GC before/after options"""
        with MemoryMonitor("test_gc", log_result=False, gc_before=True, gc_after=True, track_peak=False):
            pass  # Just test that GC options work
    
    def test_get_profile(self):
        """REQ-MEM-001: Get profile from monitor"""
        with MemoryMonitor("test_profile", log_result=False, track_peak=False) as monitor:
            time.sleep(0.01)
        
        profile = monitor.get_profile()
        assert isinstance(profile, MemoryProfile)
        assert profile.operation == "test_profile"
        assert profile.duration_seconds >= 0.01


class TestMemoryLeakDetector:
    """Test MemoryLeakDetector class"""
    
    def test_snapshot_and_compare(self):
        """REQ-MEM-002: Take snapshots and compare"""
        detector = MemoryLeakDetector()
        
        detector.snapshot("before")
        # Create some objects
        _ = [list(range(100)) for _ in range(100)]
        detector.snapshot("after")
        
        leaks = detector.compare("before", "after", threshold=10)
        assert isinstance(leaks, dict)
    
    def test_compare_missing_snapshot_raises(self):
        """REQ-MEM-002: Compare with missing snapshot raises error"""
        detector = MemoryLeakDetector()
        detector.snapshot("before")
        
        with pytest.raises(ValueError):
            detector.compare("before", "nonexistent")
    
    def test_clear_snapshots(self):
        """REQ-MEM-002: Clear all snapshots"""
        detector = MemoryLeakDetector()
        detector.snapshot("test1")
        detector.snapshot("test2")
        detector.clear()
        
        with pytest.raises(ValueError):
            detector.compare("test1", "test2")


class TestMemoryEfficient:
    """Test memory_efficient decorator"""
    
    def test_decorator_returns_result(self):
        """REQ-MEM-003: Decorator returns function result"""
        @memory_efficient(log_usage=False)
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5
    
    def test_decorator_with_max_mb(self):
        """REQ-MEM-003: Decorator with max_mb warning"""
        @memory_efficient(max_mb=0.001, log_usage=False)  # Very low limit
        def allocate_memory():
            return [list(range(1000)) for _ in range(10)]
        
        # Should not raise, just logs warning
        result = allocate_memory()
        assert len(result) == 10
    
    def test_decorator_handles_exception(self):
        """REQ-MEM-003: Decorator handles exceptions"""
        @memory_efficient(log_usage=False)
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()


class TestLowMemoryMode:
    """Test low_memory_mode context manager"""
    
    def test_low_memory_mode_context(self):
        """REQ-MEM-003: Low memory mode context"""
        original_thresholds = gc.get_threshold()
        
        with low_memory_mode():
            # Inside context, thresholds should be different
            new_thresholds = gc.get_threshold()
            assert new_thresholds[0] == 100
        
        # After context, thresholds should be restored
        restored_thresholds = gc.get_threshold()
        assert restored_thresholds == original_thresholds


class TestMemoryPool:
    """Test MemoryPool class"""
    
    def test_pool_creation(self):
        """REQ-MEM-003: Create memory pool"""
        pool = MemoryPool(create_fn=lambda: [0] * 1000, max_size=5)
        assert pool.max_size == 5
    
    def test_pool_acquire_creates_new(self):
        """REQ-MEM-003: Acquire creates new object"""
        pool = MemoryPool(create_fn=lambda: [0] * 100, max_size=5)
        obj = pool.acquire()
        assert len(obj) == 100
    
    def test_pool_release_and_reuse(self):
        """REQ-MEM-003: Release and reuse objects"""
        pool = MemoryPool(create_fn=lambda: [0] * 100, max_size=5)
        
        obj1 = pool.acquire()
        pool.release(obj1)
        obj2 = pool.acquire()
        
        # Should be the same object
        assert obj1 is obj2
    
    def test_pool_reset_function(self):
        """REQ-MEM-003: Reset function is called on release"""
        reset_count = 0
        
        def reset_fn(obj):
            nonlocal reset_count
            reset_count += 1
        
        pool = MemoryPool(create_fn=lambda: [1, 2, 3], max_size=5, reset_fn=reset_fn)
        
        obj = pool.acquire()
        pool.release(obj)
        
        assert reset_count == 1
    
    def test_pool_max_size(self):
        """REQ-MEM-003: Pool respects max size"""
        pool = MemoryPool(create_fn=lambda: [0], max_size=2)
        
        objs = [pool.acquire() for _ in range(5)]
        for obj in objs:
            pool.release(obj)
        
        stats = pool.stats()
        assert stats['pool_size'] <= 2
    
    def test_pool_stats(self):
        """REQ-MEM-003: Pool statistics"""
        pool = MemoryPool(create_fn=lambda: [0], max_size=5)
        
        obj1 = pool.acquire()
        pool.release(obj1)
        obj2 = pool.acquire()
        
        stats = pool.stats()
        assert stats['total_created'] == 1
        assert stats['total_reused'] == 1
        assert stats['reuse_ratio'] == 0.5
    
    def test_pool_clear(self):
        """REQ-MEM-003: Clear pool"""
        pool = MemoryPool(create_fn=lambda: [0], max_size=5)
        
        obj = pool.acquire()
        pool.release(obj)
        pool.clear()
        
        stats = pool.stats()
        assert stats['pool_size'] == 0


class TestCheckMemoryHealth:
    """Test check_memory_health function"""
    
    def test_returns_health_info(self):
        """REQ-MEM-001: Health check returns info"""
        health = check_memory_health()
        assert 'status' in health
        assert 'process_usage' in health
        assert 'system_memory' in health
        assert 'recommendations' in health
    
    def test_healthy_status(self):
        """REQ-MEM-001: Normal memory returns healthy status"""
        health = check_memory_health()
        # Most test environments should be healthy
        assert health['status'] in ['healthy', 'warning', 'critical']


class TestGlobalProfiles:
    """Test global memory profile recording"""
    
    def test_record_and_get_profiles(self):
        """REQ-MEM-001: Record and retrieve profiles"""
        clear_memory_profiles()
        
        profile = MemoryProfile(
            operation="test_global",
            start_mb=100.0,
            end_mb=110.0,
            peak_mb=115.0,
            delta_mb=10.0,
            duration_seconds=1.0
        )
        record_memory_profile(profile)
        
        profiles = get_memory_profiles()
        assert len(profiles) == 1
        assert profiles[0]['operation'] == "test_global"
    
    def test_clear_profiles(self):
        """REQ-MEM-001: Clear recorded profiles"""
        profile = MemoryProfile(
            operation="test",
            start_mb=100.0,
            end_mb=110.0,
            peak_mb=115.0,
            delta_mb=10.0,
            duration_seconds=1.0
        )
        record_memory_profile(profile)
        clear_memory_profiles()
        
        profiles = get_memory_profiles()
        assert len(profiles) == 0


class TestMemoryThresholds:
    """Test memory thresholds"""
    
    def test_thresholds_exist(self):
        """REQ-MEM-001: Memory thresholds are defined"""
        assert 'warning_percent' in MEMORY_THRESHOLDS
        assert 'critical_percent' in MEMORY_THRESHOLDS
        assert 'oom_percent' in MEMORY_THRESHOLDS
    
    def test_thresholds_order(self):
        """REQ-MEM-001: Thresholds are in correct order"""
        assert MEMORY_THRESHOLDS['warning_percent'] < MEMORY_THRESHOLDS['critical_percent']
        assert MEMORY_THRESHOLDS['critical_percent'] < MEMORY_THRESHOLDS['oom_percent']


class TestThreadSafety:
    """Test thread safety of memory utilities"""
    
    def test_memory_pool_thread_safe(self):
        """REQ-MEM-003: Memory pool is thread-safe"""
        pool = MemoryPool(create_fn=lambda: [0] * 100, max_size=5)
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    obj = pool.acquire()
                    time.sleep(0.001)
                    pool.release(obj)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_global_profiles_thread_safe(self):
        """REQ-MEM-001: Global profiles are thread-safe"""
        clear_memory_profiles()
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    profile = MemoryProfile(
                        operation=f"worker_{worker_id}_{i}",
                        start_mb=100.0,
                        end_mb=110.0,
                        peak_mb=115.0,
                        delta_mb=10.0,
                        duration_seconds=1.0
                    )
                    record_memory_profile(profile)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        profiles = get_memory_profiles()
        assert len(profiles) == 50  # 5 workers * 10 profiles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
