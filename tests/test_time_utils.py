#!/usr/bin/env python3
"""
Unit Tests for Time Utilities Module

Tests for time conversion, formatting, timing, and performance utilities.

Requirements Tested:
    - REQ-TIME-001: Consistent time units across the system
    - REQ-TIME-002: Standard time measurement for performance monitoring
    - REQ-PERF-001: Performance measurement with sub-millisecond precision
    - TEST-UNIT-010: Valid input testing (nominal)
    - TEST-UNIT-020: Invalid input testing (off-nominal)
    - TEST-BOUND-001: Minimum input values
    - TEST-BOUND-002: Maximum input values
"""

import asyncio
import time
import pytest

from eeg_rag.utils.time_utils import (
    TimeUnits,
    Timer,
    TimingStats,
    convert_time,
    format_duration,
    get_utc_timestamp,
    get_unix_timestamp,
    get_unix_timestamp_ms,
    record_timing,
    get_timing_stats,
    get_all_timing_stats,
    clear_timing_stats,
    timed,
    check_latency_threshold,
    LATENCY_THRESHOLDS,
)


class TestTimeUnits:
    """
    Test: REQ-TIME-001 - TimeUnits enum
    Tests for the standardized time units enumeration.
    """
    
    def test_time_units_has_all_required_units(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - all time units exist
        Expected: All required time units are defined
        """
        assert TimeUnits.NANOSECONDS is not None
        assert TimeUnits.MICROSECONDS is not None
        assert TimeUnits.MILLISECONDS is not None
        assert TimeUnits.SECONDS is not None
        assert TimeUnits.MINUTES is not None
        assert TimeUnits.HOURS is not None
    
    def test_time_units_symbols_correct(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - symbols are correct
        Expected: Each unit has the correct symbol
        """
        assert TimeUnits.NANOSECONDS.symbol == "ns"
        assert TimeUnits.MICROSECONDS.symbol == "μs"
        assert TimeUnits.MILLISECONDS.symbol == "ms"
        assert TimeUnits.SECONDS.symbol == "s"
        assert TimeUnits.MINUTES.symbol == "min"
        assert TimeUnits.HOURS.symbol == "hr"
    
    def test_time_units_multipliers_correct(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - multipliers are correct
        Expected: Each unit has the correct seconds multiplier
        """
        assert TimeUnits.NANOSECONDS.seconds_multiplier == 1e-9
        assert TimeUnits.MICROSECONDS.seconds_multiplier == 1e-6
        assert TimeUnits.MILLISECONDS.seconds_multiplier == 1e-3
        assert TimeUnits.SECONDS.seconds_multiplier == 1.0
        assert TimeUnits.MINUTES.seconds_multiplier == 60.0
        assert TimeUnits.HOURS.seconds_multiplier == 3600.0
    
    @pytest.mark.parametrize("input_str,expected", [
        ("ms", TimeUnits.MILLISECONDS),
        ("milliseconds", TimeUnits.MILLISECONDS),
        ("s", TimeUnits.SECONDS),
        ("sec", TimeUnits.SECONDS),
        ("seconds", TimeUnits.SECONDS),
        ("min", TimeUnits.MINUTES),
        ("minutes", TimeUnits.MINUTES),
        ("hr", TimeUnits.HOURS),
        ("hour", TimeUnits.HOURS),
        ("hours", TimeUnits.HOURS),
        ("ns", TimeUnits.NANOSECONDS),
        ("us", TimeUnits.MICROSECONDS),
    ])
    def test_time_units_from_string_valid(self, input_str, expected):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - parse valid unit strings
        Expected: Correct TimeUnits value returned
        """
        result = TimeUnits.from_string(input_str)
        assert result == expected
    
    def test_time_units_from_string_case_insensitive(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - case insensitive parsing
        Expected: Case variations are handled correctly
        """
        assert TimeUnits.from_string("MS") == TimeUnits.MILLISECONDS
        assert TimeUnits.from_string("Seconds") == TimeUnits.SECONDS
        assert TimeUnits.from_string("HOURS") == TimeUnits.HOURS
    
    def test_time_units_from_string_invalid_raises_error(self):
        """
        Test: REQ-TIME-001
        Scenario: Off-nominal - invalid unit string
        Expected: ValueError raised with descriptive message
        """
        with pytest.raises(ValueError) as exc_info:
            TimeUnits.from_string("invalid_unit")
        
        assert "Unknown time unit" in str(exc_info.value)
        assert "invalid_unit" in str(exc_info.value)


class TestConvertTime:
    """
    Test: REQ-TIME-001 - Time conversion
    Tests for convert_time function.
    """
    
    def test_convert_seconds_to_milliseconds(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - seconds to milliseconds
        Expected: Correct conversion
        """
        result = convert_time(2.5, TimeUnits.SECONDS, TimeUnits.MILLISECONDS)
        assert result == 2500.0
    
    def test_convert_milliseconds_to_seconds(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - milliseconds to seconds
        Expected: Correct conversion
        """
        result = convert_time(1500, TimeUnits.MILLISECONDS, TimeUnits.SECONDS)
        assert result == 1.5
    
    def test_convert_hours_to_minutes(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - hours to minutes
        Expected: Correct conversion
        """
        result = convert_time(2.0, TimeUnits.HOURS, TimeUnits.MINUTES)
        assert result == 120.0
    
    def test_convert_same_unit(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - convert to same unit
        Expected: Value unchanged
        """
        result = convert_time(100, TimeUnits.MILLISECONDS, TimeUnits.MILLISECONDS)
        assert result == 100.0
    
    def test_convert_zero_value(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - zero value
        Expected: Zero returned
        """
        result = convert_time(0, TimeUnits.SECONDS, TimeUnits.MILLISECONDS)
        assert result == 0.0
    
    def test_convert_negative_raises_error(self):
        """
        Test: REQ-FUNC-002
        Scenario: Off-nominal - negative value
        Expected: ValueError raised
        """
        with pytest.raises(ValueError) as exc_info:
            convert_time(-5, TimeUnits.SECONDS, TimeUnits.MILLISECONDS)
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_convert_very_small_value(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - very small value
        Expected: Correct precision maintained
        """
        result = convert_time(1, TimeUnits.NANOSECONDS, TimeUnits.SECONDS)
        assert result == 1e-9


class TestFormatDuration:
    """
    Test: REQ-TIME-001 - Duration formatting
    Tests for format_duration function.
    """
    
    def test_format_milliseconds_range(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - format milliseconds
        Expected: Correct ms format
        """
        result = format_duration(0.150)
        assert "150.00ms" == result
    
    def test_format_seconds_range(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - format seconds
        Expected: Correct s format
        """
        result = format_duration(2.5)
        assert "2.50s" == result
    
    def test_format_minutes_range(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - format minutes
        Expected: Correct min format
        """
        result = format_duration(150.0)
        assert "2.50min" == result
    
    def test_format_hours_range(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - format hours
        Expected: Correct hr format
        """
        result = format_duration(7200.0)
        assert "2.00hr" == result
    
    def test_format_microseconds_range(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - format microseconds
        Expected: Correct μs format
        """
        result = format_duration(0.0001)
        assert "μs" in result
    
    def test_format_zero_duration(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - zero duration
        Expected: 0.00ms format
        """
        result = format_duration(0)
        assert "0.00ms" == result
    
    def test_format_negative_duration(self):
        """
        Test: REQ-FUNC-002
        Scenario: Off-nominal - negative duration
        Expected: Negative prefix added
        """
        result = format_duration(-1.5)
        assert result.startswith("-")
    
    def test_format_custom_precision(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - custom precision
        Expected: Specified precision used
        """
        result = format_duration(0.150, precision=3)
        assert "150.000ms" == result
    
    def test_format_no_auto_scale(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - disabled auto-scale
        Expected: Seconds format forced
        """
        result = format_duration(0.150, auto_scale=False)
        assert result.endswith("s")


class TestTimer:
    """
    Test: REQ-TIME-002 - Timer class
    Tests for the Timer context manager.
    """
    
    def test_timer_context_manager(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - context manager usage
        Expected: Elapsed time recorded
        """
        with Timer() as timer:
            time.sleep(0.01)  # Sleep 10ms
        
        assert timer.elapsed >= 0.01
        assert timer.elapsed_ms >= 10.0
    
    def test_timer_manual_start_stop(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - manual start/stop
        Expected: Elapsed time recorded
        """
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        
        assert elapsed >= 0.01
        assert timer.elapsed == elapsed
    
    def test_timer_elapsed_properties(self):
        """
        Test: REQ-PERF-001
        Scenario: Nominal - elapsed time properties
        Expected: Correct unit conversions
        """
        with Timer() as timer:
            time.sleep(0.001)  # 1ms
        
        assert timer.elapsed > 0
        assert timer.elapsed_ms == timer.elapsed * 1000
        assert timer.elapsed_us == timer.elapsed * 1_000_000
        assert timer.elapsed_ns == timer.elapsed * 1_000_000_000
    
    def test_timer_not_started_stop_raises(self):
        """
        Test: REQ-ERR-001
        Scenario: Off-nominal - stop without start
        Expected: RuntimeError raised
        """
        timer = Timer()
        with pytest.raises(RuntimeError) as exc_info:
            timer.stop()
        
        assert "not started" in str(exc_info.value)
    
    def test_timer_with_name(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - named timer
        Expected: Name stored correctly
        """
        timer = Timer(name="test_operation")
        assert timer.name == "test_operation"
    
    def test_timer_is_running_property(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - is_running property
        Expected: Correct state tracking
        """
        timer = Timer()
        assert not timer.is_running
        
        timer.start()
        assert timer.is_running
        
        timer.stop()
        assert not timer.is_running
    
    def test_timer_elapsed_before_stop(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - elapsed while running
        Expected: Current elapsed time returned
        """
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        
        # Get elapsed while still running
        elapsed_running = timer.elapsed
        assert elapsed_running >= 0.01
        assert timer.is_running
    
    def test_timer_not_started_elapsed_zero(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - elapsed without start
        Expected: Zero returned
        """
        timer = Timer()
        assert timer.elapsed == 0.0


class TestTimingStats:
    """
    Test: REQ-PERF-002 - TimingStats class
    Tests for timing statistics collection.
    """
    
    def test_timing_stats_add_sample(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - add samples
        Expected: Samples collected correctly
        """
        stats = TimingStats(name="test_op")
        stats.add_sample(0.1)
        stats.add_sample(0.2)
        stats.add_sample(0.3)
        
        assert stats.count == 3
        assert len(stats.samples) == 3
    
    def test_timing_stats_calculations(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - statistics calculations
        Expected: Correct mean, min, max, total
        """
        stats = TimingStats(name="test_op")
        samples = [0.1, 0.2, 0.3, 0.4, 0.5]
        for s in samples:
            stats.add_sample(s)
        
        assert stats.count == 5
        assert stats.total_seconds == sum(samples)
        assert stats.mean_seconds == 0.3
        assert stats.min_seconds == 0.1
        assert stats.max_seconds == 0.5
    
    def test_timing_stats_percentiles(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - percentile calculations
        Expected: Correct percentiles
        """
        stats = TimingStats(name="test_op")
        # Add 100 samples from 0.01 to 1.0
        for i in range(1, 101):
            stats.add_sample(i / 100)
        
        assert stats.p50 == pytest.approx(0.50, rel=0.05)
        assert stats.p95 == pytest.approx(0.95, rel=0.05)
        assert stats.p99 == pytest.approx(0.99, rel=0.05)
    
    def test_timing_stats_empty(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - empty stats
        Expected: Zero values returned
        """
        stats = TimingStats(name="empty")
        
        assert stats.count == 0
        assert stats.total_seconds == 0.0
        assert stats.mean_seconds == 0.0
        assert stats.min_seconds == 0.0
        assert stats.max_seconds == 0.0
        assert stats.p50 == 0.0
    
    def test_timing_stats_single_sample(self):
        """
        Test: TEST-BOUND-001
        Scenario: Boundary - single sample
        Expected: Sample value for all stats
        """
        stats = TimingStats(name="single")
        stats.add_sample(0.5)
        
        assert stats.count == 1
        assert stats.mean_seconds == 0.5
        assert stats.min_seconds == 0.5
        assert stats.max_seconds == 0.5
        assert stats.p50 == 0.5
    
    def test_timing_stats_to_dict(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - serialization
        Expected: Correct dict structure
        """
        stats = TimingStats(name="test_op")
        stats.add_sample(0.1)
        stats.add_sample(0.2)
        
        result = stats.to_dict()
        
        assert result['name'] == "test_op"
        assert result['count'] == 2
        assert result['unit'] == "ms"
        assert 'mean' in result
        assert 'p95' in result
    
    def test_timing_stats_percentile_out_of_range(self):
        """
        Test: REQ-FUNC-002
        Scenario: Off-nominal - invalid percentile
        Expected: ValueError raised
        """
        stats = TimingStats(name="test")
        stats.add_sample(0.1)
        
        with pytest.raises(ValueError):
            stats.percentile(101)
        
        with pytest.raises(ValueError):
            stats.percentile(-1)


class TestTimingRegistry:
    """
    Test: REQ-PERF-002 - Global timing registry
    Tests for record_timing and related functions.
    """
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_timing_stats()
    
    def test_record_timing_creates_stats(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - record new timing
        Expected: Stats created
        """
        record_timing("test_operation", 0.1)
        
        stats = get_timing_stats("test_operation")
        assert stats is not None
        assert stats.count == 1
    
    def test_record_timing_accumulates(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - record multiple timings
        Expected: Stats accumulate
        """
        record_timing("test_op", 0.1)
        record_timing("test_op", 0.2)
        record_timing("test_op", 0.3)
        
        stats = get_timing_stats("test_op")
        assert stats.count == 3
    
    def test_get_all_timing_stats(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - get all stats
        Expected: All operations returned
        """
        record_timing("op1", 0.1)
        record_timing("op2", 0.2)
        
        all_stats = get_all_timing_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats
    
    def test_clear_timing_stats(self):
        """
        Test: REQ-PERF-002
        Scenario: Nominal - clear stats
        Expected: Registry empty
        """
        record_timing("test_op", 0.1)
        clear_timing_stats()
        
        assert get_timing_stats("test_op") is None
        assert len(get_all_timing_stats()) == 0


class TestTimedDecorator:
    """
    Test: REQ-TIME-002 - @timed decorator
    Tests for the timing decorator.
    """
    
    def test_timed_decorator_sync_function(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - sync function
        Expected: Function executes correctly
        """
        @timed()
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        assert result == "result"
    
    def test_timed_decorator_async_function(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - async function
        Expected: Async function executes correctly
        """
        @timed()
        async def test_func():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = asyncio.run(test_func())
        assert result == "async_result"
    
    def test_timed_decorator_preserves_name(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - function name preserved
        Expected: __name__ attribute preserved
        """
        @timed()
        def my_function():
            pass
        
        assert my_function.__name__ == "my_function"
    
    def test_timed_decorator_with_custom_name(self):
        """
        Test: REQ-TIME-002
        Scenario: Nominal - custom timer name
        Expected: Custom name used for logging
        """
        @timed(name="custom_operation")
        def test_func():
            return "result"
        
        # Function should still work
        result = test_func()
        assert result == "result"


class TestLatencyThresholds:
    """
    Test: REQ-PERF-001 - Latency thresholds
    Tests for check_latency_threshold function.
    """
    
    def test_threshold_within_limit(self):
        """
        Test: REQ-PERF-001
        Scenario: Nominal - within threshold
        Expected: Returns True
        """
        result = check_latency_threshold("local_retrieval", 0.050, warn=False)
        assert result is True
    
    def test_threshold_exceeded(self):
        """
        Test: REQ-PERF-001
        Scenario: Off-nominal - threshold exceeded
        Expected: Returns False
        """
        result = check_latency_threshold("local_retrieval", 0.200, warn=False)
        assert result is False
    
    def test_threshold_exactly_at_limit(self):
        """
        Test: TEST-BOUND-002
        Scenario: Boundary - exactly at threshold
        Expected: Returns True (at limit is ok)
        """
        threshold = LATENCY_THRESHOLDS["local_retrieval"]
        result = check_latency_threshold("local_retrieval", threshold, warn=False)
        assert result is True
    
    def test_unknown_operation(self):
        """
        Test: REQ-ERR-001
        Scenario: Off-nominal - unknown operation
        Expected: Returns True (permissive)
        """
        result = check_latency_threshold("unknown_operation", 100.0, warn=False)
        assert result is True


class TestTimestamps:
    """
    Test: REQ-TIME-001 - Timestamp functions
    Tests for timestamp generation functions.
    """
    
    def test_utc_timestamp_format(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - UTC timestamp format
        Expected: ISO 8601 format
        """
        timestamp = get_utc_timestamp()
        
        # Should be ISO 8601 format with Z suffix
        assert timestamp.endswith("Z")
        assert "T" in timestamp
        assert len(timestamp) == 24  # YYYY-MM-DDTHH:MM:SS.sssZ
    
    def test_unix_timestamp(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - Unix timestamp
        Expected: Current time as float
        """
        before = time.time()
        timestamp = get_unix_timestamp()
        after = time.time()
        
        assert before <= timestamp <= after
    
    def test_unix_timestamp_ms(self):
        """
        Test: REQ-TIME-001
        Scenario: Nominal - Unix timestamp in ms
        Expected: Integer milliseconds
        """
        timestamp = get_unix_timestamp_ms()
        
        assert isinstance(timestamp, int)
        # Should be roughly current time in ms
        assert timestamp > 1700000000000  # After Nov 2023
