"""
Unit tests for common utilities module.

Tests cover:
- Input validation functions
- Time unit standardization 
- Safe mathematical operations
- Error handling mechanisms
- Retry logic
- Database operation wrappers
- File system operations

All tests include both nominal and off-nominal scenarios.
"""

import unittest
import asyncio
import tempfile
import sqlite3
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    validate_range,
    standardize_time_unit,
    safe_divide,
    safe_get_nested,
    compute_content_hash,
    retry_with_backoff,
    handle_database_operation,
    ensure_directory_exists,
    format_error_message,
    format_duration_human_readable,
    SECOND, MINUTE, HOUR, MILLISECOND
)


class TestStringValidation(unittest.TestCase):
    """Test string validation functions"""

    def test_validate_non_empty_string_valid(self):
        """Test valid string validation"""
        # Nominal cases
        self.assertEqual(validate_non_empty_string("hello"), "hello")
        self.assertEqual(validate_non_empty_string("  hello  "), "hello")
        self.assertEqual(validate_non_empty_string("hello world", "query"), "hello world")
        
    def test_validate_non_empty_string_empty(self):
        """Test empty string validation"""
        # Off-nominal cases - empty strings
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string("")
        self.assertIn("cannot be empty", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string("   ")
        self.assertIn("cannot be empty", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string("\n\t  ")
        self.assertIn("cannot be empty", str(cm.exception))
    
    def test_validate_non_empty_string_none(self):
        """Test None string validation"""
        # Off-nominal cases - None values
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string(None)
        self.assertIn("cannot be None", str(cm.exception))
        
        # Test allow_none option
        self.assertEqual(validate_non_empty_string(None, allow_none=True), "")
    
    def test_validate_non_empty_string_wrong_type(self):
        """Test wrong type validation"""
        # Off-nominal cases - wrong types
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string(123)
        self.assertIn("must be a string", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string(["hello"])
        self.assertIn("must be a string", str(cm.exception))
    
    def test_validate_non_empty_string_custom_field_name(self):
        """Test custom field name in error messages"""
        with self.assertRaises(ValueError) as cm:
            validate_non_empty_string("", "query")
        self.assertIn("Value cannot be empty", str(cm.exception))


class TestNumericValidation(unittest.TestCase):
    """Test numeric validation functions"""

    def test_validate_positive_number_valid(self):
        """Test valid positive number validation"""
        # Nominal cases
        self.assertEqual(validate_positive_number(5), 5)
        self.assertEqual(validate_positive_number(3.14), 3.14)
        self.assertEqual(validate_positive_number(0.001), 0.001)
        
        # Test allow_zero option
        self.assertEqual(validate_positive_number(0, allow_zero=True), 0)
        
        # Test min_value option
        self.assertEqual(validate_positive_number(5, min_value=3), 5)
    
    def test_validate_positive_number_invalid(self):
        """Test invalid positive number validation"""
        # Off-nominal cases - negative numbers
        with self.assertRaises(ValueError) as cm:
            validate_positive_number(-1)
        self.assertIn("must be positive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_positive_number(-3.14)
        self.assertIn("must be positive", str(cm.exception))
        
        # Zero without allow_zero
        with self.assertRaises(ValueError) as cm:
            validate_positive_number(0)
        self.assertIn("must be positive", str(cm.exception))
    
    def test_validate_positive_number_none(self):
        """Test None number validation"""
        with self.assertRaises(ValueError) as cm:
            validate_positive_number(None)
        self.assertIn("cannot be None", str(cm.exception))
    
    def test_validate_positive_number_wrong_type(self):
        """Test wrong type validation"""
        with self.assertRaises(ValueError) as cm:
            validate_positive_number("5")
        self.assertIn("must be a number", str(cm.exception))
    
    def test_validate_positive_number_min_value(self):
        """Test minimum value constraint"""
        # Valid with min_value
        self.assertEqual(validate_positive_number(10, min_value=5), 10)
        
        # Invalid with min_value
        with self.assertRaises(ValueError) as cm:
            validate_positive_number(3, min_value=5)
        self.assertIn("must be >= 5", str(cm.exception))

    def test_validate_range_valid(self):
        """Test valid range validation"""
        # Nominal cases - within range
        self.assertEqual(validate_range(5, 1, 10), 5)
        self.assertEqual(validate_range(3.14, 0, 5), 3.14)
        self.assertEqual(validate_range(1, 1, 10), 1)  # Min boundary
        self.assertEqual(validate_range(10, 1, 10), 10)  # Max boundary
    
    def test_validate_range_invalid(self):
        """Test invalid range validation"""
        # Off-nominal cases - out of range
        with self.assertRaises(ValueError) as cm:
            validate_range(15, 1, 10)
        self.assertIn("must be between 1 and 10", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_range(-5, 1, 10)
        self.assertIn("must be between 1 and 10", str(cm.exception))
    
    def test_validate_range_exclusive(self):
        """Test exclusive range validation"""
        # Exclusive range
        self.assertEqual(validate_range(5, 1, 10, inclusive=False), 5)
        
        # Boundaries should fail with exclusive
        with self.assertRaises(ValueError) as cm:
            validate_range(1, 1, 10, inclusive=False)
        self.assertIn("exclusive", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            validate_range(10, 1, 10, inclusive=False)
        self.assertIn("exclusive", str(cm.exception))


class TestTimeUtilities(unittest.TestCase):
    """Test time utility functions"""

    def test_standardize_time_unit_valid(self):
        """Test valid time unit conversions"""
        # Nominal cases
        self.assertEqual(standardize_time_unit(1, "seconds"), 1.0)
        self.assertEqual(standardize_time_unit(1, "minutes"), 60.0)
        self.assertEqual(standardize_time_unit(1, "hours"), 3600.0)
        self.assertEqual(standardize_time_unit(1000, "milliseconds"), 1.0)
        
        # Test abbreviations
        self.assertEqual(standardize_time_unit(1, "s"), 1.0)
        self.assertEqual(standardize_time_unit(1, "m"), 60.0)
        self.assertEqual(standardize_time_unit(1, "h"), 3600.0)
        self.assertEqual(standardize_time_unit(1000, "ms"), 1.0)
    
    def test_standardize_time_unit_invalid(self):
        """Test invalid time unit conversions"""
        # Off-nominal cases - invalid units
        with self.assertRaises(ValueError) as cm:
            standardize_time_unit(1, "invalid")
        self.assertIn("Invalid time unit", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            standardize_time_unit(1, "")
        self.assertIn("Invalid time unit", str(cm.exception))
    
    def test_standardize_time_unit_negative(self):
        """Test negative time values"""
        # Off-nominal case - negative time
        with self.assertRaises(ValueError) as cm:
            standardize_time_unit(-1, "seconds")
        self.assertIn("must be >= 0", str(cm.exception))
    
    def test_format_duration_human_readable(self):
        """Test human-readable duration formatting"""
        # Test various durations
        self.assertEqual(format_duration_human_readable(30.5), "30.50s")
        self.assertEqual(format_duration_human_readable(90), "1m 30.00s")
        self.assertEqual(format_duration_human_readable(3661), "1h 1m 1.00s")
        self.assertEqual(format_duration_human_readable(0.5, 1), "0.5s")


class TestSafeOperations(unittest.TestCase):
    """Test safe operation utilities"""

    def test_safe_divide_normal(self):
        """Test normal division operations"""
        # Nominal cases
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(7, 3), 7/3)
        self.assertEqual(safe_divide(0, 5), 0.0)
    
    def test_safe_divide_zero_denominator(self):
        """Test division by zero handling"""
        # Off-nominal case - division by zero
        with patch('logging.warning') as mock_warning:
            result = safe_divide(10, 0, default=999)
            self.assertEqual(result, 999.0)
            mock_warning.assert_called_once()
            self.assertIn("Division by zero", mock_warning.call_args[0][0])
    
    def test_safe_get_nested_valid(self):
        """Test valid nested dictionary access"""
        # Nominal cases
        data = {
            "level1": {
                "level2": {
                    "value": "found"
                }
            }
        }
        
        self.assertEqual(
            safe_get_nested(data, ["level1", "level2", "value"]), 
            "found"
        )
        self.assertEqual(safe_get_nested(data, ["level1"]), {"level2": {"value": "found"}})
    
    def test_safe_get_nested_missing(self):
        """Test missing keys in nested access"""
        # Off-nominal cases - missing keys
        data = {"level1": {"level2": {"value": "found"}}}
        
        self.assertEqual(
            safe_get_nested(data, ["missing"], "default"), 
            "default"
        )
        self.assertEqual(
            safe_get_nested(data, ["level1", "missing"], "default"), 
            "default"
        )
        self.assertIsNone(
            safe_get_nested(data, ["level1", "missing"])
        )
    
    def test_safe_get_nested_invalid_input(self):
        """Test invalid input for nested access"""
        # Off-nominal cases - invalid input types
        self.assertEqual(safe_get_nested("not a dict", ["key"]), None)
        self.assertEqual(safe_get_nested(None, ["key"]), None)
        self.assertEqual(safe_get_nested([], ["key"]), None)


class TestContentHashing(unittest.TestCase):
    """Test content hashing utilities"""

    def test_compute_content_hash_md5(self):
        """Test MD5 content hashing"""
        # Nominal cases
        hash1 = compute_content_hash("hello world")
        hash2 = compute_content_hash("hello world")
        hash3 = compute_content_hash("different content")
        
        # Same content should produce same hash
        self.assertEqual(hash1, hash2)
        # Different content should produce different hash
        self.assertNotEqual(hash1, hash3)
        # Should be valid hex
        self.assertEqual(len(hash1), 32)  # MD5 produces 32 char hex
    
    def test_compute_content_hash_sha256(self):
        """Test SHA256 content hashing"""
        hash1 = compute_content_hash("hello world", algorithm="sha256")
        hash2 = compute_content_hash("hello world", algorithm="sha256")
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 produces 64 char hex
    
    def test_compute_content_hash_with_prefix(self):
        """Test content hashing with prefix"""
        hash_value = compute_content_hash("test", prefix="chunk")
        self.assertTrue(hash_value.startswith("chunk_"))
        self.assertEqual(len(hash_value), 18)  # "chunk_" + 12 chars
    
    def test_compute_content_hash_invalid(self):
        """Test invalid inputs for content hashing"""
        # Off-nominal cases
        with self.assertRaises(ValueError):
            compute_content_hash("")
        
        with self.assertRaises(ValueError):
            compute_content_hash("test", algorithm="invalid")
        
        with self.assertRaises(ValueError):
            compute_content_hash("   ")


class TestRetryMechanism(unittest.TestCase):
    """Test retry mechanism decorator"""

    def test_retry_success_first_attempt(self):
        """Test successful operation on first attempt"""
        @retry_with_backoff(max_retries=3)
        def successful_operation():
            return "success"
        
        result = successful_operation()
        self.assertEqual(result, "success")
    
    def test_retry_success_after_failures(self):
        """Test successful operation after some failures"""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return "success"
        
        result = flaky_operation()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
    
    def test_retry_final_failure(self):
        """Test operation that fails all retries"""
        @retry_with_backoff(max_retries=2, initial_delay=0.1)
        def always_fails():
            raise ValueError("permanent failure")
        
        with self.assertRaises(ValueError) as cm:
            always_fails()
        self.assertIn("permanent failure", str(cm.exception))
    
    def test_retry_async_function(self):
        """Test retry with async function"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, initial_delay=0.1)
        async def async_flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary failure")
            return "async success"
        
        async def run_test():
            result = await async_flaky_operation()
            return result
        
        result = asyncio.run(run_test())
        self.assertEqual(result, "async success")
        self.assertEqual(call_count, 2)


class TestDatabaseOperations(unittest.TestCase):
    """Test database operation utilities"""

    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        
        # Create test database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    
    def tearDown(self):
        """Clean up test database"""
        self.temp_dir.cleanup()
    
    def test_handle_database_operation_success(self):
        """Test successful database operation"""
        def insert_operation():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
                conn.commit()
                return "inserted"
        
        result = handle_database_operation(
            insert_operation, 
            self.db_path, 
            "insert test"
        )
        self.assertEqual(result, "inserted")
    
    def test_handle_database_operation_sqlite_error(self):
        """Test database operation with SQLite error"""
        def failing_operation():
            with sqlite3.connect(self.db_path) as conn:
                # This should fail - invalid SQL
                conn.execute("INVALID SQL")
                return "should not reach here"
        
        with self.assertRaises(sqlite3.Error) as cm:
            handle_database_operation(
                failing_operation, 
                self.db_path, 
                "invalid operation"
            )
        self.assertIn("invalid operation failed", str(cm.exception))
    
    def test_handle_database_operation_generic_error(self):
        """Test database operation with generic error"""
        def error_operation():
            raise RuntimeError("unexpected error")
        
        with self.assertRaises(RuntimeError) as cm:
            handle_database_operation(
                error_operation, 
                self.db_path, 
                "error operation"
            )
        self.assertIn("error operation failed with unexpected error", str(cm.exception))


class TestFileSystemOperations(unittest.TestCase):
    """Test file system operation utilities"""

    def setUp(self):
        """Set up test directory"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test directory"""
        self.temp_dir.cleanup()
    
    def test_ensure_directory_exists_new(self):
        """Test creating new directory"""
        new_dir = self.base_path / "new_directory"
        self.assertFalse(new_dir.exists())
        
        result = ensure_directory_exists(new_dir)
        
        self.assertEqual(result, new_dir)
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())
    
    def test_ensure_directory_exists_existing(self):
        """Test with existing directory"""
        existing_dir = self.base_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory_exists(existing_dir)
        
        self.assertEqual(result, existing_dir)
        self.assertTrue(existing_dir.exists())
    
    def test_ensure_directory_exists_file_conflict(self):
        """Test when path exists as file"""
        file_path = self.base_path / "conflicting_file"
        file_path.touch()  # Create as file
        
        with self.assertRaises(OSError) as cm:
            ensure_directory_exists(file_path)
        self.assertIn("not a directory", str(cm.exception))
    
    def test_ensure_directory_exists_nested(self):
        """Test creating nested directories"""
        nested_dir = self.base_path / "level1" / "level2" / "level3"
        
        result = ensure_directory_exists(nested_dir, create_parents=True)
        
        self.assertEqual(result, nested_dir)
        self.assertTrue(nested_dir.exists())
        self.assertTrue(nested_dir.is_dir())


class TestErrorMessageFormatting(unittest.TestCase):
    """Test error message formatting utilities"""

    def test_format_error_message_basic(self):
        """Test basic error message formatting"""
        error = ValueError("test error")
        message = format_error_message("test operation", error)
        
        self.assertIn("Operation 'test operation' failed", message)
        self.assertIn("ValueError: test error", message)
    
    def test_format_error_message_with_context(self):
        """Test error message with context"""
        error = RuntimeError("runtime error")
        context = {"user_id": 123, "operation_id": "abc"}
        
        message = format_error_message(
            "user operation", 
            error, 
            context=context
        )
        
        self.assertIn("user operation", message)
        self.assertIn("RuntimeError", message)
        self.assertIn("user_id=123", message)
        self.assertIn("operation_id=abc", message)
    
    def test_format_error_message_with_traceback(self):
        """Test error message with traceback"""
        error = Exception("test exception")
        
        with patch('traceback.format_exc', return_value="fake traceback"):
            message = format_error_message(
                "traced operation", 
                error, 
                include_traceback=True
            )
        
        self.assertIn("traced operation", message)
        self.assertIn("fake traceback", message)


class TestConstants(unittest.TestCase):
    """Test time constants"""

    def test_time_constants(self):
        """Test time constant values"""
        self.assertEqual(MILLISECOND, 0.001)
        self.assertEqual(SECOND, 1.0)
        self.assertEqual(MINUTE, 60.0)
        self.assertEqual(HOUR, 3600.0)
        
        # Test relationships
        self.assertEqual(MINUTE, 60 * SECOND)
        self.assertEqual(HOUR, 60 * MINUTE)
        self.assertEqual(SECOND, 1000 * MILLISECOND)


class TestSystemHealth(unittest.TestCase):
    """Test system health monitoring functionality"""

    def test_check_system_health_success(self):
        """Test successful system health check"""
        try:
            from eeg_rag.utils.common_utils import check_system_health, SystemHealth, SystemStatus
        except ImportError:
            self.skipTest("psutil not available - skipping system health tests")
            
        health = check_system_health()
        
        # Basic structure validation
        self.assertIsInstance(health, SystemHealth)
        self.assertIsInstance(health.status, SystemStatus)
        self.assertIsInstance(health.cpu_percent, (int, float))
        self.assertIsInstance(health.memory_percent, (int, float))
        self.assertIsInstance(health.disk_percent, (int, float))
        self.assertIsInstance(health.warnings, list)
        self.assertIsInstance(health.metrics, dict)

    def test_system_health_to_dict(self):
        """Test SystemHealth serialization"""
        try:
            from eeg_rag.utils.common_utils import check_system_health
        except ImportError:
            self.skipTest("psutil not available - skipping system health tests")
            
        health = check_system_health()
        health_dict = health.to_dict()
        
        expected_keys = {
            "status", "cpu_percent", "memory_percent", "disk_percent",
            "timestamp", "warnings", "metrics"
        }
        self.assertEqual(set(health_dict.keys()), expected_keys)


class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Test circuit breaker functionality"""

    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful operations"""
        from eeg_rag.utils.common_utils import CircuitBreaker, CircuitBreakerState
        
        cb = CircuitBreaker("test_success", failure_threshold=3)
        
        async def success_func(x):
            return x * 2
        
        result = await cb.call(success_func, 5)
        self.assertEqual(result, 10)
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 0)

    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset"""
        from eeg_rag.utils.common_utils import CircuitBreaker, CircuitBreakerState
        
        cb = CircuitBreaker("test_reset", failure_threshold=1)
        cb.failure_count = 5
        cb.state = CircuitBreakerState.OPEN
        
        cb.reset()
        
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertIsNone(cb.last_failure_time)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main()