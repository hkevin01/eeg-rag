"""
Unit Tests for Error Handling Module

Tests the error handling utilities including:
- Error codes and their properties
- Exception hierarchy
- Safe execution wrappers
- Retry decorators
- Validation utilities

Requirements Tested:
- REQ-ERR-001: Standardized error codes
- REQ-ERR-002: Structured exception handling
- REQ-REL-002: Retry mechanisms
- REQ-SEC-001: Input validation
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

from eeg_rag.utils.error_handling import (
    ErrorCode,
    EEGRAGError,
    ValidationError,
    RetrievalError,
    CitationError,
    AgentError,
    APIError,
    DatabaseError,
    safe_execute,
    safe_execute_async,
    with_error_handling,
    with_retry,
    validate_not_empty,
    validate_type,
    validate_range,
    validate_string_length,
)


class TestErrorCodes:
    """Test ErrorCode enum and its properties"""
    
    def test_success_code(self):
        """REQ-ERR-002: Success code is 0"""
        assert ErrorCode.SUCCESS == 0
    
    def test_validation_error_codes_in_range(self):
        """REQ-ERR-001: Validation errors in 1000-1999 range"""
        assert 1000 <= ErrorCode.VALIDATION_ERROR < 2000
        assert 1000 <= ErrorCode.EMPTY_QUERY < 2000
        assert 1000 <= ErrorCode.INVALID_FORMAT < 2000
    
    def test_retrieval_error_codes_in_range(self):
        """REQ-ERR-001: Retrieval errors in 2000-2999 range"""
        assert 2000 <= ErrorCode.RETRIEVAL_ERROR < 3000
        assert 2000 <= ErrorCode.INDEX_NOT_FOUND < 3000
        assert 2000 <= ErrorCode.EMBEDDING_FAILED < 3000
    
    def test_citation_error_codes_in_range(self):
        """REQ-ERR-001: Citation errors in 3000-3999 range"""
        assert 3000 <= ErrorCode.CITATION_ERROR < 4000
        assert 3000 <= ErrorCode.PMID_NOT_FOUND < 4000
        assert 3000 <= ErrorCode.CLAIM_NOT_SUPPORTED < 4000
    
    def test_agent_error_codes_in_range(self):
        """REQ-ERR-001: Agent errors in 4000-4999 range"""
        assert 4000 <= ErrorCode.AGENT_ERROR < 5000
        assert 4000 <= ErrorCode.ORCHESTRATION_FAILED < 5000
        assert 4000 <= ErrorCode.AGENT_TIMEOUT < 5000
    
    def test_api_error_codes_in_range(self):
        """REQ-ERR-001: API errors in 5000-5999 range"""
        assert 5000 <= ErrorCode.API_ERROR < 6000
        assert 5000 <= ErrorCode.RATE_LIMITED < 6000
        assert 5000 <= ErrorCode.API_AUTHENTICATION_FAILED < 6000
    
    def test_database_error_codes_in_range(self):
        """REQ-ERR-001: Database errors in 6000-6999 range"""
        assert 6000 <= ErrorCode.DATABASE_ERROR < 7000
        assert 6000 <= ErrorCode.CONNECTION_FAILED < 7000
        assert 6000 <= ErrorCode.QUERY_FAILED < 7000
    
    def test_system_error_codes_in_range(self):
        """REQ-ERR-001: System errors in 7000-7999 range"""
        assert 7000 <= ErrorCode.SYSTEM_ERROR < 8000
        assert 7000 <= ErrorCode.OUT_OF_MEMORY < 8000
        assert 7000 <= ErrorCode.CONFIGURATION_ERROR < 8000
    
    def test_error_code_is_unique(self):
        """All error codes should be unique"""
        all_codes = [e.value for e in ErrorCode]
        assert len(all_codes) == len(set(all_codes))


class TestEEGRAGError:
    """Test base exception class"""
    
    def test_create_error(self):
        """REQ-ERR-002: Create base error"""
        error = EEGRAGError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Test error"
        )
        assert "Test error" in str(error)
        assert error.code == ErrorCode.SYSTEM_ERROR
    
    def test_error_with_context(self):
        """REQ-ERR-002: Error with additional context"""
        error = EEGRAGError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Test error",
            context={"key": "value"}
        )
        assert error.context == {"key": "value"}
    
    def test_error_to_dict(self):
        """REQ-ERR-002: Error serialization"""
        error = EEGRAGError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Test error",
            context={"key": "value"}
        )
        error_dict = error.to_dict()
        assert error_dict["error_code"] == ErrorCode.SYSTEM_ERROR.value
        assert error_dict["error_name"] == "SYSTEM_ERROR"
        assert error_dict["message"] == "Test error"
    
    def test_error_recoverable_flag(self):
        """REQ-ERR-002: Error recoverable flag"""
        recoverable = EEGRAGError(
            code=ErrorCode.SEARCH_TIMEOUT,
            message="Timeout",
            recoverable=True
        )
        assert recoverable.recoverable is True
        
        unrecoverable = EEGRAGError(
            code=ErrorCode.CONFIGURATION_ERROR,
            message="Bad config",
            recoverable=False
        )
        assert unrecoverable.recoverable is False
    
    def test_error_user_message(self):
        """REQ-SEC-001: User-safe error messages"""
        error = EEGRAGError(
            code=ErrorCode.EMPTY_QUERY,
            message="Technical details here",
            user_message="Please enter a query"
        )
        assert "Technical" in error.message
        assert "Please enter" in error.user_message


class TestSpecializedExceptions:
    """Test specialized exception classes"""
    
    def test_validation_error(self):
        """REQ-ERR-002: ValidationError creation"""
        error = ValidationError(
            message="Invalid query",
            field="query",
            value=""
        )
        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.field == "query"
    
    def test_retrieval_error(self):
        """REQ-ERR-002: RetrievalError creation"""
        error = RetrievalError(
            message="Index not found",
            query="alpha waves",
            source="local"
        )
        assert error.code == ErrorCode.RETRIEVAL_ERROR
        assert "query_preview" in error.context
    
    def test_citation_error(self):
        """REQ-ERR-002: CitationError creation"""
        error = CitationError(
            message="Invalid PMID",
            pmid="12345"
        )
        assert error.code == ErrorCode.CITATION_ERROR
        assert error.pmid == "12345"
    
    def test_agent_error(self):
        """REQ-ERR-002: AgentError creation"""
        error = AgentError(
            message="Agent timeout",
            agent_name="pubmed_agent"
        )
        assert error.code == ErrorCode.AGENT_ERROR
        assert error.agent_name == "pubmed_agent"
    
    def test_api_error(self):
        """REQ-ERR-002: APIError creation"""
        error = APIError(
            message="Rate limited",
            api_name="pubmed",
            status_code=429
        )
        assert error.code == ErrorCode.API_ERROR
        assert error.status_code == 429
        assert error.api_name == "pubmed"
    
    def test_database_error(self):
        """REQ-ERR-002: DatabaseError creation"""
        error = DatabaseError(
            message="Connection failed",
            operation="insert"
        )
        assert error.code == ErrorCode.DATABASE_ERROR
        assert "operation" in error.context


class TestSafeExecute:
    """Test safe_execute wrapper"""
    
    def test_successful_execution(self):
        """REQ-ERR-002: Successful execution returns result"""
        def add(a, b):
            return a + b
        
        result = safe_execute(add, 2, 3)
        assert result == 5
    
    def test_failed_execution_returns_default(self):
        """REQ-ERR-002: Failed execution returns default"""
        def fail():
            raise ValueError("Test error")
        
        result = safe_execute(fail, default="fallback")
        assert result == "fallback"
    
    def test_failed_execution_returns_none_by_default(self):
        """REQ-ERR-002: Failed execution returns None if no default"""
        def fail():
            raise ValueError("Test error")
        
        result = safe_execute(fail)
        assert result is None
    
    def test_on_error_callback(self):
        """REQ-ERR-002: Error callback is invoked"""
        errors_caught = []
        
        def fail():
            raise ValueError("Test error")
        
        def on_error(e):
            errors_caught.append(e)
        
        safe_execute(fail, on_error=on_error)
        assert len(errors_caught) == 1
        assert isinstance(errors_caught[0], ValueError)


class TestSafeExecuteAsync:
    """Test safe_execute_async wrapper"""
    
    @pytest.mark.asyncio
    async def test_successful_async_execution(self):
        """REQ-ERR-002: Successful async execution returns result"""
        async def async_add(a, b):
            return a + b
        
        result = await safe_execute_async(async_add, 2, 3)
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_failed_async_execution_returns_default(self):
        """REQ-ERR-002: Failed async execution returns default"""
        async def async_fail():
            raise ValueError("Async test error")
        
        result = await safe_execute_async(async_fail, default="fallback")
        assert result == "fallback"
    
    @pytest.mark.asyncio
    async def test_async_on_error_callback(self):
        """REQ-ERR-002: Async error callback is invoked"""
        errors_caught = []
        
        async def async_fail():
            raise ValueError("Test error")
        
        def on_error(e):
            errors_caught.append(e)
        
        await safe_execute_async(async_fail, on_error=on_error)
        assert len(errors_caught) == 1


class TestWithErrorHandling:
    """Test with_error_handling decorator"""
    
    def test_decorator_on_success(self):
        """REQ-ERR-002: Decorator passes through on success"""
        @with_error_handling()
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    def test_decorator_converts_to_eeg_rag_error(self):
        """REQ-ERR-002: Decorator converts exceptions to EEGRAGError"""
        @with_error_handling(reraise=True)
        def failing_func():
            raise ValueError("Should be converted")
        
        with pytest.raises(EEGRAGError):
            failing_func()
    
    def test_decorator_passes_through_eeg_rag_error(self):
        """REQ-ERR-002: Decorator passes through EEGRAGError"""
        @with_error_handling(reraise=True)
        def failing_func():
            raise ValidationError(message="Already correct type")
        
        with pytest.raises(ValidationError):
            failing_func()


class TestWithRetry:
    """Test with_retry decorator"""
    
    def test_successful_first_attempt(self):
        """REQ-REL-002: No retry needed on success"""
        call_count = 0
        
        @with_retry(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """REQ-REL-002: Retry until success"""
        call_count = 0
        
        @with_retry(max_attempts=3, delay_seconds=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """REQ-REL-002: Fail after max retries"""
        call_count = 0
        
        @with_retry(max_attempts=3, delay_seconds=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(EEGRAGError):
            always_fails()
        
        assert call_count == 3
    
    def test_retry_specific_exceptions(self):
        """REQ-REL-002: Only retry specified exceptions"""
        call_count = 0
        
        @with_retry(max_attempts=3, retryable_exceptions=(ConnectionError,), delay_seconds=0.01)
        def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retriable")
        
        with pytest.raises(TypeError):
            fails_with_type_error()
        
        # Should not retry on TypeError
        assert call_count == 1


class TestValidateNotEmpty:
    """Test validate_not_empty utility"""
    
    def test_valid_string(self):
        """REQ-SEC-001: Valid non-empty string passes"""
        validate_not_empty("valid", "test_field")  # Should not raise
    
    def test_empty_string_raises(self):
        """REQ-SEC-001: Empty string raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            validate_not_empty("", "test_field")
        assert exc_info.value.field == "test_field"
    
    def test_whitespace_only_raises(self):
        """REQ-SEC-001: Whitespace-only string raises error"""
        with pytest.raises(ValidationError):
            validate_not_empty("   ", "test_field")
    
    def test_none_raises(self):
        """REQ-SEC-001: None value raises error"""
        with pytest.raises(ValidationError):
            validate_not_empty(None, "test_field")
    
    def test_empty_list_raises(self):
        """REQ-SEC-001: Empty list raises error"""
        with pytest.raises(ValidationError):
            validate_not_empty([], "test_field")
    
    def test_non_empty_list_passes(self):
        """REQ-SEC-001: Non-empty list passes"""
        validate_not_empty([1, 2, 3], "test_field")  # Should not raise


class TestValidateType:
    """Test validate_type utility"""
    
    def test_correct_type(self):
        """REQ-SEC-001: Correct type passes"""
        validate_type("test", str, "test_field")  # Should not raise
    
    def test_wrong_type_raises(self):
        """REQ-SEC-001: Wrong type raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            validate_type(123, str, "test_field")
        assert "str" in str(exc_info.value).lower()
    
    def test_multiple_types(self):
        """REQ-SEC-001: Multiple accepted types"""
        validate_type(123, (int, float), "test_field")  # Should not raise
        validate_type(12.5, (int, float), "test_field")  # Should not raise
    
    def test_subclass_accepted(self):
        """REQ-SEC-001: Subclass types are accepted"""
        validate_type(True, int, "test_field")  # bool is subclass of int


class TestValidateRange:
    """Test validate_range utility"""
    
    def test_value_in_range(self):
        """REQ-SEC-001: Value in range passes"""
        validate_range(5, 0, 10, "test_field")  # Should not raise
    
    def test_value_at_min_boundary(self):
        """REQ-SEC-001: Value at min boundary passes"""
        validate_range(0, 0, 10, "test_field")  # Should not raise
    
    def test_value_at_max_boundary(self):
        """REQ-SEC-001: Value at max boundary passes"""
        validate_range(10, 0, 10, "test_field")  # Should not raise
    
    def test_value_below_min_raises(self):
        """REQ-SEC-001: Value below min raises error"""
        with pytest.raises(ValidationError):
            validate_range(-1, 0, 10, "test_field")
    
    def test_value_above_max_raises(self):
        """REQ-SEC-001: Value above max raises error"""
        with pytest.raises(ValidationError):
            validate_range(11, 0, 10, "test_field")
    
    def test_float_range(self):
        """REQ-SEC-001: Float values work correctly"""
        validate_range(0.5, 0.0, 1.0, "test_field")  # Should not raise
    
    def test_open_ended_min(self):
        """REQ-SEC-001: Open-ended min (None)"""
        validate_range(-100, None, 10, "test_field")  # Should not raise
    
    def test_open_ended_max(self):
        """REQ-SEC-001: Open-ended max (None)"""
        validate_range(1000, 0, None, "test_field")  # Should not raise


class TestValidateStringLength:
    """Test validate_string_length utility"""
    
    def test_valid_length(self):
        """REQ-SEC-001: String within length limits passes"""
        validate_string_length("test", 1, 10, "test_field")  # Should not raise
    
    def test_too_short_raises(self):
        """REQ-SEC-001: Too short string raises error"""
        with pytest.raises(ValidationError):
            validate_string_length("a", 3, 10, "test_field")
    
    def test_too_long_raises(self):
        """REQ-SEC-001: Too long string raises error"""
        with pytest.raises(ValidationError):
            validate_string_length("a" * 100, 1, 10, "test_field")
    
    def test_exact_min_length(self):
        """REQ-SEC-001: Exact min length passes"""
        validate_string_length("abc", 3, 10, "test_field")  # Should not raise
    
    def test_exact_max_length(self):
        """REQ-SEC-001: Exact max length passes"""
        validate_string_length("abcdefghij", 1, 10, "test_field")  # Should not raise


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_error_code_boundary_values(self):
        """Test error codes at category boundaries"""
        assert ErrorCode.VALIDATION_ERROR.value >= 1000
        assert ErrorCode.RETRIEVAL_ERROR.value >= 2000
        assert ErrorCode.CITATION_ERROR.value >= 3000
    
    def test_concurrent_safe_execute(self):
        """REQ-ERR-002: Safe execute is thread-safe"""
        import concurrent.futures
        
        def task(n):
            if n % 2 == 0:
                return n * 2
            raise ValueError(f"Odd number: {n}")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(safe_execute, task, i, default="error") for i in range(10)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        assert len(results) == 10
        successful = [r for r in results if r != "error"]
        failed = [r for r in results if r == "error"]
        assert len(successful) == 5
        assert len(failed) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_async_safe_execute(self):
        """REQ-ERR-002: Async safe execute handles concurrent calls"""
        async def async_task(n):
            await asyncio.sleep(0.01)
            if n % 2 == 0:
                return n * 2
            raise ValueError(f"Odd number: {n}")
        
        tasks = [safe_execute_async(async_task, i, default="error") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        successful = [r for r in results if r != "error"]
        failed = [r for r in results if r == "error"]
        assert len(successful) == 5
        assert len(failed) == 5


class TestIntegration:
    """Integration tests for error handling"""
    
    def test_validation_in_retry_context(self):
        """REQ-ERR-002, REQ-REL-002: Validation in retry context"""
        call_count = 0
        
        @with_retry(max_attempts=2, delay_seconds=0.01, retryable_exceptions=(ValueError,))
        def validated_operation(query: str):
            nonlocal call_count
            call_count += 1
            validate_not_empty(query, "query")
            return query.upper()
        
        # Valid input should work
        result = validated_operation("test")
        assert result == "TEST"
        assert call_count == 1
    
    def test_error_hierarchy_isinstance(self):
        """REQ-ERR-002: Error hierarchy supports isinstance checks"""
        errors = [
            ValidationError(message="msg"),
            RetrievalError(message="msg"),
            CitationError(message="msg"),
            AgentError(message="msg"),
            APIError(message="msg"),
            DatabaseError(message="msg"),
        ]
        
        for error in errors:
            assert isinstance(error, EEGRAGError)
            assert isinstance(error, Exception)
    
    def test_error_chaining(self):
        """REQ-ERR-002: Error cause chaining works"""
        original = ValueError("Original error")
        eeg_error = EEGRAGError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Wrapped error",
            cause=original
        )
        assert eeg_error.cause is original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
