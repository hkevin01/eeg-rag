#!/usr/bin/env python3
"""
Comprehensive Undefined Behavior Testing Framework

This module tests various undefined behavior patterns, gotchas, and dangerous
practices across different programming paradigms. Includes A/B testing
functionality for comparing safe vs unsafe implementations.

Concepts covered:
- Sequence point violations (simulated)
- Type conversion dangers  
- Memory layout issues
- Pointer arithmetic simulation
- Concurrency pitfalls
- Integer overflow
- Floating point precision issues
- Mutable defaults
- Late binding closures
- A/B testing framework
"""

import pytest
import asyncio
import threading
import time
import random
import sys
import gc
import weakref
import ctypes
import struct
import decimal
import math
from typing import Any, Dict, List, Callable, Optional, Union, TypeVar
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
from contextlib import contextmanager
import warnings
from enum import Enum
import json
from datetime import datetime, timedelta


# =============================================================================
# A/B Testing Framework
# =============================================================================

class TestVariant(Enum):
    """Test variants for A/B testing"""
    CONTROL = "control"
    TREATMENT = "treatment"
    SAFE = "safe"
    UNSAFE = "unsafe"


@dataclass
class ABTestResult:
    """Result of an A/B test comparison"""
    variant_a: str
    variant_b: str
    performance_a: float
    performance_b: float
    error_rate_a: float
    error_rate_b: float
    reliability_score_a: float
    reliability_score_b: float
    recommendation: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "performance_a": self.performance_a,
            "performance_b": self.performance_b,
            "error_rate_a": self.error_rate_a,
            "error_rate_b": self.error_rate_b,
            "reliability_score_a": self.reliability_score_a,
            "reliability_score_b": self.reliability_score_b,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class ABTester:
    """A/B testing framework for comparing implementations"""
    
    def __init__(self):
        self.results: List[ABTestResult] = []
        self.test_data_cache: Dict[str, Any] = {}
    
    def run_comparison(
        self,
        test_name: str,
        variant_a_func: Callable,
        variant_b_func: Callable,
        test_data: List[Any],
        iterations: int = 100,
        timeout_seconds: float = 10.0
    ) -> ABTestResult:
        """Run A/B comparison between two implementations"""
        
        # Measure variant A
        perf_a, error_rate_a, reliability_a = self._measure_variant(
            variant_a_func, test_data, iterations, timeout_seconds
        )
        
        # Measure variant B
        perf_b, error_rate_b, reliability_b = self._measure_variant(
            variant_b_func, test_data, iterations, timeout_seconds
        )
        
        # Determine recommendation
        recommendation, confidence = self._analyze_results(
            perf_a, error_rate_a, reliability_a,
            perf_b, error_rate_b, reliability_b
        )
        
        result = ABTestResult(
            variant_a="A",
            variant_b="B", 
            performance_a=perf_a,
            performance_b=perf_b,
            error_rate_a=error_rate_a,
            error_rate_b=error_rate_b,
            reliability_score_a=reliability_a,
            reliability_score_b=reliability_b,
            recommendation=recommendation,
            confidence=confidence
        )
        
        self.results.append(result)
        return result
    
    def _measure_variant(
        self,
        func: Callable,
        test_data: List[Any],
        iterations: int,
        timeout_seconds: float
    ) -> tuple[float, float, float]:
        """Measure performance, error rate, and reliability of a variant"""
        times = []
        errors = 0
        
        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                
                # Run with timeout
                with self._timeout_context(timeout_seconds):
                    for data in test_data:
                        func(data)
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
            except Exception:
                errors += 1
                times.append(timeout_seconds)  # Penalty for errors
        
        avg_time = sum(times) / len(times) if times else timeout_seconds
        error_rate = errors / iterations
        reliability = 1.0 - error_rate
        
        return avg_time, error_rate, reliability
    
    @contextmanager
    def _timeout_context(self, seconds: float):
        """Context manager for timeout"""
        # Simple timeout implementation
        start = time.time()
        yield
        if time.time() - start > seconds:
            raise TimeoutError("Operation timed out")
    
    def _analyze_results(
        self,
        perf_a: float, error_a: float, reliability_a: float,
        perf_b: float, error_b: float, reliability_b: float
    ) -> tuple[str, float]:
        """Analyze results and provide recommendation"""
        
        # Weighted scoring
        score_a = (1/perf_a if perf_a > 0 else 0) * 0.4 + reliability_a * 0.6
        score_b = (1/perf_b if perf_b > 0 else 0) * 0.4 + reliability_b * 0.6
        
        diff = abs(score_a - score_b)
        confidence = min(diff * 100, 99.9)
        
        if score_a > score_b:
            recommendation = "Variant A is better"
        elif score_b > score_a:
            recommendation = "Variant B is better"
        else:
            recommendation = "No significant difference"
        
        return recommendation, confidence


# =============================================================================
# Sequence Point Violation Tests (Simulated)
# =============================================================================

class TestSequencePointViolations:
    """Tests for sequence point violation patterns"""
    
    def test_evaluation_order_dependence(self):
        """Test evaluation order dependencies"""
        
        # Simulate C-style undefined behavior: i = i++ + ++i
        # In Python, we simulate this with function calls
        
        class Counter:
            def __init__(self):
                self.value = 0
            
            def post_increment(self):
                old = self.value
                self.value += 1
                return old
            
            def pre_increment(self):
                self.value += 1
                return self.value
        
        # Unsafe: Order-dependent evaluation
        def unsafe_evaluation():
            c = Counter()
            # This simulates undefined behavior
            result = c.post_increment() + c.pre_increment() + c.value
            return result, c.value
        
        # Safe: Explicit ordering
        def safe_evaluation():
            c = Counter()
            a = c.post_increment()  # 0, c.value = 1
            b = c.pre_increment()   # 2, c.value = 2  
            result = a + b + c.value  # 0 + 2 + 2 = 4
            return result, c.value
        
        # Test multiple runs to check consistency
        results_unsafe = [unsafe_evaluation() for _ in range(10)]
        results_safe = [safe_evaluation() for _ in range(10)]
        
        # Safe version should be consistent
        assert all(r == results_safe[0] for r in results_safe)
        
        # Document the issue
        print(f"Unsafe results: {set(results_unsafe)}")
        print(f"Safe results: {set(results_safe)}")
    
    def test_side_effect_ordering(self):
        """Test side effect ordering issues"""
        
        shared_list = []
        
        def side_effect_1():
            shared_list.append("A")
            return 1
        
        def side_effect_2():
            shared_list.append("B")
            return 2
        
        # Unsafe: Order of side effects undefined
        def unsafe_side_effects():
            shared_list.clear()
            # Order of function calls in expression is undefined
            result = side_effect_1() + side_effect_2()
            return result, shared_list.copy()
        
        # Safe: Explicit ordering
        def safe_side_effects():
            shared_list.clear()
            a = side_effect_1()
            b = side_effect_2()
            result = a + b
            return result, shared_list.copy()
        
        safe_result = safe_side_effects()
        unsafe_result = unsafe_side_effects()
        
        # Both should give same numeric result
        assert safe_result[0] == unsafe_result[0] == 3
        
        # But order of side effects is guaranteed only in safe version
        assert safe_result[1] == ["A", "B"]
        print(f"Safe side effect order: {safe_result[1]}")
        print(f"Unsafe side effect order: {unsafe_result[1]}")


# =============================================================================
# Type Conversion Dangers
# =============================================================================

class TestTypeConversionDangers:
    """Tests for dangerous type conversions and coercions"""
    
    def test_implicit_conversions(self):
        """Test dangerous implicit type conversions"""
        
        # Python's dynamic typing can cause surprises
        def unsafe_mixed_operations(a, b):
            """No type checking - dangerous"""
            return a + b
        
        def safe_mixed_operations(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
            """Type-safe operations"""
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError(f"Expected numeric types, got {type(a)} and {type(b)}")
            return a + b
        
        # Test cases that can cause issues
        test_cases = [
            (5, 3),        # int + int = int
            (5.0, 3),      # float + int = float  
            ("5", "3"),    # str + str = str (concatenation)
            ([1, 2], [3]), # list + list = list (concatenation)
            (True, 2),     # bool + int = int (True=1)
        ]
        
        for a, b in test_cases:
            unsafe_result = None
            safe_result = None
            unsafe_error = None
            safe_error = None
            
            try:
                unsafe_result = unsafe_mixed_operations(a, b)
            except Exception as e:
                unsafe_error = e
            
            try:
                safe_result = safe_mixed_operations(a, b)
            except Exception as e:
                safe_error = e
            
            print(f"Input: {a} ({type(a).__name__}) + {b} ({type(b).__name__})")
            print(f"  Unsafe result: {unsafe_result} (error: {unsafe_error})")
            print(f"  Safe result: {safe_result} (error: {safe_error})")
        
        # Demonstrate the danger
        assert unsafe_mixed_operations("5", "3") == "53"  # String concatenation
        assert unsafe_mixed_operations(5, 3) == 8         # Numeric addition
        
        # Safe version prevents string operations
        with pytest.raises(TypeError):
            safe_mixed_operations("5", "3")
    
    def test_precision_loss(self):
        """Test precision loss in type conversions"""
        
        # Unsafe: Precision loss without warning
        def unsafe_precision_conversion(value: float) -> int:
            return int(value)
        
        # Safe: Check for precision loss
        def safe_precision_conversion(value: float) -> int:
            result = int(value)
            if abs(value - result) > 1e-10:
                warnings.warn(f"Precision loss: {value} -> {result}")
            return result
        
        test_values = [3.14159, 42.0, 99.999999, 1e-10]
        
        for val in test_values:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                unsafe_result = unsafe_precision_conversion(val)
                safe_result = safe_precision_conversion(val)
                
                assert unsafe_result == safe_result
                
                if abs(val - unsafe_result) > 1e-10:
                    assert len(w) > 0  # Warning should be issued
                    print(f"Precision loss warning for {val}: {w[0].message}")
    
    def test_overflow_behavior(self):
        """Test integer overflow behavior"""
        
        # Python handles big integers gracefully, but let's simulate issues
        def unsafe_overflow_prone(a: int, b: int) -> int:
            """Simulate system with fixed integer size"""
            # Simulate 32-bit signed integer overflow
            result = a * b
            if result > 2**31 - 1:
                # Wrap around (simulate undefined behavior)
                result = result % (2**32) - 2**31
            return result
        
        def safe_overflow_checked(a: int, b: int) -> int:
            """Check for overflow before operation"""
            if a != 0 and abs(b) > (2**31 - 1) // abs(a):
                raise OverflowError(f"Multiplication {a} * {b} would overflow")
            return a * b
        
        # Test cases that cause overflow in 32-bit systems
        test_cases = [
            (100000, 30000),    # Large numbers
            (2**16, 2**16),     # Powers of 2
            (-2**15, 2**16),    # Mixed signs
        ]
        
        for a, b in test_cases:
            try:
                unsafe_result = unsafe_overflow_prone(a, b)
                print(f"Unsafe {a} * {b} = {unsafe_result}")
            except Exception as e:
                print(f"Unsafe {a} * {b} failed: {e}")
            
            try:
                safe_result = safe_overflow_checked(a, b)
                print(f"Safe {a} * {b} = {safe_result}")
            except Exception as e:
                print(f"Safe {a} * {b} failed: {e}")


# =============================================================================
# Memory Layout Issues
# =============================================================================

class TestMemoryLayoutIssues:
    """Tests for memory layout and alignment issues"""
    
    def test_struct_packing_simulation(self):
        """Simulate struct packing issues"""
        
        # Unsafe: Assume specific memory layout
        class UnsafeStruct:
            def __init__(self, a: int, b: int, c: int):
                self.a = a
                self.b = b  
                self.c = c
            
            def to_bytes_unsafe(self) -> bytes:
                """Assume specific packing"""
                return struct.pack('III', self.a, self.b, self.c)
            
            @classmethod
            def from_bytes_unsafe(cls, data: bytes):
                """Assume specific unpacking"""
                a, b, c = struct.unpack('III', data)
                return cls(a, b, c)
        
        # Safe: Explicit packing control
        class SafeStruct:
            def __init__(self, a: int, b: int, c: int):
                self.a = a
                self.b = b
                self.c = c
            
            def to_bytes_safe(self) -> bytes:
                """Explicit packing with padding control"""
                return struct.pack('=III', self.a, self.b, self.c)  # Native byte order
            
            @classmethod  
            def from_bytes_safe(cls, data: bytes):
                """Explicit unpacking with validation"""
                if len(data) != struct.calcsize('=III'):
                    raise ValueError(f"Expected {struct.calcsize('=III')} bytes, got {len(data)}")
                a, b, c = struct.unpack('=III', data)
                return cls(a, b, c)
        
        # Test serialization/deserialization
        original = (42, 100, 255)
        
        # Unsafe version
        unsafe_obj = UnsafeStruct(*original)
        unsafe_bytes = unsafe_obj.to_bytes_unsafe()
        unsafe_restored = UnsafeStruct.from_bytes_unsafe(unsafe_bytes)
        
        # Safe version
        safe_obj = SafeStruct(*original)
        safe_bytes = safe_obj.to_bytes_safe()
        safe_restored = SafeStruct.from_bytes_safe(safe_bytes)
        
        # Both should work for simple cases
        assert (unsafe_restored.a, unsafe_restored.b, unsafe_restored.c) == original
        assert (safe_restored.a, safe_restored.b, safe_restored.c) == original
        
        print(f"Original: {original}")
        print(f"Unsafe bytes: {unsafe_bytes.hex()}")
        print(f"Safe bytes: {safe_bytes.hex()}")
        
        # Test with invalid data
        with pytest.raises(ValueError):
            SafeStruct.from_bytes_safe(b'\x00\x01\x02')  # Too short
    
    def test_alignment_assumptions(self):
        """Test memory alignment assumptions"""
        
        # Simulate alignment-sensitive operations
        def unsafe_unaligned_access(data: bytes, offset: int) -> int:
            """Unsafe: No alignment checking"""
            return struct.unpack_from('I', data, offset)[0]
        
        def safe_aligned_access(data: bytes, offset: int) -> int:
            """Safe: Check alignment"""
            if offset % 4 != 0:
                raise ValueError(f"Offset {offset} not aligned to 4-byte boundary")
            if offset + 4 > len(data):
                raise ValueError(f"Access beyond buffer: {offset + 4} > {len(data)}")
            return struct.unpack_from('I', data, offset)[0]
        
        # Create test data
        test_data = b'\x00' * 16
        test_data = test_data[:4] + struct.pack('I', 0x12345678) + test_data[8:]
        
        # Test aligned access (should work)
        aligned_offset = 4
        unsafe_result = unsafe_unaligned_access(test_data, aligned_offset)
        safe_result = safe_aligned_access(test_data, aligned_offset)
        assert unsafe_result == safe_result == 0x12345678
        
        # Test unaligned access
        unaligned_offset = 5
        try:
            unsafe_unaligned = unsafe_unaligned_access(test_data, unaligned_offset)
            print(f"Unsafe unaligned result: 0x{unsafe_unaligned:x}")
        except Exception as e:
            print(f"Unsafe unaligned failed: {e}")
        
        # Safe version should reject unaligned access
        with pytest.raises(ValueError, match="not aligned"):
            safe_aligned_access(test_data, unaligned_offset)
    
    def test_endianness_issues(self):
        """Test byte order issues"""
        
        value = 0x12345678
        
        # Unsafe: Assume specific endianness
        def unsafe_to_bytes() -> bytes:
            return struct.pack('I', value)  # Native endianness
        
        def unsafe_from_bytes(data: bytes) -> int:
            return struct.unpack('I', data)[0]
        
        # Safe: Explicit endianness
        def safe_to_bytes_be() -> bytes:
            return struct.pack('>I', value)  # Big endian
        
        def safe_from_bytes_be(data: bytes) -> int:
            return struct.unpack('>I', data)[0]
        
        def safe_to_bytes_le() -> bytes:
            return struct.pack('<I', value)  # Little endian
        
        def safe_from_bytes_le(data: bytes) -> int:
            return struct.unpack('<I', data)[0]
        
        # Test different endianness
        unsafe_bytes = unsafe_to_bytes()
        safe_be_bytes = safe_to_bytes_be()
        safe_le_bytes = safe_to_bytes_le()
        
        print(f"Original value: 0x{value:08x}")
        print(f"Unsafe bytes: {unsafe_bytes.hex()}")
        print(f"Big endian: {safe_be_bytes.hex()}")
        print(f"Little endian: {safe_le_bytes.hex()}")
        
        # Round trip should work
        assert unsafe_from_bytes(unsafe_bytes) == value
        assert safe_from_bytes_be(safe_be_bytes) == value
        assert safe_from_bytes_le(safe_le_bytes) == value
        
        # Cross-endian should fail or give wrong results
        if safe_be_bytes != safe_le_bytes:
            be_as_le = safe_from_bytes_le(safe_be_bytes)
            le_as_be = safe_from_bytes_be(safe_le_bytes)
            print(f"BE interpreted as LE: 0x{be_as_le:08x}")
            print(f"LE interpreted as BE: 0x{le_as_be:08x}")
            assert be_as_le != value or le_as_be != value


# =============================================================================
# Pointer Arithmetic Simulation
# =============================================================================

class TestPointerArithmeticSimulation:
    """Tests simulating pointer arithmetic undefined behavior"""
    
    def test_buffer_overflow_simulation(self):
        """Simulate buffer overflow scenarios"""
        
        # Unsafe: No bounds checking
        class UnsafeBuffer:
            def __init__(self, size: int):
                self.data = [0] * size
                self.size = size
            
            def write_unsafe(self, index: int, value: int):
                """No bounds checking - dangerous"""
                self.data[index] = value  # Can throw IndexError
            
            def read_unsafe(self, index: int) -> int:
                """No bounds checking - dangerous"""
                return self.data[index]  # Can throw IndexError
        
        # Safe: With bounds checking
        class SafeBuffer:
            def __init__(self, size: int):
                self.data = [0] * size
                self.size = size
            
            def write_safe(self, index: int, value: int):
                """With bounds checking"""
                if index < 0 or index >= self.size:
                    raise IndexError(f"Index {index} out of bounds [0, {self.size})")
                self.data[index] = value
            
            def read_safe(self, index: int) -> int:
                """With bounds checking"""
                if index < 0 or index >= self.size:
                    raise IndexError(f"Index {index} out of bounds [0, {self.size})")
                return self.data[index]
        
        # Test with buffer overflow
        size = 10
        unsafe_buf = UnsafeBuffer(size)
        safe_buf = SafeBuffer(size)
        
        # Valid access should work for both
        unsafe_buf.write_unsafe(5, 42)
        safe_buf.write_safe(5, 42)
        assert unsafe_buf.read_unsafe(5) == 42
        assert safe_buf.read_safe(5) == 42
        
        # Invalid access
        invalid_index = 15
        
        # Unsafe version might crash or corrupt memory
        with pytest.raises(IndexError):
            unsafe_buf.write_unsafe(invalid_index, 99)
        
        with pytest.raises(IndexError):
            unsafe_buf.read_unsafe(invalid_index)
        
        # Safe version should cleanly reject
        with pytest.raises(IndexError, match="out of bounds"):
            safe_buf.write_safe(invalid_index, 99)
        
        with pytest.raises(IndexError, match="out of bounds"):
            safe_buf.read_safe(invalid_index)
    
    def test_null_pointer_simulation(self):
        """Simulate null pointer dereference"""
        
        # Simulate pointers with optional values
        def unsafe_dereference(ptr: Optional[str]) -> str:
            """No null checking - dangerous"""
            return ptr.upper()  # AttributeError if None
        
        def safe_dereference(ptr: Optional[str]) -> str:
            """With null checking"""
            if ptr is None:
                raise ValueError("Cannot dereference null pointer")
            return ptr.upper()
        
        # Test valid pointer
        valid_ptr = "hello"
        assert unsafe_dereference(valid_ptr) == "HELLO"
        assert safe_dereference(valid_ptr) == "HELLO"
        
        # Test null pointer
        null_ptr = None
        
        # Unsafe version crashes
        with pytest.raises(AttributeError):
            unsafe_dereference(null_ptr)
        
        # Safe version gives meaningful error
        with pytest.raises(ValueError, match="null pointer"):
            safe_dereference(null_ptr)
    
    def test_dangling_reference_simulation(self):
        """Simulate dangling reference issues"""
        
        def create_dangling_reference():
            """Create reference that becomes invalid"""
            data = [1, 2, 3, 4, 5]
            # Return reference that might become invalid
            return data, id(data)
        
        def safe_copy_data():
            """Create safe copy"""
            data = [1, 2, 3, 4, 5]
            return data.copy(), id(data)
        
        # Create references
        unsafe_ref, unsafe_id = create_dangling_reference()
        safe_ref, safe_id = safe_copy_data()
        
        # Both should work initially
        assert unsafe_ref == [1, 2, 3, 4, 5]
        assert safe_ref == [1, 2, 3, 4, 5]
        
        # Force garbage collection (simulates reference invalidation)
        original_unsafe = unsafe_ref[:]
        original_safe = safe_ref[:]
        
        # Modify through reference
        unsafe_ref.append(6)
        safe_ref.append(6)
        
        # Both should work if still valid
        assert unsafe_ref[-1] == 6
        assert safe_ref[-1] == 6
        
        print(f"Unsafe reference ID: {unsafe_id}")
        print(f"Safe reference ID: {safe_id}")


# =============================================================================
# Concurrency Pitfalls
# =============================================================================

class TestConcurrencyPitfalls:
    """Tests for concurrency-related undefined behavior"""
    
    def test_race_condition_detection(self):
        """Test race condition scenarios"""
        
        # Unsafe: Race condition prone
        class UnsafeCounter:
            def __init__(self):
                self.value = 0
            
            def increment_unsafe(self):
                """Not thread-safe"""
                temp = self.value
                time.sleep(0.001)  # Simulate processing delay
                self.value = temp + 1
        
        # Safe: Thread-safe implementation
        class SafeCounter:
            def __init__(self):
                self.value = 0
                self.lock = threading.Lock()
            
            def increment_safe(self):
                """Thread-safe with lock"""
                with self.lock:
                    temp = self.value
                    time.sleep(0.001)  # Simulate processing delay
                    self.value = temp + 1
        
        # Test concurrent access
        def run_concurrent_test(counter, increment_func, num_threads=10):
            threads = []
            
            for _ in range(num_threads):
                thread = threading.Thread(target=increment_func)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return counter.value
        
        # Test unsafe counter
        unsafe_counter = UnsafeCounter()
        unsafe_result = run_concurrent_test(
            unsafe_counter, unsafe_counter.increment_unsafe
        )
        
        # Test safe counter
        safe_counter = SafeCounter()
        safe_result = run_concurrent_test(
            safe_counter, safe_counter.increment_safe
        )
        
        print(f"Unsafe counter final value: {unsafe_result}")
        print(f"Safe counter final value: {safe_result}")
        
        # Safe version should be exactly 10
        assert safe_result == 10
        
        # Unsafe version might be less than 10 due to race conditions
        # (though this test is probabilistic)
        assert unsafe_result <= 10
    
    def test_deadlock_detection(self):
        """Test deadlock scenarios"""
        
        # Unsafe: Potential deadlock
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        
        def unsafe_function_1():
            """Acquires locks in one order"""
            with lock1:
                time.sleep(0.1)
                with lock2:
                    return "Function 1 completed"
        
        def unsafe_function_2():
            """Acquires locks in opposite order - deadlock risk"""
            with lock2:
                time.sleep(0.1)
                with lock1:
                    return "Function 2 completed"
        
        # Safe: Consistent lock ordering
        def safe_function_1():
            """Always acquires locks in same order"""
            with lock1:
                with lock2:
                    time.sleep(0.1)
                    return "Safe function 1 completed"
        
        def safe_function_2():
            """Always acquires locks in same order"""
            with lock1:
                with lock2:
                    time.sleep(0.1)
                    return "Safe function 2 completed"
        
        # Test safe version (should complete)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(safe_function_1)
            future2 = executor.submit(safe_function_2)
            
            result1 = future1.result(timeout=1.0)
            result2 = future2.result(timeout=1.0)
            
            assert "completed" in result1
            assert "completed" in result2
        
        # Note: We don't test the unsafe version as it would actually deadlock
        print("Deadlock test completed - safe version works correctly")
    
    @pytest.mark.asyncio
    async def test_async_concurrency_issues(self):
        """Test async/await concurrency issues"""
        
        # Unsafe: Shared state without protection
        shared_data = {"count": 0}
        
        async def unsafe_async_increment():
            """Unsafe async increment"""
            for _ in range(100):
                current = shared_data["count"]
                await asyncio.sleep(0.001)  # Yield control
                shared_data["count"] = current + 1
        
        # Safe: Protected shared state
        safe_data = {"count": 0}
        data_lock = asyncio.Lock()
        
        async def safe_async_increment():
            """Safe async increment with lock"""
            for _ in range(100):
                async with data_lock:
                    current = safe_data["count"]
                    await asyncio.sleep(0.001)  # Yield control
                    safe_data["count"] = current + 1
        
        # Test unsafe version
        shared_data["count"] = 0
        await asyncio.gather(*[unsafe_async_increment() for _ in range(3)])
        unsafe_result = shared_data["count"]
        
        # Test safe version
        safe_data["count"] = 0
        await asyncio.gather(*[safe_async_increment() for _ in range(3)])
        safe_result = safe_data["count"]
        
        print(f"Unsafe async result: {unsafe_result}")
        print(f"Safe async result: {safe_result}")
        
        # Safe version should be exactly 300
        assert safe_result == 300
        
        # Unsafe version might be less due to race conditions
        assert unsafe_result <= 300


# =============================================================================
# Floating Point Issues
# =============================================================================

class TestFloatingPointIssues:
    """Tests for floating point precision and comparison issues"""
    
    def test_precision_comparison_issues(self):
        """Test floating point comparison pitfalls"""
        
        # Unsafe: Direct floating point comparison
        def unsafe_float_equals(a: float, b: float) -> bool:
            """Dangerous direct comparison"""
            return a == b
        
        # Safe: Epsilon-based comparison
        def safe_float_equals(a: float, b: float, epsilon: float = 1e-10) -> bool:
            """Safe comparison with tolerance"""
            return abs(a - b) < epsilon
        
        # Test cases that demonstrate precision issues
        test_cases = [
            (0.1 + 0.2, 0.3),           # Classic precision issue
            (1.0, 1.0000000000000002),   # Very close values
            (math.sqrt(2)**2, 2.0),     # Mathematical identity
            (0.0, -0.0),                # Signed zeros
        ]
        
        for a, b in test_cases:
            unsafe_result = unsafe_float_equals(a, b)
            safe_result = safe_float_equals(a, b)
            
            print(f"Comparing {a} and {b}:")
            print(f"  Difference: {abs(a - b)}")
            print(f"  Unsafe equals: {unsafe_result}")
            print(f"  Safe equals: {safe_result}")
            
            # Demonstrate the problem
            if a == 0.1 + 0.2 and b == 0.3:
                assert not unsafe_result  # This fails!
                assert safe_result       # This works
    
    def test_accumulation_errors(self):
        """Test floating point accumulation errors"""
        
        # Unsafe: Naive accumulation
        def unsafe_sum(values: List[float]) -> float:
            """Simple sum - can accumulate errors"""
            total = 0.0
            for value in values:
                total += value
            return total
        
        # Safe: Kahan summation algorithm
        def safe_sum(values: List[float]) -> float:
            """Kahan summation for improved accuracy"""
            total = 0.0
            compensation = 0.0
            
            for value in values:
                y = value - compensation
                t = total + y
                compensation = (t - total) - y
                total = t
            
            return total
        
        # Test with many small values
        small_values = [0.1] * 10000
        expected = 1000.0
        
        unsafe_result = unsafe_sum(small_values)
        safe_result = safe_sum(small_values)
        
        print(f"Expected sum: {expected}")
        print(f"Unsafe sum: {unsafe_result}")
        print(f"Safe sum: {safe_result}")
        print(f"Unsafe error: {abs(unsafe_result - expected)}")
        print(f"Safe error: {abs(safe_result - expected)}")
        
        # Safe version should be more accurate
        assert abs(safe_result - expected) < abs(unsafe_result - expected)
    
    def test_division_by_zero_handling(self):
        """Test division by zero and infinity handling"""
        
        # Unsafe: No special value checking
        def unsafe_divide(a: float, b: float) -> float:
            """No checking for special cases"""
            return a / b
        
        # Safe: Handle special cases
        def safe_divide(a: float, b: float) -> float:
            """Handle infinity and NaN"""
            if b == 0.0:
                if a > 0:
                    return float('inf')
                elif a < 0:
                    return float('-inf')
                else:
                    return float('nan')
            
            result = a / b
            
            # Check for overflow
            if math.isinf(result) and not math.isinf(a) and not math.isinf(b):
                raise OverflowError(f"Division {a}/{b} resulted in overflow")
            
            return result
        
        test_cases = [
            (1.0, 0.0),     # Positive / zero
            (-1.0, 0.0),    # Negative / zero  
            (0.0, 0.0),     # Zero / zero
            (1e308, 1e-308), # Potential overflow
        ]
        
        for a, b in test_cases:
            try:
                unsafe_result = unsafe_divide(a, b)
                print(f"Unsafe {a}/{b} = {unsafe_result}")
            except Exception as e:
                print(f"Unsafe {a}/{b} failed: {e}")
            
            try:
                safe_result = safe_divide(a, b)
                print(f"Safe {a}/{b} = {safe_result}")
            except Exception as e:
                print(f"Safe {a}/{b} failed: {e}")


# =============================================================================
# Mutable Defaults and Late Binding
# =============================================================================

class TestMutableDefaults:
    """Tests for mutable default argument dangers"""
    
    def test_mutable_default_arguments(self):
        """Test the classic mutable default argument pitfall"""
        
        # Unsafe: Mutable default argument
        def unsafe_append_to_list(item, target=[]):
            """Dangerous - default list is shared between calls"""
            target.append(item)
            return target
        
        # Safe: Immutable default with runtime creation
        def safe_append_to_list(item, target=None):
            """Safe - create new list each time"""
            if target is None:
                target = []
            target.append(item)
            return target
        
        # Test the dangerous behavior
        result1 = unsafe_append_to_list("first")
        result2 = unsafe_append_to_list("second")  # Shares same list!
        
        print(f"Unsafe call 1: {result1}")
        print(f"Unsafe call 2: {result2}")
        
        # Both results share the same list
        assert result1 is result2
        assert result1 == ["first", "second"]
        assert result2 == ["first", "second"]
        
        # Test the safe behavior
        safe_result1 = safe_append_to_list("first")
        safe_result2 = safe_append_to_list("second")
        
        print(f"Safe call 1: {safe_result1}")
        print(f"Safe call 2: {safe_result2}")
        
        # Results are independent
        assert safe_result1 is not safe_result2
        assert safe_result1 == ["first"]
        assert safe_result2 == ["second"]
    
    def test_late_binding_closures(self):
        """Test late binding closure issues"""
        
        # Unsafe: Late binding creates unexpected behavior
        def unsafe_create_functions():
            """Creates functions that all reference the same variable"""
            functions = []
            for i in range(3):
                functions.append(lambda: i)  # All closures refer to same 'i'
            return functions
        
        # Safe: Early binding with default arguments
        def safe_create_functions():
            """Creates functions that capture values correctly"""
            functions = []
            for i in range(3):
                functions.append(lambda x=i: x)  # Capture 'i' value
            return functions
        
        # Alternative safe: Using functools.partial
        from functools import partial
        
        def safe_create_functions_partial():
            """Creates functions using partial application"""
            def identity(x):
                return x
            
            functions = []
            for i in range(3):
                functions.append(partial(identity, i))
            return functions
        
        # Test unsafe version
        unsafe_funcs = unsafe_create_functions()
        unsafe_results = [f() for f in unsafe_funcs]
        
        # Test safe versions
        safe_funcs = safe_create_functions()
        safe_results = [f() for f in safe_funcs]
        
        partial_funcs = safe_create_functions_partial()
        partial_results = [f() for f in partial_funcs]
        
        print(f"Unsafe results: {unsafe_results}")
        print(f"Safe results: {safe_results}")
        print(f"Partial results: {partial_results}")
        
        # Unsafe version: all functions return the same value (2)
        assert unsafe_results == [2, 2, 2]
        
        # Safe versions: functions return different values
        assert safe_results == [0, 1, 2]
        assert partial_results == [0, 1, 2]
    
    def test_generator_state_confusion(self):
        """Test generator state sharing issues"""
        
        # Unsafe: Shared generator state
        def unsafe_create_generators():
            """Creates generators that share state"""
            shared_state = [0]
            
            def generator():
                while True:
                    shared_state[0] += 1
                    yield shared_state[0]
            
            return [generator() for _ in range(3)]
        
        # Safe: Independent generator state
        def safe_create_generators():
            """Creates generators with independent state"""
            def make_generator():
                state = 0
                while True:
                    state += 1
                    yield state
            
            return [make_generator() for _ in range(3)]
        
        # Test unsafe version
        unsafe_gens = unsafe_create_generators()
        unsafe_values = [next(gen) for gen in unsafe_gens]
        
        print(f"Unsafe generator values: {unsafe_values}")
        # All generators share state, so values are [1, 2, 3]
        assert unsafe_values == [1, 2, 3]
        
        # Test safe version
        safe_gens = safe_create_generators()
        safe_values = [next(gen) for gen in safe_gens]
        
        print(f"Safe generator values: {safe_values}")
        # Each generator has independent state, so values are [1, 1, 1]
        assert safe_values == [1, 1, 1]


# =============================================================================
# Comprehensive A/B Testing Integration
# =============================================================================

class TestABTestingIntegration:
    """Integration tests for A/B testing framework"""
    
    def test_ab_performance_comparison(self):
        """Test A/B comparison of different implementations"""
        
        ab_tester = ABTester()
        
        # Fast but potentially unsafe implementation
        def variant_a(data):
            return sum(data)  # Simple sum
        
        # Slower but safer implementation  
        def variant_b(data):
            if not isinstance(data, (list, tuple)):
                raise TypeError("Data must be a list or tuple")
            if not all(isinstance(x, (int, float)) for x in data):
                raise TypeError("All elements must be numeric")
            return sum(data)  # Sum with validation
        
        # Test data
        test_data = [
            [1, 2, 3, 4, 5],
            [10, 20, 30],
            [100],
            list(range(100))
        ]
        
        # Run A/B test
        result = ab_tester.run_comparison(
            test_name="sum_performance",
            variant_a_func=variant_a,
            variant_b_func=variant_b,
            test_data=test_data,
            iterations=50
        )
        
        print(f"A/B Test Results:")
        print(f"  Variant A Performance: {result.performance_a:.4f}s")
        print(f"  Variant B Performance: {result.performance_b:.4f}s")
        print(f"  Variant A Error Rate: {result.error_rate_a:.2%}")
        print(f"  Variant B Error Rate: {result.error_rate_b:.2%}")
        print(f"  Recommendation: {result.recommendation}")
        print(f"  Confidence: {result.confidence:.1f}%")
        
        # Both should have low error rates for valid data
        assert result.error_rate_a < 0.1
        assert result.error_rate_b < 0.1
        
        # Variant A should be faster
        assert result.performance_a < result.performance_b
        
        # But both should be reliable
        assert result.reliability_score_a > 0.9
        assert result.reliability_score_b > 0.9
    
    def test_ab_safety_comparison(self):
        """Test A/B comparison focusing on safety"""
        
        ab_tester = ABTester()
        
        # Unsafe: No error handling
        def unsafe_division(data):
            a, b = data
            return a / b
        
        # Safe: With error handling
        def safe_division(data):
            a, b = data
            if b == 0:
                return float('inf') if a > 0 else float('-inf') if a < 0 else float('nan')
            return a / b
        
        # Test data including edge cases
        test_data = [
            (10, 2),    # Normal case
            (5, 0),     # Division by zero
            (0, 0),     # Zero by zero
            (-3, 0),    # Negative by zero
        ]
        
        # Run A/B test
        result = ab_tester.run_comparison(
            test_name="division_safety",
            variant_a_func=unsafe_division,
            variant_b_func=safe_division,
            test_data=test_data,
            iterations=25
        )
        
        print(f"Safety A/B Test Results:")
        print(f"  Unsafe Error Rate: {result.error_rate_a:.2%}")
        print(f"  Safe Error Rate: {result.error_rate_b:.2%}")
        print(f"  Unsafe Reliability: {result.reliability_score_a:.2%}")
        print(f"  Safe Reliability: {result.reliability_score_b:.2%}")
        print(f"  Recommendation: {result.recommendation}")
        
        # Safe version should have lower error rate
        assert result.error_rate_b < result.error_rate_a
        assert result.reliability_score_b > result.reliability_score_a
        
        # Result should recommend safe version
        assert "B is better" in result.recommendation
    
    def test_ab_testing_report_generation(self):
        """Test comprehensive A/B testing report"""
        
        ab_tester = ABTester()
        
        # Run multiple tests
        tests = [
            ("string_concat", lambda x: x[0] + x[1], lambda x: "".join(x)),
            ("list_sum", lambda x: sum(x), lambda x: sum(x) if x else 0),
            ("max_value", lambda x: max(x), lambda x: max(x) if x else float('-inf')),
        ]
        
        for name, variant_a, variant_b in tests:
            test_data = [
                ["hello", "world"] if name == "string_concat" else [1, 2, 3, 4, 5],
                [] if name != "string_concat" else ["", "test"],
                [42] if name != "string_concat" else ["single"],
            ]
            
            if name == "string_concat":
                test_data = [["hello", "world"], ["", "test"], ["single"]]
            elif name == "list_sum":
                test_data = [[1, 2, 3, 4, 5], [], [42]]
            else:  # max_value
                test_data = [[1, 2, 3, 4, 5], [], [42]]
            
            ab_tester.run_comparison(
                test_name=name,
                variant_a_func=variant_a,
                variant_b_func=variant_b,
                test_data=test_data,
                iterations=20
            )
        
        # Generate comprehensive report
        print(f"\\nComprehensive A/B Testing Report:")
        print(f"Total tests run: {len(ab_tester.results)}")
        
        for i, result in enumerate(ab_tester.results):
            print(f"\\nTest {i+1}:")
            print(f"  Performance A: {result.performance_a:.4f}s")
            print(f"  Performance B: {result.performance_b:.4f}s") 
            print(f"  Reliability A: {result.reliability_score_a:.2%}")
            print(f"  Reliability B: {result.reliability_score_b:.2%}")
            print(f"  Recommendation: {result.recommendation}")
        
        # Convert to JSON for reporting
        report_data = {
            "summary": {
                "total_tests": len(ab_tester.results),
                "avg_confidence": sum(r.confidence for r in ab_tester.results) / len(ab_tester.results)
            },
            "results": [result.to_dict() for result in ab_tester.results]
        }
        
        print(f"\\nJSON Report Summary:")
        print(json.dumps(report_data["summary"], indent=2))
        
        assert len(ab_tester.results) == 3
        assert all(result.confidence > 0 for result in ab_tester.results)


# =============================================================================
# Main Test Runner and Integration
# =============================================================================

class TestUndefinedBehaviorFramework:
    """Meta-tests for the entire undefined behavior testing framework"""
    
    def test_framework_completeness(self):
        """Test that all components are properly integrated"""
        
        # Test that all test classes exist
        test_classes = [
            TestSequencePointViolations,
            TestTypeConversionDangers,
            TestMemoryLayoutIssues,
            TestPointerArithmeticSimulation,
            TestConcurrencyPitfalls,
            TestFloatingPointIssues,
            TestMutableDefaults,
            TestABTestingIntegration,
        ]
        
        for test_class in test_classes:
            assert test_class is not None
            # Check that it has test methods
            methods = [method for method in dir(test_class) if method.startswith('test_')]
            assert len(methods) > 0
            print(f"{test_class.__name__}: {len(methods)} test methods")
        
        print(f"Framework includes {len(test_classes)} test categories")
    
    def test_ab_tester_functionality(self):
        """Test core A/B testing functionality"""
        
        ab_tester = ABTester()
        assert ab_tester is not None
        assert hasattr(ab_tester, 'run_comparison')
        assert hasattr(ab_tester, 'results')
        
        # Simple functionality test
        def simple_a(x): return x * 2
        def simple_b(x): return x + x
        
        result = ab_tester.run_comparison(
            "multiplication_vs_addition",
            simple_a, simple_b,
            [1, 2, 3, 4, 5],
            iterations=10
        )
        
        assert result is not None
        assert result.variant_a == "A"
        assert result.variant_b == "B"
        assert result.performance_a >= 0
        assert result.performance_b >= 0
        assert 0 <= result.error_rate_a <= 1
        assert 0 <= result.error_rate_b <= 1
        
        print(f"A/B tester working correctly")
        print(f"Sample result: {result.recommendation}")


if __name__ == "__main__":
    """
    Run the undefined behavior testing framework
    """
    print("🔥 Undefined Behavior Testing Framework 🔥")
    print("=" * 60)
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\\n" + "=" * 60)
    print("📊 A/B Testing Framework Demo")
    print("=" * 60)
    
    # Demonstrate A/B testing
    ab_tester = ABTester()
    
    # Example comparison
    def unsafe_approach(data):
        return data[0] / data[1]  # Might crash
    
    def safe_approach(data):
        if len(data) < 2:
            return 0.0
        return data[0] / data[1] if data[1] != 0 else float('inf')
    
    test_data = [[10, 2], [5, 0], [0, 1], [1, 0]]
    
    result = ab_tester.run_comparison(
        "safety_demo",
        unsafe_approach,
        safe_approach,
        test_data,
        iterations=50
    )
    
    print(f"Demo Results:")
    print(f"  {result.recommendation}")
    print(f"  Confidence: {result.confidence:.1f}%")
    print(f"  Safe variant error rate: {result.error_rate_b:.1%}")
    print(f"  Unsafe variant error rate: {result.error_rate_a:.1%}")
    
    print("\\n🎉 Undefined Behavior Testing Framework Complete! 🎉")