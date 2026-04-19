#!/usr/bin/env python3
"""
Load Testing Framework for EEG-RAG

Stress tests the system with concurrent queries to identify:
- Performance bottlenecks
- Memory leaks
- Rate limiting behavior
- Error handling under load
- Scalability characteristics
"""

import asyncio
import time
import statistics
import psutil
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random

# Import EEG-RAG components
from ..agents.orchestrator.orchestrator_agent import OrchestratorAgent
from ..agents.base_agent import AgentQuery, QueryComplexity
from ..monitoring import PerformanceMonitor, monitor_performance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : evaluation.load_testing.LoadTestConfig
# Requirement  : `LoadTestConfig` class shall be instantiable and expose the documented interface
# Purpose      : Configuration for load testing
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
# Verification : Instantiate LoadTestConfig with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int = 10
    total_requests: int = 100
    ramp_up_time: float = 30.0  # seconds
    test_duration: Optional[float] = None  # seconds
    think_time_min: float = 1.0
    think_time_max: float = 5.0
    query_timeout: float = 30.0
    
    # Performance thresholds
    max_response_time_ms: float = 2000.0
    max_error_rate: float = 0.05
    max_memory_mb: float = 2048.0
    

# ---------------------------------------------------------------------------
# ID           : evaluation.load_testing.QueryResult
# Requirement  : `QueryResult` class shall be instantiable and expose the documented interface
# Purpose      : Result from a single query execution
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
# Verification : Instantiate QueryResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    """Result from a single query execution."""
    query_text: str
    success: bool
    response_time_ms: float
    error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    timestamp: float = 0.0
    

# ---------------------------------------------------------------------------
# ID           : evaluation.load_testing.LoadTestResults
# Requirement  : `LoadTestResults` class shall be instantiable and expose the documented interface
# Purpose      : Complete results from load testing
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
# Verification : Instantiate LoadTestResults with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class LoadTestResults:
    """Complete results from load testing."""
    config: LoadTestConfig
    query_results: List[QueryResult]
    start_time: float
    end_time: float
    total_duration: float
    
    # Aggregate metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    error_rate: float = 0.0
    
    # Response time statistics
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Resource usage
    avg_memory_usage_mb: float = 0.0
    max_memory_usage_mb: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    max_cpu_usage_percent: float = 0.0
    
    # Throughput
    queries_per_second: float = 0.0
    
    # Performance assessment
    performance_score: float = 0.0
    passed_thresholds: bool = False
    

# ---------------------------------------------------------------------------
# ID           : evaluation.load_testing.LoadTester
# Requirement  : `LoadTester` class shall be instantiable and expose the documented interface
# Purpose      : Load testing framework for EEG-RAG system
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
# Verification : Instantiate LoadTester with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class LoadTester:
    """Load testing framework for EEG-RAG system."""
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester.__init__
    # Requirement  : `__init__` shall initialize load tester
    # Purpose      : Initialize load tester
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : orchestrator: OrchestratorAgent; test_queries: Optional[List[str]] (default=None)
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
        orchestrator: OrchestratorAgent,
        test_queries: Optional[List[str]] = None
    ):
        """Initialize load tester.
        
        Args:
            orchestrator: EEG-RAG orchestrator agent to test.
            test_queries: Optional list of test queries. If None, uses defaults.
        """
        self.orchestrator = orchestrator
        self.test_queries = test_queries or self._get_default_test_queries()
        self.performance_monitor = PerformanceMonitor()
        
        # System monitoring
        self.process = psutil.Process()
        
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester._get_default_test_queries
    # Requirement  : `_get_default_test_queries` shall get default test queries covering various scenarios
    # Purpose      : Get default test queries covering various scenarios
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[str]
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
    def _get_default_test_queries(self) -> List[str]:
        """Get default test queries covering various scenarios."""
        return [
            "What are the main EEG frequency bands?",
            "How is alpha oscillation related to attention?",
            "What is the P300 ERP component?",
            "Describe the use of EEG in epilepsy diagnosis",
            "What are common EEG artifacts and how to remove them?",
            "Explain the relationship between gamma oscillations and consciousness",
            "How does sleep affect EEG patterns?",
            "What is the role of theta waves in memory formation?",
            "Describe EEG-based brain-computer interfaces",
            "How is EEG used in cognitive neuroscience research?",
            "What are the differences between scalp and intracranial EEG?",
            "Explain the concept of event-related desynchronization",
            "How do medications affect EEG patterns?",
            "What is the significance of EEG coherence analysis?",
            "Describe the use of EEG in anesthesia monitoring",
            "What are common EEG montages and their applications?",
            "How is machine learning applied to EEG analysis?",
            "What is the relationship between EEG and fMRI?",
            "Describe the clinical significance of spike and wave patterns",
            "How does age affect normal EEG patterns?"
        ]
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester.run_load_test
    # Requirement  : `run_load_test` shall run comprehensive load test
    # Purpose      : Run comprehensive load test
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: LoadTestConfig; progress_callback: Optional[Callable[[float], None]] (default=None)
    # Outputs      : LoadTestResults
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
    async def run_load_test(
        self,
        config: LoadTestConfig,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> LoadTestResults:
        """Run comprehensive load test.
        
        Args:
            config: Load testing configuration.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Complete load test results.
        """
        logger.info(f"Starting load test with {config.concurrent_users} users, "
                   f"{config.total_requests} requests")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        # Generate task schedule
        tasks = self._generate_task_schedule(config)
        
        # Execute load test
        query_results = []
        completed_tasks = 0
        
        # ---------------------------------------------------------------------------
        # ID           : evaluation.load_testing.LoadTester.execute_query_with_semaphore
        # Requirement  : `execute_query_with_semaphore` shall execute as specified
        # Purpose      : Execute query with semaphore
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : task_info
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
        async def execute_query_with_semaphore(task_info):
            nonlocal completed_tasks
            
            async with semaphore:
                result = await self._execute_single_query(
                    task_info['query'],
                    config.query_timeout
                )
                
                completed_tasks += 1
                if progress_callback:
                    progress = completed_tasks / len(tasks)
                    progress_callback(progress)
                
                return result
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[execute_query_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Filter out exceptions and convert to QueryResult objects
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                query_results.append(QueryResult(
                    query_text=tasks[i]['query'],
                    success=False,
                    response_time_ms=0.0,
                    error_message=str(result),
                    timestamp=time.time()
                ))
            else:
                query_results.append(result)
        
        end_time = time.time()
        
        # Calculate results
        test_results = self._calculate_results(
            config, query_results, start_time, end_time
        )
        
        logger.info(f"Load test completed. Success rate: {(1-test_results.error_rate)*100:.1f}%, "
                   f"Avg response time: {test_results.avg_response_time_ms:.1f}ms")
        
        return test_results
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester._generate_task_schedule
    # Requirement  : `_generate_task_schedule` shall generate schedule of query tasks
    # Purpose      : Generate schedule of query tasks
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: LoadTestConfig
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
    def _generate_task_schedule(self, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Generate schedule of query tasks."""
        tasks = []
        
        for i in range(config.total_requests):
            # Random query selection
            query = random.choice(self.test_queries)
            
            # Calculate start time with ramp-up
            start_delay = (i / config.total_requests) * config.ramp_up_time
            
            tasks.append({
                'query': query,
                'start_delay': start_delay,
                'task_id': i
            })
        
        return tasks
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester._execute_single_query
    # Requirement  : `_execute_single_query` shall execute a single query and measure performance
    # Purpose      : Execute a single query and measure performance
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_text: str; timeout: float
    # Outputs      : QueryResult
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
    async def _execute_single_query(
        self,
        query_text: str,
        timeout: float
    ) -> QueryResult:
        """Execute a single query and measure performance."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        try:
            # Create query
            query = AgentQuery(
                text=query_text,
                complexity=QueryComplexity.MEDIUM,
                context={},
                max_response_time=timeout
            )
            
            # Execute with timeout
            response = await asyncio.wait_for(
                self.orchestrator.process_query(query),
                timeout=timeout
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Measure resource usage
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu = self.process.cpu_percent()
            
            return QueryResult(
                query_text=query_text,
                success=True,
                response_time_ms=response_time_ms,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=end_cpu,
                timestamp=end_time
            )
            
        except asyncio.TimeoutError:
            return QueryResult(
                query_text=query_text,
                success=False,
                response_time_ms=timeout * 1000,
                error_message="Query timeout",
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                timestamp=time.time()
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return QueryResult(
                query_text=query_text,
                success=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                timestamp=end_time
            )
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester._calculate_results
    # Requirement  : `_calculate_results` shall calculate aggregate results from individual query results
    # Purpose      : Calculate aggregate results from individual query results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: LoadTestConfig; query_results: List[QueryResult]; start_time: float; end_time: float
    # Outputs      : LoadTestResults
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
    def _calculate_results(
        self,
        config: LoadTestConfig,
        query_results: List[QueryResult],
        start_time: float,
        end_time: float
    ) -> LoadTestResults:
        """Calculate aggregate results from individual query results."""
        total_duration = end_time - start_time
        
        # Basic counts
        total_queries = len(query_results)
        successful_queries = sum(1 for r in query_results if r.success)
        failed_queries = total_queries - successful_queries
        error_rate = failed_queries / total_queries if total_queries > 0 else 0.0
        
        # Response time statistics (only for successful queries)
        successful_times = [r.response_time_ms for r in query_results if r.success]
        
        if successful_times:
            avg_response_time = statistics.mean(successful_times)
            min_response_time = min(successful_times)
            max_response_time = max(successful_times)
            p50_response_time = statistics.median(successful_times)
            
            sorted_times = sorted(successful_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
            p50_response_time = p95_response_time = p99_response_time = 0.0
        
        # Resource usage statistics
        memory_usages = [r.memory_usage_mb for r in query_results if r.memory_usage_mb > 0]
        cpu_usages = [r.cpu_usage_percent for r in query_results if r.cpu_usage_percent > 0]
        
        avg_memory = statistics.mean(memory_usages) if memory_usages else 0.0
        max_memory = max(memory_usages) if memory_usages else 0.0
        avg_cpu = statistics.mean(cpu_usages) if cpu_usages else 0.0
        max_cpu = max(cpu_usages) if cpu_usages else 0.0
        
        # Throughput
        queries_per_second = successful_queries / total_duration if total_duration > 0 else 0.0
        
        # Performance score (0-100)
        performance_score = self._calculate_performance_score(
            config, avg_response_time, error_rate, max_memory
        )
        
        # Check if passed thresholds
        passed_thresholds = (
            avg_response_time <= config.max_response_time_ms and
            error_rate <= config.max_error_rate and
            max_memory <= config.max_memory_mb
        )
        
        return LoadTestResults(
            config=config,
            query_results=query_results,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            avg_memory_usage_mb=avg_memory,
            max_memory_usage_mb=max_memory,
            avg_cpu_usage_percent=avg_cpu,
            max_cpu_usage_percent=max_cpu,
            queries_per_second=queries_per_second,
            performance_score=performance_score,
            passed_thresholds=passed_thresholds
        )
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester._calculate_performance_score
    # Requirement  : `_calculate_performance_score` shall calculate performance score from 0-100
    # Purpose      : Calculate performance score from 0-100
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : config: LoadTestConfig; avg_response_time: float; error_rate: float; max_memory: float
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
    def _calculate_performance_score(
        self,
        config: LoadTestConfig,
        avg_response_time: float,
        error_rate: float,
        max_memory: float
    ) -> float:
        """Calculate performance score from 0-100."""
        # Response time score (40% weight)
        response_score = max(0, 100 - (avg_response_time / config.max_response_time_ms) * 100)
        
        # Error rate score (40% weight)
        error_score = max(0, 100 - (error_rate / config.max_error_rate) * 100)
        
        # Memory score (20% weight)
        memory_score = max(0, 100 - (max_memory / config.max_memory_mb) * 100)
        
        return (response_score * 0.4) + (error_score * 0.4) + (memory_score * 0.2)
    
    # ---------------------------------------------------------------------------
    # ID           : evaluation.load_testing.LoadTester.export_results
    # Requirement  : `export_results` shall export results to JSON file
    # Purpose      : Export results to JSON file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : results: LoadTestResults; output_path: Path; include_details: bool (default=True)
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
    def export_results(
        self,
        results: LoadTestResults,
        output_path: Path,
        include_details: bool = True
    ):
        """Export results to JSON file."""
        export_data = {
            'summary': {
                'total_queries': results.total_queries,
                'successful_queries': results.successful_queries,
                'error_rate': results.error_rate,
                'avg_response_time_ms': results.avg_response_time_ms,
                'p95_response_time_ms': results.p95_response_time_ms,
                'queries_per_second': results.queries_per_second,
                'performance_score': results.performance_score,
                'passed_thresholds': results.passed_thresholds
            },
            'config': asdict(results.config),
            'timestamp': results.start_time
        }
        
        if include_details:
            export_data['detailed_results'] = [
                asdict(result) for result in results.query_results
            ]
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")


# ---------------------------------------------------------------------------
# ID           : evaluation.load_testing.run_load_test_suite
# Requirement  : `run_load_test_suite` shall run complete load testing suite with different configurations
# Purpose      : Run complete load testing suite with different configurations
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : orchestrator: OrchestratorAgent; output_dir: Path; test_queries: Optional[List[str]] (default=None)
# Outputs      : Dict[str, LoadTestResults]
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
async def run_load_test_suite(
    orchestrator: OrchestratorAgent,
    output_dir: Path,
    test_queries: Optional[List[str]] = None
) -> Dict[str, LoadTestResults]:
    """Run complete load testing suite with different configurations."""
    load_tester = LoadTester(orchestrator, test_queries)
    output_dir.mkdir(exist_ok=True)
    
    test_configs = {
        'light': LoadTestConfig(
            concurrent_users=5,
            total_requests=50,
            ramp_up_time=10.0
        ),
        'medium': LoadTestConfig(
            concurrent_users=10,
            total_requests=100,
            ramp_up_time=30.0
        ),
        'heavy': LoadTestConfig(
            concurrent_users=20,
            total_requests=200,
            ramp_up_time=60.0
        )
    }
    
    results = {}
    
    for test_name, config in test_configs.items():
        logger.info(f"Running {test_name} load test...")
        
        test_results = await load_tester.run_load_test(config)
        results[test_name] = test_results
        
        # Export results
        output_file = output_dir / f"load_test_{test_name}.json"
        load_tester.export_results(test_results, output_file)
        
        logger.info(f"{test_name} test completed. Score: {test_results.performance_score:.1f}/100")
        
        # Brief pause between tests
        await asyncio.sleep(5.0)
    
    return results