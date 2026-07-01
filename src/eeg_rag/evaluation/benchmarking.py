#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for EEG-RAG

Benchmarks different aspects of the system:
- Retrieval accuracy and speed
- Generation quality and citations
- End-to-end response quality
- Memory and CPU performance
- Agent comparison
"""

import asyncio
import time
import json
import statistics
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

# Import EEG-RAG components
from ..agents.base_agent import AgentQuery, QueryComplexity
from ..agents.orchestrator.orchestrator_agent import OrchestratorAgent
from ..agents.local_agent.local_data_agent import LocalDataAgent
from ..agents.web_agent.web_search_agent import WebSearchAgent
from ..ensemble.context_aggregator import ContextAggregator
from .ground_truth_benchmarks import GroundTruthBenchmarks
from ..verification.citation_verifier import CitationVerifier
from ..monitoring import PerformanceMonitor, monitor_performance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.BenchmarkQuery
# Requirement  : `BenchmarkQuery` class shall be instantiable and expose the documented interface
# Purpose      : Query with expected characteristics for benchmarking
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
# Verification : Instantiate BenchmarkQuery with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkQuery:
    """Query with expected characteristics for benchmarking."""
    query_text: str
    complexity: QueryComplexity
    expected_topics: List[str]
    expected_citations_min: int
    expected_response_length_min: int
    category: str


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.RetrievalBenchmarkResult
# Requirement  : `RetrievalBenchmarkResult` class shall be instantiable and expose the documented interface
# Purpose      : Results from retrieval benchmarking
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
# Verification : Instantiate RetrievalBenchmarkResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class RetrievalBenchmarkResult:
    """Results from retrieval benchmarking."""
    query: str
    retrieval_time_ms: float
    documents_found: int
    relevance_score: float
    citation_accuracy: float
    redundancy_score: float = 0.0
    diversity_score: float = 1.0
    query_entity_coverage_score: float = 1.0
    query_concept_coverage_score: float = 1.0
    centrality_grounding_score: float = 0.0
    aggregation_diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.GenerationBenchmarkResult
# Requirement  : `GenerationBenchmarkResult` class shall be instantiable and expose the documented interface
# Purpose      : Results from generation benchmarking
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
# Verification : Instantiate GenerationBenchmarkResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class GenerationBenchmarkResult:
    """Results from generation benchmarking."""
    query: str
    generation_time_ms: float
    response_length: int
    citation_count: int
    citation_validity: float
    coherence_score: float
    factual_accuracy: float


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.EndToEndBenchmarkResult
# Requirement  : `EndToEndBenchmarkResult` class shall be instantiable and expose the documented interface
# Purpose      : Results from end-to-end benchmarking
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
# Verification : Instantiate EndToEndBenchmarkResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class EndToEndBenchmarkResult:
    """Results from end-to-end benchmarking."""
    query: str
    agent_type: str
    total_time_ms: float
    response_quality: float
    citation_quality: float
    user_satisfaction: float
    memory_usage_mb: float
    cpu_usage_percent: float
    redundancy_score: float = 0.0
    diversity_score: float = 1.0
    query_concept_coverage_score: float = 1.0
    centrality_grounding_score: float = 0.0


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.BenchmarkSuite
# Requirement  : `BenchmarkSuite` class shall be instantiable and expose the documented interface
# Purpose      : Complete benchmark results
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
# Verification : Instantiate BenchmarkSuite with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    retrieval_results: List[RetrievalBenchmarkResult]
    generation_results: List[GenerationBenchmarkResult]
    end_to_end_results: List[EndToEndBenchmarkResult]

    # Aggregate metrics
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    avg_citation_accuracy: float = 0.0
    avg_response_quality: float = 0.0
    avg_redundancy_score: float = 0.0
    avg_diversity_score: float = 1.0
    avg_query_entity_coverage_score: float = 1.0
    avg_query_concept_coverage_score: float = 1.0
    avg_centrality_grounding_score: float = 0.0
    avg_grounding_quality: float = 0.0
    concept_aware_grounding_score: float = 0.0
    concept_aware_ranking_ndcg: float = 0.0
    ranking_strategy_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Performance scores
    retrieval_score: float = 0.0
    generation_score: float = 0.0
    overall_score: float = 0.0


# ---------------------------------------------------------------------------
# ID           : evaluation.benchmarking.EEGRAGBenchmark
# Requirement  : `EEGRAGBenchmark` class shall be instantiable and expose the documented interface
# Purpose      : Comprehensive benchmarking suite for EEG-RAG system
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
# Verification : Instantiate EEGRAGBenchmark with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class EEGRAGBenchmark:
    """Comprehensive benchmarking suite for EEG-RAG system."""

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark.__init__
    # Requirement  : `__init__` shall initialize benchmarking suite
    # Purpose      : Initialize benchmarking suite
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : orchestrator: OrchestratorAgent; local_agent: Optional[LocalDataAgent] (default=None); web_agent: Optional[WebSearchAgent] (default=None)
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
        local_agent: Optional[LocalDataAgent] = None,
        web_agent: Optional[WebSearchAgent] = None,
        min_concept_aware_ranking_ndcg: float = 0.65,
        bootstrap_samples: int = 500,
    ):
        """Initialize benchmarking suite.

        Args:
            orchestrator: Main orchestrator agent.
            local_agent: Optional local data agent for comparison.
            web_agent: Optional web research agent for comparison.
        """
        self.orchestrator = orchestrator
        self.local_agent = local_agent
        self.web_agent = web_agent
        self.min_concept_aware_ranking_ndcg = min_concept_aware_ranking_ndcg
        self.bootstrap_samples = bootstrap_samples

        self.citation_verifier = CitationVerifier(enable_medical_validation=True)
        self.performance_monitor = PerformanceMonitor()
        self._utility_weights = self._calibrate_utility_weights_from_ground_truth()

        # Load benchmark queries
        self.benchmark_queries = self._create_benchmark_queries()

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._create_benchmark_queries
    # Requirement  : `_create_benchmark_queries` shall create comprehensive set of benchmark queries
    # Purpose      : Create comprehensive set of benchmark queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[BenchmarkQuery]
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
    def _create_benchmark_queries(self) -> List[BenchmarkQuery]:
        """Create comprehensive set of benchmark queries."""
        return [
            # Basic EEG knowledge
            BenchmarkQuery(
                query_text="What are the main EEG frequency bands and their clinical significance?",
                complexity=QueryComplexity.LOW,
                expected_topics=["delta", "theta", "alpha", "beta", "gamma"],
                expected_citations_min=3,
                expected_response_length_min=200,
                category="basic_knowledge"
            ),

            # Clinical applications
            BenchmarkQuery(
                query_text="How is EEG used in epilepsy diagnosis and what are the characteristic patterns?",
                complexity=QueryComplexity.MEDIUM,
                expected_topics=["epilepsy", "seizure", "interictal", "spike"],
                expected_citations_min=5,
                expected_response_length_min=300,
                category="clinical"
            ),

            # Research methods
            BenchmarkQuery(
                query_text="What are the latest advances in EEG-based brain-computer interfaces?",
                complexity=QueryComplexity.HIGH,
                expected_topics=["BCI", "motor imagery", "P300", "SSVEP"],
                expected_citations_min=7,
                expected_response_length_min=400,
                category="research"
            ),

            # Technical details
            BenchmarkQuery(
                query_text="Explain the technical principles of independent component analysis in EEG processing",
                complexity=QueryComplexity.HIGH,
                expected_topics=["ICA", "artifact removal", "blind source separation"],
                expected_citations_min=4,
                expected_response_length_min=350,
                category="technical"
            ),

            # Comparative questions
            BenchmarkQuery(
                query_text="Compare the advantages and limitations of scalp EEG versus intracranial EEG",
                complexity=QueryComplexity.MEDIUM,
                expected_topics=["scalp", "intracranial", "iEEG", "spatial resolution"],
                expected_citations_min=5,
                expected_response_length_min=300,
                category="comparison"
            ),

            # Developmental aspects
            BenchmarkQuery(
                query_text="How do normal EEG patterns change across the human lifespan?",
                complexity=QueryComplexity.MEDIUM,
                expected_topics=["development", "aging", "maturation", "pediatric"],
                expected_citations_min=6,
                expected_response_length_min=350,
                category="developmental"
            ),

            # Pathological conditions
            BenchmarkQuery(
                query_text="What are the EEG characteristics of different types of dementia?",
                complexity=QueryComplexity.HIGH,
                expected_topics=["dementia", "Alzheimer's", "theta", "slowing"],
                expected_citations_min=6,
                expected_response_length_min=400,
                category="pathology"
            ),

            # Methodological questions
            BenchmarkQuery(
                query_text="What are the best practices for EEG artifact detection and removal?",
                complexity=QueryComplexity.MEDIUM,
                expected_topics=["artifacts", "EOG", "EMG", "filtering"],
                expected_citations_min=5,
                expected_response_length_min=300,
                category="methodology"
            )
        ]

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark.run_full_benchmark
    # Requirement  : `run_full_benchmark` shall run complete benchmarking suite
    # Purpose      : Run complete benchmarking suite
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : BenchmarkSuite
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
    async def run_full_benchmark(self) -> BenchmarkSuite:
        """Run complete benchmarking suite."""
        logger.info("Starting comprehensive benchmark suite...")

        start_time = time.time()

        # 1. Retrieval benchmarks
        logger.info("Running retrieval benchmarks...")
        retrieval_results = await self._benchmark_retrieval()

        # 2. Generation benchmarks
        logger.info("Running generation benchmarks...")
        generation_results = await self._benchmark_generation()

        # 3. End-to-end benchmarks
        logger.info("Running end-to-end benchmarks...")
        end_to_end_results = await self._benchmark_end_to_end()

        # 4. Aggregation strategy benchmark for concept-aware grounding
        logger.info("Running aggregation strategy benchmark...")
        ranking_comparison = await self._benchmark_aggregation_strategies()
        self._enforce_ranking_regression_guard(ranking_comparison)

        # 5. Calculate aggregate metrics
        suite_results = self._calculate_aggregate_metrics(
            retrieval_results,
            generation_results,
            end_to_end_results,
            ranking_comparison=ranking_comparison,
        )

        total_time = time.time() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f}s. "
                   f"Overall score: {suite_results.overall_score:.1f}/100")

        return suite_results

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._benchmark_retrieval
    # Requirement  : `_benchmark_retrieval` shall benchmark retrieval performance
    # Purpose      : Benchmark retrieval performance
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[RetrievalBenchmarkResult]
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
    async def _benchmark_retrieval(self) -> List[RetrievalBenchmarkResult]:
        """Benchmark retrieval performance."""
        results = []

        for benchmark_query in self.benchmark_queries:
            start_time = time.time()

            try:
                # Use local agent for pure retrieval testing
                if self.local_agent:
                    query = AgentQuery(
                        text=benchmark_query.query_text,
                        complexity=benchmark_query.complexity,
                        context={}
                    )

                    # Get retrieval results
                    response = await self.local_agent.process_query(query)

                    retrieval_time_ms = (time.time() - start_time) * 1000

                    # Analyze retrieval quality
                    documents_found = len(response.sources)
                    relevance_score = await self._calculate_relevance_score(
                        benchmark_query, response.sources
                    )
                    citation_accuracy = await self._calculate_citation_accuracy(
                        response.sources
                    )
                    redundancy_score = self._calculate_source_redundancy(response.sources)
                    query_entity_coverage_score, query_concept_coverage_score = self._calculate_source_coverage(
                        benchmark_query,
                        response.sources,
                    )
                    source_diagnostics = self._source_diagnostics(response.sources)
                    diversity_score = source_diagnostics.get(
                        "diversity_score",
                        max(0.0, 1.0 - redundancy_score),
                    )
                    centrality_grounding_score = self._calculate_centrality_grounding_score(
                        response.sources
                    )

                    results.append(RetrievalBenchmarkResult(
                        query=benchmark_query.query_text,
                        retrieval_time_ms=retrieval_time_ms,
                        documents_found=documents_found,
                        relevance_score=relevance_score,
                        citation_accuracy=citation_accuracy,
                        redundancy_score=redundancy_score,
                        diversity_score=diversity_score,
                        query_entity_coverage_score=query_entity_coverage_score,
                        query_concept_coverage_score=query_concept_coverage_score,
                        centrality_grounding_score=centrality_grounding_score,
                        aggregation_diagnostics=source_diagnostics,
                    ))

            except Exception as e:
                logger.error(f"Retrieval benchmark failed for query '{benchmark_query.query_text}': {str(e)}")

                results.append(RetrievalBenchmarkResult(
                    query=benchmark_query.query_text,
                    retrieval_time_ms=0.0,
                    documents_found=0,
                    relevance_score=0.0,
                    citation_accuracy=0.0,
                    redundancy_score=0.0,
                    diversity_score=1.0,
                    query_entity_coverage_score=0.0,
                    query_concept_coverage_score=0.0,
                    centrality_grounding_score=0.0,
                    aggregation_diagnostics={},
                ))

        return results

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._benchmark_generation
    # Requirement  : `_benchmark_generation` shall benchmark generation quality
    # Purpose      : Benchmark generation quality
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[GenerationBenchmarkResult]
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
    async def _benchmark_generation(self) -> List[GenerationBenchmarkResult]:
        """Benchmark generation quality."""
        results = []

        for benchmark_query in self.benchmark_queries:
            start_time = time.time()

            try:
                query = AgentQuery(
                    text=benchmark_query.query_text,
                    complexity=benchmark_query.complexity,
                    context={}
                )

                response = await self.orchestrator.process_query(query)
                generation_time_ms = (time.time() - start_time) * 1000

                # Analyze generation quality
                response_length = len(response.content)
                citation_count = len(response.citations)

                citation_validity = await self._validate_citations(
                    response.citations
                )

                coherence_score = self._calculate_coherence_score(
                    response.content
                )

                factual_accuracy = await self._calculate_factual_accuracy(
                    benchmark_query, response.content
                )

                results.append(GenerationBenchmarkResult(
                    query=benchmark_query.query_text,
                    generation_time_ms=generation_time_ms,
                    response_length=response_length,
                    citation_count=citation_count,
                    citation_validity=citation_validity,
                    coherence_score=coherence_score,
                    factual_accuracy=factual_accuracy
                ))

            except Exception as e:
                logger.error(f"Generation benchmark failed for query '{benchmark_query.query_text}': {str(e)}")

                results.append(GenerationBenchmarkResult(
                    query=benchmark_query.query_text,
                    generation_time_ms=0.0,
                    response_length=0,
                    citation_count=0,
                    citation_validity=0.0,
                    coherence_score=0.0,
                    factual_accuracy=0.0
                ))

        return results

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._benchmark_end_to_end
    # Requirement  : `_benchmark_end_to_end` shall benchmark end-to-end performance across different agents
    # Purpose      : Benchmark end-to-end performance across different agents
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[EndToEndBenchmarkResult]
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
    async def _benchmark_end_to_end(self) -> List[EndToEndBenchmarkResult]:
        """Benchmark end-to-end performance across different agents."""
        results = []

        agents_to_test = [
            ('orchestrator', self.orchestrator)
        ]

        if self.local_agent:
            agents_to_test.append(('local', self.local_agent))

        if self.web_agent:
            agents_to_test.append(('web', self.web_agent))

        for benchmark_query in self.benchmark_queries:
            for agent_name, agent in agents_to_test:
                start_time = time.time()

                try:
                    query = AgentQuery(
                        text=benchmark_query.query_text,
                        complexity=benchmark_query.complexity,
                        context={}
                    )

                    with monitor_performance() as monitor:
                        response = await agent.process_query(query)

                    total_time_ms = (time.time() - start_time) * 1000

                    # Calculate quality metrics
                    response_quality = await self._calculate_response_quality(
                        benchmark_query, response
                    )

                    citation_quality = await self._calculate_citation_quality(
                        response.citations
                    )

                    user_satisfaction = self._calculate_user_satisfaction(
                        benchmark_query, response
                    )

                    results.append(EndToEndBenchmarkResult(
                        query=benchmark_query.query_text,
                        agent_type=agent_name,
                        total_time_ms=total_time_ms,
                        response_quality=response_quality,
                        citation_quality=citation_quality,
                        user_satisfaction=user_satisfaction,
                        memory_usage_mb=monitor.peak_memory_mb,
                        cpu_usage_percent=monitor.avg_cpu_percent,
                        redundancy_score=self._calculate_source_redundancy(getattr(response, "sources", [])),
                        diversity_score=max(0.0, 1.0 - self._calculate_source_redundancy(getattr(response, "sources", []))),
                        query_concept_coverage_score=self._calculate_source_coverage(
                            benchmark_query,
                            getattr(response, "sources", []),
                        )[1],
                        centrality_grounding_score=self._calculate_centrality_grounding_score(
                            getattr(response, "sources", []),
                        ),
                    ))

                except Exception as e:
                    logger.error(f"End-to-end benchmark failed for agent '{agent_name}', "
                               f"query '{benchmark_query.query_text}': {str(e)}")

                    results.append(EndToEndBenchmarkResult(
                        query=benchmark_query.query_text,
                        agent_type=agent_name,
                        total_time_ms=0.0,
                        response_quality=0.0,
                        citation_quality=0.0,
                        user_satisfaction=0.0,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        redundancy_score=0.0,
                        diversity_score=1.0,
                        query_concept_coverage_score=0.0,
                        centrality_grounding_score=0.0,
                    ))

        return results

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_relevance_score
    # Requirement  : `_calculate_relevance_score` shall calculate relevance score for retrieved documents
    # Purpose      : Calculate relevance score for retrieved documents
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : benchmark_query: BenchmarkQuery; sources: List[Any]
    # Outputs      : float
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
    async def _calculate_relevance_score(
        self,
        benchmark_query: BenchmarkQuery,
        sources: List[Any]
    ) -> float:
        """Calculate relevance score for retrieved documents."""
        if not sources:
            return 0.0

        # Simple keyword-based relevance scoring
        total_score = 0.0

        for source in sources:
            content = getattr(source, 'content', '') + getattr(source, 'title', '')
            content_lower = content.lower()

            topic_matches = sum(
                1 for topic in benchmark_query.expected_topics
                if topic.lower() in content_lower
            )

            relevance = topic_matches / len(benchmark_query.expected_topics)
            total_score += relevance

        return min(1.0, total_score / len(sources))

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_citation_accuracy
    # Requirement  : `_calculate_citation_accuracy` shall calculate citation accuracy for sources
    # Purpose      : Calculate citation accuracy for sources
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : sources: List[Any]
    # Outputs      : float
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
    async def _calculate_citation_accuracy(self, sources: List[Any]) -> float:
        """Calculate citation accuracy for sources."""
        if not sources:
            return 0.0

        valid_citations = 0
        total_citations = 0

        for source in sources:
            if hasattr(source, 'pmid') and source.pmid:
                total_citations += 1
                # Use citation verifier to check validity
                try:
                    is_valid = await self.citation_verifier.verify_pmid(source.pmid)
                    if is_valid:
                        valid_citations += 1
                except:
                    pass

        return valid_citations / total_citations if total_citations > 0 else 0.0

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._validate_citations
    # Requirement  : `_validate_citations` shall validate citation PMIDs
    # Purpose      : Validate citation PMIDs
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : citations: List[str]
    # Outputs      : float
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
    async def _validate_citations(self, citations: List[str]) -> float:
        """Validate citation PMIDs."""
        if not citations:
            return 0.0

        valid_count = 0

        for citation in citations:
            try:
                pmid_match = self.citation_verifier.extract_pmid(citation)
                if pmid_match:
                    is_valid = await self.citation_verifier.verify_pmid(pmid_match)
                    if is_valid:
                        valid_count += 1
            except:
                pass

        return valid_count / len(citations)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_coherence_score
    # Requirement  : `_calculate_coherence_score` shall calculate coherence score for generated content
    # Purpose      : Calculate coherence score for generated content
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : content: str
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
    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence score for generated content."""
        # Simple coherence metrics
        sentences = content.split('.')

        if len(sentences) < 2:
            return 0.5

        # Check for consistent terminology
        word_counts = defaultdict(int)
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if len(word) > 3:  # Only count meaningful words
                    word_counts[word] += 1

        # Coherence based on term repetition and length
        repeated_terms = sum(1 for count in word_counts.values() if count > 1)
        coherence = min(1.0, repeated_terms / len(word_counts)) if word_counts else 0.0

        # Length bonus
        length_score = min(1.0, len(content) / 200)

        return (coherence + length_score) / 2

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_factual_accuracy
    # Requirement  : `_calculate_factual_accuracy` shall calculate factual accuracy score
    # Purpose      : Calculate factual accuracy score
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : benchmark_query: BenchmarkQuery; content: str
    # Outputs      : float
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
    async def _calculate_factual_accuracy(
        self,
        benchmark_query: BenchmarkQuery,
        content: str
    ) -> float:
        """Calculate factual accuracy score."""
        # Check for presence of expected topics
        content_lower = content.lower()

        topic_matches = sum(
            1 for topic in benchmark_query.expected_topics
            if topic.lower() in content_lower
        )

        topic_accuracy = topic_matches / len(benchmark_query.expected_topics)

        # Check for common EEG facts
        fact_checks = [
            ("alpha", "8", "13"),  # Alpha band frequency
            ("theta", "4", "8"),   # Theta band frequency
            ("delta", "0.5", "4"), # Delta band frequency
            ("beta", "13", "30"),  # Beta band frequency
        ]

        fact_score = 0.0
        fact_count = 0

        for band, low_freq, high_freq in fact_checks:
            if band in content_lower:
                fact_count += 1
                if low_freq in content and high_freq in content:
                    fact_score += 1

        fact_accuracy = fact_score / fact_count if fact_count > 0 else 1.0

        return (topic_accuracy + fact_accuracy) / 2

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_response_quality
    # Requirement  : `_calculate_response_quality` shall calculate overall response quality
    # Purpose      : Calculate overall response quality
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : benchmark_query: BenchmarkQuery; response: Any
    # Outputs      : float
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
    async def _calculate_response_quality(
        self,
        benchmark_query: BenchmarkQuery,
        response: Any
    ) -> float:
        """Calculate overall response quality."""
        # Length score
        content_length = len(response.content)
        length_score = min(1.0, content_length / benchmark_query.expected_response_length_min)

        # Citation score
        citation_count = len(response.citations)
        citation_score = min(1.0, citation_count / benchmark_query.expected_citations_min)

        # Factual accuracy
        factual_score = await self._calculate_factual_accuracy(
            benchmark_query, response.content
        )

        # Weighted average
        return (length_score * 0.3) + (citation_score * 0.3) + (factual_score * 0.4)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_citation_quality
    # Requirement  : `_calculate_citation_quality` shall calculate citation quality score
    # Purpose      : Calculate citation quality score
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : citations: List[str]
    # Outputs      : float
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
    async def _calculate_citation_quality(self, citations: List[str]) -> float:
        """Calculate citation quality score."""
        if not citations:
            return 0.0

        # Combine validity and format scoring
        validity_score = await self._validate_citations(citations)

        # Format score (proper PMID format)
        format_score = 0.0
        for citation in citations:
            if 'PMID:' in citation and citation.replace('PMID:', '').strip().isdigit():
                format_score += 1

        format_score = format_score / len(citations)

        return (validity_score + format_score) / 2

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_user_satisfaction
    # Requirement  : `_calculate_user_satisfaction` shall calculate user satisfaction score
    # Purpose      : Calculate user satisfaction score
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : benchmark_query: BenchmarkQuery; response: Any
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
    def _calculate_user_satisfaction(
        self,
        benchmark_query: BenchmarkQuery,
        response: Any
    ) -> float:
        """Calculate user satisfaction score."""
        # Completeness score
        content = response.content.lower()
        topic_coverage = sum(
            1 for topic in benchmark_query.expected_topics
            if topic.lower() in content
        ) / len(benchmark_query.expected_topics)

        # Citation presence
        has_citations = 1.0 if response.citations else 0.0

        # Length appropriateness
        content_length = len(response.content)
        length_appropriate = 1.0 if content_length >= benchmark_query.expected_response_length_min else content_length / benchmark_query.expected_response_length_min

        return (topic_coverage * 0.5) + (has_citations * 0.25) + (length_appropriate * 0.25)

    @staticmethod
    def _source_text(source: Any) -> str:
        """Return the best available text for a retrieved source."""
        if isinstance(source, dict):
            metadata = source.get("metadata", {}) or {}
            for key in ("text", "content", "abstract", "title"):
                value = source.get(key)
                if value:
                    return str(value)
                value = metadata.get(key)
                if value:
                    return str(value)
            return ""

        metadata = getattr(source, "metadata", {}) or {}
        for key in ("text", "content", "abstract", "title"):
            value = getattr(source, key, None)
            if value:
                return str(value)
            value = metadata.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _source_metadata(source: Any) -> Dict[str, Any]:
        """Return a best-effort metadata mapping for a source."""
        if isinstance(source, dict):
            metadata = source.get("metadata")
            return metadata if isinstance(metadata, dict) else {}
        metadata = getattr(source, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _source_diagnostics(cls, sources: List[Any]) -> Dict[str, float]:
        """Extract aggregate diagnostics embedded in retrieved sources."""
        diagnostics: Dict[str, List[float]] = defaultdict(list)
        for source in sources:
            metadata = cls._source_metadata(source)
            for key in (
                "redundancy_score",
                "diversity_score",
                "query_entity_coverage_score",
                "query_concept_coverage_score",
                "centrality_score",
                "centrality_grounding_score",
            ):
                value = metadata.get(key)
                if isinstance(value, (int, float)):
                    diagnostics[key].append(float(value))

        return {
            key: statistics.mean(values)
            for key, values in diagnostics.items()
            if values
        }

    @classmethod
    def _calculate_source_redundancy(cls, sources: List[Any]) -> float:
        """Estimate redundancy among retrieved sources using lexical overlap."""
        if len(sources) <= 1:
            return 0.0

        token_sets = []
        for source in sources:
            text = cls._source_text(source).lower()
            tokens = set(token for token in text.split() if token)
            if tokens:
                token_sets.append(tokens)

        if len(token_sets) <= 1:
            return 0.0

        pairwise = []
        for i, left in enumerate(token_sets):
            for j, right in enumerate(token_sets):
                if i >= j:
                    continue
                union = left | right
                if not union:
                    continue
                pairwise.append(len(left & right) / len(union))

        return statistics.mean(pairwise) if pairwise else 0.0

    @staticmethod
    def _calculate_source_coverage(
        benchmark_query: BenchmarkQuery,
        sources: List[Any],
    ) -> Tuple[float, float]:
        """Calculate entity and concept coverage from retrieved sources."""
        combined_text = " ".join(EEGRAGBenchmark._source_text(source).lower() for source in sources)
        if not benchmark_query.expected_topics:
            return 1.0, 1.0

        topic_coverage = sum(
            1 for topic in benchmark_query.expected_topics
            if topic.lower() in combined_text
        ) / len(benchmark_query.expected_topics)

        diagnostics = EEGRAGBenchmark._source_diagnostics(sources)
        concept_coverage = diagnostics.get(
            "query_concept_coverage_score",
            diagnostics.get("query_entity_coverage_score", topic_coverage),
        )

        return topic_coverage, float(concept_coverage)

    @classmethod
    def _calculate_centrality_grounding_score(cls, sources: List[Any]) -> float:
        """Calculate a centrality-aware grounding score from source metadata."""
        diagnostics = cls._source_diagnostics(sources)
        centrality = diagnostics.get(
            "centrality_grounding_score",
            diagnostics.get("centrality_score", 0.0),
        )
        redundancy = cls._calculate_source_redundancy(sources)
        coverage = diagnostics.get("query_concept_coverage_score", diagnostics.get("query_entity_coverage_score", 1.0))

        return max(0.0, min(1.0, (0.5 * float(coverage)) + (0.3 * float(centrality)) + (0.2 * (1.0 - redundancy))))

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_aggregate_metrics
    # Requirement  : `_calculate_aggregate_metrics` shall calculate aggregate metrics from individual results
    # Purpose      : Calculate aggregate metrics from individual results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : retrieval_results: List[RetrievalBenchmarkResult]; generation_results: List[GenerationBenchmarkResult]; end_to_end_results: List[EndToEndBenchmarkResult]
    # Outputs      : BenchmarkSuite
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
    def _calculate_aggregate_metrics(
        self,
        retrieval_results: List[RetrievalBenchmarkResult],
        generation_results: List[GenerationBenchmarkResult],
        end_to_end_results: List[EndToEndBenchmarkResult],
        ranking_comparison: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BenchmarkSuite:
        """Calculate aggregate metrics from individual results."""
        # Retrieval aggregates
        retrieval_times = [r.retrieval_time_ms for r in retrieval_results if r.retrieval_time_ms > 0]
        avg_retrieval_time = statistics.mean(retrieval_times) if retrieval_times else 0.0

        # Generation aggregates
        generation_times = [r.generation_time_ms for r in generation_results if r.generation_time_ms > 0]
        avg_generation_time = statistics.mean(generation_times) if generation_times else 0.0

        # End-to-end aggregates
        total_times = [r.total_time_ms for r in end_to_end_results if r.total_time_ms > 0]
        avg_total_time = statistics.mean(total_times) if total_times else 0.0

        # Citation accuracy
        citation_accuracies = [r.citation_accuracy for r in retrieval_results]
        avg_citation_accuracy = statistics.mean(citation_accuracies) if citation_accuracies else 0.0

        # Response quality
        response_qualities = [r.response_quality for r in end_to_end_results if r.response_quality > 0]
        avg_response_quality = statistics.mean(response_qualities) if response_qualities else 0.0

        # Retrieval diagnostics
        redundancy_scores = [r.redundancy_score for r in retrieval_results if r.documents_found > 0]
        diversity_scores = [r.diversity_score for r in retrieval_results if r.documents_found > 0]
        entity_coverages = [r.query_entity_coverage_score for r in retrieval_results if r.documents_found > 0]
        concept_coverages = [r.query_concept_coverage_score for r in retrieval_results if r.documents_found > 0]
        centrality_scores = [r.centrality_grounding_score for r in retrieval_results if r.documents_found > 0]

        avg_redundancy_score = statistics.mean(redundancy_scores) if redundancy_scores else 0.0
        avg_diversity_score = statistics.mean(diversity_scores) if diversity_scores else 1.0
        avg_query_entity_coverage_score = statistics.mean(entity_coverages) if entity_coverages else 0.0
        avg_query_concept_coverage_score = statistics.mean(concept_coverages) if concept_coverages else 0.0
        avg_centrality_grounding_score = statistics.mean(centrality_scores) if centrality_scores else 0.0
        avg_grounding_quality = (
            0.5 * avg_query_concept_coverage_score
            + 0.3 * avg_centrality_grounding_score
            + 0.2 * (1.0 - avg_redundancy_score)
        )
        concept_aware_grounding_score = 0.0
        concept_aware_ranking_ndcg = 0.0
        if ranking_comparison:
            concept_aware_grounding_score = ranking_comparison.get(
                "concept_aware",
                {},
            ).get("grounding_quality", 0.0)
            concept_aware_ranking_ndcg = ranking_comparison.get(
                "concept_aware",
                {},
            ).get("ranking_ndcg", 0.0)

        # Performance scores
        retrieval_score = self._calculate_retrieval_score(retrieval_results)
        generation_score = self._calculate_generation_score(generation_results)
        overall_score = (retrieval_score + generation_score) / 2

        return BenchmarkSuite(
            retrieval_results=retrieval_results,
            generation_results=generation_results,
            end_to_end_results=end_to_end_results,
            avg_retrieval_time_ms=avg_retrieval_time,
            avg_generation_time_ms=avg_generation_time,
            avg_total_time_ms=avg_total_time,
            avg_citation_accuracy=avg_citation_accuracy,
            avg_response_quality=avg_response_quality,
            avg_redundancy_score=avg_redundancy_score,
            avg_diversity_score=avg_diversity_score,
            avg_query_entity_coverage_score=avg_query_entity_coverage_score,
            avg_query_concept_coverage_score=avg_query_concept_coverage_score,
            avg_centrality_grounding_score=avg_centrality_grounding_score,
            avg_grounding_quality=avg_grounding_quality,
            concept_aware_grounding_score=concept_aware_grounding_score,
            concept_aware_ranking_ndcg=concept_aware_ranking_ndcg,
            ranking_strategy_comparison=ranking_comparison or {},
            retrieval_score=retrieval_score,
            generation_score=generation_score,
            overall_score=overall_score
        )

    def _enforce_ranking_regression_guard(
        self,
        ranking_comparison: Dict[str, Dict[str, float]],
    ) -> None:
        """Fail evaluation if concept-aware ranking quality drops below floor."""
        concept_aware = ranking_comparison.get("concept_aware", {})
        ndcg = float(concept_aware.get("ranking_ndcg", 0.0))
        if ndcg < self.min_concept_aware_ranking_ndcg:
            raise RuntimeError(
                "Concept-aware ranking nDCG regression: "
                f"{ndcg:.3f} < required floor {self.min_concept_aware_ranking_ndcg:.3f}"
            )

    def _calibrate_utility_weights_from_ground_truth(self) -> Dict[str, float]:
        """Calibrate utility weights against ground-truth concept labels."""
        questions = GroundTruthBenchmarks.QUESTIONS
        if not questions:
            return {
                "concept": 0.50,
                "centrality": 0.30,
                "novelty": 0.20,
            }

        design_points: List[Tuple[float, float, float, float]] = []
        for question in questions:
            expected_count = max(1, len(question.expected_concepts))
            required_count = len(question.required_concepts)
            required_ratio = required_count / expected_count

            richness = min(1.0, expected_count / 10.0)
            difficulty_signal = {
                "easy": 0.35,
                "medium": 0.55,
                "hard": 0.75,
            }.get(question.difficulty.lower(), 0.5)

            target = max(0.0, min(1.0, 0.70 * required_ratio + 0.30 * richness))
            design_points.append((required_ratio, difficulty_signal, richness, target))

        candidates = [i / 20.0 for i in range(1, 19)]
        best_weights = {
            "concept": 0.50,
            "centrality": 0.30,
            "novelty": 0.20,
        }
        best_error = float("inf")

        for concept_w in candidates:
            for centrality_w in candidates:
                novelty_w = 1.0 - concept_w - centrality_w
                if novelty_w <= 0.0:
                    continue

                mse = 0.0
                for concept_signal, centrality_signal, novelty_signal, target in design_points:
                    prediction = (
                        concept_w * concept_signal
                        + centrality_w * centrality_signal
                        + novelty_w * novelty_signal
                    )
                    mse += (prediction - target) ** 2
                mse /= len(design_points)

                if mse < best_error:
                    best_error = mse
                    best_weights = {
                        "concept": round(concept_w, 3),
                        "centrality": round(centrality_w, 3),
                        "novelty": round(novelty_w, 3),
                    }

        logger.info(
            "Calibrated utility weights: concept=%.3f centrality=%.3f novelty=%.3f",
            best_weights["concept"],
            best_weights["centrality"],
            best_weights["novelty"],
        )
        return best_weights

    def _bootstrap_confidence_interval(
        self,
        values: List[float],
        samples: Optional[int] = None,
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """Estimate confidence interval with bootstrap resampling."""
        if not values:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        resample_count = samples or self.bootstrap_samples
        rng = np.random.default_rng(42)
        arr = np.asarray(values, dtype=float)
        boot = []
        for _ in range(max(100, resample_count)):
            sampled = rng.choice(arr, size=len(arr), replace=True)
            boot.append(float(np.mean(sampled)))

        lower = float(np.percentile(boot, 100 * (alpha / 2)))
        upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    def _compute_citation_utility(
        self,
        citation: Any,
        query_concepts: Dict[str, List[str]],
        max_centrality: float,
        seen_concept_groups: set,
    ) -> Tuple[float, set]:
        """Compute per-citation utility from concept coverage, centrality, novelty."""
        if not hasattr(self, "_utility_weights"):
            self._utility_weights = {
                "concept": 0.50,
                "centrality": 0.30,
                "novelty": 0.20,
            }

        title = str(getattr(citation, "title", "") or "").lower()
        abstract = str(getattr(citation, "abstract", "") or "").lower()
        citation_text = f"{title} {abstract}"

        covered_groups = {
            group
            for group, terms in query_concepts.items()
            if any(term in citation_text for term in terms)
        }

        total_groups = max(1, len(query_concepts))
        concept_coverage = len(covered_groups) / total_groups

        centrality_raw = float(
            getattr(citation, "metadata", {}).get("centrality_score", 0.0)
        )
        normalized_centrality = (
            min(1.0, centrality_raw / max_centrality) if max_centrality > 0 else 0.0
        )

        new_groups = covered_groups - seen_concept_groups
        novelty = len(new_groups) / total_groups
        updated_seen = seen_concept_groups | covered_groups

        utility = (
            self._utility_weights["concept"] * concept_coverage
            + self._utility_weights["centrality"] * normalized_centrality
            + self._utility_weights["novelty"] * novelty
        )
        return max(0.0, min(1.0, utility)), updated_seen

    def _compute_ndcg(self, utilities: List[float], k: Optional[int] = None) -> float:
        """Compute nDCG@k for ranked utilities in [0, 1]."""
        if not utilities:
            return 0.0

        cutoff = len(utilities) if k is None else max(1, min(k, len(utilities)))
        ranked = utilities[:cutoff]
        ideal = sorted(utilities, reverse=True)[:cutoff]

        def _dcg(values: List[float]) -> float:
            total = 0.0
            for idx, rel in enumerate(values):
                gain = (2.0 ** max(0.0, min(1.0, rel))) - 1.0
                discount = math.log2(idx + 2)
                total += gain / discount
            return total

        dcg = _dcg(ranked)
        idcg = _dcg(ideal)
        if idcg <= 0.0:
            return 0.0
        return dcg / idcg

    async def _benchmark_aggregation_strategies(self) -> Dict[str, Dict[str, float]]:
        """Compare weighted/diversified/concept-aware across archetypes with macro metrics."""
        archetypes = {
            "clinical": {
                "query": "EEG biomarkers for epilepsy prognosis in adults",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "c1",
                                "title": "Adult epilepsy prognosis using EEG biomarkers",
                                "abstract": "Clinical cohort with sensitivity and specificity outcomes.",
                                "year": 2024,
                                "relevance_score": 0.91,
                                "metadata": {"centrality_score": 0.81},
                            },
                            {
                                "pmid": "c2",
                                "title": "Interictal EEG markers in epilepsy",
                                "abstract": "Clinical case-control study of recurrence outcomes.",
                                "year": 2022,
                                "relevance_score": 0.87,
                                "metadata": {"centrality_score": 0.45},
                            },
                        ]
                    }
                },
            },
            "method_heavy": {
                "query": "ICA and independent component methods for EEG denoising",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "m1",
                                "title": "Independent component analysis for EEG artifact rejection",
                                "abstract": "Methodological benchmark of ICA pipelines.",
                                "year": 2023,
                                "relevance_score": 0.90,
                                "metadata": {"centrality_score": 0.76},
                            },
                            {
                                "pmid": "m2",
                                "title": "Automated EEG preprocessing with ICA",
                                "abstract": "Cross-dataset method evaluation and reproducibility analysis.",
                                "year": 2021,
                                "relevance_score": 0.86,
                                "metadata": {"centrality_score": 0.52},
                            },
                        ]
                    }
                },
            },
            "outcome_heavy": {
                "query": "EEG sensitivity, specificity, and AUC outcomes for seizure detection",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "o1",
                                "title": "Seizure detection outcomes using EEG",
                                "abstract": "Sensitivity and AUC outcomes in prospective validation.",
                                "year": 2024,
                                "relevance_score": 0.92,
                                "metadata": {"centrality_score": 0.79},
                            },
                            {
                                "pmid": "o2",
                                "title": "EEG classifier specificity across cohorts",
                                "abstract": "Outcome-focused model comparison and precision metrics.",
                                "year": 2022,
                                "relevance_score": 0.84,
                                "metadata": {"centrality_score": 0.40},
                            },
                        ]
                    }
                },
            },
            "longitudinal": {
                "query": "Longitudinal cohort studies tracking EEG biomarkers over time",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "l1",
                                "title": "Longitudinal EEG cohort biomarkers",
                                "abstract": "Prospective study with repeated EEG measures and outcomes.",
                                "year": 2023,
                                "relevance_score": 0.89,
                                "metadata": {"centrality_score": 0.74},
                            },
                            {
                                "pmid": "l2",
                                "title": "Retrospective longitudinal EEG trends",
                                "abstract": "Retrospective cohort with temporal biomarker drift analysis.",
                                "year": 2020,
                                "relevance_score": 0.82,
                                "metadata": {"centrality_score": 0.39},
                            },
                        ]
                    }
                },
            },
        }

        strategy_scores: Dict[str, Dict[str, float]] = {}
        for strategy in ("weighted", "diversified", "concept_aware"):
            per_archetype: Dict[str, Dict[str, float]] = {}
            ndcg_values: List[float] = []
            utility_values: List[float] = []
            grounding_values: List[float] = []
            coverage_values: List[float] = []
            redundancy_values: List[float] = []
            centrality_values: List[float] = []

            for archetype, fixture in archetypes.items():
                query_text = fixture["query"]
                fixture_results = fixture["results"]

                aggregator = ContextAggregator(
                    relevance_threshold=0.0,
                    max_citations=10,
                    entity_min_frequency=1,
                    ranking_strategy=strategy,
                )
                aggregated = await aggregator.aggregate(query_text, fixture_results)
                stats = aggregated.statistics
                query_concepts = aggregator._extract_query_concepts(query_text)

                redundancy = float(stats.get("redundancy_score", 0.0))
                concept_coverage = float(
                    stats.get("query_concept_coverage_score", 0.0)
                )
                centrality = 0.0
                if aggregated.citations:
                    centrality = statistics.mean(
                        float(c.metadata.get("centrality_score", 0.0))
                        for c in aggregated.citations
                    )

                max_centrality = 0.0
                if aggregated.citations:
                    max_centrality = max(
                        float(c.metadata.get("centrality_score", 0.0))
                        for c in aggregated.citations
                    )

                seen_concepts: set = set()
                utilities: List[float] = []
                for citation in aggregated.citations:
                    utility, seen_concepts = self._compute_citation_utility(
                        citation=citation,
                        query_concepts=query_concepts,
                        max_centrality=max_centrality,
                        seen_concept_groups=seen_concepts,
                    )
                    utilities.append(utility)

                ranking_ndcg = self._compute_ndcg(utilities, k=5)
                mean_utility = statistics.mean(utilities) if utilities else 0.0
                grounding_quality = max(
                    0.0,
                    min(
                        1.0,
                        (0.5 * concept_coverage)
                        + (0.3 * centrality)
                        + (0.2 * (1.0 - redundancy)),
                    ),
                )

                ndcg_values.append(ranking_ndcg)
                utility_values.append(mean_utility)
                grounding_values.append(grounding_quality)
                coverage_values.append(concept_coverage)
                redundancy_values.append(redundancy)
                centrality_values.append(centrality)

                per_archetype[archetype] = {
                    "query_concept_coverage_score": concept_coverage,
                    "redundancy_score": redundancy,
                    "centrality_grounding_score": centrality,
                    "grounding_quality": grounding_quality,
                    "mean_citation_utility": mean_utility,
                    "ranking_ndcg": ranking_ndcg,
                }

            ndcg_ci = self._bootstrap_confidence_interval(ndcg_values)
            strategy_scores[strategy] = {
                "query_concept_coverage_score": statistics.mean(coverage_values),
                "redundancy_score": statistics.mean(redundancy_values),
                "centrality_grounding_score": statistics.mean(centrality_values),
                "grounding_quality": statistics.mean(grounding_values),
                "mean_citation_utility": statistics.mean(utility_values),
                "ranking_ndcg": statistics.mean(ndcg_values),
                "ranking_ndcg_ci_lower": ndcg_ci["ci_lower"],
                "ranking_ndcg_ci_upper": ndcg_ci["ci_upper"],
                "ranking_ndcg_macro": statistics.mean(ndcg_values),
                "per_archetype": per_archetype,
            }

        return strategy_scores

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_retrieval_score
    # Requirement  : `_calculate_retrieval_score` shall calculate overall retrieval performance score
    # Purpose      : Calculate overall retrieval performance score
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : results: List[RetrievalBenchmarkResult]
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
    def _calculate_retrieval_score(self, results: List[RetrievalBenchmarkResult]) -> float:
        """Calculate overall retrieval performance score."""
        if not results:
            return 0.0

        # Performance thresholds
        max_time_ms = 100.0  # 100ms target
        min_documents = 5
        min_relevance = 0.7
        min_citation_accuracy = 0.8

        scores = []

        for result in results:
            # Time score
            time_score = max(0, 100 - (result.retrieval_time_ms / max_time_ms) * 100)

            # Document score
            doc_score = min(100, (result.documents_found / min_documents) * 100)

            # Relevance score
            relevance_score = (result.relevance_score / min_relevance) * 100

            # Citation score
            citation_score = (result.citation_accuracy / min_citation_accuracy) * 100

            # Weighted average
            total_score = (time_score * 0.3) + (doc_score * 0.2) + (relevance_score * 0.3) + (citation_score * 0.2)
            scores.append(min(100, total_score))

        return statistics.mean(scores)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark._calculate_generation_score
    # Requirement  : `_calculate_generation_score` shall calculate overall generation performance score
    # Purpose      : Calculate overall generation performance score
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : results: List[GenerationBenchmarkResult]
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
    def _calculate_generation_score(self, results: List[GenerationBenchmarkResult]) -> float:
        """Calculate overall generation performance score."""
        if not results:
            return 0.0

        # Performance thresholds
        max_time_ms = 2000.0  # 2 second target
        min_length = 200
        min_citations = 3
        min_validity = 0.8

        scores = []

        for result in results:
            # Time score
            time_score = max(0, 100 - (result.generation_time_ms / max_time_ms) * 100)

            # Length score
            length_score = min(100, (result.response_length / min_length) * 100)

            # Citation count score
            citation_count_score = min(100, (result.citation_count / min_citations) * 100)

            # Quality scores
            validity_score = result.citation_validity * 100
            coherence_score = result.coherence_score * 100
            accuracy_score = result.factual_accuracy * 100

            # Weighted average
            total_score = (
                (time_score * 0.2) +
                (length_score * 0.15) +
                (citation_count_score * 0.15) +
                (validity_score * 0.2) +
                (coherence_score * 0.15) +
                (accuracy_score * 0.15)
            )

            scores.append(min(100, total_score))

        return statistics.mean(scores)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.benchmarking.EEGRAGBenchmark.export_benchmark_results
    # Requirement  : `export_benchmark_results` shall export benchmark results to JSON file
    # Purpose      : Export benchmark results to JSON file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : results: BenchmarkSuite; output_path: Path
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
    def export_benchmark_results(
        self,
        results: BenchmarkSuite,
        output_path: Path
    ):
        """Export benchmark results to JSON file."""
        export_data = {
            'summary': {
                'overall_score': results.overall_score,
                'retrieval_score': results.retrieval_score,
                'generation_score': results.generation_score,
                'avg_total_time_ms': results.avg_total_time_ms,
                'avg_citation_accuracy': results.avg_citation_accuracy,
                'avg_response_quality': results.avg_response_quality,
                'avg_redundancy_score': results.avg_redundancy_score,
                'avg_diversity_score': results.avg_diversity_score,
                'avg_query_entity_coverage_score': results.avg_query_entity_coverage_score,
                'avg_query_concept_coverage_score': results.avg_query_concept_coverage_score,
                'avg_centrality_grounding_score': results.avg_centrality_grounding_score,
                'avg_grounding_quality': results.avg_grounding_quality,
                'concept_aware_grounding_score': results.concept_aware_grounding_score,
                'concept_aware_ranking_ndcg': results.concept_aware_ranking_ndcg,
            },
            'detailed_results': {
                'retrieval': [asdict(r) for r in results.retrieval_results],
                'generation': [asdict(r) for r in results.generation_results],
                'end_to_end': [asdict(r) for r in results.end_to_end_results],
                'ranking_strategy_comparison': results.ranking_strategy_comparison,
            },
            'timestamp': time.time()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Benchmark results exported to {output_path}")
