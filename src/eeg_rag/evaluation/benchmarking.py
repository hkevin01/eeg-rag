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
        min_archetype_ndcg_by_difficulty: Optional[Dict[str, float]] = None,
        bootstrap_samples: int = 500,
        hard_archetype_utility_margin: float = 0.0,
        hard_archetype_delta_ci_alpha: float = 0.05,
        category_adaptive_safety_floors: Optional[Dict[str, Dict[str, float]]] = None,
        risk_to_step_ridge_lambda: float = 0.15,
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
        self.min_archetype_ndcg_by_difficulty = (
            min_archetype_ndcg_by_difficulty
            if min_archetype_ndcg_by_difficulty is not None
            else {
                "easy": 0.60,
                "medium": 0.55,
                "hard": 0.50,
            }
        )
        self.bootstrap_samples = bootstrap_samples
        self.hard_archetype_utility_margin = hard_archetype_utility_margin
        self.hard_archetype_delta_ci_alpha = max(
            1e-3,
            min(0.25, float(hard_archetype_delta_ci_alpha)),
        )
        self.category_adaptive_safety_floors = (
            category_adaptive_safety_floors
            if category_adaptive_safety_floors is not None
            else {
                "general": {
                    "citation_validity_floor": 0.62,
                    "hard_utility_margin": 0.0,
                },
                "clinical": {
                    "citation_validity_floor": 0.70,
                    "hard_utility_margin": 0.01,
                },
                "bci": {
                    "citation_validity_floor": 0.68,
                    "hard_utility_margin": 0.008,
                },
            }
        )
        self.risk_to_step_ridge_lambda = max(1e-5, float(risk_to_step_ridge_lambda))

        self.citation_verifier = CitationVerifier(enable_medical_validation=True)
        self.performance_monitor = PerformanceMonitor()
        self._utility_weights = self._calibrate_utility_weights_from_ground_truth()
        self._calibration_drift_state: Dict[str, Any] = {
            "baseline_mae": None,
            "last_mae": None,
            "last_relative_shift": 0.0,
            "drift_detected": False,
            "recalibration_recommended": False,
            "checked_at": None,
        }

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

        per_archetype = concept_aware.get("per_archetype", {})
        failing_archetypes: List[str] = []
        for archetype_name, metrics in per_archetype.items():
            difficulty = str(metrics.get("difficulty", "medium")).lower()
            floor = float(
                self.min_archetype_ndcg_by_difficulty.get(
                    difficulty,
                    self.min_concept_aware_ranking_ndcg,
                )
            )
            archetype_ndcg = float(metrics.get("ranking_ndcg", 0.0))
            if archetype_ndcg < floor:
                failing_archetypes.append(
                    f"{archetype_name}({difficulty}): {archetype_ndcg:.3f} < {floor:.3f}"
                )

        if failing_archetypes:
            raise RuntimeError(
                "Per-archetype ranking regression detected: "
                + "; ".join(failing_archetypes)
            )

        self._enforce_uncertainty_adjusted_utility_guard(ranking_comparison)
        self._enforce_adaptive_safety_guards(ranking_comparison)

    def _enforce_adaptive_safety_guards(
        self,
        ranking_comparison: Dict[str, Dict[str, float]],
    ) -> None:
        """Fail evaluation when adaptive safety validators indicate regression."""
        concept = ranking_comparison.get("concept_aware", {})

        monotonic = concept.get("monotonic_safety_response")
        if isinstance(monotonic, dict) and not bool(monotonic.get("valid", True)):
            failing = monotonic.get("failing", [])
            raise RuntimeError(
                "Monotonic safety response regression detected: "
                + "; ".join(str(item) for item in failing)
            )

        temporal = concept.get("temporal_forgetting_validation")
        if isinstance(temporal, dict) and not bool(temporal.get("valid", True)):
            raise RuntimeError(
                "Temporal forgetting safety regression detected: "
                f"hard_utility_delta={float(temporal.get('hard_utility_delta', 0.0)):.3f}, "
                f"citation_validity_delta={float(temporal.get('citation_validity_delta', 0.0)):.3f}, "
                f"citation_validity_floor={float(temporal.get('citation_validity_floor', 0.0)):.3f}"
            )

    def _enforce_uncertainty_adjusted_utility_guard(
        self,
        ranking_comparison: Dict[str, Dict[str, float]],
    ) -> None:
        """Fail when concept-aware underperforms baseline on hard archetypes."""
        concept = ranking_comparison.get("concept_aware", {})
        baseline = ranking_comparison.get("weighted", {})

        concept_hard = float(
            concept.get("hard_archetype_uncertainty_adjusted_utility", 0.0)
        )
        baseline_hard = float(
            baseline.get("hard_archetype_uncertainty_adjusted_utility", 0.0)
        )
        delta = concept_hard - baseline_hard
        delta_ci = concept.get("hard_archetype_utility_delta_ci", {})
        ci_lower = float(delta_ci.get("ci_lower", delta))
        ci_upper = float(delta_ci.get("ci_upper", delta))
        ci_samples = int(delta_ci.get("samples", 0))

        if ci_samples > 0 and ci_lower < self.hard_archetype_utility_margin:
            raise RuntimeError(
                "Confidence-bounded utility regression on hard archetypes: "
                f"concept_aware={concept_hard:.3f}, baseline={baseline_hard:.3f}, "
                f"delta={delta:.3f}, ci=[{ci_lower:.3f}, {ci_upper:.3f}] < "
                f"required_margin={self.hard_archetype_utility_margin:.3f}"
            )
        if ci_samples <= 0 and delta < self.hard_archetype_utility_margin:
            raise RuntimeError(
                "Uncertainty-adjusted utility regression on hard archetypes: "
                f"concept_aware={concept_hard:.3f}, baseline={baseline_hard:.3f}, "
                f"delta={delta:.3f} < required_margin={self.hard_archetype_utility_margin:.3f}"
            )

    def _validate_monotonic_safety_response(
        self,
        hard_archetype_profiles: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Validate step-radius contraction as regret/drift risk increases."""
        failing: List[str] = []
        for archetype, profile in hard_archetype_profiles.items():
            low = float(profile.get("low_risk_max_step", 0.0))
            medium = float(profile.get("medium_risk_max_step", 0.0))
            high = float(profile.get("high_risk_max_step", 0.0))
            if not (low >= medium >= high):
                failing.append(
                    f"{archetype}: low={low:.3f}, medium={medium:.3f}, high={high:.3f}"
                )

        return {
            "valid": len(failing) == 0,
            "checked": len(hard_archetype_profiles),
            "failing": failing,
        }

    def _validate_temporal_forgetting_safety(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        citation_validity_floor: float = 0.62,
        before_per_archetype: Optional[Dict[str, Dict[str, float]]] = None,
        after_per_archetype: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Check forgetting schedule improves hard utility without citation harm."""
        before_hard = float(before.get("hard_archetype_uncertainty_adjusted_utility", 0.0))
        after_hard = float(after.get("hard_archetype_uncertainty_adjusted_utility", 0.0))
        before_valid = float(before.get("citation_validity_proxy", 0.0))
        after_valid = float(after.get("citation_validity_proxy", 0.0))

        floors = getattr(
            self,
            "category_adaptive_safety_floors",
            {
                "general": {
                    "citation_validity_floor": citation_validity_floor,
                    "hard_utility_margin": 0.0,
                }
            },
        )
        default_floor = floors.get(
            "general",
            {
                "citation_validity_floor": citation_validity_floor,
                "hard_utility_margin": 0.0,
            },
        )

        hard_improved = after_hard >= (
            before_hard + float(default_floor.get("hard_utility_margin", 0.0))
        )
        citation_safe = (
            after_valid >= float(default_floor.get("citation_validity_floor", citation_validity_floor))
            and after_valid >= (before_valid - 0.01)
        )

        category_checks: Dict[str, Dict[str, Any]] = {}
        if before_per_archetype and after_per_archetype:
            categories = set()
            for payload in before_per_archetype.values():
                if str(payload.get("difficulty", "medium")).lower() == "hard":
                    categories.add(str(payload.get("category", "general")).lower())
            for payload in after_per_archetype.values():
                if str(payload.get("difficulty", "medium")).lower() == "hard":
                    categories.add(str(payload.get("category", "general")).lower())

            for category in sorted(categories):
                before_vals = [
                    payload
                    for payload in before_per_archetype.values()
                    if str(payload.get("difficulty", "medium")).lower() == "hard"
                    and str(payload.get("category", "general")).lower() == category
                ]
                after_vals = [
                    payload
                    for payload in after_per_archetype.values()
                    if str(payload.get("difficulty", "medium")).lower() == "hard"
                    and str(payload.get("category", "general")).lower() == category
                ]
                if not before_vals or not after_vals:
                    continue

                before_cat_utility = statistics.mean(
                    float(v.get("uncertainty_adjusted_utility", 0.0))
                    for v in before_vals
                )
                after_cat_utility = statistics.mean(
                    float(v.get("uncertainty_adjusted_utility", 0.0))
                    for v in after_vals
                )
                before_cat_validity = statistics.mean(
                    float(v.get("citation_validity_proxy", 0.0))
                    for v in before_vals
                )
                after_cat_validity = statistics.mean(
                    float(v.get("citation_validity_proxy", 0.0))
                    for v in after_vals
                )

                category_floor = floors.get(category, default_floor)
                required_margin = float(category_floor.get("hard_utility_margin", 0.0))
                required_validity_floor = float(
                    category_floor.get("citation_validity_floor", citation_validity_floor)
                )

                category_hard_improved = after_cat_utility >= (
                    before_cat_utility + required_margin
                )
                category_citation_safe = (
                    after_cat_validity >= required_validity_floor
                    and after_cat_validity >= (before_cat_validity - 0.01)
                )
                category_valid = bool(category_hard_improved and category_citation_safe)
                category_checks[category] = {
                    "valid": category_valid,
                    "hard_utility_delta": after_cat_utility - before_cat_utility,
                    "citation_validity_delta": after_cat_validity - before_cat_validity,
                    "required_hard_utility_margin": required_margin,
                    "required_citation_validity_floor": required_validity_floor,
                    "hard_utility_before": before_cat_utility,
                    "hard_utility_after": after_cat_utility,
                    "citation_validity_before": before_cat_validity,
                    "citation_validity_after": after_cat_validity,
                }

        categories_valid = all(
            bool(payload.get("valid", True))
            for payload in category_checks.values()
        )
        overall_valid = bool(hard_improved and citation_safe and categories_valid)

        return {
            "valid": overall_valid,
            "hard_utility_delta": after_hard - before_hard,
            "citation_validity_delta": after_valid - before_valid,
            "citation_validity_floor": float(
                default_floor.get("citation_validity_floor", citation_validity_floor)
            ),
            "required_hard_utility_margin": float(
                default_floor.get("hard_utility_margin", 0.0)
            ),
            "hard_utility_before": before_hard,
            "hard_utility_after": after_hard,
            "citation_validity_before": before_valid,
            "citation_validity_after": after_valid,
            "category_checks": category_checks,
            "applied_category_floors": floors,
        }

    @staticmethod
    def _risk_score(
        utility: float,
        redundancy: float,
        drift_norm: float,
    ) -> float:
        """Compute normalized risk score from utility, redundancy and drift."""
        return max(
            0.0,
            min(
                1.0,
                (0.50 * (1.0 - utility))
                + (0.30 * redundancy)
                + (0.20 * drift_norm),
            ),
        )

    def _learn_risk_to_step_mapping(
        self,
        trajectories: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Learn step mapping from held-out hard-archetype trajectories."""
        if len(trajectories) < 2:
            return {
                "coefficients": {
                    "low": [0.22, -0.10],
                    "medium_ratio": [0.80, -0.15],
                    "high_ratio": [0.70, -0.20],
                },
                "heldout_mse": 0.0,
                "train_samples": len(trajectories),
                "heldout_samples": 0,
                "used_fallback": True,
            }

        train = [row for idx, row in enumerate(trajectories) if idx % 3 != 0]
        heldout = [row for idx, row in enumerate(trajectories) if idx % 3 == 0]
        if not train:
            train = trajectories
            heldout = []

        ridge_lambda = getattr(self, "risk_to_step_ridge_lambda", 0.15)

        def _fit_linear(
            rows: List[Dict[str, float]],
            y_key: str,
            default: List[float],
        ) -> np.ndarray:
            x = np.asarray(
                [[1.0, float(row.get("risk_score", 0.0))] for row in rows],
                dtype=float,
            )
            y = np.asarray([float(row.get(y_key, 0.0)) for row in rows], dtype=float)
            xtx = (x.T @ x) + (ridge_lambda * np.eye(2, dtype=float))
            try:
                beta = np.linalg.solve(xtx, x.T @ y)
            except np.linalg.LinAlgError:
                beta = np.asarray(default, dtype=float)
            return beta

        low_beta = _fit_linear(train, "target_low_step", [0.22, -0.10])
        med_beta = _fit_linear(train, "target_medium_ratio", [0.80, -0.15])
        high_beta = _fit_linear(train, "target_high_ratio", [0.70, -0.20])

        heldout_errors: List[float] = []
        for row in heldout:
            r = float(row.get("risk_score", 0.0))
            pred_low = float(low_beta[0] + (low_beta[1] * r))
            pred_med = float(med_beta[0] + (med_beta[1] * r))
            pred_high = float(high_beta[0] + (high_beta[1] * r))
            heldout_errors.extend(
                [
                    (pred_low - float(row.get("target_low_step", 0.0))) ** 2,
                    (pred_med - float(row.get("target_medium_ratio", 0.0))) ** 2,
                    (pred_high - float(row.get("target_high_ratio", 0.0))) ** 2,
                ]
            )

        return {
            "coefficients": {
                "low": [float(low_beta[0]), float(low_beta[1])],
                "medium_ratio": [float(med_beta[0]), float(med_beta[1])],
                "high_ratio": [float(high_beta[0]), float(high_beta[1])],
            },
            "heldout_mse": float(statistics.mean(heldout_errors)) if heldout_errors else 0.0,
            "train_samples": len(train),
            "heldout_samples": len(heldout),
            "used_fallback": False,
        }

    def _derive_hard_archetype_safety_profiles(
        self,
        per_archetype: Dict[str, Dict[str, float]],
        drift_summary: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        """Estimate step-radius profiles for hard archetypes by risk level."""
        drift_shift = max(0.0, float(drift_summary.get("relative_shift", 0.0)))
        drift_norm = min(1.0, drift_shift / 0.5)

        trajectories: List[Dict[str, float]] = []
        for archetype_name, metrics in per_archetype.items():
            difficulty = str(metrics.get("difficulty", "medium")).lower()
            if difficulty != "hard":
                continue
            utility = float(metrics.get("uncertainty_adjusted_utility", 0.0))
            redundancy = float(metrics.get("redundancy_score", 0.0))
            citation_validity = float(metrics.get("citation_validity_proxy", 0.0))
            risk_score = self._risk_score(utility, redundancy, drift_norm)

            target_low_step = max(
                0.06,
                min(
                    0.22,
                    0.08
                    + (0.16 * utility)
                    + (0.06 * citation_validity)
                    - (0.08 * drift_norm)
                    - (0.05 * redundancy),
                ),
            )
            target_medium_ratio = max(0.45, min(0.90, 0.82 - (0.18 * risk_score)))
            target_high_ratio = max(0.35, min(0.85, 0.72 - (0.22 * risk_score)))
            trajectories.append(
                {
                    "archetype": archetype_name,
                    "risk_score": risk_score,
                    "target_low_step": target_low_step,
                    "target_medium_ratio": target_medium_ratio,
                    "target_high_ratio": target_high_ratio,
                }
            )

        model = self._learn_risk_to_step_mapping(trajectories)
        coeffs = model.get("coefficients", {})
        low_beta = coeffs.get("low", [0.22, -0.10])
        med_beta = coeffs.get("medium_ratio", [0.80, -0.15])
        high_beta = coeffs.get("high_ratio", [0.70, -0.20])

        profiles: Dict[str, Dict[str, float]] = {}
        for archetype_name, metrics in per_archetype.items():
            difficulty = str(metrics.get("difficulty", "medium")).lower()
            if difficulty != "hard":
                continue

            utility = float(metrics.get("uncertainty_adjusted_utility", 0.0))
            redundancy = float(metrics.get("redundancy_score", 0.0))
            risk_score = self._risk_score(utility, redundancy, drift_norm)

            low = max(0.06, min(0.24, float(low_beta[0]) + (float(low_beta[1]) * risk_score)))
            medium_ratio = max(
                0.45,
                min(0.90, float(med_beta[0]) + (float(med_beta[1]) * risk_score)),
            )
            high_ratio = max(
                0.35,
                min(0.85, float(high_beta[0]) + (float(high_beta[1]) * risk_score)),
            )
            medium = max(0.04, low * medium_ratio)
            high = max(0.03, medium * high_ratio)
            if medium > low:
                medium = low
            if high > medium:
                high = medium

            profiles[archetype_name] = {
                "risk_score": risk_score,
                "low_risk_max_step": low,
                "medium_risk_max_step": medium,
                "high_risk_max_step": high,
            }

        return profiles, model

    def _hard_archetype_delta_summary(
        self,
        weighted: Dict[str, Any],
        concept_aware: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute bootstrap delta intervals for hard-archetype utility uplift."""
        weighted_per = weighted.get("per_archetype", {})
        concept_per = concept_aware.get("per_archetype", {})

        deltas: List[float] = []
        deltas_by_category: Dict[str, List[float]] = defaultdict(list)
        for name, concept_metrics in concept_per.items():
            if str(concept_metrics.get("difficulty", "medium")).lower() != "hard":
                continue
            weighted_metrics = weighted_per.get(name)
            if not weighted_metrics:
                continue
            delta = float(concept_metrics.get("uncertainty_adjusted_utility", 0.0)) - float(
                weighted_metrics.get("uncertainty_adjusted_utility", 0.0)
            )
            category = str(concept_metrics.get("category", "general")).lower()
            deltas.append(delta)
            deltas_by_category[category].append(delta)

        ci = self._bootstrap_confidence_interval(
            deltas,
            alpha=getattr(self, "hard_archetype_delta_ci_alpha", 0.05),
        )
        ci.update({"samples": len(deltas)})

        by_category = {
            category: {
                **self._bootstrap_confidence_interval(
                    values,
                    alpha=getattr(self, "hard_archetype_delta_ci_alpha", 0.05),
                ),
                "samples": len(values),
            }
            for category, values in deltas_by_category.items()
        }
        return {
            "overall": ci,
            "by_category": by_category,
        }

    def _monitor_calibration_drift(
        self,
        predicted_utilities: List[float],
        target_grounding: List[float],
    ) -> Dict[str, Any]:
        """Detect drift indicating utility-weight recalibration is needed."""
        if not predicted_utilities or not target_grounding:
            return {
                "mae": 0.0,
                "relative_shift": 0.0,
                "drift_detected": False,
                "recalibration_recommended": False,
            }

        mae = statistics.mean(
            abs(p - t)
            for p, t in zip(predicted_utilities, target_grounding)
        )
        baseline = self._calibration_drift_state.get("baseline_mae")
        if baseline is None:
            baseline = mae
            self._calibration_drift_state["baseline_mae"] = baseline

        relative_shift = (mae - baseline) / max(1e-6, baseline)
        drift_detected = bool(mae > 0.18 or relative_shift > 0.25)

        self._calibration_drift_state.update(
            {
                "last_mae": mae,
                "last_relative_shift": relative_shift,
                "drift_detected": drift_detected,
                "recalibration_recommended": drift_detected,
                "checked_at": time.time(),
            }
        )

        return {
            "mae": mae,
            "relative_shift": relative_shift,
            "drift_detected": drift_detected,
            "recalibration_recommended": drift_detected,
        }

    def _create_archetype_fixture_bank(self) -> List[Dict[str, Any]]:
        """Create a benchmark fixture bank spanning categories and difficulties."""
        return [
            {
                "name": "clinical_epilepsy",
                "category": "clinical",
                "difficulty": "medium",
                "query": "EEG biomarkers for epilepsy prognosis and seizure outcomes",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "a1",
                                "title": "Epilepsy prognosis using EEG biomarkers",
                                "abstract": "Clinical cohort with sensitivity and specificity outcomes.",
                                "year": 2024,
                                "relevance_score": 0.91,
                                "metadata": {"centrality_score": 0.81},
                            },
                            {
                                "pmid": "a2",
                                "title": "Interictal EEG markers in epilepsy",
                                "abstract": "Case-control design with recurrence outcome endpoints.",
                                "year": 2022,
                                "relevance_score": 0.86,
                                "metadata": {"centrality_score": 0.49},
                            },
                        ]
                    }
                },
            },
            {
                "name": "clinical_sleep",
                "category": "clinical",
                "difficulty": "easy",
                "query": "Sleep staging EEG markers in clinical cohorts",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "b1",
                                "title": "Sleep spindle biomarkers in mild cognitive impairment",
                                "abstract": "Clinical longitudinal outcomes with sleep spindle reduction.",
                                "year": 2023,
                                "relevance_score": 0.88,
                                "metadata": {"centrality_score": 0.70},
                            },
                            {
                                "pmid": "b2",
                                "title": "EEG sleep stage classification outcomes",
                                "abstract": "Sensitivity and AUC outcomes in sleep clinic population.",
                                "year": 2021,
                                "relevance_score": 0.84,
                                "metadata": {"centrality_score": 0.41},
                            },
                        ]
                    }
                },
            },
            {
                "name": "method_ica",
                "category": "method",
                "difficulty": "hard",
                "query": "Independent component analysis methods for EEG artifact rejection",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "c1",
                                "title": "Independent component analysis for EEG denoising",
                                "abstract": "Method benchmark for ICA preprocessing pipelines.",
                                "year": 2024,
                                "relevance_score": 0.90,
                                "metadata": {"centrality_score": 0.74},
                            },
                            {
                                "pmid": "c2",
                                "title": "ICA reproducibility in EEG preprocessing",
                                "abstract": "Cross-sectional method reproducibility with accuracy outcomes.",
                                "year": 2022,
                                "relevance_score": 0.85,
                                "metadata": {"centrality_score": 0.46},
                            },
                        ]
                    }
                },
            },
            {
                "name": "method_connectivity",
                "category": "method",
                "difficulty": "hard",
                "query": "EEG connectivity estimation methods and graph design choices",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "d1",
                                "title": "Graph-based EEG connectivity methods",
                                "abstract": "Method design with longitudinal validation cohorts.",
                                "year": 2023,
                                "relevance_score": 0.87,
                                "metadata": {"centrality_score": 0.66},
                            },
                            {
                                "pmid": "d2",
                                "title": "Functional connectivity in EEG",
                                "abstract": "Cross-sectional method outcomes for network robustness.",
                                "year": 2020,
                                "relevance_score": 0.82,
                                "metadata": {"centrality_score": 0.38},
                            },
                        ]
                    }
                },
            },
            {
                "name": "outcome_sensitivity",
                "category": "outcome",
                "difficulty": "medium",
                "query": "EEG seizure detection sensitivity and specificity outcomes",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "e1",
                                "title": "Seizure detection outcomes with EEG biomarkers",
                                "abstract": "Sensitivity and specificity in prospective validation cohorts.",
                                "year": 2024,
                                "relevance_score": 0.92,
                                "metadata": {"centrality_score": 0.79},
                            },
                            {
                                "pmid": "e2",
                                "title": "Outcome metrics for EEG seizure models",
                                "abstract": "AUC and precision-recall outcomes in retrospective design.",
                                "year": 2021,
                                "relevance_score": 0.85,
                                "metadata": {"centrality_score": 0.44},
                            },
                        ]
                    }
                },
            },
            {
                "name": "outcome_auc",
                "category": "outcome",
                "difficulty": "medium",
                "query": "AUC outcomes for EEG biomarker classification studies",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "f1",
                                "title": "AUC benchmarking for EEG biomarkers",
                                "abstract": "Outcome-focused RCT and cohort evaluations.",
                                "year": 2023,
                                "relevance_score": 0.89,
                                "metadata": {"centrality_score": 0.68},
                            },
                            {
                                "pmid": "f2",
                                "title": "EEG classifier performance outcomes",
                                "abstract": "Cross-sectional design with specificity and recall.",
                                "year": 2019,
                                "relevance_score": 0.81,
                                "metadata": {"centrality_score": 0.35},
                            },
                        ]
                    }
                },
            },
            {
                "name": "longitudinal_cohort",
                "category": "longitudinal",
                "difficulty": "hard",
                "query": "Longitudinal EEG cohort biomarkers and prognosis outcomes",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "g1",
                                "title": "Longitudinal EEG cohort biomarkers",
                                "abstract": "Prospective longitudinal outcome study over three years.",
                                "year": 2024,
                                "relevance_score": 0.90,
                                "metadata": {"centrality_score": 0.77},
                            },
                            {
                                "pmid": "g2",
                                "title": "Temporal trends in EEG biomarkers",
                                "abstract": "Retrospective longitudinal outcomes with survival endpoints.",
                                "year": 2020,
                                "relevance_score": 0.83,
                                "metadata": {"centrality_score": 0.40},
                            },
                        ]
                    }
                },
            },
            {
                "name": "longitudinal_pediatric",
                "category": "longitudinal",
                "difficulty": "medium",
                "query": "Pediatric longitudinal EEG developmental trajectories",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "h1",
                                "title": "Pediatric longitudinal EEG development",
                                "abstract": "Longitudinal cohort outcomes in pediatric epilepsy.",
                                "year": 2022,
                                "relevance_score": 0.87,
                                "metadata": {"centrality_score": 0.62},
                            },
                            {
                                "pmid": "h2",
                                "title": "Developmental EEG outcomes in childhood",
                                "abstract": "Prospective pediatric design tracking EEG biomarkers.",
                                "year": 2021,
                                "relevance_score": 0.84,
                                "metadata": {"centrality_score": 0.43},
                            },
                        ]
                    }
                },
            },
            {
                "name": "bci_motor_imagery",
                "category": "bci",
                "difficulty": "hard",
                "query": "Motor imagery EEG BCI outcomes and classifier sensitivity",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "i1",
                                "title": "Motor imagery BCI outcome evaluation",
                                "abstract": "Sensitivity outcomes and longitudinal adaptation design.",
                                "year": 2024,
                                "relevance_score": 0.88,
                                "metadata": {"centrality_score": 0.65},
                            },
                            {
                                "pmid": "i2",
                                "title": "EEG BCI classifier performance",
                                "abstract": "Cross-sectional design with AUC and recall outcomes.",
                                "year": 2021,
                                "relevance_score": 0.82,
                                "metadata": {"centrality_score": 0.37},
                            },
                        ]
                    }
                },
            },
            {
                "name": "erp_p300",
                "category": "erp",
                "difficulty": "medium",
                "query": "P300 latency outcomes in clinical EEG oddball paradigms",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "j1",
                                "title": "P300 clinical outcomes in EEG",
                                "abstract": "Case-control outcomes with latency and amplitude sensitivity.",
                                "year": 2023,
                                "relevance_score": 0.88,
                                "metadata": {"centrality_score": 0.64},
                            },
                            {
                                "pmid": "j2",
                                "title": "ERP biomarker outcomes in oddball tasks",
                                "abstract": "Cross-sectional design and specificity estimates.",
                                "year": 2020,
                                "relevance_score": 0.83,
                                "metadata": {"centrality_score": 0.39},
                            },
                        ]
                    }
                },
            },
            {
                "name": "pathology_dementia",
                "category": "pathology",
                "difficulty": "medium",
                "query": "EEG biomarkers and outcomes in dementia cohorts",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "k1",
                                "title": "Dementia EEG outcomes and biomarkers",
                                "abstract": "Prospective cohort outcomes for dementia progression.",
                                "year": 2024,
                                "relevance_score": 0.90,
                                "metadata": {"centrality_score": 0.73},
                            },
                            {
                                "pmid": "k2",
                                "title": "EEG slowing in dementia",
                                "abstract": "Cross-sectional design and clinical outcome measures.",
                                "year": 2019,
                                "relevance_score": 0.80,
                                "metadata": {"centrality_score": 0.34},
                            },
                        ]
                    }
                },
            },
            {
                "name": "preprocessing_artifact",
                "category": "preprocessing",
                "difficulty": "easy",
                "query": "Artifact rejection methods and quality outcomes in EEG preprocessing",
                "results": {
                    "local": {
                        "data": [
                            {
                                "pmid": "l1",
                                "title": "EEG artifact rejection outcomes",
                                "abstract": "Method outcomes with ICA and filtering sensitivity metrics.",
                                "year": 2023,
                                "relevance_score": 0.87,
                                "metadata": {"centrality_score": 0.60},
                            },
                            {
                                "pmid": "l2",
                                "title": "Preprocessing design in EEG",
                                "abstract": "Cross-sectional method comparison with specificity outcomes.",
                                "year": 2020,
                                "relevance_score": 0.82,
                                "metadata": {"centrality_score": 0.36},
                            },
                        ]
                    }
                },
            },
        ]

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

    @staticmethod
    def _citation_metadata_completeness(citations: List[Any]) -> float:
        """Estimate completeness ratio for citation metadata fields."""
        if not citations:
            return 0.0

        required_fields = ("pmid", "title", "year", "doi")
        completeness_scores: List[float] = []
        for citation in citations:
            metadata = getattr(citation, "metadata", {}) or {}
            year = (
                getattr(citation, "year", None)
                if hasattr(citation, "year")
                else metadata.get("year")
            )
            score = 0.0
            if getattr(citation, "pmid", None):
                score += 1.0
            if getattr(citation, "title", None):
                score += 1.0
            if year is not None:
                score += 1.0
            if getattr(citation, "doi", None) or metadata.get("doi"):
                score += 1.0
            completeness_scores.append(score / len(required_fields))

        return float(statistics.mean(completeness_scores))

    async def _benchmark_aggregation_strategies(self) -> Dict[str, Dict[str, float]]:
        """Compare weighted/diversified/concept-aware across archetypes with macro metrics."""
        archetypes = self._create_archetype_fixture_bank()

        strategy_scores: Dict[str, Dict[str, float]] = {}
        for strategy in ("weighted", "diversified", "concept_aware"):
            per_archetype: Dict[str, Dict[str, float]] = {}
            ndcg_values: List[float] = []
            utility_values: List[float] = []
            uncertainty_adjusted_utility_values: List[float] = []
            citation_count_values: List[float] = []
            metadata_completeness_values: List[float] = []
            grounding_values: List[float] = []
            coverage_values: List[float] = []
            redundancy_values: List[float] = []
            centrality_values: List[float] = []
            ndcg_by_category: Dict[str, List[float]] = defaultdict(list)
            ndcg_by_difficulty: Dict[str, List[float]] = defaultdict(list)
            hard_uncertainty_adjusted_utility_values: List[float] = []
            citation_validity_values: List[float] = []

            for fixture in archetypes:
                archetype_name = str(fixture["name"])
                category = str(fixture["category"])
                difficulty = str(fixture["difficulty"])
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

                citation_count = float(len(aggregated.citations))
                metadata_completeness = self._citation_metadata_completeness(
                    aggregated.citations
                )

                ranking_ndcg = self._compute_ndcg(utilities, k=5)
                mean_utility = statistics.mean(utilities) if utilities else 0.0
                utility_dispersion = statistics.pstdev(utilities) if len(utilities) > 1 else 0.0
                uncertainty_adjusted_utility = max(
                    0.0,
                    min(1.0, mean_utility - (0.5 * utility_dispersion)),
                )
                grounding_quality = max(
                    0.0,
                    min(
                        1.0,
                        (0.5 * concept_coverage)
                        + (0.3 * centrality)
                        + (0.2 * (1.0 - redundancy)),
                    ),
                )
                citation_validity_proxy = max(
                    0.0,
                    min(
                        1.0,
                        (0.55 * concept_coverage)
                        + (0.25 * centrality)
                        + (0.20 * (1.0 - redundancy)),
                    ),
                )

                ndcg_values.append(ranking_ndcg)
                ndcg_by_category[category].append(ranking_ndcg)
                ndcg_by_difficulty[difficulty].append(ranking_ndcg)
                utility_values.append(mean_utility)
                uncertainty_adjusted_utility_values.append(uncertainty_adjusted_utility)
                citation_count_values.append(citation_count)
                metadata_completeness_values.append(metadata_completeness)
                grounding_values.append(grounding_quality)
                coverage_values.append(concept_coverage)
                redundancy_values.append(redundancy)
                centrality_values.append(centrality)
                if difficulty.lower() == "hard":
                    hard_uncertainty_adjusted_utility_values.append(
                        uncertainty_adjusted_utility
                    )
                citation_validity_values.append(citation_validity_proxy)

                per_archetype[archetype_name] = {
                    "category": category,
                    "difficulty": difficulty,
                    "query_concept_coverage_score": concept_coverage,
                    "redundancy_score": redundancy,
                    "centrality_grounding_score": centrality,
                    "grounding_quality": grounding_quality,
                    "mean_citation_utility": mean_utility,
                    "uncertainty_adjusted_utility": uncertainty_adjusted_utility,
                    "citation_validity_proxy": citation_validity_proxy,
                    "citation_count": citation_count,
                    "metadata_completeness": metadata_completeness,
                    "ranking_ndcg": ranking_ndcg,
                }

            ndcg_ci = self._bootstrap_confidence_interval(ndcg_values)
            drift_summary = self._monitor_calibration_drift(
                predicted_utilities=utility_values,
                target_grounding=grounding_values,
            )
            hard_safety_profiles, risk_to_step_model = self._derive_hard_archetype_safety_profiles(
                per_archetype=per_archetype,
                drift_summary=drift_summary,
            )
            monotonic_validation = self._validate_monotonic_safety_response(
                hard_safety_profiles
            )
            strategy_scores[strategy] = {
                "query_concept_coverage_score": statistics.mean(coverage_values),
                "redundancy_score": statistics.mean(redundancy_values),
                "centrality_grounding_score": statistics.mean(centrality_values),
                "grounding_quality": statistics.mean(grounding_values),
                "mean_citation_utility": statistics.mean(utility_values),
                "uncertainty_adjusted_utility": statistics.mean(
                    uncertainty_adjusted_utility_values
                ),
                "total_papers_evaluated": int(sum(citation_count_values)),
                "avg_papers_per_archetype": statistics.mean(citation_count_values),
                "metadata_completeness_rate": statistics.mean(
                    metadata_completeness_values
                ),
                "hard_archetype_uncertainty_adjusted_utility": (
                    statistics.mean(hard_uncertainty_adjusted_utility_values)
                    if hard_uncertainty_adjusted_utility_values
                    else 0.0
                ),
                "citation_validity_proxy": statistics.mean(citation_validity_values),
                "ranking_ndcg": statistics.mean(ndcg_values),
                "ranking_ndcg_ci_lower": ndcg_ci["ci_lower"],
                "ranking_ndcg_ci_upper": ndcg_ci["ci_upper"],
                "ranking_ndcg_macro": statistics.mean(ndcg_values),
                "stratified_macro_ndcg_by_category": {
                    key: statistics.mean(value)
                    for key, value in ndcg_by_category.items()
                },
                "stratified_macro_ndcg_by_difficulty": {
                    key: statistics.mean(value)
                    for key, value in ndcg_by_difficulty.items()
                },
                "calibration_drift": drift_summary,
                "hard_archetype_safety_profiles": hard_safety_profiles,
                "risk_to_step_model": risk_to_step_model,
                "monotonic_safety_response": monotonic_validation,
                "per_archetype": per_archetype,
            }

        if "weighted" in strategy_scores and "concept_aware" in strategy_scores:
            delta_summary = self._hard_archetype_delta_summary(
                weighted=strategy_scores["weighted"],
                concept_aware=strategy_scores["concept_aware"],
            )
            strategy_scores["concept_aware"]["hard_archetype_utility_delta_ci"] = (
                delta_summary["overall"]
            )
            strategy_scores["concept_aware"]["hard_archetype_utility_delta_by_category"] = (
                delta_summary["by_category"]
            )
            strategy_scores["concept_aware"]["temporal_forgetting_validation"] = (
                self._validate_temporal_forgetting_safety(
                    before=strategy_scores["weighted"],
                    after=strategy_scores["concept_aware"],
                    before_per_archetype=strategy_scores["weighted"].get("per_archetype", {}),
                    after_per_archetype=strategy_scores["concept_aware"].get("per_archetype", {}),
                )
            )

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
                'concept_aware_ranking_ndcg_ci': {
                    'lower': results.ranking_strategy_comparison.get('concept_aware', {}).get('ranking_ndcg_ci_lower', 0.0),
                    'upper': results.ranking_strategy_comparison.get('concept_aware', {}).get('ranking_ndcg_ci_upper', 0.0),
                },
                'concept_aware_calibration_drift': results.ranking_strategy_comparison.get('concept_aware', {}).get('calibration_drift', {}),
                'concept_aware_hard_utility_delta_ci': results.ranking_strategy_comparison.get('concept_aware', {}).get('hard_archetype_utility_delta_ci', {}),
                'adaptive_safety': {
                    'monotonic_safety_response': results.ranking_strategy_comparison.get('concept_aware', {}).get('monotonic_safety_response', {}),
                    'temporal_forgetting_validation': results.ranking_strategy_comparison.get('concept_aware', {}).get('temporal_forgetting_validation', {}),
                    'risk_to_step_model': results.ranking_strategy_comparison.get('concept_aware', {}).get('risk_to_step_model', {}),
                    'hard_archetype_utility_delta_by_category': results.ranking_strategy_comparison.get('concept_aware', {}).get('hard_archetype_utility_delta_by_category', {}),
                },
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
