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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

# Import EEG-RAG components
from ..agents.base_agent import AgentQuery, QueryComplexity
from ..agents.orchestrator.orchestrator_agent import OrchestratorAgent
from ..agents.local_agent.local_data_agent import LocalDataAgent
from ..agents.web_agent.web_research_agent import WebResearchAgent
from ..verification.citation_verifier import CitationVerifier
from ..monitoring import PerformanceMonitor, monitor_performance

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """Query with expected characteristics for benchmarking."""
    query_text: str
    complexity: QueryComplexity
    expected_topics: List[str]
    expected_citations_min: int
    expected_response_length_min: int
    category: str
    

@dataclass
class RetrievalBenchmarkResult:
    """Results from retrieval benchmarking."""
    query: str
    retrieval_time_ms: float
    documents_found: int
    relevance_score: float
    citation_accuracy: float
    

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
    
    # Performance scores
    retrieval_score: float = 0.0
    generation_score: float = 0.0
    overall_score: float = 0.0
    

class EEGRAGBenchmark:
    """Comprehensive benchmarking suite for EEG-RAG system."""
    
    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        local_agent: Optional[LocalDataAgent] = None,
        web_agent: Optional[WebResearchAgent] = None
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
        
        self.citation_verifier = CitationVerifier(enable_medical_validation=True)
        self.performance_monitor = PerformanceMonitor()
        
        # Load benchmark queries
        self.benchmark_queries = self._create_benchmark_queries()
    
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
        
        # 4. Calculate aggregate metrics
        suite_results = self._calculate_aggregate_metrics(
            retrieval_results, generation_results, end_to_end_results
        )
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.2f}s. "
                   f"Overall score: {suite_results.overall_score:.1f}/100")
        
        return suite_results
    
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
                    
                    results.append(RetrievalBenchmarkResult(
                        query=benchmark_query.query_text,
                        retrieval_time_ms=retrieval_time_ms,
                        documents_found=documents_found,
                        relevance_score=relevance_score,
                        citation_accuracy=citation_accuracy
                    ))
                
            except Exception as e:
                logger.error(f"Retrieval benchmark failed for query '{benchmark_query.query_text}': {str(e)}")
                
                results.append(RetrievalBenchmarkResult(
                    query=benchmark_query.query_text,
                    retrieval_time_ms=0.0,
                    documents_found=0,
                    relevance_score=0.0,
                    citation_accuracy=0.0
                ))
        
        return results
    
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
                        cpu_usage_percent=monitor.avg_cpu_percent
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
                        cpu_usage_percent=0.0
                    ))
        
        return results
    
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
    
    def _calculate_aggregate_metrics(
        self,
        retrieval_results: List[RetrievalBenchmarkResult],
        generation_results: List[GenerationBenchmarkResult],
        end_to_end_results: List[EndToEndBenchmarkResult]
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
            retrieval_score=retrieval_score,
            generation_score=generation_score,
            overall_score=overall_score
        )
    
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
                'avg_response_quality': results.avg_response_quality
            },
            'detailed_results': {
                'retrieval': [asdict(r) for r in results.retrieval_results],
                'generation': [asdict(r) for r in results.generation_results],
                'end_to_end': [asdict(r) for r in results.end_to_end_results]
            },
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {output_path}")