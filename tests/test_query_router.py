#!/usr/bin/env python3
"""
Comprehensive Test Suite for Production Query Router

Tests the enhanced query routing system with medical domain optimizations,
performance validation, and edge case handling. Ensures reliable agent
selection for EEG-RAG queries across all complexity levels.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from src.eeg_rag.core.query_router import (
    QueryRouter, QueryType, RoutingResult, route_query
)


class TestQueryRouterBasics:
    """Test basic query router functionality."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_router_initialization(self, router):
        """Test router initializes correctly."""
        assert hasattr(router, 'query_patterns')
        assert hasattr(router, 'agent_routing')
        assert hasattr(router, 'eeg_keywords')
        assert len(router.query_patterns) >= 6
    
    def test_definitional_queries(self, router):
        """Test classification of definitional queries."""
        queries = [
            "What is EEG?",
            "Define electroencephalography",
            "Explain brain waves"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.DEFINITIONAL
            assert result.recommended_agent == 'local_agent'
    
    def test_recent_literature_queries(self, router):
        """Test classification of recent literature queries."""
        queries = [
            "Recent research on EEG in epilepsy",
            "Latest findings about brain-computer interfaces",
            "Current trends in EEG analysis 2024"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.RECENT_LITERATURE
            assert result.recommended_agent == 'web_agent'
    
    def test_comparative_queries(self, router):
        """Test classification of comparative queries."""
        queries = [
            "Compare EEG vs fMRI for seizure detection",
            "Difference between alpha and beta waves",
            "Which is better: bipolar or referential montage?"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.COMPARATIVE
            assert result.recommended_agent == 'graph_agent'
    
    def test_methodological_queries(self, router):
        """Test classification of methodological queries."""
        queries = [
            "How to perform EEG electrode placement?",
            "Protocol for EEG artifact removal",
            "Steps for seizure detection algorithm"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.METHODOLOGICAL
            assert result.recommended_agent == 'local_agent'
    
    def test_clinical_queries(self, router):
        """Test classification of clinical queries."""
        queries = [
            "EEG findings in epilepsy patients",
            "Clinical applications of sleep EEG",
            "Patient monitoring with continuous EEG"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.CLINICAL
            assert result.recommended_agent == 'local_agent'
    
    def test_statistical_queries(self, router):
        """Test classification of statistical queries."""
        queries = [
            "Statistical analysis of EEG coherence",
            "Power spectral density significance testing",
            "Correlation between EEG features and behavior"
        ]
        
        for query in queries:
            result = router.route_query(query)
            assert result.query_type == QueryType.STATISTICAL
            assert result.recommended_agent == 'local_agent'


class TestComplexityAssessment:
    """Test query complexity assessment."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_simple_complexity(self, router):
        """Test simple query complexity detection."""
        simple_queries = [
            "What is EEG?",
            "Define alpha waves"
        ]
        
        for query in simple_queries:
            result = router.route_query(query)
            assert result.complexity == 'simple'
    
    def test_medium_complexity(self, router):
        """Test medium query complexity detection."""
        medium_queries = [
            "How does EEG compare to MEG for source localization?",
            "Effect of caffeine on EEG alpha power"
        ]
        
        for query in medium_queries:
            result = router.route_query(query)
            assert result.complexity in ['medium', 'complex']
    
    def test_complex_routing_to_orchestrator(self, router):
        """Test complex queries route to orchestrator."""
        complex_query = "Analyze the multifactorial relationship between EEG connectivity patterns, cognitive performance, and neuropharmacological interventions in epilepsy patients"
        
        result = router.route_query(complex_query)
        assert result.complexity == 'complex'
        assert result.recommended_agent == 'orchestrator'


class TestPerformance:
    """Test routing performance."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_routing_latency(self, router):
        """Test routing latency is within bounds."""
        query = "What is EEG alpha power?"
        
        start_time = time.time()
        result = router.route_query(query)
        end_time = time.time()
        
        latency = end_time - start_time
        assert latency < 1.0  # Should be under 1 second
        assert result is not None
    
    def test_batch_routing(self, router):
        """Test batch routing performance."""
        queries = ["What is EEG?"] * 10
        
        start_time = time.time()
        results = [router.route_query(q) for q in queries]
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(queries)
        
        assert avg_time < 0.5  # Average under 500ms per query
        assert len(results) == len(queries)


class TestEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_empty_query(self, router):
        """Test handling of empty queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            router.route_query("")
    
    def test_very_long_query(self, router):
        """Test handling of very long queries."""
        long_query = "What is EEG? " * 50
        
        result = router.route_query(long_query)
        assert result is not None
        assert result.confidence >= 0.0
    
    def test_special_characters(self, router):
        """Test handling of queries with special characters."""
        special_queries = [
            "What is µV in EEG measurements?",
            "Alpha waves (8-12 Hz) significance"
        ]
        
        for query in special_queries:
            result = router.route_query(query)
            assert result is not None


class TestRoutingResult:
    """Test RoutingResult functionality."""
    
    def test_routing_result_creation(self):
        """Test RoutingResult creation."""
        result = RoutingResult(
            query_type=QueryType.DEFINITIONAL,
            confidence=0.85,
            recommended_agent='local_agent',
            reasoning="Test reasoning",
            keywords=['eeg', 'test'],
            complexity='simple'
        )
        
        assert result.query_type == QueryType.DEFINITIONAL
        assert result.confidence == 0.85
    
    def test_to_dict_serialization(self):
        """Test conversion to dictionary."""
        result = RoutingResult(
            query_type=QueryType.COMPARATIVE,
            confidence=0.75,
            recommended_agent='graph_agent',
            reasoning="Comparison query",
            keywords=['compare', 'eeg'],
            complexity='medium'
        )
        
        result_dict = result.to_dict()
        assert result_dict['query_type'] == 'comparative'
        assert result_dict['confidence'] == 0.75


class TestEEGOptimization:
    """Test EEG domain optimizations."""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_eeg_keyword_boost(self, router):
        """Test EEG keywords boost confidence."""
        eeg_query = "EEG electroencephalography seizure detection"
        generic_query = "What is data analysis?"
        
        eeg_result = router.route_query(eeg_query)
        generic_result = router.route_query(generic_query)
        
        # EEG query should have higher or equal confidence
        assert eeg_result.confidence >= generic_result.confidence


class TestConvenienceFunction:
    """Test convenience routing function."""
    
    def test_quick_route_function(self):
        """Test the convenience route_query function."""
        result = route_query("What is EEG?")
        
        assert result is not None
        assert result.query_type == QueryType.DEFINITIONAL
        assert result.recommended_agent == 'local_agent'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])