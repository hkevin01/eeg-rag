#!/usr/bin/env python3
"""
Tests for Query Routing System
"""

import pytest
from src.eeg_rag.core.query_router import (
    QueryRouter, QueryType, RoutingResult, route_query
)


class TestQueryType:
    """Test query type enum"""
    
    def test_query_types(self):
        """Test all query types are defined"""
        expected_types = {
            'definitional', 'recent_literature', 'comparative',
            'methodological', 'clinical', 'statistical', 'unknown'
        }
        
        actual_types = {qt.value for qt in QueryType}
        assert actual_types == expected_types


class TestRoutingResult:
    """Test routing result data class"""
    
    def test_creation(self):
        """Test result creation"""
        result = RoutingResult(
            query_type=QueryType.DEFINITIONAL,
            confidence=0.85,
            recommended_agent="local_agent",
            reasoning="Query asks for definition",
            keywords=["what", "definition"],
            complexity="simple"
        )
        
        assert result.query_type == QueryType.DEFINITIONAL
        assert result.confidence == 0.85
        assert result.recommended_agent == "local_agent"
    
    def test_to_dict(self):
        """Test result serialization"""
        result = RoutingResult(
            query_type=QueryType.CLINICAL,
            confidence=0.75,
            recommended_agent="local_agent",
            reasoning="Clinical question detected",
            keywords=["patient", "treatment"],
            complexity="medium"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['query_type'] == 'clinical'
        assert result_dict['confidence'] == 0.75
        assert result_dict['recommended_agent'] == 'local_agent'
        assert result_dict['keywords'] == ['patient', 'treatment']


class TestQueryRouter:
    """Test query routing functionality"""
    
    @pytest.fixture
    def router(self):
        return QueryRouter()
    
    def test_init(self, router):
        """Test router initialization"""
        assert router.query_patterns is not None
        assert router.agent_routing is not None
        assert len(router.eeg_keywords) > 0
        
        # Check that all query types have patterns
        for query_type in QueryType:
            if query_type != QueryType.UNKNOWN:
                assert query_type in router.query_patterns
    
    def test_definitional_queries(self, router):
        """Test definitional query routing"""
        queries = [
            "What is EEG?",
            "Define electroencephalography",
            "Explain what sleep spindles mean",
            "Describe brain-computer interfaces"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.DEFINITIONAL
            assert result.confidence > 0.3
            assert result.recommended_agent in ["local_agent", "orchestrator"]
    
    def test_recent_literature_queries(self, router):
        """Test recent literature query routing"""
        queries = [
            "Recent research on EEG for epilepsy detection",
            "Latest findings in brain-computer interfaces",
            "Current studies on sleep EEG analysis",
            "New advances in motor imagery classification",
            "What are the cutting edge developments in 2024?"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.RECENT_LITERATURE
            assert result.confidence > 0.3
            assert result.recommended_agent == "web_agent"
    
    def test_comparative_queries(self, router):
        """Test comparative query routing"""
        queries = [
            "Compare EEG vs fMRI for epilepsy detection",
            "Difference between alpha and beta waves",
            "Which is better: invasive or non-invasive EEG?",
            "Advantages of deep learning versus traditional methods"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.COMPARATIVE
            assert result.confidence > 0.3
            assert result.recommended_agent == "graph_agent"
    
    def test_methodological_queries(self, router):
        """Test methodological query routing"""
        queries = [
            "How to perform EEG preprocessing?",
            "Methods for artifact removal in EEG",
            "Protocol for sleep stage classification",
            "Procedure for spike detection",
            "Steps to train a BCI classifier"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.METHODOLOGICAL
            assert result.confidence > 0.3
            assert result.recommended_agent == "local_agent"
    
    def test_clinical_queries(self, router):
        """Test clinical query routing"""
        queries = [
            "Clinical applications of EEG in epilepsy",
            "Patient outcomes with EEG monitoring",
            "Treatment efficacy using brain stimulation",
            "Diagnosis of sleep disorders with EEG",
            "Side effects of EEG electrode placement"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.CLINICAL
            assert result.confidence > 0.3
            assert result.recommended_agent == "local_agent"
    
    def test_statistical_queries(self, router):
        """Test statistical query routing"""
        queries = [
            "Statistical analysis of EEG coherence data",
            "Correlation between EEG power and behavior",
            "ANOVA results for sleep stage classification",
            "P-values for seizure prediction accuracy",
            "Sample size calculation for EEG study"
        ]
        
        for query in queries:
            result = router.route_query(query)
            
            assert result.query_type == QueryType.STATISTICAL
            assert result.confidence > 0.3
            assert result.recommended_agent == "local_agent"
    
    def test_eeg_relevance_boost(self, router):
        """Test EEG-specific confidence boosting"""
        # Query with EEG terms should get boosted confidence
        eeg_query = "What is electroencephalography alpha rhythm analysis?"
        general_query = "What is signal analysis?"
        
        eeg_result = router.route_query(eeg_query)
        general_result = router.route_query(general_query)
        
        # EEG query should have higher confidence due to domain relevance
        assert eeg_result.confidence >= general_result.confidence
    
    def test_complexity_assessment(self, router):
        """Test query complexity assessment"""
        simple_query = "What is EEG?"
        medium_query = "How do sleep spindles relate to memory consolidation?"
        complex_query = "Compare and analyze the interaction between multiple EEG biomarkers in various sleep stages and their correlation with cognitive performance metrics."
        
        simple_result = router.route_query(simple_query)
        medium_result = router.route_query(medium_query)
        complex_result = router.route_query(complex_query)
        
        assert simple_result.complexity == "simple"
        assert medium_result.complexity == "medium"
        assert complex_result.complexity == "complex"
    
    def test_complex_query_agent_override(self, router):
        """Test that complex queries get routed to orchestrator"""
        complex_query = "Analyze and compare multiple methodological approaches for seizure detection while evaluating their clinical efficacy and statistical significance across different patient populations."
        
        result = router.route_query(complex_query)
        
        assert result.complexity == "complex"
        # Complex queries should use orchestrator regardless of detected type
        if result.query_type != QueryType.RECENT_LITERATURE:
            assert result.recommended_agent == "orchestrator"
    
    def test_keyword_extraction(self, router):
        """Test keyword extraction"""
        query = "What is the clinical significance of EEG alpha waves in epilepsy patients?"
        
        result = router.route_query(query)
        
        # Should extract meaningful keywords
        keywords = result.keywords
        assert len(keywords) > 0
        # Should not contain stop words
        assert "the" not in keywords
        assert "is" not in keywords
    
    def test_calculate_type_score(self, router):
        """Test type scoring calculation"""
        query = "what is the definition of EEG?"
        
        config = router.query_patterns[QueryType.DEFINITIONAL]
        score = router._calculate_type_score(query, config)
        
        assert 0 <= score <= 1
        assert score > 0  # Should match some patterns
    
    def test_calculate_eeg_relevance(self, router):
        """Test EEG relevance calculation"""
        eeg_query = "electroencephalography seizure epilepsy brain waves"
        non_eeg_query = "weather forecast prediction model"
        
        eeg_score = router._calculate_eeg_relevance(eeg_query)
        non_eeg_score = router._calculate_eeg_relevance(non_eeg_query)
        
        assert eeg_score > non_eeg_score
        assert 0 <= eeg_score <= 1
        assert 0 <= non_eeg_score <= 1
    
    def test_generate_reasoning(self, router):
        """Test reasoning generation"""
        reasoning = router._generate_reasoning(
            "What is EEG?",
            QueryType.DEFINITIONAL,
            0.85,
            "simple"
        )
        
        assert "definition" in reasoning.lower()
        assert "high" in reasoning.lower()  # high confidence
        assert "simple" in reasoning.lower()
    
    def test_extract_key_terms(self, router):
        """Test key term extraction"""
        query = "The effectiveness of deep learning methods for EEG seizure detection"
        
        terms = router._extract_key_terms(query)
        
        assert "effectiveness" in terms
        assert "deep" in terms or "learning" in terms
        assert "eeg" in terms
        assert "seizure" in terms
        assert "detection" in terms
        
        # Should not contain stop words
        assert "the" not in terms
        assert "of" not in terms
        assert "for" not in terms
    
    def test_get_routing_stats(self, router):
        """Test routing statistics"""
        stats = router.get_routing_stats()
        
        assert 'supported_query_types' in stats
        assert 'agent_mapping' in stats
        assert 'eeg_keywords_count' in stats
        
        assert len(stats['supported_query_types']) == len(QueryType)
        assert stats['eeg_keywords_count'] > 0
    
    def test_add_custom_pattern(self, router):
        """Test adding custom patterns"""
        initial_patterns = len(router.query_patterns[QueryType.DEFINITIONAL]['patterns'])
        
        router.add_custom_pattern(
            QueryType.DEFINITIONAL,
            r'\bexplicate\b',
            ['explicate', 'clarify']
        )
        
        final_patterns = len(router.query_patterns[QueryType.DEFINITIONAL]['patterns'])
        assert final_patterns == initial_patterns + 1
        
        # Test with custom pattern
        result = router.route_query("Please explicate the concept of EEG")
        assert result.query_type == QueryType.DEFINITIONAL
    
    def test_ambiguous_queries(self, router):
        """Test handling of ambiguous queries"""
        ambiguous_queries = [
            "EEG",  # Too short
            "Tell me about stuff",  # Too vague
            "Is this working?",  # Unclear
            "The quick brown fox jumps"  # Irrelevant
        ]
        
        for query in ambiguous_queries:
            result = router.route_query(query)
            
            # Should still return a result with low confidence
            assert isinstance(result, RoutingResult)
            assert 0 <= result.confidence <= 1


class TestConvenienceFunction:
    """Test convenience routing function"""
    
    def test_route_query_function(self):
        """Test standalone routing function"""
        query = "What is epilepsy?"
        
        result = route_query(query)
        
        assert isinstance(result, RoutingResult)
        assert result.query_type == QueryType.DEFINITIONAL
        assert result.confidence > 0


class TestIntegration:
    """Integration tests for query routing"""
    
    def test_real_world_queries(self):
        """Test routing with realistic EEG research queries"""
        router = QueryRouter()
        
        test_cases = [
            {
                'query': "How effective are convolutional neural networks for automated seizure detection in long-term EEG monitoring?",
                'expected_type': QueryType.METHODOLOGICAL,
                'expected_complexity': 'complex'
            },
            {
                'query': "What are the latest 2024 developments in brain-computer interface research?",
                'expected_type': QueryType.RECENT_LITERATURE,
                'expected_agent': 'web_agent'
            },
            {
                'query': "Define sleep spindle morphology",
                'expected_type': QueryType.DEFINITIONAL,
                'expected_complexity': 'simple'
            },
            {
                'query': "Compare performance of ICA versus wavelet denoising for EEG artifact removal",
                'expected_type': QueryType.COMPARATIVE,
                'expected_agent': 'graph_agent'
            }
        ]
        
        for case in test_cases:
            result = router.route_query(case['query'])
            
            if 'expected_type' in case:
                assert result.query_type == case['expected_type'], f"Failed for: {case['query']}"
            
            if 'expected_complexity' in case:
                assert result.complexity == case['expected_complexity'], f"Failed for: {case['query']}"
            
            if 'expected_agent' in case:
                assert result.recommended_agent == case['expected_agent'], f"Failed for: {case['query']}"


if __name__ == "__main__":
    pytest.main([__file__])
