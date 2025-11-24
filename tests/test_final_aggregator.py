"""
Tests for Final Aggregator
"""

import pytest
from datetime import datetime
from eeg_rag.ensemble import (
    FinalAggregator,
    FinalAnswer,
    HallucinationDetector,
    ResponseValidator,
    CitationFormatter,
    Citation,
    Entity,
    AggregatedContext,
    GenerationResult,
    EnsembleResponse,
    LLMProvider
)


# Test fixtures
@pytest.fixture
def sample_citations():
    """Create sample citations for testing"""
    return [
        Citation(
            pmid="12345678",
            title="Alpha oscillations in visual cortex",
            authors=["Smith J", "Jones M", "Brown K"],
            year=2020,
            journal="Nature Neuroscience",
            doi="10.1038/nn.2020.1",
            abstract="Alpha oscillations (8-13 Hz) play a crucial role in visual attention and processing.",
            relevance_score=0.95
        ),
        Citation(
            pmid="87654321",
            title="Theta rhythms and memory encoding",
            authors=["Johnson A", "Williams R"],
            year=2021,
            journal="Journal of Neuroscience",
            doi="10.1523/jneurosci.2021.2",
            abstract="Theta rhythms (4-8 Hz) are essential for memory encoding in the hippocampus.",
            relevance_score=0.88
        ),
        Citation(
            pmid="11223344",
            title="EEG biomarkers in Alzheimer's disease",
            authors=["Davis L"],
            year=2022,
            journal="Brain",
            abstract="Alpha and theta power changes are reliable biomarkers for Alzheimer's disease progression.",
            relevance_score=0.92
        )
    ]


@pytest.fixture
def sample_entities():
    """Create sample entities for testing"""
    return [
        Entity(
            text="alpha oscillations",
            entity_type="biomarker",
            frequency=3,
            contexts=["visual cortex", "attention", "8-13 Hz"],
            confidence=0.95
        ),
        Entity(
            text="theta rhythms",
            entity_type="biomarker",
            frequency=2,
            contexts=["hippocampus", "memory"],
            confidence=0.90
        )
    ]


@pytest.fixture
def sample_context(sample_citations, sample_entities):
    """Create sample aggregated context"""
    return AggregatedContext(
        query="What are the roles of alpha and theta oscillations in cognition?",
        citations=sample_citations,
        entities=sample_entities,
        total_sources=3,
        agent_contributions={'local_agent': 2, 'web_agent': 1},
        relevance_threshold=0.8,
        timestamp=datetime.utcnow().isoformat(),
        statistics={'total_chunks': 5, 'unique_papers': 3}
    )


@pytest.fixture
def sample_generation_results():
    """Create sample generation results"""
    return [
        GenerationResult(
            model_name="gpt-4",
            provider=LLMProvider.OPENAI,
            response="Alpha oscillations (8-13 Hz) are crucial for visual attention and processing in the cortex. Theta rhythms (4-8 Hz) play an essential role in memory encoding, particularly in the hippocampus.",
            confidence=0.92,
            generation_time=1.2,
            tokens_used=45
        ),
        GenerationResult(
            model_name="gemini-pro",
            provider=LLMProvider.GOOGLE,
            response="Research shows that alpha oscillations support visual processing and attention, while theta rhythms are important for memory functions in the hippocampus.",
            confidence=0.88,
            generation_time=1.5,
            tokens_used=40
        )
    ]


@pytest.fixture
def sample_ensemble_response(sample_generation_results):
    """Create sample ensemble response"""
    return EnsembleResponse(
        final_response="Alpha oscillations (8-13 Hz) are crucial for visual attention and processing in the cortex. Theta rhythms (4-8 Hz) play an essential role in memory encoding, particularly in the hippocampus. These oscillations serve as important biomarkers for cognitive function.",
        confidence_score=0.90,
        diversity_score=0.75,
        contributing_models=["gpt-4", "gemini-pro"],
        individual_responses=sample_generation_results,
        voting_results={'consensus': True, 'agreement': 0.85},
        timestamp=datetime.utcnow().isoformat(),
        statistics={'total_models': 2, 'successful': 2}
    )


# Test HallucinationDetector
class TestHallucinationDetector:
    """Test hallucination detection"""
    
    def test_detect_numeric_claims_without_citations(self):
        """Test detection of numeric claims without citations"""
        response = "Alpha waves occur at 10 Hz and theta at 6 Hz in 85% of subjects."
        context = AggregatedContext(
            query="test",
            citations=[],
            entities=[],
            total_sources=0,
            agent_contributions={},
            relevance_threshold=0.8,
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        is_safe, warnings = HallucinationDetector.detect_hallucinations(response, context)
        
        assert not is_safe
        assert any("numeric claims" in w.lower() for w in warnings)
    
    def test_detect_causal_language_insufficient_citations(self):
        """Test detection of strong causal claims with few citations"""
        response = "Alpha oscillations always cause improved attention."
        
        # Create context with only 1 citation
        citation = Citation(
            title="Test paper",
            relevance_score=0.9
        )
        context = AggregatedContext(
            query="test",
            citations=[citation],
            entities=[],
            total_sources=1,
            agent_contributions={},
            relevance_threshold=0.8,
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        is_safe, warnings = HallucinationDetector.detect_hallucinations(response, context)
        
        assert not is_safe
        assert any("causal claim" in w.lower() for w in warnings)
    
    def test_detect_medical_advice_without_disclaimer(self):
        """Test detection of medical advice without disclaimer"""
        response = "Patients should take 10mg medication daily for treatment."
        
        context = AggregatedContext(
            query="test",
            citations=[],
            entities=[],
            total_sources=0,
            agent_contributions={},
            relevance_threshold=0.8,
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        is_safe, warnings = HallucinationDetector.detect_hallucinations(response, context)
        
        assert not is_safe
        assert any("medical advice" in w.lower() for w in warnings)
    
    def test_safe_response_with_citations(self, sample_context):
        """Test that well-cited responses pass validation"""
        response = "Alpha oscillations (8-13 Hz) are important for visual processing. Theta rhythms support memory encoding."
        
        is_safe, warnings = HallucinationDetector.detect_hallucinations(response, sample_context)
        
        # Should be safe or only have low-priority warnings
        assert is_safe or all('low citation density' in w.lower() for w in warnings)


# Test ResponseValidator
class TestResponseValidator:
    """Test response validation"""
    
    def test_validate_with_good_overlap(self, sample_context):
        """Test validation with good term overlap"""
        response = "Alpha oscillations are crucial for visual attention processing in the cortex. Theta rhythms play an essential role in memory encoding in the hippocampus."
        
        is_valid, overlap_score, issues = ResponseValidator.validate_response(
            response, sample_context, min_overlap=0.3
        )
        
        assert is_valid
        assert overlap_score >= 0.3
        assert len(issues) == 0 or 'entities' in issues[0].lower()
    
    def test_validate_with_low_overlap(self, sample_context):
        """Test validation with low term overlap"""
        response = "Quantum mechanics describes particle behavior using wavefunctions and probability distributions."
        
        is_valid, overlap_score, issues = ResponseValidator.validate_response(
            response, sample_context, min_overlap=0.3
        )
        
        assert not is_valid
        assert overlap_score < 0.3
        assert len(issues) > 0
    
    def test_validate_without_citations(self):
        """Test validation without citations"""
        response = "Test response with some content."
        
        context = AggregatedContext(
            query="test",
            citations=[],
            entities=[],
            total_sources=0,
            agent_contributions={},
            relevance_threshold=0.8,
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        is_valid, overlap_score, issues = ResponseValidator.validate_response(response, context)
        
        assert not is_valid
        assert overlap_score == 0.0
        assert any("no source citations" in i.lower() for i in issues)


# Test CitationFormatter
class TestCitationFormatter:
    """Test citation formatting"""
    
    def test_extract_citation_markers(self):
        """Test extraction of citation markers"""
        text = "Alpha waves [1] are important [2,3] for cognition [Smith2020]."
        
        markers = CitationFormatter.extract_citation_markers(text)
        
        assert len(markers) == 3
        assert "1" in markers
        assert "2,3" in markers
        assert "Smith2020" in markers
    
    def test_format_inline_citation_numeric(self, sample_citations):
        """Test numeric inline citation formatting"""
        citation = sample_citations[0]
        
        formatted = CitationFormatter.format_inline_citation(citation, style="numeric")
        
        assert formatted == "[12345678]"
    
    def test_format_inline_citation_author_year(self, sample_citations):
        """Test author-year inline citation formatting"""
        citation = sample_citations[0]
        
        formatted = CitationFormatter.format_inline_citation(citation, style="author-year")
        
        assert "(Smith, 2020)" in formatted or "(J, 2020)" in formatted
    
    def test_format_bibliography_entry_apa(self, sample_citations):
        """Test APA bibliography entry formatting"""
        citation = sample_citations[0]
        
        formatted = CitationFormatter.format_bibliography_entry(citation, index=1, style="apa")
        
        assert "[1]" in formatted
        assert "Smith" in formatted or "et al." in formatted
        assert "(2020)" in formatted
        assert "Alpha oscillations" in formatted
        assert "Nature Neuroscience" in formatted
        assert "PMID:12345678" in formatted


# Test FinalAggregator
class TestFinalAggregator:
    """Test Final Aggregator"""
    
    def test_initialization(self):
        """Test aggregator initialization"""
        aggregator = FinalAggregator(
            hallucination_threshold=0.8,
            validation_threshold=0.4,
            min_confidence=0.6
        )
        
        assert aggregator.hallucination_threshold == 0.8
        assert aggregator.validation_threshold == 0.4
        assert aggregator.min_confidence == 0.6
        assert aggregator.stats['total_aggregations'] == 0
    
    def test_aggregate_basic(self, sample_ensemble_response, sample_context):
        """Test basic aggregation"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "What are alpha and theta oscillations?"
        )
        
        assert isinstance(result, FinalAnswer)
        assert len(result.answer_text) > 0
        assert len(result.citations) == 3
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.sources) > 0
    
    def test_aggregate_confidence_scoring(self, sample_ensemble_response, sample_context):
        """Test confidence scoring in aggregation"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        # Confidence should be adjusted based on validation
        assert result.confidence <= sample_ensemble_response.confidence_score
        assert 'ensemble_confidence' in result.metadata
        assert 'validation_overlap' in result.metadata
    
    def test_aggregate_citation_ranking(self, sample_ensemble_response, sample_context):
        """Test that citations are ranked by relevance"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        # Citations should be sorted by relevance score (descending)
        for i in range(len(result.citations) - 1):
            assert result.citations[i].relevance_score >= result.citations[i + 1].relevance_score
    
    def test_aggregate_warning_generation(self):
        """Test warning generation for problematic responses"""
        aggregator = FinalAggregator(min_confidence=0.9)
        
        # Create low-quality response
        bad_response = EnsembleResponse(
            final_response="Quantum mechanics explains EEG.",
            confidence_score=0.5,
            diversity_score=0.3,
            contributing_models=["model1"],
            individual_responses=[
                GenerationResult(
                    model_name="model1",
                    provider=LLMProvider.MOCK,
                    response="Quantum mechanics explains EEG.",
                    confidence=0.5,
                    generation_time=1.0,
                    tokens_used=10
                )
            ],
            voting_results={},
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        context = AggregatedContext(
            query="test",
            citations=[],
            entities=[],
            total_sources=0,
            agent_contributions={},
            relevance_threshold=0.8,
            timestamp=datetime.utcnow().isoformat(),
            statistics={}
        )
        
        result = aggregator.aggregate(bad_response, context, "Test query")
        
        assert len(result.warnings) > 0
    
    def test_aggregate_statistics_tracking(self, sample_ensemble_response, sample_context):
        """Test statistics tracking across aggregations"""
        aggregator = FinalAggregator()
        
        # Perform multiple aggregations
        for i in range(3):
            aggregator.aggregate(sample_ensemble_response, sample_context, f"Query {i}")
        
        stats = aggregator.get_statistics()
        
        assert stats['total_aggregations'] == 3
        assert stats['avg_confidence'] > 0
        assert stats['avg_citation_count'] > 0
    
    def test_aggregate_metadata_enrichment(self, sample_ensemble_response, sample_context):
        """Test metadata enrichment in final answer"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        assert 'ensemble_confidence' in result.metadata
        assert 'validation_overlap' in result.metadata
        assert 'hallucination_safe' in result.metadata
        assert 'citation_count' in result.metadata
        assert 'source_agents' in result.metadata
        assert 'diversity_score' in result.metadata
    
    def test_aggregate_source_extraction(self, sample_ensemble_response, sample_context):
        """Test extraction of source IDs (PMIDs, DOIs)"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        assert len(result.sources) == 3
        assert any("PMID:" in s for s in result.sources)
    
    def test_to_dict_conversion(self, sample_ensemble_response, sample_context):
        """Test FinalAnswer to_dict conversion"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        result_dict = result.to_dict()
        
        assert 'answer_text' in result_dict
        assert 'citations' in result_dict
        assert 'confidence' in result_dict
        assert 'sources' in result_dict
        assert 'query' in result_dict
        assert isinstance(result_dict['citations'], list)
    
    def test_to_markdown_conversion(self, sample_ensemble_response, sample_context):
        """Test FinalAnswer to_markdown conversion"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "What are alpha oscillations?"
        )
        
        markdown = result.to_markdown()
        
        assert "# Query:" in markdown
        assert "## Answer" in markdown
        assert "Confidence:" in markdown
        if result.citations:
            assert "## References" in markdown
        if result.warnings:
            assert "## Warnings" in markdown
    
    def test_reset_statistics(self, sample_ensemble_response, sample_context):
        """Test statistics reset"""
        aggregator = FinalAggregator()
        
        # Perform aggregation
        aggregator.aggregate(sample_ensemble_response, sample_context, "Test")
        
        assert aggregator.stats['total_aggregations'] > 0
        
        # Reset
        aggregator.reset_statistics()
        
        assert aggregator.stats['total_aggregations'] == 0
        assert aggregator.stats['avg_confidence'] == 0.0


# Integration tests
class TestFinalAggregatorIntegration:
    """Integration tests for Final Aggregator"""
    
    def test_end_to_end_aggregation(self, sample_ensemble_response, sample_context):
        """Test complete end-to-end aggregation pipeline"""
        aggregator = FinalAggregator(
            hallucination_threshold=0.7,
            validation_threshold=0.3,
            min_confidence=0.5
        )
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "What are the roles of alpha and theta oscillations?"
        )
        
        # Verify all components
        assert isinstance(result, FinalAnswer)
        assert len(result.answer_text) > 0
        assert len(result.citations) > 0
        assert result.confidence > 0
        assert len(result.sources) > 0
        assert result.query == "What are the roles of alpha and theta oscillations?"
        assert 'ensemble_confidence' in result.metadata
        assert 'statistics' in result.__dict__
        
        # Verify markdown export works
        markdown = result.to_markdown()
        assert len(markdown) > 0
        
        # Verify dict export works
        result_dict = result.to_dict()
        assert 'answer_text' in result_dict
    
    def test_quality_assessment_pipeline(self, sample_ensemble_response, sample_context):
        """Test complete quality assessment pipeline"""
        aggregator = FinalAggregator()
        
        result = aggregator.aggregate(
            sample_ensemble_response,
            sample_context,
            "Test query"
        )
        
        # Verify hallucination detection ran
        assert 'hallucination_safe' in result.metadata
        
        # Verify validation ran
        assert 'validation_overlap' in result.metadata
        
        # Verify confidence adjustment
        assert result.confidence != sample_ensemble_response.confidence_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
