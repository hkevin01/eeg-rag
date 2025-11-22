"""
Test Suite for Generation Ensemble (Component 11/12)
Tests multi-LLM generation, voting, diversity, and confidence weighting
"""

import pytest
import asyncio
from eeg_rag.ensemble import (
    GenerationEnsemble,
    GenerationResult,
    EnsembleResponse,
    LLMProvider,
    MockLLMClient
)


class TestLLMProvider:
    """Tests for LLMProvider enum"""
    
    def test_llm_provider_values(self):
        """Test LLM provider enum values"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.MOCK.value == "mock"


class TestGenerationResult:
    """Tests for GenerationResult dataclass"""
    
    def test_generation_result_creation(self):
        """Test creating a generation result"""
        result = GenerationResult(
            model_name="gpt-4",
            provider=LLMProvider.OPENAI,
            response="Test response",
            confidence=0.95,
            generation_time=1.5,
            tokens_used=100
        )
        
        assert result.model_name == "gpt-4"
        assert result.provider == LLMProvider.OPENAI
        assert result.confidence == 0.95
        assert result.tokens_used == 100
    
    def test_generation_result_to_dict(self):
        """Test generation result serialization"""
        result = GenerationResult(
            model_name="gemini-pro",
            provider=LLMProvider.GOOGLE,
            response="Test",
            confidence=0.88,
            generation_time=1.0,
            tokens_used=50
        )
        
        result_dict = result.to_dict()
        assert result_dict['model_name'] == "gemini-pro"
        assert result_dict['provider'] == "google"
        assert result_dict['confidence'] == 0.88


class TestEnsembleResponse:
    """Tests for EnsembleResponse dataclass"""
    
    def test_ensemble_response_creation(self):
        """Test creating ensemble response"""
        results = [
            GenerationResult("gpt-4", LLMProvider.OPENAI, "Test", 0.9, 1.0, 50)
        ]
        
        response = EnsembleResponse(
            final_response="Final answer",
            confidence_score=0.9,
            diversity_score=0.3,
            contributing_models=["gpt-4"],
            individual_responses=results,
            voting_results={},
            timestamp="2024-01-01",
            statistics={}
        )
        
        assert response.final_response == "Final answer"
        assert response.confidence_score == 0.9
        assert len(response.contributing_models) == 1
    
    def test_ensemble_response_to_dict(self):
        """Test ensemble response serialization"""
        response = EnsembleResponse(
            final_response="Test",
            confidence_score=0.85,
            diversity_score=0.2,
            contributing_models=["gpt-4"],
            individual_responses=[],
            voting_results={},
            timestamp="2024-01-01",
            statistics={}
        )
        
        result = response.to_dict()
        assert 'final_response' in result
        assert 'confidence_score' in result
        assert result['diversity_score'] == 0.2


class TestMockLLMClient:
    """Tests for MockLLMClient"""
    
    @pytest.mark.asyncio
    async def test_mock_client_gpt(self):
        """Test mock GPT-4 client"""
        client = MockLLMClient("gpt-4", LLMProvider.OPENAI)
        result = await client.generate("Test prompt")
        
        assert "GPT-4 Response" in result.response
        assert result.confidence == 0.92
        assert result.provider == LLMProvider.OPENAI
    
    @pytest.mark.asyncio
    async def test_mock_client_gemini(self):
        """Test mock Gemini client"""
        client = MockLLMClient("gemini-pro", LLMProvider.GOOGLE)
        result = await client.generate("Test prompt")
        
        assert "Gemini Response" in result.response
        assert result.confidence == 0.88
    
    @pytest.mark.asyncio
    async def test_mock_client_claude(self):
        """Test mock Claude client"""
        client = MockLLMClient("claude-3", LLMProvider.ANTHROPIC)
        result = await client.generate("Test prompt")
        
        assert "Claude Response" in result.response
        assert result.confidence == 0.90


class TestGenerationEnsemble:
    """Tests for GenerationEnsemble class"""
    
    def test_ensemble_initialization(self):
        """Test creating generation ensemble"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        assert ensemble.temperature == 0.7
        assert ensemble.max_tokens == 1000
        assert ensemble.timeout == 30.0
        assert len(ensemble.models) == 3
    
    def test_ensemble_custom_models(self):
        """Test ensemble with custom model configuration"""
        models = [
            {'name': 'gpt-4', 'provider': LLMProvider.OPENAI, 'weight': 1.0}
        ]
        ensemble = GenerationEnsemble(models=models, use_mock=True)
        
        assert len(ensemble.models) == 1
        assert ensemble.models[0]['name'] == 'gpt-4'
    
    @pytest.mark.asyncio
    async def test_generate_single_model(self):
        """Test generation with single model"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate(
            "Test query about EEG",
            use_all_models=False
        )
        
        assert isinstance(response, EnsembleResponse)
        assert len(response.individual_responses) == 1
        assert response.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_generate_all_models(self):
        """Test generation with all models"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate(
            "What is alpha power in EEG?",
            use_all_models=True
        )
        
        assert isinstance(response, EnsembleResponse)
        assert len(response.individual_responses) == 3
        assert len(response.contributing_models) == 3
    
    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test generation with additional context"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        context = {
            'citations': [
                {'title': 'Study 1'},
                {'title': 'Study 2'}
            ],
            'entities': [
                {'text': 'alpha power'},
                {'text': 'frontal cortex'}
            ]
        }
        
        response = await ensemble.generate(
            "Test query",
            context=context,
            use_all_models=False
        )
        
        assert isinstance(response, EnsembleResponse)
        # Context should be included in prompt
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test response caching"""
        ensemble = GenerationEnsemble(use_mock=True, enable_caching=True)
        
        # First generation
        response1 = await ensemble.generate("Test query", use_all_models=False)
        
        # Second generation (should hit cache)
        response2 = await ensemble.generate("Test query", use_all_models=False)
        
        assert response1.final_response == response2.final_response
        assert ensemble.stats['cache_hits'] == 1
    
    @pytest.mark.asyncio
    async def test_caching_disabled(self):
        """Test with caching disabled"""
        ensemble = GenerationEnsemble(use_mock=True, enable_caching=False)
        
        await ensemble.generate("Test query", use_all_models=False)
        await ensemble.generate("Test query", use_all_models=False)
        
        # Should not hit cache
        assert ensemble.stats['cache_hits'] == 0
    
    @pytest.mark.asyncio
    async def test_confidence_weighting(self):
        """Test confidence-based weighting"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate("Test", use_all_models=True)
        
        # Should select highest confidence * weight response
        assert response.final_response is not None
        assert response.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_diversity_scoring(self):
        """Test diversity score calculation"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate("Test", use_all_models=True)
        
        # With 3 different models, should have some diversity
        assert response.diversity_score >= 0
        assert response.diversity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test statistics tracking"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        await ensemble.generate("Test 1", use_all_models=False)
        await ensemble.generate("Test 2", use_all_models=False)
        
        stats = ensemble.get_statistics()
        
        assert stats['total_generations'] == 2
        assert stats['successful_generations'] == 2
        assert stats['total_tokens'] > 0
    
    def test_get_statistics(self):
        """Test getting statistics"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        stats = ensemble.get_statistics()
        
        assert 'total_generations' in stats
        assert 'success_rate' in stats
        assert 'model_usage' in stats
    
    def test_clear_cache(self):
        """Test clearing cache"""
        ensemble = GenerationEnsemble(use_mock=True, enable_caching=True)
        ensemble.cache['test_key'] = None
        
        ensemble.clear_cache()
        
        assert len(ensemble.cache) == 0
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        ensemble = GenerationEnsemble(use_mock=True)
        ensemble.stats['total_generations'] = 10
        
        ensemble.reset_statistics()
        
        assert ensemble.stats['total_generations'] == 0
    
    @pytest.mark.asyncio
    async def test_voting_results(self):
        """Test voting results metadata"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate("Test", use_all_models=True)
        
        assert 'method' in response.voting_results
        assert 'weights' in response.voting_results
        assert response.voting_results['method'] == 'weighted_confidence'
    
    @pytest.mark.asyncio
    async def test_response_statistics(self):
        """Test response statistics"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        response = await ensemble.generate("Test", use_all_models=True)
        
        assert 'total_time' in response.statistics
        assert 'models_used' in response.statistics
        assert 'total_tokens' in response.statistics
        assert 'average_confidence' in response.statistics
    
    @pytest.mark.asyncio
    async def test_model_usage_tracking(self):
        """Test model usage tracking"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        await ensemble.generate("Test", use_all_models=True)
        
        stats = ensemble.get_statistics()
        
        # All models should be used once
        assert stats['model_usage']['gpt-4'] == 1
        assert stats['model_usage']['gemini-pro'] == 1
        assert stats['model_usage']['claude-3'] == 1
    
    @pytest.mark.asyncio
    async def test_token_tracking(self):
        """Test token usage tracking"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        await ensemble.generate("Test query", use_all_models=True)
        
        stats = ensemble.get_statistics()
        assert stats['total_tokens'] > 0
    
    @pytest.mark.asyncio
    async def test_success_rate_calculation(self):
        """Test success rate calculation"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        await ensemble.generate("Test 1", use_all_models=False)
        await ensemble.generate("Test 2", use_all_models=False)
        
        stats = ensemble.get_statistics()
        assert stats['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_average_time_tracking(self):
        """Test average time tracking"""
        ensemble = GenerationEnsemble(use_mock=True)
        
        await ensemble.generate("Test 1", use_all_models=False)
        await ensemble.generate("Test 2", use_all_models=False)
        
        stats = ensemble.get_statistics()
        assert stats['average_time'] > 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Test cache hit rate calculation"""
        ensemble = GenerationEnsemble(use_mock=True, enable_caching=True)
        
        # First call
        await ensemble.generate("Test", use_all_models=False)
        # Second call (cache hit)
        await ensemble.generate("Test", use_all_models=False)
        
        stats = ensemble.get_statistics()
        assert stats['cache_hit_rate'] == 0.5  # 1 hit out of 2 generations
    
    @pytest.mark.asyncio
    async def test_custom_temperature(self):
        """Test custom temperature setting"""
        ensemble = GenerationEnsemble(
            use_mock=True,
            temperature=0.5,
            max_tokens=500
        )
        
        assert ensemble.temperature == 0.5
        assert ensemble.max_tokens == 500
