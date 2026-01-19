#!/usr/bin/env python3
"""
Comprehensive Test Suite for Production Semantic Chunker

Tests the enhanced semantic chunking system with medical optimizations.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.eeg_rag.nlp.semantic_chunker import (
    SemanticChunker, ChunkingStrategy, ChunkResult,
    chunk_medical_text, chunk_eeg_paper
)


class TestSemanticChunkerBasics:
    """Test basic semantic chunker functionality."""
    
    def test_chunker_initialization(self):
        """Test chunker initializes correctly."""
        chunker = SemanticChunker()
        
        assert chunker.enable_medical_optimization is True
        assert chunker.enable_caching is True
        assert chunker.max_cache_size == 1000
    
    def test_chunker_no_medical(self):
        """Test chunker without medical optimization."""
        chunker = SemanticChunker(enable_medical_optimization=False)
        assert chunker.enable_medical_optimization is False


class TestChunkingStrategies:
    """Test different chunking strategies."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker()
    
    @pytest.fixture
    def sample_text(self):
        return """
        EEG is a method to record brain electrical activity.
        Alpha waves are 8-12 Hz oscillations.
        Beta waves are 13-30 Hz oscillations.
        Clinical applications include epilepsy diagnosis.
        """
    
    def test_fixed_chunking(self, chunker, sample_text):
        """Test fixed-size chunking strategy."""
        result = chunker.chunk_text(
            sample_text,
            strategy=ChunkingStrategy.FIXED,
            target_chunk_size=80,
            min_chunk_size=20
        )
        
        assert result.strategy_used == ChunkingStrategy.FIXED
        assert len(result.chunks) > 0
        assert all(isinstance(chunk, str) for chunk in result.chunks)
    
    def test_adaptive_chunking(self, chunker, sample_text):
        """Test adaptive strategy selection."""
        result = chunker.chunk_text(
            sample_text,
            strategy=ChunkingStrategy.ADAPTIVE,
            target_chunk_size=90,
            min_chunk_size=25
        )
        
        assert result.strategy_used in [
            ChunkingStrategy.FIXED, ChunkingStrategy.SENTENCE,
            ChunkingStrategy.MEDICAL, ChunkingStrategy.PARAGRAPH
        ]
        assert len(result.chunks) > 0


class TestMedicalOptimizations:
    """Test medical content preservation."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(enable_medical_optimization=True)
    
    def test_citation_preservation(self, chunker):
        """Test that citations are preserved."""
        text_with_citations = """
        EEG shows alpha differences.
        Study by Smith et al. (2023) found significant results.
        PMID: 12345678 provides methodology details.
        """
        
        result = chunker.chunk_text(
            text_with_citations,
            strategy=ChunkingStrategy.MEDICAL,
            target_chunk_size=100,
            min_chunk_size=30
        )
        
        all_text = ' '.join(result.chunks)
        assert 'Smith et al. (2023)' in all_text
        assert 'PMID: 12345678' in all_text
    
    def test_measurement_preservation(self, chunker):
        """Test measurement preservation."""
        text_with_measurements = """
        Alpha activity at 10.5 Hz with 25 µV amplitude.
        Statistical significance p < 0.05 with n = 45.
        """
        
        result = chunker.chunk_text(
            text_with_measurements,
            strategy=ChunkingStrategy.MEDICAL,
            target_chunk_size=80,
            min_chunk_size=20
        )
        
        all_text = ' '.join(result.chunks)
        assert '10.5 Hz' in all_text
        assert '25 µV' in all_text
        assert 'p < 0.05' in all_text


class TestQualityMetrics:
    """Test chunk quality metrics."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker()
    
    def test_quality_metrics_computation(self, chunker):
        """Test quality metrics are computed."""
        text = "This is a test. " * 50
        
        result = chunker.chunk_text(
            text,
            strategy=ChunkingStrategy.FIXED,
            target_chunk_size=100,
            min_chunk_size=25
        )
        
        metrics = result.quality_metrics
        assert 'total_chunks' in metrics
        assert 'avg_chunk_size' in metrics
        assert 'min_chunk_size' in metrics
        assert 'max_chunk_size' in metrics
        
        assert metrics['total_chunks'] == len(result.chunks)
        assert metrics['avg_chunk_size'] > 0
    
    def test_chunk_result_methods(self, chunker):
        """Test ChunkResult utility methods."""
        text = "First content. Second content. Third content."
        
        result = chunker.chunk_text(
            text,
            strategy=ChunkingStrategy.FIXED,
            target_chunk_size=25,
            min_chunk_size=10
        )
        
        # Test get_chunk_by_index
        chunk_info = result.get_chunk_by_index(0)
        assert chunk_info is not None
        
        # Test invalid index
        assert result.get_chunk_by_index(999) is None
        
        # Test find_chunks_containing
        matching = result.find_chunks_containing("content")
        assert len(matching) >= 0
        
        # Test to_dict
        result_dict = result.to_dict()
        assert 'total_chunks' in result_dict
        assert 'strategy_used' in result_dict


class TestPerformance:
    """Test chunking performance."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker()
    
    def test_chunking_latency(self, chunker):
        """Test chunking latency."""
        text = "Test sentence. " * 100
        
        start_time = time.time()
        result = chunker.chunk_text(text, target_chunk_size=200)
        end_time = time.time()
        
        latency = end_time - start_time
        assert latency < 10.0  # Should be reasonable
        assert result is not None
    
    def test_large_text_handling(self, chunker):
        """Test large text handling."""
        large_text = "EEG analysis sentence. " * 500
        
        result = chunker.chunk_text(
            large_text,
            strategy=ChunkingStrategy.FIXED,
            target_chunk_size=200,
            min_chunk_size=50
        )
        
        assert len(result.chunks) > 5
        assert all(len(chunk) > 0 for chunk in result.chunks)


class TestEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker()
    
    def test_empty_text(self, chunker):
        """Test empty text handling."""
        with pytest.raises(ValueError):
            chunker.chunk_text("")
    
    def test_invalid_parameters(self, chunker):
        """Test invalid parameters."""
        text = "Valid text."
        
        with pytest.raises(ValueError):
            chunker.chunk_text(
                text,
                target_chunk_size=50,
                min_chunk_size=100  # Invalid: min > target
            )
    
    def test_special_characters(self, chunker):
        """Test special characters."""
        text_with_special = "EEG µV with α-waves and β-rhythms."
        
        result = chunker.chunk_text(text_with_special, 
                                   target_chunk_size=50, 
                                   min_chunk_size=15)
        
        assert len(result.chunks) > 0
        all_text = ' '.join(result.chunks)
        assert 'µV' in all_text


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_chunk_medical_text_function(self):
        """Test medical text convenience function."""
        medical_text = """
        Patient had seizures. EEG showed spikes.
        Treatment by Smith et al. (2023).
        """
        
        result = chunk_medical_text(medical_text, target_size=80)
        
        assert isinstance(result, ChunkResult)
        assert result.strategy_used == ChunkingStrategy.MEDICAL
        assert len(result.chunks) > 0
    
    def test_chunk_eeg_paper_function(self):
        """Test EEG paper convenience function."""
        eeg_text = """
        EEG Analysis Study
        
        Alpha waves (8-12 Hz) during rest.
        Beta activity (13-30 Hz) during tasks.
        """
        
        result = chunk_eeg_paper(eeg_text, target_size=80)
        
        assert isinstance(result, ChunkResult)
        assert len(result.chunks) > 0


class TestIntegration:
    """Test realistic usage scenarios."""
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(enable_medical_optimization=True)
    
    def test_eeg_paper_chunking(self, chunker):
        """Test realistic EEG paper chunking."""
        eeg_paper = """
        Title: EEG Analysis in Epilepsy
        
        Abstract: We studied 50 patients with epilepsy using EEG.
        
        Methods: EEG recorded at 512 Hz with 32 channels.
        Artifacts removed using ICA filtering.
        
        Results: Sensitivity 92.5% (CI: 88-96%).
        Specificity 87% with p < 0.05 significance.
        
        References: Smith et al. (2023). PMID: 12345.
        """
        
        result = chunker.chunk_text(
            eeg_paper,
            strategy=ChunkingStrategy.ADAPTIVE,
            target_chunk_size=150,
            min_chunk_size=40
        )
        
        assert len(result.chunks) >= 2
        
        # Check medical content preservation
        all_text = ' '.join(result.chunks)
        assert '92.5%' in all_text
        assert 'p < 0.05' in all_text
        assert 'PMID: 12345' in all_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])