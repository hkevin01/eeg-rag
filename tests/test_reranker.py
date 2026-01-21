"""
Tests for Cross-Encoder Reranker

Coverage:
- CrossEncoderReranker initialization
- Reranking with various candidate sets
- Score combination logic
- Edge cases (empty candidates, single candidate)
- NoOpReranker fallback
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from eeg_rag.retrieval.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankedResult
)


class TestCrossEncoderReranker:
    """Test cross-encoder reranker"""
    
    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder model"""
        with patch('eeg_rag.retrieval.reranker.CROSS_ENCODER_AVAILABLE', True):
            with patch('eeg_rag.retrieval.reranker.CrossEncoder') as mock_ce:
                mock_model = MagicMock()
                mock_ce.return_value = mock_model
                yield mock_model
    
    def test_initialization(self, mock_cross_encoder):
        """Test reranker initialization"""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            combine_weight=0.7
        )
        
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.combine_weight == 0.7
        assert reranker.model is not None
    
    def test_rerank_basic(self, mock_cross_encoder):
        """Test basic reranking"""
        # Setup mock to return scores
        mock_cross_encoder.predict.return_value = np.array([2.5, 1.0, 3.0])
        
        reranker = CrossEncoderReranker()
        
        candidates = [
            {'doc_id': 'doc1', 'text': 'EEG alpha waves', 'score': 0.5, 'metadata': {}},
            {'doc_id': 'doc2', 'text': 'Beta oscillations', 'score': 0.8, 'metadata': {}},
            {'doc_id': 'doc3', 'text': 'Delta activity', 'score': 0.3, 'metadata': {}}
        ]
        
        results = reranker.rerank("EEG frequency bands", candidates)
        
        # Check all results returned
        assert len(results) == 3
        
        # Check result structure
        assert all(isinstance(r, RerankedResult) for r in results)
        assert all(hasattr(r, 'doc_id') for r in results)
        assert all(hasattr(r, 'original_score') for r in results)
        assert all(hasattr(r, 'rerank_score') for r in results)
        assert all(hasattr(r, 'final_score') for r in results)
        
        # Check sorted by final score (descending)
        assert results[0].final_score >= results[1].final_score
        assert results[1].final_score >= results[2].final_score
    
    def test_rerank_top_k(self, mock_cross_encoder):
        """Test reranking with top_k limit"""
        mock_cross_encoder.predict.return_value = np.array([2.5, 1.0, 3.0, 0.5])
        
        reranker = CrossEncoderReranker()
        
        candidates = [
            {'doc_id': f'doc{i}', 'text': f'Text {i}', 'score': 0.5, 'metadata': {}}
            for i in range(4)
        ]
        
        results = reranker.rerank("query", candidates, top_k=2)
        
        # Should return only top 2
        assert len(results) == 2
    
    def test_rerank_empty_candidates(self, mock_cross_encoder):
        """Test reranking with empty candidate list"""
        reranker = CrossEncoderReranker()
        
        results = reranker.rerank("query", [])
        
        assert results == []
        # Should not call model
        mock_cross_encoder.predict.assert_not_called()
    
    def test_rerank_single_candidate(self, mock_cross_encoder):
        """Test reranking with single candidate"""
        mock_cross_encoder.predict.return_value = np.array([1.5])
        
        reranker = CrossEncoderReranker()
        
        candidates = [
            {'doc_id': 'doc1', 'text': 'Single document', 'score': 0.7, 'metadata': {}}
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert len(results) == 1
        assert results[0].doc_id == 'doc1'
    
    def test_score_combination(self, mock_cross_encoder):
        """Test score combination logic"""
        # High rerank score
        mock_cross_encoder.predict.return_value = np.array([5.0])
        
        reranker = CrossEncoderReranker(combine_weight=0.7)
        
        candidates = [
            {'doc_id': 'doc1', 'text': 'Test doc', 'score': 0.2, 'metadata': {}}
        ]
        
        results = reranker.rerank("query", candidates)
        
        # Final score should be weighted combination
        # High rerank score should boost low original score
        assert results[0].final_score > results[0].original_score
        assert results[0].rerank_score == 5.0
    
    def test_rerank_with_metadata(self, mock_cross_encoder):
        """Test that metadata is preserved"""
        mock_cross_encoder.predict.return_value = np.array([1.0])
        
        reranker = CrossEncoderReranker()
        
        candidates = [
            {
                'doc_id': 'doc1',
                'text': 'Test',
                'score': 0.5,
                'metadata': {'pmid': '12345', 'year': 2023},
                'chunk_id': 'chunk_1'
            }
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert results[0].metadata == {'pmid': '12345', 'year': 2023}
        assert results[0].chunk_id == 'chunk_1'
    
    def test_get_statistics(self, mock_cross_encoder):
        """Test statistics retrieval"""
        reranker = CrossEncoderReranker(
            model_name="test-model",
            combine_weight=0.6
        )
        
        stats = reranker.get_statistics()
        
        assert stats['model_name'] == "test-model"
        assert stats['combine_weight'] == 0.6
        assert stats['available'] == True


class TestNoOpReranker:
    """Test no-op reranker fallback"""
    
    def test_initialization(self):
        """Test NoOp reranker initialization"""
        reranker = NoOpReranker()
        assert reranker is not None
    
    def test_rerank_passthrough(self):
        """Test that results pass through unchanged"""
        reranker = NoOpReranker()
        
        candidates = [
            {'doc_id': 'doc1', 'text': 'Text 1', 'score': 0.8, 'metadata': {}},
            {'doc_id': 'doc2', 'text': 'Text 2', 'score': 0.5, 'metadata': {}}
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert len(results) == 2
        assert results[0].doc_id == 'doc1'
        assert results[0].final_score == 0.8
        assert results[0].rerank_score == 0.0
        assert results[1].doc_id == 'doc2'
        assert results[1].final_score == 0.5
    
    def test_rerank_top_k(self):
        """Test top_k with NoOp reranker"""
        reranker = NoOpReranker()
        
        candidates = [
            {'doc_id': f'doc{i}', 'text': f'Text {i}', 'score': 0.5, 'metadata': {}}
            for i in range(5)
        ]
        
        results = reranker.rerank("query", candidates, top_k=3)
        
        assert len(results) == 3
    
    def test_get_statistics(self):
        """Test NoOp statistics"""
        reranker = NoOpReranker()
        
        stats = reranker.get_statistics()
        
        assert stats['available'] == False
        assert stats['model_name'] == 'noop'


class TestRerankedResult:
    """Test RerankedResult dataclass"""
    
    def test_creation(self):
        """Test creating RerankedResult"""
        result = RerankedResult(
            doc_id="test_doc",
            text="Test text",
            original_score=0.5,
            rerank_score=2.0,
            final_score=0.75,
            metadata={'key': 'value'},
            chunk_id='chunk_1'
        )
        
        assert result.doc_id == "test_doc"
        assert result.text == "Test text"
        assert result.original_score == 0.5
        assert result.rerank_score == 2.0
        assert result.final_score == 0.75
        assert result.metadata == {'key': 'value'}
        assert result.chunk_id == 'chunk_1'
    
    def test_optional_chunk_id(self):
        """Test that chunk_id is optional"""
        result = RerankedResult(
            doc_id="test",
            text="text",
            original_score=0.5,
            rerank_score=1.0,
            final_score=0.6,
            metadata={}
        )
        
        assert result.chunk_id is None
