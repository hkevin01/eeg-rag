#!/usr/bin/env python3
"""
Tests for Hybrid Retrieval System
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.eeg_rag.retrieval.hybrid_retriever import (
    HybridRetriever, RetrievalResult
)


class TestRetrievalResult:
    """Test retrieval result data class"""
    
    def test_creation(self):
        """Test result creation"""
        result = RetrievalResult(
            doc_id="doc1",
            score=0.85,
            content="Test document about EEG",
            metadata={"source": "test"},
            bm25_score=0.7,
            dense_score=0.9
        )
        
        assert result.doc_id == "doc1"
        assert result.score == 0.85
        assert result.bm25_score == 0.7
        assert result.dense_score == 0.9
    
    def test_to_dict(self):
        """Test result serialization"""
        result = RetrievalResult(
            doc_id="doc1",
            score=0.85,
            content="A" * 300,  # Long content
            metadata={"test": True}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['doc_id'] == "doc1"
        assert result_dict['score'] == 0.85
        assert len(result_dict['content']) <= 203  # Truncated + "..."
        assert result_dict['metadata']['test'] is True


class TestHybridRetriever:
    """Test hybrid retrieval functionality"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            "EEG measures electrical activity in the brain using electrodes.",
            "Epilepsy is characterized by recurrent seizures and abnormal brain activity.",
            "Sleep stages can be identified through EEG patterns including sleep spindles.",
            "Brain-computer interfaces use EEG signals to control external devices.",
            "Motor imagery tasks activate specific brain regions visible in EEG."
        ]
    
    @pytest.fixture
    def sample_metadata(self):
        return [
            {"title": "EEG Basics", "year": 2020},
            {"title": "Epilepsy Research", "year": 2021},
            {"title": "Sleep Studies", "year": 2022},
            {"title": "BCI Development", "year": 2023},
            {"title": "Motor Imagery", "year": 2024}
        ]
    
    @pytest.fixture
    def retriever(self, sample_documents, sample_metadata):
        retriever = HybridRetriever(alpha=0.6, fusion_method='weighted_sum')
        retriever.add_documents(sample_documents, sample_metadata)
        return retriever
    
    def test_init(self):
        """Test retriever initialization"""
        retriever = HybridRetriever(alpha=0.7, fusion_method='max')
        
        assert retriever.alpha == 0.7
        assert retriever.fusion_method == 'max'
        assert retriever.documents == []
        assert retriever.bm25 is None
    
    def test_add_documents(self, sample_documents, sample_metadata):
        """Test document addition"""
        retriever = HybridRetriever()
        retriever.add_documents(sample_documents, sample_metadata)
        
        assert len(retriever.documents) == 5
        assert len(retriever.doc_metadata) == 5
        assert retriever.bm25 is not None
    
    def test_tokenize_for_bm25(self):
        """Test BM25 tokenization"""
        retriever = HybridRetriever()
        
        text = "EEG electroencephalogram shows brain-computer interface activity."
        tokens = retriever._tokenize_for_bm25(text)
        
        assert "eeg" in tokens  # Should normalize
        assert "brain_computer" in tokens  # Should handle compound terms
        assert "the" not in tokens  # Should remove stopwords
    
    @patch('rank_bm25.BM25Okapi')
    def test_build_bm25_index(self, mock_bm25, sample_documents):
        """Test BM25 index building"""
        retriever = HybridRetriever()
        retriever.documents = sample_documents
        
        retriever._build_bm25_index()
        
        mock_bm25.assert_called_once()
    
    def test_get_bm25_scores_no_index(self, retriever):
        """Test BM25 scoring without index"""
        retriever.bm25 = None
        
        scores = retriever._get_bm25_scores("test query", 5)
        
        assert scores == {}
    
    @patch('rank_bm25.BM25Okapi')
    def test_get_bm25_scores_with_index(self, mock_bm25, retriever):
        """Test BM25 scoring with index"""
        # Mock BM25 instance
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_scores.return_value = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
        retriever.bm25 = mock_bm25_instance
        
        scores = retriever._get_bm25_scores("EEG seizure", 3)
        
        assert len(scores) <= 3
        assert all(0 <= score <= 1 for score in scores.values())
    
    def test_get_dense_scores_no_retriever(self, retriever):
        """Test dense scoring without dense retriever"""
        retriever.dense_retriever = None
        
        scores = retriever._get_dense_scores("test query", 5)
        
        assert scores == {}
    
    def test_get_dense_scores_with_mock_retriever(self, retriever):
        """Test dense scoring with mock retriever"""
        # Mock dense retriever
        mock_dense = Mock()
        mock_results = [
            {'doc_id': 0, 'score': 0.9},
            {'doc_id': 1, 'score': 0.7},
            {'doc_id': 2, 'score': 0.8}
        ]
        mock_dense.search.return_value = mock_results
        retriever.dense_retriever = mock_dense
        
        scores = retriever._get_dense_scores("EEG", 3)
        
        assert len(scores) == 3
        assert scores[0] == 0.9
    
    def test_combine_scores_weighted_sum(self, retriever):
        """Test score combination with weighted sum"""
        retriever.alpha = 0.6  # 60% dense, 40% BM25
        retriever.fusion_method = 'weighted_sum'
        
        bm25_scores = {0: 0.8, 1: 0.4}
        dense_scores = {0: 0.6, 1: 0.9}
        
        results = retriever._combine_scores(bm25_scores, dense_scores, 5)
        
        assert len(results) == 2
        
        # Check score calculation: 0.6 * dense + 0.4 * bm25
        doc0_expected = 0.6 * 0.6 + 0.4 * 0.8  # = 0.68
        doc1_expected = 0.6 * 0.9 + 0.4 * 0.4  # = 0.7
        
        # Results should be sorted by score (highest first)
        assert results[0].score == doc1_expected
        assert results[1].score == doc0_expected
    
    def test_combine_scores_max_fusion(self, retriever):
        """Test score combination with max fusion"""
        retriever.fusion_method = 'max'
        
        bm25_scores = {0: 0.8, 1: 0.4}
        dense_scores = {0: 0.6, 1: 0.9}
        
        results = retriever._combine_scores(bm25_scores, dense_scores, 5)
        
        # Max fusion should take the maximum of each score
        doc0_expected = max(0.8, 0.6)  # = 0.8
        doc1_expected = max(0.4, 0.9)  # = 0.9
        
        assert results[0].score == doc1_expected
        assert results[1].score == doc0_expected
    
    def test_search_no_documents(self):
        """Test search with no documents"""
        retriever = HybridRetriever()
        
        results = retriever.search("test query")
        
        assert results == []
    
    @patch.object(HybridRetriever, '_get_bm25_scores')
    @patch.object(HybridRetriever, '_get_dense_scores')
    def test_search_with_results(self, mock_dense, mock_bm25, retriever):
        """Test search with mock results"""
        mock_bm25.return_value = {0: 0.8, 1: 0.6}
        mock_dense.return_value = {0: 0.7, 1: 0.9}
        
        results = retriever.search("EEG epilepsy", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].score >= results[1].score  # Should be sorted
    
    def test_get_stats(self, retriever):
        """Test statistics retrieval"""
        stats = retriever.get_stats()
        
        assert 'total_documents' in stats
        assert 'bm25_available' in stats
        assert 'dense_retriever_available' in stats
        assert 'alpha' in stats
        assert 'fusion_method' in stats
        
        assert stats['total_documents'] == 5
        assert stats['alpha'] == 0.6
    
    def test_save_load_config(self, retriever, tmp_path):
        """Test configuration save/load"""
        config_file = tmp_path / "config.json"
        
        # Save config
        retriever.save_config(str(config_file))
        
        # Create new retriever and load config
        new_retriever = HybridRetriever()
        new_retriever.load_config(str(config_file))
        
        assert new_retriever.alpha == retriever.alpha
        assert new_retriever.fusion_method == retriever.fusion_method
    
    def test_eeg_specific_tokenization(self):
        """Test EEG-specific text processing"""
        retriever = HybridRetriever()
        
        text = "The electroencephalography showed event-related potentials in motor imagery tasks."
        tokens = retriever._tokenize_for_bm25(text)
        
        # Should convert to abbreviations
        assert "eeg" in tokens
        assert "erp" in tokens
        assert "motor_imagery" in tokens
        
        # Should keep EEG-specific terms even if they're typically stop words
        text_with_coords = "The alpha waves at C3 and C4 electrodes."
        tokens = retriever._tokenize_for_bm25(text_with_coords)
        assert "alpha" in tokens
        assert "c3" in tokens
        assert "c4" in tokens


class TestIntegration:
    """Integration tests for hybrid retrieval"""
    
    def test_end_to_end_search(self):
        """Test complete search workflow"""
        documents = [
            "EEG electroencephalography measures brain electrical activity.",
            "Epileptic seizures show characteristic spike patterns in EEG recordings.",
            "Sleep spindles are transient bursts of rhythmic brain wave activity.",
            "Brain-computer interfaces decode motor intentions from EEG signals."
        ]
        
        retriever = HybridRetriever(alpha=0.5)
        retriever.add_documents(documents)
        
        # Test different queries
        results = retriever.search("epilepsy seizure detection", top_k=2)
        
        assert len(results) <= 2
        assert all(r.score > 0 for r in results)
        
        # Check that epilepsy-related document ranks high
        top_result = results[0]
        assert "epilep" in top_result.content.lower() or "seizure" in top_result.content.lower()


if __name__ == "__main__":
    pytest.main([__file__])
