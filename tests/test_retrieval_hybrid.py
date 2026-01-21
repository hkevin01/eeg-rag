"""
Unit tests for hybrid retrieval system.

Tests BM25, Dense, Hybrid retrievers and Query Expansion.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from eeg_rag.retrieval import (
    BM25Retriever, BM25Result,
    DenseRetriever, DenseResult,
    HybridRetriever, HybridResult,
    EEGQueryExpander
)


class TestBM25Retriever:
    """Test BM25 sparse retriever."""
    
    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            {"id": "1", "text": "CNN for epilepsy seizure detection using EEG", "metadata": {"year": 2019}},
            {"id": "2", "text": "RNN for sleep staging classification", "metadata": {"year": 2020}},
            {"id": "3", "text": "Motor imagery BCI using deep learning", "metadata": {"year": 2021}},
        ]
    
    @pytest.fixture
    def retriever(self):
        """Create BM25 retriever with temp cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield BM25Retriever(cache_dir=tmpdir)
    
    def test_initialization(self, retriever):
        """Test BM25 retriever initialization."""
        assert retriever.bm25 is None
        assert len(retriever.documents) == 0
        assert len(retriever.tokenized_corpus) == 0
    
    def test_tokenize(self, retriever):
        """Test text tokenization."""
        tokens = retriever._tokenize("CNN for Seizure Detection")
        assert tokens == ["cnn", "for", "seizure", "detection"]
    
    def test_index_documents(self, retriever, sample_docs):
        """Test document indexing."""
        retriever.index_documents(sample_docs)
        
        assert retriever.bm25 is not None
        assert len(retriever.documents) == 3
        assert len(retriever.tokenized_corpus) == 3
    
    def test_search(self, retriever, sample_docs):
        """Test BM25 search."""
        retriever.index_documents(sample_docs)
        
        results = retriever.search("seizure detection", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, BM25Result) for r in results)
        assert results[0].doc_id == "1"  # Most relevant
        assert results[0].score > 0
    
    def test_search_no_results(self, retriever, sample_docs):
        """Test search with no matches."""
        retriever.index_documents(sample_docs)
        
        results = retriever.search("quantum computing", top_k=5)
        
        # Should return all docs but with low/zero scores
        assert len(results) == 3
    
    def test_search_without_index(self, retriever):
        """Test search fails without index."""
        with pytest.raises(ValueError, match="No BM25 index available"):
            retriever.search("test query")


class TestDenseRetriever:
    """Test Dense semantic retriever."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Mock VectorDB for testing."""
        mock_vdb = Mock()
        mock_vdb.search.return_value = [
            Mock(doc_id="1", score=0.85, payload={"text": "CNN seizure", "metadata": {}}, chunk_id="c1"),
            Mock(doc_id="2", score=0.75, payload={"text": "RNN sleep", "metadata": {}}, chunk_id="c2"),
        ]
        return mock_vdb
    
    def test_initialization(self):
        """Test dense retriever initialization."""
        with patch('eeg_rag.retrieval.dense_retriever.VectorDB'):
            retriever = DenseRetriever(url="http://test:6333", collection_name="test")
            assert retriever is not None
    
    def test_search(self, mock_vector_db):
        """Test dense search."""
        with patch('eeg_rag.retrieval.dense_retriever.VectorDB', return_value=mock_vector_db):
            retriever = DenseRetriever()
            retriever.vector_db = mock_vector_db
            
            results = retriever.search("seizure detection", top_k=5)
            
            assert len(results) == 2
            assert all(isinstance(r, DenseResult) for r in results)
            assert results[0].doc_id == "1"
            assert results[0].score == 0.85


class TestHybridRetriever:
    """Test Hybrid retriever with RRF fusion."""
    
    @pytest.fixture
    def sample_docs(self):
        """Sample documents."""
        return [
            {"id": "1", "text": "CNN epilepsy seizure detection EEG", "metadata": {}},
            {"id": "2", "text": "RNN sleep staging classification", "metadata": {}},
            {"id": "3", "text": "Deep learning motor imagery BCI", "metadata": {}},
        ]
    
    @pytest.fixture
    def bm25_retriever(self, sample_docs):
        """Create indexed BM25 retriever."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = BM25Retriever(cache_dir=tmpdir)
            retriever.index_documents(sample_docs)
            yield retriever
    
    @pytest.fixture
    def mock_dense_retriever(self):
        """Mock dense retriever."""
        mock = Mock()
        mock.search.return_value = [
            DenseResult(doc_id="1", score=0.85, text="CNN epilepsy", metadata={}, chunk_id="c1"),
            DenseResult(doc_id="3", score=0.75, text="Deep learning", metadata={}, chunk_id="c3"),
        ]
        return mock
    
    def test_initialization_no_expansion(self, bm25_retriever, mock_dense_retriever):
        """Test hybrid retriever init without query expansion."""
        hybrid = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=mock_dense_retriever,
            use_query_expansion=False
        )
        
        assert hybrid.bm25_weight == 0.5
        assert hybrid.dense_weight == 0.5
        assert hybrid.rrf_k == 60
        assert hybrid.query_expander is None
    
    def test_initialization_with_expansion(self, bm25_retriever, mock_dense_retriever):
        """Test hybrid retriever init with query expansion."""
        hybrid = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=mock_dense_retriever,
            use_query_expansion=True
        )
        
        assert hybrid.query_expander is not None
        assert isinstance(hybrid.query_expander, EEGQueryExpander)
    
    def test_search_basic(self, bm25_retriever, mock_dense_retriever):
        """Test basic hybrid search."""
        hybrid = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=mock_dense_retriever,
            use_query_expansion=False
        )
        
        results = hybrid.search("seizure detection", top_k=3, retrieve_k=10)
        
        assert len(results) <= 3
        assert all(isinstance(r, HybridResult) for r in results)
        assert all(hasattr(r, 'rrf_score') for r in results)
        assert all(hasattr(r, 'bm25_rank') for r in results)
        assert all(hasattr(r, 'dense_rank') for r in results)
    
    def test_rrf_fusion(self, bm25_retriever, mock_dense_retriever):
        """Test RRF score computation."""
        hybrid = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=mock_dense_retriever,
            bm25_weight=0.5,
            dense_weight=0.5,
            rrf_k=60
        )
        
        results = hybrid.search("seizure", top_k=5, retrieve_k=10)
        
        # Results should be sorted by RRF score
        for i in range(len(results) - 1):
            assert results[i].rrf_score >= results[i+1].rrf_score
    
    def test_custom_weights(self, bm25_retriever, mock_dense_retriever):
        """Test custom BM25/Dense weights."""
        hybrid = HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=mock_dense_retriever,
            bm25_weight=0.7,
            dense_weight=0.3,
            use_query_expansion=False
        )
        
        results = hybrid.search("seizure", top_k=3, retrieve_k=10)
        
        assert len(results) > 0
        # BM25 should have more influence
        assert results[0].bm25_score > 0 or results[0].dense_score > 0


class TestEEGQueryExpander:
    """Test EEG domain query expansion."""
    
    @pytest.fixture
    def expander(self):
        """Create query expander."""
        return EEGQueryExpander()
    
    def test_initialization(self, expander):
        """Test expander initialization."""
        assert len(expander.synonyms) > 100
        assert "cnn" in expander.synonyms
        assert "seizure" in expander.synonyms
        assert "eeg" in expander.synonyms
    
    def test_expand_acronym(self, expander):
        """Test acronym expansion."""
        expanded = expander.expand("CNN", max_expansions=2)
        
        assert "cnn" in expanded.lower()
        assert "convolutional" in expanded.lower() or "neural" in expanded.lower()
    
    def test_expand_medical_term(self, expander):
        """Test medical term expansion."""
        expanded = expander.expand("seizure", max_expansions=2)
        
        assert "seizure" in expanded.lower()
        assert any(term in expanded.lower() for term in ["epilepsy", "epileptic", "ictal"])
    
    def test_expand_frequency_band(self, expander):
        """Test frequency band expansion."""
        expanded = expander.expand("alpha band", max_expansions=2)
        
        assert "alpha" in expanded.lower()
        # Should include Hz range or wave terminology
        assert "hz" in expanded.lower() or "wave" in expanded.lower()
    
    def test_no_expansion_for_unknown(self, expander):
        """Test no expansion for unknown terms."""
        original = "quantum computing"
        expanded = expander.expand(original, max_expansions=2)
        
        # Should keep original terms
        assert "quantum" in expanded.lower()
        assert "computing" in expanded.lower()
    
    def test_get_synonyms(self, expander):
        """Test synonym lookup."""
        syns = expander.get_synonyms("cnn")
        
        assert len(syns) > 0
        assert "convolutional neural network" in syns
    
    def test_has_synonyms(self, expander):
        """Test synonym existence check."""
        assert expander.has_synonyms("cnn") is True
        assert expander.has_synonyms("seizure") is True
        assert expander.has_synonyms("unknownterm123") is False
    
    def test_bigram_expansion(self, expander):
        """Test multi-word phrase expansion."""
        expanded = expander.expand("motor imagery", max_expansions=2)
        
        # Should expand bigram "motor imagery"
        assert "motor" in expanded.lower()
        assert "imagery" in expanded.lower()
        # Should include MI or movement imagination
        assert "mi" in expanded.lower() or "movement" in expanded.lower()


class TestIntegration:
    """Integration tests for full retrieval pipeline."""
    
    @pytest.fixture
    def full_system(self):
        """Set up full retrieval system with real components."""
        docs = [
            {"id": "1", "text": "CNN for epilepsy seizure detection using deep learning", "metadata": {"year": 2019}},
            {"id": "2", "text": "RNN for sleep staging and classification", "metadata": {"year": 2020}},
            {"id": "3", "text": "Motor imagery BCI using convolutional networks", "metadata": {"year": 2021}},
            {"id": "4", "text": "Epileptic seizure prediction with LSTM networks", "metadata": {"year": 2022}},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create BM25 retriever
            bm25 = BM25Retriever(cache_dir=tmpdir)
            bm25.index_documents(docs)
            
            # Mock dense retriever
            dense = Mock()
            dense.search.return_value = [
                DenseResult(doc_id="1", score=0.9, text=docs[0]["text"], metadata={}),
                DenseResult(doc_id="4", score=0.85, text=docs[3]["text"], metadata={}),
                DenseResult(doc_id="2", score=0.7, text=docs[1]["text"], metadata={}),
            ]
            
            # Create hybrid
            hybrid = HybridRetriever(
                bm25_retriever=bm25,
                dense_retriever=dense,
                use_query_expansion=True
            )
            
            yield hybrid
    
    def test_end_to_end_search(self, full_system):
        """Test complete search pipeline."""
        results = full_system.search(
            query="CNN seizure detection",
            top_k=3,
            retrieve_k=10
        )
        
        assert len(results) == 3
        assert results[0].doc_id is not None
        assert results[0].rrf_score > 0
        assert results[0].text is not None
    
    def test_query_expansion_in_pipeline(self, full_system):
        """Test query expansion is applied in search."""
        # Disable expansion
        full_system.use_query_expansion = False
        full_system.query_expander = None
        results_no_exp = full_system.search("CNN", top_k=3, retrieve_k=10)
        
        # Enable expansion
        full_system.use_query_expansion = True
        full_system.query_expander = EEGQueryExpander()
        results_with_exp = full_system.search("CNN", top_k=3, retrieve_k=10)
        
        # Both should return results
        assert len(results_no_exp) > 0
        assert len(results_with_exp) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
