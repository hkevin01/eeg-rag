# tests/test_search_history.py
"""
Tests for the search history system.
Tests database models, history manager, and search integration.
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

from eeg_rag.db.history_models import (
    SearchHistoryDB, SearchResult, SearchQuery, SearchSession
)
from eeg_rag.services.history_manager import HistoryManager
from eeg_rag.search.search_with_history import SearchWithHistory, create_search_with_history


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def history_db(temp_db):
    """Create a SearchHistoryDB instance for testing."""
    return SearchHistoryDB(db_path=temp_db)


@pytest.fixture
def history_manager(temp_db):
    """Create a HistoryManager instance for testing."""
    return HistoryManager(db_path=temp_db)


@pytest.fixture
def sample_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            paper_id="pmid_12345678",
            title="EEG Analysis for Epilepsy Detection Using Deep Learning",
            authors=["Smith J", "Jones K", "Brown M"],
            year=2024,
            abstract="This paper presents a novel deep learning approach for detecting epileptic seizures from EEG signals...",
            source="pubmed",
            relevance_score=0.95,
            pmid="12345678",
            doi="10.1234/example.2024.001"
        ),
        SearchResult(
            paper_id="arxiv_2024.12345",
            title="Transformer Models for EEG Signal Classification",
            authors=["Lee S", "Kim H"],
            year=2024,
            abstract="We propose a transformer-based architecture for multi-class EEG classification...",
            source="arxiv",
            relevance_score=0.87,
            url="https://arxiv.org/abs/2024.12345"
        ),
        SearchResult(
            paper_id="pmid_87654321",
            title="P300 Component Analysis in Brain-Computer Interfaces",
            authors=["Garcia M", "Wang L", "Martinez A"],
            year=2023,
            abstract="The P300 event-related potential is a key component in EEG-based BCIs...",
            source="pubmed",
            relevance_score=0.82,
            pmid="87654321"
        )
    ]


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            paper_id="test_001",
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2024,
            abstract="Test abstract",
            source="pubmed",
            relevance_score=0.9
        )
        
        assert result.paper_id == "test_001"
        assert result.title == "Test Paper"
        assert len(result.authors) == 2
        assert result.year == 2024
        assert result.relevance_score == 0.9
    
    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        result = SearchResult(
            paper_id="test_001",
            title="Test Paper",
            authors=["Author A"],
            year=2024,
            abstract="Test",
            source="pubmed",
            relevance_score=0.9
        )
        
        d = result.to_dict()
        assert d['paper_id'] == "test_001"
        assert d['title'] == "Test Paper"
        assert isinstance(d['authors'], list)
    
    def test_search_result_from_dict(self):
        """Test creating search result from dictionary."""
        data = {
            'paper_id': 'test_002',
            'title': 'From Dict Paper',
            'authors': ['Author X'],
            'year': 2023,
            'abstract': 'From dict',
            'source': 'arxiv',
            'relevance_score': 0.85
        }
        
        result = SearchResult.from_dict(data)
        assert result.paper_id == 'test_002'
        assert result.source == 'arxiv'


class TestSearchHistoryDB:
    """Tests for SearchHistoryDB."""
    
    def test_database_initialization(self, history_db):
        """Test that database initializes correctly."""
        assert history_db.db_path.exists()
    
    def test_save_search(self, history_db, sample_results):
        """Test saving a search query."""
        query = history_db.save_search(
            query_text="EEG epilepsy deep learning",
            query_type="hybrid",
            results=sample_results,
            execution_time_ms=150.5
        )
        
        assert query.id is not None
        assert query.query_text == "EEG epilepsy deep learning"
        assert query.query_type == "hybrid"
        assert query.result_count == 3
        assert query.execution_time_ms == 150.5
    
    def test_get_recent_searches(self, history_db, sample_results):
        """Test retrieving recent searches."""
        # Save multiple searches
        history_db.save_search("query 1", "hybrid", sample_results[:1], 100)
        history_db.save_search("query 2", "semantic", sample_results[1:2], 120)
        history_db.save_search("query 3", "natural", sample_results, 80)
        
        recent = history_db.get_recent_searches(limit=10)
        
        assert len(recent) == 3
        # Should be ordered by timestamp DESC
        assert recent[0].query_text == "query 3"
    
    def test_search_history(self, history_db, sample_results):
        """Test searching through history."""
        history_db.save_search("EEG epilepsy detection", "hybrid", sample_results, 100)
        history_db.save_search("P300 brain computer interface", "hybrid", sample_results, 100)
        history_db.save_search("EEG sleep staging", "semantic", sample_results, 100)
        
        matches = history_db.search_history("EEG")
        
        assert len(matches) == 2
    
    def test_toggle_star(self, history_db, sample_results):
        """Test toggling star status."""
        query = history_db.save_search("test query", "hybrid", sample_results, 100)
        
        # Initially not starred
        new_status = history_db.toggle_star(query.id)
        assert new_status is True
        
        # Toggle again
        new_status = history_db.toggle_star(query.id)
        assert new_status is False
    
    def test_add_note(self, history_db, sample_results):
        """Test adding notes to a search."""
        query = history_db.save_search("test query", "hybrid", sample_results, 100)
        
        history_db.add_note(query.id, "This is a useful search")
        
        retrieved = history_db.get_search_by_id(query.id)
        assert retrieved.notes == "This is a useful search"
    
    def test_record_click(self, history_db, sample_results):
        """Test recording result clicks."""
        query = history_db.save_search("test query", "hybrid", sample_results, 100)
        
        history_db.record_click(query.id, "pmid_12345678")
        history_db.record_click(query.id, "arxiv_2024.12345")
        
        retrieved = history_db.get_search_by_id(query.id)
        assert len(retrieved.clicked_results) == 2
        assert "pmid_12345678" in retrieved.clicked_results
    
    def test_delete_search(self, history_db, sample_results):
        """Test deleting a search."""
        query = history_db.save_search("to delete", "hybrid", sample_results, 100)
        query_id = query.id
        
        history_db.delete_search(query_id)
        
        retrieved = history_db.get_search_by_id(query_id)
        assert retrieved is None
    
    def test_clear_history(self, history_db, sample_results):
        """Test clearing history."""
        # Save some searches
        history_db.save_search("query 1", "hybrid", sample_results, 100)
        q2 = history_db.save_search("query 2", "hybrid", sample_results, 100)
        history_db.toggle_star(q2.id)  # Star one
        history_db.save_search("query 3", "hybrid", sample_results, 100)
        
        # Clear keeping starred
        count = history_db.clear_history(keep_starred=True)
        
        assert count == 2  # Only unstarred deleted
        
        remaining = history_db.get_recent_searches(limit=10)
        assert len(remaining) == 1
        assert remaining[0].starred is True
    
    def test_create_session(self, history_db):
        """Test creating a session."""
        session = history_db.create_session(name="Epilepsy Research", topic="Detection methods")
        
        assert session.id is not None
        assert session.name == "Epilepsy Research"
        assert session.topic == "Detection methods"
    
    def test_save_paper(self, history_db, sample_results):
        """Test saving a paper."""
        history_db.save_paper(sample_results[0], notes="Important paper", tags=["epilepsy", "deep-learning"])
        
        papers = history_db.get_saved_papers()
        
        assert len(papers) == 1
        assert papers[0]['paper_id'] == "pmid_12345678"
        assert papers[0]['notes'] == "Important paper"
        assert "epilepsy" in papers[0]['tags']
    
    def test_get_search_stats(self, history_db, sample_results):
        """Test getting search statistics."""
        history_db.save_search("query 1", "hybrid", sample_results, 100)
        history_db.save_search("query 2", "semantic", sample_results[:1], 120)
        
        stats = history_db.get_search_stats()
        
        assert stats['total_searches'] == 2
        assert 'by_type' in stats
        assert stats['by_type'].get('hybrid', 0) == 1
        assert stats['by_type'].get('semantic', 0) == 1


class TestHistoryManager:
    """Tests for HistoryManager."""
    
    def test_manager_initialization(self, history_manager):
        """Test manager initializes correctly."""
        assert history_manager.db is not None
    
    def test_record_search(self, history_manager, sample_results):
        """Test recording a search."""
        query = history_manager.record_search(
            query_text="EEG analysis",
            query_type="hybrid",
            results=sample_results,
            execution_time_ms=200
        )
        
        assert query.id is not None
        assert query.query_text == "EEG analysis"
    
    def test_session_management(self, history_manager, sample_results):
        """Test session creation and usage."""
        session = history_manager.start_session(name="Test Session", topic="EEG Research")
        
        assert history_manager.get_current_session() is not None
        assert session.name == "Test Session"
        
        # Record search in session
        query = history_manager.record_search(
            query_text="in session query",
            query_type="hybrid",
            results=sample_results,
            execution_time_ms=100
        )
        
        assert query.session_id == session.id
        
        history_manager.end_session()
        assert history_manager.get_current_session() is None
    
    def test_clear_old_history(self, history_manager, sample_results):
        """Test clearing old history."""
        history_manager.record_search("query", "hybrid", sample_results, 100)
        
        # Clear with 0 days should clear all non-starred
        count = history_manager.clear_old_history(days=0, keep_starred=True)
        
        assert count == 1
    
    def test_get_stats(self, history_manager, sample_results):
        """Test getting statistics."""
        history_manager.record_search("q1", "hybrid", sample_results, 100)
        history_manager.record_search("q2", "semantic", sample_results, 100)
        
        stats = history_manager.get_stats()
        
        assert stats['total_searches'] == 2


class TestSearchWithHistory:
    """Tests for SearchWithHistory wrapper."""
    
    @pytest.mark.asyncio
    async def test_search_with_history(self, temp_db, sample_results):
        """Test search wrapper records history."""
        manager = HistoryManager(db_path=temp_db)
        
        async def mock_search(query_text, **kwargs):
            return [
                {
                    'paper_id': 'test_001',
                    'title': 'Test Result',
                    'authors': ['Author A'],
                    'year': 2024,
                    'abstract': 'Test abstract',
                    'source': 'pubmed',
                    'relevance_score': 0.9
                }
            ]
        
        search = SearchWithHistory(search_fn=mock_search, history_manager=manager)
        
        results = await search.search("test query", query_type="hybrid")
        
        assert len(results) == 1
        assert results[0].title == "Test Result"
        
        # Check history was recorded
        recent = manager.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0].query_text == "test query"
    
    @pytest.mark.asyncio
    async def test_record_click(self, temp_db, sample_results):
        """Test recording clicks through wrapper."""
        manager = HistoryManager(db_path=temp_db)
        
        async def mock_search(query_text, **kwargs):
            return sample_results
        
        search = SearchWithHistory(search_fn=mock_search, history_manager=manager)
        
        results = await search.search("test query")
        search.record_click("pmid_12345678")
        
        # Verify click was recorded
        query = manager.get_recent(limit=1)[0]
        assert "pmid_12345678" in query.clicked_results
    
    @pytest.mark.asyncio
    async def test_record_feedback(self, temp_db, sample_results):
        """Test recording feedback through wrapper."""
        manager = HistoryManager(db_path=temp_db)
        
        async def mock_search(query_text, **kwargs):
            return sample_results
        
        search = SearchWithHistory(search_fn=mock_search, history_manager=manager)
        
        await search.search("test query")
        search.record_feedback("helpful")
        
        query = manager.get_recent(limit=1)[0]
        assert query.user_feedback == "helpful"


class TestDataPersistence:
    """Tests for data persistence across sessions."""
    
    def test_data_persists(self, temp_db, sample_results):
        """Test that data persists when reopening database."""
        # Create and save data
        db1 = SearchHistoryDB(db_path=temp_db)
        db1.save_search("persistent query", "hybrid", sample_results, 100)
        db1.save_paper(sample_results[0], notes="Saved paper")
        
        # Reopen database
        db2 = SearchHistoryDB(db_path=temp_db)
        
        searches = db2.get_recent_searches(limit=10)
        papers = db2.get_saved_papers()
        
        assert len(searches) == 1
        assert searches[0].query_text == "persistent query"
        assert len(papers) == 1
        assert papers[0]['notes'] == "Saved paper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
