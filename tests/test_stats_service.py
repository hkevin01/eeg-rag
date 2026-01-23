# tests/test_stats_service.py
"""
Comprehensive tests for StatsService.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from eeg_rag.services.stats_service import StatsService, IndexStats, get_stats_service


class TestIndexStats:
    """Tests for IndexStats dataclass."""
    
    def test_index_stats_creation(self):
        """Test creating an IndexStats object."""
        stats = IndexStats(
            total_papers=100,
            papers_by_source={"pubmed": 80, "arxiv": 20},
            papers_with_abstracts=95,
            papers_with_embeddings=90,
            date_range={"min_year": 2020, "max_year": 2024},
            last_updated=datetime.now(),
            index_health={"status": "healthy", "issues": []}
        )
        
        assert stats.total_papers == 100
        assert stats.papers_by_source["pubmed"] == 80
        assert stats.papers_with_abstracts == 95
    
    def test_index_stats_to_dict(self):
        """Test converting IndexStats to dictionary."""
        now = datetime.now()
        stats = IndexStats(
            total_papers=50,
            papers_by_source={"pubmed": 50},
            papers_with_abstracts=45,
            papers_with_embeddings=40,
            date_range={"min_year": 2021, "max_year": 2023},
            last_updated=now,
            index_health={"status": "healthy", "issues": []}
        )
        
        result = stats.to_dict()
        
        assert result["total_papers"] == 50
        assert result["papers_by_source"] == {"pubmed": 50}
        assert result["last_updated"] == now.isoformat()


class TestStatsService:
    """Tests for StatsService."""
    
    @pytest.fixture
    def temp_corpus_dir(self):
        """Create a temporary corpus directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            corpus_dir.mkdir()
            
            # Create test JSONL file
            papers = [
                {"pmid": "12345678", "title": "Paper 1", "abstract": "Abstract 1", "year": 2020, "source": "pubmed"},
                {"pmid": "12345679", "title": "Paper 2", "abstract": "Abstract 2", "year": 2021, "source": "pubmed"},
                {"pmid": "12345680", "title": "Paper 3", "abstract": "", "year": 2022, "source": "arxiv"},
                {"pmid": "12345681", "title": "Paper 4", "abstract": "Abstract 4", "year": 2023, "source": "pubmed"},
                {"pmid": "12345682", "title": "Paper 5", "abstract": "Abstract 5", "year": 2024, "source": "arxiv"},
            ]
            
            jsonl_file = corpus_dir / "test_corpus.jsonl"
            with open(jsonl_file, 'w') as f:
                for paper in papers:
                    f.write(json.dumps(paper) + "\n")
            
            yield corpus_dir
    
    def test_stats_service_init(self, temp_corpus_dir):
        """Test StatsService initialization."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        assert service.corpus_dir == temp_corpus_dir
        assert service.cache_ttl == 300  # default 5 minutes
    
    def test_get_total_papers_from_corpus(self, temp_corpus_dir):
        """Test getting total papers from corpus files."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        total = service.get_total_papers()
        
        assert total == 5
    
    def test_get_papers_by_source(self, temp_corpus_dir):
        """Test getting papers grouped by source."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        by_source = service.get_papers_by_source()
        
        assert by_source.get("pubmed", 0) == 3
        assert by_source.get("arxiv", 0) == 2
    
    def test_get_full_stats(self, temp_corpus_dir):
        """Test getting full statistics."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        stats = service.get_full_stats()
        
        assert isinstance(stats, IndexStats)
        assert stats.total_papers == 5
        assert stats.papers_with_abstracts == 4  # One paper has empty abstract
        assert stats.date_range["min_year"] == 2020
        assert stats.date_range["max_year"] == 2024
    
    def test_get_display_stats(self, temp_corpus_dir):
        """Test getting display-formatted stats."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        display = service.get_display_stats()
        
        assert display["papers_indexed"] == "5"
        assert display["ai_agents"] == "8"
        assert display["citation_accuracy"] == "99.2%"
    
    def test_verify_counts(self, temp_corpus_dir):
        """Test verification report."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        report = service.verify_counts()
        
        assert report["verified_total"] == 5
        assert report["display_total"] == "5"
        assert "corpus_files" in report
        assert "test_corpus.jsonl" in report["corpus_files"]
    
    def test_cache_functionality(self, temp_corpus_dir):
        """Test caching of statistics."""
        service = StatsService(corpus_dir=temp_corpus_dir, cache_ttl_seconds=60)
        
        # First call - populates cache
        total1 = service.get_total_papers()
        
        # Second call - should use cache
        total2 = service.get_total_papers()
        
        assert total1 == total2 == 5
        
        # Verify cache exists
        assert service._is_cache_valid("total_papers")
    
    def test_invalidate_cache(self, temp_corpus_dir):
        """Test cache invalidation."""
        service = StatsService(corpus_dir=temp_corpus_dir)
        
        # Populate cache
        service.get_total_papers()
        assert service._is_cache_valid("total_papers")
        
        # Invalidate
        service.invalidate_cache()
        assert not service._is_cache_valid("total_papers")
    
    def test_empty_corpus(self):
        """Test behavior with empty corpus directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            
            service = StatsService(corpus_dir=empty_dir)
            
            assert service.get_total_papers() == 0
            assert service.get_papers_by_source() == {}
    
    def test_nonexistent_database(self):
        """Test behavior when database doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StatsService(
                papers_db_path=Path(tmpdir) / "nonexistent.db",
                corpus_dir=Path(tmpdir) / "nonexistent"
            )
            
            report = service.verify_counts()
            
            assert "Papers database not found" in report["inconsistencies"][0]


class TestGetStatsService:
    """Tests for singleton getter."""
    
    def test_get_stats_service_singleton(self):
        """Test that get_stats_service returns singleton."""
        service1 = get_stats_service()
        service2 = get_stats_service()
        
        assert service1 is service2
    
    def test_get_stats_service_type(self):
        """Test that get_stats_service returns StatsService."""
        service = get_stats_service()
        
        assert isinstance(service, StatsService)


class TestLargeNumbers:
    """Tests for large number formatting."""
    
    @pytest.fixture
    def large_corpus(self):
        """Create corpus with many papers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir)
            
            # Create 1500 papers
            papers = [
                {"pmid": f"{10000000 + i}", "title": f"Paper {i}", "abstract": f"Abstract {i}", "year": 2020}
                for i in range(1500)
            ]
            
            jsonl_file = corpus_dir / "large_corpus.jsonl"
            with open(jsonl_file, 'w') as f:
                for paper in papers:
                    f.write(json.dumps(paper) + "\n")
            
            yield corpus_dir
    
    def test_large_paper_count_display(self, large_corpus):
        """Test formatting of large paper counts."""
        service = StatsService(corpus_dir=large_corpus)
        
        display = service.get_display_stats()
        
        # Should be comma-formatted
        assert display["papers_indexed"] == "1,500"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
