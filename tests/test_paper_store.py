#!/usr/bin/env python3
"""
Comprehensive tests for PaperStore - Production Paper Database

Tests cover:
- Paper CRUD operations
- Batch insertion with deduplication
- FTS5 full-text search
- Statistics and analytics
- Edge cases and boundary conditions
- Performance under load
- Concurrent access
"""

import pytest
import tempfile
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor
import time
import threading

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.db.paper_store import Paper, PaperStore


class TestPaperDataclass:
    """Test the Paper dataclass."""
    
    def test_paper_creation_minimal(self):
        """Test creating a paper with minimal fields."""
        paper = Paper(
            paper_id="test_001",
            title="Test Paper",
            abstract="Test abstract"
        )
        assert paper.paper_id == "test_001"
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test abstract"
        assert paper.authors == []
        assert paper.year is None
        assert paper.source == "unknown"
    
    def test_paper_creation_full(self):
        """Test creating a paper with all fields."""
        paper = Paper(
            paper_id="test_002",
            title="Full Test Paper",
            abstract="Full test abstract",
            authors=["Author One", "Author Two"],
            year=2024,
            source="pubmed",
            doi="10.1234/test",
            pmid="12345678",
            arxiv_id="1234.56789",
            s2_id="abc123",
            openalex_id="W123456",
            url="https://example.com/paper",
            pdf_url="https://example.com/paper.pdf",
            journal="Test Journal",
            venue="Test Conference",
            citation_count=42,
            keywords=["EEG", "neuroscience", "P300"]
        )
        assert paper.paper_id == "test_002"
        assert len(paper.authors) == 2
        assert paper.year == 2024
        assert paper.citation_count == 42
        assert "EEG" in paper.keywords
    
    def test_paper_to_dict(self):
        """Test Paper to_dict method."""
        paper = Paper(
            paper_id="test_003",
            title="Dict Test",
            abstract="Testing dict conversion",
            authors=["Author"],
            year=2023
        )
        d = paper.to_dict()
        assert d["paper_id"] == "test_003"
        assert d["title"] == "Dict Test"
        assert d["authors"] == ["Author"]
        assert d["year"] == 2023
    
    def test_paper_from_dict(self):
        """Test Paper from_dict method."""
        data = {
            "paper_id": "test_004",
            "title": "From Dict Test",
            "abstract": "Testing from dict",
            "authors": ["Author A", "Author B"],
            "year": 2022,
            "source": "openalex"
        }
        paper = Paper.from_dict(data)
        assert paper.paper_id == "test_004"
        assert paper.title == "From Dict Test"
        assert len(paper.authors) == 2
        assert paper.year == 2022
        assert paper.source == "openalex"
    
    def test_paper_from_dict_with_json_string_authors(self):
        """Test Paper from_dict with JSON string authors."""
        data = {
            "paper_id": "test_005",
            "title": "JSON Authors Test",
            "abstract": "Test",
            "authors": '["Author One", "Author Two"]'
        }
        paper = Paper.from_dict(data)
        assert paper.authors == ["Author One", "Author Two"]
    
    def test_paper_equality(self):
        """Test Paper equality based on paper_id."""
        paper1 = Paper(paper_id="same_id", title="Paper 1", abstract="Abstract 1")
        paper2 = Paper(paper_id="same_id", title="Paper 2", abstract="Abstract 2")
        # Both have same ID
        assert paper1.paper_id == paper2.paper_id


class TestPaperStoreBasic:
    """Basic PaperStore functionality tests."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_store_initialization(self, temp_db):
        """Test that store initializes correctly."""
        assert temp_db is not None
        assert temp_db.get_total_count() == 0
    
    def test_add_single_paper(self, temp_db):
        """Test adding a single paper."""
        paper = Paper(
            paper_id="add_001",
            title="Single Paper Test",
            abstract="Testing single paper addition",
            year=2024
        )
        result = temp_db.add_paper(paper)
        assert result is True
        assert temp_db.get_total_count() == 1
    
    def test_add_duplicate_paper(self, temp_db):
        """Test adding duplicate paper is handled."""
        paper = Paper(
            paper_id="dup_001",
            title="Original Title",
            abstract="Original abstract"
        )
        temp_db.add_paper(paper)
        
        # Try to add again
        result = temp_db.add_paper(paper)
        assert result is False  # Should not add duplicate
        assert temp_db.get_total_count() == 1
    
    def test_update_existing_paper(self, temp_db):
        """Test updating an existing paper."""
        paper1 = Paper(
            paper_id="update_001",
            title="Original Title",
            abstract="Original abstract",
            citation_count=10
        )
        temp_db.add_paper(paper1)
        
        paper2 = Paper(
            paper_id="update_001",
            title="Updated Title",
            abstract="Updated abstract",
            citation_count=20
        )
        temp_db.add_paper(paper2, update_if_exists=True)
        
        # Retrieve and verify
        retrieved = temp_db.get_paper("update_001")
        assert retrieved is not None
        assert retrieved.title == "Updated Title"
        assert retrieved.citation_count == 20
    
    def test_get_paper(self, temp_db):
        """Test retrieving a paper by ID."""
        paper = Paper(
            paper_id="get_001",
            title="Get Test",
            abstract="Testing retrieval"
        )
        temp_db.add_paper(paper)
        
        retrieved = temp_db.get_paper("get_001")
        assert retrieved is not None
        assert retrieved.paper_id == "get_001"
        assert retrieved.title == "Get Test"
    
    def test_get_nonexistent_paper(self, temp_db):
        """Test retrieving a non-existent paper."""
        retrieved = temp_db.get_paper("nonexistent_id")
        assert retrieved is None
    
    def test_delete_paper(self, temp_db):
        """Test deleting a paper."""
        paper = Paper(
            paper_id="del_001",
            title="Delete Test",
            abstract="Will be deleted"
        )
        temp_db.add_paper(paper)
        assert temp_db.get_total_count() == 1
        
        result = temp_db.delete_paper("del_001")
        assert result is True
        assert temp_db.get_total_count() == 0
    
    def test_delete_nonexistent_paper(self, temp_db):
        """Test deleting a non-existent paper."""
        result = temp_db.delete_paper("nonexistent_id")
        assert result is False


class TestPaperStoreBatch:
    """Test batch operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_batch_insert_small(self, temp_db):
        """Test batch insertion of small set."""
        papers = [
            Paper(paper_id=f"batch_{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
            for i in range(10)
        ]
        added, updated, skipped = temp_db.add_papers_batch(papers)
        assert added == 10
        assert updated == 0
        assert skipped == 0
        assert temp_db.get_total_count() == 10
    
    def test_batch_insert_with_duplicates(self, temp_db):
        """Test batch insertion with internal duplicates."""
        papers = [
            Paper(paper_id="dup_001", title="Paper 1", abstract="Abstract 1"),
            Paper(paper_id="dup_002", title="Paper 2", abstract="Abstract 2"),
            Paper(paper_id="dup_001", title="Paper 1 Dup", abstract="Abstract 1 Dup"),
        ]
        added, updated, skipped = temp_db.add_papers_batch(papers)
        assert added == 2  # Only 2 unique papers
        assert skipped >= 0
    
    def test_batch_insert_large(self, temp_db):
        """Test batch insertion of larger set."""
        papers = [
            Paper(
                paper_id=f"large_{i:06d}",
                title=f"Large Batch Paper {i}",
                abstract=f"Testing large batch insertion paper number {i}",
                year=2020 + (i % 5)
            )
            for i in range(1000)
        ]
        added, updated, skipped = temp_db.add_papers_batch(papers)
        assert added == 1000
        assert temp_db.get_total_count() == 1000
    
    def test_batch_insert_with_existing(self, temp_db):
        """Test batch insertion with some existing papers."""
        # Add some papers first
        papers1 = [
            Paper(paper_id=f"exist_{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
            for i in range(5)
        ]
        temp_db.add_papers_batch(papers1)
        
        # Add more, some overlapping
        papers2 = [
            Paper(paper_id=f"exist_{i}", title=f"Paper {i} New", abstract=f"Abstract {i}")
            for i in range(3, 8)  # 3,4 overlap, 5,6,7 new
        ]
        added, updated, skipped = temp_db.add_papers_batch(papers2, update_if_exists=False)
        assert added == 3  # Only 5,6,7 added
        assert skipped == 2  # 3,4 skipped
        assert temp_db.get_total_count() == 8


class TestPaperStoreSearch:
    """Test full-text search functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with test data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        
        # Add EEG-related papers
        papers = [
            Paper(
                paper_id="eeg_001",
                title="EEG-based Brain-Computer Interface for Motor Imagery",
                abstract="This study presents a novel brain-computer interface using EEG signals for motor imagery classification. We achieved 85% accuracy using deep learning.",
                keywords=["EEG", "BCI", "motor imagery", "deep learning"],
                year=2023
            ),
            Paper(
                paper_id="eeg_002",
                title="P300 Event-Related Potential in Attention Studies",
                abstract="We investigated the P300 component during attention tasks using high-density EEG. Alpha band activity was significantly modulated.",
                keywords=["P300", "ERP", "attention", "alpha"],
                year=2022
            ),
            Paper(
                paper_id="eeg_003",
                title="Epilepsy Detection Using Scalp Electroencephalography",
                abstract="Machine learning approach for automated epilepsy detection from scalp EEG recordings. Seizure onset detection achieved 92% sensitivity.",
                keywords=["epilepsy", "seizure", "detection", "machine learning"],
                year=2024
            ),
            Paper(
                paper_id="sleep_001",
                title="Sleep Stage Classification with Polysomnography",
                abstract="Deep neural network for automatic sleep staging using polysomnography data including EEG, EMG, and EOG channels.",
                keywords=["sleep", "polysomnography", "classification"],
                year=2023
            ),
            Paper(
                paper_id="neuro_001",
                title="Theta Oscillations During Working Memory Tasks",
                abstract="Frontal theta oscillations measured with EEG correlate with working memory load. Gamma band activity was also observed in parietal regions.",
                keywords=["theta", "working memory", "oscillations"],
                year=2021
            ),
        ]
        store.add_papers_batch(papers)
        
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_search_simple_term(self, temp_db):
        """Test simple term search."""
        results = temp_db.search_papers("EEG")
        assert len(results) >= 3  # Multiple papers mention EEG
    
    def test_search_phrase(self, temp_db):
        """Test phrase search."""
        results = temp_db.search_papers("brain-computer interface")
        assert len(results) >= 1
        assert any("motor imagery" in r.title.lower() for r in results)
    
    def test_search_technical_term(self, temp_db):
        """Test search for technical EEG terms."""
        results = temp_db.search_papers("P300")
        assert len(results) >= 1
        assert any("P300" in r.title for r in results)
    
    def test_search_frequency_band(self, temp_db):
        """Test search for frequency bands."""
        results = temp_db.search_papers("theta oscillations")
        assert len(results) >= 1
    
    def test_search_clinical_term(self, temp_db):
        """Test search for clinical terms."""
        results = temp_db.search_papers("epilepsy seizure")
        assert len(results) >= 1
    
    def test_search_limit(self, temp_db):
        """Test search with limit."""
        results = temp_db.search_papers("EEG", limit=2)
        assert len(results) <= 2
    
    def test_search_no_results(self, temp_db):
        """Test search with no results."""
        results = temp_db.search_papers("quantum computing blockchain")
        assert len(results) == 0
    
    def test_search_empty_query(self, temp_db):
        """Test search with empty query."""
        results = temp_db.search_papers("")
        assert isinstance(results, list)
    
    def test_search_special_characters(self, temp_db):
        """Test search with special characters."""
        results = temp_db.search_papers("EEG-based")
        assert isinstance(results, list)


class TestPaperStoreStatistics:
    """Test statistics and analytics."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with varied data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        
        papers = [
            Paper(
                paper_id=f"stat_{i:03d}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                year=2020 + (i % 5),
                source="pubmed" if i % 2 == 0 else "openalex",
                pmid=f"1234{i:04d}" if i % 3 == 0 else None,
                doi=f"10.1234/{i:04d}" if i % 2 == 0 else None
            )
            for i in range(100)
        ]
        store.add_papers_batch(papers)
        
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_get_statistics(self, temp_db):
        """Test getting statistics."""
        stats = temp_db.get_statistics()
        assert stats["total_papers"] == 100
        assert "by_source" in stats
        assert "pubmed" in stats["by_source"]
        assert "openalex" in stats["by_source"]
    
    def test_year_range(self, temp_db):
        """Test year range in statistics."""
        stats = temp_db.get_statistics()
        year_range = stats.get("year_range", {})
        assert year_range.get("min") == 2020
        assert year_range.get("max") == 2024
    
    def test_pmid_coverage(self, temp_db):
        """Test PMID coverage calculation."""
        stats = temp_db.get_statistics()
        pmid_coverage = stats.get("pmid_coverage", 0)
        # About 1/3 have PMIDs
        assert 30 <= pmid_coverage <= 40
    
    def test_doi_coverage(self, temp_db):
        """Test DOI coverage calculation."""
        stats = temp_db.get_statistics()
        doi_coverage = stats.get("doi_coverage", 0)
        # About 1/2 have DOIs
        assert 45 <= doi_coverage <= 55
    
    def test_database_size(self, temp_db):
        """Test database size is reported."""
        stats = temp_db.get_statistics()
        db_size = stats.get("db_size_mb", 0)
        assert db_size > 0


class TestPaperStoreEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_very_long_title(self, temp_db):
        """Test paper with very long title."""
        long_title = "A" * 10000
        paper = Paper(
            paper_id="long_title",
            title=long_title,
            abstract="Normal abstract"
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("long_title")
        assert retrieved.title == long_title
    
    def test_very_long_abstract(self, temp_db):
        """Test paper with very long abstract."""
        long_abstract = "B" * 100000
        paper = Paper(
            paper_id="long_abstract",
            title="Normal title",
            abstract=long_abstract
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("long_abstract")
        assert retrieved.abstract == long_abstract
    
    def test_unicode_content(self, temp_db):
        """Test paper with unicode content."""
        paper = Paper(
            paper_id="unicode_001",
            title="Étude sur l'électroencéphalographie 脑电图研究 🧠",
            abstract="Testing unicode: αβγδ θ波 μ律 日本語テスト",
            authors=["José García", "田中太郎", "Müller"]
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("unicode_001")
        assert "électroencéphalographie" in retrieved.title
        assert "αβγδ" in retrieved.abstract
    
    def test_empty_abstract(self, temp_db):
        """Test paper with empty abstract."""
        paper = Paper(
            paper_id="empty_abstract",
            title="Paper with no abstract",
            abstract=""
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("empty_abstract")
        assert retrieved.abstract == ""
    
    def test_many_authors(self, temp_db):
        """Test paper with many authors."""
        authors = [f"Author {i}" for i in range(100)]
        paper = Paper(
            paper_id="many_authors",
            title="Collaborative Study",
            abstract="Many authors test",
            authors=authors
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("many_authors")
        assert len(retrieved.authors) == 100
    
    def test_special_chars_in_id(self, temp_db):
        """Test paper with special characters in ID."""
        paper = Paper(
            paper_id="doi:10.1234/special_id",
            title="Special ID Paper",
            abstract="Testing special chars"
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("doi:10.1234/special_id")
        assert retrieved is not None
    
    def test_null_year(self, temp_db):
        """Test paper with null year."""
        paper = Paper(
            paper_id="null_year",
            title="Unknown Year Paper",
            abstract="No year specified",
            year=None
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("null_year")
        assert retrieved.year is None
    
    def test_future_year(self, temp_db):
        """Test paper with future year (preprints)."""
        paper = Paper(
            paper_id="future_year",
            title="Future Paper",
            abstract="Preprint for next year",
            year=2030
        )
        temp_db.add_paper(paper)
        retrieved = temp_db.get_paper("future_year")
        assert retrieved.year == 2030


class TestPaperStoreLookups:
    """Test lookup by various identifiers."""
    
    @pytest.fixture
    def temp_db(self):
        """Create database with papers having various IDs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        
        papers = [
            Paper(
                paper_id="lookup_001",
                title="Paper with PMID",
                abstract="Has PMID",
                pmid="12345678"
            ),
            Paper(
                paper_id="lookup_002",
                title="Paper with DOI",
                abstract="Has DOI",
                doi="10.1234/test.001"
            ),
            Paper(
                paper_id="lookup_003",
                title="Paper with arXiv",
                abstract="Has arXiv",
                arxiv_id="2401.12345"
            ),
            Paper(
                paper_id="lookup_004",
                title="Paper with OpenAlex",
                abstract="Has OpenAlex",
                openalex_id="W1234567890"
            ),
        ]
        store.add_papers_batch(papers)
        
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_get_by_pmid(self, temp_db):
        """Test lookup by PMID."""
        paper = temp_db.get_paper_by_pmid("12345678")
        assert paper is not None
        assert paper.paper_id == "lookup_001"
    
    def test_get_by_doi(self, temp_db):
        """Test lookup by DOI."""
        paper = temp_db.get_paper_by_doi("10.1234/test.001")
        assert paper is not None
        assert paper.paper_id == "lookup_002"
    
    def test_get_by_nonexistent_pmid(self, temp_db):
        """Test lookup by non-existent PMID."""
        paper = temp_db.get_paper_by_pmid("99999999")
        assert paper is None
    
    def test_get_by_nonexistent_doi(self, temp_db):
        """Test lookup by non-existent DOI."""
        paper = temp_db.get_paper_by_doi("10.9999/none")
        assert paper is None


class TestPaperStorePerformance:
    """Performance tests for PaperStore."""
    
    @pytest.fixture
    def large_db(self):
        """Create a larger database for performance testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        
        # Add 10,000 papers
        papers = [
            Paper(
                paper_id=f"perf_{i:06d}",
                title=f"Performance Test Paper {i} on EEG and Brain-Computer Interface",
                abstract=f"This paper {i} discusses electroencephalography and neural signal processing for brain-computer interface applications.",
                year=2010 + (i % 15),
                source="openalex",
                keywords=["EEG", "BCI", "neural", "signal"]
            )
            for i in range(10000)
        ]
        store.add_papers_batch(papers)
        
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_search_performance(self, large_db):
        """Test search performance on 10K papers."""
        start = time.time()
        results = large_db.search_papers("electroencephalography brain-computer")
        elapsed = time.time() - start
        
        assert len(results) > 0
        assert elapsed < 1.0  # Should complete in under 1 second
    
    def test_batch_insert_performance(self, large_db):
        """Test batch insert performance."""
        papers = [
            Paper(
                paper_id=f"new_perf_{i:04d}",
                title=f"New Performance Paper {i}",
                abstract=f"Testing batch insert performance {i}",
                year=2024
            )
            for i in range(1000)
        ]
        
        start = time.time()
        added, _, _ = large_db.add_papers_batch(papers)
        elapsed = time.time() - start
        
        assert added == 1000
        assert elapsed < 5.0  # Should complete in under 5 seconds
    
    def test_statistics_performance(self, large_db):
        """Test statistics calculation performance."""
        start = time.time()
        stats = large_db.get_statistics()
        elapsed = time.time() - start
        
        assert stats["total_papers"] >= 10000
        assert elapsed < 1.0


class TestPaperStoreConcurrency:
    """Test concurrent access to PaperStore."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store, db_path
        store.close()
        os.unlink(db_path)
    
    def test_concurrent_reads(self, temp_db):
        """Test concurrent read operations."""
        store, db_path = temp_db
        
        # Add some papers first
        papers = [
            Paper(paper_id=f"conc_r_{i}", title=f"Paper {i}", abstract=f"Abstract {i}")
            for i in range(100)
        ]
        store.add_papers_batch(papers)
        
        results = []
        errors = []
        
        def read_task(paper_id):
            try:
                result = store.get_paper(paper_id)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(read_task, f"conc_r_{i}")
                for i in range(100)
            ]
            for f in futures:
                f.result()
        
        assert len(errors) == 0
        assert len(results) == 100


class TestPaperStoreGetPapers:
    """Test get_papers pagination."""
    
    @pytest.fixture
    def temp_db(self):
        """Create database with papers for pagination."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        
        papers = [
            Paper(
                paper_id=f"page_{i:04d}",
                title=f"Page Test Paper {i}",
                abstract=f"Pagination test {i}",
                year=2020 + (i % 5)
            )
            for i in range(50)
        ]
        store.add_papers_batch(papers)
        
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_get_papers_default(self, temp_db):
        """Test get_papers with default parameters."""
        papers = temp_db.get_papers()
        assert len(papers) <= 100
    
    def test_get_papers_with_limit(self, temp_db):
        """Test get_papers with limit."""
        papers = temp_db.get_papers(limit=10)
        assert len(papers) == 10
    
    def test_get_papers_with_offset(self, temp_db):
        """Test get_papers with offset."""
        papers1 = temp_db.get_papers(limit=10, offset=0)
        papers2 = temp_db.get_papers(limit=10, offset=10)
        
        # Should be different papers
        ids1 = {p.paper_id for p in papers1}
        ids2 = {p.paper_id for p in papers2}
        assert len(ids1.intersection(ids2)) == 0
    
    def test_get_papers_pagination(self, temp_db):
        """Test full pagination through all papers."""
        all_papers = []
        offset = 0
        limit = 10
        
        while True:
            batch = temp_db.get_papers(limit=limit, offset=offset)
            if not batch:
                break
            all_papers.extend(batch)
            offset += limit
        
        assert len(all_papers) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
