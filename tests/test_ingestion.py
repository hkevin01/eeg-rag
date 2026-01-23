#!/usr/bin/env python3
"""
Tests for the ingestion system including OpenAlex client.

Tests cover:
- OpenAlex client functionality
- Paper ingestion pipeline
- Ingestion statistics
- Error handling and rate limiting
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, date

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.db.paper_store import Paper, PaperStore


class TestOpenAlexClient:
    """Test OpenAlex client functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAlex client."""
        # We'll mock the client to avoid actual API calls
        with patch('eeg_rag.ingestion.openalex_client.aiohttp.ClientSession') as mock_session:
            yield mock_session
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initializes with proper headers."""
        from eeg_rag.ingestion.openalex_client import OpenAlexClient
        
        client = OpenAlexClient(email="test@example.com")
        assert client.email == "test@example.com"
        assert "OpenAlex" in client.base_url or "api.openalex.org" in client.base_url
    
    @pytest.mark.asyncio
    async def test_work_dataclass(self):
        """Test Work dataclass creation."""
        from eeg_rag.ingestion.openalex_client import Work
        
        work = Work(
            openalex_id="W123456",
            title="Test EEG Paper",
            abstract="Testing EEG signals",
            publication_date=date(2024, 1, 15)
        )
        assert work.openalex_id == "W123456"
        assert work.title == "Test EEG Paper"
        assert work.publication_date.year == 2024


class TestPaperIngestion:
    """Test paper ingestion from various sources."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary paper store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_ingest_single_paper(self, temp_store):
        """Test ingesting a single paper."""
        paper = Paper(
            paper_id="ingest_001",
            title="EEG-Based Brain-Computer Interface",
            abstract="Novel BCI using EEG signals for motor imagery classification",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            source="openalex",
            doi="10.1234/test.001",
            keywords=["EEG", "BCI", "motor imagery"]
        )
        
        result = temp_store.add_paper(paper)
        assert result is True
        
        retrieved = temp_store.get_paper("ingest_001")
        assert retrieved is not None
        assert retrieved.title == paper.title
        assert retrieved.authors == paper.authors
    
    def test_ingest_batch_papers(self, temp_store):
        """Test batch ingestion of papers."""
        papers = []
        for i in range(100):
            papers.append(Paper(
                paper_id=f"batch_ingest_{i:03d}",
                title=f"EEG Study {i}: Neural Oscillations",
                abstract=f"Study {i} investigating theta and alpha oscillations in EEG recordings",
                authors=["Researcher A", "Researcher B"],
                year=2020 + (i % 5),
                source="openalex",
                keywords=["EEG", "oscillations", "theta", "alpha"]
            ))
        
        added, updated, skipped = temp_store.add_papers_batch(papers)
        assert added == 100
        assert temp_store.get_total_count() == 100
    
    def test_ingest_with_deduplication(self, temp_store):
        """Test that duplicate papers are handled correctly."""
        paper1 = Paper(
            paper_id="dup_test_001",
            title="P300 Event-Related Potential Study",
            abstract="Investigation of P300 component in attention tasks",
            year=2023,
            source="openalex"
        )
        temp_store.add_paper(paper1)
        
        # Try to add same paper with different ID but same content
        paper2 = Paper(
            paper_id="dup_test_002",
            title="P300 Event-Related Potential Study",
            abstract="Investigation of P300 component in attention tasks",
            year=2023,
            source="pubmed"
        )
        result = temp_store.add_paper(paper2)
        assert result is False  # Should be detected as duplicate
        assert temp_store.get_total_count() == 1
    
    def test_ingest_with_pmid_deduplication(self, temp_store):
        """Test deduplication by PMID."""
        paper1 = Paper(
            paper_id="pmid_test_001",
            title="Original Paper",
            abstract="Original abstract",
            pmid="12345678",
            source="pubmed"
        )
        temp_store.add_paper(paper1)
        
        # Different paper_id but same PMID
        paper2 = Paper(
            paper_id="pmid_test_002",
            title="Same Paper Different Source",
            abstract="Slightly different abstract",
            pmid="12345678",
            source="openalex"
        )
        result = temp_store.add_paper(paper2)
        assert result is False
        assert temp_store.get_total_count() == 1
    
    def test_ingest_eeg_specific_content(self, temp_store):
        """Test ingestion of EEG-specific content."""
        eeg_papers = [
            Paper(
                paper_id="eeg_freq_001",
                title="Delta Oscillations During Sleep",
                abstract="Study of 0.5-4Hz delta waves during deep sleep stages",
                keywords=["delta", "sleep", "EEG"],
                year=2023
            ),
            Paper(
                paper_id="eeg_freq_002",
                title="Theta Activity in Working Memory",
                abstract="4-8Hz theta oscillations correlate with memory load",
                keywords=["theta", "working memory", "EEG"],
                year=2023
            ),
            Paper(
                paper_id="eeg_freq_003",
                title="Alpha Rhythm and Visual Attention",
                abstract="8-13Hz alpha suppression during visual attention tasks",
                keywords=["alpha", "attention", "EEG"],
                year=2023
            ),
            Paper(
                paper_id="eeg_freq_004",
                title="Beta Oscillations in Motor Control",
                abstract="13-30Hz beta activity in motor cortex during movement",
                keywords=["beta", "motor", "EEG"],
                year=2024
            ),
            Paper(
                paper_id="eeg_freq_005",
                title="Gamma Band Activity in Cognition",
                abstract="30-100Hz gamma oscillations in cognitive processing",
                keywords=["gamma", "cognition", "EEG"],
                year=2024
            ),
        ]
        
        added, _, _ = temp_store.add_papers_batch(eeg_papers)
        assert added == 5
        
        # Search for frequency band terms
        delta_results = temp_store.search_papers("delta oscillations sleep")
        assert len(delta_results) >= 1
        
        gamma_results = temp_store.search_papers("gamma cognition")
        assert len(gamma_results) >= 1
    
    def test_ingest_clinical_content(self, temp_store):
        """Test ingestion of clinical EEG content."""
        clinical_papers = [
            Paper(
                paper_id="clin_001",
                title="Epilepsy Detection Using Machine Learning",
                abstract="Automated seizure detection from scalp EEG",
                keywords=["epilepsy", "seizure", "detection"],
                year=2024
            ),
            Paper(
                paper_id="clin_002",
                title="BCI for ALS Patients",
                abstract="Brain-computer interface using P300 speller",
                keywords=["BCI", "ALS", "P300"],
                year=2024
            ),
        ]
        
        added, _, _ = temp_store.add_papers_batch(clinical_papers)
        assert added == 2
        
        # Search for clinical terms
        epilepsy_results = temp_store.search_papers("epilepsy seizure detection")
        assert len(epilepsy_results) >= 1
    
    def test_ingest_international_content(self, temp_store):
        """Test ingestion with international characters."""
        papers = [
            Paper(
                paper_id="intl_001",
                title="Étude de l'activité cérébrale par EEG",
                abstract="Analyse des oscillations cérébrales θ et α",
                authors=["François Müller", "田中太郎"],
                year=2024
            ),
            Paper(
                paper_id="intl_002",
                title="脑电图研究：注意力与记忆",
                abstract="研究EEG在注意力和工作记忆任务中的应用",
                authors=["张伟", "李娜"],
                year=2024
            ),
        ]
        
        added, _, _ = temp_store.add_papers_batch(papers)
        assert added == 2
        
        # Verify content is preserved
        paper = temp_store.get_paper("intl_001")
        assert "François" in paper.authors[0]


class TestIngestionStatistics:
    """Test ingestion statistics tracking."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary paper store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_statistics_after_ingestion(self, temp_store):
        """Test that statistics are updated after ingestion."""
        papers = [
            Paper(
                paper_id=f"stat_{i:03d}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                year=2020 + (i % 5),
                source="openalex" if i % 2 == 0 else "pubmed",
                pmid=f"1234{i:04d}" if i % 2 == 1 else None,
                doi=f"10.1234/{i:04d}" if i % 2 == 0 else None
            )
            for i in range(50)
        ]
        
        temp_store.add_papers_batch(papers)
        stats = temp_store.get_statistics()
        
        assert stats['total_papers'] == 50
        assert 'openalex' in stats['by_source']
        assert 'pubmed' in stats['by_source']
        assert stats['by_source']['openalex'] == 25
        assert stats['by_source']['pubmed'] == 25
    
    def test_year_range_statistics(self, temp_store):
        """Test year range in statistics."""
        papers = [
            Paper(
                paper_id=f"year_{i:03d}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                year=2015 + i
            )
            for i in range(10)
        ]
        
        temp_store.add_papers_batch(papers)
        stats = temp_store.get_statistics()
        
        assert stats['year_range']['min'] == 2015
        assert stats['year_range']['max'] == 2024
    
    def test_coverage_statistics(self, temp_store):
        """Test PMID and DOI coverage statistics."""
        papers = [
            Paper(
                paper_id=f"cov_{i:03d}",
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                pmid=f"1234{i:04d}" if i < 30 else None,
                doi=f"10.1234/{i:04d}" if i < 40 else None
            )
            for i in range(100)
        ]
        
        temp_store.add_papers_batch(papers)
        stats = temp_store.get_statistics()
        
        assert stats['pmid_coverage'] == 30.0
        assert stats['doi_coverage'] == 40.0


class TestIngestionErrorHandling:
    """Test error handling during ingestion."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary paper store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        store = PaperStore(db_path)
        yield store
        store.close()
        os.unlink(db_path)
    
    def test_ingest_missing_title(self, temp_store):
        """Test handling paper with empty title."""
        paper = Paper(
            paper_id="empty_title",
            title="",
            abstract="This paper has no title"
        )
        result = temp_store.add_paper(paper)
        # Should still add, empty title is valid
        assert result is True
    
    def test_ingest_very_long_abstract(self, temp_store):
        """Test handling paper with very long abstract."""
        long_abstract = "A" * 500000  # 500KB of text
        paper = Paper(
            paper_id="long_abstract",
            title="Paper with long abstract",
            abstract=long_abstract
        )
        result = temp_store.add_paper(paper)
        assert result is True
        
        retrieved = temp_store.get_paper("long_abstract")
        assert len(retrieved.abstract) == 500000


class TestIngestionRateLimiting:
    """Test rate limiting and polite access."""
    
    @pytest.mark.asyncio
    async def test_client_respects_rate_limits(self):
        """Test that client has rate limiting configured."""
        from eeg_rag.ingestion.openalex_client import OpenAlexClient
        
        client = OpenAlexClient(email="test@example.com")
        # Client should have reasonable delays between requests
        assert hasattr(client, 'delay') or hasattr(client, 'rate_limit_delay')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
