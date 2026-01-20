"""
Comprehensive tests for EEGQueryEngine class.

Tests the query processing, response generation,
source retrieval, and citation handling functionality.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch, AsyncMock


class TestEEGQueryEngineInit:
    """Tests for EEGQueryEngine initialization."""
    
    def test_engine_init(self, query_engine):
        """Test query engine initialization."""
        assert query_engine is not None
        
    def test_knowledge_base_defined(self, query_engine):
        """Test that knowledge base is defined."""
        assert hasattr(query_engine, 'KNOWLEDGE_BASE')
        assert "seizure" in query_engine.KNOWLEDGE_BASE
        assert "sleep" in query_engine.KNOWLEDGE_BASE
        assert "bci" in query_engine.KNOWLEDGE_BASE
        assert "default" in query_engine.KNOWLEDGE_BASE
        
    def test_knowledge_base_structure(self, query_engine):
        """Test knowledge base entry structure."""
        for key, entry in query_engine.KNOWLEDGE_BASE.items():
            assert "response" in entry
            assert "sources" in entry
            assert isinstance(entry["response"], str)
            assert isinstance(entry["sources"], list)


class TestQueryProcessing:
    """Tests for query processing functionality."""
    
    @pytest.mark.asyncio
    async def test_query_returns_result(self, query_engine):
        """Test that query returns a QueryResult."""
        from eeg_rag.web_ui.app import QueryResult
        
        result = await query_engine.query("What are CNNs for EEG?", max_sources=3)
        
        assert isinstance(result, QueryResult)
        
    @pytest.mark.asyncio
    async def test_query_contains_response(self, query_engine):
        """Test that query result contains response."""
        result = await query_engine.query("Test query")
        
        assert result.response is not None
        assert len(result.response) > 0
        
    @pytest.mark.asyncio
    async def test_query_contains_sources(self, query_engine):
        """Test that query result contains sources."""
        result = await query_engine.query("Test query", max_sources=5)
        
        assert result.sources is not None
        assert isinstance(result.sources, list)
        
    @pytest.mark.asyncio
    async def test_query_contains_citations(self, query_engine):
        """Test that query result contains citations."""
        result = await query_engine.query("Test query")
        
        assert result.citations is not None
        assert isinstance(result.citations, list)
        for citation in result.citations:
            assert citation.startswith("PMID:")
            
    @pytest.mark.asyncio
    async def test_query_contains_timestamp(self, query_engine):
        """Test that query result contains timestamp."""
        result = await query_engine.query("Test query")
        
        assert result.timestamp is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(result.timestamp)
        
    @pytest.mark.asyncio
    async def test_query_processing_time(self, query_engine):
        """Test that processing time is tracked."""
        result = await query_engine.query("Test query")
        
        assert result.processing_time_ms > 0
        
    @pytest.mark.asyncio
    async def test_query_confidence(self, query_engine):
        """Test that confidence score is returned."""
        result = await query_engine.query("Test query")
        
        assert 0 <= result.confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_query_id_generated(self, query_engine):
        """Test that unique query ID is generated."""
        result = await query_engine.query("Test query")
        
        assert result.query_id is not None
        assert len(result.query_id) > 0


class TestDomainRouting:
    """Tests for query routing to appropriate knowledge domains."""
    
    @pytest.mark.asyncio
    async def test_seizure_query_routing(self, query_engine):
        """Test routing of seizure-related queries."""
        seizure_queries = [
            "seizure detection methods",
            "epilepsy classification",
            "ictal patterns in EEG"
        ]
        
        for query in seizure_queries:
            result = await query_engine.query(query)
            # Should route to seizure knowledge base
            assert "seizure" in result.response.lower() or "cnn" in result.response.lower()
            
    @pytest.mark.asyncio
    async def test_sleep_query_routing(self, query_engine):
        """Test routing of sleep-related queries."""
        sleep_queries = [
            "sleep stage classification",
            "automatic sleep staging",
            "polysomnography analysis"
        ]
        
        for query in sleep_queries:
            result = await query_engine.query(query)
            assert "sleep" in result.response.lower()
            
    @pytest.mark.asyncio
    async def test_bci_query_routing(self, query_engine):
        """Test routing of BCI-related queries."""
        bci_queries = [
            "BCI motor imagery",
            "brain-computer interface design",
            "P300 speller system"
        ]
        
        for query in bci_queries:
            result = await query_engine.query(query)
            assert "bci" in result.response.lower() or "motor" in result.response.lower()
            
    @pytest.mark.asyncio
    async def test_default_routing(self, query_engine):
        """Test routing of generic queries to default response."""
        generic_queries = [
            "general EEG analysis",
            "deep learning overview",
            "neural network applications"
        ]
        
        for query in generic_queries:
            result = await query_engine.query(query)
            assert result.response is not None
            assert len(result.response) > 100


class TestSourceHandling:
    """Tests for source retrieval and formatting."""
    
    @pytest.mark.asyncio
    async def test_max_sources_respected(self, query_engine):
        """Test that max_sources parameter is respected."""
        result = await query_engine.query("Test query", max_sources=2)
        
        assert len(result.sources) <= 2
        
    @pytest.mark.asyncio
    async def test_source_structure(self, query_engine):
        """Test source entry structure."""
        result = await query_engine.query("Test query", max_sources=5)
        
        for source in result.sources:
            assert "title" in source
            assert "pmid" in source
            assert "year" in source
            assert "relevance" in source
            
    @pytest.mark.asyncio
    async def test_source_relevance_ordering(self, query_engine):
        """Test that sources are ordered by relevance."""
        result = await query_engine.query("Test query", max_sources=5)
        
        if len(result.sources) > 1:
            for i in range(len(result.sources) - 1):
                assert result.sources[i]["relevance"] >= result.sources[i+1]["relevance"]
                
    @pytest.mark.asyncio
    async def test_pmid_format(self, query_engine):
        """Test PMID format in sources."""
        result = await query_engine.query("Test query")
        
        for source in result.sources:
            pmid = source["pmid"]
            assert pmid.isdigit()
            assert 7 <= len(pmid) <= 8
            
    @pytest.mark.asyncio
    async def test_year_validity(self, query_engine):
        """Test year validity in sources."""
        result = await query_engine.query("Test query")
        
        for source in result.sources:
            year = source["year"]
            assert 1990 <= year <= 2030


class TestQueryResult:
    """Tests for QueryResult dataclass."""
    
    def test_query_result_creation(self, sample_query_result_data):
        """Test QueryResult creation."""
        from eeg_rag.web_ui.app import QueryResult
        
        result = QueryResult(**sample_query_result_data)
        
        assert result.query == sample_query_result_data["query"]
        assert result.response == sample_query_result_data["response"]
        assert result.confidence == sample_query_result_data["confidence"]
        
    def test_query_result_auto_id(self, sample_query_result_data):
        """Test automatic query ID generation."""
        from eeg_rag.web_ui.app import QueryResult
        
        data = sample_query_result_data.copy()
        data.pop("query_id", None)  # Remove if present
        
        result = QueryResult(**data)
        
        assert result.query_id is not None
        assert len(result.query_id) == 12  # MD5 hash truncated to 12 chars
        
    def test_query_result_unique_ids(self, sample_query_result_data):
        """Test that different queries get unique IDs."""
        from eeg_rag.web_ui.app import QueryResult
        
        data1 = sample_query_result_data.copy()
        data1["query"] = "Query 1"
        data1["timestamp"] = datetime.now().isoformat()
        
        data2 = sample_query_result_data.copy()
        data2["query"] = "Query 2"
        data2["timestamp"] = datetime.now().isoformat()
        
        result1 = QueryResult(**data1)
        result2 = QueryResult(**data2)
        
        assert result1.query_id != result2.query_id


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_query(self, query_engine):
        """Test handling of empty query."""
        result = await query_engine.query("")
        
        # Should still return a result (default response)
        assert result is not None
        assert result.response is not None
        
    @pytest.mark.asyncio
    async def test_very_long_query(self, query_engine):
        """Test handling of very long query."""
        long_query = "What is the best architecture for EEG? " * 100
        
        result = await query_engine.query(long_query)
        
        assert result is not None
        assert result.response is not None
        
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, query_engine):
        """Test handling of special characters."""
        special_query = "What about CNN™? <script>alert('test')</script>"
        
        result = await query_engine.query(special_query)
        
        assert result is not None
        
    @pytest.mark.asyncio
    async def test_unicode_query(self, query_engine):
        """Test handling of unicode in query."""
        unicode_query = "Qué arquitectura para EEG? 什么是最好的?"
        
        result = await query_engine.query(unicode_query)
        
        assert result is not None
        
    @pytest.mark.asyncio
    async def test_max_sources_zero(self, query_engine):
        """Test with max_sources=0."""
        result = await query_engine.query("Test", max_sources=0)
        
        assert len(result.sources) == 0
        
    @pytest.mark.asyncio
    async def test_max_sources_large(self, query_engine):
        """Test with very large max_sources."""
        result = await query_engine.query("Test", max_sources=100)
        
        # Should still work but limited to available sources
        assert result is not None
        assert len(result.sources) <= 100


class TestCitationHandling:
    """Tests for citation handling."""
    
    @pytest.mark.asyncio
    async def test_citation_format(self, query_engine):
        """Test citation format."""
        result = await query_engine.query("Test query")
        
        for citation in result.citations:
            assert citation.startswith("PMID:")
            pmid = citation.replace("PMID:", "")
            assert pmid.isdigit()
            
    @pytest.mark.asyncio
    async def test_citations_match_sources(self, query_engine):
        """Test that citations match sources."""
        result = await query_engine.query("Test query", max_sources=3)
        
        source_pmids = {source["pmid"] for source in result.sources}
        citation_pmids = {c.replace("PMID:", "") for c in result.citations}
        
        # All source PMIDs should be in citations
        for pmid in source_pmids:
            assert pmid in citation_pmids


class TestConcurrency:
    """Tests for concurrent query handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, query_engine):
        """Test handling of concurrent queries."""
        queries = [
            "seizure detection",
            "sleep staging",
            "BCI motor imagery",
            "emotion recognition"
        ]
        
        tasks = [query_engine.query(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            assert result.response is not None
            
    @pytest.mark.asyncio
    async def test_query_independence(self, query_engine):
        """Test that queries are independent."""
        result1 = await query_engine.query("seizure")
        result2 = await query_engine.query("sleep")
        
        # Results should be different
        assert result1.response != result2.response
        assert result1.query_id != result2.query_id
