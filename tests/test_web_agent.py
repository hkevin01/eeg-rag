"""
Test suite for Web Search Agent (Agent 2).

Tests cover:
- PubMedArticle creation and serialization
- SearchResult creation and serialization
- RateLimiter functionality
- WebSearchAgent initialization
- ESearch execution (mocked)
- EFetch execution (mocked)
- XML parsing
- Caching behavior
- Rate limiting
- Error handling
- Statistics tracking
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
import xml.etree.ElementTree as ET

from eeg_rag.agents.web_agent.web_search_agent import (
    WebSearchAgent,
    PubMedArticle,
    SearchResult,
    RateLimiter
)


# ============================================================================
# Test PubMedArticle
# ============================================================================

def test_pubmed_article_creation():
    """Test creating a PubMedArticle instance."""
    article = PubMedArticle(
        pmid="12345678",
        title="Test Article",
        authors=["Smith J", "Jones A"],
        abstract="This is a test abstract.",
        journal="Test Journal",
        pub_date="2024-01-15",
        doi="10.1234/test.2024",
        mesh_terms=["EEG", "Seizures"],
        keywords=["epilepsy", "biomarkers"]
    )
    
    assert article.pmid == "12345678"
    assert article.title == "Test Article"
    assert len(article.authors) == 2
    assert article.abstract == "This is a test abstract."
    assert article.journal == "Test Journal"
    assert article.pub_date == "2024-01-15"
    assert article.doi == "10.1234/test.2024"
    assert len(article.mesh_terms) == 2
    assert len(article.keywords) == 2


def test_pubmed_article_to_dict():
    """Test converting PubMedArticle to dictionary."""
    article = PubMedArticle(
        pmid="12345678",
        title="Test Article",
        authors=["Smith J"],
        abstract="Test abstract",
        journal="Test Journal",
        pub_date="2024"
    )
    
    article_dict = article.to_dict()
    assert isinstance(article_dict, dict)
    assert article_dict["pmid"] == "12345678"
    assert article_dict["title"] == "Test Article"
    assert article_dict["authors"] == ["Smith J"]


def test_pubmed_article_optional_fields():
    """Test PubMedArticle with optional fields."""
    article = PubMedArticle(
        pmid="12345678",
        title="Test",
        authors=[],
        abstract="Test",
        journal="Test",
        pub_date="2024"
    )
    
    assert article.doi is None
    assert article.mesh_terms == []
    assert article.keywords == []


# ============================================================================
# Test SearchResult
# ============================================================================

def test_search_result_creation():
    """Test creating a SearchResult instance."""
    articles = [
        PubMedArticle(
            pmid="123",
            title="Article 1",
            authors=["Author A"],
            abstract="Abstract 1",
            journal="Journal 1",
            pub_date="2024"
        )
    ]
    
    result = SearchResult(
        query="test query",
        count=100,
        articles=articles,
        web_env="test_env",
        query_key="1"
    )
    
    assert result.query == "test query"
    assert result.count == 100
    assert len(result.articles) == 1
    assert result.web_env == "test_env"
    assert result.query_key == "1"


def test_search_result_to_dict():
    """Test converting SearchResult to dictionary."""
    articles = [
        PubMedArticle(
            pmid="123",
            title="Test",
            authors=[],
            abstract="Test",
            journal="Test",
            pub_date="2024"
        )
    ]
    
    result = SearchResult(
        query="test",
        count=1,
        articles=articles
    )
    
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["query"] == "test"
    assert result_dict["count"] == 1
    assert len(result_dict["articles"]) == 1


# ============================================================================
# Test RateLimiter
# ============================================================================

@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    limiter = RateLimiter(requests_per_second=10.0)
    
    start_time = time.time()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.time() - start_time
    
    # Should have waited at least 0.1 seconds (1/10)
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_rate_limiter_interval():
    """Test rate limiter interval calculation."""
    limiter = RateLimiter(requests_per_second=5.0)
    assert limiter.min_interval == 0.2  # 1/5 = 0.2


@pytest.mark.asyncio
async def test_rate_limiter_no_wait_first_request():
    """Test that first request doesn't wait."""
    limiter = RateLimiter(requests_per_second=10.0)
    
    start_time = time.time()
    await limiter.acquire()
    elapsed = time.time() - start_time
    
    # First request should be immediate
    assert elapsed < 0.05


# ============================================================================
# Test WebSearchAgent Initialization
# ============================================================================

def test_web_search_agent_initialization():
    """Test Web Search Agent initialization."""
    agent = WebSearchAgent(
        name="TestAgent",
        email="test@example.com",
        tool="test-tool"
    )
    
    assert agent.name == "TestAgent"
    assert agent.email == "test@example.com"
    assert agent.tool == "test-tool"
    assert agent.api_key is None


def test_web_search_agent_with_api_key():
    """Test agent initialization with API key (higher rate limit)."""
    agent = WebSearchAgent(
        api_key="test_api_key"
    )
    
    assert agent.api_key == "test_api_key"
    assert agent.rate_limiter.requests_per_second == 10.0


def test_web_search_agent_without_api_key():
    """Test agent initialization without API key (lower rate limit)."""
    agent = WebSearchAgent()
    
    assert agent.api_key is None
    assert agent.rate_limiter.requests_per_second == 3.0


# ============================================================================
# Test Query Hashing
# ============================================================================

def test_query_hash_generation():
    """Test query hash generation for caching."""
    agent = WebSearchAgent()
    
    hash1 = agent._get_query_hash("test query", max_results=20)
    hash2 = agent._get_query_hash("test query", max_results=20)
    hash3 = agent._get_query_hash("test query", max_results=30)
    
    assert hash1 == hash2  # Same parameters = same hash
    assert hash1 != hash3  # Different parameters = different hash


# ============================================================================
# Test XML Parsing
# ============================================================================

def test_parse_article_complete():
    """Test parsing complete PubMed article XML."""
    xml_string = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Test Article Title</ArticleTitle>
                <Abstract>
                    <AbstractText>This is the abstract text.</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Smith</LastName>
                        <Initials>J</Initials>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Test Journal</Title>
                </Journal>
            </Article>
            <PubDate>
                <Year>2024</Year>
                <Month>Jan</Month>
            </PubDate>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>EEG</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
        <PubmedData>
            <ArticleIdList>
                <ArticleId IdType="doi">10.1234/test</ArticleId>
            </ArticleIdList>
        </PubmedData>
    </PubmedArticle>
    """
    
    agent = WebSearchAgent()
    root = ET.fromstring(xml_string)
    article = agent._parse_article(root)
    
    assert article is not None
    assert article.pmid == "12345678"
    assert article.title == "Test Article Title"
    assert len(article.authors) == 1
    assert article.doi == "10.1234/test"


def test_parse_article_minimal():
    """Test parsing minimal PubMed article XML."""
    xml_string = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Minimal Article</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """
    
    agent = WebSearchAgent()
    root = ET.fromstring(xml_string)
    article = agent._parse_article(root)
    
    assert article is not None
    assert article.pmid == "12345678"
    assert article.title == "Minimal Article"
    assert article.abstract == "No abstract available"


def test_parse_article_missing_pmid():
    """Test parsing article without PMID returns None."""
    xml_string = """
    <PubmedArticle>
        <MedlineCitation>
            <Article>
                <ArticleTitle>No PMID Article</ArticleTitle>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """
    
    agent = WebSearchAgent()
    root = ET.fromstring(xml_string)
    article = agent._parse_article(root)
    
    assert article is None


# ============================================================================
# Test Execute Method (with mocking)
# ============================================================================

@pytest.mark.asyncio
async def test_execute_basic_search():
    """Test basic search execution."""
    agent = WebSearchAgent()
    
    # Mock the internal methods
    mock_search_result = {
        "count": 100,
        "ids": ["12345678", "23456789"],
        "web_env": "test_env",
        "query_key": "1"
    }
    
    mock_articles = [
        PubMedArticle(
            pmid="12345678",
            title="Test Article 1",
            authors=["Smith J"],
            abstract="Abstract 1",
            journal="Journal 1",
            pub_date="2024"
        )
    ]
    
    agent._search_pubmed = AsyncMock(return_value=mock_search_result)
    agent._fetch_articles = AsyncMock(return_value=mock_articles)
    
    result = await agent.execute("test query")
    
    assert result["success"] is True
    assert result["query"] == "test query"
    assert result["count"] == 100
    assert len(result["articles"]) == 1
    assert result["cached"] is False


@pytest.mark.asyncio
async def test_execute_with_cache():
    """Test that caching works correctly."""
    agent = WebSearchAgent()
    
    # Mock methods for first call
    mock_search_result = {
        "count": 10,
        "ids": ["12345678"],
        "web_env": None,
        "query_key": None
    }
    
    mock_articles = [
        PubMedArticle(
            pmid="12345678",
            title="Cached Article",
            authors=[],
            abstract="Test",
            journal="Test",
            pub_date="2024"
        )
    ]
    
    agent._search_pubmed = AsyncMock(return_value=mock_search_result)
    agent._fetch_articles = AsyncMock(return_value=mock_articles)
    
    # First call - should hit API
    result1 = await agent.execute("test query")
    assert result1["cached"] is False
    assert agent.cache_misses == 1
    
    # Second call - should hit cache
    result2 = await agent.execute("test query")
    assert result2["cached"] is True
    assert agent.cache_hits == 1
    
    # Should only call API methods once
    assert agent._search_pubmed.call_count == 1
    assert agent._fetch_articles.call_count == 1


@pytest.mark.asyncio
async def test_execute_with_context_parameters():
    """Test execute with context parameters."""
    agent = WebSearchAgent()
    
    mock_search_result = {"count": 0, "ids": [], "web_env": None, "query_key": None}
    agent._search_pubmed = AsyncMock(return_value=mock_search_result)
    agent._fetch_articles = AsyncMock(return_value=[])
    
    context = {
        "max_results": 50,
        "mindate": "2020/01/01",
        "maxdate": "2024/12/31",
        "sort": "pub_date"
    }
    
    result = await agent.execute("test query", context=context)
    
    # Verify search was called with correct parameters
    agent._search_pubmed.assert_called_once()
    call_args = agent._search_pubmed.call_args
    assert call_args[1]["retmax"] == 50
    assert call_args[1]["mindate"] == "2020/01/01"
    assert call_args[1]["maxdate"] == "2024/12/31"
    assert call_args[1]["sort"] == "pub_date"


@pytest.mark.asyncio
async def test_execute_error_handling():
    """Test error handling during execution."""
    agent = WebSearchAgent()
    
    # Mock method to raise an exception
    agent._search_pubmed = AsyncMock(side_effect=RuntimeError("API Error"))
    
    result = await agent.execute("test query")
    
    assert result["success"] is False
    assert "error" in result
    assert "API Error" in result["error"]
    assert agent.failed_executions == 1


# ============================================================================
# Test Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_statistics_tracking():
    """Test statistics tracking."""
    agent = WebSearchAgent()
    
    # Execute some searches
    mock_search_result = {"count": 10, "ids": ["123"], "web_env": None, "query_key": None}
    mock_articles = [
        PubMedArticle(
            pmid="123",
            title="Test",
            authors=[],
            abstract="Test",
            journal="Test",
            pub_date="2024"
        )
    ]
    
    agent._search_pubmed = AsyncMock(return_value=mock_search_result)
    agent._fetch_articles = AsyncMock(return_value=mock_articles)
    
    await agent.execute("query 1")
    await agent.execute("query 2")
    await agent.execute("query 1")  # Should hit cache
    
    stats = agent.get_statistics()
    
    assert stats["total_executions"] == 2  # Only non-cached executions
    assert stats["successful_executions"] == 2
    assert stats["total_searches"] == 2  # Only 2 actual searches (1 cached)
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert stats["cache_hit_rate"] == pytest.approx(1/3)


def test_clear_cache():
    """Test cache clearing functionality."""
    agent = WebSearchAgent()
    
    # Add some mock cache entries
    agent.cache["key1"] = Mock()
    agent.cache["key2"] = Mock()
    agent.cache_hits = 10
    agent.cache_misses = 5
    
    agent.clear_cache()
    
    assert len(agent.cache) == 0
    assert agent.cache_hits == 0
    assert agent.cache_misses == 0


# ============================================================================
# Test Integration
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow from search to result."""
    agent = WebSearchAgent()
    
    # Create realistic mock data
    search_result = {
        "count": 150,
        "ids": ["12345678", "23456789", "34567890"],
        "web_env": "mock_env",
        "query_key": "1"
    }
    
    articles = [
        PubMedArticle(
            pmid="12345678",
            title="EEG biomarkers in epilepsy",
            authors=["Smith J", "Jones A"],
            abstract="This study investigates EEG biomarkers...",
            journal="Epilepsia",
            pub_date="2024-01-15",
            doi="10.1234/epi.2024.001",
            mesh_terms=["Epilepsy", "Electroencephalography"],
            keywords=["seizures", "biomarkers"]
        ),
        PubMedArticle(
            pmid="23456789",
            title="Machine learning for seizure prediction",
            authors=["Brown K"],
            abstract="We present a novel ML approach...",
            journal="Nature Neuroscience",
            pub_date="2024-02-20",
            mesh_terms=["Machine Learning", "Seizures"]
        )
    ]
    
    # Mock with side effect to increment counter
    def mock_fetch(pmids):
        agent.total_articles_fetched += len(pmids)
        return articles
    
    agent._search_pubmed = AsyncMock(return_value=search_result)
    agent._fetch_articles = AsyncMock(side_effect=mock_fetch)
    
    # Execute search
    result = await agent.execute("EEG biomarkers epilepsy")
    
    # Verify result structure
    assert result["success"] is True
    assert result["count"] == 150
    assert len(result["articles"]) == 2
    assert result["articles"][0]["pmid"] == "12345678"
    assert "EEG biomarkers" in result["articles"][0]["title"]
    
    # Verify statistics were updated
    stats = agent.get_statistics()
    assert stats["total_executions"] == 1
    assert stats["successful_executions"] == 1
    assert stats["total_articles_fetched"] == 3  # 3 articles as per the search result IDs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
