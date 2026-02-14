"""
Tests for EEG Bibliometrics Module

Comprehensive test suite for pyBiblioNet integration.

Requirements tested:
- REQ-BIB-001: OpenAlex integration
- REQ-BIB-002: Citation network analysis
- REQ-BIB-003: Co-authorship network analysis
- REQ-BIB-004: Centrality metrics
- REQ-BIB-005: RAG integration
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from eeg_rag.bibliometrics.eeg_biblionet import (
    EEGArticle,
    EEGAuthor,
    EEGBiblioNet,
    EEGResearchDomain,
    NetworkMetrics,
    build_eeg_citation_network,
    build_eeg_coauthorship_network,
    get_influential_authors,
    get_influential_papers,
    retrieve_eeg_articles,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_openalex_article() -> Dict[str, Any]:
    """Sample OpenAlex API response for an EEG article."""
    return {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.1234/eeg.2023.001",
        "title": "Deep Learning for EEG-Based Epilepsy Detection",
        "abstract": "This study presents a novel deep learning approach for detecting epileptic seizures from EEG signals.",
        "publication_date": "2023-06-15",
        "cited_by_count": 42,
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Jane Smith",
                },
                "institutions": [{"display_name": "MIT"}],
            },
            {
                "author": {
                    "id": "https://openalex.org/A2",
                    "display_name": "John Doe",
                },
                "institutions": [{"display_name": "Stanford"}],
            },
        ],
        "primary_location": {
            "source": {
                "display_name": "Journal of Neural Engineering",
            }
        },
        "topics": [
            {"display_name": "Electroencephalography"},
            {"display_name": "Epilepsy"},
            {"display_name": "Deep Learning"},
        ],
        "referenced_works": [
            "https://openalex.org/W11111",
            "https://openalex.org/W22222",
        ],
        "ids": {
            "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678",
        },
    }


@pytest.fixture
def sample_eeg_article() -> EEGArticle:
    """Sample EEGArticle for testing."""
    return EEGArticle(
        openalex_id="https://openalex.org/W12345",
        doi="https://doi.org/10.1234/eeg.2023.001",
        pmid="12345678",
        title="Deep Learning for EEG-Based Epilepsy Detection",
        abstract="Novel deep learning approach for epileptic seizure detection.",
        authors=["Jane Smith", "John Doe"],
        publication_date="2023-06-15",
        cited_by_count=42,
        venue="Journal of Neural Engineering",
        topics=["Electroencephalography", "Epilepsy", "Deep Learning"],
        referenced_works=["W11111", "W22222"],
    )


@pytest.fixture
def sample_articles() -> List[EEGArticle]:
    """Multiple sample articles for network testing."""
    return [
        EEGArticle(
            openalex_id="W001",
            title="EEG Signal Processing Methods",
            authors=["Author A", "Author B"],
            cited_by_count=100,
            referenced_works=["W002", "W003"],
        ),
        EEGArticle(
            openalex_id="W002",
            title="Brain-Computer Interface Design",
            authors=["Author B", "Author C"],
            cited_by_count=80,
            referenced_works=["W003"],
        ),
        EEGArticle(
            openalex_id="W003",
            title="Epilepsy Detection Using Machine Learning",
            authors=["Author C", "Author D"],
            cited_by_count=60,
            referenced_works=[],
        ),
        EEGArticle(
            openalex_id="W004",
            title="Sleep Stage Classification with EEG",
            authors=["Author A", "Author D"],
            cited_by_count=40,
            referenced_works=["W001", "W002"],
        ),
    ]


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory."""
    cache_dir = tmp_path / "biblionet_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_networkx_graph():
    """Create a mock NetworkX graph for testing."""
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 10
    mock_graph.number_of_edges.return_value = 15
    mock_graph.is_directed.return_value = True
    mock_graph.degree.return_value = [(f"node{i}", i) for i in range(10)]
    return mock_graph


# =============================================================================
# EEGResearchDomain Tests
# =============================================================================


class TestEEGResearchDomain:
    """Tests for EEGResearchDomain enum."""
    
    def test_domain_values(self):
        """Test all domain values are defined."""
        assert EEGResearchDomain.EPILEPSY.value == "epilepsy"
        assert EEGResearchDomain.SLEEP.value == "sleep"
        assert EEGResearchDomain.BCI.value == "brain_computer_interface"
        assert EEGResearchDomain.COGNITIVE.value == "cognitive_neuroscience"
        assert EEGResearchDomain.CLINICAL.value == "clinical_neurophysiology"
        assert EEGResearchDomain.SIGNAL_PROCESSING.value == "signal_processing"
        assert EEGResearchDomain.GENERAL.value == "general_eeg"
    
    def test_domain_count(self):
        """Test correct number of domains."""
        assert len(EEGResearchDomain) == 7


# =============================================================================
# EEGArticle Tests
# =============================================================================


class TestEEGArticle:
    """Tests for EEGArticle dataclass."""
    
    def test_article_creation(self, sample_eeg_article: EEGArticle):
        """Test basic article creation."""
        assert sample_eeg_article.openalex_id == "https://openalex.org/W12345"
        assert sample_eeg_article.title == "Deep Learning for EEG-Based Epilepsy Detection"
        assert len(sample_eeg_article.authors) == 2
        assert sample_eeg_article.cited_by_count == 42
    
    def test_article_to_dict(self, sample_eeg_article: EEGArticle):
        """Test conversion to dictionary."""
        article_dict = sample_eeg_article.to_dict()
        
        assert isinstance(article_dict, dict)
        assert article_dict["openalex_id"] == sample_eeg_article.openalex_id
        assert article_dict["title"] == sample_eeg_article.title
        assert article_dict["authors"] == sample_eeg_article.authors
        assert article_dict["cited_by_count"] == sample_eeg_article.cited_by_count
    
    def test_article_from_openalex(self, sample_openalex_article: Dict[str, Any]):
        """Test creation from OpenAlex API response."""
        article = EEGArticle.from_openalex(sample_openalex_article)
        
        assert article.openalex_id == "https://openalex.org/W12345"
        assert article.doi == "https://doi.org/10.1234/eeg.2023.001"
        assert article.pmid == "12345678"
        assert article.title == "Deep Learning for EEG-Based Epilepsy Detection"
        assert len(article.authors) == 2
        assert "Jane Smith" in article.authors
        assert "John Doe" in article.authors
        assert article.venue == "Journal of Neural Engineering"
        assert len(article.topics) == 3
    
    def test_article_from_openalex_missing_fields(self):
        """Test handling of missing fields in OpenAlex response."""
        minimal_data = {
            "id": "W99999",
            "title": "Minimal Article",
        }
        
        article = EEGArticle.from_openalex(minimal_data)
        
        assert article.openalex_id == "W99999"
        assert article.title == "Minimal Article"
        assert article.doi is None
        assert article.pmid is None
        assert article.authors == []
        assert article.topics == []
    
    def test_article_default_values(self):
        """Test article with default values."""
        article = EEGArticle(openalex_id="W123")
        
        assert article.openalex_id == "W123"
        assert article.title == ""
        assert article.abstract == ""
        assert article.authors == []
        assert article.cited_by_count == 0
        assert article.centrality_score == 0.0


# =============================================================================
# EEGAuthor Tests
# =============================================================================


class TestEEGAuthor:
    """Tests for EEGAuthor dataclass."""
    
    def test_author_creation(self):
        """Test basic author creation."""
        author = EEGAuthor(
            openalex_id="A123",
            name="Dr. Jane Smith",
            affiliations=["MIT", "Harvard"],
            works_count=50,
            cited_by_count=1000,
            h_index=25,
        )
        
        assert author.openalex_id == "A123"
        assert author.name == "Dr. Jane Smith"
        assert len(author.affiliations) == 2
        assert author.h_index == 25
    
    def test_author_to_dict(self):
        """Test conversion to dictionary."""
        author = EEGAuthor(
            openalex_id="A123",
            name="Dr. Jane Smith",
            works_count=50,
        )
        
        author_dict = author.to_dict()
        
        assert isinstance(author_dict, dict)
        assert author_dict["openalex_id"] == "A123"
        assert author_dict["name"] == "Dr. Jane Smith"
        assert author_dict["works_count"] == 50
    
    def test_author_default_values(self):
        """Test author with default values."""
        author = EEGAuthor(openalex_id="A999", name="Unknown Author")
        
        assert author.affiliations == []
        assert author.works_count == 0
        assert author.cited_by_count == 0
        assert author.h_index == 0
        assert author.centrality_score == 0.0
        assert author.collaboration_count == 0


# =============================================================================
# NetworkMetrics Tests
# =============================================================================


class TestNetworkMetrics:
    """Tests for NetworkMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test basic metrics creation."""
        metrics = NetworkMetrics(
            node_count=100,
            edge_count=500,
            density=0.05,
            avg_clustering=0.3,
            num_components=5,
            avg_degree=10.0,
            max_degree=50,
            diameter=8,
        )
        
        assert metrics.node_count == 100
        assert metrics.edge_count == 500
        assert metrics.density == 0.05
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NetworkMetrics(
            node_count=100,
            edge_count=500,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["node_count"] == 100
        assert metrics_dict["edge_count"] == 500
    
    def test_metrics_default_values(self):
        """Test default values."""
        metrics = NetworkMetrics()
        
        assert metrics.node_count == 0
        assert metrics.edge_count == 0
        assert metrics.density == 0.0
        assert metrics.avg_clustering == 0.0
        assert metrics.num_components == 0


# =============================================================================
# EEGBiblioNet Initialization Tests
# =============================================================================


class TestEEGBiblioNetInit:
    """Tests for EEGBiblioNet initialization."""
    
    def test_init_with_email(self, temp_cache_dir: Path):
        """Test initialization with valid email."""
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        assert biblio.email == "test@example.com"
        assert biblio.cache_dir == temp_cache_dir
        assert biblio.use_cache is True
        assert biblio.articles == []
    
    def test_init_without_email(self):
        """Test initialization without email raises error."""
        with pytest.raises(ValueError, match="Email is required"):
            EEGBiblioNet(email="")
    
    def test_init_creates_cache_dir(self, tmp_path: Path):
        """Test that cache directory is created if not exists."""
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=cache_dir,
        )
        
        assert cache_dir.exists()
    
    def test_init_disables_cache(self, temp_cache_dir: Path):
        """Test initialization with cache disabled."""
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
            use_cache=False,
        )
        
        assert biblio.use_cache is False
    
    def test_query_patterns_defined(self, temp_cache_dir: Path):
        """Test that all query patterns are defined."""
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        for domain in EEGResearchDomain:
            assert domain in biblio.EEG_QUERY_PATTERNS
            pattern = biblio.EEG_QUERY_PATTERNS[domain]
            assert isinstance(pattern, str)
            assert len(pattern) > 0
            # All patterns should include EEG-related terms
            assert "EEG" in pattern or "electroencephalography" in pattern


# =============================================================================
# EEGBiblioNet Search Tests (Mocked)
# =============================================================================


class TestEEGBiblioNetSearch:
    """Tests for EEGBiblioNet search functionality (mocked)."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_search_without_pybiblionet(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test that search raises error when pyBiblioNet not installed."""
        mock_check.return_value = False
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio._pybiblionet_available = False
        
        with pytest.raises(RuntimeError, match="pyBiblioNet is not installed"):
            biblio.search_eeg_literature()


# =============================================================================
# EEGBiblioNet Network Building Tests
# =============================================================================


class TestEEGBiblioNetNetworks:
    """Tests for network building functionality."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_build_citation_network_no_articles(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test building citation network without articles raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio._pybiblionet_available = True
        
        with pytest.raises(ValueError, match="No articles retrieved"):
            biblio.build_citation_network()
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_build_coauthorship_network_no_articles(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test building co-authorship network without articles raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio._pybiblionet_available = True
        
        with pytest.raises(ValueError, match="No articles retrieved"):
            biblio.build_coauthorship_network()


# =============================================================================
# EEGBiblioNet Centrality Tests
# =============================================================================


class TestEEGBiblioNetCentrality:
    """Tests for centrality computation."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_citation_centrality_no_network(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test computing centrality without network raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Citation network not built"):
            biblio.compute_citation_centrality()
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_coauthorship_centrality_no_network(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test computing co-authorship centrality without network raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Co-authorship network not built"):
            biblio.compute_coauthorship_centrality()
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_invalid_centrality_method(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
        mock_networkx_graph: MagicMock,
    ):
        """Test invalid centrality method raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio.citation_graph = mock_networkx_graph
        
        with patch("networkx.pagerank"):
            with pytest.raises(ValueError, match="Unknown centrality method"):
                biblio.compute_citation_centrality(method="invalid_method")


# =============================================================================
# EEGBiblioNet Network Metrics Tests
# =============================================================================


class TestEEGBiblioNetMetrics:
    """Tests for network metrics computation."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_metrics_no_network(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test getting metrics without network raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Citation network not built"):
            biblio.get_network_metrics(network_type="citation")
        
        with pytest.raises(ValueError, match="Co-authorship network not built"):
            biblio.get_network_metrics(network_type="coauthorship")
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_metrics_invalid_type(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test getting metrics with invalid type raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Unknown network type"):
            biblio.get_network_metrics(network_type="invalid")


# =============================================================================
# EEGBiblioNet Community Detection Tests
# =============================================================================


class TestEEGBiblioNetCommunities:
    """Tests for community detection."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_detect_communities_no_network(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test detecting communities without network raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Citation network not built"):
            biblio.detect_communities(network_type="citation")
        
        with pytest.raises(ValueError, match="Co-authorship network not built"):
            biblio.detect_communities(network_type="coauthorship")
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_detect_communities_invalid_type(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test detecting communities with invalid type raises error."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        with pytest.raises(ValueError, match="Unknown network type"):
            biblio.detect_communities(network_type="invalid")


# =============================================================================
# EEGBiblioNet RAG Integration Tests
# =============================================================================


class TestEEGBiblioNetRAG:
    """Tests for RAG integration functionality."""
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_articles_for_rag_empty(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
    ):
        """Test getting articles for RAG with no articles."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        
        rag_articles = biblio.get_articles_for_rag()
        
        assert rag_articles == []
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_articles_for_rag_with_articles(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test getting articles for RAG with articles."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio.articles = sample_articles
        
        rag_articles = biblio.get_articles_for_rag()
        
        assert len(rag_articles) == 4
        
        # Check structure of RAG document
        for doc in rag_articles:
            assert "id" in doc
            assert "content" in doc
            assert "metadata" in doc
            assert "source" in doc["metadata"]
            assert doc["metadata"]["source"] == "OpenAlex"
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_articles_for_rag_filtered_by_citations(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test filtering articles by citation count."""
        mock_check.return_value = True
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio.articles = sample_articles
        
        # Filter for high-citation articles (>= 80)
        rag_articles = biblio.get_articles_for_rag(min_citations=80)
        
        assert len(rag_articles) == 2  # Only articles with 100 and 80 citations
    
    @patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet")
    def test_get_articles_for_rag_filtered_by_centrality(
        self,
        mock_check: MagicMock,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test filtering articles by centrality score."""
        mock_check.return_value = True
        
        # Set centrality scores
        sample_articles[0].centrality_score = 0.5
        sample_articles[1].centrality_score = 0.3
        sample_articles[2].centrality_score = 0.1
        sample_articles[3].centrality_score = 0.05
        
        biblio = EEGBiblioNet(
            email="test@example.com",
            cache_dir=temp_cache_dir,
        )
        biblio.articles = sample_articles
        
        # Filter for high-centrality articles
        rag_articles = biblio.get_articles_for_rag(min_centrality=0.2)
        
        assert len(rag_articles) == 2  # Only articles with 0.5 and 0.3 centrality


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_retrieve_eeg_articles_no_pybiblionet(self):
        """Test retrieve function without pyBiblioNet."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = False
            
            with pytest.raises(RuntimeError, match="pyBiblioNet is not installed"):
                retrieve_eeg_articles(email="test@example.com")
    
    def test_build_citation_network_function(self, sample_articles: List[EEGArticle]):
        """Test convenience function for building citation network."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as check_mock:
            check_mock.return_value = True
            
            with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet.build_citation_network") as build_mock:
                build_mock.return_value = MagicMock()
                
                result = build_eeg_citation_network(
                    articles=sample_articles,
                    output_path="/tmp/test.gml",
                )
                
                build_mock.assert_called_once()
    
    def test_build_coauthorship_network_function(self, sample_articles: List[EEGArticle]):
        """Test convenience function for building co-authorship network."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as check_mock:
            check_mock.return_value = True
            
            with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet.build_coauthorship_network") as build_mock:
                build_mock.return_value = MagicMock()
                
                result = build_eeg_coauthorship_network(
                    articles=sample_articles,
                    output_path="/tmp/test.gml",
                )
                
                build_mock.assert_called_once()
    
    def test_get_influential_papers_function(self, sample_articles: List[EEGArticle]):
        """Test convenience function for getting influential papers."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as check_mock:
            check_mock.return_value = True
            
            with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet.get_influential_papers") as inf_mock:
                inf_mock.return_value = sample_articles[:3]
                
                result = get_influential_papers(
                    articles=sample_articles,
                    top_n=3,
                )
                
                assert len(result) == 3
    
    def test_get_influential_authors_function(self, sample_articles: List[EEGArticle]):
        """Test convenience function for getting influential authors."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as check_mock:
            check_mock.return_value = True
            
            with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet.get_influential_authors") as inf_mock:
                inf_mock.return_value = [("Author A", 0.5), ("Author B", 0.3)]
                
                result = get_influential_authors(
                    articles=sample_articles,
                    top_n=2,
                )
                
                assert len(result) == 2
                assert result[0][0] == "Author A"


# =============================================================================
# Integration Tests (with networkx)
# =============================================================================


class TestNetworkXIntegration:
    """Integration tests with NetworkX (requires networkx)."""
    
    def test_centrality_computation_with_real_networkx(
        self,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test centrality computation with real NetworkX graph."""
        import networkx as nx
        
        # Create a simple directed graph for testing
        G = nx.DiGraph()
        G.add_edges_from([
            ("W001", "W002"),
            ("W001", "W003"),
            ("W002", "W003"),
            ("W004", "W001"),
            ("W004", "W002"),
        ])
        
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = True
            
            biblio = EEGBiblioNet(
                email="test@example.com",
                cache_dir=temp_cache_dir,
            )
            biblio.articles = sample_articles
            biblio.citation_graph = G
            
            # Test PageRank
            centrality = biblio.compute_citation_centrality(method="pagerank")
            assert len(centrality) == 4
            assert all(0 <= v <= 1 for v in centrality.values())
            
            # Test betweenness
            centrality = biblio.compute_citation_centrality(method="betweenness")
            assert len(centrality) == 4
            
            # Test degree
            centrality = biblio.compute_citation_centrality(method="degree")
            assert len(centrality) == 4
    
    def test_network_metrics_with_real_networkx(
        self,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test network metrics with real NetworkX graph."""
        import networkx as nx
        
        # Create a simple graph for testing
        G = nx.Graph()
        G.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("A", "C"),
        ])
        
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = True
            
            biblio = EEGBiblioNet(
                email="test@example.com",
                cache_dir=temp_cache_dir,
            )
            biblio.coauthorship_graph = G
            
            metrics = biblio.get_network_metrics(network_type="coauthorship")
            
            assert metrics.node_count == 4
            assert metrics.edge_count == 4
            assert 0 < metrics.density < 1
            assert metrics.num_components == 1
            assert metrics.avg_degree > 0
    
    def test_community_detection_with_real_networkx(
        self,
        temp_cache_dir: Path,
    ):
        """Test community detection with real NetworkX graph."""
        import networkx as nx
        
        # Create a graph with clear community structure
        G = nx.Graph()
        # Community 1: A-B-C (densely connected)
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        # Community 2: D-E-F (densely connected)
        G.add_edges_from([("D", "E"), ("E", "F"), ("D", "F")])
        # Bridge between communities
        G.add_edge("C", "D")
        
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = True
            
            biblio = EEGBiblioNet(
                email="test@example.com",
                cache_dir=temp_cache_dir,
            )
            biblio.coauthorship_graph = G
            
            communities = biblio.detect_communities(
                network_type="coauthorship",
                method="label_propagation",
            )
            
            assert len(communities) == 6  # All nodes assigned
            assert all(isinstance(v, int) for v in communities.values())


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_article_list(self, temp_cache_dir: Path):
        """Test handling of empty article list."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = True
            
            biblio = EEGBiblioNet(
                email="test@example.com",
                cache_dir=temp_cache_dir,
            )
            
            rag_articles = biblio.get_articles_for_rag()
            assert rag_articles == []
    
    def test_article_with_all_none_values(self):
        """Test article creation with all None/empty values."""
        article = EEGArticle(
            openalex_id="W000",
            doi=None,
            pmid=None,
            title="",
            abstract="",
        )
        
        assert article.openalex_id == "W000"
        assert article.doi is None
        assert article.pmid is None
    
    def test_article_from_openalex_with_null_abstract(self):
        """Test parsing OpenAlex response with null abstract."""
        data = {
            "id": "W123",
            "title": "Test Article",
            "abstract": None,  # Explicitly null
        }
        
        article = EEGArticle.from_openalex(data)
        
        assert article.abstract == ""  # Should be empty string, not None
    
    def test_rag_filtering_combinations(
        self,
        temp_cache_dir: Path,
        sample_articles: List[EEGArticle],
    ):
        """Test RAG filtering with multiple criteria."""
        with patch("eeg_rag.bibliometrics.eeg_biblionet.EEGBiblioNet._check_pybiblionet") as mock:
            mock.return_value = True
            
            sample_articles[0].centrality_score = 0.5
            sample_articles[0].topics = ["Epilepsy", "Deep Learning"]
            sample_articles[1].centrality_score = 0.3
            sample_articles[1].topics = ["BCI", "Motor Imagery"]
            
            biblio = EEGBiblioNet(
                email="test@example.com",
                cache_dir=temp_cache_dir,
            )
            biblio.articles = sample_articles
            
            # Multiple filters
            rag_articles = biblio.get_articles_for_rag(
                min_citations=50,
                min_centrality=0.2,
                topics=["Epilepsy"],
            )
            
            assert len(rag_articles) == 1
            assert rag_articles[0]["id"] == "W001"
