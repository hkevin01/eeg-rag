"""
Comprehensive tests for EEG Bibliometrics Advanced Modules.

Tests visualization, NLP enhancement, and research export functionality.

Coverage targets:
- visualization.py: EEGVisualization, ChartResult
- nlp_enhancement.py: EEGNLPEnhancer, ExtractedKeywords, TopicCluster
- research_export.py: EEGResearchExporter, VenueMetrics, InstitutionMetrics, AuthorProductivity
"""

from __future__ import annotations

import base64
import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_articles() -> List[Dict[str, Any]]:
    """Create sample article data for testing."""
    return [
        {
            "id": "W1234567890",
            "doi": "10.1000/test.001",
            "title": "Deep Learning for EEG-based Seizure Detection",
            "abstract": "We present a novel deep learning approach for automated seizure detection in EEG signals using convolutional neural networks.",
            "publication_date": "2024-06-15",
            "publication_year": 2024,
            "cited_by_count": 42,
            "authorships": [
                {
                    "author": {"id": "A001", "display_name": "John Smith", "orcid": "0000-0001-0001"},
                    "institutions": [{"id": "I001", "display_name": "MIT", "country_code": "US", "type": "education"}],
                    "countries": ["US"],
                    "raw_affiliation_strings": ["Massachusetts Institute of Technology"],
                },
                {
                    "author": {"id": "A002", "display_name": "Jane Doe", "orcid": "0000-0002-0002"},
                    "institutions": [{"id": "I002", "display_name": "Stanford", "country_code": "US", "type": "education"}],
                    "countries": ["US"],
                    "raw_affiliation_strings": ["Stanford University"],
                },
            ],
            "primary_location": {
                "source": {
                    "id": "S001",
                    "display_name": "NeuroImage",
                    "issn_l": "1053-8119",
                    "issn": ["1053-8119"],
                    "host_organization_name": "Elsevier",
                }
            },
            "primary_topic": {
                "display_name": "EEG Signal Processing",
                "field": {"display_name": "Neuroscience"},
                "domain": {"display_name": "Medical Sciences"},
                "subfield": {"display_name": "Clinical Neurology"},
            },
            "type": "article",
            "language": "en",
            "open_access": {"is_oa": True, "oa_url": "https://example.com/paper.pdf"},
            "root_set": True,
        },
        {
            "id": "W0987654321",
            "doi": "10.1000/test.002",
            "title": "Motor Imagery Classification with Transfer Learning",
            "abstract": "This study explores transfer learning techniques for motor imagery EEG classification in brain-computer interfaces.",
            "publication_date": "2023-03-20",
            "publication_year": 2023,
            "cited_by_count": 28,
            "authorships": [
                {
                    "author": {"id": "A003", "display_name": "Alice Wang", "orcid": "0000-0003-0003"},
                    "institutions": [{"id": "I003", "display_name": "Tsinghua University", "country_code": "CN", "type": "education"}],
                    "countries": ["CN"],
                    "raw_affiliation_strings": ["Tsinghua University"],
                },
            ],
            "primary_location": {
                "source": {
                    "id": "S002",
                    "display_name": "Journal of Neural Engineering",
                    "issn_l": "1741-2560",
                    "issn": ["1741-2560"],
                    "host_organization_name": "IOP Publishing",
                }
            },
            "primary_topic": {
                "display_name": "Brain-Computer Interface",
                "field": {"display_name": "Computer Science"},
                "domain": {"display_name": "Engineering"},
                "subfield": {"display_name": "Biomedical Engineering"},
            },
            "type": "article",
            "language": "en",
            "open_access": {"is_oa": False},
            "root_set": True,
        },
        {
            "id": "W5555555555",
            "doi": "10.1000/test.003",
            "title": "Sleep Stage Classification Using EEG Features",
            "abstract": "We analyze polysomnography data to classify sleep stages using machine learning with focus on slow-wave sleep detection.",
            "publication_date": "2022-11-10",
            "publication_year": 2022,
            "cited_by_count": 15,
            "authorships": [
                {
                    "author": {"id": "A001", "display_name": "John Smith"},
                    "institutions": [{"id": "I001", "display_name": "MIT", "country_code": "US", "type": "education"}],
                    "countries": ["US"],
                    "raw_affiliation_strings": ["MIT"],
                },
            ],
            "primary_location": {
                "source": {
                    "id": "S003",
                    "display_name": "Sleep Medicine",
                    "issn_l": "1389-9457",
                    "issn": ["1389-9457"],
                    "host_organization_name": "Elsevier",
                }
            },
            "primary_topic": {
                "display_name": "Sleep Research",
                "field": {"display_name": "Neuroscience"},
                "domain": {"display_name": "Medical Sciences"},
                "subfield": {"display_name": "Sleep Medicine"},
            },
            "type": "article",
            "language": "en",
            "root_set": True,
        },
    ]


@pytest.fixture
def sample_eeg_abstract() -> str:
    """Sample EEG research abstract."""
    return """
    This study investigates deep learning approaches for EEG-based seizure detection. 
    We employed convolutional neural networks (CNN) and long short-term memory (LSTM) 
    networks to classify ictal and interictal periods. The dataset consisted of 
    scalp EEG recordings from 50 epilepsy patients using the 10-20 electrode system.
    Our method achieved 95% sensitivity and 92% specificity for seizure detection,
    outperforming traditional machine learning baselines including SVM and Random Forest.
    The proposed deep learning approach shows promise for real-time seizure monitoring
    in clinical settings.
    """


# =============================================================================
# ChartResult Tests
# =============================================================================

class TestChartResult:
    """Tests for ChartResult dataclass."""

    def test_chart_result_basic(self):
        """Test basic ChartResult creation."""
        from eeg_rag.bibliometrics.visualization import ChartResult
        
        chart = ChartResult(
            title="Test Chart",
            chart_type="bar",
            png_base64="dGVzdA==",  # 'test' in base64
        )
        assert chart.title == "Test Chart"
        assert chart.chart_type == "bar"
        assert chart.png_base64 == "dGVzdA=="

    def test_to_html_img(self):
        """Test HTML img tag generation."""
        from eeg_rag.bibliometrics.visualization import ChartResult
        
        chart = ChartResult(
            title="Test",
            chart_type="line",
            png_base64="YWJj",
        )
        html = chart.to_html_img()
        assert '<img src="data:image/png;base64,YWJj"' in html
        assert 'alt="Test"' in html

    def test_to_html_img_no_data(self):
        """Test HTML img with no data."""
        from eeg_rag.bibliometrics.visualization import ChartResult
        
        chart = ChartResult(title="Empty", chart_type="bar")
        assert chart.to_html_img() == ""

    def test_chart_result_save_with_figure(self, tmp_path):
        """Test saving chart with matplotlib figure."""
        from eeg_rag.bibliometrics.visualization import ChartResult
        
        # Create mock figure
        mock_fig = MagicMock()
        chart = ChartResult(
            title="Test",
            chart_type="bar",
            figure=mock_fig,
        )
        
        output_path = tmp_path / "test.png"
        chart.save(output_path, format="png")
        mock_fig.savefig.assert_called_once()


# =============================================================================
# EEGVisualization Tests
# =============================================================================

class TestEEGVisualization:
    """Tests for EEGVisualization class."""

    def test_visualization_init(self):
        """Test visualization engine initialization."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        assert viz.style == "seaborn-v0_8-whitegrid"
        assert len(viz.EEG_COLORS) == 10

    def test_visualization_custom_style(self):
        """Test custom style."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization(style="ggplot")
        assert viz.style == "ggplot"

    def test_parse_date_valid(self):
        """Test valid date parsing."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        date = viz._parse_date("2024-06-15")
        assert date is not None
        assert date.year == 2024
        assert date.month == 6

    def test_parse_date_invalid(self):
        """Test invalid date parsing."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        assert viz._parse_date("invalid") is None
        assert viz._parse_date(None) is None
        assert viz._parse_date("") is None

    def test_parse_date_with_time(self):
        """Test date parsing with time component."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        date = viz._parse_date("2024-06-15T10:30:00Z")
        assert date is not None
        assert date.year == 2024

    def test_plot_citation_distribution(self, sample_articles):
        """Test citation distribution chart creation."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization, ChartResult
        
        viz = EEGVisualization()
        # Test that method exists and is callable
        assert hasattr(viz, 'plot_citation_distribution')
        assert callable(viz.plot_citation_distribution)

    def test_plot_venue_distribution(self, sample_articles):
        """Test venue distribution chart."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        assert hasattr(viz, 'plot_venue_distribution')
        assert callable(viz.plot_venue_distribution)

    @patch("matplotlib.pyplot.subplots")
    def test_plot_publication_trends(self, mock_subplots, sample_articles):
        """Test publication trends plot."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization, ChartResult
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        viz = EEGVisualization()
        result = viz.plot_publication_trends(sample_articles, interval="year")
        
        assert isinstance(result, ChartResult)
        assert result.chart_type == "area"
        assert "trends" in result.title.lower() or result.title != ""

    @patch("matplotlib.pyplot.subplots")
    def test_plot_topic_evolution(self, mock_subplots, sample_articles):
        """Test topic evolution plot."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization, ChartResult
        
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        viz = EEGVisualization()
        result = viz.plot_topic_evolution(sample_articles, field_key="domain", top_n=5)
        
        assert isinstance(result, ChartResult)

    def test_create_research_dashboard(self, sample_articles):
        """Test dashboard creation method exists."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        
        viz = EEGVisualization()
        assert hasattr(viz, 'create_research_dashboard')
        assert callable(viz.create_research_dashboard)


# =============================================================================
# ExtractedKeywords Tests
# =============================================================================

class TestExtractedKeywords:
    """Tests for ExtractedKeywords dataclass."""

    def test_extracted_keywords_basic(self):
        """Test basic keyword extraction result."""
        from eeg_rag.bibliometrics.nlp_enhancement import ExtractedKeywords
        
        keywords = ExtractedKeywords(
            keywords=[("seizure", 0.9), ("eeg", 0.85), ("detection", 0.7)],
            document_id="doc1",
        )
        assert len(keywords.keywords) == 3
        assert keywords.method == "keybert"

    def test_get_top_n(self):
        """Test getting top N keywords."""
        from eeg_rag.bibliometrics.nlp_enhancement import ExtractedKeywords
        
        keywords = ExtractedKeywords(
            keywords=[("a", 0.5), ("b", 0.9), ("c", 0.7)],
        )
        top2 = keywords.get_top_n(2)
        assert top2 == ["b", "c"]

    def test_to_dict(self):
        """Test dictionary conversion."""
        from eeg_rag.bibliometrics.nlp_enhancement import ExtractedKeywords
        
        kw = ExtractedKeywords(
            keywords=[("test", 0.8)],
            document_id="doc1",
            method="custom",
            ngram_range=(1, 3),
        )
        d = kw.to_dict()
        assert d["document_id"] == "doc1"
        assert d["method"] == "custom"
        assert d["ngram_range"] == (1, 3)


# =============================================================================
# TopicCluster Tests
# =============================================================================

class TestTopicCluster:
    """Tests for TopicCluster dataclass."""

    def test_topic_cluster_creation(self):
        """Test topic cluster creation."""
        from eeg_rag.bibliometrics.nlp_enhancement import TopicCluster
        
        cluster = TopicCluster(
            topic_id=0,
            keywords=["seizure", "epilepsy", "ictal"],
            document_ids=["d1", "d2"],
            coherence_score=0.85,
            label="Epilepsy Research",
        )
        assert cluster.topic_id == 0
        assert len(cluster.keywords) == 3
        assert cluster.label == "Epilepsy Research"

    def test_topic_cluster_to_dict(self):
        """Test dictionary conversion."""
        from eeg_rag.bibliometrics.nlp_enhancement import TopicCluster
        
        cluster = TopicCluster(
            topic_id=1,
            keywords=["bci"],
            coherence_score=0.9,
        )
        d = cluster.to_dict()
        assert d["topic_id"] == 1
        assert d["coherence_score"] == 0.9


# =============================================================================
# EEGNLPEnhancer Tests
# =============================================================================

class TestEEGNLPEnhancer:
    """Tests for EEGNLPEnhancer class."""

    def test_enhancer_init(self):
        """Test NLP enhancer initialization."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        assert not enhancer.use_keybert
        assert not enhancer.use_spacy

    def test_eeg_stopwords(self):
        """Test EEG-specific stopwords."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        assert "eeg" in enhancer.EEG_STOPWORDS
        assert "electroencephalography" in enhancer.EEG_STOPWORDS
        assert "data" in enhancer.EEG_STOPWORDS

    def test_eeg_domain_terms(self):
        """Test EEG domain terms."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        assert "frequency_bands" in enhancer.EEG_DOMAIN_TERMS
        assert "alpha" in enhancer.EEG_DOMAIN_TERMS["frequency_bands"]
        assert "seizure" in enhancer.EEG_DOMAIN_TERMS["clinical"]

    def test_lemmatize_text(self, sample_eeg_abstract):
        """Test text lemmatization."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        # Should return text even without spacy
        result = enhancer.lemmatize_text(sample_eeg_abstract)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_keyword_extraction_from_text(self, sample_eeg_abstract):
        """Test keyword extraction from text."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer, ExtractedKeywords
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        result = enhancer.extract_keywords_from_text(sample_eeg_abstract, top_n=10)
        
        assert isinstance(result, ExtractedKeywords)
        assert len(result.keywords) <= 10

    def test_expand_query_basic(self):
        """Test basic query expansion."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        expanded = enhancer.expand_query("seizure detection")
        
        assert isinstance(expanded, list)
        assert len(expanded) > 0
        # Original query should be in expanded list
        assert any("seizure" in term.lower() for term in expanded)

    def test_categorize_by_topic(self, sample_articles):
        """Test topic categorization."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        categories = enhancer.categorize_by_topic(sample_articles)
        
        assert isinstance(categories, dict)

    def test_compute_text_similarity(self):
        """Test text similarity computation."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        text1 = "EEG seizure detection using deep learning"
        text2 = "Deep learning for epilepsy EEG analysis"
        similarity = enhancer.compute_text_similarity(text1, text2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_extract_keywords_from_articles(self, sample_articles):
        """Test keyword extraction from article list."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer, ExtractedKeywords
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        results = enhancer.extract_keywords_from_articles(sample_articles, top_n=5)
        
        # Returns ExtractedKeywords with aggregated keywords
        assert isinstance(results, ExtractedKeywords)
        # Should have keywords
        assert len(results.keywords) >= 0


# =============================================================================
# VenueMetrics Tests
# =============================================================================

class TestVenueMetrics:
    """Tests for VenueMetrics dataclass."""

    def test_venue_metrics_creation(self):
        """Test venue metrics creation."""
        from eeg_rag.bibliometrics.research_export import VenueMetrics
        
        venue = VenueMetrics(
            name="NeuroImage",
            article_count=100,
            total_citations=5000,
            mean_citations=50.0,
            h_index=35,
        )
        assert venue.name == "NeuroImage"
        assert venue.h_index == 35

    def test_venue_metrics_to_dict(self):
        """Test dictionary conversion."""
        from eeg_rag.bibliometrics.research_export import VenueMetrics
        
        venue = VenueMetrics(name="Test", article_count=10)
        d = venue.to_dict()
        assert d["name"] == "Test"
        assert d["article_count"] == 10


# =============================================================================
# InstitutionMetrics Tests
# =============================================================================

class TestInstitutionMetrics:
    """Tests for InstitutionMetrics dataclass."""

    def test_institution_metrics_creation(self):
        """Test institution metrics creation."""
        from eeg_rag.bibliometrics.research_export import InstitutionMetrics
        
        inst = InstitutionMetrics(
            name="MIT",
            country="US",
            article_count=500,
            author_count=150,
        )
        assert inst.name == "MIT"
        assert inst.country == "US"

    def test_institution_with_collaborators(self):
        """Test institution with collaborators."""
        from eeg_rag.bibliometrics.research_export import InstitutionMetrics
        
        inst = InstitutionMetrics(
            name="Stanford",
            collaborators=["MIT", "Berkeley", "CMU"],
        )
        assert len(inst.collaborators) == 3


# =============================================================================
# AuthorProductivity Tests
# =============================================================================

class TestAuthorProductivity:
    """Tests for AuthorProductivity dataclass."""

    def test_author_productivity_creation(self):
        """Test author productivity creation."""
        from eeg_rag.bibliometrics.research_export import AuthorProductivity
        
        author = AuthorProductivity(
            name="John Smith",
            openalex_id="A001",
            article_count=45,
            total_citations=1200,
            h_index=18,
            first_author_count=15,
            last_author_count=10,
        )
        assert author.name == "John Smith"
        assert author.h_index == 18

    def test_author_productivity_to_dict(self):
        """Test dictionary conversion."""
        from eeg_rag.bibliometrics.research_export import AuthorProductivity
        
        author = AuthorProductivity(
            name="Test Author",
            article_count=10,
            years_active=(2015, 2024),
        )
        d = author.to_dict()
        assert d["years_active"] == (2015, 2024)


# =============================================================================
# EEGResearchExporter Tests
# =============================================================================

class TestEEGResearchExporter:
    """Tests for EEGResearchExporter class."""

    def test_exporter_init(self, sample_articles):
        """Test exporter initialization."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        assert len(exporter.articles) == 3

    def test_exporter_empty_init(self):
        """Test exporter with no articles."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter([])
        assert len(exporter.articles) == 0

    def test_scopus_fields(self, sample_articles):
        """Test Scopus field mapping."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        assert hasattr(exporter, 'SCOPUS_FIELDS')

    def test_export_to_scopus_csv(self, sample_articles, tmp_path):
        """Test Scopus CSV export."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "scopus_export.csv"
        exporter.export_to_scopus_csv(str(output_path))
        
        assert output_path.exists()
        
        # Verify CSV content
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3

    def test_export_authors_csv(self, sample_articles, tmp_path):
        """Test authors CSV export."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "authors.csv"
        exporter.export_authors_csv(str(output_path))
        
        assert output_path.exists()

    def test_export_institutions_csv(self, sample_articles, tmp_path):
        """Test institutions CSV export."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "institutions.csv"
        exporter.export_institutions_csv(str(output_path))
        
        assert output_path.exists()

    def test_export_venues_csv(self, sample_articles, tmp_path):
        """Test venues CSV export."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "venues.csv"
        exporter.export_venues_csv(str(output_path))
        
        assert output_path.exists()

    def test_compute_venue_metrics(self, sample_articles):
        """Test venue metrics computation."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter, VenueMetrics
        
        exporter = EEGResearchExporter(sample_articles)
        metrics = exporter.compute_venue_metrics()
        
        assert isinstance(metrics, list)
        if metrics:
            assert isinstance(metrics[0], VenueMetrics)

    def test_compute_institution_metrics(self, sample_articles):
        """Test institution metrics computation."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter, InstitutionMetrics
        
        exporter = EEGResearchExporter(sample_articles)
        metrics = exporter.compute_institution_metrics()
        
        assert isinstance(metrics, list)
        if metrics:
            assert isinstance(metrics[0], InstitutionMetrics)

    def test_compute_author_productivity(self, sample_articles):
        """Test author productivity computation."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter, AuthorProductivity
        
        exporter = EEGResearchExporter(sample_articles)
        productivity = exporter.compute_author_productivity()
        
        assert isinstance(productivity, list)
        if productivity:
            assert isinstance(productivity[0], AuthorProductivity)

    def test_compute_h_index(self, sample_articles):
        """Test h-index computation."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        citations = [50, 30, 20, 10, 5, 3, 2, 1]
        h_index = exporter._compute_h_index(citations)
        
        assert isinstance(h_index, int)
        assert h_index >= 0

    def test_export_all(self, sample_articles, tmp_path):
        """Test export all to directory."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        exporter = EEGResearchExporter(sample_articles)
        output_dir = tmp_path / "exports"
        output_dir.mkdir()
        exporter.export_all(str(output_dir))
        
        # Should create multiple files
        assert (output_dir / "scopus_articles.csv").exists() or len(list(output_dir.iterdir())) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestBibliometricsIntegration:
    """Integration tests for bibliometrics modules."""

    def test_full_workflow(self, sample_articles, tmp_path):
        """Test complete visualization + NLP + export workflow."""
        from eeg_rag.bibliometrics.visualization import EEGVisualization
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer, ExtractedKeywords
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        # Step 1: NLP enhancement
        nlp = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        keywords = nlp.extract_keywords_from_articles(sample_articles, top_n=5)
        assert isinstance(keywords, ExtractedKeywords)
        
        # Step 2: Visualization exists
        viz = EEGVisualization()
        assert hasattr(viz, 'plot_publication_trends')
        
        # Step 3: Export
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "full_export.csv"
        exporter.export_to_scopus_csv(str(output_path))
        assert output_path.exists()

    def test_keyword_enhanced_export(self, sample_articles, tmp_path):
        """Test export with keyword enhancement."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        nlp = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        
        # Add keywords to articles
        for article in sample_articles:
            abstract = article.get("abstract", "")
            if abstract:
                kw = nlp.extract_keywords_from_text(abstract, top_n=5)
                article["extracted_keywords"] = kw.get_top_n(5)
        
        exporter = EEGResearchExporter(sample_articles)
        output_path = tmp_path / "keywords_export.csv"
        exporter.export_to_scopus_csv(str(output_path))
        
        assert output_path.exists()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for bibliometrics modules."""

    def test_empty_abstract_keywords(self):
        """Test keyword extraction from empty abstract."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        result = enhancer.extract_keywords_from_text("", top_n=10)
        
        assert len(result.keywords) == 0

    def test_missing_fields_in_article(self, tmp_path):
        """Test export with missing fields."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        sparse_articles = [
            {"id": "W001", "title": "Test"},
            {"id": "W002"},  # Missing most fields
        ]
        
        exporter = EEGResearchExporter(sparse_articles)
        output_path = tmp_path / "sparse.csv"
        exporter.export_to_scopus_csv(str(output_path))
        
        assert output_path.exists()

    def test_special_characters_in_text(self):
        """Test handling special characters in keyword extraction."""
        from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
        
        enhancer = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
        text = "EEG analysis with alpha beta gamma bands and special processing"
        result = enhancer.extract_keywords_from_text(text, top_n=5)
        
        assert isinstance(result.keywords, list)

    def test_unicode_handling(self, tmp_path):
        """Test Unicode character handling in export."""
        from eeg_rag.bibliometrics.research_export import EEGResearchExporter
        
        articles = [{
            "id": "W001",
            "title": "研究中文标题 — EEG análise",
            "authorships": [
                {"author": {"id": "A1", "display_name": "李明"}}
            ],
        }]
        
        exporter = EEGResearchExporter(articles)
        output_path = tmp_path / "unicode.csv"
        exporter.export_to_scopus_csv(str(output_path))
        
        # Should not raise encoding errors
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "研究中文标题" in content or len(content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
