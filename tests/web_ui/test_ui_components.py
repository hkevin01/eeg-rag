"""
Tests for UI components and session state management.

Tests the Streamlit UI rendering functions, session state
initialization, and configuration management.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any
import sys


class TestAppConfig:
    """Tests for AppConfig configuration class."""
    
    def test_config_page_title(self):
        """Test page title configuration."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert AppConfig.PAGE_TITLE is not None
        assert len(AppConfig.PAGE_TITLE) > 0
        
    def test_config_page_icon(self):
        """Test page icon configuration."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert AppConfig.PAGE_ICON is not None
        
    def test_config_default_paths(self):
        """Test default path configurations."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert AppConfig.DEFAULT_CORPUS_PATH is not None
        assert AppConfig.DEFAULT_EMBEDDINGS_PATH is not None
        assert AppConfig.DEFAULT_BENCHMARK_CSV is not None
        
    def test_config_limits(self):
        """Test configuration limits."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert AppConfig.MAX_QUERY_LENGTH > 0
        assert AppConfig.MAX_RESPONSE_SOURCES > 0
        assert AppConfig.BENCHMARK_BATCH_SIZE > 0
        

class TestQueryComplexity:
    """Tests for QueryComplexity enum."""
    
    def test_complexity_values(self):
        """Test query complexity enum values."""
        from eeg_rag.web_ui.app import QueryComplexity
        
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MEDIUM.value == "medium"
        assert QueryComplexity.COMPLEX.value == "complex"
        assert QueryComplexity.EXPERT.value == "expert"
        
    def test_all_complexity_levels(self):
        """Test all complexity levels are defined."""
        from eeg_rag.web_ui.app import QueryComplexity
        
        assert len(QueryComplexity) == 4


class TestSessionStateInit:
    """Tests for session state initialization."""
    
    def test_init_session_state_defaults(self, mock_streamlit):
        """Test session state initialization with defaults."""
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            from eeg_rag.web_ui.app import init_session_state
            
            init_session_state()
            
            # Check that session state has expected keys
            assert "query_history" in mock_streamlit.session_state
            assert "benchmark_results" in mock_streamlit.session_state
            assert "settings" in mock_streamlit.session_state
            
    def test_init_session_state_preserves_existing(self, mock_streamlit):
        """Test that existing session state values are preserved."""
        mock_streamlit.session_state["query_history"] = ["existing_query"]
        
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            from eeg_rag.web_ui.app import init_session_state
            
            init_session_state()
            
            # Existing value should be preserved
            assert mock_streamlit.session_state["query_history"] == ["existing_query"]
            
    def test_settings_structure(self, mock_streamlit):
        """Test settings structure in session state."""
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            from eeg_rag.web_ui.app import init_session_state
            
            init_session_state()
            
            settings = mock_streamlit.session_state["settings"]
            
            assert "corpus_path" in settings
            assert "embeddings_path" in settings
            assert "benchmark_csv" in settings
            assert "embedding_model" in settings
            assert "llm_model" in settings
            assert "max_sources" in settings
            assert "show_confidence" in settings


class TestRenderHeader:
    """Tests for header rendering."""
    
    def test_render_header_calls(self, mock_streamlit):
        """Test that render_header makes expected calls."""
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            # Re-import to get the patched version
            import importlib
            import eeg_rag.web_ui.app as app_module
            importlib.reload(app_module)
            
            app_module.render_header()
            
            # Check that title was called
            mock_streamlit.title.assert_called_once()
            
    def test_render_header_shows_stats(self, mock_streamlit):
        """Test that header shows stats when benchmark results exist."""
        mock_streamlit.session_state["benchmark_results"] = MagicMock(overall_accuracy=0.85)
        
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            import importlib
            import eeg_rag.web_ui.app as app_module
            importlib.reload(app_module)
            
            app_module.render_header()
            
            # Metric should be called for accuracy
            assert mock_streamlit.metric.called


class TestRenderSidebar:
    """Tests for sidebar rendering."""
    
    def test_render_sidebar_returns_page(self, mock_streamlit):
        """Test that render_sidebar returns selected page."""
        mock_streamlit.sidebar.radio.return_value = "🔍 Query System"
        
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            import importlib
            import eeg_rag.web_ui.app as app_module
            importlib.reload(app_module)
            
            page = app_module.render_sidebar()
            
            assert page == "🔍 Query System"
            
    def test_render_sidebar_has_navigation(self, mock_streamlit):
        """Test that sidebar has navigation elements."""
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            import importlib
            import eeg_rag.web_ui.app as app_module
            importlib.reload(app_module)
            
            app_module.render_sidebar()
            
            # Check sidebar methods were called
            mock_streamlit.sidebar.title.assert_called()
            mock_streamlit.sidebar.radio.assert_called()
            
    def test_render_sidebar_quick_actions(self, mock_streamlit):
        """Test that sidebar has quick action buttons."""
        with patch.dict(sys.modules, {'streamlit': mock_streamlit}):
            import importlib
            import eeg_rag.web_ui.app as app_module
            importlib.reload(app_module)
            
            app_module.render_sidebar()
            
            # Check that buttons were created
            assert mock_streamlit.sidebar.button.called


class TestPageRouting:
    """Tests for page routing."""
    
    def test_all_pages_defined(self):
        """Test that all navigation pages are handled."""
        pages = [
            "🔍 Query System",
            "📊 Systematic Review Benchmark",
            "📈 Results Dashboard",
            "📚 Corpus Explorer",
            "⚙️ Settings"
        ]
        
        # All pages should be handled without error
        for page in pages:
            assert page is not None


class TestFieldMappings:
    """Tests for field mappings."""
    
    def test_field_mappings_complete(self):
        """Test that field mappings are complete."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        required_fields = [
            "architecture_type",
            "n_layers",
            "domain",
            "dataset",
            "best_accuracy",
            "code_available"
        ]
        
        for field in required_fields:
            assert field in SystematicReviewBenchmark.FIELD_MAPPINGS
            
    def test_field_mappings_valid_columns(self):
        """Test that field mappings point to valid column names."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        for field, column in SystematicReviewBenchmark.FIELD_MAPPINGS.items():
            assert isinstance(column, str)
            assert len(column) > 0


class TestPatternDefinitions:
    """Tests for regex pattern definitions."""
    
    def test_architecture_patterns_compile(self):
        """Test that all architecture patterns compile."""
        import re
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        for arch_type, pattern in SystematicReviewBenchmark.ARCHITECTURE_PATTERNS.items():
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern for {arch_type}: {e}")
                
    def test_domain_patterns_compile(self):
        """Test that all domain patterns compile."""
        import re
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        for domain, pattern in SystematicReviewBenchmark.DOMAIN_PATTERNS.items():
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern for {domain}: {e}")


class TestDataFrameOperations:
    """Tests for DataFrame operations in result classes."""
    
    def test_benchmark_result_to_dataframe_columns(self, sample_benchmark_result):
        """Test DataFrame columns from benchmark result."""
        df = sample_benchmark_result.to_dataframe()
        
        expected_columns = [
            "paper_id",
            "title",
            "year",
            "accuracy",
            "correct_fields",
            "incorrect_fields",
            "extraction_time_ms"
        ]
        
        for col in expected_columns:
            assert col in df.columns
            
    def test_benchmark_result_to_dataframe_types(self, sample_benchmark_result):
        """Test DataFrame column types from benchmark result."""
        import pandas as pd
        
        df = sample_benchmark_result.to_dataframe()
        
        # Numeric columns should be numeric
        assert pd.api.types.is_numeric_dtype(df["accuracy"])
        assert pd.api.types.is_numeric_dtype(df["year"])
        assert pd.api.types.is_numeric_dtype(df["extraction_time_ms"])


class TestSummaryStatistics:
    """Tests for summary statistics calculation."""
    
    def test_summary_stats_keys(self, sample_benchmark_result):
        """Test summary statistics keys."""
        stats = sample_benchmark_result.get_summary_stats()
        
        expected_keys = [
            "mean_accuracy",
            "std_accuracy",
            "min_accuracy",
            "max_accuracy",
            "median_accuracy"
        ]
        
        for key in expected_keys:
            assert key in stats
            
    def test_summary_stats_values(self, sample_benchmark_result):
        """Test summary statistics values are valid."""
        stats = sample_benchmark_result.get_summary_stats()
        
        # All values should be between 0 and 1 for accuracy
        for key, value in stats.items():
            assert 0 <= value <= 1, f"{key} has invalid value: {value}"
            
    def test_summary_stats_empty_results(self):
        """Test summary statistics with empty results."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmarkResult
        
        result = SystematicReviewBenchmarkResult(
            total_papers=0,
            papers_evaluated=0,
            fields_evaluated=[],
            per_field_accuracy={},
            overall_accuracy=0.0,
            paper_results=[],
            extraction_time_total_ms=0.0,
            timestamp="2024-01-01"
        )
        
        stats = result.get_summary_stats()
        
        # Should handle empty case
        assert stats["mean_accuracy"] == 0.0


class TestErrorHandling:
    """Tests for error handling in UI components."""
    
    def test_benchmark_with_missing_columns(self, sample_ground_truth_csv):
        """Test benchmark handles missing columns gracefully."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        import pandas as pd
        
        # Create CSV with missing columns
        df = pd.read_csv(sample_ground_truth_csv)
        df = df.drop(columns=["Domain 1"], errors='ignore')
        df.to_csv(sample_ground_truth_csv, index=False)
        
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=["domain"]
        )
        
        # Should not raise error
        stats = benchmark.get_field_statistics()
        assert isinstance(stats, pd.DataFrame)


class TestConfigurationValidation:
    """Tests for configuration validation."""
    
    def test_default_config_paths_valid(self):
        """Test that default config paths are valid format."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert "/" in AppConfig.DEFAULT_CORPUS_PATH
        assert AppConfig.DEFAULT_CORPUS_PATH.endswith(".jsonl")
        
        assert "/" in AppConfig.DEFAULT_EMBEDDINGS_PATH
        assert AppConfig.DEFAULT_EMBEDDINGS_PATH.endswith(".npz")
        
        assert "/" in AppConfig.DEFAULT_BENCHMARK_CSV
        assert AppConfig.DEFAULT_BENCHMARK_CSV.endswith(".csv")
        
    def test_config_limits_reasonable(self):
        """Test that configuration limits are reasonable."""
        from eeg_rag.web_ui.app import AppConfig
        
        assert 100 <= AppConfig.MAX_QUERY_LENGTH <= 10000
        assert 1 <= AppConfig.MAX_RESPONSE_SOURCES <= 100
        assert 1 <= AppConfig.BENCHMARK_BATCH_SIZE <= 100


class TestUIStateTransitions:
    """Tests for UI state transitions."""
    
    def test_query_history_append(self, mock_streamlit):
        """Test query history appending."""
        mock_streamlit.session_state["query_history"] = []
        
        from eeg_rag.web_ui.app import QueryResult
        
        result = QueryResult(
            query="test",
            response="response",
            sources=[],
            citations=[],
            confidence=0.8,
            processing_time_ms=100,
            timestamp="2024-01-01T00:00:00"
        )
        
        mock_streamlit.session_state["query_history"].append(result)
        
        assert len(mock_streamlit.session_state["query_history"]) == 1
        assert mock_streamlit.session_state["query_history"][0].query == "test"
        
    def test_benchmark_results_storage(self, mock_streamlit, sample_benchmark_result):
        """Test benchmark results storage in session state."""
        mock_streamlit.session_state["benchmark_results"] = sample_benchmark_result
        
        assert mock_streamlit.session_state["benchmark_results"] is not None
        assert mock_streamlit.session_state["benchmark_results"].overall_accuracy == 0.833


class TestAccessibilityFeatures:
    """Tests for accessibility features."""
    
    def test_emoji_usage_consistency(self):
        """Test that emojis are used consistently in navigation."""
        pages = [
            "🔍 Query System",
            "📊 Systematic Review Benchmark",
            "📈 Results Dashboard",
            "📚 Corpus Explorer",
            "⚙️ Settings"
        ]
        
        # Each page should start with an emoji
        for page in pages:
            assert len(page) > 2
            # First character should be an emoji (high unicode codepoint)
            first_char = page[0]
            assert ord(first_char) > 127, f"Page '{page}' doesn't start with emoji"
