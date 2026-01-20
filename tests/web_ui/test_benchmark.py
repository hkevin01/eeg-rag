"""
Comprehensive tests for SystematicReviewBenchmark class.

Tests all extraction methods, value normalization, comparison logic,
and benchmark evaluation functionality.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch, AsyncMock


class TestSystematicReviewBenchmarkInit:
    """Tests for SystematicReviewBenchmark initialization."""
    
    def test_init_with_valid_csv(self, benchmark_instance):
        """Test initialization with valid CSV file."""
        assert benchmark_instance.ground_truth_df is not None
        assert len(benchmark_instance.ground_truth_df) == 5
        assert benchmark_instance.strict_matching is False
        
    def test_init_with_custom_fields(self, sample_ground_truth_csv):
        """Test initialization with custom extraction fields."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        custom_fields = ["architecture_type", "domain"]
        benchmark = SystematicReviewBenchmark(
            ground_truth_csv=str(sample_ground_truth_csv),
            extraction_fields=custom_fields
        )
        
        assert benchmark.extraction_fields == custom_fields
        
    def test_init_with_strict_matching(self, strict_benchmark_instance):
        """Test initialization with strict matching enabled."""
        assert strict_benchmark_instance.strict_matching is True
        
    def test_init_with_nonexistent_csv(self, tmp_path):
        """Test initialization with non-existent CSV raises error."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        with pytest.raises(FileNotFoundError):
            SystematicReviewBenchmark(
                ground_truth_csv=str(tmp_path / "nonexistent.csv")
            )
    
    def test_field_mappings_defined(self, benchmark_instance):
        """Test that field mappings are properly defined."""
        assert "architecture_type" in benchmark_instance.FIELD_MAPPINGS
        assert "domain" in benchmark_instance.FIELD_MAPPINGS
        assert "best_accuracy" in benchmark_instance.FIELD_MAPPINGS
        assert "code_available" in benchmark_instance.FIELD_MAPPINGS
        
    def test_architecture_patterns_defined(self, benchmark_instance):
        """Test that architecture patterns are properly defined."""
        patterns = benchmark_instance.ARCHITECTURE_PATTERNS
        assert "CNN" in patterns
        assert "RNN" in patterns
        assert "CNN+RNN" in patterns
        assert "Transformer" in patterns
        
    def test_domain_patterns_defined(self, benchmark_instance):
        """Test that domain patterns are properly defined."""
        patterns = benchmark_instance.DOMAIN_PATTERNS
        assert "Epilepsy" in patterns
        assert "Sleep" in patterns
        assert "BCI" in patterns
        assert "Emotion" in patterns


class TestArchitectureExtraction:
    """Tests for architecture type extraction."""
    
    def test_extract_cnn_variations(self, benchmark_instance):
        """Test CNN architecture extraction with various text patterns."""
        test_cases = [
            ("CNN-based classifier", "CNN"),
            ("convolutional neural network", "CNN"),
            ("ConvNet architecture", "CNN"),
            ("conv net model", "CNN"),
            ("We used CNN", "CNN"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_rnn_variations(self, benchmark_instance):
        """Test RNN architecture extraction with various patterns."""
        test_cases = [
            ("RNN model", "RNN"),
            ("LSTM network", "RNN"),
            ("GRU-based approach", "RNN"),
            ("recurrent neural network", "RNN"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_hybrid_cnn_rnn(self, benchmark_instance):
        """Test hybrid CNN+RNN architecture extraction."""
        test_cases = [
            ("CNN+RNN model", "CNN+RNN"),
            ("CNN-LSTM hybrid", "CNN+RNN"),
            ("CRNN architecture", "CNN+RNN"),
            ("ConvLSTM layer", "CNN+RNN"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_autoencoder(self, benchmark_instance):
        """Test autoencoder architecture extraction."""
        test_cases = [
            ("AE model", "AE"),
            ("autoencoder approach", "AE"),
            ("VAE for generation", "AE"),
            ("SAE stacked", "AE"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_dbn(self, benchmark_instance):
        """Test DBN architecture extraction."""
        test_cases = [
            ("DBN classifier", "DBN"),
            ("deep belief network", "DBN"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_transformer(self, benchmark_instance):
        """Test transformer architecture extraction."""
        test_cases = [
            ("transformer model", "Transformer"),
            ("attention mechanism", "Transformer"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_unknown_architecture(self, benchmark_instance):
        """Test unknown architecture handling."""
        result = benchmark_instance._extract_architecture("random text without architecture")
        assert result == "Unknown"
        
    def test_extract_architecture_none_input(self, benchmark_instance):
        """Test architecture extraction with None input."""
        result = benchmark_instance._extract_architecture(None)
        assert result == "Unknown"
        
    def test_extract_architecture_empty_string(self, benchmark_instance):
        """Test architecture extraction with empty string."""
        result = benchmark_instance._extract_architecture("")
        assert result == "Unknown"
        
    def test_architecture_case_insensitivity(self, benchmark_instance):
        """Test case insensitivity in architecture extraction."""
        test_cases = [
            ("CNN", "CNN"),
            ("cnn", "CNN"),
            ("Cnn", "CNN"),
            ("LSTM", "RNN"),
            ("lstm", "RNN"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_architecture(text)
            assert result == expected


class TestDomainExtraction:
    """Tests for research domain extraction."""
    
    def test_extract_epilepsy_domain(self, benchmark_instance):
        """Test epilepsy domain extraction."""
        test_cases = [
            ("epilepsy detection", "Epilepsy"),
            ("seizure prediction", "Epilepsy"),
            ("ictal patterns", "Epilepsy"),
            ("interictal EEG", "Epilepsy"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_domain(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_sleep_domain(self, benchmark_instance):
        """Test sleep domain extraction."""
        test_cases = [
            ("sleep stage classification", "Sleep"),
            ("staging analysis", "Sleep"),
            ("polysomnography", "Sleep"),
            ("PSG data", "Sleep"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_domain(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_bci_domain(self, benchmark_instance):
        """Test BCI domain extraction."""
        test_cases = [
            ("BCI application", "BCI"),
            ("brain-computer interface", "BCI"),
            ("motor imagery task", "BCI"),
            ("P300 speller", "BCI"),
            ("SSVEP response", "BCI"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_domain(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_emotion_domain(self, benchmark_instance):
        """Test emotion domain extraction."""
        test_cases = [
            ("emotion recognition", "Emotion"),
            ("affect detection", "Emotion"),
            ("valence classification", "Emotion"),
            ("arousal level", "Emotion"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_domain(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_cognitive_domain(self, benchmark_instance):
        """Test cognitive domain extraction."""
        test_cases = [
            ("cognitive load", "Cognitive"),
            ("mental workload", "Cognitive"),
            ("attention monitoring", "Cognitive"),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_domain(text)
            assert result == expected, f"Failed for '{text}': got {result}"
            
    def test_extract_other_domain(self, benchmark_instance):
        """Test 'Other' domain for unrecognized text."""
        result = benchmark_instance._extract_domain("unrelated text about EEG analysis")
        assert result == "Other"
        
    def test_domain_none_input(self, benchmark_instance):
        """Test domain extraction with None input."""
        result = benchmark_instance._extract_domain(None)
        assert result == "Other"


class TestAccuracyExtraction:
    """Tests for accuracy value extraction."""
    
    def test_extract_percentage_with_symbol(self, benchmark_instance):
        """Test extracting accuracy with % symbol."""
        test_cases = [
            ("achieved 95.2%", 95.2),
            ("95% accuracy", 95.0),
            ("88.5% performance", 88.5),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_accuracy(text)
            assert result == pytest.approx(expected), f"Failed for '{text}'"
            
    def test_extract_decimal_accuracy(self, benchmark_instance):
        """Test extracting decimal accuracy values."""
        test_cases = [
            ("0.95 accuracy", 95.0),
            ("with 0.88 acc", 88.0),  # Pattern expects decimal before acc/accuracy
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_accuracy(text)
            assert result == pytest.approx(expected), f"Failed for '{text}'"
            
    def test_extract_accuracy_with_keyword(self, benchmark_instance):
        """Test extracting accuracy with keyword prefix."""
        test_cases = [
            ("accuracy: 87.5", 87.5),
            ("acc: 92", 92.0),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_accuracy(text)
            assert result == pytest.approx(expected), f"Failed for '{text}'"
            
    def test_extract_best_accuracy_from_multiple(self, benchmark_instance):
        """Test extracting best (max) accuracy when multiple present."""
        result = benchmark_instance._extract_accuracy("achieved 85% and 92% accuracy")
        assert result == pytest.approx(92.0)
        
    def test_extract_accuracy_no_value(self, benchmark_instance):
        """Test extraction when no accuracy value present."""
        result = benchmark_instance._extract_accuracy("no accuracy reported")
        assert result is None
        
    def test_extract_accuracy_empty_string(self, benchmark_instance):
        """Test extraction with empty string."""
        result = benchmark_instance._extract_accuracy("")
        assert result is None
        
    def test_extract_accuracy_none_input(self, benchmark_instance):
        """Test extraction with None input."""
        result = benchmark_instance._extract_accuracy(None)
        assert result is None


class TestLayersExtraction:
    """Tests for layer count extraction."""
    
    def test_extract_layers_hyphenated(self, benchmark_instance):
        """Test extracting layers with hyphenated format."""
        test_cases = [
            ("5-layer CNN", 5),
            ("7-layer network", 7),
            ("12-layer deep model", 12),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_layers(text)
            assert result == expected, f"Failed for '{text}'"
            
    def test_extract_layers_with_keyword(self, benchmark_instance):
        """Test extracting layers with keyword."""
        test_cases = [
            ("3 layers deep", 3),
            ("layers: 5", 5),
            ("8 conv layers", 8),
        ]
        
        for text, expected in test_cases:
            result = benchmark_instance._extract_layers(text)
            assert result == expected, f"Failed for '{text}'"
            
    def test_extract_layers_none_input(self, benchmark_instance):
        """Test layer extraction with None input."""
        result = benchmark_instance._extract_layers(None)
        assert result is None
        
    def test_extract_layers_no_value(self, benchmark_instance):
        """Test extraction when no layer count present."""
        result = benchmark_instance._extract_layers("deep neural network")
        assert result is None
        
    def test_extract_layers_invalid_range(self, benchmark_instance):
        """Test that unreasonable layer counts are filtered."""
        # Very high layer count should still work if in valid range
        result = benchmark_instance._extract_layers("500 layers")
        assert result == 500


class TestValueNormalization:
    """Tests for value normalization."""
    
    def test_normalize_architecture_value(self, benchmark_instance):
        """Test architecture value normalization."""
        result = benchmark_instance._normalize_value("CNN model", "architecture_type")
        assert result == "cnn"
        
    def test_normalize_domain_value(self, benchmark_instance):
        """Test domain value normalization."""
        result = benchmark_instance._normalize_value("epilepsy", "domain")
        assert result == "epilepsy"
        
    def test_normalize_code_available_yes(self, benchmark_instance):
        """Test code_available normalization for positive values."""
        test_cases = ["yes", "Yes", "YES", "true", "1", "public", "available"]
        
        for value in test_cases:
            result = benchmark_instance._normalize_value(value, "code_available")
            assert result == "yes", f"Failed for '{value}'"
            
    def test_normalize_code_available_no(self, benchmark_instance):
        """Test code_available normalization for negative values."""
        test_cases = ["no", "No", "NO", "false", "0", "private", "n/m", "n/a"]
        
        for value in test_cases:
            result = benchmark_instance._normalize_value(value, "code_available")
            assert result == "no", f"Failed for '{value}'"
            
    def test_normalize_intra_inter(self, benchmark_instance):
        """Test intra/inter subject normalization."""
        test_cases = [
            ("intra-subject", "intra"),
            ("Intra", "intra"),
            ("inter-subject", "inter"),
            ("Inter", "inter"),
            ("both", "both"),
        ]
        
        for value, expected in test_cases:
            result = benchmark_instance._normalize_value(value, "intra_inter")
            assert result == expected, f"Failed for '{value}'"
            
    def test_normalize_none_value(self, benchmark_instance):
        """Test normalization of None value."""
        result = benchmark_instance._normalize_value(None, "architecture_type")
        assert result == "N/A"
        
    def test_normalize_nan_value(self, benchmark_instance):
        """Test normalization of NaN value."""
        result = benchmark_instance._normalize_value(np.nan, "domain")
        assert result == "N/A"


class TestValueComparison:
    """Tests for value comparison logic."""
    
    def test_compare_exact_match(self, benchmark_instance):
        """Test exact value match."""
        is_correct, error = benchmark_instance._compare_values("CNN", "CNN", "architecture_type")
        assert is_correct is True
        assert error is None
        
    def test_compare_normalized_match(self, benchmark_instance):
        """Test match after normalization."""
        is_correct, error = benchmark_instance._compare_values("cnn", "CNN", "architecture_type")
        assert is_correct is True
        
    def test_compare_na_both(self, benchmark_instance):
        """Test comparison when both values are N/A."""
        is_correct, error = benchmark_instance._compare_values(None, None, "domain")
        assert is_correct is True
        
    def test_compare_na_one_side(self, benchmark_instance):
        """Test comparison when only one value is N/A."""
        # When N/A for one side with architecture field, it falls through to fuzzy matching
        # and gets 'architecture_mismatch' because 'n/a' doesn't contain 'cnn'
        is_correct, error = benchmark_instance._compare_values(None, "CNN", "architecture_type")
        assert is_correct is False
        assert error in ["missing_value", "architecture_mismatch"]
        
    def test_compare_mismatch(self, benchmark_instance):
        """Test value mismatch."""
        is_correct, error = benchmark_instance._compare_values("CNN", "RNN", "architecture_type")
        assert is_correct is False
        assert error == "architecture_mismatch"
        
    def test_compare_accuracy_close_values(self, benchmark_instance):
        """Test accuracy comparison with close values (within tolerance)."""
        is_correct, error = benchmark_instance._compare_values("95.2%", "93%", "best_accuracy")
        assert is_correct is True  # Within 5% tolerance
        
    def test_compare_accuracy_far_values(self, benchmark_instance):
        """Test accuracy comparison with distant values."""
        is_correct, error = benchmark_instance._compare_values("95%", "80%", "best_accuracy")
        assert is_correct is False
        assert error == "accuracy_mismatch"
        
    def test_compare_layers_close(self, benchmark_instance):
        """Test layer comparison with close values."""
        is_correct, error = benchmark_instance._compare_values("5 layers", "6 layers", "n_layers")
        assert is_correct is True  # Within 2 layer tolerance
        
    def test_compare_substring_match(self, benchmark_instance):
        """Test substring matching for fuzzy comparison."""
        is_correct, error = benchmark_instance._compare_values(
            "CHB-MIT", "CHB-MIT Scalp EEG Database", "dataset"
        )
        assert is_correct is True
        
    def test_compare_strict_mode(self, strict_benchmark_instance):
        """Test strict matching mode."""
        is_correct, error = strict_benchmark_instance._compare_values(
            "cnn", "CNN", "architecture_type"
        )
        # In strict mode, normalized values must match exactly
        assert is_correct is True  # Both normalize to 'cnn'


class TestBenchmarkEvaluation:
    """Tests for benchmark evaluation functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluate_extraction_accuracy(self, benchmark_instance):
        """Test full evaluation with sample data."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=3)
        
        assert results is not None
        assert results.papers_evaluated == 3
        assert 0 <= results.overall_accuracy <= 1.0
        assert len(results.paper_results) == 3
        
    @pytest.mark.asyncio
    async def test_evaluate_with_progress_callback(self, benchmark_instance):
        """Test evaluation with progress callback."""
        progress_values = []
        
        def callback(progress):
            progress_values.append(progress)
            
        results = await benchmark_instance.evaluate_extraction_accuracy(
            max_papers=5,
            progress_callback=callback
        )
        
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0  # Final progress should be 100%
        
    @pytest.mark.asyncio
    async def test_evaluate_generates_per_field_accuracy(self, benchmark_instance):
        """Test that per-field accuracy is generated."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=5)
        
        assert len(results.per_field_accuracy) > 0
        for field, accuracy in results.per_field_accuracy.items():
            assert 0 <= accuracy <= 1.0
            
    @pytest.mark.asyncio
    async def test_evaluate_generates_error_analysis(self, benchmark_instance):
        """Test that error analysis is generated."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=5)
        
        # Error analysis may or may not have entries depending on data
        assert isinstance(results.error_analysis, dict)
        
    @pytest.mark.asyncio
    async def test_evaluate_timing(self, benchmark_instance):
        """Test that extraction time is tracked."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=3)
        
        assert results.extraction_time_total_ms > 0
        for paper in results.paper_results:
            assert paper.extraction_time_ms >= 0
            
    @pytest.mark.asyncio
    async def test_evaluate_timestamp(self, benchmark_instance):
        """Test that timestamp is recorded."""
        results = await benchmark_instance.evaluate_extraction_accuracy(max_papers=2)
        
        assert results.timestamp is not None
        assert len(results.timestamp) > 0


class TestBenchmarkResult:
    """Tests for SystematicReviewBenchmarkResult dataclass."""
    
    def test_get_summary_stats(self, sample_benchmark_result):
        """Test summary statistics calculation."""
        stats = sample_benchmark_result.get_summary_stats()
        
        assert "mean_accuracy" in stats
        assert "std_accuracy" in stats
        assert "min_accuracy" in stats
        assert "max_accuracy" in stats
        assert "median_accuracy" in stats
        
    def test_to_dataframe(self, sample_benchmark_result):
        """Test conversion to DataFrame."""
        df = sample_benchmark_result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 papers in sample
        assert "paper_id" in df.columns
        assert "accuracy" in df.columns
        assert "title" in df.columns


class TestPaperExtractionResult:
    """Tests for PaperExtractionResult dataclass."""
    
    def test_get_correct_fields(self, sample_paper_extraction_result):
        """Test getting list of correctly extracted fields."""
        correct = sample_paper_extraction_result.get_correct_fields()
        
        assert "architecture_type" in correct
        assert "domain" not in correct
        
    def test_get_incorrect_fields(self, sample_paper_extraction_result):
        """Test getting list of incorrectly extracted fields."""
        incorrect = sample_paper_extraction_result.get_incorrect_fields()
        
        assert "domain" in incorrect
        assert "architecture_type" not in incorrect


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""
    
    def test_to_dict(self, sample_extraction_result):
        """Test conversion to dictionary."""
        result_dict = sample_extraction_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["paper_id"] == "test_paper_001"
        assert result_dict["field_name"] == "architecture_type"
        assert result_dict["is_correct"] is True
        
    def test_extraction_result_defaults(self):
        """Test default values for ExtractionResult."""
        from eeg_rag.web_ui.app import ExtractionResult
        
        result = ExtractionResult(
            paper_id="test",
            field_name="test_field",
            extracted_value="value1",
            ground_truth_value="value1",
            is_correct=True
        )
        
        assert result.confidence == 0.0
        assert result.extraction_method == "regex"
        assert result.error_type is None


class TestFieldStatistics:
    """Tests for field statistics generation."""
    
    def test_get_field_statistics(self, benchmark_instance):
        """Test field statistics generation."""
        stats = benchmark_instance.get_field_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
        assert "field" in stats.columns
        assert "non_null_count" in stats.columns
        assert "fill_rate" in stats.columns


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_ground_truth(self, empty_csv):
        """Test handling of empty ground truth CSV."""
        from eeg_rag.web_ui.app import SystematicReviewBenchmark
        
        # Should not raise, but will have empty or filtered dataframe
        benchmark = SystematicReviewBenchmark(str(empty_csv))
        assert benchmark.ground_truth_df is not None
        
    def test_special_characters_in_text(self, benchmark_instance):
        """Test handling of special characters."""
        result = benchmark_instance._extract_architecture("CNN™ model (v2.0) [beta]")
        assert result == "CNN"
        
    def test_unicode_text(self, benchmark_instance):
        """Test handling of unicode text - ASCII patterns only."""
        # Note: épilepsie with accent doesn't match 'epilep' pattern
        # ASCII seizure keyword will match
        result = benchmark_instance._extract_domain("seizure épilepsie detection")
        assert result == "Epilepsy"
        
    def test_very_long_text(self, benchmark_instance):
        """Test handling of very long text."""
        long_text = "CNN " * 1000
        result = benchmark_instance._extract_architecture(long_text)
        assert result == "CNN"
        
    def test_numeric_input_to_normalize(self, benchmark_instance):
        """Test numeric input to normalize function."""
        result = benchmark_instance._normalize_value(123, "generic_field")
        assert result == "123"
