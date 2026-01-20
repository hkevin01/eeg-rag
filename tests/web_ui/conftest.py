"""
Test fixtures for web UI tests.

Provides mock data, sample papers, and test configurations
for comprehensive testing of the EEG-RAG web interface.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Any
from datetime import datetime


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ground_truth_data() -> Dict[str, List[Any]]:
    """Create sample ground truth data matching Roy et al. 2019 structure."""
    return {
        "Citation": ["Author2018a", "Author2019b", "Author2020c", "Author2017d", "Author2021e"],
        "Title": [
            "Deep Learning for EEG Seizure Detection using CNN",
            "LSTM-based Sleep Stage Classification from PSG",
            "CNN+RNN Hybrid Architecture for BCI Motor Imagery",
            "Autoencoder for EEG Emotion Recognition",
            "Deep Belief Network for Cognitive Load Assessment"
        ],
        "Year": [2018, 2019, 2020, 2017, 2021],
        "Authors": [
            "Smith et al.",
            "Jones et al.",
            "Wang et al.",
            "Garcia et al.",
            "Kim et al."
        ],
        "Architecture (clean)": ["CNN", "RNN", "CNN+RNN", "AE", "DBN"],
        "Layers (clean)": ["5", "3", "7", "4", "6"],
        "Domain 1": ["Epilepsy", "Sleep", "BCI", "Emotion", "Cognitive"],
        "Dataset name": ["CHB-MIT", "Sleep-EDF", "BCI-IV", "DEAP", "Custom"],
        "Intra/Inter subject": ["Intra", "Inter", "Both", "Intra", "Inter"],
        "Features (clean)": ["Raw", "Processed", "Raw", "Processed", "Raw"],
        "Results": ["95.2%", "87.3%", "82.1%", "78.5%", "91.8%"],
        "Code available": ["Yes", "No", "Yes", "N/M", "No"],
        "Dataset accessibility": ["Public", "Private", "Public", "Public", "Private"],
        "Preprocessing (clean)": ["Bandpass", "None", "ICA", "Artifact removal", "Normalization"],
        "Optimizer (clean)": ["Adam", "SGD", "RMSprop", "Adam", "Momentum"],
        "Activation function": ["ReLU", "tanh", "ELU", "Sigmoid", "ReLU"],
        "Regularization (clean)": ["Dropout", "L2", "Dropout+L2", "None", "BatchNorm"]
    }


@pytest.fixture
def sample_ground_truth_csv(sample_ground_truth_data, tmp_path) -> Path:
    """Create a temporary ground truth CSV file for testing."""
    df = pd.DataFrame(sample_ground_truth_data)
    csv_path = tmp_path / "test_ground_truth.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_paper_texts() -> Dict[str, str]:
    """Sample paper abstracts/texts for extraction testing."""
    return {
        "seizure_cnn": """
            This paper presents a convolutional neural network (CNN) for 
            automated seizure detection from EEG signals. We used a 5-layer 
            deep architecture trained on the CHB-MIT dataset. The model 
            achieved 95.2% accuracy with dropout regularization.
            Raw EEG signals were bandpass filtered before processing.
        """,
        "sleep_lstm": """
            We propose an LSTM-based recurrent neural network for automatic
            sleep stage classification from polysomnography data. Using the
            Sleep-EDF dataset, our 3-layer model achieved 87.3% accuracy.
            Features were extracted using wavelet transforms.
        """,
        "bci_hybrid": """
            A hybrid CNN-LSTM architecture for motor imagery BCI classification
            is presented. The 7-layer model processes raw EEG from BCI Competition IV
            dataset, achieving 82.1% accuracy. Both intra and inter-subject
            evaluations were performed.
        """,
        "emotion_ae": """
            An autoencoder (AE) approach for emotion recognition from EEG is
            proposed. The 4-layer model was tested on the DEAP dataset with
            processed features, achieving 78.5% valence classification accuracy.
        """,
        "cognitive_dbn": """
            Deep Belief Networks (DBN) are applied to cognitive workload 
            assessment. Our 6-layer DBN trained on custom EEG data achieved
            91.8% accuracy for mental workload classification.
        """
    }


@pytest.fixture
def sample_query_result_data() -> Dict[str, Any]:
    """Sample query result data for testing."""
    return {
        "query": "What CNNs are used for seizure detection?",
        "response": "Based on the literature, CNNs with 5-7 layers achieve best results...",
        "sources": [
            {"title": "EEGNet for Seizure Detection", "pmid": "12345678", "year": 2019, "relevance": 0.95},
            {"title": "DeepConvNet Analysis", "pmid": "23456789", "year": 2018, "relevance": 0.88}
        ],
        "citations": ["PMID:12345678", "PMID:23456789"],
        "confidence": 0.85,
        "processing_time_ms": 250.5,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# Mock Objects Fixtures
# =============================================================================

@pytest.fixture
def mock_streamlit():
    """Create mock Streamlit module for testing."""
    mock_st = MagicMock()
    
    # Mock session state as a dict-like object
    mock_st.session_state = {}
    
    # Mock columns to return list of mock contexts
    def mock_columns(spec):
        if isinstance(spec, list):
            return [MagicMock() for _ in spec]
        return [MagicMock() for _ in range(spec)]
    
    mock_st.columns = mock_columns
    mock_st.spinner = MagicMock()
    mock_st.progress = MagicMock()
    mock_st.text_area = MagicMock(return_value="test query")
    mock_st.text_input = MagicMock(return_value="test input")
    mock_st.button = MagicMock(return_value=False)
    mock_st.selectbox = MagicMock(return_value="option1")
    mock_st.multiselect = MagicMock(return_value=["field1", "field2"])
    mock_st.slider = MagicMock(return_value=5)
    mock_st.checkbox = MagicMock(return_value=True)
    mock_st.number_input = MagicMock(return_value=50)
    mock_st.dataframe = MagicMock()
    mock_st.bar_chart = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.header = MagicMock()
    mock_st.title = MagicMock()
    mock_st.success = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.info = MagicMock()
    mock_st.download_button = MagicMock()
    mock_st.expander = MagicMock()
    mock_st.container = MagicMock()
    mock_st.set_page_config = MagicMock()
    mock_st.radio = MagicMock(return_value="🔍 Query System")
    mock_st.select_slider = MagicMock(return_value="Medium")
    mock_st.rerun = MagicMock()
    mock_st.code = MagicMock()
    mock_st.empty = MagicMock(return_value=MagicMock())
    mock_st.caption = MagicMock()
    
    # Mock sidebar
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.title = MagicMock()
    mock_st.sidebar.radio = MagicMock(return_value="🔍 Query System")
    mock_st.sidebar.button = MagicMock(return_value=False)
    mock_st.sidebar.markdown = MagicMock()
    mock_st.sidebar.success = MagicMock()
    mock_st.sidebar.download_button = MagicMock()
    
    return mock_st


@pytest.fixture
def mock_async_loop():
    """Create mock event loop for async operations."""
    loop = MagicMock()
    loop.run_until_complete = MagicMock(side_effect=lambda x: x)
    return loop


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def benchmark_instance(sample_ground_truth_csv):
    """Create a SystematicReviewBenchmark instance for testing."""
    from eeg_rag.web_ui.app import SystematicReviewBenchmark
    
    return SystematicReviewBenchmark(
        ground_truth_csv=str(sample_ground_truth_csv),
        extraction_fields=["architecture_type", "domain", "best_accuracy", "code_available"],
        strict_matching=False
    )


@pytest.fixture
def strict_benchmark_instance(sample_ground_truth_csv):
    """Create a SystematicReviewBenchmark with strict matching."""
    from eeg_rag.web_ui.app import SystematicReviewBenchmark
    
    return SystematicReviewBenchmark(
        ground_truth_csv=str(sample_ground_truth_csv),
        extraction_fields=["architecture_type", "domain"],
        strict_matching=True
    )


# =============================================================================
# Query Engine Fixtures  
# =============================================================================

@pytest.fixture
def query_engine():
    """Create EEGQueryEngine instance for testing."""
    from eeg_rag.web_ui.app import EEGQueryEngine
    return EEGQueryEngine()


# =============================================================================
# Data Classes Fixtures
# =============================================================================

@pytest.fixture
def sample_extraction_result():
    """Create sample ExtractionResult for testing."""
    from eeg_rag.web_ui.app import ExtractionResult
    
    return ExtractionResult(
        paper_id="test_paper_001",
        field_name="architecture_type",
        extracted_value="CNN",
        ground_truth_value="CNN",
        is_correct=True,
        confidence=0.95,
        extraction_method="regex",
        error_type=None
    )


@pytest.fixture
def sample_paper_extraction_result():
    """Create sample PaperExtractionResult for testing."""
    from eeg_rag.web_ui.app import ExtractionResult, PaperExtractionResult
    
    return PaperExtractionResult(
        paper_id="test_paper_001",
        title="Test Paper on EEG Analysis",
        year=2021,
        authors="Test Author et al.",
        field_results={
            "architecture_type": ExtractionResult(
                paper_id="test_paper_001",
                field_name="architecture_type",
                extracted_value="CNN",
                ground_truth_value="CNN",
                is_correct=True,
                confidence=0.95
            ),
            "domain": ExtractionResult(
                paper_id="test_paper_001",
                field_name="domain",
                extracted_value="Sleep",
                ground_truth_value="Epilepsy",
                is_correct=False,
                confidence=0.6,
                error_type="mismatch"
            )
        },
        overall_accuracy=0.5,
        extraction_time_ms=15.3
    )


@pytest.fixture
def sample_benchmark_result():
    """Create sample SystematicReviewBenchmarkResult for testing."""
    from eeg_rag.web_ui.app import (
        SystematicReviewBenchmarkResult, 
        PaperExtractionResult,
        ExtractionResult
    )
    
    paper_results = [
        PaperExtractionResult(
            paper_id=f"paper_{i}",
            title=f"Test Paper {i}",
            year=2020 + i,
            authors=f"Author {i}",
            field_results={
                "architecture_type": ExtractionResult(
                    paper_id=f"paper_{i}",
                    field_name="architecture_type",
                    extracted_value="CNN",
                    ground_truth_value="CNN",
                    is_correct=True,
                    confidence=0.9
                )
            },
            overall_accuracy=0.8 + (i * 0.05),
            extraction_time_ms=10.0 + i
        )
        for i in range(3)
    ]
    
    return SystematicReviewBenchmarkResult(
        total_papers=100,
        papers_evaluated=3,
        fields_evaluated=["architecture_type", "domain", "best_accuracy"],
        per_field_accuracy={"architecture_type": 0.9, "domain": 0.85, "best_accuracy": 0.75},
        overall_accuracy=0.833,
        paper_results=paper_results,
        extraction_time_total_ms=150.5,
        timestamp="2024-01-15 10:30:00",
        error_analysis={"mismatch": 5, "missing_value": 2}
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config():
    """Test configuration values."""
    return {
        "corpus_path": "data/test_corpus/test.jsonl",
        "embeddings_path": "data/test_corpus/embeddings.npz",
        "benchmark_csv": "data/test/ground_truth.csv",
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gpt-4",
        "max_sources": 5,
        "show_confidence": True
    }


@pytest.fixture
def empty_csv(tmp_path) -> Path:
    """Create an empty CSV file for edge case testing."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("Column1,Column2\n")
    return csv_path


@pytest.fixture
def malformed_csv(tmp_path) -> Path:
    """Create a malformed CSV file for error handling testing."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("col1,col2\nval1,val2,extra\nval3\n")
    return csv_path


# =============================================================================
# Architecture Pattern Test Data
# =============================================================================

@pytest.fixture
def architecture_test_cases() -> List[Dict[str, str]]:
    """Test cases for architecture extraction."""
    return [
        {"text": "We used a CNN for classification", "expected": "CNN"},
        {"text": "convolutional neural network approach", "expected": "CNN"},
        {"text": "The ConvNet architecture", "expected": "CNN"},
        {"text": "LSTM network was applied", "expected": "RNN"},
        {"text": "recurrent neural network with GRU", "expected": "RNN"},
        {"text": "CNN-LSTM hybrid model", "expected": "CNN+RNN"},
        {"text": "CRNN architecture", "expected": "CNN+RNN"},
        {"text": "ConvLSTM for temporal modeling", "expected": "CNN+RNN"},
        {"text": "autoencoder for feature learning", "expected": "AE"},
        {"text": "Variational AE (VAE) was used", "expected": "AE"},
        {"text": "deep belief network", "expected": "DBN"},
        {"text": "fully connected neural network", "expected": "FC"},
        {"text": "MLP classifier", "expected": "FC"},
        {"text": "transformer architecture with attention", "expected": "Transformer"},
        {"text": "GAN for data augmentation", "expected": "GAN"},
        {"text": "RBM for pretraining", "expected": "RBM"},
        {"text": "Some other method", "expected": "Unknown"},
    ]


@pytest.fixture
def domain_test_cases() -> List[Dict[str, str]]:
    """Test cases for domain extraction."""
    return [
        {"text": "epilepsy detection and seizure prediction", "expected": "Epilepsy"},
        {"text": "ictal and interictal EEG patterns", "expected": "Epilepsy"},
        {"text": "sleep stage classification", "expected": "Sleep"},
        {"text": "polysomnography analysis", "expected": "Sleep"},
        {"text": "BCI motor imagery", "expected": "BCI"},
        {"text": "P300 speller interface", "expected": "BCI"},
        {"text": "SSVEP-based brain-computer interface", "expected": "BCI"},
        {"text": "emotion recognition from EEG", "expected": "Emotion"},
        {"text": "valence and arousal classification", "expected": "Emotion"},
        {"text": "cognitive workload assessment", "expected": "Cognitive"},
        {"text": "mental attention monitoring", "expected": "Cognitive"},
        {"text": "random text without domain", "expected": "Other"},
    ]


@pytest.fixture
def accuracy_test_cases() -> List[Dict[str, Any]]:
    """Test cases for accuracy extraction."""
    return [
        {"text": "achieved 95.2%", "expected": 95.2},
        {"text": "accuracy: 87.5", "expected": 87.5},
        {"text": "acc: 92%", "expected": 92.0},
        {"text": "0.95 accuracy", "expected": 95.0},
        {"text": "accuracy of 88.3%", "expected": 88.3},
        {"text": "no accuracy reported", "expected": None},
        {"text": "", "expected": None},
    ]


@pytest.fixture
def layers_test_cases() -> List[Dict[str, Any]]:
    """Test cases for layer extraction."""
    return [
        {"text": "5-layer CNN", "expected": 5},
        {"text": "7 layers deep", "expected": 7},
        {"text": "layers: 3", "expected": 3},
        {"text": "12 conv layers", "expected": 12},
        {"text": "no layer info", "expected": None},
    ]
