"""Pytest configuration for systematic review tests."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def roy_et_al_metadata():
    """Metadata about the Roy et al. (2019) systematic review."""
    return {
        'title': 'Deep learning-based electroencephalography analysis: a systematic review',
        'pmid': '31151119',
        'doi': '10.1088/1741-2552/ab260c',
        'year': 2019,
        'papers_reviewed': 154,
        'time_period': '2010-2018',
        'github': 'https://github.com/hubertjb/dl-eeg-review',
        'spreadsheet': 'https://docs.google.com/spreadsheets/d/1smpU0WSlSq-Al2u_QH3djGn68mTuHlth2fNJTrD3wa8/',
        'data_items': [
            'architecture_type', 'n_layers', 'domain', 'dataset',
            'n_subjects', 'n_channels', 'sampling_rate', 'input_type',
            'cross_validation', 'best_accuracy', 'baseline_accuracy',
            'code_available', 'data_available'
        ]
    }


@pytest.fixture
def mock_pubmed_response():
    """Mock PubMed API response for testing."""
    return {
        '31151119': {
            'title': 'Deep learning-based electroencephalography analysis: a systematic review',
            'authors': ['Roy Y', 'Banville H', 'Albuquerque I', 'Gramfort A', 'Falk TH', 'Faubert J'],
            'journal': 'J Neural Eng',
            'year': 2019,
            'abstract': 'Context: Electroencephalography (EEG) is a complex signal...'
        }
    }


@pytest.fixture
def extraction_accuracy_thresholds():
    """Accuracy thresholds for different extraction fields."""
    return {
        'architecture_type': 0.90,  # High accuracy expected
        'domain': 0.85,
        'n_layers': 0.80,
        'n_subjects': 0.75,
        'n_channels': 0.80,
        'sampling_rate': 0.85,
        'best_accuracy': 0.70,  # Harder to extract consistently
        'code_available': 0.90,
        'cross_validation': 0.75,
        'input_type': 0.70
    }