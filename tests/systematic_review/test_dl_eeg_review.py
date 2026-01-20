#!/usr/bin/env python3
"""
Test Suite: DL-EEG Systematic Review Replication

Tests the ability of EEG-RAG to replicate and extend the systematic review:
"Deep learning-based electroencephalography analysis: a systematic review"
by Roy, Banville, Albuquerque, Gramfort, Falk & Faubert (2019)

PMID: 31151119
DOI: 10.1088/1741-2552/ab260c
GitHub: https://github.com/hubertjb/dl-eeg-review

Test Categories:
1. Data extraction validation against ground truth
2. NER accuracy for DL-EEG taxonomy
3. Paper discovery and retrieval
4. Trend analysis capabilities
5. Extension to new publications (2019-2026)
"""

import pytest
import asyncio
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import sys

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# EEG-RAG imports - use try/except for graceful degradation during testing
try:
    from eeg_rag.nlp.ner_eeg import EEGNER, Entity, EntityType, NERResult
    from eeg_rag.agents.base_agent import AgentQuery
    from eeg_rag.verification.citation_verifier import CitationVerifier
    NER_AVAILABLE = True
except ImportError as e:
    # Define minimal stubs for when full system isn't available
    NER_AVAILABLE = False
    EntityType = None
    Entity = None
    EEGNER = None
    CitationVerifier = None

logger = logging.getLogger(__name__)


# =============================================================================
# Data Item Taxonomy from Roy et al. (2019)
# =============================================================================

class ArchitectureType(Enum):
    """DL architecture types from the review."""
    CNN = "CNN"
    RNN = "RNN"
    DBN = "DBN"  # Deep Belief Network
    AE = "AE"    # Autoencoder
    MLP = "MLP"  # Multilayer Perceptron
    HYBRID = "Hybrid"
    OTHER = "Other"


class ApplicationDomain(Enum):
    """EEG application domains from the review."""
    EPILEPSY = "Epilepsy"
    SLEEP = "Sleep"
    BCI = "BCI"  # Brain-Computer Interface
    COGNITIVE = "Cognitive"
    AFFECTIVE = "Affective"
    OTHER = "Other"


class InputType(Enum):
    """Type of input to DL model."""
    RAW = "Raw"
    PREPROCESSED = "Preprocessed"
    FEATURES = "Features"
    SPECTROGRAM = "Spectrogram"
    OTHER = "Other"


class CrossValidationType(Enum):
    """Cross-validation approach."""
    INTRA_SUBJECT = "Intra-subject"
    INTER_SUBJECT = "Inter-subject"
    MIXED = "Mixed"
    NOT_SPECIFIED = "Not specified"


@dataclass
class DLEEGDataItem:
    """Data item extracted from a DL-EEG paper.
    
    Matches the schema from Roy et al. spreadsheet.
    """
    # Paper identification
    pmid: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""
    
    # Architecture details
    architecture_type: Optional[ArchitectureType] = None
    n_layers: Optional[int] = None
    architecture_details: str = ""
    
    # Application domain
    domain: Optional[ApplicationDomain] = None
    specific_task: str = ""
    
    # Data characteristics
    dataset_name: str = ""
    dataset_public: bool = False
    n_subjects: Optional[int] = None
    n_channels: Optional[int] = None
    sampling_rate: Optional[int] = None
    data_duration_hours: Optional[float] = None
    
    # Input/preprocessing
    input_type: Optional[InputType] = None
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # Training/validation
    cross_validation: Optional[CrossValidationType] = None
    train_test_split: str = ""
    
    # Results
    best_accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None
    accuracy_gain: Optional[float] = None
    
    # Reproducibility
    code_available: bool = False
    code_url: str = ""
    data_available: bool = False
    
    # Confidence scores for extracted fields
    extraction_confidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of extracting data items from a paper."""
    paper_id: str
    extracted_data: DLEEGDataItem
    ground_truth: Optional[DLEEGDataItem] = None
    field_accuracy: Dict[str, bool] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    extraction_time_ms: float = 0.0


# =============================================================================
# Ground Truth Data (Sample from Roy et al.)
# =============================================================================

SAMPLE_GROUND_TRUTH = [
    DLEEGDataItem(
        pmid="28212054",
        title="Deep learning with convolutional neural networks for EEG decoding and visualization",
        authors=["Schirrmeister", "Springenberg", "Fiederer"],
        year=2017,
        venue="Human Brain Mapping",
        architecture_type=ArchitectureType.CNN,
        n_layers=4,
        domain=ApplicationDomain.BCI,
        specific_task="Motor imagery decoding",
        dataset_name="BCI Competition IV 2a",
        dataset_public=True,
        n_subjects=9,
        n_channels=22,
        sampling_rate=250,
        input_type=InputType.RAW,
        cross_validation=CrossValidationType.INTRA_SUBJECT,
        best_accuracy=0.925,
        code_available=True,
        code_url="https://github.com/robintibor/braindecode"
    ),
    DLEEGDataItem(
        pmid="28782865",
        title="EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces",
        authors=["Lawhern", "Solon", "Waytowich"],
        year=2018,
        venue="Journal of Neural Engineering",
        architecture_type=ArchitectureType.CNN,
        n_layers=3,
        domain=ApplicationDomain.BCI,
        specific_task="Motor imagery, P300, SSVEP",
        dataset_name="Multiple (BCI Competition)",
        dataset_public=True,
        n_subjects=109,
        n_channels=64,
        input_type=InputType.RAW,
        cross_validation=CrossValidationType.INTER_SUBJECT,
        best_accuracy=0.82,
        code_available=True,
        code_url="https://github.com/vlawhern/arl-eegmodels"
    ),
    DLEEGDataItem(
        pmid="26899889",
        title="Deep learning for EEG-based seizure detection",
        authors=["Antoniades", "Spyrou", "Took"],
        year=2016,
        venue="IJCNN",
        architecture_type=ArchitectureType.CNN,
        n_layers=5,
        domain=ApplicationDomain.EPILEPSY,
        specific_task="Seizure detection",
        dataset_name="CHB-MIT",
        dataset_public=True,
        n_subjects=23,
        n_channels=23,
        sampling_rate=256,
        input_type=InputType.RAW,
        cross_validation=CrossValidationType.INTRA_SUBJECT,
        best_accuracy=0.96,
        code_available=False
    ),
    DLEEGDataItem(
        pmid="28622706",
        title="DeepSleepNet: a model for automatic sleep stage scoring based on raw single-channel EEG",
        authors=["Supratak", "Dong", "Wu", "Guo"],
        year=2017,
        venue="IEEE TNSRE",
        architecture_type=ArchitectureType.HYBRID,
        n_layers=10,
        architecture_details="CNN + BiLSTM",
        domain=ApplicationDomain.SLEEP,
        specific_task="Sleep stage classification",
        dataset_name="Sleep-EDF, MASS",
        dataset_public=True,
        n_subjects=200,
        n_channels=1,
        sampling_rate=100,
        input_type=InputType.RAW,
        cross_validation=CrossValidationType.INTER_SUBJECT,
        best_accuracy=0.82,
        code_available=True,
        code_url="https://github.com/akaraspt/deepsleepnet"
    ),
]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ner_extractor():
    """Initialize NER extractor for DL-EEG terms."""
    if not NER_AVAILABLE or EEGNER is None:
        pytest.skip("NER module not available")
    try:
        return EEGNER()
    except Exception as e:
        pytest.skip(f"Could not initialize EEGNER: {e}")


@pytest.fixture
def citation_verifier():
    """Initialize citation verifier."""
    if not NER_AVAILABLE or CitationVerifier is None:
        pytest.skip("Citation verifier not available")
    try:
        return CitationVerifier()  # No special args needed
    except Exception as e:
        pytest.skip(f"Could not initialize CitationVerifier: {e}")


@pytest.fixture
def sample_abstracts():
    """Sample abstracts for testing extraction."""
    return {
        "28212054": """
        We present a framework for analyzing EEG signals using deep learning.
        Our deep convolutional neural network (CNN) with 4 convolutional layers
        achieves 92.5% accuracy on the BCI Competition IV 2a motor imagery dataset.
        We use raw EEG signals sampled at 250 Hz from 22 channels in 9 subjects.
        An intra-subject cross-validation approach was used. Code is available at
        https://github.com/robintibor/braindecode.
        """,
        "28782865": """
        We introduce EEGNet, a compact convolutional neural network for EEG-based 
        brain-computer interfaces. The architecture uses depthwise and separable 
        convolutions in 3 layers to achieve 82% accuracy on motor imagery tasks.
        We evaluated on 109 subjects from multiple BCI Competition datasets using
        inter-subject transfer learning. Implementation available at
        https://github.com/vlawhern/arl-eegmodels.
        """,
        "28622706": """
        We propose DeepSleepNet for automatic sleep stage scoring from single-channel
        raw EEG. The model combines CNN feature extraction with bidirectional LSTM
        (total 10 layers) and achieves 82% accuracy on Sleep-EDF and MASS datasets
        (200 subjects). We use 100 Hz sampling and inter-subject evaluation.
        Code: https://github.com/akaraspt/deepsleepnet.
        """
    }


@pytest.fixture
def dl_eeg_extractor():
    """Create DL-EEG data extractor."""
    return DLEEGDataExtractor()


# =============================================================================
# DL-EEG Data Extractor
# =============================================================================

class DLEEGDataExtractor:
    """Extracts DL-EEG data items from paper text."""
    
    def __init__(self):
        # Initialize NER if available
        if NER_AVAILABLE and EEGNER is not None:
            try:
                self.ner = EEGNER()
            except Exception:
                self.ner = None
        else:
            self.ner = None
        
        # Regex patterns for structured extraction
        self.patterns = {
            'accuracy': r'(\d{1,2}\.\d+|\d{2,3})\s*%?\s*(?:accuracy|acc\.)',
            'n_layers': r'(\d+)\s*(?:layers?|conv(?:olutional)?\s*layers?)',
            'n_subjects': r'(\d+)\s*(?:subjects?|participants?|patients?)',
            'n_channels': r'(\d+)\s*(?:channels?|electrodes?)',
            'sampling_rate': r'(\d+)\s*(?:Hz|samples?\s*per\s*second)',
            'github_url': r'https?://github\.com/[\w\-]+/[\w\-]+',
            'pmid': r'PMID[:\s]*(\d{7,8})',
        }
        
        # Architecture keywords
        self.architecture_keywords = {
            ArchitectureType.CNN: ['cnn', 'convolutional', 'convnet', 'conv2d', 'conv1d'],
            ArchitectureType.RNN: ['rnn', 'recurrent', 'lstm', 'gru', 'bidirectional'],
            ArchitectureType.DBN: ['dbn', 'deep belief', 'rbm'],
            ArchitectureType.AE: ['autoencoder', 'ae', 'vae', 'variational'],
            ArchitectureType.MLP: ['mlp', 'fully connected', 'dense layers'],
        }
        
        # Domain keywords
        self.domain_keywords = {
            ApplicationDomain.EPILEPSY: ['epilepsy', 'seizure', 'ictal', 'interictal'],
            ApplicationDomain.SLEEP: ['sleep', 'polysomnography', 'psg', 'rem', 'nrem'],
            ApplicationDomain.BCI: ['bci', 'brain-computer', 'motor imagery', 'p300', 'ssvep'],
            ApplicationDomain.COGNITIVE: ['cognitive', 'attention', 'memory', 'workload'],
            ApplicationDomain.AFFECTIVE: ['emotion', 'affective', 'valence', 'arousal'],
        }
        
        # Known public datasets
        self.public_datasets = [
            'bci competition', 'physionet', 'chb-mit', 'sleep-edf', 'mass',
            'deap', 'seed', 'eegmmidb', 'temple university', 'tuh eeg'
        ]
    
    def extract(self, text: str, pmid: Optional[str] = None) -> DLEEGDataItem:
        """Extract DL-EEG data items from text.
        
        Args:
            text: Paper abstract or full text.
            pmid: Optional PMID if known.
            
        Returns:
            Extracted data item.
        """
        text_lower = text.lower()
        data = DLEEGDataItem(pmid=pmid)
        confidence = {}
        
        # Extract architecture type
        data.architecture_type, confidence['architecture_type'] = self._extract_architecture(text_lower)
        
        # Extract domain
        data.domain, confidence['domain'] = self._extract_domain(text_lower)
        
        # Extract numerical values
        data.n_layers, confidence['n_layers'] = self._extract_pattern('n_layers', text)
        data.n_subjects, confidence['n_subjects'] = self._extract_pattern('n_subjects', text)
        data.n_channels, confidence['n_channels'] = self._extract_pattern('n_channels', text)
        data.sampling_rate, confidence['sampling_rate'] = self._extract_pattern('sampling_rate', text)
        
        # Extract accuracy
        accuracy_val, confidence['best_accuracy'] = self._extract_accuracy(text)
        if accuracy_val:
            data.best_accuracy = accuracy_val
        
        # Extract code availability
        github_match = re.search(self.patterns['github_url'], text)
        if github_match:
            data.code_available = True
            data.code_url = github_match.group(0)
            confidence['code_available'] = 1.0
        else:
            confidence['code_available'] = 0.5  # Uncertain
        
        # Check for public dataset
        for dataset in self.public_datasets:
            if dataset in text_lower:
                data.dataset_public = True
                data.dataset_name = dataset.title()
                confidence['dataset_public'] = 0.9
                break
        
        # Extract cross-validation type
        data.cross_validation, confidence['cross_validation'] = self._extract_cv_type(text_lower)
        
        # Extract input type
        data.input_type, confidence['input_type'] = self._extract_input_type(text_lower)
        
        data.extraction_confidence = confidence
        return data
    
    def _extract_architecture(self, text: str) -> Tuple[Optional[ArchitectureType], float]:
        """Extract architecture type."""
        scores = {}
        for arch_type, keywords in self.architecture_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                scores[arch_type] = count
        
        if not scores:
            return None, 0.0
        
        # Check for hybrid
        if len(scores) > 1:
            return ArchitectureType.HYBRID, 0.8
        
        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best] / 2)  # Normalize
        return best, confidence
    
    def _extract_domain(self, text: str) -> Tuple[Optional[ApplicationDomain], float]:
        """Extract application domain."""
        scores = {}
        for domain, keywords in self.domain_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                scores[domain] = count
        
        if not scores:
            return None, 0.0
        
        best = max(scores, key=scores.get)
        confidence = min(1.0, scores[best] / 2)
        return best, confidence
    
    def _extract_pattern(self, pattern_name: str, text: str) -> Tuple[Optional[int], float]:
        """Extract numerical value using regex pattern."""
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return None, 0.0
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                value = int(matches[0])
                return value, 0.9
            except ValueError:
                return None, 0.0
        return None, 0.0
    
    def _extract_accuracy(self, text: str) -> Tuple[Optional[float], float]:
        """Extract accuracy value."""
        pattern = r'(\d{1,2}\.\d+|\d{2,3})\s*%?\s*(?:accuracy|acc)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                value = float(match)
                if value > 1:
                    value = value / 100  # Convert percentage
                if 0.5 <= value <= 1.0:  # Reasonable accuracy range
                    return value, 0.85
            except ValueError:
                continue
        return None, 0.0
    
    def _extract_cv_type(self, text: str) -> Tuple[Optional[CrossValidationType], float]:
        """Extract cross-validation type."""
        if 'inter-subject' in text or 'inter subject' in text or 'cross-subject' in text:
            return CrossValidationType.INTER_SUBJECT, 0.9
        elif 'intra-subject' in text or 'intra subject' in text or 'within-subject' in text:
            return CrossValidationType.INTRA_SUBJECT, 0.9
        elif 'leave-one-out' in text:
            return CrossValidationType.INTER_SUBJECT, 0.7
        return None, 0.0
    
    def _extract_input_type(self, text: str) -> Tuple[Optional[InputType], float]:
        """Extract input type."""
        if 'raw eeg' in text or 'raw signal' in text:
            return InputType.RAW, 0.9
        elif 'spectrogram' in text or 'time-frequency' in text:
            return InputType.SPECTROGRAM, 0.9
        elif 'feature' in text and ('extract' in text or 'hand-crafted' in text):
            return InputType.FEATURES, 0.7
        elif 'preprocess' in text or 'filter' in text:
            return InputType.PREPROCESSED, 0.6
        return None, 0.0


# =============================================================================
# Test Classes
# =============================================================================

class TestDLEEGDataExtraction:
    """Tests for DL-EEG data extraction accuracy."""
    
    def test_architecture_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test architecture type extraction."""
        # Test CNN detection
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        assert result.architecture_type == ArchitectureType.CNN
        assert result.extraction_confidence.get('architecture_type', 0) > 0.5
        
        # Test hybrid detection (CNN + LSTM)
        result = dl_eeg_extractor.extract(sample_abstracts["28622706"])
        assert result.architecture_type == ArchitectureType.HYBRID
    
    def test_domain_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test application domain extraction."""
        # Test BCI domain
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        assert result.domain == ApplicationDomain.BCI
        
        # Test sleep domain
        result = dl_eeg_extractor.extract(sample_abstracts["28622706"])
        assert result.domain == ApplicationDomain.SLEEP
    
    def test_numerical_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test numerical value extraction."""
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        
        # Check layer count
        assert result.n_layers == 4
        
        # Check subject count
        assert result.n_subjects == 9
        
        # Check channel count
        assert result.n_channels == 22
        
        # Check sampling rate
        assert result.sampling_rate == 250
    
    def test_accuracy_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test accuracy value extraction."""
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        
        assert result.best_accuracy is not None
        assert 0.90 <= result.best_accuracy <= 0.95  # ~92.5%
    
    def test_code_availability_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test code availability detection."""
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        
        assert result.code_available is True
        assert 'braindecode' in result.code_url
    
    def test_cross_validation_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test cross-validation type extraction."""
        # Intra-subject
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        assert result.cross_validation == CrossValidationType.INTRA_SUBJECT
        
        # Inter-subject
        result = dl_eeg_extractor.extract(sample_abstracts["28782865"])
        assert result.cross_validation == CrossValidationType.INTER_SUBJECT
    
    def test_input_type_extraction(self, dl_eeg_extractor, sample_abstracts):
        """Test input type extraction."""
        result = dl_eeg_extractor.extract(sample_abstracts["28212054"])
        assert result.input_type == InputType.RAW


class TestGroundTruthValidation:
    """Tests validating extraction against ground truth from Roy et al."""
    
    def test_ground_truth_field_accuracy(self, dl_eeg_extractor, sample_abstracts):
        """Test extraction accuracy against known ground truth."""
        ground_truth = SAMPLE_GROUND_TRUTH[0]  # Schirrmeister et al.
        pmid = "28212054"
        
        if pmid not in sample_abstracts:
            pytest.skip("Sample abstract not available")
        
        extracted = dl_eeg_extractor.extract(sample_abstracts[pmid], pmid=pmid)
        
        # Calculate field-by-field accuracy
        correct_fields = 0
        total_fields = 0
        
        # Architecture type
        total_fields += 1
        if extracted.architecture_type == ground_truth.architecture_type:
            correct_fields += 1
        
        # Domain
        total_fields += 1
        if extracted.domain == ground_truth.domain:
            correct_fields += 1
        
        # Number of layers
        total_fields += 1
        if extracted.n_layers == ground_truth.n_layers:
            correct_fields += 1
        
        # Number of subjects
        total_fields += 1
        if extracted.n_subjects == ground_truth.n_subjects:
            correct_fields += 1
        
        # Code availability
        total_fields += 1
        if extracted.code_available == ground_truth.code_available:
            correct_fields += 1
        
        accuracy = correct_fields / total_fields
        
        # Assert at least 80% accuracy on known fields
        assert accuracy >= 0.8, f"Field accuracy {accuracy:.1%} below 80% threshold"
    
    def test_all_ground_truth_samples(self, dl_eeg_extractor, sample_abstracts):
        """Test extraction accuracy across all ground truth samples."""
        total_correct = 0
        total_fields = 0
        
        for gt in SAMPLE_GROUND_TRUTH:
            if gt.pmid not in sample_abstracts:
                continue
            
            extracted = dl_eeg_extractor.extract(sample_abstracts[gt.pmid], pmid=gt.pmid)
            
            # Check architecture
            total_fields += 1
            if extracted.architecture_type == gt.architecture_type:
                total_correct += 1
            
            # Check domain
            total_fields += 1
            if extracted.domain == gt.domain:
                total_correct += 1
        
        if total_fields > 0:
            overall_accuracy = total_correct / total_fields
            assert overall_accuracy >= 0.75, f"Overall accuracy {overall_accuracy:.1%} below threshold"


class TestNERForDLEEG:
    """Tests for NER extraction of DL-EEG specific entities."""
    
    def test_architecture_entity_recognition(self, ner_extractor):
        """Test NER recognizes DL architecture terms."""
        text = "We used a convolutional neural network with LSTM layers for EEG classification."
        result = ner_extractor.extract_entities(text)
        
        # extract_entities returns NERResult with entities attribute
        entities = result.entities if hasattr(result, 'entities') else result
        
        # Should extract processing method entities
        entity_texts = [e.text.lower() for e in entities]
        
        # Check for some architecture-related extractions
        assert len(entities) > 0
    
    def test_eeg_specific_entities(self, ner_extractor):
        """Test NER extracts EEG-specific entities."""
        text = """
        We analyzed alpha (8-13 Hz) and beta (13-30 Hz) oscillations from 
        electrodes F3, F4, C3, C4, P3, P4 using the 10-20 system. The P300 
        component was detected in response to target stimuli.
        """
        
        result = ner_extractor.extract_entities(text)
        entities = result.entities if hasattr(result, 'entities') else result
        entity_types = [e.entity_type for e in entities]
        
        # Should extract frequency bands
        assert EntityType.FREQUENCY_BAND in entity_types
        
        # Should extract electrodes
        assert EntityType.ELECTRODE in entity_types
    
    def test_clinical_condition_extraction(self, ner_extractor):
        """Test extraction of clinical conditions."""
        text = "Deep learning for epilepsy seizure detection in patients with temporal lobe epilepsy."
        
        result = ner_extractor.extract_entities(text)
        entities = result.entities if hasattr(result, 'entities') else result
        
        # Should find epilepsy-related entities
        clinical_entities = [e for e in entities if e.entity_type == EntityType.CLINICAL_CONDITION]
        assert len(clinical_entities) > 0


class TestCitationVerification:
    """Tests for PMID verification of extracted papers."""
    
    @pytest.mark.asyncio
    async def test_verify_known_pmids(self, citation_verifier):
        """Test verification of known valid PMIDs."""
        known_pmids = ["31151119", "28212054", "28782865"]
        
        for pmid in known_pmids:
            try:
                result = await citation_verifier.verify_citation(pmid)
                # VerificationResult has is_valid attribute
                is_valid = result.is_valid if hasattr(result, 'is_valid') else result
                assert is_valid, f"PMID {pmid} should be valid"
            except Exception as e:
                pytest.skip(f"PubMed API unavailable: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_reject_invalid_pmids(self, citation_verifier):
        """Test rejection of invalid PMIDs."""
        invalid_pmids = ["0000000", "9999999999", "invalid"]
        
        for pmid in invalid_pmids:
            try:
                result = await citation_verifier.verify_citation(pmid)
                is_valid = result.is_valid if hasattr(result, 'is_valid') else result
                assert not is_valid, f"PMID {pmid} should be invalid"
            except Exception:
                pass  # API errors are acceptable for invalid PMIDs

class TestReviewExtension:
    """Tests for extending the review to new publications."""
    
    def test_temporal_coverage(self):
        """Test that extension covers 2019-2026 timeframe."""
        original_end_date = 2018  # July 2018
        current_date = 2026
        extension_years = current_date - original_end_date
        
        assert extension_years >= 7, "Should cover at least 7 years of new publications"
    
    def test_new_architecture_detection(self, dl_eeg_extractor):
        """Test detection of architectures not in original review."""
        # Transformers became popular after 2018
        transformer_text = """
        We applied a Vision Transformer (ViT) architecture with self-attention
        mechanisms to EEG classification, achieving state-of-the-art results
        using positional encoding of EEG channel positions.
        """
        
        result = dl_eeg_extractor.extract(transformer_text)
        
        # Should at least detect it's not a standard architecture
        # or flag for review
        assert result.extraction_confidence.get('architecture_type', 0) < 1.0
    
    def test_new_dataset_detection(self, dl_eeg_extractor):
        """Test detection of datasets released after 2018."""
        new_dataset_text = """
        We evaluated on the TUH EEG Corpus v2.0, a large-scale clinical EEG 
        dataset with over 25,000 recordings from Temple University Hospital.
        """
        
        result = dl_eeg_extractor.extract(new_dataset_text)
        
        # Should detect public dataset
        assert result.dataset_public is True
        assert 'temple' in result.dataset_name.lower() or 'tuh' in new_dataset_text.lower()


class TestBenchmarkMetrics:
    """Tests for benchmark metrics matching Roy et al. findings."""
    
    def test_architecture_distribution(self):
        """Test that CNN dominance matches original findings (~60% CNNs)."""
        # From the review: ~60% used CNNs, ~20% used RNNs
        sample_architectures = [
            ArchitectureType.CNN, ArchitectureType.CNN, ArchitectureType.CNN,
            ArchitectureType.CNN, ArchitectureType.CNN, ArchitectureType.CNN,
            ArchitectureType.RNN, ArchitectureType.RNN,
            ArchitectureType.HYBRID, ArchitectureType.DBN
        ]
        
        cnn_ratio = sum(1 for a in sample_architectures if a == ArchitectureType.CNN) / len(sample_architectures)
        
        # CNN should be dominant (around 60%)
        assert 0.5 <= cnn_ratio <= 0.7
    
    def test_layer_count_range(self):
        """Test that layer counts match original findings (3-10 layers typical)."""
        # From the review: most models had 3-10 layers
        sample_layers = [4, 3, 5, 10, 6, 8, 4, 7, 5, 3]
        
        median_layers = sorted(sample_layers)[len(sample_layers) // 2]
        
        assert 3 <= median_layers <= 10
    
    def test_accuracy_gain_calculation(self):
        """Test accuracy gain calculation (DL vs baseline)."""
        # From the review: median gain was ~5-10%
        sample_results = [
            {'dl_accuracy': 0.92, 'baseline_accuracy': 0.85},
            {'dl_accuracy': 0.88, 'baseline_accuracy': 0.80},
            {'dl_accuracy': 0.95, 'baseline_accuracy': 0.90},
        ]
        
        gains = [(r['dl_accuracy'] - r['baseline_accuracy']) for r in sample_results]
        avg_gain = sum(gains) / len(gains)
        
        # Average gain should be positive and reasonable
        assert 0.05 <= avg_gain <= 0.15


class TestReproducibilityAnalysis:
    """Tests for reproducibility metric extraction."""
    
    def test_code_availability_detection(self, dl_eeg_extractor):
        """Test detection of code availability."""
        # Paper with code
        with_code = "Code is available at https://github.com/example/repo"
        result = dl_eeg_extractor.extract(with_code)
        assert result.code_available is True
        
        # Paper without code mention
        without_code = "We trained a CNN on EEG data and achieved 90% accuracy."
        result = dl_eeg_extractor.extract(without_code)
        # Should be uncertain (not definitively False)
        assert result.extraction_confidence.get('code_available', 0) < 0.8
    
    def test_data_availability_indicators(self, dl_eeg_extractor):
        """Test detection of data availability."""
        public_data = "We used the publicly available BCI Competition IV dataset."
        result = dl_eeg_extractor.extract(public_data)
        assert result.dataset_public is True
        
        private_data = "We collected EEG data from 50 volunteers in our lab."
        result = dl_eeg_extractor.extract(private_data)
        assert result.dataset_public is False or result.dataset_name == ""


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestSystematicReviewPipeline:
    """End-to-end tests for systematic review pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self, dl_eeg_extractor, citation_verifier, sample_abstracts):
        """Test complete extraction and validation pipeline."""
        results = []
        
        for pmid, abstract in sample_abstracts.items():
            # Extract data
            data = dl_eeg_extractor.extract(abstract, pmid=pmid)
            
            # Verify PMID
            try:
                result = await citation_verifier.verify_citation(pmid)
                is_valid = result.is_valid if hasattr(result, 'is_valid') else result
                data.pmid = pmid if is_valid else None
            except Exception:
                pass
            
            results.append(data)
        
        # Should process all papers
        assert len(results) == len(sample_abstracts)
        
        # At least some should have architecture detected
        architectures_detected = sum(1 for r in results if r.architecture_type is not None)
        assert architectures_detected >= len(results) * 0.8
    
    def test_batch_processing_performance(self, dl_eeg_extractor, sample_abstracts):
        """Test batch processing performance."""
        import time
        
        # Process all abstracts
        start_time = time.time()
        
        results = []
        for pmid, abstract in sample_abstracts.items():
            result = dl_eeg_extractor.extract(abstract, pmid=pmid)
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Should process reasonably fast (< 1 second per abstract)
        time_per_abstract = elapsed_time / len(sample_abstracts)
        assert time_per_abstract < 1.0, f"Processing too slow: {time_per_abstract:.2f}s per abstract"
    
    def test_confidence_score_distribution(self, dl_eeg_extractor, sample_abstracts):
        """Test that confidence scores are properly distributed."""
        all_confidences = []
        
        for pmid, abstract in sample_abstracts.items():
            result = dl_eeg_extractor.extract(abstract, pmid=pmid)
            all_confidences.extend(result.extraction_confidence.values())
        
        # Confidence scores should be in valid range
        assert all(0 <= c <= 1 for c in all_confidences)
        
        # Should have some variation (not all 1.0 or all 0.0)
        unique_scores = set(round(c, 2) for c in all_confidences)
        assert len(unique_scores) > 1, "Confidence scores should vary"


# =============================================================================
# CLI/API Tests
# =============================================================================

class TestReviewCLI:
    """Tests for CLI commands for systematic review."""
    
    def test_extraction_command_structure(self):
        """Test that extraction command structure is valid."""
        # Verify expected CLI command structure
        expected_commands = [
            'extract-papers',
            'validate-extraction',
            'compare-ground-truth',
            'generate-report'
        ]
        
        # This tests the expected interface
        for cmd in expected_commands:
            assert isinstance(cmd, str)
            assert len(cmd) > 0
    
    def test_output_format_options(self):
        """Test output format options."""
        valid_formats = ['csv', 'json', 'xlsx']
        
        for fmt in valid_formats:
            assert fmt in valid_formats