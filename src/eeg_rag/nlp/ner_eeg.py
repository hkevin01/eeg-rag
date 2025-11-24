"""
Named Entity Recognition (NER) for EEG Terms
Identifies and extracts EEG-specific entities from research papers

Requirements:
- REQ-NER-001: Identify EEG frequency bands (delta, theta, alpha, beta, gamma)
- REQ-NER-002: Recognize brain regions and electrode locations
- REQ-NER-003: Extract clinical conditions and disorders
- REQ-NER-004: Identify EEG features and biomarkers
- REQ-NER-005: Recognize measurement units and metrics
- REQ-NER-006: Extract experimental paradigms and tasks
- REQ-NER-007: Identify signal processing methods
- REQ-NER-008: Support batch processing of documents
- REQ-NER-009: Provide confidence scores for entities
- REQ-NER-010: Export entities in structured format
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple, Optional
from enum import Enum
import re
from collections import defaultdict
import json


class EntityType(Enum):
    """Types of EEG-related entities"""
    FREQUENCY_BAND = "frequency_band"
    BRAIN_REGION = "brain_region"
    ELECTRODE = "electrode"
    CLINICAL_CONDITION = "clinical_condition"
    BIOMARKER = "biomarker"
    MEASUREMENT_UNIT = "measurement_unit"
    SIGNAL_FEATURE = "signal_feature"
    EXPERIMENTAL_TASK = "experimental_task"
    PROCESSING_METHOD = "processing_method"
    EEG_PHENOMENON = "eeg_phenomenon"
    COGNITIVE_STATE = "cognitive_state"
    HARDWARE = "hardware"


@dataclass
class Entity:
    """Represents an extracted named entity"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'type': self.entity_type.value,
            'start': self.start_pos,
            'end': self.end_pos,
            'confidence': self.confidence,
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class NERResult:
    """Result from NER processing"""
    text: str
    entities: List[Entity]
    entity_counts: Dict[str, int]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text_length': len(self.text),
            'total_entities': len(self.entities),
            'entities': [e.to_dict() for e in self.entities],
            'entity_counts': self.entity_counts,
            'processing_time': self.processing_time
        }


class EEGTerminologyDatabase:
    """
    Comprehensive database of EEG terminology
    Organized by entity type for efficient lookup
    """
    
    # Frequency Bands
    FREQUENCY_BANDS = {
        # Standard bands
        'delta': {'range': (0.5, 4), 'description': 'Deep sleep, unconsciousness'},
        'theta': {'range': (4, 8), 'description': 'Drowsiness, meditation, memory'},
        'alpha': {'range': (8, 13), 'description': 'Relaxed wakefulness, eyes closed'},
        'beta': {'range': (13, 30), 'description': 'Active thinking, concentration'},
        'gamma': {'range': (30, 100), 'description': 'Higher cognitive processing'},
        'mu': {'range': (8, 13), 'description': 'Motor cortex rhythm'},
        
        # Sub-bands
        'low alpha': {'range': (8, 10), 'description': 'Lower alpha band'},
        'high alpha': {'range': (10, 13), 'description': 'Upper alpha band'},
        'low beta': {'range': (13, 20), 'description': 'Lower beta band'},
        'high beta': {'range': (20, 30), 'description': 'Upper beta band'},
        'low gamma': {'range': (30, 50), 'description': 'Lower gamma band'},
        'high gamma': {'range': (50, 100), 'description': 'Upper gamma band'},
        
        # Combined terms
        'theta-alpha': {'range': (4, 13), 'description': 'Combined theta-alpha'},
        'alpha-beta': {'range': (8, 30), 'description': 'Combined alpha-beta'},
    }
    
    # Brain Regions (anatomical)
    BRAIN_REGIONS = [
        # Lobes
        'frontal lobe', 'frontal cortex', 'prefrontal cortex', 'prefrontal',
        'parietal lobe', 'parietal cortex',
        'temporal lobe', 'temporal cortex',
        'occipital lobe', 'occipital cortex',
        'limbic system',
        
        # Specific regions
        'motor cortex', 'premotor cortex', 'supplementary motor area',
        'somatosensory cortex', 'primary somatosensory cortex',
        'visual cortex', 'primary visual cortex',
        'auditory cortex', 'primary auditory cortex',
        'cingulate cortex', 'anterior cingulate', 'posterior cingulate',
        'hippocampus', 'amygdala', 'thalamus', 'hypothalamus',
        'basal ganglia', 'striatum', 'cerebellum',
        'corpus callosum', 'brainstem',
        
        # Functional areas
        'dorsolateral prefrontal cortex', 'dlpfc',
        'ventromedial prefrontal cortex', 'vmpfc',
        'orbitofrontal cortex', 'ofc',
        'superior temporal gyrus', 'middle temporal gyrus',
        'fusiform gyrus', 'parahippocampal gyrus',
        'precuneus', 'cuneus',
        'insula', 'insular cortex',
    ]
    
    # Electrode Locations (10-20 system)
    ELECTRODES = [
        # Frontal
        'Fp1', 'Fp2', 'Fpz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz',
        'AF3', 'AF4', 'AF7', 'AF8', 'AFz',
        
        # Central
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz',
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
        
        # Parietal
        'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz',
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz',
        
        # Temporal
        'T3', 'T4', 'T5', 'T6', 'T7', 'T8',
        'TP7', 'TP8', 'FT7', 'FT8',
        
        # Occipital
        'O1', 'O2', 'Oz',
        'PO3', 'PO4', 'PO7', 'PO8', 'POz',
        
        # Reference electrodes
        'A1', 'A2', 'M1', 'M2',
        
        # Common references
        'average reference', 'linked mastoids', 'common reference',
    ]
    
    # Clinical Conditions
    CLINICAL_CONDITIONS = [
        # Neurological disorders
        'epilepsy', 'seizure', 'seizures', 'epileptic seizure',
        'focal seizure', 'generalized seizure', 'absence seizure',
        'tonic-clonic seizure', 'myoclonic seizure',
        'status epilepticus',
        
        "alzheimer's disease", 'alzheimer disease', 'dementia',
        'vascular dementia', 'lewy body dementia',
        "parkinson's disease", 'parkinson disease',
        "huntington's disease", 'huntington disease',
        
        'stroke', 'ischemic stroke', 'hemorrhagic stroke',
        'traumatic brain injury', 'tbi', 'concussion',
        'encephalopathy', 'encephalitis', 'meningitis',
        
        'multiple sclerosis', 'ms',
        'amyotrophic lateral sclerosis', 'als',
        'brain tumor', 'glioma', 'meningioma',
        
        # Psychiatric disorders
        'depression', 'major depressive disorder', 'mdd',
        'anxiety', 'anxiety disorder', 'generalized anxiety disorder',
        'panic disorder', 'social anxiety',
        'schizophrenia', 'psychosis',
        'bipolar disorder', 'mania', 'manic episode',
        'post-traumatic stress disorder', 'ptsd',
        'attention deficit hyperactivity disorder', 'adhd',
        'autism spectrum disorder', 'autism', 'asd',
        'obsessive-compulsive disorder', 'ocd',
        
        # Sleep disorders
        'insomnia', 'sleep apnea', 'narcolepsy',
        'restless leg syndrome', 'rls',
        'rem sleep behavior disorder',
        
        # Cognitive impairments
        'mild cognitive impairment', 'mci',
        'cognitive decline', 'cognitive impairment',
        'memory impairment', 'executive dysfunction',
    ]
    
    # EEG Biomarkers and Features
    BIOMARKERS = [
        # Event-related potentials
        'P300', 'P3', 'P3a', 'P3b',
        'N100', 'N1', 'N170', 'N200', 'N2', 'N400',
        'P100', 'P1', 'P200', 'P2',
        'mismatch negativity', 'mmn',
        'error-related negativity', 'ern',
        'contingent negative variation', 'cnv',
        'bereitschaftspotential', 'readiness potential',
        
        # Oscillatory features
        'alpha asymmetry', 'frontal alpha asymmetry',
        'theta-beta ratio', 'theta/beta ratio',
        'peak alpha frequency', 'individual alpha frequency',
        'alpha blocking', 'alpha suppression',
        'beta rebound', 'beta desynchronization',
        'gamma burst', 'gamma oscillation',
        'spindle', 'sleep spindle', 'sigma spindle',
        'k-complex', 'k complex',
        
        # Power measures
        'absolute power', 'relative power',
        'power spectral density', 'psd',
        'band power', 'spectral power',
        
        # Synchrony measures
        'coherence', 'phase coherence', 'phase locking',
        'phase synchronization', 'phase-amplitude coupling',
        'cross-frequency coupling',
        
        # Other features
        'amplitude', 'peak amplitude', 'mean amplitude',
        'latency', 'peak latency',
        'frequency', 'peak frequency', 'dominant frequency',
        'complexity', 'entropy', 'sample entropy',
        'fractal dimension',
    ]
    
    # Measurement Units
    MEASUREMENT_UNITS = [
        'μV', 'microvolt', 'microvolts', 'uV',
        'mV', 'millivolt', 'millivolts',
        'Hz', 'hertz',
        'ms', 'millisecond', 'milliseconds',
        's', 'second', 'seconds',
        'min', 'minute', 'minutes',
        'dB', 'decibel', 'decibels',
        'μV²/Hz', 'μV2/Hz',
    ]
    
    # Signal Features
    SIGNAL_FEATURES = [
        'amplitude modulation', 'frequency modulation',
        'phase', 'phase angle', 'phase shift',
        'synchrony', 'desynchrony', 'synchronization',
        'burst', 'paroxysm', 'spike', 'sharp wave',
        'spike-wave', 'spike and wave',
        'slow wave', 'fast wave',
        'artifact', 'artifacts', 'noise',
        'baseline', 'baseline drift',
        'epoch', 'epochs', 'trial', 'trials',
        'waveform', 'morphology',
        'symmetry', 'asymmetry',
    ]
    
    # Experimental Tasks/Paradigms
    EXPERIMENTAL_TASKS = [
        # Cognitive tasks
        'oddball task', 'oddball paradigm', 'auditory oddball', 'visual oddball',
        'go/no-go task', 'go nogo task',
        'stroop task', 'stroop test',
        'n-back task', 'n back task', 'working memory task',
        'flanker task',
        'simon task',
        'posner task', 'attention cueing',
        
        # Memory tasks
        'recognition task', 'recall task',
        'encoding task', 'retrieval task',
        'associative memory', 'episodic memory task',
        
        # Motor tasks
        'motor imagery', 'motor execution',
        'finger tapping', 'hand movement',
        'motor planning',
        
        # Emotional tasks
        'emotional face processing', 'face recognition',
        'affective processing',
        
        # Sleep studies
        'sleep study', 'polysomnography', 'psg',
        'sleep staging', 'sleep architecture',
        
        # States
        'eyes open', 'eyes closed',
        'resting state', 'rest condition',
        'active condition', 'task condition',
    ]
    
    # Signal Processing Methods
    PROCESSING_METHODS = [
        # Filtering
        'bandpass filter', 'lowpass filter', 'highpass filter',
        'notch filter', 'butterworth filter',
        'filtering', 'digital filter',
        
        # Artifact removal
        'artifact rejection', 'artifact removal',
        'ica', 'independent component analysis',
        'pca', 'principal component analysis',
        'baseline correction',
        
        # Time-frequency analysis
        'fourier transform', 'fft', 'fast fourier transform',
        'wavelet transform', 'wavelet analysis',
        'hilbert transform',
        'short-time fourier transform', 'stft',
        'spectrogram',
        
        # Decomposition
        'empirical mode decomposition', 'emd',
        'variational mode decomposition', 'vmd',
        
        # Source localization
        'source localization', 'source analysis',
        'beamforming', 'dipole fitting',
        'loreta', 'sloreta',
        
        # Connectivity
        'connectivity analysis', 'functional connectivity',
        'granger causality', 'transfer entropy',
        
        # Classification
        'machine learning', 'deep learning',
        'support vector machine', 'svm',
        'linear discriminant analysis', 'lda',
        'neural network', 'convolutional neural network', 'cnn',
    ]
    
    # EEG Phenomena
    EEG_PHENOMENA = [
        'alpha rhythm', 'mu rhythm', 'beta rhythm',
        'rolandic mu', 'posterior alpha',
        'lambda waves',
        'positive occipital sharp transients', 'posts',
        'wicket spikes',
        'breach rhythm',
        'photic driving',
        'hyperventilation response',
    ]
    
    # Cognitive States
    COGNITIVE_STATES = [
        'attention', 'focused attention', 'sustained attention',
        'alertness', 'vigilance',
        'consciousness', 'awareness',
        'wakefulness', 'arousal',
        'drowsiness', 'sleepiness',
        'sleep', 'rem sleep', 'nrem sleep', 'deep sleep',
        'sleep stage 1', 'sleep stage 2', 'sleep stage 3', 'sleep stage 4',
        'meditation', 'relaxation',
        'stress', 'mental effort',
        'cognitive load', 'workload',
    ]
    
    # Hardware
    HARDWARE = [
        'electrode', 'electrodes', 'electrode cap', 'electrode montage',
        'amplifier', 'eeg amplifier',
        'headset', 'eeg headset',
        'dry electrode', 'wet electrode', 'gel electrode',
        'active electrode', 'passive electrode',
        'high-density eeg', 'hdeg',
        'wireless eeg', 'mobile eeg',
        'eeg system', 'eeg device',
    ]


class EEGNER:
    """
    Named Entity Recognition for EEG terminology
    
    Uses dictionary-based matching with context awareness
    and confidence scoring.
    """
    
    def __init__(self):
        """Initialize EEG NER system"""
        self.terminology = EEGTerminologyDatabase()
        
        # Build lookup dictionaries for efficient matching
        self._build_lookup_tables()
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_entities_found': 0,
            'entities_by_type': defaultdict(int)
        }
    
    def _build_lookup_tables(self):
        """Build efficient lookup tables for entity matching"""
        self.entity_patterns = {
            EntityType.FREQUENCY_BAND: self.terminology.FREQUENCY_BANDS.keys(),
            EntityType.BRAIN_REGION: self.terminology.BRAIN_REGIONS,
            EntityType.ELECTRODE: self.terminology.ELECTRODES,
            EntityType.CLINICAL_CONDITION: self.terminology.CLINICAL_CONDITIONS,
            EntityType.BIOMARKER: self.terminology.BIOMARKERS,
            EntityType.MEASUREMENT_UNIT: self.terminology.MEASUREMENT_UNITS,
            EntityType.SIGNAL_FEATURE: self.terminology.SIGNAL_FEATURES,
            EntityType.EXPERIMENTAL_TASK: self.terminology.EXPERIMENTAL_TASKS,
            EntityType.PROCESSING_METHOD: self.terminology.PROCESSING_METHODS,
            EntityType.EEG_PHENOMENON: self.terminology.EEG_PHENOMENA,
            EntityType.COGNITIVE_STATE: self.terminology.COGNITIVE_STATES,
            EntityType.HARDWARE: self.terminology.HARDWARE,
        }
        
        # Create compiled regex patterns for faster matching
        self.compiled_patterns = {}
        for entity_type, terms in self.entity_patterns.items():
            # Sort by length (longest first) to match longer phrases first
            sorted_terms = sorted(terms, key=len, reverse=True)
            # Escape special regex characters and create word boundary pattern
            escaped_terms = [re.escape(term) for term in sorted_terms]
            pattern = r'\b(' + '|'.join(escaped_terms) + r')\b'
            self.compiled_patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
    
    def extract_entities(
        self,
        text: str,
        context_window: int = 50,
        min_confidence: float = 0.0
    ) -> NERResult:
        """
        Extract EEG-related entities from text
        
        Args:
            text: Input text to process
            context_window: Number of characters for context (before and after)
            min_confidence: Minimum confidence threshold for entities
            
        Returns:
            NERResult with extracted entities
        """
        import time
        start_time = time.time()
        
        entities = []
        entity_counts = defaultdict(int)
        
        # Extract entities for each type
        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                entity_text = match.group()
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract context
                context_start = max(0, start_pos - context_window)
                context_end = min(len(text), end_pos + context_window)
                context = text[context_start:context_end]
                
                # Calculate confidence (can be enhanced with ML)
                confidence = self._calculate_confidence(
                    entity_text, entity_type, context
                )
                
                if confidence >= min_confidence:
                    # Get metadata for frequency bands
                    metadata = {}
                    if entity_type == EntityType.FREQUENCY_BAND:
                        band_info = self.terminology.FREQUENCY_BANDS.get(
                            entity_text.lower(), {}
                        )
                        if band_info:
                            metadata['frequency_range'] = band_info.get('range')
                            metadata['description'] = band_info.get('description')
                    
                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        context=context,
                        metadata=metadata
                    )
                    entities.append(entity)
                    entity_counts[entity_type.value] += 1
        
        # Sort entities by position
        entities.sort(key=lambda e: e.start_pos)
        
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlaps(entities)
        
        # Update statistics
        self.stats['documents_processed'] += 1
        self.stats['total_entities_found'] += len(entities)
        for entity_type in entity_counts:
            self.stats['entities_by_type'][entity_type] += entity_counts[entity_type]
        
        processing_time = time.time() - start_time
        
        return NERResult(
            text=text,
            entities=entities,
            entity_counts=dict(entity_counts),
            processing_time=processing_time
        )
    
    def _calculate_confidence(
        self,
        entity_text: str,
        entity_type: EntityType,
        context: str
    ) -> float:
        """
        Calculate confidence score for entity
        
        Factors:
        - Length of match (longer = more specific)
        - Context relevance
        - Capitalization (for electrodes)
        """
        confidence = 0.8  # Base confidence
        
        # Bonus for longer matches (more specific)
        if len(entity_text) > 10:
            confidence += 0.1
        elif len(entity_text) > 5:
            confidence += 0.05
        
        # Bonus for electrodes with correct capitalization
        if entity_type == EntityType.ELECTRODE:
            if entity_text[0].isupper():
                confidence += 0.1
        
        # Context relevance (simple keyword check)
        eeg_keywords = ['eeg', 'electroencephalog', 'brain', 'neural', 'cortex']
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in eeg_keywords):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence ones"""
        if not entities:
            return entities
        
        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start_pos, -e.confidence)
        )
        
        result = []
        last_end = -1
        
        for entity in sorted_entities:
            if entity.start_pos >= last_end:
                result.append(entity)
                last_end = entity.end_pos
        
        return result
    
    def extract_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[NERResult]:
        """Extract entities from multiple texts"""
        return [self.extract_entities(text, **kwargs) for text in texts]
    
    def get_entity_summary(self, ner_result: NERResult) -> Dict[str, Any]:
        """Get summary statistics for extracted entities"""
        return {
            'total_entities': len(ner_result.entities),
            'entity_types': len(ner_result.entity_counts),
            'entity_counts': ner_result.entity_counts,
            'most_common_type': max(
                ner_result.entity_counts.items(),
                key=lambda x: x[1]
            )[0] if ner_result.entity_counts else None,
            'unique_entities': len(set(e.text.lower() for e in ner_result.entities)),
            'avg_confidence': sum(e.confidence for e in ner_result.entities) / len(ner_result.entities) if ner_result.entities else 0
        }
    
    def export_entities_to_json(
        self,
        ner_result: NERResult,
        output_path: str
    ):
        """Export entities to JSON file"""
        data = {
            'text_length': len(ner_result.text),
            'processing_time': ner_result.processing_time,
            'summary': self.get_entity_summary(ner_result),
            'entities': [e.to_dict() for e in ner_result.entities]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall NER statistics"""
        return {
            'documents_processed': self.stats['documents_processed'],
            'total_entities_found': self.stats['total_entities_found'],
            'avg_entities_per_doc': (
                self.stats['total_entities_found'] / self.stats['documents_processed']
                if self.stats['documents_processed'] > 0 else 0
            ),
            'entities_by_type': dict(self.stats['entities_by_type']),
            'terminology_size': sum(
                len(terms) for terms in self.entity_patterns.values()
            )
        }
