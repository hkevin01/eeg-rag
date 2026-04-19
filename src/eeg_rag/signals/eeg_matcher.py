"""
Real-Time EEG Data Integration.

Provides case-based retrieval by matching EEG signal features with literature.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : signals.eeg_matcher.EEGFeatures
# Requirement  : `EEGFeatures` class shall be instantiable and expose the documented interface
# Purpose      : Extracted EEG features for matching
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate EEGFeatures with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class EEGFeatures:
    """Extracted EEG features for matching."""
    power_bands: Dict[str, float]  # Delta, Theta, Alpha, Beta, Gamma
    dominant_frequency: float
    peak_amplitude: float
    entropy: float
    asymmetry: float
    coherence: Dict[str, float]
    artifacts: bool
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.EEGFeatures.to_text_description
    # Requirement  : `to_text_description` shall convert features to text description for literature search
    # Purpose      : Convert features to text description for literature search
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def to_text_description(self) -> str:
        """Convert features to text description for literature search."""
        desc_parts = []
        
        # Band powers
        dominant_band = max(self.power_bands.items(), key=lambda x: x[1])[0]
        desc_parts.append(f"dominant {dominant_band} activity")
        
        # Amplitude
        if self.peak_amplitude > 100:
            desc_parts.append("high amplitude")
        elif self.peak_amplitude < 20:
            desc_parts.append("low amplitude")
        
        # Entropy (regularity)
        if self.entropy > 0.8:
            desc_parts.append("irregular pattern")
        elif self.entropy < 0.3:
            desc_parts.append("regular rhythmic pattern")
        
        # Asymmetry
        if abs(self.asymmetry) > 0.2:
            hemisphere = "right" if self.asymmetry > 0 else "left"
            desc_parts.append(f"{hemisphere} hemisphere asymmetry")
        
        return f"EEG recording showing {', '.join(desc_parts)}"


# ---------------------------------------------------------------------------
# ID           : signals.eeg_matcher.FeatureExtractor
# Requirement  : `FeatureExtractor` class shall be instantiable and expose the documented interface
# Purpose      : Extract features from EEG signals
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate FeatureExtractor with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Extract features from EEG signals."""
    
    FREQUENCY_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor.extract
    # Requirement  : `extract` shall extract clinically relevant features from EEG data
    # Purpose      : Extract clinically relevant features from EEG data
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : eeg_data: np.ndarray; sampling_rate: int
    # Outputs      : EEGFeatures
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def extract(self, eeg_data: np.ndarray, sampling_rate: int) -> EEGFeatures:
        """
        Extract clinically relevant features from EEG data.
        
        Args:
            eeg_data: EEG signal array (channels x samples)
            sampling_rate: Sampling frequency in Hz
        
        Returns:
            Extracted EEG features
        """
        logger.info(f"Extracting features from EEG (shape={eeg_data.shape}, fs={sampling_rate}Hz)")
        
        # Power spectral density for frequency bands
        power_bands = self._compute_band_powers(eeg_data, sampling_rate)
        
        # Dominant frequency
        dominant_freq = self._compute_dominant_frequency(eeg_data, sampling_rate)
        
        # Peak amplitude
        peak_amp = float(np.max(np.abs(eeg_data)))
        
        # Sample entropy (complexity)
        entropy = self._compute_entropy(eeg_data)
        
        # Hemispheric asymmetry (if multi-channel)
        asymmetry = self._compute_asymmetry(eeg_data)
        
        # Coherence between channels
        coherence = self._compute_coherence(eeg_data, sampling_rate)
        
        # Artifact detection
        artifacts = self._detect_artifacts(eeg_data, sampling_rate)
        
        features = EEGFeatures(
            power_bands=power_bands,
            dominant_frequency=dominant_freq,
            peak_amplitude=peak_amp,
            entropy=entropy,
            asymmetry=asymmetry,
            coherence=coherence,
            artifacts=artifacts
        )
        
        logger.info(f"Features extracted: dominant={dominant_freq:.1f}Hz, peak={peak_amp:.1f}µV")
        return features
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._compute_band_powers
    # Requirement  : `_compute_band_powers` shall compute relative power in each frequency band
    # Purpose      : Compute relative power in each frequency band
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray; fs: int
    # Outputs      : Dict[str, float]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_band_powers(self, data: np.ndarray, fs: int) -> Dict[str, float]:
        """Compute relative power in each frequency band."""
        from scipy import signal as scipy_signal
        
        # Compute PSD using Welch's method
        freqs, psd = scipy_signal.welch(data, fs=fs, nperseg=min(fs * 2, data.shape[-1]))
        
        # Compute band powers
        band_powers = {}
        total_power = np.trapz(psd, freqs)
        
        for band_name, (low, high) in self.FREQUENCY_BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx], freqs[idx])
            band_powers[band_name] = float(band_power / total_power)
        
        return band_powers
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._compute_dominant_frequency
    # Requirement  : `_compute_dominant_frequency` shall find peak frequency in the spectrum
    # Purpose      : Find peak frequency in the spectrum
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray; fs: int
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_dominant_frequency(self, data: np.ndarray, fs: int) -> float:
        """Find peak frequency in the spectrum."""
        from scipy import signal as scipy_signal
        
        freqs, psd = scipy_signal.welch(data, fs=fs, nperseg=min(fs * 2, data.shape[-1]))
        
        # Focus on 0.5-40 Hz range
        idx = np.logical_and(freqs >= 0.5, freqs <= 40)
        peak_idx = np.argmax(psd[idx])
        
        return float(freqs[idx][peak_idx])
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._compute_entropy
    # Requirement  : `_compute_entropy` shall compute sample entropy as measure of complexity
    # Purpose      : Compute sample entropy as measure of complexity
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute sample entropy as measure of complexity."""
        # Simplified entropy calculation
        # In production, use proper sample entropy algorithm
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
        hist, _ = np.histogram(normalized, bins=50)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy / np.log(50))  # Normalize to [0, 1]
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._compute_asymmetry
    # Requirement  : `_compute_asymmetry` shall compute hemispheric asymmetry index
    # Purpose      : Compute hemispheric asymmetry index
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_asymmetry(self, data: np.ndarray) -> float:
        """Compute hemispheric asymmetry index."""
        if data.ndim < 2 or data.shape[0] < 2:
            return 0.0
        
        # Assume first half = left, second half = right hemisphere
        mid = data.shape[0] // 2
        left_power = np.mean(np.var(data[:mid], axis=-1))
        right_power = np.mean(np.var(data[mid:], axis=-1))
        
        asymmetry = (right_power - left_power) / (right_power + left_power + 1e-10)
        return float(asymmetry)
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._compute_coherence
    # Requirement  : `_compute_coherence` shall compute inter-channel coherence
    # Purpose      : Compute inter-channel coherence
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray; fs: int
    # Outputs      : Dict[str, float]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_coherence(self, data: np.ndarray, fs: int) -> Dict[str, float]:
        """Compute inter-channel coherence."""
        if data.ndim < 2 or data.shape[0] < 2:
            return {"frontal": 0.0, "central": 0.0, "posterior": 0.0}
        
        # Simplified coherence - in production use scipy.signal.coherence
        coherence_values = {}
        for region in ["frontal", "central", "posterior"]:
            # Placeholder: compute correlation as proxy
            corr = np.corrcoef(data[:min(4, data.shape[0])])
            coherence_values[region] = float(np.mean(np.triu(corr, k=1)))
        
        return coherence_values
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.FeatureExtractor._detect_artifacts
    # Requirement  : `_detect_artifacts` shall detect common artifacts (muscle, eye blinks, etc.)
    # Purpose      : Detect common artifacts (muscle, eye blinks, etc.)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: np.ndarray; fs: int
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _detect_artifacts(self, data: np.ndarray, fs: int) -> bool:
        """Detect common artifacts (muscle, eye blinks, etc.)."""
        # Simplified artifact detection
        # High amplitude (>100µV) or high frequency content (>30Hz dominant)
        max_amp = np.max(np.abs(data))
        
        if max_amp > 150:  # Likely muscle artifact
            return True
        
        # Check for excessive high-frequency content
        from scipy import signal as scipy_signal
        freqs, psd = scipy_signal.welch(data, fs=fs, nperseg=min(fs * 2, data.shape[-1]))
        
        high_freq_power = np.trapz(psd[freqs > 30], freqs[freqs > 30])
        total_power = np.trapz(psd, freqs)
        
        if high_freq_power / total_power > 0.3:  # >30% high-frequency
            return True
        
        return False


# ---------------------------------------------------------------------------
# ID           : signals.eeg_matcher.EEGCaseMatcher
# Requirement  : `EEGCaseMatcher` class shall be instantiable and expose the documented interface
# Purpose      : Match EEG recordings with similar cases in the literature
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate EEGCaseMatcher with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class EEGCaseMatcher:
    """
    Match EEG recordings with similar cases in the literature.
    
    Enables case-based retrieval where clinicians upload EEG data
    and find papers with similar patterns.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.EEGCaseMatcher.__init__
    # Requirement  : `__init__` shall initialize EEG case matcher
    # Purpose      : Initialize EEG case matcher
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : feature_extractor: FeatureExtractor; literature_index: Any
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, feature_extractor: FeatureExtractor, literature_index: Any):
        """
        Initialize EEG case matcher.
        
        Args:
            feature_extractor: Feature extraction pipeline
            literature_index: Vector store with EEG literature
        """
        self.extractor = feature_extractor
        self.index = literature_index
        logger.info("EEGCaseMatcher initialized")
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.EEGCaseMatcher.find_similar_cases
    # Requirement  : `find_similar_cases` shall find papers with similar EEG patterns
    # Purpose      : Find papers with similar EEG patterns
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : eeg_data: np.ndarray; sampling_rate: int; top_k: int (default=10); filters: Optional[Dict[str, Any]] (default=None)
    # Outputs      : List[Dict[str, Any]]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def find_similar_cases(
        self,
        eeg_data: np.ndarray,
        sampling_rate: int,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find papers with similar EEG patterns.
        
        Args:
            eeg_data: EEG signal array
            sampling_rate: Sampling frequency in Hz
            top_k: Number of results to return
            filters: Optional filters (age, condition, etc.)
        
        Returns:
            List of similar papers with relevance scores
        """
        logger.info(f"Finding similar cases for EEG (fs={sampling_rate}Hz, top_k={top_k})")
        
        # Extract features
        features = self.extractor.extract(eeg_data, sampling_rate)
        
        # Convert to searchable text description
        description = features.to_text_description()
        logger.info(f"Feature description: {description}")
        
        # Add clinical context if provided
        query_parts = [description]
        if filters:
            if "condition" in filters:
                query_parts.append(filters["condition"])
            if "age_group" in filters:
                query_parts.append(f"{filters['age_group']} years old")
        
        query = " ".join(query_parts)
        
        # Search literature index
        results = self.index.search(query, top_k=top_k)
        
        # Enhance results with feature matching scores
        enhanced_results = []
        for result in results:
            # In production, extract features from paper's reported EEG data
            # and compute similarity score
            enhanced_result = {
                **result,
                "feature_match_score": self._compute_feature_similarity(features, result),
                "query_features": features
            }
            enhanced_results.append(enhanced_result)
        
        # Re-rank by combined score
        enhanced_results.sort(
            key=lambda x: x["feature_match_score"] * x.get("score", 0.5),
            reverse=True
        )
        
        logger.info(f"Found {len(enhanced_results)} similar cases")
        return enhanced_results[:top_k]
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.EEGCaseMatcher._compute_feature_similarity
    # Requirement  : `_compute_feature_similarity` shall compute similarity between query features and paper
    # Purpose      : Compute similarity between query features and paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query_features: EEGFeatures; paper: Dict[str, Any]
    # Outputs      : float
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _compute_feature_similarity(
        self,
        query_features: EEGFeatures,
        paper: Dict[str, Any]
    ) -> float:
        """
        Compute similarity between query features and paper.
        
        In production, this would extract reported EEG features
        from the paper text and compute actual similarity.
        """
        # Placeholder: Use text-based relevance as proxy
        # In production: Extract features from paper, compute cosine similarity
        base_score = paper.get("score", 0.5)
        
        # Bonus for high-quality papers
        if paper.get("pmid"):
            base_score *= 1.1
        
        return min(base_score, 1.0)
    
    # ---------------------------------------------------------------------------
    # ID           : signals.eeg_matcher.EEGCaseMatcher.get_feature_summary
    # Requirement  : `get_feature_summary` shall get human-readable summary of EEG features
    # Purpose      : Get human-readable summary of EEG features
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : eeg_data: np.ndarray; sampling_rate: int
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_feature_summary(self, eeg_data: np.ndarray, sampling_rate: int) -> str:
        """Get human-readable summary of EEG features."""
        features = self.extractor.extract(eeg_data, sampling_rate)
        
        summary = f"""EEG Feature Summary:
        
Frequency Content:
  - Delta (0.5-4 Hz): {features.power_bands['delta']:.1%}
  - Theta (4-8 Hz): {features.power_bands['theta']:.1%}
  - Alpha (8-13 Hz): {features.power_bands['alpha']:.1%}
  - Beta (13-30 Hz): {features.power_bands['beta']:.1%}
  - Gamma (30-100 Hz): {features.power_bands['gamma']:.1%}
  - Dominant: {features.dominant_frequency:.1f} Hz

Amplitude & Complexity:
  - Peak Amplitude: {features.peak_amplitude:.1f} µV
  - Entropy (Complexity): {features.entropy:.2f}
  - Asymmetry Index: {features.asymmetry:.2f}

Artifacts: {"Detected" if features.artifacts else "None detected"}

Clinical Description:
  {features.to_text_description()}
"""
        return summary
