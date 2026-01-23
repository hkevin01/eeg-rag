"""
Real-Time EEG Data Integration.

Provides case-based retrieval by matching EEG signal features with literature.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


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


class FeatureExtractor:
    """Extract features from EEG signals."""
    
    FREQUENCY_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    
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
    
    def _compute_dominant_frequency(self, data: np.ndarray, fs: int) -> float:
        """Find peak frequency in the spectrum."""
        from scipy import signal as scipy_signal
        
        freqs, psd = scipy_signal.welch(data, fs=fs, nperseg=min(fs * 2, data.shape[-1]))
        
        # Focus on 0.5-40 Hz range
        idx = np.logical_and(freqs >= 0.5, freqs <= 40)
        peak_idx = np.argmax(psd[idx])
        
        return float(freqs[idx][peak_idx])
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute sample entropy as measure of complexity."""
        # Simplified entropy calculation
        # In production, use proper sample entropy algorithm
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-10)
        hist, _ = np.histogram(normalized, bins=50)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy / np.log(50))  # Normalize to [0, 1]
    
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


class EEGCaseMatcher:
    """
    Match EEG recordings with similar cases in the literature.
    
    Enables case-based retrieval where clinicians upload EEG data
    and find papers with similar patterns.
    """
    
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
