"""
Evidence Ranker

Grades evidence quality based on study type, methodology, and clinical relevance.

Requirements Covered:
- REQ-SYNTH-010: Evidence quality assessment
- REQ-SYNTH-011: Study type classification
- REQ-SYNTH-012: Methodology scoring
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EvidenceLevel(Enum):
    """Evidence quality levels based on study design."""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion
    UNKNOWN = "unknown"
    
    @property
    def numeric_score(self) -> float:
        """Convert to numeric score (0-1)."""
        scores = {
            "1a": 1.0, "1b": 0.9,
            "2a": 0.8, "2b": 0.7,
            "3a": 0.6, "3b": 0.5,
            "4": 0.3, "5": 0.1,
            "unknown": 0.0
        }
        return scores.get(self.value, 0.0)


@dataclass
class EvidenceScore:
    """Comprehensive evidence quality score."""
    overall_score: float
    evidence_level: EvidenceLevel
    study_type: str
    sample_size_score: float = 0.0
    methodology_score: float = 0.0
    recency_score: float = 0.0
    citation_score: float = 0.0
    clinical_relevance: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "evidence_level": self.evidence_level.value,
            "study_type": self.study_type,
            "sample_size_score": self.sample_size_score,
            "methodology_score": self.methodology_score,
            "recency_score": self.recency_score,
            "citation_score": self.citation_score,
            "clinical_relevance": self.clinical_relevance,
            "confidence": self.confidence
        }


class EvidenceRanker:
    """
    Ranks evidence quality based on multiple factors.
    
    Implements evidence-based medicine hierarchy with
    EEG-specific considerations.
    """
    
    # Study type patterns
    STUDY_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
        "systematic_review": [
            (r"systematic\s+review", 1.0),
            (r"meta[\-\s]?analysis", 1.0),
            (r"cochrane\s+review", 1.0),
            (r"umbrella\s+review", 0.95),
        ],
        "randomized_controlled_trial": [
            (r"randomized?\s+controlled?\s+trial", 1.0),
            (r"\brct\b", 0.95),
            (r"double[\-\s]?blind", 0.9),
            (r"placebo[\-\s]?controlled", 0.85),
            (r"randomized?\s+clinical\s+trial", 0.95),
        ],
        "cohort_study": [
            (r"cohort\s+study", 1.0),
            (r"prospective\s+study", 0.9),
            (r"longitudinal\s+study", 0.85),
            (r"follow[\-\s]?up\s+study", 0.8),
        ],
        "case_control": [
            (r"case[\-\s]?control", 1.0),
            (r"retrospective\s+analysis", 0.8),
            (r"matched\s+controls?", 0.85),
        ],
        "case_series": [
            (r"case\s+series", 1.0),
            (r"case\s+report", 0.8),
            (r"single[\-\s]?case", 0.7),
        ],
        "cross_sectional": [
            (r"cross[\-\s]?sectional", 1.0),
            (r"survey\s+study", 0.8),
        ],
        "narrative_review": [
            (r"(?<!systematic\s)review", 0.5),
            (r"literature\s+review", 0.5),
            (r"narrative\s+review", 0.6),
        ],
        "animal_study": [
            (r"animal\s+study", 1.0),
            (r"in[\-\s]?vivo", 0.8),
            (r"rodent", 0.9),
            (r"\bmice\b|\bmouse\b|\brat\b|\brats\b", 0.9),
        ],
        "in_vitro": [
            (r"in[\-\s]?vitro", 1.0),
            (r"cell\s+culture", 0.9),
        ]
    }
    
    # Study type to evidence level mapping
    STUDY_TO_LEVEL: Dict[str, EvidenceLevel] = {
        "systematic_review": EvidenceLevel.LEVEL_1A,
        "randomized_controlled_trial": EvidenceLevel.LEVEL_1B,
        "cohort_study": EvidenceLevel.LEVEL_2B,
        "case_control": EvidenceLevel.LEVEL_3B,
        "case_series": EvidenceLevel.LEVEL_4,
        "cross_sectional": EvidenceLevel.LEVEL_4,
        "narrative_review": EvidenceLevel.LEVEL_5,
        "animal_study": EvidenceLevel.LEVEL_5,
        "in_vitro": EvidenceLevel.LEVEL_5,
    }
    
    # EEG-specific quality indicators
    EEG_QUALITY_INDICATORS = {
        "high_density": (r"high[\-\s]?density|hdEEG|\b64\+?\s*channels?|\b128\+?\s*channels?|\b256\s*channels?", 0.1),
        "blind_analysis": (r"blind\s+analysis|blinded?\s+rater", 0.15),
        "standardized_protocol": (r"10[\-\s]?20\s+system|standard\s+montage|international\s+system", 0.05),
        "quantitative": (r"qEEG|quantitative\s+EEG|spectral\s+analysis|power\s+spectral", 0.1),
        "validated_measures": (r"validated|reliability|validity|inter[\-\s]?rater", 0.1),
        "control_group": (r"control\s+group|healthy\s+controls?|age[\-\s]?matched", 0.1),
    }
    
    # Sample size tiers
    SAMPLE_SIZE_TIERS = [
        (1000, 1.0),
        (500, 0.9),
        (200, 0.8),
        (100, 0.7),
        (50, 0.5),
        (20, 0.3),
        (10, 0.2),
        (0, 0.1),
    ]
    
    def __init__(self):
        """Initialize evidence ranker."""
        # Compile patterns for efficiency
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, float]]] = {}
        for study_type, patterns in self.STUDY_PATTERNS.items():
            self._compiled_patterns[study_type] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]
        
        self._eeg_patterns = {
            name: (re.compile(pattern, re.IGNORECASE), weight)
            for name, (pattern, weight) in self.EEG_QUALITY_INDICATORS.items()
        }
    
    def rank_evidence(self, paper: Dict[str, Any]) -> EvidenceScore:
        """
        Calculate comprehensive evidence score for a paper.
        
        Args:
            paper: Paper data dictionary
            
        Returns:
            EvidenceScore with detailed breakdown
        """
        # Get text to analyze
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        pub_types = paper.get("publication_types", []) or []
        full_text = f"{title} {abstract} {' '.join(pub_types)}"
        
        # Detect study type
        study_type, type_confidence = self._detect_study_type(full_text)
        
        # Get evidence level
        evidence_level = self.STUDY_TO_LEVEL.get(study_type, EvidenceLevel.UNKNOWN)
        
        # Calculate component scores
        sample_size_score = self._calculate_sample_size_score(paper)
        methodology_score = self._calculate_methodology_score(full_text)
        recency_score = self._calculate_recency_score(paper.get("year"))
        citation_score = self._calculate_citation_score(paper)
        clinical_relevance = self._calculate_clinical_relevance(full_text)
        
        # Calculate overall score with weights
        overall_score = (
            evidence_level.numeric_score * 0.35 +
            sample_size_score * 0.15 +
            methodology_score * 0.20 +
            recency_score * 0.10 +
            citation_score * 0.10 +
            clinical_relevance * 0.10
        )
        
        return EvidenceScore(
            overall_score=min(1.0, overall_score),
            evidence_level=evidence_level,
            study_type=study_type,
            sample_size_score=sample_size_score,
            methodology_score=methodology_score,
            recency_score=recency_score,
            citation_score=citation_score,
            clinical_relevance=clinical_relevance,
            confidence=type_confidence
        )
    
    def _detect_study_type(self, text: str) -> Tuple[str, float]:
        """
        Detect study type from text.
        
        Returns:
            Tuple of (study_type, confidence)
        """
        best_type = "unknown"
        best_score = 0.0
        
        for study_type, patterns in self._compiled_patterns.items():
            type_score = 0.0
            for pattern, weight in patterns:
                if pattern.search(text):
                    type_score = max(type_score, weight)
            
            if type_score > best_score:
                best_score = type_score
                best_type = study_type
        
        return best_type, best_score
    
    def _calculate_sample_size_score(self, paper: Dict[str, Any]) -> float:
        """Calculate sample size score."""
        # Try to extract sample size from paper data
        sample_size = paper.get("sample_size")
        
        if sample_size is None:
            # Try to extract from abstract
            abstract = paper.get("abstract", "") or ""
            matches = re.findall(r"n\s*=\s*(\d+)|(\d+)\s*participants?|(\d+)\s*subjects?|(\d+)\s*patients?", abstract, re.IGNORECASE)
            if matches:
                # Get the largest number found
                numbers = []
                for match in matches:
                    for group in match:
                        if group:
                            numbers.append(int(group))
                if numbers:
                    sample_size = max(numbers)
        
        if sample_size is None:
            return 0.3  # Default score
        
        # Map to tier
        for threshold, score in self.SAMPLE_SIZE_TIERS:
            if sample_size >= threshold:
                return score
        
        return 0.1
    
    def _calculate_methodology_score(self, text: str) -> float:
        """Calculate methodology quality score based on EEG-specific indicators."""
        score = 0.5  # Base score
        
        for name, (pattern, weight) in self._eeg_patterns.items():
            if pattern.search(text):
                score += weight
        
        return min(1.0, score)
    
    def _calculate_recency_score(self, year: Optional[int]) -> float:
        """Calculate recency score."""
        if not year:
            return 0.3
        
        from datetime import datetime
        current_year = datetime.now().year
        age = current_year - year
        
        if age <= 2:
            return 1.0
        elif age <= 5:
            return 0.9
        elif age <= 10:
            return 0.7
        elif age <= 15:
            return 0.5
        elif age <= 20:
            return 0.3
        else:
            return 0.1
    
    def _calculate_citation_score(self, paper: Dict[str, Any]) -> float:
        """Calculate citation impact score."""
        citations = paper.get("citation_count", 0) or 0
        year = paper.get("year")
        
        if not year:
            # Raw citation score
            if citations >= 100:
                return 1.0
            elif citations >= 50:
                return 0.8
            elif citations >= 20:
                return 0.6
            elif citations >= 5:
                return 0.4
            else:
                return 0.2
        
        # Calculate citations per year
        from datetime import datetime
        age = max(1, datetime.now().year - year)
        citations_per_year = citations / age
        
        if citations_per_year >= 20:
            return 1.0
        elif citations_per_year >= 10:
            return 0.9
        elif citations_per_year >= 5:
            return 0.7
        elif citations_per_year >= 2:
            return 0.5
        elif citations_per_year >= 1:
            return 0.3
        else:
            return 0.1
    
    def _calculate_clinical_relevance(self, text: str) -> float:
        """Calculate clinical relevance score."""
        clinical_indicators = [
            (r"clinical\s+trial", 0.2),
            (r"patient", 0.1),
            (r"diagnosis|diagnostic", 0.15),
            (r"treatment|therapy|therapeutic", 0.15),
            (r"prognosis|prognostic", 0.1),
            (r"outcome", 0.1),
            (r"efficacy|effectiveness", 0.1),
            (r"safety|adverse", 0.05),
        ]
        
        score = 0.0
        for pattern, weight in clinical_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        return min(1.0, score)
    
    def rank_papers(
        self,
        papers: List[Dict[str, Any]],
        min_evidence_level: Optional[EvidenceLevel] = None
    ) -> List[Tuple[Dict[str, Any], EvidenceScore]]:
        """
        Rank a list of papers by evidence quality.
        
        Args:
            papers: List of paper dictionaries
            min_evidence_level: Optional minimum evidence level filter
            
        Returns:
            Sorted list of (paper, score) tuples
        """
        results = []
        
        for paper in papers:
            score = self.rank_evidence(paper)
            
            # Filter by evidence level if specified
            if min_evidence_level:
                if score.evidence_level.numeric_score < min_evidence_level.numeric_score:
                    continue
            
            results.append((paper, score))
        
        # Sort by overall score descending
        results.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return results
    
    def get_evidence_summary(
        self,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a set of papers.
        
        Args:
            papers: List of papers
            
        Returns:
            Summary statistics
        """
        scores = [self.rank_evidence(p) for p in papers]
        
        # Count by evidence level
        level_counts: Dict[str, int] = {}
        for score in scores:
            level = score.evidence_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count by study type
        type_counts: Dict[str, int] = {}
        for score in scores:
            type_counts[score.study_type] = type_counts.get(score.study_type, 0) + 1
        
        # Calculate average scores
        if scores:
            avg_overall = sum(s.overall_score for s in scores) / len(scores)
            avg_methodology = sum(s.methodology_score for s in scores) / len(scores)
            avg_recency = sum(s.recency_score for s in scores) / len(scores)
        else:
            avg_overall = avg_methodology = avg_recency = 0.0
        
        return {
            "total_papers": len(papers),
            "evidence_levels": level_counts,
            "study_types": type_counts,
            "average_overall_score": avg_overall,
            "average_methodology_score": avg_methodology,
            "average_recency_score": avg_recency,
            "high_quality_count": sum(1 for s in scores if s.overall_score >= 0.7),
            "low_quality_count": sum(1 for s in scores if s.overall_score < 0.3),
        }
