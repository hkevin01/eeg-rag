"""
Influence Scorer for Semantic Scholar Papers

Calculates research influence scores based on multiple factors.

Requirements Covered:
- REQ-S2-001: Citation impact assessment
- REQ-S2-002: Influential citation identification
"""

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class InfluenceScorer:
    """Score papers by research influence."""
    
    # Top venues in EEG/neuroscience research
    TOP_VENUES = {
        # Nature family
        "nature", "nature neuroscience", "nature methods", "nature communications",
        # Science family  
        "science", "science advances",
        # High-impact journals
        "cell", "pnas", "lancet", "lancet neurology",
        "new england journal of medicine", "nejm",
        # Neuro specialty
        "neuroimage", "brain", "cerebral cortex", "journal of neuroscience",
        "human brain mapping", "clinical neurophysiology",
        # IEEE/computing
        "ieee transactions on neural systems and rehabilitation engineering",
        "ieee transactions on biomedical engineering",
        "ieee transactions on pattern analysis",
        # ML venues
        "neurips", "icml", "iclr", "aaai", "ijcai",
    }
    
    # Venue impact multipliers
    VENUE_MULTIPLIERS = {
        "nature": 1.0,
        "science": 1.0,
        "cell": 0.95,
        "pnas": 0.85,
        "neuroimage": 0.80,
        "brain": 0.80,
        "ieee transactions": 0.75,
        "neurips": 0.85,
        "icml": 0.82,
    }
    
    def __init__(self, current_year: int = 2026):
        """
        Initialize influence scorer.
        
        Args:
            current_year: Current year for recency calculations
        """
        self.current_year = current_year
        logger.debug("InfluenceScorer initialized")
    
    def calculate_influence_score(
        self,
        citation_count: int = 0,
        influential_citation_count: int = 0,
        year: Optional[int] = None,
        venue: Optional[str] = None,
        reference_count: int = 0,
        is_open_access: bool = False,
        fields_of_study: Optional[list] = None
    ) -> float:
        """
        Calculate influence score based on multiple factors.
        
        Args:
            citation_count: Total citations
            influential_citation_count: Citations deemed influential by S2
            year: Publication year
            venue: Journal/conference name
            reference_count: Number of references
            is_open_access: Whether paper is open access
            fields_of_study: List of fields
            
        Returns:
            Influence score between 0 and 1
        """
        scores = []
        weights = []
        
        # 1. Citation component (35%)
        citation_score = self._calculate_citation_score(citation_count)
        scores.append(citation_score)
        weights.append(0.35)
        
        # 2. Influential citation ratio (25%)
        influential_score = self._calculate_influential_ratio(
            citation_count, influential_citation_count
        )
        scores.append(influential_score)
        weights.append(0.25)
        
        # 3. Recency component (20%)
        recency_score = self._calculate_recency_score(year)
        scores.append(recency_score)
        weights.append(0.20)
        
        # 4. Venue component (15%)
        venue_score = self._calculate_venue_score(venue)
        scores.append(venue_score)
        weights.append(0.15)
        
        # 5. Accessibility bonus (5%)
        access_score = 1.0 if is_open_access else 0.3
        scores.append(access_score)
        weights.append(0.05)
        
        # Weighted combination
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, total_score))
    
    def _calculate_citation_score(self, citation_count: int) -> float:
        """Calculate citation score using log scale."""
        if citation_count <= 0:
            return 0.0
        
        # Log scale normalization
        # 1 citation -> ~0.2
        # 10 citations -> ~0.5
        # 100 citations -> ~0.8
        # 1000+ citations -> ~1.0
        log_citations = math.log10(citation_count + 1)
        normalized = min(1.0, log_citations / 3.0)  # log10(1000) = 3
        
        return normalized
    
    def _calculate_influential_ratio(
        self,
        citation_count: int,
        influential_count: int
    ) -> float:
        """Calculate influential citation ratio."""
        if citation_count <= 0:
            return 0.0
        
        if influential_count <= 0:
            return 0.1  # Base score even without influential citations
        
        ratio = influential_count / citation_count
        
        # Boost for having any influential citations
        base_boost = 0.3
        ratio_contribution = ratio * 0.7
        
        return min(1.0, base_boost + ratio_contribution)
    
    def _calculate_recency_score(self, year: Optional[int]) -> float:
        """Calculate recency score with decay."""
        if year is None:
            return 0.5  # Unknown year gets middle score
        
        age = self.current_year - year
        
        if age <= 0:
            return 1.0  # Current or future year
        elif age <= 2:
            return 0.95  # Very recent
        elif age <= 5:
            return 0.85 - (age - 2) * 0.05  # Recent
        elif age <= 10:
            return 0.70 - (age - 5) * 0.04  # Moderate
        elif age <= 20:
            return 0.50 - (age - 10) * 0.02  # Older
        else:
            return max(0.1, 0.30 - (age - 20) * 0.01)  # Very old
    
    def _calculate_venue_score(self, venue: Optional[str]) -> float:
        """Calculate venue prestige score."""
        if venue is None:
            return 0.4  # Unknown venue
        
        venue_lower = venue.lower()
        
        # Check for exact matches or partial matches
        for top_venue in self.TOP_VENUES:
            if top_venue in venue_lower:
                # Check multipliers
                for mult_key, mult_val in self.VENUE_MULTIPLIERS.items():
                    if mult_key in venue_lower:
                        return mult_val
                return 0.75  # Default top venue score
        
        # Check for known patterns
        if "ieee" in venue_lower:
            return 0.70
        if "acm" in venue_lower:
            return 0.65
        if "springer" in venue_lower or "elsevier" in venue_lower:
            return 0.55
        if "arxiv" in venue_lower:
            return 0.35  # Preprints
        
        return 0.4  # Unknown venue
    
    def score_paper(self, paper: Dict[str, Any]) -> float:
        """
        Calculate influence score for a paper dictionary.
        
        Args:
            paper: Paper data dictionary with S2 fields
            
        Returns:
            Influence score
        """
        return self.calculate_influence_score(
            citation_count=paper.get("citation_count", 0) or 0,
            influential_citation_count=paper.get("influential_citation_count", 0) or 0,
            year=paper.get("year"),
            venue=paper.get("venue") or paper.get("journal"),
            reference_count=paper.get("reference_count", 0) or 0,
            is_open_access=paper.get("is_open_access", False),
            fields_of_study=paper.get("fields_of_study")
        )
    
    def rank_papers(
        self,
        papers: list,
        key: str = "influence_score"
    ) -> list:
        """
        Rank papers by influence score.
        
        Args:
            papers: List of paper dictionaries
            key: Key to store score under
            
        Returns:
            Sorted list with scores added
        """
        for paper in papers:
            paper[key] = self.score_paper(paper)
        
        return sorted(papers, key=lambda p: p.get(key, 0), reverse=True)
