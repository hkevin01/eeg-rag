"""
Research Gap Detector

Identifies research gaps, methodological limitations, and unanswered questions.

Requirements Covered:
- REQ-SYNTH-020: Research gap identification
- REQ-SYNTH-021: Methodology limitation detection
- REQ-SYNTH-022: Future direction extraction
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : agents.synthesis_agent.gap_detector.GapType
# Requirement  : `GapType` class shall be instantiable and expose the documented interface
# Purpose      : Types of research gaps
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
# Verification : Instantiate GapType with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GapType(Enum):
    """Types of research gaps."""
    METHODOLOGICAL = "methodological"
    SAMPLE_SIZE = "sample_size"
    POPULATION = "population"
    LONGITUDINAL = "longitudinal"
    REPLICATION = "replication"
    MECHANISM = "mechanism"
    CLINICAL = "clinical"
    TECHNOLOGY = "technology"
    STANDARDIZATION = "standardization"


# ---------------------------------------------------------------------------
# ID           : agents.synthesis_agent.gap_detector.ResearchGap
# Requirement  : `ResearchGap` class shall be instantiable and expose the documented interface
# Purpose      : Represents an identified research gap
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
# Verification : Instantiate ResearchGap with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ResearchGap:
    """Represents an identified research gap."""
    gap_type: GapType
    description: str
    confidence: float
    source_papers: List[str] = field(default_factory=list)
    suggested_studies: List[str] = field(default_factory=list)
    priority: float = 0.5
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.ResearchGap.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gap_type": self.gap_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "source_papers": self.source_papers,
            "suggested_studies": self.suggested_studies,
            "priority": self.priority
        }


# ---------------------------------------------------------------------------
# ID           : agents.synthesis_agent.gap_detector.GapDetector
# Requirement  : `GapDetector` class shall be instantiable and expose the documented interface
# Purpose      : Detects research gaps from a corpus of papers
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
# Verification : Instantiate GapDetector with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GapDetector:
    """
    Detects research gaps from a corpus of papers.
    
    Analyzes limitations sections, future work mentions,
    and identifies underexplored areas.
    """
    
    # Limitation indicators
    LIMITATION_PATTERNS = [
        (r"limitation(?:s)?\s+(?:of\s+)?(?:this|the|our)\s+study", 1.0),
        (r"(?:a|one)\s+limitation\s+(?:is|was)", 0.9),
        (r"(?:small|limited)\s+sample\s+size", 0.85),
        (r"lack(?:s|ed|ing)?\s+(?:of\s+)?(?:a\s+)?control\s+group", 0.9),
        (r"(?:cannot|could\s+not)\s+(?:be\s+)?generalized?", 0.8),
        (r"cross[\-\s]?sectional\s+design", 0.7),
        (r"retrospective\s+(?:nature|design|study)", 0.7),
        (r"selection\s+bias", 0.85),
        (r"confounding\s+(?:factors?|variables?)", 0.8),
        (r"further\s+(?:research|studies?|investigation)", 0.75),
    ]
    
    # Future work indicators
    FUTURE_WORK_PATTERNS = [
        (r"future\s+(?:research|studies?|work|investigation)", 1.0),
        (r"(?:should|need\s+to)\s+be\s+(?:further\s+)?(?:investigated|explored|examined)", 0.9),
        (r"remains?\s+(?:to\s+be\s+)?(?:unclear|unknown|unexplored)", 0.85),
        (r"warrants?\s+(?:further|additional)\s+(?:investigation|research|study)", 0.9),
        (r"more\s+(?:research|studies?)\s+(?:is|are)\s+needed", 0.85),
        (r"(?:yet\s+)?to\s+be\s+(?:determined|established|elucidated)", 0.8),
    ]
    
    # EEG-specific gap patterns
    EEG_GAP_PATTERNS = {
        GapType.METHODOLOGICAL: [
            (r"(?:different|varying|inconsistent)\s+(?:EEG\s+)?(?:methods?|methodolog(?:y|ies)|techniques?)", 0.8),
            (r"lack\s+of\s+(?:standardized?|uniform)\s+(?:methods?|protocols?)", 0.9),
            (r"(?:artifact|noise)\s+(?:removal|rejection)\s+(?:varies?|differ)", 0.7),
        ],
        GapType.SAMPLE_SIZE: [
            (r"(?:small|limited|modest)\s+(?:sample\s+size|number\s+of\s+(?:subjects?|participants?|patients?))", 0.9),
            (r"(?:only\s+)?\d{1,2}\s+(?:subjects?|participants?|patients?)", 0.85),
            (r"n\s*=\s*\d{1,2}\b", 0.8),
        ],
        GapType.POPULATION: [
            (r"(?:limited|lack\s+of)\s+(?:diverse|varied|heterogeneous)\s+(?:population|sample)", 0.85),
            (r"(?:only\s+)?(?:adult|pediatric|elderly)\s+(?:population|subjects?|patients?)", 0.7),
            (r"(?:single|one)\s+(?:center|site|hospital)", 0.75),
            (r"(?:healthy|normal)\s+(?:subjects?|controls?)\s+only", 0.7),
        ],
        GapType.LONGITUDINAL: [
            (r"cross[\-\s]?sectional\s+(?:design|study|nature)", 0.8),
            (r"lack\s+of\s+(?:follow[\-\s]?up|longitudinal)\s+data", 0.9),
            (r"(?:no|without)\s+long[\-\s]?term\s+(?:follow[\-\s]?up|assessment)", 0.85),
            (r"single\s+(?:time[\-\s]?point|session|recording)", 0.75),
        ],
        GapType.REPLICATION: [
            (r"(?:needs?|requires?)\s+(?:independent\s+)?(?:replication|validation)", 0.9),
            (r"(?:first|initial|preliminary)\s+(?:study|findings?|results?)", 0.7),
            (r"(?:not\s+)?(?:been\s+)?replicated", 0.85),
            (r"(?:single|one)\s+(?:study|dataset|cohort)", 0.75),
        ],
        GapType.MECHANISM: [
            (r"(?:underlying|neural|neurophysiological)\s+mechanism", 0.8),
            (r"(?:unclear|unknown)\s+(?:how|why|mechanism)", 0.85),
            (r"(?:causal|mechanistic)\s+(?:relationship|link)", 0.8),
        ],
        GapType.CLINICAL: [
            (r"clinical\s+(?:utility|application|relevance|translation)", 0.8),
            (r"(?:real[\-\s]?world|clinical)\s+(?:setting|practice|validation)", 0.85),
            (r"(?:diagnostic|prognostic)\s+(?:accuracy|value|utility)", 0.8),
        ],
        GapType.TECHNOLOGY: [
            (r"(?:limited|outdated|older)\s+(?:equipment|technology|hardware)", 0.8),
            (r"(?:low[\-\s]?)?density\s+EEG", 0.7),
            (r"(?:standard|conventional)\s+(?:EEG\s+)?(?:system|equipment)", 0.6),
        ],
        GapType.STANDARDIZATION: [
            (r"(?:lack|absence)\s+of\s+(?:standard|consensus|guidelines?)", 0.9),
            (r"(?:no|without)\s+(?:gold\s+)?standard", 0.85),
            (r"(?:different|varying|inconsistent)\s+(?:definitions?|criteria|thresholds?)", 0.8),
        ],
    }
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector.__init__
    # Requirement  : `__init__` shall initialize gap detector
    # Purpose      : Initialize gap detector
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def __init__(self):
        """Initialize gap detector."""
        # Compile patterns
        self._limitation_patterns = [
            (re.compile(p, re.IGNORECASE), w) for p, w in self.LIMITATION_PATTERNS
        ]
        self._future_patterns = [
            (re.compile(p, re.IGNORECASE), w) for p, w in self.FUTURE_WORK_PATTERNS
        ]
        self._gap_patterns = {
            gap_type: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
            for gap_type, patterns in self.EEG_GAP_PATTERNS.items()
        }
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector.detect_gaps
    # Requirement  : `detect_gaps` shall detect research gaps from a collection of papers
    # Purpose      : Detect research gaps from a collection of papers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : papers: List[Dict[str, Any]]; min_confidence: float (default=0.5)
    # Outputs      : List[ResearchGap]
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
    def detect_gaps(
        self,
        papers: List[Dict[str, Any]],
        min_confidence: float = 0.5
    ) -> List[ResearchGap]:
        """
        Detect research gaps from a collection of papers.
        
        Args:
            papers: List of paper dictionaries
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of identified research gaps
        """
        all_gaps: Dict[GapType, List[Tuple[str, float, str]]] = {
            gap_type: [] for gap_type in GapType
        }
        
        for paper in papers:
            paper_id = paper.get("pmid") or paper.get("paper_id") or paper.get("title", "")[:50]
            text = self._get_paper_text(paper)
            
            # Detect gaps from this paper
            for gap_type, patterns in self._gap_patterns.items():
                for pattern, weight in patterns:
                    match = pattern.search(text)
                    if match:
                        # Extract context around match
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end].strip()
                        all_gaps[gap_type].append((context, weight, paper_id))
        
        # Aggregate and create ResearchGap objects
        gaps = []
        for gap_type, findings in all_gaps.items():
            if not findings:
                continue
            
            # Aggregate confidence
            avg_confidence = sum(f[1] for f in findings) / len(findings)
            if avg_confidence < min_confidence:
                continue
            
            # Get unique source papers
            source_papers = list(set(f[2] for f in findings))
            
            # Generate description
            description = self._generate_gap_description(gap_type, findings)
            
            # Calculate priority based on frequency and confidence
            priority = min(1.0, (len(findings) / len(papers)) * avg_confidence * 2)
            
            # Generate suggested studies
            suggested = self._generate_suggestions(gap_type)
            
            gaps.append(ResearchGap(
                gap_type=gap_type,
                description=description,
                confidence=avg_confidence,
                source_papers=source_papers[:10],  # Limit
                suggested_studies=suggested,
                priority=priority
            ))
        
        # Sort by priority
        gaps.sort(key=lambda g: g.priority, reverse=True)
        
        return gaps
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector._get_paper_text
    # Requirement  : `_get_paper_text` shall extract searchable text from paper
    # Purpose      : Extract searchable text from paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]
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
    def _get_paper_text(self, paper: Dict[str, Any]) -> str:
        """Extract searchable text from paper."""
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        return f"{title} {abstract}"
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector._generate_gap_description
    # Requirement  : `_generate_gap_description` shall generate a description for the identified gap
    # Purpose      : Generate a description for the identified gap
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : gap_type: GapType; findings: List[Tuple[str, float, str]]
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
    def _generate_gap_description(
        self,
        gap_type: GapType,
        findings: List[Tuple[str, float, str]]
    ) -> str:
        """Generate a description for the identified gap."""
        descriptions = {
            GapType.METHODOLOGICAL: "Inconsistent or non-standardized methodological approaches across studies limit comparability and reproducibility of findings.",
            GapType.SAMPLE_SIZE: "Many studies are limited by small sample sizes, reducing statistical power and generalizability of results.",
            GapType.POPULATION: "Research has focused on limited populations, restricting applicability to diverse patient groups or clinical contexts.",
            GapType.LONGITUDINAL: "Cross-sectional designs predominate, limiting understanding of temporal dynamics and long-term outcomes.",
            GapType.REPLICATION: "Key findings await independent replication and validation in larger or different cohorts.",
            GapType.MECHANISM: "The underlying neural mechanisms remain incompletely understood and require further investigation.",
            GapType.CLINICAL: "Translation of research findings to clinical practice needs additional validation and utility studies.",
            GapType.TECHNOLOGY: "Technological limitations in EEG equipment and analysis methods may constrain findings.",
            GapType.STANDARDIZATION: "Lack of standardized definitions, protocols, or criteria limits cross-study comparisons.",
        }
        
        base_desc = descriptions.get(gap_type, "Research gap identified.")
        paper_count = len(set(f[2] for f in findings))
        
        return f"{base_desc} Identified across {paper_count} paper(s)."
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector._generate_suggestions
    # Requirement  : `_generate_suggestions` shall generate suggested studies to address the gap
    # Purpose      : Generate suggested studies to address the gap
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : gap_type: GapType
    # Outputs      : List[str]
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
    def _generate_suggestions(self, gap_type: GapType) -> List[str]:
        """Generate suggested studies to address the gap."""
        suggestions = {
            GapType.METHODOLOGICAL: [
                "Conduct systematic comparison of methodological approaches",
                "Develop and validate standardized protocols",
                "Multi-center study with harmonized methods"
            ],
            GapType.SAMPLE_SIZE: [
                "Large-scale multi-center cohort study",
                "Meta-analysis of existing smaller studies",
                "Collaborative data sharing initiative"
            ],
            GapType.POPULATION: [
                "Studies including diverse demographic groups",
                "Multi-center international studies",
                "Age-stratified population studies"
            ],
            GapType.LONGITUDINAL: [
                "Prospective longitudinal cohort study",
                "Long-term follow-up of existing cohorts",
                "Repeated measures study design"
            ],
            GapType.REPLICATION: [
                "Independent replication study",
                "External validation in different cohort",
                "Pre-registered replication attempt"
            ],
            GapType.MECHANISM: [
                "Mechanistic investigation with multimodal imaging",
                "Animal or computational modeling studies",
                "Combined EEG-fMRI investigation"
            ],
            GapType.CLINICAL: [
                "Clinical validation study",
                "Real-world effectiveness trial",
                "Implementation research in clinical settings"
            ],
            GapType.TECHNOLOGY: [
                "Studies using high-density EEG systems",
                "Comparison of analysis pipelines",
                "Technology advancement validation"
            ],
            GapType.STANDARDIZATION: [
                "Consensus guidelines development",
                "Expert panel recommendations",
                "Comparative methodology study"
            ],
        }
        
        return suggestions.get(gap_type, ["Additional research needed"])
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector.extract_limitations
    # Requirement  : `extract_limitations` shall extract explicit limitations from a single paper
    # Purpose      : Extract explicit limitations from a single paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]
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
    def extract_limitations(
        self,
        paper: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract explicit limitations from a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            List of limitation descriptions
        """
        text = self._get_paper_text(paper)
        limitations = []
        
        for pattern, weight in self._limitation_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 150)
                context = text[start:end].strip()
                
                limitations.append({
                    "text": context,
                    "confidence": weight,
                    "pattern_matched": match.group()
                })
        
        return limitations
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector.extract_future_directions
    # Requirement  : `extract_future_directions` shall extract future research directions from a single paper
    # Purpose      : Extract future research directions from a single paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]
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
    def extract_future_directions(
        self,
        paper: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract future research directions from a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            List of future direction descriptions
        """
        text = self._get_paper_text(paper)
        directions = []
        
        for pattern, weight in self._future_patterns:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 150)
                context = text[start:end].strip()
                
                directions.append({
                    "text": context,
                    "confidence": weight,
                    "pattern_matched": match.group()
                })
        
        return directions
    
    # ---------------------------------------------------------------------------
    # ID           : agents.synthesis_agent.gap_detector.GapDetector.get_gap_summary
    # Requirement  : `get_gap_summary` shall get summary of detected gaps
    # Purpose      : Get summary of detected gaps
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : gaps: List[ResearchGap]
    # Outputs      : Dict[str, Any]
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
    def get_gap_summary(
        self,
        gaps: List[ResearchGap]
    ) -> Dict[str, Any]:
        """
        Get summary of detected gaps.
        
        Args:
            gaps: List of research gaps
            
        Returns:
            Summary statistics
        """
        if not gaps:
            return {
                "total_gaps": 0,
                "by_type": {},
                "high_priority_count": 0,
                "average_confidence": 0.0
            }
        
        by_type = {}
        for gap in gaps:
            type_name = gap.gap_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total_gaps": len(gaps),
            "by_type": by_type,
            "high_priority_count": sum(1 for g in gaps if g.priority >= 0.7),
            "average_confidence": sum(g.confidence for g in gaps) / len(gaps),
            "top_gaps": [g.to_dict() for g in gaps[:5]]
        }
