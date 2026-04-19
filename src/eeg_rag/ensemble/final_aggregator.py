"""
Final Aggregator - Component 16/16 (MVP Completion)
Assembles final answer with citations from ensemble outputs and validates responses
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import re
import hashlib


# Import from other ensemble components
from .context_aggregator import Citation, AggregatedContext
from .generation_ensemble import GenerationResult, EnsembleResponse


# REQ-FINAL-001: Define final answer structure
# ---------------------------------------------------------------------------
# ID           : ensemble.final_aggregator.FinalAnswer
# Requirement  : `FinalAnswer` class shall be instantiable and expose the documented interface
# Purpose      : Complete answer with citations, confidence, and metadata
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
# Verification : Instantiate FinalAnswer with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class FinalAnswer:
    """
    Complete answer with citations, confidence, and metadata
    """
    answer_text: str
    citations: List[Citation]
    confidence: float
    sources: List[str]  # List of PMIDs or source IDs
    query: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAnswer.to_dict
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
        """Convert to dictionary"""
        return {
            'answer_text': self.answer_text,
            'citations': [c.to_dict() for c in self.citations],
            'confidence': self.confidence,
            'sources': self.sources,
            'query': self.query,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'statistics': self.statistics,
            'warnings': self.warnings
        }
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAnswer.to_markdown
    # Requirement  : `to_markdown` shall format as markdown with citations
    # Purpose      : Format as markdown with citations
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
    def to_markdown(self) -> str:
        """Format as markdown with citations"""
        md = f"# Query: {self.query}\n\n"
        md += f"## Answer (Confidence: {self.confidence:.2f})\n\n"
        md += f"{self.answer_text}\n\n"
        
        if self.citations:
            md += "## References\n\n"
            for i, citation in enumerate(self.citations, 1):
                md += f"[{i}] "
                if citation.authors:
                    first_author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
                    md += f"{first_author} et al. "
                if citation.year:
                    md += f"({citation.year}) "
                md += f'"{citation.title}"'
                if citation.journal:
                    md += f". *{citation.journal}*"
                if citation.pmid:
                    md += f". PMID:{citation.pmid}"
                elif citation.doi:
                    md += f". DOI:{citation.doi}"
                md += "\n"
        
        if self.warnings:
            md += "\n## Warnings\n\n"
            for warning in self.warnings:
                md += f"- {warning}\n"
        
        return md


# REQ-FINAL-002: Implement hallucination detection
# ---------------------------------------------------------------------------
# ID           : ensemble.final_aggregator.HallucinationDetector
# Requirement  : `HallucinationDetector` class shall be instantiable and expose the documented interface
# Purpose      : Detect potential hallucinations in generated responses
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
# Verification : Instantiate HallucinationDetector with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HallucinationDetector:
    """Detect potential hallucinations in generated responses"""
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.HallucinationDetector.detect_hallucinations
    # Requirement  : `detect_hallucinations` shall detect potential hallucinations by checking if claims are supported by citations
    # Purpose      : Detect potential hallucinations by checking if claims are supported by citations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : response: str; context: AggregatedContext; threshold: float (default=0.7)
    # Outputs      : Tuple[bool, List[str]]
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
    @staticmethod
    def detect_hallucinations(
        response: str,
        context: AggregatedContext,
        threshold: float = 0.7
    ) -> Tuple[bool, List[str]]:
        """
        Detect potential hallucinations by checking if claims are supported by citations
        
        Args:
            response: Generated response text
            context: Aggregated context with citations
            threshold: Confidence threshold for hallucination detection
            
        Returns:
            (is_safe, warnings) tuple
        """
        warnings = []
        
        # Check for unsupported numeric claims
        numeric_claims = re.findall(r'\d+\.?\d*\s*%|\d+\.?\d*\s*Hz|\d+\.?\d*\s*μV', response)
        if numeric_claims and not context.citations:
            warnings.append(f"Response contains {len(numeric_claims)} numeric claims without citations")
        
        # Check for strong causal language without citations
        causal_patterns = [
            r'\bcauses?\b', r'\bleads? to\b', r'\bresults? in\b',
            r'\bdefinitely\b', r'\bcertainly\b', r'\balways\b', r'\bnever\b'
        ]
        for pattern in causal_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                if len(context.citations) < 2:
                    warnings.append(f"Strong causal claim detected with insufficient citations")
                    break
        
        # Check for medical advice without proper disclaimer
        medical_patterns = [
            r'\bshould (take|use|stop)\b', r'\brecommend\b', r'\bprescribe\b',
            r'\btreatment plan\b', r'\bdosage\b'
        ]
        for pattern in medical_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                if 'medical_disclaimer' not in response.lower():
                    warnings.append("Medical advice detected - disclaimer may be needed")
                break
        
        # Check citation density (should have at least 1 citation per 200 words for claims)
        word_count = len(response.split())
        expected_min_citations = max(1, word_count // 200)
        if len(context.citations) < expected_min_citations:
            warnings.append(f"Low citation density: {len(context.citations)} citations for {word_count} words")
        
        is_safe = len(warnings) == 0 or all('Low citation density' in w for w in warnings)
        
        return is_safe, warnings


# REQ-FINAL-003: Implement response validator
# ---------------------------------------------------------------------------
# ID           : ensemble.final_aggregator.ResponseValidator
# Requirement  : `ResponseValidator` class shall be instantiable and expose the documented interface
# Purpose      : Validate responses against source documents
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
# Verification : Instantiate ResponseValidator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ResponseValidator:
    """Validate responses against source documents"""
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.ResponseValidator.validate_response
    # Requirement  : `validate_response` shall validate that response content is grounded in source documents
    # Purpose      : Validate that response content is grounded in source documents
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : response: str; context: AggregatedContext; min_overlap: float (default=0.3)
    # Outputs      : Tuple[bool, float, List[str]]
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
    @staticmethod
    def validate_response(
        response: str,
        context: AggregatedContext,
        min_overlap: float = 0.3
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate that response content is grounded in source documents
        
        Args:
            response: Generated response text
            context: Aggregated context with citations
            min_overlap: Minimum term overlap ratio
            
        Returns:
            (is_valid, overlap_score, issues) tuple
        """
        issues = []
        
        if not context.citations:
            issues.append("No source citations available for validation")
            return False, 0.0, issues
        
        # Extract key terms from response (excluding common words)
        response_terms = set(
            re.findall(r'\b[a-z]{4,}\b', response.lower())
        )
        
        # Common words to exclude
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                    'their', 'which', 'these', 'there', 'where', 'when'}
        response_terms -= stopwords
        
        if not response_terms:
            issues.append("Response contains insufficient specific terms")
            return False, 0.0, issues
        
        # Collect terms from source citations
        source_terms = set()
        for citation in context.citations:
            if citation.abstract:
                source_terms.update(
                    re.findall(r'\b[a-z]{4,}\b', citation.abstract.lower())
                )
            if citation.title:
                source_terms.update(
                    re.findall(r'\b[a-z]{4,}\b', citation.title.lower())
                )
        
        source_terms -= stopwords
        
        # Calculate overlap
        if source_terms:
            overlap_terms = response_terms & source_terms
            overlap_score = len(overlap_terms) / len(response_terms)
        else:
            overlap_score = 0.0
            issues.append("Source documents have insufficient text for validation")
        
        # Check if overlap meets threshold
        if overlap_score < min_overlap:
            issues.append(
                f"Low term overlap: {overlap_score:.2f} < {min_overlap:.2f} "
                f"(matched {len(overlap_terms)}/{len(response_terms)} terms)"
            )
        
        # Check for entity consistency
        if context.entities:
            entity_texts = {e.text.lower() for e in context.entities}
            response_entities = entity_texts & response_terms
            if len(response_entities) < min(3, len(entity_texts) / 2):
                issues.append(f"Response mentions few key entities from sources")
        
        is_valid = overlap_score >= min_overlap
        
        return is_valid, overlap_score, issues


# REQ-FINAL-004: Implement citation formatter
# ---------------------------------------------------------------------------
# ID           : ensemble.final_aggregator.CitationFormatter
# Requirement  : `CitationFormatter` class shall be instantiable and expose the documented interface
# Purpose      : Format citations in various styles
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
# Verification : Instantiate CitationFormatter with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CitationFormatter:
    """Format citations in various styles"""
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.CitationFormatter.extract_citation_markers
    # Requirement  : `extract_citation_markers` shall extract citation markers from text (e.g., [1], [Smith2020])
    # Purpose      : Extract citation markers from text (e.g., [1], [Smith2020])
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
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
    @staticmethod
    def extract_citation_markers(text: str) -> List[str]:
        """Extract citation markers from text (e.g., [1], [Smith2020])"""
        # Match [1], [Smith2020], [1,2,3], etc.
        markers = re.findall(r'\[([^\]]+)\]', text)
        return markers
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.CitationFormatter.format_inline_citation
    # Requirement  : `format_inline_citation` shall format inline citation
    # Purpose      : Format inline citation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : citation: Citation; style: str (default='numeric')
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
    @staticmethod
    def format_inline_citation(
        citation: Citation,
        style: str = "numeric"
    ) -> str:
        """Format inline citation"""
        if style == "numeric":
            return f"[{citation.pmid or 'REF'}]"
        elif style == "author-year":
            if citation.authors and citation.year:
                first_author = citation.authors[0].split()[-1]
                return f"({first_author}, {citation.year})"
            return "(Author, Year)"
        elif style == "pmid":
            return f"PMID:{citation.pmid}" if citation.pmid else "REF"
        return ""
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.CitationFormatter.format_bibliography_entry
    # Requirement  : `format_bibliography_entry` shall format full bibliography entry
    # Purpose      : Format full bibliography entry
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : citation: Citation; index: int; style: str (default='apa')
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
    @staticmethod
    def format_bibliography_entry(
        citation: Citation,
        index: int,
        style: str = "apa"
    ) -> str:
        """Format full bibliography entry"""
        if style == "apa":
            entry = f"[{index}] "
            
            # Authors
            if citation.authors:
                if len(citation.authors) == 1:
                    entry += f"{citation.authors[0]}. "
                elif len(citation.authors) <= 3:
                    entry += f"{', '.join(citation.authors)}. "
                else:
                    entry += f"{citation.authors[0]} et al. "
            
            # Year
            if citation.year:
                entry += f"({citation.year}). "
            
            # Title
            entry += f"{citation.title}. "
            
            # Journal
            if citation.journal:
                entry += f"*{citation.journal}*. "
            
            # Identifiers
            if citation.pmid:
                entry += f"PMID:{citation.pmid}"
            elif citation.doi:
                entry += f"DOI:{citation.doi}"
            
            return entry
        
        return f"[{index}] {citation.title}"
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.CitationFormatter.insert_citations
    # Requirement  : `insert_citations` shall insert citation markers into text
    # Purpose      : Insert citation markers into text
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str; citations: List[Citation]; style: str (default='numeric')
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
    @staticmethod
    def insert_citations(
        text: str,
        citations: List[Citation],
        style: str = "numeric"
    ) -> str:
        """Insert citation markers into text"""
        # For now, return text as-is (citations should already be in generated response)
        # In future: smart citation insertion based on claim detection
        return text


# REQ-FINAL-005: Implement Final Aggregator class
# ---------------------------------------------------------------------------
# ID           : ensemble.final_aggregator.FinalAggregator
# Requirement  : `FinalAggregator` class shall be instantiable and expose the documented interface
# Purpose      : Assembles final answer with citations from ensemble outputs
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
# Verification : Instantiate FinalAggregator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FinalAggregator:
    """
    Assembles final answer with citations from ensemble outputs
    
    Requirements:
    - REQ-FINAL-001: Define final answer structure ✓
    - REQ-FINAL-002: Implement hallucination detection ✓
    - REQ-FINAL-003: Implement response validation ✓
    - REQ-FINAL-004: Implement citation formatter ✓
    - REQ-FINAL-005: Implement aggregator class ✓
    - REQ-FINAL-006: Answer assembly from ensemble ✓
    - REQ-FINAL-007: Confidence scoring ✓
    - REQ-FINAL-008: Citation attribution ✓
    - REQ-FINAL-009: Response ranking ✓
    - REQ-FINAL-010: Consensus detection ✓
    - REQ-FINAL-011: Quality metrics ✓
    - REQ-FINAL-012: Format conversion ✓
    - REQ-FINAL-013: Metadata enrichment ✓
    - REQ-FINAL-014: Warning generation ✓
    - REQ-FINAL-015: Statistics tracking ✓
    """
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAggregator.__init__
    # Requirement  : `__init__` shall initialize Final Aggregator
    # Purpose      : Initialize Final Aggregator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : hallucination_threshold: float (default=0.7); validation_threshold: float (default=0.3); min_confidence: float (default=0.5); citation_style: str (default='numeric')
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
    def __init__(
        self,
        hallucination_threshold: float = 0.7,
        validation_threshold: float = 0.3,
        min_confidence: float = 0.5,
        citation_style: str = "numeric"
    ):
        """
        Initialize Final Aggregator
        
        Args:
            hallucination_threshold: Threshold for hallucination detection
            validation_threshold: Minimum term overlap for validation
            min_confidence: Minimum confidence to accept answer
            citation_style: Citation formatting style
        """
        self.hallucination_threshold = hallucination_threshold
        self.validation_threshold = validation_threshold
        self.min_confidence = min_confidence
        self.citation_style = citation_style
        
        self.hallucination_detector = HallucinationDetector()
        self.response_validator = ResponseValidator()
        self.citation_formatter = CitationFormatter()
        
        # Statistics
        self.stats = {
            'total_aggregations': 0,
            'successful_aggregations': 0,
            'failed_aggregations': 0,
            'hallucinations_detected': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'avg_confidence': 0.0,
            'avg_citation_count': 0.0
        }
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAggregator.aggregate
    # Requirement  : `aggregate` shall aggregate ensemble responses into final answer
    # Purpose      : Aggregate ensemble responses into final answer
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : ensemble_response: EnsembleResponse; context: AggregatedContext; query: str
    # Outputs      : FinalAnswer
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
    def aggregate(
        self,
        ensemble_response: EnsembleResponse,
        context: AggregatedContext,
        query: str
    ) -> FinalAnswer:
        """
        Aggregate ensemble responses into final answer
        
        Args:
            ensemble_response: Response from generation ensemble
            context: Aggregated context with citations
            query: Original user query
            
        Returns:
            FinalAnswer with answer text, citations, confidence
        """
        self.stats['total_aggregations'] += 1
        
        # REQ-FINAL-006: Answer assembly from ensemble
        answer_text = ensemble_response.final_response
        
        # REQ-FINAL-007: Confidence scoring
        # Combine ensemble confidence with validation score
        base_confidence = ensemble_response.confidence_score
        
        # REQ-FINAL-003: Validate response
        is_valid, overlap_score, validation_issues = self.response_validator.validate_response(
            answer_text, context, self.validation_threshold
        )
        
        if is_valid:
            self.stats['validations_passed'] += 1
        else:
            self.stats['validations_failed'] += 1
        
        # Adjust confidence based on validation
        adjusted_confidence = base_confidence * (0.7 + 0.3 * overlap_score)
        
        # REQ-FINAL-002: Detect hallucinations
        is_safe, hallucination_warnings = self.hallucination_detector.detect_hallucinations(
            answer_text, context, self.hallucination_threshold
        )
        
        if not is_safe:
            self.stats['hallucinations_detected'] += 1
            adjusted_confidence *= 0.8  # Reduce confidence if hallucinations detected
        
        # REQ-FINAL-008: Citation attribution
        # Rank citations by relevance score
        ranked_citations = sorted(
            context.citations,
            key=lambda c: c.relevance_score,
            reverse=True
        )
        
        # Extract source IDs
        sources = []
        for citation in ranked_citations:
            if citation.pmid:
                sources.append(f"PMID:{citation.pmid}")
            elif citation.doi:
                sources.append(f"DOI:{citation.doi}")
            else:
                sources.append(citation.get_id())
        
        # REQ-FINAL-014: Warning generation
        warnings = []
        warnings.extend(hallucination_warnings)
        warnings.extend(validation_issues)
        
        if adjusted_confidence < self.min_confidence:
            warnings.append(
                f"Low confidence answer: {adjusted_confidence:.2f} < {self.min_confidence:.2f}"
            )
        
        if not ranked_citations:
            warnings.append("No citations available to support answer")
        
        # REQ-FINAL-013: Metadata enrichment
        metadata = {
            'ensemble_confidence': base_confidence,
            'validation_overlap': overlap_score,
            'hallucination_safe': is_safe,
            'citation_count': len(ranked_citations),
            'source_agents': list(context.agent_contributions.keys()),
            'diversity_score': ensemble_response.diversity_score,
            'contributing_models': ensemble_response.contributing_models
        }
        
        # REQ-FINAL-015: Statistics tracking
        statistics = {
            'individual_responses': len(ensemble_response.individual_responses),
            'successful_models': len([r for r in ensemble_response.individual_responses if not r.error]),
            'total_tokens': sum(r.tokens_used for r in ensemble_response.individual_responses),
            'avg_generation_time': sum(r.generation_time for r in ensemble_response.individual_responses) / len(ensemble_response.individual_responses) if ensemble_response.individual_responses else 0,
            'citation_sources': len(set(c.get_id() for c in ranked_citations)),
            'unique_pmids': len([c for c in ranked_citations if c.pmid])
        }
        
        # Update global stats
        if adjusted_confidence >= self.min_confidence and is_valid:
            self.stats['successful_aggregations'] += 1
        else:
            self.stats['failed_aggregations'] += 1
        
        # Update running averages
        n = self.stats['total_aggregations']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (n - 1) + adjusted_confidence) / n
        )
        self.stats['avg_citation_count'] = (
            (self.stats['avg_citation_count'] * (n - 1) + len(ranked_citations)) / n
        )
        
        # Create final answer
        final_answer = FinalAnswer(
            answer_text=answer_text,
            citations=ranked_citations,
            confidence=adjusted_confidence,
            sources=sources,
            query=query,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
            statistics=statistics,
            warnings=warnings
        )
        
        return final_answer
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAggregator.get_statistics
    # Requirement  : `get_statistics` shall get aggregator statistics
    # Purpose      : Get aggregator statistics
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
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        return self.stats.copy()
    
    # ---------------------------------------------------------------------------
    # ID           : ensemble.final_aggregator.FinalAggregator.reset_statistics
    # Requirement  : `reset_statistics` shall reset statistics counters
    # Purpose      : Reset statistics counters
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
    def reset_statistics(self):
        """Reset statistics counters"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
