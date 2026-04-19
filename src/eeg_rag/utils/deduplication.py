"""
Deduplication utilities for EEG-RAG.

Handles duplicate detection across papers from multiple sources using:
1. DOI/PMID exact matching
2. Fuzzy title matching for preprints/versions
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : utils.deduplication.PaperDeduplicator
# Requirement  : `PaperDeduplicator` class shall be instantiable and expose the documented interface
# Purpose      : Deduplicate papers from multiple sources
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
# Verification : Instantiate PaperDeduplicator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class PaperDeduplicator:
    """Deduplicate papers from multiple sources."""
    
    # Minimum similarity score for title matching (0-1)
    TITLE_SIMILARITY_THRESHOLD = 0.85
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.__init__
    # Requirement  : `__init__` shall initialize deduplicator
    # Purpose      : Initialize deduplicator
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
        """Initialize deduplicator."""
        self.seen_dois: Set[str] = set()
        self.seen_pmids: Set[str] = set()
        self.seen_arxiv_ids: Set[str] = set()
        self.seen_titles: Dict[str, str] = {}  # normalized_title -> original_doc_id
        
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.normalize_title
    # Requirement  : `normalize_title` shall normalize title for comparison
    # Purpose      : Normalize title for comparison
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : title: str
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
    def normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Handle non-string titles (e.g., float, None)
        if not isinstance(title, str):
            title = str(title) if title is not None else ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove punctuation and extra whitespace
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        
        # Remove common stop words that don't affect identity
        stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
        words = [w for w in title.split() if w not in stop_words]
        
        return ' '.join(words).strip()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.extract_identifiers
    # Requirement  : `extract_identifiers` shall extract DOI, PMID, and arXiv ID from paper metadata
    # Purpose      : Extract DOI, PMID, and arXiv ID from paper metadata
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict
    # Outputs      : Tuple[Optional[str], Optional[str], Optional[str]]
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
    def extract_identifiers(self, paper: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract DOI, PMID, and arXiv ID from paper metadata."""
        # Try different field names for DOI
        doi = paper.get('doi') or paper.get('DOI') or paper.get('external_ids', {}).get('DOI')
        if doi:
            doi = doi.lower().strip()
            # Remove common DOI prefixes
            doi = re.sub(r'^https?://doi\.org/', '', doi)
            doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
        
        # Try different field names for PMID
        pmid = paper.get('pmid') or paper.get('PMID') or paper.get('external_ids', {}).get('PubMed')
        if pmid:
            pmid = str(pmid).strip()
            # Remove "PMID:" prefix if present
            pmid = re.sub(r'^pmid:?\s*', '', pmid, flags=re.IGNORECASE)
        
        # Try different field names for arXiv ID
        arxiv_id = paper.get('arxiv_id') or paper.get('external_ids', {}).get('ArXiv')
        if arxiv_id:
            arxiv_id = arxiv_id.strip()
        
        return doi, pmid, arxiv_id
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.title_similarity
    # Requirement  : `title_similarity` shall calculate similarity between two titles (0-1)
    # Purpose      : Calculate similarity between two titles (0-1)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : title1: str; title2: str
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
    def title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles (0-1)."""
        norm1 = self.normalize_title(title1)
        norm2 = self.normalize_title(title2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use sequence matcher for fuzzy comparison
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.is_duplicate
    # Requirement  : `is_duplicate` shall check if paper is a duplicate
    # Purpose      : Check if paper is a duplicate
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict
    # Outputs      : Tuple[bool, Optional[str]]
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
    def is_duplicate(self, paper: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if paper is a duplicate.
        
        Returns:
            (is_duplicate: bool, reason: str or None)
        """
        # Extract identifiers
        doi, pmid, arxiv_id = self.extract_identifiers(paper)
        
        # Check DOI
        if doi:
            if doi in self.seen_dois:
                return True, f"Duplicate DOI: {doi}"
        
        # Check PMID
        if pmid:
            if pmid in self.seen_pmids:
                return True, f"Duplicate PMID: {pmid}"
        
        # Check arXiv ID
        if arxiv_id:
            if arxiv_id in self.seen_arxiv_ids:
                return True, f"Duplicate arXiv ID: {arxiv_id}"
        
        # Check title similarity
        title = paper.get('title') or paper.get('Title') or ''
        
        # Handle non-string titles
        if title and not isinstance(title, str):
            title = str(title) if title is not None else ''
        
        if title:
            normalized_title = self.normalize_title(title)
            
            # Exact match
            if normalized_title in self.seen_titles:
                title_preview = str(title)[:50] if title else "N/A"
                return True, f"Duplicate title (exact): {title_preview}..."
            
            # Fuzzy match
            for seen_title_norm, seen_doc_id in self.seen_titles.items():
                similarity = self.title_similarity(title, seen_title_norm)
                if similarity >= self.TITLE_SIMILARITY_THRESHOLD:
                    title_preview = str(title)[:50] if title else "N/A"
                    return True, f"Duplicate title (fuzzy {similarity:.2f}): {title_preview}..."
        
        return False, None
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.add_paper
    # Requirement  : `add_paper` shall add paper to seen set after confirming it's not a duplicate
    # Purpose      : Add paper to seen set after confirming it's not a duplicate
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict; doc_id: str
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
    def add_paper(self, paper: Dict, doc_id: str):
        """Add paper to seen set after confirming it's not a duplicate."""
        doi, pmid, arxiv_id = self.extract_identifiers(paper)
        
        if doi:
            self.seen_dois.add(doi)
        if pmid:
            self.seen_pmids.add(pmid)
        if arxiv_id:
            self.seen_arxiv_ids.add(arxiv_id)
        
        title = paper.get('title') or paper.get('Title') or ''
        if title:
            normalized_title = self.normalize_title(title)
            if normalized_title:
                self.seen_titles[normalized_title] = doc_id
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.deduplicate_papers
    # Requirement  : `deduplicate_papers` shall deduplicate a list of papers
    # Purpose      : Deduplicate a list of papers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : papers: List[Dict]
    # Outputs      : Tuple[List[Dict], List[Dict]]
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
    def deduplicate_papers(self, papers: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Deduplicate a list of papers.
        
        Returns:
            (unique_papers, duplicates)
        """
        unique = []
        duplicates = []
        
        self.reset()
        
        for i, paper in enumerate(papers):
            doc_id = paper.get('doc_id') or paper.get('id') or f"paper_{i}"
            
            is_dup, reason = self.is_duplicate(paper)
            
            if is_dup:
                paper['_duplicate_reason'] = reason
                duplicates.append(paper)
                logger.debug(f"Duplicate found: {reason}")
            else:
                unique.append(paper)
                self.add_paper(paper, doc_id)
        
        logger.info(f"Deduplication complete: {len(unique)} unique, {len(duplicates)} duplicates from {len(papers)} total")
        
        return unique, duplicates
    
    # ---------------------------------------------------------------------------
    # ID           : utils.deduplication.PaperDeduplicator.reset
    # Requirement  : `reset` shall reset the deduplicator state
    # Purpose      : Reset the deduplicator state
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
    def reset(self):
        """Reset the deduplicator state."""
        self.seen_dois.clear()
        self.seen_pmids.clear()
        self.seen_arxiv_ids.clear()
        self.seen_titles.clear()


# ---------------------------------------------------------------------------
# ID           : utils.deduplication.deduplicate_papers
# Requirement  : `deduplicate_papers` shall convenience function to deduplicate papers
# Purpose      : Convenience function to deduplicate papers
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : papers: List[Dict]; threshold: float (default=0.85)
# Outputs      : Tuple[List[Dict], List[Dict]]
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
def deduplicate_papers(papers: List[Dict], threshold: float = 0.85) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function to deduplicate papers.
    
    Args:
        papers: List of paper dictionaries
        threshold: Title similarity threshold (0-1)
        
    Returns:
        (unique_papers, duplicate_papers)
    """
    deduplicator = PaperDeduplicator()
    deduplicator.TITLE_SIMILARITY_THRESHOLD = threshold
    return deduplicator.deduplicate_papers(papers)
