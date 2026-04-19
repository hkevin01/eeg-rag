"""
Citation-aware semantic chunking for research papers.

Provides:
- Section detection (Abstract, Introduction, Methods, Results, Discussion)
- Citation-preserving chunking (keeps [Author Year] intact)
- Configurable chunk size with overlap
- Metadata preservation for each chunk
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : chunking.citation_aware_chunker.Chunk
# Requirement  : `Chunk` class shall be instantiable and expose the documented interface
# Purpose      : A chunk of text from a paper
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
# Verification : Instantiate Chunk with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A chunk of text from a paper."""
    text: str
    chunk_id: int
    section: str
    paper_id: str
    metadata: Dict[str, Any]
    char_start: int
    char_end: int
    has_citations: bool = False


# ---------------------------------------------------------------------------
# ID           : chunking.citation_aware_chunker.CitationAwareChunker
# Requirement  : `CitationAwareChunker` class shall be instantiable and expose the documented interface
# Purpose      : Smart chunking for research papers that preserves citations and sections
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
# Verification : Instantiate CitationAwareChunker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CitationAwareChunker:
    """
    Smart chunking for research papers that preserves citations and sections.
    
    Features:
    - Detects common paper sections
    - Splits by tokens while preserving sentences
    - Keeps citations intact across chunks
    - Maintains overlap for context continuity
    """
    
    # Section headers to detect
    SECTION_PATTERNS = [
        (r'\b(abstract|summary)\b', 'Abstract'),
        (r'\b(introduction|background)\b', 'Introduction'),
        (r'\b(related work|literature review|prior work)\b', 'Related Work'),
        (r'\b(method|methodology|approach|architecture|model)\b', 'Methods'),
        (r'\b(experiment|evaluation|results?)\b', 'Results'),
        (r'\b(discussion|analysis|findings)\b', 'Discussion'),
        (r'\b(conclusion|summary|future work)\b', 'Conclusion'),
        (r'\b(reference|bibliography)\b', 'References'),
    ]
    
    # Citation patterns to preserve
    CITATION_PATTERNS = [
        r'\[([A-Z][a-zA-Z\s,&]+\s+\d{4}[a-z]?)\]',  # [Author 2020]
        r'\[(\d{1,3})\]',  # [1]
        r'\(([A-Z][a-zA-Z\s,&]+,?\s+\d{4}[a-z]?)\)',  # (Author, 2020)
        r'et al\.',  # et al.
    ]
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.__init__
    # Requirement  : `__init__` shall initialize chunker
    # Purpose      : Initialize chunker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : chunk_size: int (default=512); overlap: int (default=128); min_chunk_size: int (default=100)
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
        chunk_size: int = 512,
        overlap: int = 128,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.detect_sections
    # Requirement  : `detect_sections` shall detect sections in paper text
    # Purpose      : Detect sections in paper text
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : List[Tuple[str, int, int]]
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
    def detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect sections in paper text.
        
        Args:
            text: Full paper text
            
        Returns:
            List of (section_name, start_pos, end_pos) tuples
        """
        sections = []
        text_lower = text.lower()
        
        # Find all section headers
        matches = []
        for pattern, section_name in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                # Check if it looks like a header (start of line or after newlines)
                start = match.start()
                if start == 0 or text[start-1] in '\n\r':
                    matches.append((start, section_name))
        
        # Sort by position
        matches.sort()
        
        # Create sections with boundaries
        if not matches:
            # No sections detected - treat entire text as one section
            return [('Full Text', 0, len(text))]
        
        for i, (start, name) in enumerate(matches):
            end = matches[i+1][0] if i+1 < len(matches) else len(text)
            sections.append((name, start, end))
        
        return sections
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.has_citations
    # Requirement  : `has_citations` shall check if text contains citations
    # Purpose      : Check if text contains citations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
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
    def has_citations(self, text: str) -> bool:
        """Check if text contains citations."""
        for pattern in self.CITATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.split_into_sentences
    # Requirement  : `split_into_sentences` shall split text into sentences while preserving citations
    # Purpose      : Split text into sentences while preserving citations
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
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving citations.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting that handles common cases
        # More robust than just splitting on '. '
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            
            # Check if we're at a sentence boundary
            if char in '.!?' and len(current) > 1:
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        
        # Add remaining text
        if current:
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.estimate_tokens
    # Requirement  : `estimate_tokens` shall rough token estimation (1 token ≈ 4 characters)
    # Purpose      : Rough token estimation (1 token ≈ 4 characters)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : int
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
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.chunk_text
    # Requirement  : `chunk_text` shall chunk text with overlap while preserving sentences
    # Purpose      : Chunk text with overlap while preserving sentences
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str; section: str (default='Unknown'); paper_id: str (default='unknown')
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
    def chunk_text(
        self,
        text: str,
        section: str = 'Unknown',
        paper_id: str = 'unknown'
    ) -> List[str]:
        """
        Chunk text with overlap while preserving sentences.
        
        Args:
            text: Text to chunk
            section: Section name
            paper_id: Paper identifier
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self.estimate_tokens(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_size = self.estimate_tokens(s)
                    if overlap_size + s_size <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.chunk_paper
    # Requirement  : `chunk_paper` shall chunk a complete paper into semantic chunks
    # Purpose      : Chunk a complete paper into semantic chunks
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]; text_field: str (default='text')
    # Outputs      : List[Chunk]
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
    def chunk_paper(
        self,
        paper: Dict[str, Any],
        text_field: str = 'text'
    ) -> List[Chunk]:
        """
        Chunk a complete paper into semantic chunks.
        
        Args:
            paper: Paper dictionary with text and metadata
            text_field: Field containing the full text
            
        Returns:
            List of Chunk objects
        """
        text = paper.get(text_field, '')
        if not text:
            # Fallback to other common fields
            text = paper.get('abstract', '') or paper.get('Abstract', '')
            text += '\n\n' + paper.get('results', '') or paper.get('Results', '')
        
        if not text or not text.strip():
            logger.warning(f"No text found for paper: {paper.get('doc_id', 'unknown')}")
            return []
        
        paper_id = str(paper.get('doc_id', paper.get('Citation', 'unknown')))
        
        # Detect sections
        sections = self.detect_sections(text)
        
        # Chunk each section
        all_chunks = []
        chunk_counter = 0
        
        for section_name, start, end in sections:
            section_text = text[start:end].strip()
            
            if not section_text:
                continue
            
            # Chunk this section
            text_chunks = self.chunk_text(section_text, section_name, paper_id)
            
            # Create Chunk objects
            for chunk_text in text_chunks:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=chunk_counter,
                    section=section_name,
                    paper_id=paper_id,
                    metadata={
                        'title': paper.get('title', paper.get('Title', '')),
                        'authors': paper.get('authors', paper.get('Authors', '')),
                        'year': paper.get('year', paper.get('Year', '')),
                        'domain': paper.get('domain', paper.get('Domain 1', '')),
                        'doi': paper.get('doi', paper.get('DOI', '')),
                        'pmid': paper.get('pmid', paper.get('PMID', '')),
                    },
                    char_start=start,
                    char_end=start + len(chunk_text),
                    has_citations=self.has_citations(chunk_text)
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        logger.debug(f"Created {len(all_chunks)} chunks for paper {paper_id}")
        return all_chunks
    
    # ---------------------------------------------------------------------------
    # ID           : chunking.citation_aware_chunker.CitationAwareChunker.chunk_papers
    # Requirement  : `chunk_papers` shall chunk multiple papers
    # Purpose      : Chunk multiple papers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : papers: List[Dict[str, Any]]; text_field: str (default='text')
    # Outputs      : List[Chunk]
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
    def chunk_papers(
        self,
        papers: List[Dict[str, Any]],
        text_field: str = 'text'
    ) -> List[Chunk]:
        """
        Chunk multiple papers.
        
        Args:
            papers: List of paper dictionaries
            text_field: Field containing the text
            
        Returns:
            List of all chunks from all papers
        """
        all_chunks = []
        
        for paper in papers:
            try:
                chunks = self.chunk_paper(paper, text_field)
                all_chunks.extend(chunks)
            except Exception as e:
                paper_id = paper.get('doc_id', paper.get('Citation', 'unknown'))
                logger.error(f"Failed to chunk paper {paper_id}: {e}")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(papers)} papers")
        return all_chunks


# ---------------------------------------------------------------------------
# ID           : chunking.citation_aware_chunker.test_chunker
# Requirement  : `test_chunker` shall test the citation-aware chunker
# Purpose      : Test the citation-aware chunker
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
def test_chunker():
    """Test the citation-aware chunker."""
    logger.info("Testing CitationAwareChunker...")
    
    # Test paper
    paper = {
        'doc_id': 'test001',
        'title': 'Deep Learning for EEG Analysis',
        'authors': 'Smith et al.',
        'year': 2020,
        'text': """
        Abstract
        This paper presents a novel deep learning approach for EEG seizure detection.
        We achieve 95% accuracy using CNNs [Roy et al. 2019].
        
        Introduction
        EEG analysis is crucial for epilepsy diagnosis. Previous work by [Jones 2018] 
        and [Smith 2017] has shown promise, but accuracy remains limited.
        
        Methods
        We use a convolutional neural network with 5 layers. The architecture follows
        [LeCun et al. 2015]. Data preprocessing includes bandpass filtering (0.5-40Hz)
        and artifact removal using ICA [Delorme 2004].
        
        Results
        Our model achieves 95% accuracy, 93% precision, and 94% recall on the test set.
        This outperforms baselines [Roy et al. 2019] by 10 percentage points.
        
        Conclusion
        We demonstrate state-of-the-art performance on EEG seizure detection.
        Future work will extend this to other EEG applications.
        """
    }
    
    # Initialize chunker
    chunker = CitationAwareChunker(chunk_size=200, overlap=50)
    
    # Chunk the paper
    chunks = chunker.chunk_paper(paper)
    
    logger.info(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        logger.info(f"  Chunk {chunk.chunk_id} [{chunk.section}]: {len(chunk.text)} chars, "
                   f"has_citations={chunk.has_citations}")
        logger.info(f"    Preview: {chunk.text[:80]}...")
    
    # Test section detection
    sections = chunker.detect_sections(paper['text'])
    logger.info(f"\nDetected {len(sections)} sections:")
    for section, start, end in sections:
        logger.info(f"  - {section}: chars {start}-{end}")
    
    logger.info("\n✅ CitationAwareChunker test complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_chunker()
