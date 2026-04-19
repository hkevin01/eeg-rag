"""
Intelligent document chunking for EEG literature.
"""

import re
from dataclasses import dataclass
from typing import Iterator, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# ID           : ingestion.chunker.ChunkType
# Requirement  : `ChunkType` class shall be instantiable and expose the documented interface
# Purpose      : Types of document chunks for classification
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
# Verification : Instantiate ChunkType with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ChunkType(Enum):
    """Types of document chunks for classification."""
    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    FULL_TEXT = "full_text"


# ---------------------------------------------------------------------------
# ID           : ingestion.chunker.DocumentChunk
# Requirement  : `DocumentChunk` class shall be instantiable and expose the documented interface
# Purpose      : A chunk of a document for indexing
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
# Verification : Instantiate DocumentChunk with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """A chunk of a document for indexing."""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: ChunkType
    chunk_index: int
    
    # Metadata preserved from parent
    pmid: Optional[str]
    doi: Optional[str]
    title: str
    authors: list[str]
    publication_year: Optional[int]
    journal: str
    
    # Chunk-specific
    start_char: int
    end_char: int
    token_count: int


# ---------------------------------------------------------------------------
# ID           : ingestion.chunker.EEGDocumentChunker
# Requirement  : `EEGDocumentChunker` class shall be instantiable and expose the documented interface
# Purpose      : Chunker optimized for EEG research documents
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
# Verification : Instantiate EEGDocumentChunker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class EEGDocumentChunker:
    """
    Chunker optimized for EEG research documents.
    Uses semantic boundaries and preserves important context.
    """
    
    SECTION_PATTERNS = {
        ChunkType.INTRODUCTION: r'(?i)^\s*(introduction|background)\s*$',
        ChunkType.METHODS: r'(?i)^\s*(methods?|materials?\s*(and|&)\s*methods?|experimental\s*design)\s*$',
        ChunkType.RESULTS: r'(?i)^\s*(results?|findings)\s*$',
        ChunkType.DISCUSSION: r'(?i)^\s*(discussion|interpretation)\s*$',
        ChunkType.CONCLUSION: r'(?i)^\s*(conclusions?|summary)\s*$',
    }
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker.__init__
    # Requirement  : `__init__` shall initialize document chunker
    # Purpose      : Initialize document chunker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : chunk_size: int (default=512); chunk_overlap: int (default=50); min_chunk_size: int (default=100); preserve_sentences: bool (default=True)
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
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        preserve_sentences: bool = True
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size to emit
            preserve_sentences: Whether to avoid breaking sentences
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_sentences = preserve_sentences
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker.chunk_document
    # Requirement  : `chunk_document` shall chunk a unified document into indexable pieces
    # Purpose      : Chunk a unified document into indexable pieces
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : doc: dict
    # Outputs      : Iterator[DocumentChunk]
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
    def chunk_document(self, doc: dict) -> Iterator[DocumentChunk]:
        """
        Chunk a unified document into indexable pieces.
        
        Args:
            doc: UnifiedDocument as dict
            
        Yields:
            DocumentChunk objects
        """
        doc_id = doc["doc_id"]
        chunk_index = 0
        
        # Chunk title + abstract together (always important)
        title_abstract = f"{doc['title']}\n\n{doc.get('abstract', '')}"
        
        for chunk in self._create_chunks(
            title_abstract,
            ChunkType.ABSTRACT,
            doc_id,
            doc,
            chunk_index
        ):
            yield chunk
            chunk_index += 1
        
        # Chunk full text if available
        full_text = doc.get("full_text")
        if full_text:
            sections = self._split_into_sections(full_text)
            
            for section_type, section_text in sections:
                for chunk in self._create_chunks(
                    section_text,
                    section_type,
                    doc_id,
                    doc,
                    chunk_index
                ):
                    yield chunk
                    chunk_index += 1
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker._split_into_sections
    # Requirement  : `_split_into_sections` shall split full text into labeled sections
    # Purpose      : Split full text into labeled sections
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : list[tuple[ChunkType, str]]
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
    def _split_into_sections(self, text: str) -> list[tuple[ChunkType, str]]:
        """Split full text into labeled sections."""
        sections = []
        current_type = ChunkType.FULL_TEXT
        current_text = []
        
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            new_section = None
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line_stripped):
                    new_section = section_type
                    break
            
            if new_section:
                # Save previous section
                if current_text:
                    sections.append((current_type, '\n'.join(current_text)))
                current_type = new_section
                current_text = []
            else:
                current_text.append(line)
        
        # Don't forget last section
        if current_text:
            sections.append((current_type, '\n'.join(current_text)))
        
        return sections
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker._create_chunks
    # Requirement  : `_create_chunks` shall create chunks from text with proper overlap
    # Purpose      : Create chunks from text with proper overlap
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str; chunk_type: ChunkType; doc_id: str; doc: dict; start_index: int
    # Outputs      : Iterator[DocumentChunk]
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
    def _create_chunks(
        self,
        text: str,
        chunk_type: ChunkType,
        doc_id: str,
        doc: dict,
        start_index: int
    ) -> Iterator[DocumentChunk]:
        """Create chunks from text with proper overlap."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return
        
        # Split into sentences for cleaner boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_start = 0
        char_offset = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                # Yield current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    yield DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{start_index}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_type=chunk_type,
                        chunk_index=start_index,
                        pmid=doc.get("pmid"),
                        doi=doc.get("doi"),
                        title=doc.get("title", ""),
                        authors=doc.get("authors", []),
                        publication_year=doc.get("publication_year"),
                        journal=doc.get("journal", ""),
                        start_char=chunk_start,
                        end_char=char_offset,
                        token_count=current_length
                    )
                    start_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                
                for sent in reversed(current_chunk):
                    sent_len = len(sent.split())
                    if overlap_length + sent_len <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                chunk_start = char_offset - sum(len(s) + 1 for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
            char_offset += len(sentence) + 1
        
        # Yield final chunk
        if current_chunk and current_length >= self.min_chunk_size // 5:
            chunk_text = ' '.join(current_chunk)
            yield DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{start_index}",
                doc_id=doc_id,
                content=chunk_text,
                chunk_type=chunk_type,
                chunk_index=start_index,
                pmid=doc.get("pmid"),
                doi=doc.get("doi"),
                title=doc.get("title", ""),
                authors=doc.get("authors", []),
                publication_year=doc.get("publication_year"),
                journal=doc.get("journal", ""),
                start_char=chunk_start,
                end_char=char_offset,
                token_count=current_length
            )
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker._split_sentences
    # Requirement  : `_split_sentences` shall split text into sentences, handling abbreviations
    # Purpose      : Split text into sentences, handling abbreviations
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : list[str]
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
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling abbreviations."""
        # Simple sentence splitter (could use nltk/spacy for better results)
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Fig|et al|i\.e|e\.g)\.',
                      r'\1<DOT>', text, flags=re.IGNORECASE)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]

    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker.chunk_batch
    # Requirement  : `chunk_batch` shall chunk multiple documents
    # Purpose      : Chunk multiple documents
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : documents: list[dict]
    # Outputs      : Iterator[DocumentChunk]
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
    def chunk_batch(self, documents: list[dict]) -> Iterator[DocumentChunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of UnifiedDocument dicts
            
        Yields:
            DocumentChunk objects from all documents
        """
        for doc in documents:
            yield from self.chunk_document(doc)
    
    # ---------------------------------------------------------------------------
    # ID           : ingestion.chunker.EEGDocumentChunker.estimate_chunk_count
    # Requirement  : `estimate_chunk_count` shall estimate the number of chunks a document will produce
    # Purpose      : Estimate the number of chunks a document will produce
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : doc: dict
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
    def estimate_chunk_count(self, doc: dict) -> int:
        """
        Estimate the number of chunks a document will produce.
        
        Args:
            doc: UnifiedDocument as dict
            
        Returns:
            Estimated chunk count
        """
        text_length = len(doc.get("title", ""))
        text_length += len(doc.get("abstract", ""))
        
        if doc.get("full_text"):
            text_length += len(doc["full_text"])
        
        # Rough estimate: ~5 chars per word, chunk_size words per chunk
        estimated_words = text_length / 5
        return max(1, int(estimated_words / self.chunk_size))
