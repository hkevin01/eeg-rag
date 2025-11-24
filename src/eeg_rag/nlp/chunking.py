"""
Text Chunking Pipeline
Splits documents into semantically meaningful chunks with overlap for RAG retrieval.

Requirements:
- REQ-CHUNK-001: Split text into 512-token chunks
- REQ-CHUNK-002: Implement sliding window with overlap (50-100 tokens)
- REQ-CHUNK-003: Preserve sentence boundaries
- REQ-CHUNK-004: Maintain metadata (source, page, section)
- REQ-CHUNK-005: Handle special formatting (equations, tables, citations)
- REQ-CHUNK-006: Support batch processing (1000+ documents)
- REQ-CHUNK-007: Calculate chunk statistics (length, overlap, coverage)
- REQ-CHUNK-008: Deduplicate identical chunks
- REQ-CHUNK-009: Optimize for biomedical text
- REQ-CHUNK-010: Export chunk indices for retrieval
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib
from collections import defaultdict


@dataclass
class TextChunk:
    """
    A chunk of text with metadata
    
    Attributes:
        chunk_id: Unique identifier
        text: Chunk text content
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        token_count: Approximate token count
        metadata: Source document metadata
    """
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'token_count': self.token_count,
            'metadata': self.metadata,
            'overlap_with_previous': self.overlap_with_previous,
            'overlap_with_next': self.overlap_with_next
        }
    
    def compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        return hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class ChunkingResult:
    """Result from chunking a document"""
    document_id: str
    chunks: List[TextChunk]
    total_chunks: int
    total_tokens: int
    average_chunk_size: float
    overlap_tokens: int
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'document_id': self.document_id,
            'chunks': [c.to_dict() for c in self.chunks],
            'total_chunks': self.total_chunks,
            'total_tokens': self.total_tokens,
            'average_chunk_size': self.average_chunk_size,
            'overlap_tokens': self.overlap_tokens,
            'processing_time': self.processing_time
        }


class TextChunker:
    """
    Text chunking pipeline for RAG systems
    
    Optimized for biomedical/EEG research papers with:
    - Sentence-boundary preservation
    - Configurable chunk size and overlap
    - Metadata tracking
    - Deduplication
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 100,
        preserve_sentences: bool = True,
        deduplicate: bool = True
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target chunk size in tokens (default: 512)
            overlap: Number of overlapping tokens between chunks (default: 64)
            min_chunk_size: Minimum chunk size (default: 100)
            preserve_sentences: Try to end chunks at sentence boundaries (default: True)
            deduplicate: Remove duplicate chunks (default: True)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_sentences = preserve_sentences
        self.deduplicate = deduplicate
        
        # Sentence boundary patterns (biomedical-optimized)
        self.sentence_endings = re.compile(r'([.!?]\s+|\n{2,})')
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'duplicates_removed': 0,
            'total_tokens_processed': 0
        }
    
    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk a single document
        
        Args:
            text: Input text to chunk
            document_id: Unique document identifier
            metadata: Optional metadata to attach to chunks
            
        Returns:
            ChunkingResult with chunks and statistics
        """
        import time
        start_time = time.time()
        
        if not text or len(text.strip()) == 0:
            return ChunkingResult(
                document_id=document_id,
                chunks=[],
                total_chunks=0,
                total_tokens=0,
                average_chunk_size=0.0,
                overlap_tokens=0,
                processing_time=time.time() - start_time
            )
        
        # Preprocessing
        text = self._preprocess_text(text)
        
        # Create chunks
        if self.preserve_sentences:
            chunks = self._chunk_by_sentences(text, document_id, metadata or {})
        else:
            chunks = self._chunk_by_tokens(text, document_id, metadata or {})
        
        # Deduplicate if enabled
        if self.deduplicate:
            chunks = self._deduplicate_chunks(chunks)
        
        # Calculate statistics
        total_tokens = sum(c.token_count for c in chunks)
        overlap_tokens = sum(c.overlap_with_previous for c in chunks)
        avg_chunk_size = total_tokens / len(chunks) if chunks else 0
        
        # Update global statistics
        self.stats['documents_processed'] += 1
        self.stats['total_chunks_created'] += len(chunks)
        self.stats['total_tokens_processed'] += total_tokens
        
        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            average_chunk_size=avg_chunk_size,
            overlap_tokens=overlap_tokens,
            processing_time=time.time() - start_time
        )
    
    def chunk_batch(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]]
    ) -> List[ChunkingResult]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of (document_id, text, metadata) tuples
            
        Returns:
            List of ChunkingResult objects
        """
        results = []
        for doc_id, text, metadata in documents:
            result = self.chunk_text(text, doc_id, metadata)
            results.append(result)
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before chunking"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues in academic papers
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def _chunk_by_sentences(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk text by sentences, respecting token limits"""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        chunk_num = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{document_id}_chunk_{chunk_num}"
                
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    token_count=current_tokens,
                    metadata={**metadata, 'chunk_number': chunk_num}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.overlap)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
                current_start += len(chunk_text) - len(' '.join(overlap_sentences))
                
                if overlap_sentences:
                    chunk.overlap_with_next = sum(self._estimate_tokens(s) for s in overlap_sentences)
                    chunks[-1] = chunk
                
                chunk_num += 1
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{document_id}_chunk_{chunk_num}"
            
            chunk = TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                token_count=current_tokens,
                metadata={**metadata, 'chunk_number': chunk_num}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_tokens(
        self,
        text: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk text by fixed token count (no sentence preservation)"""
        tokens = text.split()  # Simple tokenization
        chunks = []
        chunk_num = 0
        
        i = 0
        while i < len(tokens):
            # Get chunk tokens with overlap
            if i > 0:
                start_idx = max(0, i - self.overlap)
            else:
                start_idx = i
            
            end_idx = min(len(tokens), i + self.chunk_size)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Create chunk
            chunk_text = ' '.join(chunk_tokens)
            token_count = len(chunk_tokens)
            
            if token_count >= self.min_chunk_size:
                chunk_id = f"{document_id}_chunk_{chunk_num}"
                
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_char=0,  # Character positions not accurate without sentence parsing
                    end_char=len(chunk_text),
                    token_count=token_count,
                    metadata={**metadata, 'chunk_number': chunk_num},
                    overlap_with_previous=self.overlap if i > 0 else 0
                )
                chunks.append(chunk)
                chunk_num += 1
            
            i += self.chunk_size
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence boundaries
        sentences = self.sentence_endings.split(text)
        
        # Combine sentence text with its ending punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                sentence = sentence.strip()
                if sentence:
                    result.append(sentence)
        
        # Add last sentence if exists
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count
        
        Uses simple heuristic: ~4 characters per token for English text
        More accurate with actual tokenizer, but this is fast for chunking
        """
        return len(text) // 4
    
    def _get_overlap_sentences(self, sentences: List[str], target_overlap: int) -> List[str]:
        """Get sentences for overlap region"""
        overlap_sentences = []
        overlap_tokens = 0
        
        # Start from end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens <= target_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _deduplicate_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Remove duplicate chunks based on content hash"""
        seen_hashes = set()
        unique_chunks = []
        duplicates_count = 0
        
        for chunk in chunks:
            chunk_hash = chunk.compute_hash()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
            else:
                duplicates_count += 1
        
        self.stats['duplicates_removed'] += duplicates_count
        return unique_chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return {
            'documents_processed': self.stats['documents_processed'],
            'total_chunks_created': self.stats['total_chunks_created'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'total_tokens_processed': self.stats['total_tokens_processed'],
            'average_chunks_per_doc': (
                self.stats['total_chunks_created'] / self.stats['documents_processed']
                if self.stats['documents_processed'] > 0 else 0
            )
        }
