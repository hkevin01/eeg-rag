#!/usr/bin/env python3
"""
Semantic Chunking with Boundary Detection

Creates semantically coherent chunks that respect document structure.
Improves retrieval quality by maintaining context boundaries.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A semantic chunk of text"""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: str
    metadata: Dict[str, Any]
    boundary_score: float = 0.0  # How strong the semantic boundary is
    tokens: int = 0
    sentences: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'text_preview': self.text[:200] + '...' if len(self.text) > 200 else self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'length': len(self.text),
            'tokens': self.tokens,
            'sentences': self.sentences,
            'boundary_score': self.boundary_score,
            'metadata': self.metadata
        }


class SemanticChunker:
    """Creates semantically coherent chunks with boundary detection"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 512,
                 overlap: int = 50,
                 similarity_threshold: float = 0.6,
                 min_chunk_size: int = 100):
        """
        Initialize semantic chunker
        
        Args:
            model_name: Sentence transformer model for embeddings
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            similarity_threshold: Threshold for semantic boundary detection
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        
        # Load sentence transformer
        try:
            self.sentence_model = SentenceTransformer(model_name)
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def chunk_text(self, text: str, doc_id: str = "doc", 
                   metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Main chunking method"""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Step 2: Detect semantic boundaries
        boundaries = self._detect_semantic_boundaries(sentences)
        
        # Step 3: Create chunks respecting boundaries
        chunks = self._create_chunks_from_boundaries(sentences, boundaries, doc_id, metadata)
        
        # Step 4: Post-process chunks (merge small chunks, handle overlaps)
        final_chunks = self._post_process_chunks(chunks)
        
        return final_chunks
    
    def chunk_document(self, document: str, doc_id: str = "doc",
                      preserve_structure: bool = True) -> List[Chunk]:
        """Chunk a document with structure preservation"""
        if preserve_structure:
            return self._chunk_with_structure_preservation(document, doc_id)
        else:
            return self.chunk_text(document, doc_id)
    
    def _split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions"""
        # Enhanced sentence splitting for scientific text
        
        # Handle common abbreviations that shouldn't trigger splits
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'et al.', 'i.e.', 'e.g.',
            'vs.', 'etc.', 'Fig.', 'Table', 'Ref.', 'Vol.', 'No.',
            'p.', 'pp.', 'Hz.', 'mV.', 'µV.'
        }
        
        # Protect abbreviations
        protected_text = text
        replacements = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            protected_text = protected_text.replace(abbrev, placeholder)
            replacements[placeholder] = abbrev
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences_raw = re.split(sentence_pattern, protected_text)
        
        # Restore abbreviations and track positions
        sentences = []
        current_pos = 0
        
        for sentence_raw in sentences_raw:
            # Restore abbreviations
            sentence = sentence_raw
            for placeholder, abbrev in replacements.items():
                sentence = sentence.replace(placeholder, abbrev)
            
            # Find actual position in original text
            start_pos = text.find(sentence, current_pos)
            if start_pos == -1:  # Fallback
                start_pos = current_pos
            
            end_pos = start_pos + len(sentence)
            
            sentences.append((sentence.strip(), start_pos, end_pos))
            current_pos = end_pos
        
        return [s for s in sentences if s[0]]  # Filter empty sentences
    
    def _detect_semantic_boundaries(self, sentences: List[Tuple[str, int, int]]) -> List[float]:
        """Detect semantic boundaries between sentences"""
        if not self.sentence_model or len(sentences) < 2:
            # Fallback: use simple heuristics
            return self._heuristic_boundaries(sentences)
        
        try:
            # Extract sentence texts
            sentence_texts = [s[0] for s in sentences]
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(sentence_texts, convert_to_tensor=True)
            
            # Calculate similarities between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cos_sim(embeddings[i], embeddings[i + 1]).item()
                similarities.append(sim)
            
            # Convert similarities to boundary scores (lower similarity = stronger boundary)
            boundary_scores = [1.0 - sim for sim in similarities]
            
            return boundary_scores
            
        except Exception as e:
            logger.warning(f"Error in semantic boundary detection: {e}")
            return self._heuristic_boundaries(sentences)
    
    def _heuristic_boundaries(self, sentences: List[Tuple[str, int, int]]) -> List[float]:
        """Fallback heuristic boundary detection"""
        boundaries = []
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i][0]
            next_sentence = sentences[i + 1][0]
            
            # Heuristic rules for strong boundaries
            boundary_score = 0.5  # Default
            
            # Check for topic transition indicators
            transition_words = [
                'however', 'moreover', 'furthermore', 'additionally',
                'in contrast', 'on the other hand', 'meanwhile',
                'subsequently', 'in conclusion', 'finally'
            ]
            
            if any(word in next_sentence.lower()[:50] for word in transition_words):
                boundary_score += 0.3
            
            # Check for paragraph-like breaks (double newlines in original)
            if '\n\n' in current_sentence or current_sentence.endswith('\n'):
                boundary_score += 0.4
            
            # Check for section headers or numbered lists
            if re.match(r'^\d+\.', next_sentence.strip()):
                boundary_score += 0.4
            
            if next_sentence.isupper() and len(next_sentence) < 100:
                boundary_score += 0.5  # Likely a header
            
            # Check for different topics (very basic keyword overlap)
            current_words = set(re.findall(r'\b\w+\b', current_sentence.lower()))
            next_words = set(re.findall(r'\b\w+\b', next_sentence.lower()))
            
            if current_words and next_words:
                overlap = len(current_words & next_words) / len(current_words | next_words)
                if overlap < 0.2:  # Low overlap indicates topic change
                    boundary_score += 0.2
            
            boundaries.append(min(boundary_score, 1.0))
        
        return boundaries
    
    def _create_chunks_from_boundaries(self, sentences: List[Tuple[str, int, int]], 
                                     boundaries: List[float],
                                     doc_id: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Create chunks based on semantic boundaries"""
        chunks = []
        current_chunk_sentences = []
        current_chunk_start = 0
        
        for i, (sentence, start, end) in enumerate(sentences):
            current_chunk_sentences.append((sentence, start, end))
            
            # Check if we should end this chunk
            should_end_chunk = False
            
            # Size-based chunking
            current_text = ' '.join([s[0] for s in current_chunk_sentences])
            current_tokens = self._estimate_tokens(current_text)
            
            if current_tokens >= self.chunk_size:
                should_end_chunk = True
            
            # Boundary-based chunking
            if i < len(boundaries) and boundaries[i] > self.similarity_threshold:
                # Strong semantic boundary, consider ending chunk
                if current_tokens >= self.min_chunk_size:  # Don't create tiny chunks
                    should_end_chunk = True
            
            # End of document
            if i == len(sentences) - 1:
                should_end_chunk = True
            
            if should_end_chunk and current_chunk_sentences:
                # Create chunk
                chunk_text = ' '.join([s[0] for s in current_chunk_sentences])
                chunk_start = current_chunk_sentences[0][1]
                chunk_end = current_chunk_sentences[-1][2]
                
                chunk = Chunk(
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    metadata={**metadata, 'doc_id': doc_id},
                    boundary_score=boundaries[i-1] if i > 0 and i <= len(boundaries) else 0.0,
                    tokens=self._estimate_tokens(chunk_text),
                    sentences=len(current_chunk_sentences)
                )
                
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if i < len(sentences) - 1:  # Not the last sentence
                    overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                    current_chunk_sentences = overlap_sentences
                else:
                    current_chunk_sentences = []
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """Get overlap sentences for next chunk"""
        if not sentences:
            return []
        
        # Calculate how many sentences to overlap based on overlap tokens
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take sentences from the end
        for sentence, start, end in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, (sentence, start, end))
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks (merge small ones, validate)"""
        if not chunks:
            return []
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # If chunk is too small, try to merge with previous or next
            if chunk.tokens < self.min_chunk_size and len(chunks) > 1:
                # Try to merge with previous chunk
                if processed_chunks and processed_chunks[-1].tokens + chunk.tokens <= self.chunk_size * 1.5:
                    # Merge with previous
                    prev_chunk = processed_chunks[-1]
                    merged_text = prev_chunk.text + ' ' + chunk.text
                    
                    merged_chunk = Chunk(
                        text=merged_text,
                        start_pos=prev_chunk.start_pos,
                        end_pos=chunk.end_pos,
                        chunk_id=prev_chunk.chunk_id,
                        metadata=prev_chunk.metadata,
                        boundary_score=min(prev_chunk.boundary_score, chunk.boundary_score),
                        tokens=self._estimate_tokens(merged_text),
                        sentences=prev_chunk.sentences + chunk.sentences
                    )
                    
                    processed_chunks[-1] = merged_chunk
                    continue
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _chunk_with_structure_preservation(self, document: str, doc_id: str) -> List[Chunk]:
        """Chunk document while preserving structure (sections, paragraphs)"""
        # Detect structural elements
        sections = self._detect_sections(document)
        
        all_chunks = []
        
        for section_title, section_text, start_pos in sections:
            # Create metadata for this section
            section_metadata = {
                'section_title': section_title,
                'doc_id': doc_id
            }
            
            # Chunk this section
            section_chunks = self.chunk_text(section_text, doc_id, section_metadata)
            
            # Adjust positions
            for chunk in section_chunks:
                chunk.start_pos += start_pos
                chunk.end_pos += start_pos
            
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _detect_sections(self, document: str) -> List[Tuple[str, str, int]]:
        """Detect sections in a document"""
        # Simple section detection based on headers
        sections = []
        
        # Look for numbered sections, headers, etc.
        section_patterns = [
            r'^\d+\.\s+([A-Z][^\n]+)\n',  # 1. Title
            r'^([A-Z][A-Z\s]+)\n',  # ALL CAPS HEADERS
            r'^#+\s+([^\n]+)\n',  # Markdown headers
            r'^([A-Z][^\n]{5,50})\n\n'  # Title-like lines followed by blank line
        ]
        
        current_pos = 0
        current_section = "Introduction"
        section_start = 0
        
        for line_num, line in enumerate(document.split('\n')):
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # End previous section
                    if current_pos > section_start:
                        section_text = document[section_start:current_pos]
                        sections.append((current_section, section_text, section_start))
                    
                    # Start new section
                    current_section = match.group(1).strip()
                    section_start = current_pos
                    break
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Add final section
        if current_pos > section_start:
            section_text = document[section_start:]
            sections.append((current_section, section_text, section_start))
        
        # If no sections detected, treat as single section
        if not sections:
            sections = [("Document", document, 0)]
        
        return sections
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple estimation: 1 token ≈ 4 characters for English text
        return max(len(text) // 4, 1)
    
    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunking results"""
        if not chunks:
            return {'total_chunks': 0}
        
        token_counts = [chunk.tokens for chunk in chunks]
        sentence_counts = [chunk.sentences for chunk in chunks]
        boundary_scores = [chunk.boundary_score for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': np.mean(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_sentences_per_chunk': np.mean(sentence_counts),
            'avg_boundary_score': np.mean(boundary_scores),
            'chunk_size_target': self.chunk_size,
            'overlap_tokens': self.overlap,
            'similarity_threshold': self.similarity_threshold
        }


# Convenience function for quick chunking
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Chunk]:
    """Quick function to chunk text"""
    chunker = SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk_text(text)
