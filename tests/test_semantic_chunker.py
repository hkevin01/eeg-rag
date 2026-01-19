#!/usr/bin/env python3
"""
Tests for Semantic Chunking System
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.eeg_rag.core.semantic_chunker import (
    SemanticChunker, Chunk, chunk_text
)


class TestChunk:
    """Test chunk data class"""
    
    def test_creation(self):
        """Test chunk creation"""
        chunk = Chunk(
            text="This is a test chunk about EEG.",
            start_pos=0,
            end_pos=33,
            chunk_id="doc1_chunk_0",
            metadata={"section": "introduction"},
            boundary_score=0.75,
            tokens=8,
            sentences=1
        )
        
        assert chunk.text == "This is a test chunk about EEG."
        assert chunk.start_pos == 0
        assert chunk.end_pos == 33
        assert chunk.chunk_id == "doc1_chunk_0"
        assert chunk.boundary_score == 0.75
    
    def test_to_dict(self):
        """Test chunk serialization"""
        chunk = Chunk(
            text="A" * 300,  # Long text
            start_pos=0,
            end_pos=300,
            chunk_id="test_chunk",
            metadata={"test": True},
            tokens=75,
            sentences=3
        )
        
        chunk_dict = chunk.to_dict()
        
        assert chunk_dict['chunk_id'] == "test_chunk"
        assert len(chunk_dict['text_preview']) <= 203  # Should be truncated
        assert chunk_dict['tokens'] == 75
        assert chunk_dict['sentences'] == 3
        assert chunk_dict['metadata']['test'] is True


class TestSemanticChunker:
    """Test semantic chunking functionality"""
    
    @pytest.fixture
    def sample_text(self):
        return (
            "EEG measures electrical activity in the brain. "
            "It uses electrodes placed on the scalp. "
            "This technique is non-invasive and painless. "
            "Epilepsy research relies heavily on EEG analysis. "
            "Seizures show characteristic patterns in EEG recordings. "
            "Sleep studies also use EEG to identify sleep stages. "
            "Brain-computer interfaces decode EEG signals for control."
        )
    
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(
            chunk_size=100,
            overlap=20,
            similarity_threshold=0.6,
            min_chunk_size=50
        )
    
    def test_init(self, chunker):
        """Test chunker initialization"""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
        assert chunker.similarity_threshold == 0.6
        assert chunker.min_chunk_size == 50
    
    def test_init_no_model(self):
        """Test initialization when sentence transformer fails"""
        with patch('sentence_transformers.SentenceTransformer', side_effect=Exception("Model load failed")):
            chunker = SemanticChunker()
            assert chunker.sentence_model is None
    
    def test_split_into_sentences(self, chunker):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Dr. Smith said this."
        
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) >= 3  # Should handle abbreviations correctly
        assert all(len(s) == 3 for s in sentences)  # Each should have (text, start, end)
        assert all(isinstance(s[1], int) and isinstance(s[2], int) for s in sentences)
    
    def test_split_into_sentences_with_abbreviations(self, chunker):
        """Test sentence splitting with scientific abbreviations"""
        text = "EEG was recorded at 256 Hz. Fig. 1 shows the results. The p. value was significant."
        
        sentences = chunker._split_into_sentences(text)
        
        # Should not split on abbreviations
        sentence_texts = [s[0] for s in sentences]
        assert any("Fig. 1" in s for s in sentence_texts)
        assert any("256 Hz." in s for s in sentence_texts)
    
    def test_estimate_tokens(self, chunker):
        """Test token estimation"""
        text = "This is a test sentence with approximately eight words."
        
        tokens = chunker._estimate_tokens(text)
        
        assert tokens > 0
        # Should be roughly text length / 4
        expected = max(len(text) // 4, 1)
        assert abs(tokens - expected) <= 5  # Allow some variance
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_detect_semantic_boundaries_with_model(self, mock_transformer, chunker):
        """Test semantic boundary detection with model"""
        # Mock sentence transformer
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1]])  # 3 sentences
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        chunker.sentence_model = mock_model
        
        sentences = [
            ("First sentence about EEG.", 0, 20),
            ("Second sentence about fMRI.", 21, 45),  # Different topic
            ("Third sentence about EEG again.", 46, 75)
        ]
        
        boundaries = chunker._detect_semantic_boundaries(sentences)
        
        assert len(boundaries) == 2  # n-1 boundaries for n sentences
        assert all(0 <= b <= 1 for b in boundaries)
    
    def test_detect_semantic_boundaries_without_model(self, chunker):
        """Test heuristic boundary detection"""
        chunker.sentence_model = None
        
        sentences = [
            ("First sentence.", 0, 15),
            ("However, the second sentence shows contrast.", 16, 60),
            ("1. This looks like a numbered list.", 61, 95)
        ]
        
        boundaries = chunker._detect_semantic_boundaries(sentences)
        
        assert len(boundaries) == 2
        assert boundaries[0] > 0.5  # "However" should trigger high boundary
        assert boundaries[1] > 0.5  # Numbered list should trigger high boundary
    
    def test_heuristic_boundaries(self, chunker):
        """Test specific heuristic boundary rules"""
        # Test transition words
        sentences = [
            ("First sentence.", 0, 15),
            ("However, this is different.", 16, 40)
        ]
        
        boundaries = chunker._heuristic_boundaries(sentences)
        assert boundaries[0] > 0.5  # "However" should create strong boundary
        
        # Test numbered lists
        sentences = [
            ("Previous content.", 0, 15),
            ("1. First item in list.", 16, 40)
        ]
        
        boundaries = chunker._heuristic_boundaries(sentences)
        assert boundaries[0] > 0.5  # Numbered list should create strong boundary
    
    def test_chunk_text_empty(self, chunker):
        """Test chunking empty text"""
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []
    
    def test_chunk_text_basic(self, chunker, sample_text):
        """Test basic text chunking"""
        chunks = chunker.chunk_text(sample_text, "doc1", {"test": True})
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.chunk_id.startswith("doc1_chunk_") for chunk in chunks)
        assert all(chunk.metadata.get("test") is True for chunk in chunks)
        
        # Check chunk properties
        for chunk in chunks:
            assert chunk.tokens > 0
            assert chunk.sentences > 0
            assert chunk.start_pos < chunk.end_pos
    
    def test_get_overlap_sentences(self, chunker):
        """Test overlap sentence calculation"""
        sentences = [
            ("Sentence one.", 0, 12),
            ("Sentence two is longer.", 13, 35),
            ("Sentence three.", 36, 50)
        ]
        
        overlap_sentences = chunker._get_overlap_sentences(sentences)
        
        # Should include sentences that fit within overlap token limit
        assert len(overlap_sentences) <= len(sentences)
        
        total_overlap_tokens = sum(chunker._estimate_tokens(s[0]) for s in overlap_sentences)
        assert total_overlap_tokens <= chunker.overlap
    
    def test_post_process_chunks(self, chunker):
        """Test chunk post-processing"""
        # Create test chunks, one too small
        chunks = [
            Chunk(
                text="Small chunk.",
                start_pos=0, end_pos=12,
                chunk_id="doc1_chunk_0",
                metadata={},
                tokens=3  # Below minimum
            ),
            Chunk(
                text="This is a larger chunk with more content.",
                start_pos=13, end_pos=55,
                chunk_id="doc1_chunk_1",
                metadata={},
                tokens=10
            )
        ]
        
        processed = chunker._post_process_chunks(chunks)
        
        # Should merge small chunk with larger one
        assert len(processed) == 1
        assert "Small chunk. This is a larger chunk" in processed[0].text
    
    def test_detect_sections(self, chunker):
        """Test section detection"""
        document = """
# Introduction
This is the introduction section.

# Methods
This section describes the methods.

# Results
Results are presented here.
        """.strip()
        
        sections = chunker._detect_sections(document)
        
        assert len(sections) >= 3
        section_titles = [s[0] for s in sections]
        assert any("Introduction" in title for title in section_titles)
        assert any("Methods" in title for title in section_titles)
        assert any("Results" in title for title in section_titles)
    
    def test_chunk_with_structure_preservation(self, chunker):
        """Test structure-preserving chunking"""
        document = """
# Background
EEG is a neuroimaging technique. It measures electrical activity.

# Methods  
We collected data from subjects. Analysis was performed using Python.
        """.strip()
        
        chunks = chunker._chunk_with_structure_preservation(document, "doc1")
        
        assert len(chunks) > 0
        
        # Check that section metadata is preserved
        section_titles = [chunk.metadata.get('section_title', '') for chunk in chunks]
        assert any("Background" in title for title in section_titles)
        assert any("Methods" in title for title in section_titles)
    
    def test_chunk_document_with_structure(self, chunker):
        """Test document chunking with structure preservation"""
        document = "# Title\nContent here.\n\n# Section\nMore content."
        
        chunks = chunker.chunk_document(document, "doc1", preserve_structure=True)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_chunk_document_without_structure(self, chunker, sample_text):
        """Test document chunking without structure preservation"""
        chunks = chunker.chunk_document(sample_text, "doc1", preserve_structure=False)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_get_chunking_stats(self, chunker, sample_text):
        """Test chunking statistics"""
        chunks = chunker.chunk_text(sample_text, "doc1")
        stats = chunker.get_chunking_stats(chunks)
        
        assert 'total_chunks' in stats
        assert 'avg_tokens_per_chunk' in stats
        assert 'min_tokens' in stats
        assert 'max_tokens' in stats
        assert 'avg_sentences_per_chunk' in stats
        assert 'chunk_size_target' in stats
        
        assert stats['total_chunks'] == len(chunks)
        assert stats['chunk_size_target'] == chunker.chunk_size
    
    def test_get_chunking_stats_empty(self, chunker):
        """Test statistics with empty chunk list"""
        stats = chunker.get_chunking_stats([])
        
        assert stats['total_chunks'] == 0


class TestConvenienceFunction:
    """Test convenience chunking function"""
    
    def test_chunk_text_function(self):
        """Test standalone chunking function"""
        text = "This is a test. This is another sentence. And one more."
        
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_short_text(self):
        """Test chunking very short text"""
        chunker = SemanticChunker(chunk_size=100, min_chunk_size=20)
        
        text = "Short."
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].text == "Short."
    
    def test_text_without_sentences(self):
        """Test text without proper sentence boundaries"""
        chunker = SemanticChunker()
        
        text = "word1 word2 word3 word4 word5"  # No sentence endings
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) >= 1
    
    def test_very_long_text(self):
        """Test chunking very long text"""
        chunker = SemanticChunker(chunk_size=100, overlap=20)
        
        # Create long text
        sentences = [f"This is sentence number {i} about EEG research." for i in range(50)]
        text = " ".join(sentences)
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 1
        # Check overlap exists
        for i in range(len(chunks) - 1):
            # Some content should overlap between adjacent chunks
            assert chunks[i].end_pos > chunks[i+1].start_pos or chunks[i+1].start_pos <= chunks[i].end_pos + chunker.overlap
    
    def test_single_very_long_sentence(self):
        """Test chunking single extremely long sentence"""
        chunker = SemanticChunker(chunk_size=100)
        
        # Single sentence longer than chunk size
        text = "This is an extremely long sentence that goes on and on and contains many words and phrases about EEG research and brain activity and neural signals and electrical measurements and electrode placement and signal processing and data analysis and machine learning and artificial intelligence and deep learning and convolutional neural networks and feature extraction."
        
        chunks = chunker.chunk_text(text, "doc1")
        
        # Should still create chunks even from single long sentence
        assert len(chunks) >= 1


class TestIntegration:
    """Integration tests for semantic chunking"""
    
    def test_real_eeg_paper_chunking(self):
        """Test chunking realistic EEG research paper content"""
        paper_text = """
Electroencephalography (EEG) is a non-invasive neuroimaging technique that measures electrical activity in the brain. It provides high temporal resolution, making it suitable for real-time brain monitoring.

Epilepsy is a neurological disorder characterized by recurrent seizures. EEG plays a crucial role in epilepsy diagnosis and monitoring. Seizure detection algorithms analyze EEG patterns to identify abnormal brain activity.

Deep learning methods have shown promising results in automated seizure detection. Convolutional neural networks (CNNs) can extract relevant features from raw EEG signals. However, the performance depends on data quality and preprocessing steps.

Sleep stage classification is another important application of EEG analysis. Sleep spindles and K-complexes are characteristic patterns in non-REM sleep. Automated sleep scoring can assist clinicians in sleep disorder diagnosis.
        """.strip()
        
        chunker = SemanticChunker(
            chunk_size=200,
            overlap=30,
            similarity_threshold=0.6
        )
        
        chunks = chunker.chunk_text(paper_text, "paper1", {"type": "research"})
        
        assert len(chunks) > 1
        
        # Check that chunks maintain topical coherence
        chunk_texts = [chunk.text for chunk in chunks]
        
        # First chunk should be about EEG basics
        assert any("neuroimaging" in chunk or "electrical activity" in chunk for chunk in chunk_texts)
        
        # Should have chunks about epilepsy and sleep
        assert any("epilepsy" in chunk.lower() or "seizure" in chunk.lower() for chunk in chunk_texts)
        assert any("sleep" in chunk.lower() for chunk in chunk_texts)
        
        # Check metadata is preserved
        assert all(chunk.metadata.get("type") == "research" for chunk in chunks)
        
        # Check statistics
        stats = chunker.get_chunking_stats(chunks)
        assert stats['total_chunks'] == len(chunks)
        assert stats['avg_tokens_per_chunk'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
