"""
Integration Tests for Newly Implemented Components
Tests Agent 3, Agent 4, Text Chunking, Corpus Builder, and PubMedBERT Embeddings
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import shutil

from src.eeg_rag.agents.graph_agent.graph_agent import GraphAgent
from src.eeg_rag.agents.citation_agent.citation_validator import (
    CitationValidator, ValidationStatus
)
from src.eeg_rag.nlp.chunking import TextChunker
from src.eeg_rag.rag.corpus_builder import EEGCorpusBuilder
from src.eeg_rag.rag.embeddings import PubMedBERTEmbedder


class TestGraphAgent:
    """Test Knowledge Graph Agent (Agent 3)"""

    def test_graph_agent_initialization(self):
        """Test agent can be initialized"""
        agent = GraphAgent(use_mock=True)
        assert agent.name == "GraphAgent"
        assert agent.agent_type == "graph"
        assert "graph_query" in agent.capabilities

    @pytest.mark.asyncio
    async def test_graph_agent_query_execution(self):
        """Test agent can execute queries"""
        agent = GraphAgent(use_mock=True)
        result = await agent.execute("Find biomarkers that predict epilepsy")

        assert result.query_text == "Find biomarkers that predict epilepsy"
        assert len(result.nodes) > 0
        assert result.execution_time > 0
        assert result.cypher_query  # Should have generated Cypher

    @pytest.mark.asyncio
    async def test_graph_agent_caching(self):
        """Test query caching works"""
        agent = GraphAgent(use_mock=True)

        # First query - cache miss
        result1 = await agent.execute("Find biomarkers")
        cache_misses_1 = agent.cache_misses

        # Same query - cache hit
        result2 = await agent.execute("Find biomarkers")
        cache_hits_after = agent.cache_hits

        assert cache_hits_after > 0
        assert result1.nodes == result2.nodes

    @pytest.mark.asyncio
    async def test_graph_agent_relationship_traversal(self):
        """Test relationship extraction"""
        agent = GraphAgent(use_mock=True)
        result = await agent.execute("What predicts outcomes?")

        assert len(result.relationships) > 0
        rel = result.relationships[0]
        assert hasattr(rel, 'source_id')
        assert hasattr(rel, 'target_id')
        assert hasattr(rel, 'strength')


class TestCitationValidator:
    """Test Citation Validation Agent (Agent 4)"""

    def test_citation_validator_initialization(self):
        """Test validator can be initialized"""
        validator = CitationValidator(use_mock=True)
        assert validator.name == "CitationValidator"
        assert "citation_validation" in validator.capabilities

    @pytest.mark.asyncio
    async def test_validate_known_citation(self):
        """Test validating a known citation"""
        validator = CitationValidator(use_mock=True)
        result = await validator.validate('12345678')

        assert result.citation_id == '12345678'
        assert result.status == ValidationStatus.VALID
        assert result.title == 'P300 amplitude in epilepsy diagnosis'
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_validate_unknown_citation(self):
        """Test validating unknown citation"""
        validator = CitationValidator(use_mock=True)
        result = await validator.validate('99999999')

        assert result.status == ValidationStatus.INVALID
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_retraction_detection(self):
        """Test retracted paper detection"""
        validator = CitationValidator(use_mock=True)
        result = await validator.validate('34567890')  # Known retracted

        assert result.status == ValidationStatus.RETRACTED
        assert result.is_retracted is True
        assert result.retraction_notice is not None

    @pytest.mark.asyncio
    async def test_impact_score_calculation(self):
        """Test impact score calculation"""
        validator = CitationValidator(use_mock=True)
        result = await validator.validate('12345678')

        impact_total = result.impact_score.calculate_total()
        assert 0 <= impact_total <= 100
        assert impact_total > 0  # Should have some score

    @pytest.mark.asyncio
    async def test_batch_validation(self):
        """Test batch validation"""
        validator = CitationValidator(use_mock=True)
        citation_ids = ['12345678', '23456789', '34567890']

        results = await validator.validate_batch(citation_ids)

        assert len(results) == 3
        assert all(r.citation_id in citation_ids for r in results)


class TestTextChunker:
    """Test Text Chunking Pipeline"""

    def test_chunker_initialization(self):
        """Test chunker can be initialized"""
        chunker = TextChunker(chunk_size=512, overlap=64)
        assert chunker.chunk_size == 512
        assert chunker.overlap == 64

    def test_simple_chunking(self):
        """Test basic text chunking"""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "This is a test sentence. " * 20

        result = chunker.chunk_text(text, 'test_doc')

        assert result.total_chunks > 0
        assert result.total_tokens > 0
        assert result.document_id == 'test_doc'

    def test_sentence_preservation(self):
        """Test sentence boundary preservation"""
        chunker = TextChunker(chunk_size=30, overlap=5, preserve_sentences=True)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        result = chunker.chunk_text(text, 'test_doc')

        # Check that chunks end at sentence boundaries (roughly)
        for chunk in result.chunks:
            assert chunk.text.strip().endswith(('.', '?', '!')) or chunk == result.chunks[-1]

    def test_overlap_calculation(self):
        """Test chunk overlap works"""
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "word " * 100

        result = chunker.chunk_text(text, 'test_doc')

        if result.total_chunks > 1:
            assert result.overlap_tokens > 0

    def test_metadata_preservation(self):
        """Test metadata is preserved in chunks"""
        chunker = TextChunker(chunk_size=50, overlap=10)
        text = "Test text. " * 10
        metadata = {'source': 'test', 'author': 'TestUser'}

        result = chunker.chunk_text(text, 'test_doc', metadata)

        for chunk in result.chunks:
            assert 'source' in chunk.metadata
            assert chunk.metadata['source'] == 'test'

    def test_batch_chunking(self):
        """Test batch processing"""
        chunker = TextChunker(chunk_size=50, overlap=10)
        # Use longer text to ensure chunks are created
        long_text = "This is a longer test sentence with multiple words to ensure proper chunking behavior. " * 5
        documents = [
            ('doc1', long_text, {}),
            ('doc2', long_text, {}),
            ('doc3', long_text, {})
        ]

        results = chunker.chunk_batch(documents)

        assert len(results) == 3
        assert all(r.total_chunks > 0 for r in results)


class TestCorpusBuilder:
    """Test EEG Corpus Builder"""

    @pytest.fixture
    def temp_corpus_dir(self, tmp_path):
        """Create temporary directory for corpus"""
        corpus_dir = tmp_path / "test_corpus"
        corpus_dir.mkdir()
        yield corpus_dir
        # Cleanup
        if corpus_dir.exists():
            shutil.rmtree(corpus_dir)

    @pytest.mark.asyncio
    async def test_corpus_builder_initialization(self, temp_corpus_dir):
        """Test builder can be initialized"""
        builder = EEGCorpusBuilder(
            output_dir=temp_corpus_dir,
            target_count=10,
            use_mock=True
        )
        assert builder.target_count == 10
        assert builder.use_mock is True

    @pytest.mark.asyncio
    async def test_mock_corpus_generation(self, temp_corpus_dir):
        """Test mock corpus generation"""
        builder = EEGCorpusBuilder(
            output_dir=temp_corpus_dir,
            target_count=10,
            use_mock=True
        )

        stats = await builder.build_corpus()

        assert stats['papers_fetched'] == 10
        assert stats['total_time'] > 0

    @pytest.mark.asyncio
    async def test_corpus_file_creation(self, temp_corpus_dir):
        """Test corpus files are created"""
        builder = EEGCorpusBuilder(
            output_dir=temp_corpus_dir,
            target_count=5,
            use_mock=True
        )

        await builder.build_corpus()

        # Check files exist
        jsonl_files = list(temp_corpus_dir.glob("*.jsonl"))
        metadata_files = list(temp_corpus_dir.glob("corpus_metadata.json"))

        assert len(jsonl_files) > 0
        assert len(metadata_files) > 0

    @pytest.mark.asyncio
    async def test_paper_structure(self, temp_corpus_dir):
        """Test generated papers have correct structure"""
        builder = EEGCorpusBuilder(
            output_dir=temp_corpus_dir,
            target_count=3,
            use_mock=True
        )

        await builder.build_corpus()

        # Check paper attributes
        papers = list(builder.papers.values())
        assert len(papers) == 3

        for paper in papers:
            assert paper.pmid
            assert paper.title
            assert paper.abstract
            assert len(paper.authors) > 0
            assert paper.year > 2000


class TestPubMedBERTEmbedder:
    """Test PubMedBERT Embedding Generator"""

    def test_embedder_initialization(self):
        """Test embedder can be initialized"""
        embedder = PubMedBERTEmbedder(use_mock=True, batch_size=32)
        assert embedder.batch_size == 32
        assert embedder.use_mock is True

    def test_single_text_embedding(self):
        """Test embedding single text"""
        embedder = PubMedBERTEmbedder(use_mock=True)
        texts = ["EEG biomarkers for epilepsy"]

        result = embedder.embed_texts(texts, show_progress=False)

        assert result.total_chunks == 1
        assert len(result.embeddings) == 1
        assert result.embeddings[0].embedding.shape == (768,)

    def test_batch_embedding(self):
        """Test batch embedding generation"""
        embedder = PubMedBERTEmbedder(use_mock=True, batch_size=2)
        texts = [
            "EEG biomarkers for epilepsy",
            "P300 amplitude in cognition",
            "Theta oscillations in sleep"
        ]

        result = embedder.embed_texts(texts, show_progress=False)

        assert result.total_chunks == 3
        assert len(result.embeddings) == 3
        assert result.batch_size == 2

    def test_embedding_normalization(self):
        """Test embeddings are normalized"""
        embedder = PubMedBERTEmbedder(use_mock=True)
        texts = ["Test text for embedding"]

        result = embedder.embed_texts(texts, show_progress=False)
        embedding = result.embeddings[0].embedding

        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_embedding_consistency(self):
        """Test same text produces same embedding"""
        embedder = PubMedBERTEmbedder(use_mock=True)
        text = "EEG biomarkers for epilepsy"

        result1 = embedder.embed_texts([text], show_progress=False)
        result2 = embedder.embed_texts([text], show_progress=False)

        emb1 = result1.embeddings[0].embedding
        emb2 = result2.embeddings[0].embedding

        assert np.allclose(emb1, emb2)

    def test_embedding_save_load(self, tmp_path):
        """Test saving and loading embeddings"""
        embedder = PubMedBERTEmbedder(use_mock=True)
        texts = ["Text 1", "Text 2", "Text 3"]
        chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]

        result = embedder.embed_texts(texts, chunk_ids, show_progress=False)

        # Save
        save_path = tmp_path / "embeddings.npz"
        embedder.save_embeddings(result.embeddings, save_path)

        # Load
        loaded = embedder.load_embeddings(save_path)

        assert len(loaded) == 3
        assert all(isinstance(emb, np.ndarray) for emb in loaded)


class TestIntegrationPipeline:
    """Integration tests showing components working together"""

    @pytest.mark.asyncio
    async def test_corpus_to_embeddings_pipeline(self, tmp_path):
        """Test full pipeline: corpus -> chunks -> embeddings"""
        # Step 1: Build corpus
        corpus_dir = tmp_path / "corpus"
        builder = EEGCorpusBuilder(
            output_dir=corpus_dir,
            target_count=3,
            use_mock=True
        )
        await builder.build_corpus()
        papers = list(builder.papers.values())

        # Step 2: Chunk papers
        chunker = TextChunker(chunk_size=100, overlap=20)
        all_chunks = []
        for paper in papers:
            result = chunker.chunk_text(
                paper.abstract,
                paper.pmid,
                {'title': paper.title}
            )
            all_chunks.extend(result.chunks)

        # Step 3: Generate embeddings
        embedder = PubMedBERTEmbedder(use_mock=True)
        texts = [chunk.text for chunk in all_chunks]
        chunk_ids = [chunk.chunk_id for chunk in all_chunks]

        emb_result = embedder.embed_texts(texts, chunk_ids, show_progress=False)

        # Verify pipeline
        assert len(papers) == 3
        assert len(all_chunks) > 0
        assert len(emb_result.embeddings) == len(all_chunks)
        assert all(emb.embedding.shape == (768,) for emb in emb_result.embeddings)

    @pytest.mark.asyncio
    async def test_citations_and_graph_integration(self):
        """Test citation validator and graph agent together"""
        # Validate citations
        validator = CitationValidator(use_mock=True)
        citations = ['12345678', '23456789']
        validation_results = await validator.validate_batch(citations)

        # Query graph for related entities
        graph_agent = GraphAgent(use_mock=True)
        graph_result = await graph_agent.execute(
            "Find biomarkers related to validated papers"
        )

        # Verify both work together
        assert len(validation_results) == 2
        assert all(r.status in [ValidationStatus.VALID, ValidationStatus.RETRACTED]
                   for r in validation_results)
        assert len(graph_result.nodes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
