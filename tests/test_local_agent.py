"""
Unit tests for Local Data Agent

Tests cover:
- FAISS vector store operations
- Citation handling
- Search functionality
- Document indexing
"""

import unittest
import asyncio
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.agents.base_agent import AgentQuery
from eeg_rag.agents.local_agent.local_data_agent import (
    LocalDataAgent,
    FAISSVectorStore,
    Citation,
    SearchResult
)


class TestCitation(unittest.TestCase):
    """Test Citation dataclass"""

    def test_citation_creation(self):
        """Test creating a citation"""
        citation = Citation(
            pmid="12345678",
            doi="10.1000/xyz123",
            title="EEG Study",
            authors=["Smith J", "Doe J"],
            journal="J Neurosci",
            year=2024
        )

        self.assertEqual(citation.pmid, "12345678")
        self.assertEqual(citation.title, "EEG Study")
        self.assertEqual(len(citation.authors), 2)

    def test_citation_formatting(self):
        """Test citation formatting"""
        citation = Citation(
            pmid="12345678",
            title="EEG Study",
            authors=["Smith J", "Doe J", "Johnson A"],
            journal="J Neurosci",
            year=2024
        )

        formatted = citation.format_citation()

        self.assertIn("Smith J", formatted)
        self.assertIn("(2024)", formatted)
        self.assertIn("EEG Study", formatted)
        self.assertIn("PMID: 12345678", formatted)

    def test_citation_many_authors(self):
        """Test citation with many authors"""
        citation = Citation(
            title="Test",
            authors=["A", "B", "C", "D", "E"],
            journal="Test J",
            year=2024
        )

        formatted = citation.format_citation()
        self.assertIn("et al.", formatted)

    def test_citation_to_dict(self):
        """Test citation serialization"""
        citation = Citation(
            pmid="12345",
            title="Test",
            authors=["A"],
            year=2024
        )

        data = citation.to_dict()

        self.assertEqual(data["pmid"], "12345")
        self.assertEqual(data["title"], "Test")
        self.assertIsInstance(data["authors"], list)


class TestSearchResult(unittest.TestCase):
    """Test SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating a search result"""
        citation = Citation(title="Test", authors=["A"], year=2024)
        result = SearchResult(
            document_id="doc1",
            content="Test content",
            citation=citation,
            relevance_score=0.95
        )

        self.assertEqual(result.document_id, "doc1")
        self.assertEqual(result.relevance_score, 0.95)

    def test_search_result_to_dict(self):
        """Test search result serialization"""
        citation = Citation(title="Test", authors=["A"], year=2024)
        result = SearchResult(
            document_id="doc1",
            content="Test content",
            citation=citation,
            relevance_score=0.95,
            metadata={"source": "test"}
        )

        data = result.to_dict()

        self.assertEqual(data["document_id"], "doc1")
        self.assertEqual(data["relevance_score"], 0.95)
        self.assertIn("citation", data)
        self.assertEqual(data["metadata"]["source"], "test")


class TestFAISSVectorStore(unittest.TestCase):
    """Test FAISSVectorStore class"""

    def setUp(self):
        """Create fresh vector store for each test"""
        self.store = FAISSVectorStore(dimension=128, index_type="Flat")

    def test_store_initialization(self):
        """Test vector store initialization"""
        self.assertEqual(self.store.dimension, 128)
        self.assertEqual(self.store.index_type, "Flat")
        self.assertEqual(len(self.store.documents), 0)

    def test_add_documents(self):
        """Test adding documents to store"""
        embeddings = np.random.randn(3, 128).astype('float32')
        documents = [
            {"content": "Doc 1", "citation": {}},
            {"content": "Doc 2", "citation": {}},
            {"content": "Doc 3", "citation": {}}
        ]

        doc_ids = self.store.add_documents(embeddings, documents)

        self.assertEqual(len(doc_ids), 3)
        self.assertEqual(len(self.store.documents), 3)
        self.assertEqual(doc_ids, [0, 1, 2])

    def test_search_empty_store(self):
        """Test searching empty store"""
        query_embedding = np.random.randn(128)
        results = self.store.search(query_embedding, k=5)

        self.assertEqual(len(results), 0)

    def test_search_with_documents(self):
        """Test searching with documents"""
        # Add documents
        embeddings = np.random.randn(5, 128).astype('float32')
        documents = [
            {"content": f"Doc {i}", "citation": {}}
            for i in range(5)
        ]
        self.store.add_documents(embeddings, documents)

        # Search
        query_embedding = embeddings[0]  # Use first doc as query
        results = self.store.search(query_embedding, k=3)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # First result should be doc 0 (exact match)
        self.assertEqual(results[0][0], 0)

    def test_save_and_load(self):
        """Test persistence"""
        # Add documents
        embeddings = np.random.randn(3, 128).astype('float32')
        documents = [
            {"content": f"Doc {i}", "citation": {}}
            for i in range(3)
        ]
        self.store.add_documents(embeddings, documents)

        # Save
        temp_dir = Path(tempfile.mkdtemp())
        self.store.save(temp_dir)

        # Load into new store
        new_store = FAISSVectorStore(dimension=128)
        new_store.load(temp_dir)

        self.assertEqual(len(new_store.documents), 3)
        self.assertEqual(new_store.dimension, 128)

    def test_statistics(self):
        """Test getting statistics"""
        embeddings = np.random.randn(5, 128).astype('float32')
        documents = [{"content": f"Doc {i}", "citation": {}} for i in range(5)]
        self.store.add_documents(embeddings, documents)

        stats = self.store.get_statistics()

        self.assertEqual(stats["total_documents"], 5)
        self.assertEqual(stats["dimension"], 128)
        self.assertIn("using_faiss", stats)


class TestLocalDataAgent(unittest.IsolatedAsyncioTestCase):
    """Test LocalDataAgent class"""

    async def asyncSetUp(self):
        """Set up test environment"""
        self.agent = LocalDataAgent(
            embedding_dimension=128,
            config={"top_k": 3, "min_relevance_score": 0.1}
        )

        # Add some test documents
        documents = [
            {
                "content": "EEG alpha waves in relaxation",
                "citation": {
                    "title": "Alpha Waves Study",
                    "authors": ["Smith J"],
                    "year": 2024,
                    "journal": "J Neurosci"
                },
                "metadata": {"topic": "alpha"}
            },
            {
                "content": "P300 event-related potential component",
                "citation": {
                    "title": "P300 Study",
                    "authors": ["Doe J"],
                    "year": 2023,
                    "journal": "Brain Res"
                },
                "metadata": {"topic": "erp"}
            },
            {
                "content": "Sleep stage classification using EEG",
                "citation": {
                    "title": "Sleep EEG",
                    "authors": ["Johnson A"],
                    "year": 2024,
                    "journal": "Sleep Med"
                },
                "metadata": {"topic": "sleep"}
            }
        ]

        self.agent.add_documents(documents)

    async def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_type.value, "local_data")
        self.assertEqual(self.agent.name, "local_data_agent")
        self.assertIsNotNone(self.agent.vector_store)

    async def test_execute_search(self):
        """Test executing a search"""
        query = AgentQuery(
            text="alpha waves relaxation",
            intent="factual"
        )

        result = await self.agent.execute(query)

        self.assertTrue(result.success)
        self.assertIn("results", result.data)
        self.assertIn("total_results", result.data)
        self.assertGreater(result.data["total_results"], 0)

    async def test_search_results_structure(self):
        """Test search results have correct structure"""
        query = AgentQuery(text="P300 ERP", intent="factual")

        result = await self.agent.execute(query)

        self.assertTrue(result.success)
        results = result.data["results"]

        if len(results) > 0:
            first_result = results[0]
            self.assertIn("document_id", first_result)
            self.assertIn("content", first_result)
            self.assertIn("citation", first_result)
            self.assertIn("relevance_score", first_result)

    async def test_search_timing(self):
        """Test search completes quickly"""
        query = AgentQuery(text="EEG sleep", intent="factual")

        result = await self.agent.execute(query)

        self.assertTrue(result.success)
        # Should complete in less than 1 second (much less ideally)
        self.assertLess(result.elapsed_time, 1.0)

    async def test_empty_query(self):
        """Test validation of empty query"""
        # Empty query should raise ValueError during AgentQuery creation
        with self.assertRaises(ValueError) as cm:
            AgentQuery(text="", intent="factual")
        
        self.assertIn("cannot be empty", str(cm.exception))

    def test_add_documents(self):
        """Test adding new documents"""
        initial_count = len(self.agent.vector_store.documents)

        new_docs = [
            {
                "content": "New document about theta waves",
                "citation": {
                    "title": "Theta Study",
                    "authors": ["New Author"],
                    "year": 2024
                }
            }
        ]

        doc_ids = self.agent.add_documents(new_docs)

        self.assertEqual(len(doc_ids), 1)
        self.assertEqual(
            len(self.agent.vector_store.documents),
            initial_count + 1
        )

    def test_statistics(self):
        """Test getting agent statistics"""
        stats = self.agent.get_statistics()

        self.assertIn("total_executions", stats)
        self.assertIn("vector_store", stats)
        self.assertIn("total_documents", stats["vector_store"])

    async def test_relevance_filtering(self):
        """Test results are filtered by relevance score"""
        # Create agent with high relevance threshold
        agent = LocalDataAgent(
            embedding_dimension=128,
            config={"top_k": 10, "min_relevance_score": 0.9}
        )

        # Add one document
        agent.add_documents([{
            "content": "Test document",
            "citation": {"title": "Test", "authors": [], "year": 2024}
        }])

        query = AgentQuery(text="completely unrelated query", intent="factual")
        result = await agent.execute(query)

        # With high threshold, might get no results
        self.assertTrue(result.success)
        # Number of results should be <= top_k
        self.assertLessEqual(result.data["total_results"], 10)


if __name__ == "__main__":
    unittest.main()
