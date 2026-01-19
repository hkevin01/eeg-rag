#!/usr/bin/env python3
"""
Comprehensive Demo of EEG-RAG Production Features

Demonstrates all 5 production-ready improvements:
1. RAG Evaluation Framework
2. Citation Verification with Hallucination Detection
3. Hybrid Retrieval (BM25 + Dense)
4. Query Routing based on Question Type
5. Semantic Chunking with Boundary Detection
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

# Import all production components
from src.eeg_rag.verification import CitationVerifier, HallucinationDetector, verify_answer_citations
from src.eeg_rag.retrieval import HybridRetriever
from src.eeg_rag.core import QueryRouter, SemanticChunker, QueryType

# Import evaluation framework
import sys
sys.path.append('tests/evaluation')
from rag_evaluator import RAGEvaluator

# Sample EEG research documents
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "EEG-based Epilepsy Detection using Deep Learning",
        "content": """Electroencephalography (EEG) is a non-invasive technique for measuring brain electrical activity. 
In epilepsy research, EEG plays a crucial role in seizure detection and characterization. Recent advances in deep learning have shown promising results for automated seizure detection. 
Convolutional neural networks (CNNs) can effectively extract spatiotemporal features from EEG signals. 
Our study achieved 95% accuracy in seizure detection using a hybrid CNN-LSTM architecture (PMID: 12345678). 
The model was trained on 1000 hours of EEG data from 200 patients with focal epilepsy.""",
        "metadata": {"year": 2023, "journal": "NeuroImage", "pmid": "12345678"}
    },
    {
        "id": "doc2", 
        "title": "Sleep Stage Classification from EEG Signals",
        "content": """Sleep studies rely heavily on EEG analysis for accurate sleep stage classification. 
Sleep spindles and K-complexes are characteristic patterns in non-REM sleep stages. 
Traditional sleep scoring is time-consuming and subject to inter-rater variability. 
Automated sleep stage classification using machine learning can improve consistency and efficiency. 
Our random forest model achieved 87% accuracy across all sleep stages (PMID: 87654321). 
Feature extraction focused on spectral power in delta (0.5-4 Hz), theta (4-8 Hz), and alpha (8-13 Hz) bands.""",
        "metadata": {"year": 2022, "journal": "Sleep Medicine", "pmid": "87654321"}
    },
    {
        "id": "doc3",
        "title": "Brain-Computer Interfaces for Motor Imagery",
        "content": """Brain-computer interfaces (BCIs) enable direct communication between the brain and external devices. 
Motor imagery tasks activate specific brain regions, generating detectable EEG patterns. 
Common spatial patterns (CSP) is a widely used feature extraction method for motor imagery classification. 
However, CSP performance degrades with session-to-session variability and subject differences. 
Adaptive algorithms can improve BCI performance by accounting for signal non-stationarity. 
Our adaptive CSP method showed 15% improvement over standard CSP in online BCI control (PMID: 11111111).""",
        "metadata": {"year": 2024, "journal": "IEEE Trans BME", "pmid": "11111111"}
    },
    {
        "id": "doc4",
        "title": "EEG Preprocessing and Artifact Removal",
        "content": """EEG signals are contaminated by various artifacts including eye movements, muscle activity, and line noise. 
Proper preprocessing is essential for reliable EEG analysis and interpretation. 
Independent component analysis (ICA) is effective for removing ocular and muscular artifacts. 
Wavelet denoising can reduce high-frequency noise while preserving signal characteristics. 
The choice of preprocessing pipeline significantly affects downstream analysis results. 
A systematic comparison showed ICA outperforms wavelet methods for artifact removal (PMID: 99999999).""",
        "metadata": {"year": 2021, "journal": "Clinical Neurophysiology", "pmid": "99999999"}
    }
]

# Sample queries for testing
TEST_QUERIES = [
    "What is electroencephalography?",  # Definitional
    "Recent advances in EEG-based seizure detection",  # Recent literature
    "Compare ICA versus wavelet denoising for EEG preprocessing",  # Comparative
    "How to perform sleep stage classification from EEG?",  # Methodological
    "Clinical applications of BCI for motor rehabilitation",  # Clinical
    "Statistical significance of EEG biomarkers in epilepsy"  # Statistical
]


class ProductionDemo:
    """Comprehensive demo of all production features"""
    
    def __init__(self):
        # Initialize all components
        self.citation_verifier = CitationVerifier(email="demo@eeg-rag.org")
        self.hallucination_detector = HallucinationDetector(self.citation_verifier)
        self.hybrid_retriever = HybridRetriever(alpha=0.6, fusion_method='weighted_sum')
        self.query_router = QueryRouter()
        self.semantic_chunker = SemanticChunker(chunk_size=300, overlap=50)
        self.rag_evaluator = RAGEvaluator()
        
        # Setup data
        self.documents = SAMPLE_DOCUMENTS
        self.chunks = []
        
    async def setup_system(self):
        """Initialize the system with sample data"""
        print("🔧 Setting up EEG-RAG Production System...\n")
        
        # 1. Semantic Chunking
        print("📄 Creating semantic chunks...")
        all_chunks = []
        for doc in self.documents:
            chunks = self.semantic_chunker.chunk_text(
                doc['content'], 
                doc['id'], 
                doc['metadata']
            )
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        print(f"Created {len(self.chunks)} semantic chunks")
        
        # Get chunking statistics
        chunk_stats = self.semantic_chunker.get_chunking_stats(self.chunks)
        print(f"Average tokens per chunk: {chunk_stats['avg_tokens_per_chunk']:.1f}")
        print(f"Average sentences per chunk: {chunk_stats['avg_sentences_per_chunk']:.1f}\n")
        
        # 2. Hybrid Retrieval Setup
        print("🔍 Setting up hybrid retrieval...")
        chunk_texts = [chunk.text for chunk in self.chunks]
        chunk_metadata = [chunk.metadata for chunk in self.chunks]
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        
        self.hybrid_retriever.add_documents(chunk_texts, chunk_metadata, chunk_ids)
        
        retrieval_stats = self.hybrid_retriever.get_stats()
        print(f"Indexed {retrieval_stats['total_documents']} chunks")
        print(f"BM25 available: {retrieval_stats['bm25_available']}")
        print(f"Dense retriever available: {retrieval_stats['dense_retriever_available']}")
        print(f"Fusion method: {retrieval_stats['fusion_method']}\n")
    
    async def demo_query_routing(self):
        """Demonstrate intelligent query routing"""
        print("🗺️  DEMO: Query Routing\n")
        print("Testing different query types and routing decisions...\n")
        
        for query in TEST_QUERIES:
            result = self.query_router.route_query(query)
            
            print(f"Query: '{query}'")
            print(f"  → Type: {result.query_type.value}")
            print(f"  → Agent: {result.recommended_agent}")
            print(f"  → Confidence: {result.confidence:.2f}")
            print(f"  → Complexity: {result.complexity}")
            print(f"  → Keywords: {result.keywords[:5]}")
            print(f"  → Reasoning: {result.reasoning}")
            print()
        
        # Show routing statistics
        routing_stats = self.query_router.get_routing_stats()
        print(f"Supported query types: {len(routing_stats['supported_query_types'])}")
        print(f"EEG keywords: {routing_stats['eeg_keywords_count']}\n")
    
    def demo_semantic_chunking(self):
        """Demonstrate semantic chunking capabilities"""
        print("✂️  DEMO: Semantic Chunking\n")
        
        # Show chunk details
        print(f"Total chunks created: {len(self.chunks)}\n")
        
        for i, chunk in enumerate(self.chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1} ({chunk.chunk_id}):")
            print(f"  Text: {chunk.text[:100]}...")
            print(f"  Tokens: {chunk.tokens}")
            print(f"  Sentences: {chunk.sentences}")
            print(f"  Boundary score: {chunk.boundary_score:.2f}")
            print(f"  Metadata: {chunk.metadata}")
            print()
        
        # Show statistics
        stats = self.semantic_chunker.get_chunking_stats(self.chunks)
        print("Chunking Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print()
    
    def demo_hybrid_retrieval(self):
        """Demonstrate hybrid retrieval capabilities"""
        print("🔍 DEMO: Hybrid Retrieval (BM25 + Dense)\n")
        
        test_queries = [
            "seizure detection accuracy",
            "sleep spindles classification",
            "motor imagery BCI performance"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            
            # Get hybrid search results
            results = self.hybrid_retriever.search(query, top_k=3)
            
            print("Top Results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result.score:.3f} (BM25: {result.bm25_score:.3f}, Dense: {result.dense_score:.3f})")
                print(f"     Doc: {result.doc_id}")
                print(f"     Text: {result.content[:100]}...")
            print()
    
    async def demo_citation_verification(self):
        """Demonstrate citation verification and hallucination detection"""
        print("🔬 DEMO: Citation Verification & Hallucination Detection\n")
        
        # Test answers with different citation patterns
        test_answers = [
            "EEG-based seizure detection achieved 95% accuracy using CNN-LSTM architecture (PMID: 12345678). This approach outperforms traditional methods.",
            
            "Sleep stage classification always works perfectly with 100% accuracy (PMID: 99999999). All patients show complete improvement without any side effects.",
            
            "Motor imagery BCIs are definitely the best approach for all neurological conditions. Recent studies prove this conclusively."
        ]
        
        for i, answer in enumerate(test_answers):
            print(f"Test Answer {i+1}:")
            print(f"'{answer[:100]}...'\n")
            
            # Check for hallucination
            hallucination_result = await self.hallucination_detector.check_answer(answer)
            
            print("Hallucination Analysis:")
            print(f"  Overall Score: {hallucination_result['hallucination_score']:.3f} (0=good, 1=bad)")
            print(f"  Citation Accuracy: {hallucination_result['citation_accuracy']:.3f}")
            print(f"  Verified Citations: {hallucination_result['verified_citations']}")
            print(f"  Invalid Citations: {hallucination_result['invalid_citations']}")
            print(f"  Unsupported Claims: {hallucination_result['unsupported_claims']}")
            print(f"  Total Claims: {hallucination_result['total_claims']}")
            print("  Pattern Flags:")
            for pattern, score in hallucination_result['pattern_flags'].items():
                if score > 0:
                    print(f"    {pattern}: {score:.3f}")
            print()
    
    async def demo_rag_evaluation(self):
        """Demonstrate RAG evaluation framework"""
        print("📊 DEMO: RAG Evaluation Framework\n")
        
        # Load benchmark queries
        try:
            with open('tests/evaluation/eeg_benchmark.json', 'r') as f:
                benchmark = json.load(f)
        except FileNotFoundError:
            print("Benchmark file not found, creating sample benchmark...")
            benchmark = {
                "queries": [
                    {
                        "id": "q1",
                        "question": "What are the main EEG biomarkers for epilepsy?",
                        "expected_pmids": ["12345678"],
                        "key_concepts": ["seizure detection", "EEG", "epilepsy"],
                        "difficulty": "medium"
                    },
                    {
                        "id": "q2",
                        "question": "How effective is deep learning for EEG analysis?",
                        "expected_pmids": ["12345678", "11111111"],
                        "key_concepts": ["deep learning", "CNN", "accuracy"],
                        "difficulty": "medium"
                    }
                ]
            }
        
        print(f"Evaluating {len(benchmark['queries'])} benchmark queries...\n")
        
        # Mock evaluation results (in real system, this would run actual queries)
        for query_data in benchmark['queries'][:2]:
            print(f"Query: {query_data['question']}")
            
            # Simulate retrieval results
            retrieved_docs = [
                {"doc_id": "doc1", "score": 0.85, "content": self.documents[0]['content']},
                {"doc_id": "doc2", "score": 0.72, "content": self.documents[1]['content']}
            ]
            
            # Simulate generated answer
            generated_answer = f"Based on recent research, EEG analysis shows promising results with deep learning approaches achieving high accuracy (PMID: {query_data['expected_pmids'][0]})."
            
            # Evaluate retrieval
            retrieval_metrics = self.rag_evaluator.evaluate_retrieval(
                query_data['question'],
                retrieved_docs,
                query_data['expected_pmids']
            )
            
            print("Retrieval Metrics:")
            print(f"  MRR: {retrieval_metrics['mrr']:.3f}")
            print(f"  Recall@3: {retrieval_metrics['recall_at_3']:.3f}")
            print(f"  Precision@3: {retrieval_metrics['precision_at_3']:.3f}")
            print(f"  NDCG@3: {retrieval_metrics['ndcg_at_3']:.3f}")
            
            # Evaluate generation
            generation_metrics = self.rag_evaluator.evaluate_generation(
                query_data['question'],
                generated_answer,
                [doc['content'] for doc in retrieved_docs],
                query_data['key_concepts']
            )
            
            print("Generation Metrics:")
            print(f"  Faithfulness: {generation_metrics['faithfulness']:.3f}")
            print(f"  Relevance: {generation_metrics['relevance']:.3f}")
            print(f"  Entity Coverage: {generation_metrics['entity_coverage']:.3f}")
            print(f"  Citation Accuracy: {generation_metrics['citation_accuracy']:.3f}")
            print()
    
    async def run_full_demo(self):
        """Run the complete production features demo"""
        print("🚀 EEG-RAG Production Features Demo")
        print("=" * 50)
        print("Demonstrating 5 production-ready improvements:\n")
        
        # Setup
        await self.setup_system()
        
        # Demo each component
        await self.demo_query_routing()
        print("-" * 50)
        
        self.demo_semantic_chunking()
        print("-" * 50)
        
        self.demo_hybrid_retrieval()
        print("-" * 50)
        
        await self.demo_citation_verification()
        print("-" * 50)
        
        await self.demo_rag_evaluation()
        print("-" * 50)
        
        print("✅ Demo completed! All production features demonstrated successfully.")
        print("\nKey Improvements Implemented:")
        print("1. ✅ RAG Evaluation Framework - Domain-specific benchmarks")
        print("2. ✅ Citation Verification - PubMed integration & hallucination detection")
        print("3. ✅ Hybrid Retrieval - BM25 + dense retrieval fusion")
        print("4. ✅ Query Routing - Intelligent agent selection")
        print("5. ✅ Semantic Chunking - Boundary-aware text segmentation")
        print("\n🎯 EEG-RAG is now production-ready for research applications!")


async def main():
    """Run the production features demo"""
    demo = ProductionDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
