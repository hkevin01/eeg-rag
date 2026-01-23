"""
Demo: All 5 Major Improvements for EEG-RAG

Demonstrates:
1. Response generation with multiple LLM providers
2. Real-time EEG signal integration
3. Feedback collection and learning
4. Citation network analysis
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eeg_rag.generation.response_generator import (
    ResponseGenerator,
    GenerationConfig,
    Document
)
from eeg_rag.signals.eeg_matcher import (
    EEGCaseMatcher,
    FeatureExtractor
)
from eeg_rag.feedback.learning import (
    FeedbackCollector,
    Feedback,
    TrainingDataset
)
from eeg_rag.analysis.citation_network import (
    CitationNetworkAnalyzer,
    ResearchFront
)


# ============== Improvement #2: LLM Response Generation ==============

async def demo_response_generation():
    """Demo multi-provider LLM response generation."""
    print("\n" + "="*60)
    print("DEMO: LLM Response Generation with Fallback")
    print("="*60)
    
    # Create sample retrieved documents
    documents = [
        Document(
            content="Alpha oscillations (8-13 Hz) are associated with wakeful relaxation.",
            pmid="12345678",
            title="EEG Alpha Rhythms in Cognitive Processing"
        ),
        Document(
            content="Theta activity increases during memory encoding tasks.",
            pmid="87654321",
            title="Theta Oscillations and Memory Formation"
        )
    ]
    
    # Initialize generator with fallback chain
    config = GenerationConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )
    
    generator = ResponseGenerator(config=config)
    
    # Generate response
    query = "What EEG frequency bands are involved in cognitive tasks?"
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(documents)} documents")
    print("\nGenerating response with citation...")
    
    try:
        response = await generator.generate_response(query, documents)
        
        print(f"\n✅ Generated Response ({response['provider']}):")
        print(f"{response['answer'][:300]}...")
        print(f"\nCitations: {response['citations']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============== Improvement #3: EEG Signal Integration ==============

def demo_eeg_signal_matching():
    """Demo real-time EEG signal analysis and case matching."""
    print("\n" + "="*60)
    print("DEMO: Real-Time EEG Signal Integration")
    print("="*60)
    
    # Generate synthetic EEG signal (250 Hz, 10 seconds, 4 channels)
    sampling_rate = 250
    duration = 10
    n_channels = 4
    
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Simulate multi-channel EEG with different frequency components
    eeg_data = np.zeros((n_channels, len(t)))
    
    # Channel 0: Strong alpha (10 Hz)
    eeg_data[0] = 50 * np.sin(2 * np.pi * 10 * t)
    
    # Channel 1: Theta + alpha
    eeg_data[1] = 30 * np.sin(2 * np.pi * 6 * t) + 40 * np.sin(2 * np.pi * 10 * t)
    
    # Channel 2: Beta activity
    eeg_data[2] = 20 * np.sin(2 * np.pi * 20 * t)
    
    # Channel 3: Mixed frequencies
    eeg_data[3] = (25 * np.sin(2 * np.pi * 8 * t) + 
                   15 * np.sin(2 * np.pi * 15 * t))
    
    # Add noise
    eeg_data += np.random.normal(0, 5, eeg_data.shape)
    
    print(f"\nSimulated EEG data: {eeg_data.shape}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Duration: {duration}s")
    
    # Extract features
    extractor = FeatureExtractor(sampling_rate=sampling_rate)
    features = extractor.extract(eeg_data)
    
    print(f"\n✅ Extracted Features:")
    print(f"  Delta power: {features.delta_power:.2f} μV²")
    print(f"  Theta power: {features.theta_power:.2f} μV²")
    print(f"  Alpha power: {features.alpha_power:.2f} μV²")
    print(f"  Beta power: {features.beta_power:.2f} μV²")
    print(f"  Gamma power: {features.gamma_power:.2f} μV²")
    print(f"  Dominant frequency: {features.dominant_frequency:.1f} Hz")
    print(f"  Peak amplitude: {features.peak_amplitude:.1f} μV")
    print(f"  Sample entropy: {features.sample_entropy:.3f}")
    print(f"  Asymmetry index: {features.asymmetry_index:.3f}")
    
    # Find similar cases
    matcher = EEGCaseMatcher(mock_retriever=True)
    
    print(f"\n✅ Feature Summary:")
    summary = matcher.get_feature_summary(features)
    print(summary)
    
    print("\nSearching for similar cases in literature...")
    similar_cases = matcher.find_similar_cases(eeg_data, sampling_rate, top_k=3)
    
    print(f"\nFound {len(similar_cases)} similar cases")
    for i, case in enumerate(similar_cases, 1):
        print(f"\n  {i}. Similarity: {case['similarity']:.2%}")
        print(f"     PMID: {case.get('pmid', 'N/A')}")
        print(f"     Title: {case.get('title', 'Mock result')[:60]}...")


# ============== Improvement #4: Feedback & Learning ==============

def demo_feedback_collection():
    """Demo feedback collection and training data generation."""
    print("\n" + "="*60)
    print("DEMO: Continuous Learning via Feedback")
    print("="*60)
    
    collector = FeedbackCollector()
    
    # Simulate user feedback
    feedback_items = [
        Feedback(
            query_id="q1",
            rating=5,
            clicked_pmids=["12345678", "23456789"],
            ignored_pmids=["34567890"]
        ),
        Feedback(
            query_id="q2",
            rating=4,
            clicked_pmids=["45678901"],
            ignored_pmids=["56789012", "67890123"]
        ),
        Feedback(
            query_id="q3",
            rating=2,
            clicked_pmids=[],
            ignored_pmids=["78901234"]
        )
    ]
    
    queries = [
        "What are alpha oscillations?",
        "EEG biomarkers for epilepsy",
        "Sleep spindles detection"
    ]
    
    print("\nRecording user feedback...")
    for query, feedback in zip(queries, feedback_items):
        collector.record_feedback(query, feedback)
        print(f"  ✓ Query: {query[:40]}... (rating={feedback.rating})")
    
    # Generate training data
    print("\nGenerating training dataset from feedback...")
    dataset = collector.generate_training_data(min_rating=4)
    
    print(f"\n✅ Training Dataset:")
    print(f"  Total pairs: {len(dataset.pairs)}")
    print(f"  Positive examples: {len([p for p in dataset.pairs if p.score >= 0.8])}")
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"\n✅ Feedback Statistics:")
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Average rating: {stats['avg_rating']:.2f}/5")
    print(f"  Positive ratio: {stats['positive_ratio']:.1%}")
    print(f"  Click-through rate: {stats['click_through_rate']:.1%}")
    
    # Export for training
    output_path = "/tmp/eeg_rag_training.jsonl"
    dataset.export_jsonl(output_path)
    print(f"\n✅ Exported training data to: {output_path}")


# ============== Improvement #5: Citation Network Analysis ==============

def demo_citation_network():
    """Demo citation network analysis and research fronts."""
    print("\n" + "="*60)
    print("DEMO: Citation Network Analysis")
    print("="*60)
    
    # Note: Requires graph database connection in production
    # Using mock for demonstration
    
    class MockGraphStore:
        """Mock graph store for demo."""
        pass
    
    analyzer = CitationNetworkAnalyzer(graph_store=MockGraphStore())
    
    print("\nAnalyzing research fronts for 'seizure detection'...")
    fronts = analyzer.find_research_fronts("seizure detection", years=3)
    
    print(f"\n✅ Detected {len(fronts)} research fronts")
    if fronts:
        for front in fronts:
            print(f"  - {front}")
    else:
        print("  (No fronts detected - requires live graph database)")
    
    print("\nGenerating literature map for 'EEG classification'...")
    lit_map = analyzer.generate_literature_map("EEG classification", top_k=20)
    
    print(f"\n✅ Literature Map:")
    print(f"  Nodes (papers): {lit_map['metadata']['total_papers']}")
    print(f"  Edges (citations): {lit_map['metadata']['total_citations']}")
    print(f"  Clusters: {lit_map['metadata']['num_clusters']}")
    
    print("\nAnalyzing topic trend: 'deep learning EEG' (2020-2024)...")
    trend = analyzer.analyze_trend("deep learning EEG", 2020, 2024)
    
    print(f"\n✅ Trend Analysis:")
    print(f"  Time period: {trend['time_period']}")
    print(f"  Total papers: {trend['total_papers']}")
    print(f"  Growth rate: {trend['growth_rate']:.1%} per year")
    print(f"  Yearly counts: {trend['yearly_counts']}")


# ============== Main Demo ==============

async def main():
    """Run all improvement demos."""
    print("\n" + "="*70)
    print("EEG-RAG: Top 5 Improvements Demo")
    print("="*70)
    print("\nDemonstrating production-ready enhancements:")
    print("  1. FastAPI Production Deployment (see k8s/ directory)")
    print("  2. Full LLM Response Generation")
    print("  3. Real-Time EEG Data Integration")
    print("  4. Continuous Learning Pipeline")
    print("  5. Automated Citation Network Analysis")
    
    # Run async demos
    await demo_response_generation()
    
    # Run sync demos
    demo_eeg_signal_matching()
    demo_feedback_collection()
    demo_citation_network()
    
    print("\n" + "="*70)
    print("✅ All demos completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Deploy with: kubectl apply -f k8s/")
    print("  2. Start API: uvicorn eeg_rag.api.main:app --reload")
    print("  3. Test auth: Use JWT tokens from middleware.create_access_token()")
    print("  4. Monitor: Check /metrics endpoint for telemetry")
    print("  5. Collect feedback: Use /feedback API endpoint")


if __name__ == "__main__":
    asyncio.run(main())
