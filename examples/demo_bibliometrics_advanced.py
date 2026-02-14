#!/usr/bin/env python3
"""
Advanced Bibliometrics Demo for EEG-RAG

Demonstrates the visualization, NLP enhancement, and research export features
of the bibliometrics module.

Usage:
    python examples/demo_bibliometrics_advanced.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import bibliometrics components
from eeg_rag.bibliometrics import (
    # Core
    EEGBiblioNet,
    EEGArticle,
    # Visualization
    EEGVisualization,
    ChartResult,
    # NLP Enhancement
    EEGNLPEnhancer,
    ExtractedKeywords,
    TopicCluster,
    # Research Export
    EEGResearchExporter,
    VenueMetrics,
    InstitutionMetrics,
    AuthorProductivity,
)


def create_sample_articles():
    """Create sample EEG articles for demonstration."""
    return [
        EEGArticle(
            openalex_id="W1234567890",
            title="EEG-based seizure detection using deep learning",
            abstract="We present a novel deep learning approach for automated "
                     "seizure detection using EEG signals. Our method achieves "
                     "95% sensitivity and 98% specificity on the CHB-MIT dataset.",
            authors=["Smith J", "Jones A", "Williams R"],
            year=2023,
            venue="Journal of Neural Engineering",
            doi="10.1088/1741-2552/abc123",
            citation_count=45,
            concepts=["seizure detection", "deep learning", "EEG classification"],
            references=["W9876543210", "W1111111111"],
        ),
        EEGArticle(
            openalex_id="W1234567891",
            title="Alpha oscillations and cognitive load in working memory tasks",
            abstract="This study investigates the relationship between alpha band "
                     "(8-13 Hz) oscillations and cognitive load during working "
                     "memory tasks. Results show increased alpha suppression under "
                     "high cognitive load conditions.",
            authors=["Brown M", "Davis K", "Smith J"],
            year=2022,
            venue="NeuroImage",
            doi="10.1016/j.neuroimage.2022.xyz",
            citation_count=78,
            concepts=["alpha oscillations", "working memory", "cognitive load"],
            references=["W1234567890", "W2222222222"],
        ),
        EEGArticle(
            openalex_id="W1234567892",
            title="Sleep spindle detection methods: A comparative study",
            abstract="We compare various automated methods for sleep spindle "
                     "detection including traditional signal processing and "
                     "machine learning approaches. The optimal approach depends "
                     "on data quality and computational constraints.",
            authors=["Wilson T", "Johnson P", "Anderson L"],
            year=2024,
            venue="Sleep Medicine Reviews",
            doi="10.1016/j.smrv.2024.abc",
            citation_count=12,
            concepts=["sleep spindles", "signal processing", "machine learning"],
            references=["W3333333333"],
        ),
        EEGArticle(
            openalex_id="W1234567893",
            title="P300-based brain-computer interface for locked-in patients",
            abstract="This paper presents a P300-based BCI system designed for "
                     "patients with locked-in syndrome. The system achieves "
                     "reliable communication with minimal training.",
            authors=["Garcia E", "Lee C", "Kim H"],
            year=2023,
            venue="IEEE Transactions on Biomedical Engineering",
            doi="10.1109/tbme.2023.def",
            citation_count=34,
            concepts=["P300", "BCI", "locked-in syndrome", "communication"],
            references=["W1234567890", "W4444444444"],
        ),
        EEGArticle(
            openalex_id="W1234567894",
            title="Theta oscillations in spatial navigation: An EEG study",
            abstract="Using high-density EEG, we investigated theta band "
                     "(4-8 Hz) oscillations during spatial navigation in a "
                     "virtual reality environment. Strong theta synchronization "
                     "was observed in frontal-parietal networks.",
            authors=["Martinez R", "Smith J", "Thompson K"],
            year=2022,
            venue="Cerebral Cortex",
            doi="10.1093/cercor/2022",
            citation_count=56,
            concepts=["theta oscillations", "spatial navigation", "virtual reality"],
            references=["W1234567891", "W5555555555"],
        ),
    ]


async def demo_visualization():
    """Demonstrate visualization features."""
    print("\n" + "="*60)
    print("📊 VISUALIZATION DEMO")
    print("="*60)
    
    articles = create_sample_articles()
    viz = EEGVisualization(style="seaborn-v0_8-whitegrid")
    
    # 1. Publication Trends
    print("\n1. Creating publication trends chart...")
    trends = viz.plot_publication_trends(
        articles,
        title="EEG Research Publication Trends"
    )
    print(f"   Chart type: {trends.chart_type}")
    print(f"   Data points: {trends.metadata.get('year_count', 'N/A')}")
    
    # 2. Top Authors
    print("\n2. Creating top authors chart...")
    authors = viz.plot_top_authors(
        articles,
        top_n=5,
        title="Most Prolific EEG Researchers"
    )
    print(f"   Chart type: {authors.chart_type}")
    print(f"   Authors shown: {min(5, len(set(a for art in articles for a in art.authors)))}")
    
    # 3. Citation Distribution
    print("\n3. Creating citation distribution...")
    citations = viz.plot_citation_distribution(
        articles,
        title="Citation Distribution in EEG Literature"
    )
    print(f"   Chart type: {citations.chart_type}")
    print(f"   Total articles: {len(articles)}")
    
    # 4. Research Dashboard
    print("\n4. Creating research dashboard...")
    dashboard = viz.create_research_dashboard(
        articles,
        title="EEG Research Overview Dashboard"
    )
    print(f"   Dashboard panels: {dashboard.metadata.get('num_panels', 4)}")
    
    print("\n✅ Visualization demo complete!")
    return viz


async def demo_nlp_enhancement():
    """Demonstrate NLP enhancement features."""
    print("\n" + "="*60)
    print("🔤 NLP ENHANCEMENT DEMO")
    print("="*60)
    
    articles = create_sample_articles()
    nlp = EEGNLPEnhancer()
    
    # 1. Keyword Extraction
    print("\n1. Extracting keywords from sample abstract...")
    sample_text = articles[0].abstract
    keywords = nlp.extract_keywords_from_text(sample_text, top_n=5)
    print(f"   Top keywords: {[kw for kw, _ in keywords.keywords[:5]]}")
    print(f"   Total extracted: {len(keywords.keywords)}")
    
    # 2. Keyword Extraction from Articles
    print("\n2. Extracting keywords from all articles...")
    all_keywords = nlp.extract_keywords_from_articles(articles, top_n=10)
    print(f"   Top research keywords: {[kw for kw, _ in all_keywords.keywords[:5]]}")
    
    # 3. Query Expansion
    print("\n3. Expanding EEG research queries...")
    queries = [
        "seizure detection",
        "alpha oscillations",
        "BCI systems"
    ]
    for query in queries:
        expanded = nlp.expand_query(query)
        print(f"   '{query}' -> {expanded[:3]}...")
    
    # 4. Topic Clustering
    print("\n4. Categorizing articles by topic...")
    clusters = nlp.categorize_by_topic(articles, num_topics=3)
    for cluster in clusters:
        print(f"   Topic '{cluster.topic_label}': {len(cluster.article_ids)} articles")
        print(f"      Keywords: {cluster.keywords[:3]}")
    
    # 5. Text Similarity
    print("\n5. Computing text similarity...")
    text1 = articles[0].abstract
    text2 = articles[1].abstract
    similarity = nlp.compute_text_similarity(text1, text2)
    print(f"   Similarity between first two abstracts: {similarity:.3f}")
    
    # 6. Keyword Trends
    print("\n6. Analyzing keyword trends...")
    trends = nlp.get_keyword_trends(articles)
    print(f"   Trending keywords by year:")
    for year, words in list(trends.items())[:3]:
        print(f"      {year}: {words[:3]}")
    
    print("\n✅ NLP Enhancement demo complete!")
    return nlp


async def demo_research_export():
    """Demonstrate research export features."""
    print("\n" + "="*60)
    print("📤 RESEARCH EXPORT DEMO")
    print("="*60)
    
    articles = create_sample_articles()
    exporter = EEGResearchExporter()
    
    # 1. Compute Metrics
    print("\n1. Computing venue metrics...")
    venue_metrics = exporter.compute_venue_metrics(articles)
    for venue, metrics in list(venue_metrics.items())[:3]:
        print(f"   {venue[:30]:30} - {metrics.total_articles} articles, "
              f"avg citations: {metrics.avg_citations:.1f}")
    
    # 2. Author Productivity
    print("\n2. Computing author productivity...")
    author_prod = exporter.compute_author_productivity(articles)
    for author, prod in list(author_prod.items())[:3]:
        print(f"   {author:20} - {prod.total_publications} pubs, "
              f"h-index: {prod.h_index}")
    
    # 3. Export Options
    print("\n3. Export formats available:")
    print("   - Scopus-compatible CSV (export_to_scopus_csv)")
    print("   - Authors CSV (export_authors_csv)")
    print("   - Venues CSV (export_venues_csv)")
    print("   - Institutions CSV (export_institutions_csv)")
    print("   - Complete ZIP package (export_all)")
    
    # 4. Collaboration Network
    print("\n4. Computing collaboration network data...")
    collab_data = exporter.get_collaboration_network_data(articles)
    print(f"   Nodes (authors): {collab_data['num_nodes']}")
    print(f"   Edges (collaborations): {collab_data['num_edges']}")
    
    print("\n✅ Research Export demo complete!")
    return exporter


async def demo_integrated_workflow():
    """Demonstrate integrated workflow with all components."""
    print("\n" + "="*60)
    print("🔗 INTEGRATED WORKFLOW DEMO")
    print("="*60)
    
    articles = create_sample_articles()
    
    # Initialize all components
    viz = EEGVisualization()
    nlp = EEGNLPEnhancer()
    exporter = EEGResearchExporter()
    
    # Step 1: Analyze with NLP
    print("\n1. NLP Analysis...")
    keywords = nlp.extract_keywords_from_articles(articles)
    clusters = nlp.categorize_by_topic(articles, num_topics=2)
    
    # Step 2: Visualize Results
    print("2. Creating Visualizations...")
    dashboard = viz.create_research_dashboard(articles, "EEG Research Analysis")
    
    # Step 3: Compute Metrics
    print("3. Computing Research Metrics...")
    metrics = exporter.compute_venue_metrics(articles)
    productivity = exporter.compute_author_productivity(articles)
    
    # Step 4: Generate Summary
    print("\n📋 ANALYSIS SUMMARY:")
    print(f"   Total articles analyzed: {len(articles)}")
    print(f"   Key research themes: {[kw for kw, _ in keywords.keywords[:3]]}")
    print(f"   Topic clusters identified: {len(clusters)}")
    print(f"   Unique venues: {len(metrics)}")
    print(f"   Active researchers: {len(productivity)}")
    
    # Most cited work
    most_cited = max(articles, key=lambda a: a.citation_count)
    print(f"\n   Most cited work: '{most_cited.title[:50]}...'")
    print(f"   Citations: {most_cited.citation_count}")
    
    # Most prolific author
    top_author = max(productivity.items(), key=lambda x: x[1].total_publications)
    print(f"\n   Most prolific author: {top_author[0]}")
    print(f"   Publications: {top_author[1].total_publications}")
    
    print("\n✅ Integrated workflow demo complete!")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  EEG-RAG ADVANCED BIBLIOMETRICS DEMO")
    print("  Visualization | NLP | Export")
    print("="*60)
    
    await demo_visualization()
    await demo_nlp_enhancement()
    await demo_research_export()
    await demo_integrated_workflow()
    
    print("\n" + "="*60)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
