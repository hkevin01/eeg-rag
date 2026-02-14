#!/usr/bin/env python3
"""
EEG Bibliometrics Integration Demo

Demonstrates pyBiblioNet integration for network-based bibliometric analysis
of EEG research literature.

Requirements:
    pip install pybiblionet networkx

Usage:
    python examples/demo_bibliometrics.py

Note:
    Requires pyBiblioNet to be installed for full functionality.
    Without pyBiblioNet, demonstrates data structures and mock workflow.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def demo_data_structures():
    """Demonstrate bibliometrics data structures (no external deps needed)."""
    from eeg_rag.bibliometrics import EEGArticle, EEGAuthor, NetworkMetrics
    from eeg_rag.bibliometrics.eeg_biblionet import EEGResearchDomain
    
    print("\n" + "=" * 60)
    print("EEG Bibliometrics Data Structures Demo")
    print("=" * 60)
    
    # 1. EEGResearchDomain - Pre-defined research domains
    print("\n1. Available EEG Research Domains:")
    for domain in EEGResearchDomain:
        print(f"   - {domain.name}: {domain.value}")
    
    # 2. EEGArticle - Article data structure
    print("\n2. Creating EEGArticle:")
    article = EEGArticle(
        openalex_id="https://openalex.org/W12345678",
        doi="https://doi.org/10.1016/j.clinph.2023.001",
        pmid="12345678",
        title="Deep Learning for EEG-Based Epileptic Seizure Detection",
        abstract="This study presents a novel CNN-LSTM architecture for detecting epileptic seizures from scalp EEG recordings.",
        authors=["Jane Smith", "John Doe", "Alice Johnson"],
        publication_date="2023-06-15",
        cited_by_count=42,
        venue="Clinical Neurophysiology",
        topics=["Electroencephalography", "Epilepsy", "Deep Learning", "Seizure Detection"],
        referenced_works=["W111", "W222", "W333"],
        centrality_score=0.0856,
    )
    
    print(f"   Title: {article.title}")
    print(f"   Authors: {', '.join(article.authors)}")
    print(f"   Citations: {article.cited_by_count}")
    print(f"   Venue: {article.venue}")
    print(f"   Topics: {', '.join(article.topics[:3])}...")
    
    # 3. Serialize to dict
    print("\n3. Article as dictionary (for RAG ingestion):")
    article_dict = article.to_dict()
    print(f"   Keys: {list(article_dict.keys())}")
    
    # 4. Parse from OpenAlex format
    print("\n4. Parsing OpenAlex API response:")
    openalex_data = {
        "id": "https://openalex.org/W99999999",
        "doi": "https://doi.org/10.1000/example",
        "title": "Motor Imagery Classification with EEG-based BCI",
        "abstract": "Novel approach for motor imagery classification.",
        "publication_date": "2024-01-15",
        "cited_by_count": 15,
        "authorships": [
            {"author": {"id": "A1", "display_name": "Dr. Neural"}},
            {"author": {"id": "A2", "display_name": "Prof. Brain"}},
        ],
        "primary_location": {
            "source": {"display_name": "Journal of Neural Engineering"}
        },
        "topics": [
            {"display_name": "Brain-Computer Interface"},
            {"display_name": "Motor Imagery"},
        ],
        "ids": {
            "pmid": "https://pubmed.ncbi.nlm.nih.gov/98765432"
        },
    }
    
    parsed = EEGArticle.from_openalex(openalex_data)
    print(f"   Parsed: {parsed.title}")
    print(f"   Authors: {parsed.authors}")
    print(f"   PMID: {parsed.pmid}")
    
    # 5. EEGAuthor
    print("\n5. Creating EEGAuthor:")
    author = EEGAuthor(
        openalex_id="https://openalex.org/A12345",
        name="Dr. Jane Smith",
        affiliations=["MIT", "Harvard Medical School"],
        works_count=127,
        cited_by_count=4500,
        h_index=35,
        centrality_score=0.125,
        collaboration_count=89,
    )
    print(f"   Name: {author.name}")
    print(f"   h-index: {author.h_index}")
    print(f"   Citations: {author.cited_by_count}")
    
    # 6. NetworkMetrics
    print("\n6. Network Metrics Example:")
    metrics = NetworkMetrics(
        node_count=1500,
        edge_count=8500,
        density=0.0076,
        avg_clustering=0.42,
        num_components=12,
        avg_degree=11.33,
        max_degree=256,
        diameter=8,
    )
    print(f"   Nodes: {metrics.node_count}")
    print(f"   Edges: {metrics.edge_count}")
    print(f"   Density: {metrics.density:.4f}")
    print(f"   Avg Clustering: {metrics.avg_clustering:.2f}")
    print(f"   Diameter: {metrics.diameter}")


def demo_with_pybiblionet():
    """Demonstrate full workflow with pyBiblioNet (requires installation)."""
    try:
        from eeg_rag.bibliometrics import EEGBiblioNet
        from eeg_rag.bibliometrics.eeg_biblionet import EEGResearchDomain
    except ImportError:
        print("\nCould not import EEGBiblioNet. Skipping full demo.")
        return
    
    print("\n" + "=" * 60)
    print("EEG Bibliometrics with pyBiblioNet (Full Demo)")
    print("=" * 60)
    
    # Check if pyBiblioNet is available
    try:
        biblio = EEGBiblioNet(
            email="demo@example.com",
            cache_dir=Path("/tmp/eeg_biblionet_demo"),
            use_cache=True,
        )
    except Exception as e:
        print(f"\nNote: Could not initialize EEGBiblioNet: {e}")
        print("This demo requires pyBiblioNet: pip install pybiblionet")
        return
    
    if not biblio._pybiblionet_available:
        print("\n⚠️  pyBiblioNet not installed. Showing mock workflow...")
        print("   Install with: pip install pybiblionet")
        print("   Then: python -m spacy download en_core_web_sm")
        print("\nExample workflow (would run with pyBiblioNet):")
        print("""
    # 1. Retrieve EEG epilepsy research from OpenAlex
    articles = biblio.search_eeg_literature(
        domain=EEGResearchDomain.EPILEPSY,
        from_date="2023-01-01",
        max_results=500,
    )
    
    # 2. Build citation network
    citation_graph = biblio.build_citation_network(
        output_path="eeg_citations.gml"
    )
    
    # 3. Build co-authorship network  
    coauthorship_graph = biblio.build_coauthorship_network(
        output_path="eeg_coauthorship.gml"
    )
    
    # 4. Find influential papers
    top_papers = biblio.get_influential_papers(top_n=10)
    for paper in top_papers:
        print(f"{paper.title}: {paper.centrality_score:.4f}")
    
    # 5. Find influential authors
    top_authors = biblio.get_influential_authors(top_n=10)
    for author, score in top_authors:
        print(f"{author}: {score:.4f}")
    
    # 6. Get articles for RAG ingestion
    rag_docs = biblio.get_articles_for_rag(
        min_citations=10,
        min_centrality=0.01,
    )
        """)
        return
    
    # Full demo with pyBiblioNet
    print("\n1. Searching for EEG epilepsy literature...")
    try:
        articles = biblio.search_eeg_literature(
            domain=EEGResearchDomain.EPILEPSY,
            from_date="2024-01-01",
            max_results=50,  # Small sample for demo
        )
        print(f"   Retrieved {len(articles)} articles")
        
        if articles:
            print(f"\n   Sample article: {articles[0].title[:60]}...")
            
            print("\n2. Building citation network...")
            G = biblio.build_citation_network()
            print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            
            print("\n3. Computing centrality metrics...")
            biblio.compute_citation_centrality(method="pagerank")
            
            print("\n4. Getting influential papers...")
            top_papers = biblio.get_influential_papers(top_n=5)
            for i, paper in enumerate(top_papers, 1):
                print(f"   {i}. {paper.title[:50]}... (score: {paper.centrality_score:.4f})")
            
            print("\n5. Preparing articles for RAG ingestion...")
            rag_docs = biblio.get_articles_for_rag(min_citations=5)
            print(f"   Prepared {len(rag_docs)} articles for RAG")
            
    except Exception as e:
        print(f"   Error during demo: {e}")
        print("   This may be due to API rate limits or network issues.")


def demo_eeg_query_patterns():
    """Show the pre-defined EEG query patterns."""
    from eeg_rag.bibliometrics.eeg_biblionet import EEGBiblioNet, EEGResearchDomain
    
    print("\n" + "=" * 60)
    print("Pre-defined EEG Query Patterns for OpenAlex")
    print("=" * 60)
    
    for domain, pattern in EEGBiblioNet.EEG_QUERY_PATTERNS.items():
        print(f"\n{domain.name}:")
        print(f"  Pattern: {pattern}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("EEG-RAG Bibliometrics Integration Demo")
    print("pyBiblioNet for Network-Based Bibliometric Analysis")
    print("=" * 60)
    
    # Demo data structures (always works)
    demo_data_structures()
    
    # Show query patterns
    demo_eeg_query_patterns()
    
    # Full demo (requires pyBiblioNet)
    demo_with_pybiblionet()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    print("""
Next Steps:
1. Install pyBiblioNet: pip install pybiblionet
2. Install spaCy model: python -m spacy download en_core_web_sm
3. Set your email for OpenAlex API (polite pool)
4. Run full bibliometric analysis on EEG research

For more information:
- pyBiblioNet: https://github.com/giorgioavena/pybiblionet
- OpenAlex API: https://docs.openalex.org/
    """)


if __name__ == "__main__":
    main()
