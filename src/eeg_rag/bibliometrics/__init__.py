"""
EEG-RAG Bibliometrics Module

Network-based bibliometric analysis for EEG research using pyBiblioNet.

This module provides:
- EEG literature retrieval from OpenAlex
- Citation and co-authorship network construction
- Centrality and influence metrics for EEG papers/authors
- Community detection in EEG research domains
- Integration with RAG for bibliometric-enhanced retrieval

Requirements:
- REQ-BIB-001: OpenAlex integration for article retrieval
- REQ-BIB-002: Citation network analysis
- REQ-BIB-003: Co-authorship network analysis
- REQ-BIB-004: Centrality metrics computation
- REQ-BIB-005: RAG enhancement with bibliometric data
"""

from .eeg_biblionet import (
    EEGBiblioNet,
    EEGArticle,
    EEGAuthor,
    NetworkMetrics,
    retrieve_eeg_articles,
    build_eeg_citation_network,
    build_eeg_coauthorship_network,
    get_influential_papers,
    get_influential_authors,
)

__all__ = [
    "EEGBiblioNet",
    "EEGArticle", 
    "EEGAuthor",
    "NetworkMetrics",
    "retrieve_eeg_articles",
    "build_eeg_citation_network",
    "build_eeg_coauthorship_network",
    "get_influential_papers",
    "get_influential_authors",
]
