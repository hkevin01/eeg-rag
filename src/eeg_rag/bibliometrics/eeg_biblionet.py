"""
EEG BiblioNet Integration Module

Integrates pyBiblioNet for network-based bibliometric analysis of EEG research.

This module wraps pyBiblioNet functionality with EEG-specific features:
- Pre-defined EEG query patterns for OpenAlex
- Citation network analysis for EEG literature
- Co-authorship network analysis in EEG research
- Influence metrics for papers and authors
- Integration hooks for RAG enhancement

Requirements:
- REQ-BIB-001: OpenAlex integration for article retrieval
- REQ-BIB-002: Citation network analysis
- REQ-BIB-003: Co-authorship network analysis
- REQ-BIB-004: Centrality metrics computation
- REQ-BIB-005: RAG enhancement with bibliometric data

Author: EEG-RAG Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class EEGResearchDomain(Enum):
    """
    EEG research domains for targeted bibliometric analysis.
    
    REQ-BIB-001: Pre-defined research domains for EEG literature retrieval.
    """
    
    EPILEPSY = "epilepsy"
    SLEEP = "sleep"
    BCI = "brain_computer_interface"
    COGNITIVE = "cognitive_neuroscience"
    CLINICAL = "clinical_neurophysiology"
    SIGNAL_PROCESSING = "signal_processing"
    GENERAL = "general_eeg"


@dataclass
class EEGArticle:
    """
    Represents an EEG research article with bibliometric metadata.
    
    REQ-BIB-001: Article data structure for bibliometric analysis.
    
    Attributes:
        openalex_id: OpenAlex unique identifier
        doi: Digital Object Identifier
        pmid: PubMed identifier
        title: Article title
        abstract: Article abstract
        authors: List of author names
        publication_date: Publication date
        cited_by_count: Number of citations
        venue: Journal or conference name
        topics: List of research topics
        referenced_works: List of cited work IDs
        citation_count: Incoming citation count
        centrality_score: Computed centrality in citation network
    """
    
    openalex_id: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    cited_by_count: int = 0
    venue: str = ""
    topics: List[str] = field(default_factory=list)
    referenced_works: List[str] = field(default_factory=list)
    citation_count: int = 0
    centrality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "openalex_id": self.openalex_id,
            "doi": self.doi,
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "cited_by_count": self.cited_by_count,
            "venue": self.venue,
            "topics": self.topics,
            "referenced_works": self.referenced_works,
            "citation_count": self.citation_count,
            "centrality_score": self.centrality_score,
        }
    
    @classmethod
    def from_openalex(cls, data: Dict[str, Any]) -> "EEGArticle":
        """
        Create EEGArticle from OpenAlex API response.
        
        REQ-BIB-001: Parse OpenAlex article data.
        
        Args:
            data: OpenAlex article data dictionary
            
        Returns:
            EEGArticle instance
        """
        # Extract author names
        authors = []
        authorships = data.get("authorships", [])
        for authorship in authorships:
            author_info = authorship.get("author", {})
            display_name = author_info.get("display_name", "")
            if display_name:
                authors.append(display_name)
        
        # Extract topics
        topics = []
        for topic in data.get("topics", []):
            topic_name = topic.get("display_name", "")
            if topic_name:
                topics.append(topic_name)
        
        # Extract venue
        venue = ""
        primary_location = data.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source", {})
            if source:
                venue = source.get("display_name", "")
        
        # Extract PMID from IDs
        pmid = None
        ids = data.get("ids", {})
        if "pmid" in ids:
            pmid_url = ids["pmid"]
            if pmid_url and "pubmed.ncbi.nlm.nih.gov/" in pmid_url:
                pmid = pmid_url.split("/")[-1]
        
        return cls(
            openalex_id=data.get("id", ""),
            doi=data.get("doi"),
            pmid=pmid,
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            authors=authors,
            publication_date=data.get("publication_date"),
            cited_by_count=data.get("cited_by_count", 0),
            venue=venue,
            topics=topics,
            referenced_works=data.get("referenced_works", []),
        )


@dataclass
class EEGAuthor:
    """
    Represents an EEG researcher with bibliometric metrics.
    
    REQ-BIB-003: Author data structure for co-authorship analysis.
    
    Attributes:
        openalex_id: OpenAlex unique identifier
        name: Author display name
        affiliations: List of institutional affiliations
        works_count: Number of publications
        cited_by_count: Total citations across works
        h_index: Computed h-index
        centrality_score: Centrality in co-authorship network
        collaboration_count: Number of unique co-authors
    """
    
    openalex_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    works_count: int = 0
    cited_by_count: int = 0
    h_index: int = 0
    centrality_score: float = 0.0
    collaboration_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "openalex_id": self.openalex_id,
            "name": self.name,
            "affiliations": self.affiliations,
            "works_count": self.works_count,
            "cited_by_count": self.cited_by_count,
            "h_index": self.h_index,
            "centrality_score": self.centrality_score,
            "collaboration_count": self.collaboration_count,
        }


@dataclass
class NetworkMetrics:
    """
    Network-level metrics for bibliometric analysis.
    
    REQ-BIB-004: Network metrics data structure.
    
    Attributes:
        node_count: Number of nodes in network
        edge_count: Number of edges in network
        density: Network density (0-1)
        avg_clustering: Average clustering coefficient
        num_components: Number of connected components
        avg_degree: Average node degree
        max_degree: Maximum node degree
        diameter: Network diameter (longest shortest path)
    """
    
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_clustering: float = 0.0
    num_components: int = 0
    avg_degree: float = 0.0
    max_degree: int = 0
    diameter: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "avg_clustering": self.avg_clustering,
            "num_components": self.num_components,
            "avg_degree": self.avg_degree,
            "max_degree": self.max_degree,
            "diameter": self.diameter,
        }


class EEGBiblioNet:
    """
    Main class for EEG-specific bibliometric network analysis.
    
    Wraps pyBiblioNet functionality with EEG domain expertise.
    
    REQ-BIB-001 through REQ-BIB-005: Comprehensive bibliometric analysis.
    
    Attributes:
        email: Email for OpenAlex API (polite pool)
        cache_dir: Directory for caching retrieved data
        articles: Retrieved EEG articles
        citation_graph: NetworkX citation graph
        coauthorship_graph: NetworkX co-authorship graph
        
    Example:
        >>> biblio = EEGBiblioNet(email="researcher@university.edu")
        >>> articles = await biblio.search_eeg_literature(
        ...     domain=EEGResearchDomain.EPILEPSY,
        ...     from_date="2020-01-01"
        ... )
        >>> influential = biblio.get_influential_papers(top_n=10)
    """
    
    # EEG-specific query patterns for OpenAlex
    # REQ-BIB-001: Pre-defined EEG query patterns
    EEG_QUERY_PATTERNS: Dict[EEGResearchDomain, str] = {
        EEGResearchDomain.GENERAL: "(EEG|electroencephalography|electroencephalogram)",
        EEGResearchDomain.EPILEPSY: "(EEG|electroencephalography)( )(epilepsy|seizure|ictal|interictal)",
        EEGResearchDomain.SLEEP: "(EEG|electroencephalography)( )(sleep|polysomnography|PSG|sleep staging)",
        EEGResearchDomain.BCI: "(EEG|electroencephalography)( )(BCI|brain-computer interface|brain machine interface|BMI|motor imagery)",
        EEGResearchDomain.COGNITIVE: "(EEG|electroencephalography)( )(ERP|event-related potential|P300|N400|cognitive)",
        EEGResearchDomain.CLINICAL: "(EEG|electroencephalography)( )(clinical|neurophysiology|neurology|diagnosis)",
        EEGResearchDomain.SIGNAL_PROCESSING: "(EEG|electroencephalography)( )(signal processing|artifact|filtering|preprocessing|ICA)",
    }
    
    def __init__(
        self,
        email: str,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize EEGBiblioNet instance.
        
        REQ-BIB-001: Initialize bibliometric analysis system.
        
        Args:
            email: Email for OpenAlex API (required for polite pool)
            cache_dir: Directory for caching retrieved data
            use_cache: Whether to use cached data when available
            
        Raises:
            ValueError: If email is not provided
        """
        if not email:
            raise ValueError("Email is required for OpenAlex API access")
        
        self.email = email
        self.use_cache = use_cache
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "eeg_biblionet_cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.articles: List[EEGArticle] = []
        self.authors: Dict[str, EEGAuthor] = {}
        self.citation_graph = None
        self.coauthorship_graph = None
        
        # Check if pyBiblioNet is available
        self._pybiblionet_available = self._check_pybiblionet()
        
        logger.info(
            f"EEGBiblioNet initialized with email={email}, "
            f"cache_dir={self.cache_dir}, pybiblionet_available={self._pybiblionet_available}"
        )
    
    def _check_pybiblionet(self) -> bool:
        """
        Check if pyBiblioNet is installed and available.
        
        Returns:
            True if pyBiblioNet is available, False otherwise
        """
        try:
            from pybiblionet.openalex.core import retrieve_articles
            return True
        except ImportError:
            logger.warning(
                "pyBiblioNet not installed. Install with: pip install pybiblionet"
            )
            return False
    
    def search_eeg_literature(
        self,
        domain: EEGResearchDomain = EEGResearchDomain.GENERAL,
        custom_query: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[EEGArticle]:
        """
        Search for EEG literature using OpenAlex via pyBiblioNet.
        
        REQ-BIB-001: Retrieve EEG articles from OpenAlex.
        
        Args:
            domain: EEG research domain to search
            custom_query: Custom query pattern (overrides domain)
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            max_results: Maximum number of articles to retrieve
            
        Returns:
            List of EEGArticle objects
            
        Raises:
            RuntimeError: If pyBiblioNet is not available
            
        Example:
            >>> articles = biblio.search_eeg_literature(
            ...     domain=EEGResearchDomain.EPILEPSY,
            ...     from_date="2023-01-01"
            ... )
        """
        if not self._pybiblionet_available:
            raise RuntimeError(
                "pyBiblioNet is not installed. Install with: pip install pybiblionet"
            )
        
        from pybiblionet.openalex.core import (
            string_generator_from_lite_regex,
            retrieve_articles,
        )
        
        # Determine query pattern
        if custom_query:
            query_pattern = custom_query
        else:
            query_pattern = self.EEG_QUERY_PATTERNS[domain]
        
        # Generate query strings from pattern
        queries = string_generator_from_lite_regex(query_pattern)
        logger.info(f"Generated {len(queries)} query variations from pattern: {query_pattern}")
        
        # Set default dates if not provided
        if not from_date:
            # Default to last 2 years
            two_years_ago = datetime.now() - timedelta(days=730)
            from_date = two_years_ago.strftime("%Y-%m-%d")
        
        # Check cache first
        cache_key = f"{domain.value}_{from_date}_{to_date}_{hash(query_pattern)}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if self.use_cache and cache_file.exists():
            logger.info(f"Loading cached articles from {cache_file}")
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            self.articles = [EEGArticle(**article) for article in cached_data]
            return self.articles
        
        # Retrieve articles from OpenAlex
        logger.info(
            f"Retrieving EEG articles: domain={domain.value}, "
            f"from_date={from_date}, to_date={to_date}"
        )
        
        json_file = retrieve_articles(
            queries=queries,
            mail=self.email,
            from_publication_date=from_date,
            to_publication_date=to_date,
        )
        
        # Load and parse articles
        with open(json_file, "r") as f:
            raw_articles = json.load(f)
        
        # Convert to EEGArticle objects
        self.articles = []
        for raw in raw_articles[:max_results]:
            article = EEGArticle.from_openalex(raw)
            self.articles.append(article)
        
        logger.info(f"Retrieved {len(self.articles)} EEG articles")
        
        # Cache results
        if self.use_cache:
            with open(cache_file, "w") as f:
                json.dump([a.to_dict() for a in self.articles], f)
        
        return self.articles
    
    def build_citation_network(
        self,
        output_path: Optional[Union[str, Path]] = None,
        base_set_only: bool = True,
    ) -> Any:
        """
        Build citation network from retrieved articles.
        
        REQ-BIB-002: Construct citation network for EEG literature.
        
        Args:
            output_path: Path to save network as GML file
            base_set_only: If True, only include articles in base set
            
        Returns:
            NetworkX DiGraph representing citation network
            
        Raises:
            RuntimeError: If pyBiblioNet is not available
            ValueError: If no articles have been retrieved
            
        Example:
            >>> G = biblio.build_citation_network(output_path="eeg_citations.gml")
            >>> print(f"Network has {G.number_of_nodes()} nodes")
        """
        if not self._pybiblionet_available:
            raise RuntimeError(
                "pyBiblioNet is not installed. Install with: pip install pybiblionet"
            )
        
        if not self.articles:
            raise ValueError("No articles retrieved. Call search_eeg_literature first.")
        
        from pybiblionet.openalex.core import create_citation_graph
        
        # Convert articles back to raw format for pyBiblioNet
        raw_articles = [a.to_dict() for a in self.articles]
        
        # Create output path if not provided
        if not output_path:
            output_path = self.cache_dir / "eeg_citation_network.gml"
        else:
            output_path = Path(output_path)
        
        # Build citation graph
        self.citation_graph = create_citation_graph(
            articles=raw_articles,
            G_path=str(output_path),
            base_set=base_set_only,
        )
        
        logger.info(
            f"Built citation network with {self.citation_graph.number_of_nodes()} nodes "
            f"and {self.citation_graph.number_of_edges()} edges"
        )
        
        return self.citation_graph
    
    def build_coauthorship_network(
        self,
        output_path: Optional[Union[str, Path]] = None,
        base_set_only: bool = True,
    ) -> Any:
        """
        Build co-authorship network from retrieved articles.
        
        REQ-BIB-003: Construct co-authorship network for EEG researchers.
        
        Args:
            output_path: Path to save network as GML file
            base_set_only: If True, only include authors from base set
            
        Returns:
            NetworkX Graph representing co-authorship network
            
        Raises:
            RuntimeError: If pyBiblioNet is not available
            ValueError: If no articles have been retrieved
            
        Example:
            >>> G = biblio.build_coauthorship_network()
            >>> central = biblio.get_central_authors()
        """
        if not self._pybiblionet_available:
            raise RuntimeError(
                "pyBiblioNet is not installed. Install with: pip install pybiblionet"
            )
        
        if not self.articles:
            raise ValueError("No articles retrieved. Call search_eeg_literature first.")
        
        from pybiblionet.openalex.core import create_coauthorship_graph
        
        # Convert articles back to raw format for pyBiblioNet
        raw_articles = [a.to_dict() for a in self.articles]
        
        # Create output path if not provided
        if not output_path:
            output_path = self.cache_dir / "eeg_coauthorship_network.gml"
        else:
            output_path = Path(output_path)
        
        # Build co-authorship graph
        self.coauthorship_graph = create_coauthorship_graph(
            articles=raw_articles,
            G_path=str(output_path),
            base_set=base_set_only,
        )
        
        logger.info(
            f"Built co-authorship network with {self.coauthorship_graph.number_of_nodes()} nodes "
            f"and {self.coauthorship_graph.number_of_edges()} edges"
        )
        
        return self.coauthorship_graph
    
    def compute_citation_centrality(
        self,
        method: str = "pagerank",
    ) -> Dict[str, float]:
        """
        Compute centrality metrics for citation network.
        
        REQ-BIB-004: Compute influence metrics for EEG papers.
        
        Args:
            method: Centrality method ("pagerank", "betweenness", "eigenvector", "degree")
            
        Returns:
            Dictionary mapping article IDs to centrality scores
            
        Raises:
            ValueError: If citation network has not been built
            
        Example:
            >>> centrality = biblio.compute_citation_centrality(method="pagerank")
            >>> top_paper_id = max(centrality, key=centrality.get)
        """
        if self.citation_graph is None:
            raise ValueError("Citation network not built. Call build_citation_network first.")
        
        import networkx as nx
        
        # Compute centrality based on method
        if method == "pagerank":
            centrality = nx.pagerank(self.citation_graph)
        elif method == "betweenness":
            centrality = nx.betweenness_centrality(self.citation_graph)
        elif method == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(self.citation_graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge, using PageRank")
                centrality = nx.pagerank(self.citation_graph)
        elif method == "degree":
            centrality = dict(self.citation_graph.degree())
            # Normalize
            max_degree = max(centrality.values()) if centrality else 1
            centrality = {k: v / max_degree for k, v in centrality.items()}
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        # Update article centrality scores
        for article in self.articles:
            article.centrality_score = centrality.get(article.openalex_id, 0.0)
        
        logger.info(f"Computed {method} centrality for {len(centrality)} nodes")
        
        return centrality
    
    def compute_coauthorship_centrality(
        self,
        method: str = "betweenness",
    ) -> Dict[str, float]:
        """
        Compute centrality metrics for co-authorship network.
        
        REQ-BIB-004: Compute influence metrics for EEG researchers.
        
        Args:
            method: Centrality method ("betweenness", "degree", "closeness", "eigenvector")
            
        Returns:
            Dictionary mapping author IDs to centrality scores
            
        Raises:
            ValueError: If co-authorship network has not been built
        """
        if self.coauthorship_graph is None:
            raise ValueError("Co-authorship network not built. Call build_coauthorship_network first.")
        
        import networkx as nx
        
        # Compute centrality based on method
        if method == "betweenness":
            centrality = nx.betweenness_centrality(self.coauthorship_graph)
        elif method == "degree":
            centrality = dict(self.coauthorship_graph.degree())
            max_degree = max(centrality.values()) if centrality else 1
            centrality = {k: v / max_degree for k, v in centrality.items()}
        elif method == "closeness":
            centrality = nx.closeness_centrality(self.coauthorship_graph)
        elif method == "eigenvector":
            try:
                centrality = nx.eigenvector_centrality(self.coauthorship_graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge, using degree")
                centrality = dict(self.coauthorship_graph.degree())
                max_degree = max(centrality.values()) if centrality else 1
                centrality = {k: v / max_degree for k, v in centrality.items()}
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        logger.info(f"Computed {method} centrality for {len(centrality)} authors")
        
        return centrality
    
    def get_influential_papers(
        self,
        top_n: int = 10,
        method: str = "pagerank",
    ) -> List[EEGArticle]:
        """
        Get most influential EEG papers based on citation network.
        
        REQ-BIB-004: Identify influential EEG research papers.
        
        Args:
            top_n: Number of top papers to return
            method: Centrality method for ranking
            
        Returns:
            List of top EEGArticle objects sorted by influence
            
        Example:
            >>> top_papers = biblio.get_influential_papers(top_n=10)
            >>> for paper in top_papers:
            ...     print(f"{paper.title}: {paper.centrality_score:.4f}")
        """
        # Build network if not already built
        if self.citation_graph is None:
            self.build_citation_network()
        
        # Compute centrality
        self.compute_citation_centrality(method=method)
        
        # Sort articles by centrality
        sorted_articles = sorted(
            self.articles,
            key=lambda a: a.centrality_score,
            reverse=True,
        )
        
        return sorted_articles[:top_n]
    
    def get_influential_authors(
        self,
        top_n: int = 10,
        method: str = "betweenness",
    ) -> List[Tuple[str, float]]:
        """
        Get most influential EEG researchers based on co-authorship network.
        
        REQ-BIB-004: Identify influential EEG researchers.
        
        Args:
            top_n: Number of top authors to return
            method: Centrality method for ranking
            
        Returns:
            List of (author_name, centrality_score) tuples
        """
        # Build network if not already built
        if self.coauthorship_graph is None:
            self.build_coauthorship_network()
        
        # Compute centrality
        centrality = self.compute_coauthorship_centrality(method=method)
        
        # Sort by centrality
        sorted_authors = sorted(
            centrality.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_authors[:top_n]
    
    def get_network_metrics(
        self,
        network_type: str = "citation",
    ) -> NetworkMetrics:
        """
        Get network-level metrics for the specified network.
        
        REQ-BIB-004: Compute network structure metrics.
        
        Args:
            network_type: "citation" or "coauthorship"
            
        Returns:
            NetworkMetrics object with computed metrics
            
        Raises:
            ValueError: If specified network has not been built
        """
        import networkx as nx
        
        if network_type == "citation":
            if self.citation_graph is None:
                raise ValueError("Citation network not built")
            G = self.citation_graph
        elif network_type == "coauthorship":
            if self.coauthorship_graph is None:
                raise ValueError("Co-authorship network not built")
            G = self.coauthorship_graph
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Compute metrics
        metrics = NetworkMetrics(
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            density=nx.density(G),
        )
        
        # Compute clustering (different for directed/undirected)
        if G.is_directed():
            metrics.avg_clustering = nx.average_clustering(G.to_undirected())
        else:
            metrics.avg_clustering = nx.average_clustering(G)
        
        # Connected components
        if G.is_directed():
            metrics.num_components = nx.number_weakly_connected_components(G)
        else:
            metrics.num_components = nx.number_connected_components(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            metrics.avg_degree = sum(degrees) / len(degrees)
            metrics.max_degree = max(degrees)
        
        # Diameter (only for connected graphs)
        try:
            if G.is_directed():
                if nx.is_weakly_connected(G):
                    metrics.diameter = nx.diameter(G.to_undirected())
            else:
                if nx.is_connected(G):
                    metrics.diameter = nx.diameter(G)
        except (nx.NetworkXError, nx.NetworkXException):
            metrics.diameter = -1  # Indicate diameter couldn't be computed
        
        return metrics
    
    def detect_communities(
        self,
        network_type: str = "coauthorship",
        method: str = "louvain",
    ) -> Dict[str, int]:
        """
        Detect communities in the network.
        
        REQ-BIB-004: Community detection for research clusters.
        
        Args:
            network_type: "citation" or "coauthorship"
            method: Community detection method ("louvain", "label_propagation")
            
        Returns:
            Dictionary mapping node IDs to community labels
        """
        import networkx as nx
        
        if network_type == "citation":
            if self.citation_graph is None:
                raise ValueError("Citation network not built")
            G = self.citation_graph.to_undirected()
        elif network_type == "coauthorship":
            if self.coauthorship_graph is None:
                raise ValueError("Co-authorship network not built")
            G = self.coauthorship_graph
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        if method == "louvain":
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G)
                # Convert to node -> community mapping
                node_to_community = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_to_community[node] = i
                return node_to_community
            except ImportError:
                logger.warning("Louvain not available, using label propagation")
                method = "label_propagation"
        
        if method == "label_propagation":
            from networkx.algorithms.community import label_propagation_communities
            communities = list(label_propagation_communities(G))
            node_to_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = i
            return node_to_community
        
        raise ValueError(f"Unknown community detection method: {method}")
    
    def export_to_csv(
        self,
        output_dir: Union[str, Path],
        include_authors: bool = True,
        include_venues: bool = True,
    ) -> Dict[str, Path]:
        """
        Export bibliometric data to CSV files.
        
        REQ-BIB-001: Export functionality for further analysis.
        
        Args:
            output_dir: Directory for output CSV files
            include_authors: Whether to export author data
            include_venues: Whether to export venue data
            
        Returns:
            Dictionary mapping data type to output file path
        """
        if not self._pybiblionet_available:
            raise RuntimeError(
                "pyBiblioNet is not installed. Install with: pip install pybiblionet"
            )
        
        from pybiblionet.openalex.core import (
            export_articles_to_csv,
            export_authors_to_csv,
            export_venues_to_csv,
        )
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        # Export articles
        articles_path = output_dir / "eeg_articles.csv"
        export_articles_to_csv(
            json_data=[a.to_dict() for a in self.articles],
            fields=["openalex_id", "doi", "pmid", "title", "publication_date", "cited_by_count", "venue"],
            csv_path=str(articles_path),
        )
        outputs["articles"] = articles_path
        
        if include_authors:
            authors_path = output_dir / "eeg_authors.csv"
            export_authors_to_csv(
                json_data=[a.to_dict() for a in self.articles],
                fields=["name", "affiliations", "works_count", "cited_by_count"],
                csv_path=str(authors_path),
            )
            outputs["authors"] = authors_path
        
        if include_venues:
            venues_path = output_dir / "eeg_venues.csv"
            export_venues_to_csv(
                json_data=[a.to_dict() for a in self.articles],
                fields=["venue", "count"],
                csv_path=str(venues_path),
            )
            outputs["venues"] = venues_path
        
        logger.info(f"Exported bibliometric data to {output_dir}")
        
        return outputs
    
    def get_articles_for_rag(
        self,
        min_citations: int = 0,
        min_centrality: float = 0.0,
        topics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get articles formatted for RAG ingestion.
        
        REQ-BIB-005: Integration with RAG pipeline.
        
        Args:
            min_citations: Minimum citation count filter
            min_centrality: Minimum centrality score filter
            topics: List of topic keywords to filter by
            
        Returns:
            List of article dictionaries formatted for RAG ingestion
            
        Example:
            >>> rag_articles = biblio.get_articles_for_rag(
            ...     min_citations=10,
            ...     min_centrality=0.01
            ... )
            >>> for article in rag_articles:
            ...     rag_system.ingest(article)
        """
        filtered = []
        
        for article in self.articles:
            # Apply filters
            if article.cited_by_count < min_citations:
                continue
            if article.centrality_score < min_centrality:
                continue
            if topics:
                article_topics_lower = [t.lower() for t in article.topics]
                if not any(topic.lower() in article_topics_lower for topic in topics):
                    continue
            
            # Format for RAG
            rag_doc = {
                "id": article.openalex_id,
                "doi": article.doi,
                "pmid": article.pmid,
                "title": article.title,
                "content": article.abstract,
                "metadata": {
                    "source": "OpenAlex",
                    "authors": article.authors,
                    "publication_date": article.publication_date,
                    "venue": article.venue,
                    "cited_by_count": article.cited_by_count,
                    "centrality_score": article.centrality_score,
                    "topics": article.topics,
                },
            }
            filtered.append(rag_doc)
        
        logger.info(
            f"Prepared {len(filtered)} articles for RAG ingestion "
            f"(filtered from {len(self.articles)} total)"
        )
        
        return filtered


# Convenience functions for direct usage
def retrieve_eeg_articles(
    email: str,
    domain: EEGResearchDomain = EEGResearchDomain.GENERAL,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_results: int = 1000,
) -> List[EEGArticle]:
    """
    Convenience function to retrieve EEG articles.
    
    REQ-BIB-001: Simple interface for article retrieval.
    
    Args:
        email: Email for OpenAlex API
        domain: EEG research domain
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        max_results: Maximum articles to retrieve
        
    Returns:
        List of EEGArticle objects
    """
    biblio = EEGBiblioNet(email=email)
    return biblio.search_eeg_literature(
        domain=domain,
        from_date=from_date,
        to_date=to_date,
        max_results=max_results,
    )


def build_eeg_citation_network(
    articles: List[EEGArticle],
    output_path: Optional[str] = None,
) -> Any:
    """
    Convenience function to build citation network from articles.
    
    REQ-BIB-002: Simple interface for citation network construction.
    
    Args:
        articles: List of EEGArticle objects
        output_path: Optional path to save network
        
    Returns:
        NetworkX DiGraph
    """
    biblio = EEGBiblioNet(email="placeholder@example.com")
    biblio.articles = articles
    return biblio.build_citation_network(output_path=output_path)


def build_eeg_coauthorship_network(
    articles: List[EEGArticle],
    output_path: Optional[str] = None,
) -> Any:
    """
    Convenience function to build co-authorship network from articles.
    
    REQ-BIB-003: Simple interface for co-authorship network construction.
    
    Args:
        articles: List of EEGArticle objects
        output_path: Optional path to save network
        
    Returns:
        NetworkX Graph
    """
    biblio = EEGBiblioNet(email="placeholder@example.com")
    biblio.articles = articles
    return biblio.build_coauthorship_network(output_path=output_path)


def get_influential_papers(
    articles: List[EEGArticle],
    top_n: int = 10,
) -> List[EEGArticle]:
    """
    Convenience function to get influential papers from a list.
    
    REQ-BIB-004: Simple interface for influence analysis.
    
    Args:
        articles: List of EEGArticle objects
        top_n: Number of top papers to return
        
    Returns:
        List of top EEGArticle objects
    """
    biblio = EEGBiblioNet(email="placeholder@example.com")
    biblio.articles = articles
    return biblio.get_influential_papers(top_n=top_n)


def get_influential_authors(
    articles: List[EEGArticle],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Convenience function to get influential authors from a list.
    
    REQ-BIB-004: Simple interface for author influence analysis.
    
    Args:
        articles: List of EEGArticle objects
        top_n: Number of top authors to return
        
    Returns:
        List of (author_name, centrality_score) tuples
    """
    biblio = EEGBiblioNet(email="placeholder@example.com")
    biblio.articles = articles
    return biblio.get_influential_authors(top_n=top_n)
