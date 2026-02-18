"""
EEG Bibliometrics Visualization Module

Provides publication trends, topic evolution, author productivity,
and keyword analysis charts for EEG research visualization.

Requirements:
- REQ-VIS-001: Publication trends over time
- REQ-VIS-002: Topic/domain evolution charts
- REQ-VIS-003: Author productivity analysis
- REQ-VIS-004: Keyword trend visualization
- REQ-VIS-005: Network visualization
"""

from __future__ import annotations

import base64
import io
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ChartResult:
    """
    Container for chart output with multiple formats.
    
    Supports embedding in HTML, saving to file, and display in notebooks.
    """
    
    title: str
    chart_type: str
    png_base64: Optional[str] = None
    svg_data: Optional[str] = None
    html_data: Optional[str] = None
    figure: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_html_img(self) -> str:
        """Return HTML img tag for embedding."""
        if self.png_base64:
            return f'<img src="data:image/png;base64,{self.png_base64}" alt="{self.title}" />'
        return ""
    
    def save(self, path: Union[str, Path], format: str = "png") -> None:
        """Save chart to file."""
        path = Path(path)
        if self.figure is not None:
            self.figure.savefig(path, format=format, dpi=150, bbox_inches='tight')
        elif self.png_base64 and format == "png":
            with open(path, "wb") as f:
                f.write(base64.b64decode(self.png_base64))


class EEGVisualization:
    """
    Visualization engine for EEG bibliometric data.
    
    Generates publication trends, topic evolution, author charts,
    and keyword analysis visualizations.
    
    Example:
        >>> viz = EEGVisualization()
        >>> chart = viz.plot_publication_trends(articles, interval="year")
        >>> chart.save("trends.png")
    """
    
    # EEG-specific color palette
    EEG_COLORS = [
        "#1f77b4",  # Blue - EEG signal color
        "#ff7f0e",  # Orange - Alert/abnormal
        "#2ca02c",  # Green - Normal
        "#d62728",  # Red - Seizure/critical
        "#9467bd",  # Purple - Sleep
        "#8c564b",  # Brown - Artifact
        "#e377c2",  # Pink - Cognitive
        "#7f7f7f",  # Gray - Reference
        "#bcbd22",  # Yellow-green - Motor
        "#17becf",  # Cyan - BCI
    ]
    
    def __init__(self, style: str = "seaborn-v0_8-whitegrid") -> None:
        """
        Initialize visualization engine.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        self._plt = None
        self._np = None
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """Check if visualization dependencies are available."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for server/web apps
            import matplotlib.pyplot as plt
            import numpy as np
            self._plt = plt
            self._np = np
            return True
        except ImportError as e:
            logger.warning(f"matplotlib not available for visualization: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up matplotlib: {e}")
            return False
    
    def _figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse publication date string."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            try:
                return datetime.strptime(date_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                return None
    
    def plot_publication_trends(
        self,
        articles: List[Any],
        interval: str = "year",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        color_root: str = "#1f77b4",
        color_base: str = "#a6cee3",
        show_cumulative: bool = False,
    ) -> ChartResult:
        """
        Plot publication trends over time.
        
        REQ-VIS-001: Visualize publication volume trends.
        
        Args:
            articles: List of EEGArticle objects or dicts
            interval: Time interval ("month", "quarter", "year")
            date_from: Start date filter
            date_to: End date filter
            color_root: Color for root set articles
            color_base: Color for base set articles
            show_cumulative: Whether to show cumulative line
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        np = self._np
        
        # Aggregate by interval
        counts = defaultdict(int)
        for article in articles:
            if hasattr(article, 'publication_date'):
                date_str = article.publication_date
            elif isinstance(article, dict):
                date_str = article.get('publication_date')
            else:
                continue
            
            pub_date = self._parse_date(date_str)
            if not pub_date:
                continue
            
            if date_from and pub_date < date_from:
                continue
            if date_to and pub_date > date_to:
                continue
            
            # Create time key based on interval
            if interval == "month":
                key = pub_date.strftime("%Y-%m")
            elif interval == "quarter":
                quarter = (pub_date.month - 1) // 3 + 1
                key = f"{pub_date.year}-Q{quarter}"
            else:  # year
                key = str(pub_date.year)
            
            counts[key] += 1
        
        if not counts:
            # Return empty chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            result = ChartResult(
                title="Publication Trends",
                chart_type="area",
                png_base64=self._figure_to_base64(fig),
                figure=fig,
            )
            plt.close(fig)
            return result
        
        # Sort by date
        sorted_keys = sorted(counts.keys())
        values = [counts[k] for k in sorted_keys]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(sorted_keys))
        ax.fill_between(x, 0, values, alpha=0.7, color=color_root, label="Publications")
        ax.plot(x, values, color=color_root, linewidth=2)
        
        if show_cumulative:
            cumulative = np.cumsum(values)
            ax2 = ax.twinx()
            ax2.plot(x, cumulative, color=color_base, linewidth=2, linestyle='--', label="Cumulative")
            ax2.set_ylabel("Cumulative Publications", fontsize=12)
            ax2.legend(loc='upper left')
        
        # Style
        ax.set_title(f"EEG Publication Trends by {interval.title()}", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"Publication {interval.title()}", fontsize=12)
        ax.set_ylabel("Number of Publications", fontsize=12)
        
        # X-axis ticks
        tick_step = max(1, len(sorted_keys) // 10)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(sorted_keys[::tick_step], rotation=45, ha='right')
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        result = ChartResult(
            title="EEG Publication Trends",
            chart_type="area",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"interval": interval, "total_articles": sum(values)},
        )
        plt.close(fig)
        return result
    
    def plot_topic_evolution(
        self,
        articles: List[Any],
        field_key: str = "domain",
        interval: str = "year",
        top_n: int = 8,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> ChartResult:
        """
        Plot topic/domain evolution over time.
        
        REQ-VIS-002: Visualize research domain evolution.
        
        Args:
            articles: List of EEGArticle objects or dicts
            field_key: Field to use ("domain" or "field")
            interval: Time interval
            top_n: Number of top topics to show
            date_from: Start date filter
            date_to: End date filter
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        np = self._np
        
        # Extract topics by time period
        topic_counts = defaultdict(lambda: defaultdict(int))
        overall_topics = Counter()
        
        for article in articles:
            # Get publication date
            if hasattr(article, 'publication_date'):
                date_str = article.publication_date
                topics = getattr(article, 'topics', [])
            elif isinstance(article, dict):
                date_str = article.get('publication_date')
                topics = article.get('topics', [])
            else:
                continue
            
            pub_date = self._parse_date(date_str)
            if not pub_date:
                continue
            
            if date_from and pub_date < date_from:
                continue
            if date_to and pub_date > date_to:
                continue
            
            # Time key
            if interval == "month":
                key = pub_date.strftime("%Y-%m")
            elif interval == "quarter":
                quarter = (pub_date.month - 1) // 3 + 1
                key = f"{pub_date.year}-Q{quarter}"
            else:
                key = str(pub_date.year)
            
            # Count topics (use first topic as primary)
            if topics:
                primary_topic = topics[0] if isinstance(topics[0], str) else str(topics[0])
                topic_counts[key][primary_topic] += 1
                overall_topics[primary_topic] += 1
        
        if not topic_counts:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'No topic data available', ha='center', va='center', fontsize=14)
            result = ChartResult(title="Topic Evolution", chart_type="bar", figure=fig)
            plt.close(fig)
            return result
        
        # Get top N topics
        top_topics = [t for t, _ in overall_topics.most_common(top_n)]
        sorted_keys = sorted(topic_counts.keys())
        
        # Create stacked bar data
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(sorted_keys))
        width = 0.8
        bottom = np.zeros(len(sorted_keys))
        
        for i, topic in enumerate(top_topics):
            values = [topic_counts[k].get(topic, 0) for k in sorted_keys]
            ax.bar(x, values, width, label=topic[:30], bottom=bottom, 
                   color=self.EEG_COLORS[i % len(self.EEG_COLORS)], alpha=0.85)
            bottom += np.array(values)
        
        # Style
        ax.set_title(f"Top {top_n} EEG Research Topics by {interval.title()}", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"Publication {interval.title()}", fontsize=12)
        ax.set_ylabel("Number of Articles", fontsize=12)
        
        tick_step = max(1, len(sorted_keys) // 10)
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels(sorted_keys[::tick_step], rotation=45, ha='right')
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        result = ChartResult(
            title="EEG Topic Evolution",
            chart_type="stacked_bar",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"top_topics": top_topics, "interval": interval},
        )
        plt.close(fig)
        return result
    
    def plot_top_authors(
        self,
        articles: List[Any],
        top_n: int = 10,
        by_citations: bool = False,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> ChartResult:
        """
        Plot top authors by publications or citations.
        
        REQ-VIS-003: Visualize author productivity and impact.
        
        Args:
            articles: List of EEGArticle objects or dicts
            top_n: Number of top authors to display
            by_citations: If True, rank by citations; else by publication count
            date_from: Start date filter
            date_to: End date filter
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        
        # Count author publications/citations
        author_counts = Counter()
        author_citations = Counter()
        
        for article in articles:
            if hasattr(article, 'publication_date'):
                date_str = article.publication_date
                authors = getattr(article, 'authors', [])
                citations = getattr(article, 'cited_by_count', 0)
            elif isinstance(article, dict):
                date_str = article.get('publication_date')
                authors = article.get('authors', [])
                citations = article.get('cited_by_count', 0)
            else:
                continue
            
            pub_date = self._parse_date(date_str)
            if pub_date:
                if date_from and pub_date < date_from:
                    continue
                if date_to and pub_date > date_to:
                    continue
            
            for author in authors:
                author_counts[author] += 1
                author_citations[author] += citations
        
        # Get top authors
        if by_citations:
            top_authors = author_citations.most_common(top_n)
            metric = "Citations"
        else:
            top_authors = author_counts.most_common(top_n)
            metric = "Publications"
        
        if not top_authors:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No author data available', ha='center', va='center')
            result = ChartResult(title="Top Authors", chart_type="bar", figure=fig)
            plt.close(fig)
            return result
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))
        
        names = [a[:40] for a, _ in reversed(top_authors)]  # Truncate long names
        values = [v for _, v in reversed(top_authors)]
        
        y_pos = range(len(names))
        bars = ax.barh(y_pos, values, color=self.EEG_COLORS[:len(names)], alpha=0.85)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:,}', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"Top {top_n} EEG Researchers by {metric}", fontsize=16, fontweight='bold')
        
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        result = ChartResult(
            title=f"Top EEG Researchers by {metric}",
            chart_type="horizontal_bar",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"top_authors": [a for a, _ in top_authors], "metric": metric},
        )
        plt.close(fig)
        return result
    
    def plot_citation_distribution(
        self,
        articles: List[Any],
        bins: int = 30,
        log_scale: bool = True,
    ) -> ChartResult:
        """
        Plot citation count distribution.
        
        Args:
            articles: List of EEGArticle objects or dicts
            bins: Number of histogram bins
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        np = self._np
        
        # Extract citation counts
        citations = []
        for article in articles:
            if hasattr(article, 'cited_by_count'):
                citations.append(article.cited_by_count)
            elif isinstance(article, dict):
                citations.append(article.get('cited_by_count', 0))
        
        if not citations:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No citation data', ha='center', va='center')
            result = ChartResult(title="Citation Distribution", chart_type="histogram", figure=fig)
            plt.close(fig)
            return result
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter out extreme outliers for better visualization
        citations = np.array(citations)
        p99 = np.percentile(citations, 99)
        plot_citations = citations[citations <= p99]
        
        ax.hist(plot_citations, bins=bins, color=self.EEG_COLORS[0], alpha=0.7, edgecolor='white')
        
        if log_scale:
            ax.set_yscale('log')
        
        # Add statistics
        mean_cit = np.mean(citations)
        median_cit = np.median(citations)
        ax.axvline(mean_cit, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cit:.1f}')
        ax.axvline(median_cit, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cit:.1f}')
        
        ax.set_title("EEG Paper Citation Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Citation Count", fontsize=12)
        ax.set_ylabel("Number of Papers" + (" (log scale)" if log_scale else ""), fontsize=12)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        result = ChartResult(
            title="Citation Distribution",
            chart_type="histogram",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"mean": float(mean_cit), "median": float(median_cit), "total": len(citations)},
        )
        plt.close(fig)
        return result
    
    def plot_venue_distribution(
        self,
        articles: List[Any],
        top_n: int = 15,
    ) -> ChartResult:
        """
        Plot publication venue distribution.
        
        Args:
            articles: List of EEGArticle objects or dicts
            top_n: Number of top venues to display
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        
        # Count venues
        venue_counts = Counter()
        for article in articles:
            if hasattr(article, 'venue'):
                venue = article.venue
            elif isinstance(article, dict):
                venue = article.get('venue', '')
            else:
                continue
            
            if venue:
                venue_counts[venue] += 1
        
        top_venues = venue_counts.most_common(top_n)
        
        if not top_venues:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No venue data', ha='center', va='center')
            result = ChartResult(title="Venue Distribution", chart_type="bar", figure=fig)
            plt.close(fig)
            return result
        
        fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.5)))
        
        names = [v[:50] for v, _ in reversed(top_venues)]
        values = [c for _, c in reversed(top_venues)]
        
        y_pos = range(len(names))
        colors = [self.EEG_COLORS[i % len(self.EEG_COLORS)] for i in range(len(names))]
        ax.barh(y_pos, values, color=colors, alpha=0.85)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Number of Publications", fontsize=12)
        ax.set_title(f"Top {top_n} EEG Publication Venues", fontsize=16, fontweight='bold')
        
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        result = ChartResult(
            title="EEG Publication Venues",
            chart_type="horizontal_bar",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"top_venues": [v for v, _ in top_venues]},
        )
        plt.close(fig)
        return result
    
    def plot_network_graph(
        self,
        graph: Any,
        node_size_attr: Optional[str] = None,
        node_color_attr: Optional[str] = None,
        max_nodes: int = 100,
        layout: str = "spring",
        title: str = "Network Graph",
    ) -> ChartResult:
        """
        Plot network graph visualization.
        
        REQ-VIS-005: Network structure visualization.
        
        Args:
            graph: NetworkX graph object
            node_size_attr: Node attribute for sizing
            node_color_attr: Node attribute for coloring
            max_nodes: Maximum nodes to display
            layout: Layout algorithm ("spring", "kamada_kawai", "circular")
            title: Chart title
            
        Returns:
            ChartResult with the generated chart
        """
        if self._plt is None or self._np is None:
            raise RuntimeError("Matplotlib and numpy are required for visualization. Please install: pip install matplotlib numpy")
        
        plt = self._plt
        np = self._np
        
        try:
            import networkx as nx
        except ImportError:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'NetworkX not available', ha='center', va='center')
            result = ChartResult(title=title, chart_type="network", figure=fig)
            plt.close(fig)
            return result
        
        # Sample if too large
        if graph.number_of_nodes() > max_nodes:
            # Get most connected nodes
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            graph = graph.subgraph(top_nodes).copy()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Compute layout
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        else:  # spring
            pos = nx.spring_layout(graph, k=2/np.sqrt(graph.number_of_nodes()), iterations=50)
        
        # Node sizes based on degree
        degrees = dict(graph.degree())
        node_sizes = [50 + degrees[n] * 20 for n in graph.nodes()]
        
        # Node colors (community-based if available)
        node_colors = [self.EEG_COLORS[hash(str(n)) % len(self.EEG_COLORS)] for n in graph.nodes()]
        
        # Draw
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)
        
        # Labels for top nodes
        if graph.number_of_nodes() <= 30:
            labels = {n: str(n)[:20] for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        result = ChartResult(
            title=title,
            chart_type="network",
            png_base64=self._figure_to_base64(fig),
            figure=fig,
            metadata={"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        )
        plt.close(fig)
        return result
    
    def create_research_dashboard(
        self,
        articles: List[Any],
        graph: Optional[Any] = None,
    ) -> Dict[str, ChartResult]:
        """
        Create a comprehensive research dashboard with multiple charts.
        
        Args:
            articles: List of EEGArticle objects
            graph: Optional citation or co-authorship graph
            
        Returns:
            Dictionary of chart name to ChartResult
        """
        dashboard = {}
        
        # Publication trends
        dashboard["trends"] = self.plot_publication_trends(articles, interval="year")
        
        # Topic evolution
        dashboard["topics"] = self.plot_topic_evolution(articles, interval="year", top_n=6)
        
        # Top authors by publications
        dashboard["authors"] = self.plot_top_authors(articles, top_n=10)
        
        # Top authors by citations
        dashboard["authors_citations"] = self.plot_top_authors(articles, top_n=10, by_citations=True)
        
        # Citation distribution
        dashboard["citations"] = self.plot_citation_distribution(articles)
        
        # Venue distribution
        dashboard["venues"] = self.plot_venue_distribution(articles, top_n=12)
        
        # Network if provided
        if graph is not None:
            dashboard["network"] = self.plot_network_graph(
                graph, max_nodes=80, title="Research Network"
            )
        
        return dashboard
