#!/usr/bin/env python3
"""
RAG Pipeline Integration for Bibliometrics

Integrates bibliometric visualizations and analysis into RAG search results.
Provides visual research context alongside LLM-generated responses.

Requirements:
- REQ-BIB-020: Integration with RAG query results
- REQ-BIB-021: Visualization generation for relevant queries
- REQ-BIB-022: NLP enhancement for query expansion
- REQ-BIB-023: Research metrics display

Usage:
    from eeg_rag.bibliometrics.rag_integration import BibliometricEnhancer
    
    enhancer = BibliometricEnhancer()
    enhanced_result = await enhancer.enhance_rag_result(query, rag_result)
"""

import asyncio
import logging
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .eeg_biblionet import EEGBiblioNet, EEGArticle
from .visualization import EEGVisualization, ChartResult
from .nlp_enhancement import EEGNLPEnhancer, ExtractedKeywords
from .research_export import EEGResearchExporter

logger = logging.getLogger(__name__)


@dataclass
class BibliometricInsight:
    """A single bibliometric insight to include in RAG results."""
    insight_type: str  # "trend", "author", "citation", "keyword", "venue"
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    chart_base64: Optional[str] = None  # Base64 encoded chart image
    relevance_score: float = 1.0


@dataclass
class EnhancedRAGResult:
    """RAG result enhanced with bibliometric data and visualizations."""
    original_response: str
    query: str
    insights: List[BibliometricInsight] = field(default_factory=list)
    related_articles: List[Dict[str, Any]] = field(default_factory=list)
    expanded_keywords: List[str] = field(default_factory=list)
    research_summary: Optional[Dict[str, Any]] = None
    charts: Dict[str, str] = field(default_factory=dict)  # name -> base64
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_response": self.original_response,
            "query": self.query,
            "insights": [
                {
                    "type": i.insight_type,
                    "title": i.title,
                    "description": i.description,
                    "data": i.data,
                    "has_chart": i.chart_base64 is not None,
                    "relevance": i.relevance_score,
                }
                for i in self.insights
            ],
            "related_articles": self.related_articles,
            "expanded_keywords": self.expanded_keywords,
            "research_summary": self.research_summary,
            "charts_available": list(self.charts.keys()),
            "processing_time_ms": self.processing_time_ms,
        }


class BibliometricEnhancer:
    """
    Enhances RAG results with bibliometric insights and visualizations.
    
    This class integrates the bibliometrics module into the RAG pipeline,
    providing visual research context for EEG-related queries.
    
    Example:
        >>> enhancer = BibliometricEnhancer()
        >>> result = await enhancer.enhance_rag_result(
        ...     query="seizure detection methods",
        ...     rag_response="Seizure detection can use various methods...",
        ...     cited_articles=[article1, article2]
        ... )
        >>> print(result.insights[0].title)
        'Publication Trends in Seizure Detection'
    """
    
    def __init__(
        self,
        enable_visualizations: bool = True,
        enable_nlp: bool = True,
        enable_metrics: bool = True,
        auto_fetch_articles: bool = False,
        max_articles: int = 50,
        chart_style: str = "seaborn-v0_8-whitegrid"
    ):
        """
        Initialize the bibliometric enhancer.
        
        Args:
            enable_visualizations: Generate visualization charts
            enable_nlp: Use NLP for keyword extraction and expansion
            enable_metrics: Compute research metrics
            auto_fetch_articles: Fetch articles from OpenAlex if needed
            max_articles: Maximum articles for analysis
            chart_style: Matplotlib style for charts
        """
        self.enable_visualizations = enable_visualizations
        self.enable_nlp = enable_nlp
        self.enable_metrics = enable_metrics
        self.auto_fetch_articles = auto_fetch_articles
        self.max_articles = max_articles
        
        # Initialize components lazily
        self._viz: Optional[EEGVisualization] = None
        self._nlp: Optional[EEGNLPEnhancer] = None
        self._exporter: Optional[EEGResearchExporter] = None
        self._biblionet: Optional[EEGBiblioNet] = None
        self._chart_style = chart_style
        
        logger.info("BibliometricEnhancer initialized")
    
    @property
    def viz(self) -> EEGVisualization:
        """Lazy-load visualization component."""
        if self._viz is None:
            self._viz = EEGVisualization(style=self._chart_style)
        return self._viz
    
    @property
    def nlp(self) -> EEGNLPEnhancer:
        """Lazy-load NLP component."""
        if self._nlp is None:
            self._nlp = EEGNLPEnhancer()
        return self._nlp
    
    @property
    def exporter(self) -> EEGResearchExporter:
        """Lazy-load exporter component."""
        if self._exporter is None:
            self._exporter = EEGResearchExporter()
        return self._exporter
    
    @property
    def biblionet(self) -> EEGBiblioNet:
        """Lazy-load biblionet component."""
        if self._biblionet is None:
            self._biblionet = EEGBiblioNet()
        return self._biblionet
    
    async def enhance_rag_result(
        self,
        query: str,
        rag_response: str,
        cited_articles: Optional[List[EEGArticle]] = None,
        include_trends: bool = True,
        include_author_analysis: bool = True,
        include_keywords: bool = True,
        include_citations: bool = True
    ) -> EnhancedRAGResult:
        """
        Enhance a RAG result with bibliometric insights.
        
        Args:
            query: The original search query
            rag_response: The LLM-generated response
            cited_articles: Articles cited in the response (optional)
            include_trends: Include publication trends visualization
            include_author_analysis: Include top authors chart
            include_keywords: Include keyword analysis
            include_citations: Include citation distribution
            
        Returns:
            EnhancedRAGResult with visualizations and insights
        """
        import time
        start_time = time.time()
        
        insights = []
        charts = {}
        expanded_keywords = []
        research_summary = None
        related_articles = []
        
        # Get articles if not provided
        articles = cited_articles or []
        if not articles and self.auto_fetch_articles:
            try:
                articles = await self._fetch_relevant_articles(query)
            except Exception as e:
                logger.warning(f"Failed to fetch articles: {e}")
        
        # NLP Enhancement
        if self.enable_nlp:
            try:
                expanded = self.nlp.expand_query(query)
                expanded_keywords = expanded[:10] if expanded else []
                
                # Extract keywords from response
                if rag_response:
                    keywords = self.nlp.extract_keywords_from_text(
                        rag_response, top_n=5
                    )
                    insights.append(BibliometricInsight(
                        insight_type="keyword",
                        title="Key Concepts in Response",
                        description="Most relevant concepts extracted from the response",
                        data={"keywords": [kw for kw, score in keywords.keywords]}
                    ))
            except Exception as e:
                logger.warning(f"NLP enhancement failed: {e}")
        
        # Generate visualizations if we have articles
        if articles and self.enable_visualizations:
            # Publication Trends
            if include_trends:
                try:
                    trends_insight, trends_chart = self._generate_trends_insight(
                        articles, query
                    )
                    if trends_insight:
                        insights.append(trends_insight)
                        charts["publication_trends"] = trends_chart
                except Exception as e:
                    logger.warning(f"Trends visualization failed: {e}")
            
            # Author Analysis
            if include_author_analysis:
                try:
                    author_insight, author_chart = self._generate_author_insight(
                        articles
                    )
                    if author_insight:
                        insights.append(author_insight)
                        charts["top_authors"] = author_chart
                except Exception as e:
                    logger.warning(f"Author analysis failed: {e}")
            
            # Citation Distribution
            if include_citations:
                try:
                    citation_insight, citation_chart = self._generate_citation_insight(
                        articles
                    )
                    if citation_insight:
                        insights.append(citation_insight)
                        charts["citations"] = citation_chart
                except Exception as e:
                    logger.warning(f"Citation analysis failed: {e}")
        
        # Compute research metrics
        if articles and self.enable_metrics:
            try:
                research_summary = self._compute_research_summary(articles)
                
                # Add summary insight
                insights.append(BibliometricInsight(
                    insight_type="summary",
                    title="Research Landscape Summary",
                    description="Overview of the research area based on cited literature",
                    data=research_summary
                ))
            except Exception as e:
                logger.warning(f"Metrics computation failed: {e}")
        
        # Format related articles
        if articles:
            related_articles = [
                {
                    "title": art.title,
                    "authors": art.authors[:3],
                    "year": art.year,
                    "venue": art.venue,
                    "citations": art.citation_count,
                    "doi": art.doi,
                }
                for art in articles[:10]
            ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return EnhancedRAGResult(
            original_response=rag_response,
            query=query,
            insights=insights,
            related_articles=related_articles,
            expanded_keywords=expanded_keywords,
            research_summary=research_summary,
            charts=charts,
            processing_time_ms=processing_time
        )
    
    def _generate_trends_insight(
        self,
        articles: List[EEGArticle],
        query: str
    ) -> Tuple[Optional[BibliometricInsight], Optional[str]]:
        """Generate publication trends insight."""
        if len(articles) < 3:
            return None, None
        
        chart_result = self.viz.plot_publication_trends(
            articles,
            title=f"Publication Trends: {query[:30]}..."
        )
        
        # Convert to base64
        chart_base64 = self._chart_to_base64(chart_result)
        
        # Compute trend direction
        years = [art.year for art in articles if art.year]
        if years:
            recent_count = sum(1 for y in years if y >= max(years) - 2)
            older_count = len(years) - recent_count
            trend = "increasing" if recent_count > older_count else "stable"
        else:
            trend = "unknown"
        
        insight = BibliometricInsight(
            insight_type="trend",
            title=f"Publication Trends in {query[:30]}...",
            description=f"Research activity appears to be {trend} in this area.",
            data={
                "total_articles": len(articles),
                "year_range": (min(years), max(years)) if years else None,
                "trend_direction": trend,
            },
            chart_base64=chart_base64,
            relevance_score=0.9
        )
        
        return insight, chart_base64
    
    def _generate_author_insight(
        self,
        articles: List[EEGArticle]
    ) -> Tuple[Optional[BibliometricInsight], Optional[str]]:
        """Generate top authors insight."""
        if len(articles) < 2:
            return None, None
        
        chart_result = self.viz.plot_top_authors(
            articles,
            top_n=5,
            title="Leading Researchers in This Area"
        )
        
        chart_base64 = self._chart_to_base64(chart_result)
        
        # Get productivity metrics
        productivity = self.exporter.compute_author_productivity(articles)
        top_authors = list(productivity.items())[:5]
        
        insight = BibliometricInsight(
            insight_type="author",
            title="Leading Researchers",
            description="Most prolific authors in the relevant literature.",
            data={
                "top_authors": [
                    {"name": name, "publications": prod.total_publications}
                    for name, prod in top_authors
                ],
                "total_unique_authors": len(productivity),
            },
            chart_base64=chart_base64,
            relevance_score=0.8
        )
        
        return insight, chart_base64
    
    def _generate_citation_insight(
        self,
        articles: List[EEGArticle]
    ) -> Tuple[Optional[BibliometricInsight], Optional[str]]:
        """Generate citation distribution insight."""
        if len(articles) < 3:
            return None, None
        
        chart_result = self.viz.plot_citation_distribution(
            articles,
            title="Citation Impact Distribution"
        )
        
        chart_base64 = self._chart_to_base64(chart_result)
        
        citations = [art.citation_count for art in articles]
        
        insight = BibliometricInsight(
            insight_type="citation",
            title="Citation Impact",
            description="Distribution of citation counts in the relevant literature.",
            data={
                "total_citations": sum(citations),
                "avg_citations": sum(citations) / len(citations),
                "max_citations": max(citations),
                "min_citations": min(citations),
            },
            chart_base64=chart_base64,
            relevance_score=0.85
        )
        
        return insight, chart_base64
    
    def _compute_research_summary(
        self,
        articles: List[EEGArticle]
    ) -> Dict[str, Any]:
        """Compute summary statistics for research area."""
        venue_metrics = self.exporter.compute_venue_metrics(articles)
        author_productivity = self.exporter.compute_author_productivity(articles)
        
        years = [art.year for art in articles if art.year]
        citations = [art.citation_count for art in articles]
        
        return {
            "total_articles": len(articles),
            "year_range": {
                "start": min(years) if years else None,
                "end": max(years) if years else None,
            },
            "citation_stats": {
                "total": sum(citations),
                "average": sum(citations) / len(citations) if citations else 0,
                "max": max(citations) if citations else 0,
            },
            "venues": {
                "unique_count": len(venue_metrics),
                "top_venues": [
                    {"name": name, "articles": m.total_articles}
                    for name, m in list(venue_metrics.items())[:3]
                ],
            },
            "authors": {
                "unique_count": len(author_productivity),
                "top_authors": [
                    {"name": name, "h_index": p.h_index}
                    for name, p in list(author_productivity.items())[:3]
                ],
            },
        }
    
    def _chart_to_base64(self, chart_result: ChartResult) -> Optional[str]:
        """Convert chart to base64 string for embedding."""
        if chart_result.figure is None:
            return None
        
        try:
            buf = BytesIO()
            chart_result.figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            import matplotlib.pyplot as plt
            plt.close(chart_result.figure)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to convert chart to base64: {e}")
            return None
    
    async def _fetch_relevant_articles(
        self,
        query: str,
        max_results: int = 20
    ) -> List[EEGArticle]:
        """Fetch relevant articles from OpenAlex."""
        try:
            # Expand query for better coverage
            expanded = self.nlp.expand_query(query)
            search_terms = [query] + expanded[:3]
            
            all_articles = []
            for term in search_terms[:2]:  # Limit API calls
                articles = await asyncio.to_thread(
                    self.biblionet.search_articles,
                    term,
                    max_results=max_results // 2
                )
                all_articles.extend(articles)
            
            # Deduplicate by ID
            seen = set()
            unique = []
            for art in all_articles:
                if art.openalex_id not in seen:
                    seen.add(art.openalex_id)
                    unique.append(art)
            
            return unique[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            return []
    
    def should_enhance(self, query: str) -> bool:
        """
        Determine if a query would benefit from bibliometric enhancement.
        
        Args:
            query: The search query
            
        Returns:
            True if the query is research-oriented and would benefit from
            bibliometric context
        """
        # Research-oriented keywords
        research_indicators = [
            "research", "study", "paper", "literature", "review",
            "published", "citation", "author", "trends", "methods",
            "analysis", "comparison", "state of the art", "recent",
            "advances", "development", "approach", "technique"
        ]
        
        query_lower = query.lower()
        
        # Check for research indicators
        has_research_intent = any(
            indicator in query_lower for indicator in research_indicators
        )
        
        # Check minimum query length
        is_substantial = len(query.split()) >= 3
        
        return has_research_intent or is_substantial
