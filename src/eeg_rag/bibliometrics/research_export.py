"""
EEG Bibliometrics Research Export Module

Provides Scopus-compatible CSV exports, institutional collaboration analysis,
venue quality metrics, and author productivity tracking for EEG research.

Requirements:
- REQ-EXP-001: Scopus-compatible CSV exports
- REQ-EXP-002: Institutional collaboration network analysis
- REQ-EXP-003: Venue/journal quality metrics
- REQ-EXP-004: Author productivity and h-index estimation
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class VenueMetrics:
    """
    Quality metrics for a publication venue (journal/conference).
    
    Attributes:
        name: Venue name
        article_count: Number of articles published
        total_citations: Total citations across all articles
        mean_citations: Average citations per article
        h_index: Estimated venue h-index
        top_authors: Most published authors in venue
    """
    name: str
    article_count: int = 0
    total_citations: int = 0
    mean_citations: float = 0.0
    h_index: int = 0
    top_authors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "article_count": self.article_count,
            "total_citations": self.total_citations,
            "mean_citations": self.mean_citations,
            "h_index": self.h_index,
            "top_authors": self.top_authors,
        }


@dataclass
class InstitutionMetrics:
    """
    Metrics for a research institution.
    
    Attributes:
        name: Institution name
        country: Country code
        article_count: Number of articles
        author_count: Number of unique authors
        total_citations: Total citations
        collaborators: Top collaborating institutions
    """
    name: str
    country: str = ""
    article_count: int = 0
    author_count: int = 0
    total_citations: int = 0
    collaborators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "country": self.country,
            "article_count": self.article_count,
            "author_count": self.author_count,
            "total_citations": self.total_citations,
            "collaborators": self.collaborators,
        }


@dataclass
class AuthorProductivity:
    """
    Author productivity metrics.
    
    Attributes:
        name: Author name
        openalex_id: OpenAlex author ID
        article_count: Number of publications
        total_citations: Total citations received
        h_index: Estimated h-index
        first_author_count: Papers as first author
        last_author_count: Papers as last/senior author
        coauthor_count: Number of unique coauthors
        venues: Top publication venues
        years_active: Range of publication years
    """
    name: str
    openalex_id: str = ""
    article_count: int = 0
    total_citations: int = 0
    h_index: int = 0
    first_author_count: int = 0
    last_author_count: int = 0
    coauthor_count: int = 0
    venues: List[str] = field(default_factory=list)
    years_active: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "openalex_id": self.openalex_id,
            "article_count": self.article_count,
            "total_citations": self.total_citations,
            "h_index": self.h_index,
            "first_author_count": self.first_author_count,
            "last_author_count": self.last_author_count,
            "coauthor_count": self.coauthor_count,
            "venues": self.venues,
            "years_active": self.years_active,
        }


class EEGResearchExporter:
    """
    Research export engine for EEG bibliometric data.
    
    Provides CSV exports, institutional analysis, venue metrics,
    and author productivity tracking.
    
    Example:
        >>> exporter = EEGResearchExporter(articles)
        >>> exporter.export_to_scopus_csv("output/eeg_articles.csv")
        >>> venue_metrics = exporter.compute_venue_metrics()
    """
    
    # Scopus CSV field mapping
    SCOPUS_FIELDS = [
        "Authors", "Author full names", "Author(s) ID", "Title", "Year",
        "Source title", "Volume", "Issue", "Art. No.", "Page start", "Page end",
        "Page count", "Cited by", "DOI", "Link", "Affiliations",
        "Authors with affiliations", "Abstract", "Author Keywords",
        "Index Keywords", "Document Type", "Publication Stage", "Open Access",
        "Source", "EID",
    ]
    
    def __init__(self, articles: List[Any]) -> None:
        """
        Initialize research exporter.
        
        Args:
            articles: List of EEGArticle objects or article dicts
        """
        self.articles = articles
        self._processed_articles: List[Dict[str, Any]] = []
        self._process_articles()
    
    def _process_articles(self) -> None:
        """Process articles into standard dict format."""
        self._processed_articles = []
        
        for article in self.articles:
            if hasattr(article, 'to_dict'):
                data = article.to_dict()
            elif isinstance(article, dict):
                data = article
            else:
                continue
            
            self._processed_articles.append(data)
    
    def _compute_h_index(self, citation_counts: List[int]) -> int:
        """
        Compute h-index from list of citation counts.
        
        Args:
            citation_counts: List of citation counts per paper
            
        Returns:
            h-index value
        """
        if not citation_counts:
            return 0
        
        sorted_citations = sorted(citation_counts, reverse=True)
        h = 0
        for i, citations in enumerate(sorted_citations, 1):
            if citations >= i:
                h = i
            else:
                break
        return h
    
    def export_to_scopus_csv(
        self,
        output_path: Union[str, Path],
        include_abstracts: bool = True,
    ) -> Path:
        """
        Export articles to Scopus-compatible CSV format.
        
        REQ-EXP-001: Scopus-compatible CSV export.
        
        Args:
            output_path: Path to output CSV file
            include_abstracts: Whether to include abstracts
            
        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.SCOPUS_FIELDS)
            writer.writeheader()
            
            for article in self._processed_articles:
                row = self._format_scopus_row(article, include_abstracts)
                writer.writerow(row)
        
        logger.info(f"Exported {len(self._processed_articles)} articles to {output_path}")
        return output_path
    
    def _format_scopus_row(self, article: Dict[str, Any], include_abstracts: bool) -> Dict[str, str]:
        """Format article data as Scopus CSV row."""
        authors = article.get('authors', [])
        
        # Extract year from publication date
        pub_date = article.get('publication_date', '')
        year = ''
        if pub_date:
            try:
                year = str(datetime.strptime(pub_date[:10], "%Y-%m-%d").year)
            except (ValueError, TypeError):
                year = pub_date[:4] if len(pub_date) >= 4 else ''
        
        # Format authors
        authors_str = "; ".join(authors) if authors else ""
        author_ids = "; ".join([
            article.get('openalex_id', '').split('/')[-1]
        ] * len(authors)) if authors else ""
        
        return {
            "Authors": authors_str,
            "Author full names": authors_str,
            "Author(s) ID": author_ids,
            "Title": article.get('title', ''),
            "Year": year,
            "Source title": article.get('venue', ''),
            "Volume": "",
            "Issue": "",
            "Art. No.": "",
            "Page start": "",
            "Page end": "",
            "Page count": "",
            "Cited by": str(article.get('cited_by_count', 0)),
            "DOI": (article.get('doi', '') or '').replace('https://doi.org/', ''),
            "Link": article.get('doi', ''),
            "Affiliations": "",
            "Authors with affiliations": authors_str,
            "Abstract": article.get('abstract', '') if include_abstracts else "",
            "Author Keywords": "; ".join(article.get('topics', [])[:5]),
            "Index Keywords": "",
            "Document Type": "Article",
            "Publication Stage": "Final",
            "Open Access": "",
            "Source": "OpenAlex",
            "EID": article.get('openalex_id', '').split('/')[-1],
        }
    
    def export_authors_csv(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export author data to CSV.
        
        REQ-EXP-004: Author productivity export.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute author metrics
        author_metrics = self.compute_author_productivity()
        
        fields = [
            "Name", "OpenAlex ID", "Article Count", "Total Citations",
            "H-Index", "First Author Count", "Last Author Count",
            "Coauthor Count", "Top Venues", "Years Active"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            
            for author in author_metrics:
                writer.writerow({
                    "Name": author.name,
                    "OpenAlex ID": author.openalex_id,
                    "Article Count": author.article_count,
                    "Total Citations": author.total_citations,
                    "H-Index": author.h_index,
                    "First Author Count": author.first_author_count,
                    "Last Author Count": author.last_author_count,
                    "Coauthor Count": author.coauthor_count,
                    "Top Venues": "; ".join(author.venues[:3]),
                    "Years Active": f"{author.years_active[0]}-{author.years_active[1]}",
                })
        
        logger.info(f"Exported {len(author_metrics)} authors to {output_path}")
        return output_path
    
    def export_venues_csv(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export venue metrics to CSV.
        
        REQ-EXP-003: Venue quality metrics export.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        venue_metrics = self.compute_venue_metrics()
        
        fields = [
            "Venue Name", "Article Count", "Total Citations",
            "Mean Citations", "H-Index", "Top Authors"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            
            for venue in venue_metrics:
                writer.writerow({
                    "Venue Name": venue.name,
                    "Article Count": venue.article_count,
                    "Total Citations": venue.total_citations,
                    "Mean Citations": f"{venue.mean_citations:.2f}",
                    "H-Index": venue.h_index,
                    "Top Authors": "; ".join(venue.top_authors[:5]),
                })
        
        logger.info(f"Exported {len(venue_metrics)} venues to {output_path}")
        return output_path
    
    def export_institutions_csv(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Export institution metrics to CSV.
        
        REQ-EXP-002: Institutional analysis export.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Path to created CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        institution_metrics = self.compute_institution_metrics()
        
        fields = [
            "Institution Name", "Country", "Article Count",
            "Author Count", "Total Citations", "Top Collaborators"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            
            for inst in institution_metrics:
                writer.writerow({
                    "Institution Name": inst.name,
                    "Country": inst.country,
                    "Article Count": inst.article_count,
                    "Author Count": inst.author_count,
                    "Total Citations": inst.total_citations,
                    "Top Collaborators": "; ".join(inst.collaborators[:5]),
                })
        
        logger.info(f"Exported {len(institution_metrics)} institutions to {output_path}")
        return output_path
    
    def export_all(
        self,
        output_dir: Union[str, Path],
        prefix: str = "eeg_research",
    ) -> Dict[str, Path]:
        """
        Export all data types to CSV files.
        
        Args:
            output_dir: Directory for output files
            prefix: Filename prefix
            
        Returns:
            Dict mapping data type to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        outputs['articles'] = self.export_to_scopus_csv(output_dir / f"{prefix}_articles.csv")
        outputs['authors'] = self.export_authors_csv(output_dir / f"{prefix}_authors.csv")
        outputs['venues'] = self.export_venues_csv(output_dir / f"{prefix}_venues.csv")
        outputs['institutions'] = self.export_institutions_csv(output_dir / f"{prefix}_institutions.csv")
        
        # Also export JSON summary
        summary = {
            "total_articles": len(self._processed_articles),
            "unique_authors": len(set(
                author 
                for a in self._processed_articles 
                for author in a.get('authors', [])
            )),
            "unique_venues": len(set(
                a.get('venue', '') 
                for a in self._processed_articles 
                if a.get('venue')
            )),
            "total_citations": sum(a.get('cited_by_count', 0) for a in self._processed_articles),
            "exported_files": {k: str(v) for k, v in outputs.items()},
        }
        
        summary_path = output_dir / f"{prefix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        outputs['summary'] = summary_path
        
        logger.info(f"Exported all data to {output_dir}")
        return outputs
    
    def compute_venue_metrics(self, top_n: int = 50) -> List[VenueMetrics]:
        """
        Compute quality metrics for publication venues.
        
        REQ-EXP-003: Venue/journal quality metrics.
        
        Args:
            top_n: Number of top venues to return
            
        Returns:
            List of VenueMetrics sorted by article count
        """
        venue_data: Dict[str, Dict] = defaultdict(lambda: {
            'articles': [],
            'citations': [],
            'authors': [],
        })
        
        for article in self._processed_articles:
            venue = article.get('venue', '')
            if not venue:
                continue
            
            venue_data[venue]['articles'].append(article)
            venue_data[venue]['citations'].append(article.get('cited_by_count', 0))
            venue_data[venue]['authors'].extend(article.get('authors', []))
        
        metrics = []
        for venue_name, data in venue_data.items():
            article_count = len(data['articles'])
            citations = data['citations']
            total_citations = sum(citations)
            
            # Compute h-index for venue
            h_index = self._compute_h_index(citations)
            
            # Top authors
            author_counts = Counter(data['authors'])
            top_authors = [a for a, _ in author_counts.most_common(10)]
            
            metrics.append(VenueMetrics(
                name=venue_name,
                article_count=article_count,
                total_citations=total_citations,
                mean_citations=total_citations / article_count if article_count > 0 else 0,
                h_index=h_index,
                top_authors=top_authors,
            ))
        
        # Sort by article count
        metrics.sort(key=lambda x: x.article_count, reverse=True)
        return metrics[:top_n]
    
    def compute_institution_metrics(self, top_n: int = 50) -> List[InstitutionMetrics]:
        """
        Compute metrics for research institutions.
        
        REQ-EXP-002: Institutional collaboration analysis.
        
        Note: This uses author affiliations if available in the data.
        
        Args:
            top_n: Number of top institutions to return
            
        Returns:
            List of InstitutionMetrics sorted by article count
        """
        # For now, extract institutions from author affiliations if available
        # This is a simplified implementation
        inst_data: Dict[str, Dict] = defaultdict(lambda: {
            'articles': [],
            'authors': set(),
            'citations': 0,
            'collaborators': Counter(),
        })
        
        for article in self._processed_articles:
            # Extract affiliations (if available in metadata)
            affiliations = article.get('affiliations', [])
            authors = article.get('authors', [])
            citations = article.get('cited_by_count', 0)
            
            # If no explicit affiliations, try to infer from author structure
            if not affiliations and authors:
                # Use placeholder - in real implementation would parse from data
                for author in authors:
                    # Try to extract affiliation from author info
                    if isinstance(author, dict):
                        affil = author.get('affiliation', author.get('institution', ''))
                        if affil:
                            affiliations.append(affil)
            
            # Track institution data
            for affil in affiliations:
                if affil:
                    inst_data[affil]['articles'].append(article)
                    inst_data[affil]['authors'].update(authors if isinstance(authors[0], str) else [])
                    inst_data[affil]['citations'] += citations
                    
                    # Track collaborations
                    for other_affil in affiliations:
                        if other_affil and other_affil != affil:
                            inst_data[affil]['collaborators'][other_affil] += 1
        
        metrics = []
        for inst_name, data in inst_data.items():
            top_collabs = [c for c, _ in data['collaborators'].most_common(10)]
            
            metrics.append(InstitutionMetrics(
                name=inst_name,
                country="",  # Would require additional parsing
                article_count=len(data['articles']),
                author_count=len(data['authors']),
                total_citations=data['citations'],
                collaborators=top_collabs,
            ))
        
        metrics.sort(key=lambda x: x.article_count, reverse=True)
        return metrics[:top_n]
    
    def compute_author_productivity(self, top_n: int = 100) -> List[AuthorProductivity]:
        """
        Compute productivity metrics for authors.
        
        REQ-EXP-004: Author productivity and h-index estimation.
        
        Args:
            top_n: Number of top authors to return
            
        Returns:
            List of AuthorProductivity sorted by article count
        """
        author_data: Dict[str, Dict] = defaultdict(lambda: {
            'articles': [],
            'citations': [],
            'coauthors': set(),
            'venues': [],
            'years': [],
            'first_author': 0,
            'last_author': 0,
        })
        
        for article in self._processed_articles:
            authors = article.get('authors', [])
            citations = article.get('cited_by_count', 0)
            venue = article.get('venue', '')
            
            # Parse year
            pub_date = article.get('publication_date', '')
            year = 0
            if pub_date:
                try:
                    year = datetime.strptime(pub_date[:10], "%Y-%m-%d").year
                except (ValueError, TypeError):
                    pass
            
            for i, author in enumerate(authors):
                author_name = author if isinstance(author, str) else str(author)
                
                author_data[author_name]['articles'].append(article)
                author_data[author_name]['citations'].append(citations)
                author_data[author_name]['coauthors'].update(
                    a for a in authors if a != author
                )
                if venue:
                    author_data[author_name]['venues'].append(venue)
                if year:
                    author_data[author_name]['years'].append(year)
                
                # Track authorship position
                if i == 0:
                    author_data[author_name]['first_author'] += 1
                if i == len(authors) - 1 and len(authors) > 1:
                    author_data[author_name]['last_author'] += 1
        
        metrics = []
        for author_name, data in author_data.items():
            citations = data['citations']
            h_index = self._compute_h_index(citations)
            
            # Top venues
            venue_counts = Counter(data['venues'])
            top_venues = [v for v, _ in venue_counts.most_common(5)]
            
            # Years active
            years = data['years']
            years_active = (min(years), max(years)) if years else (0, 0)
            
            metrics.append(AuthorProductivity(
                name=author_name,
                article_count=len(data['articles']),
                total_citations=sum(citations),
                h_index=h_index,
                first_author_count=data['first_author'],
                last_author_count=data['last_author'],
                coauthor_count=len(data['coauthors']),
                venues=top_venues,
                years_active=years_active,
            ))
        
        metrics.sort(key=lambda x: x.article_count, reverse=True)
        return metrics[:top_n]
    
    def get_collaboration_network_data(self) -> Dict[str, Any]:
        """
        Get data for building an institutional collaboration network.
        
        REQ-EXP-002: Institutional collaboration network.
        
        Returns:
            Dict with 'nodes' (institutions) and 'edges' (collaborations)
        """
        inst_metrics = self.compute_institution_metrics(top_n=100)
        
        nodes = []
        edges = []
        edge_set = set()
        
        for inst in inst_metrics:
            nodes.append({
                'id': inst.name,
                'label': inst.name[:30],
                'article_count': inst.article_count,
                'citations': inst.total_citations,
            })
            
            for collab in inst.collaborators:
                edge = tuple(sorted([inst.name, collab]))
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append({
                        'source': inst.name,
                        'target': collab,
                        'weight': 1,  # Could be collaboration count
                    })
        
        return {
            'nodes': nodes,
            'edges': edges,
        }


def export_eeg_research(
    articles: List[Any],
    output_dir: Union[str, Path],
    prefix: str = "eeg_research",
) -> Dict[str, Path]:
    """
    Convenience function to export all EEG research data.
    
    Args:
        articles: List of EEGArticle objects or dicts
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict mapping data type to output path
    """
    exporter = EEGResearchExporter(articles)
    return exporter.export_all(output_dir, prefix)


def compute_eeg_venue_metrics(articles: List[Any]) -> List[VenueMetrics]:
    """
    Convenience function to compute venue metrics.
    
    Args:
        articles: List of EEGArticle objects or dicts
        
    Returns:
        List of VenueMetrics
    """
    exporter = EEGResearchExporter(articles)
    return exporter.compute_venue_metrics()


def compute_eeg_author_productivity(articles: List[Any]) -> List[AuthorProductivity]:
    """
    Convenience function to compute author productivity.
    
    Args:
        articles: List of EEGArticle objects or dicts
        
    Returns:
        List of AuthorProductivity
    """
    exporter = EEGResearchExporter(articles)
    return exporter.compute_author_productivity()
