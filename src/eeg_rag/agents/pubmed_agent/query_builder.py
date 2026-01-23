"""
PubMed Query Builder

Builds optimized PubMed queries with filters and MeSH expansion.

Requirements Covered:
- REQ-PUBMED-006: Smart query construction
- REQ-PUBMED-007: Date range filtering
- REQ-PUBMED-008: Article type filtering
"""

import logging
from typing import List, Optional, Tuple

from .mesh_expander import MeSHExpander

logger = logging.getLogger(__name__)


class PubMedQueryBuilder:
    """Build optimized PubMed queries."""
    
    # Common article types for filtering
    ARTICLE_TYPES = {
        "review": "Review",
        "systematic_review": "Systematic Review",
        "meta_analysis": "Meta-Analysis",
        "clinical_trial": "Clinical Trial",
        "randomized_controlled_trial": "Randomized Controlled Trial",
        "case_report": "Case Reports",
        "comparative_study": "Comparative Study",
        "evaluation_study": "Evaluation Study",
        "validation_study": "Validation Study",
        "journal_article": "Journal Article",
    }
    
    def __init__(self, mesh_expander: Optional[MeSHExpander] = None):
        """
        Initialize query builder.
        
        Args:
            mesh_expander: MeSH expander instance (creates new if None)
        """
        self.mesh_expander = mesh_expander or MeSHExpander()
        logger.info("PubMedQueryBuilder initialized")
    
    def build_query(
        self,
        query: str,
        use_mesh: bool = True,
        date_range: Optional[Tuple[int, int]] = None,
        article_types: Optional[List[str]] = None,
        humans_only: bool = False,
        english_only: bool = False,
        exclude_reviews: bool = False,
        journal_filter: Optional[List[str]] = None
    ) -> str:
        """
        Build a comprehensive PubMed query.
        
        Args:
            query: Base search query
            use_mesh: Whether to expand with MeSH terms
            date_range: Tuple of (start_year, end_year)
            article_types: List of article type keys (see ARTICLE_TYPES)
            humans_only: Filter to human studies only
            english_only: Filter to English language only
            exclude_reviews: Exclude review articles
            journal_filter: List of journal names to filter by
            
        Returns:
            Formatted PubMed query string
        """
        parts = []
        
        # Main query with optional MeSH expansion
        if use_mesh:
            main_query = self.mesh_expander.expand_query(query)
        else:
            main_query = f'({query}[Title/Abstract])'
        
        parts.append(main_query)
        
        # Date filter
        if date_range:
            start_year, end_year = date_range
            parts.append(f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])')
        
        # Article type filter
        if article_types:
            type_filters = []
            for type_key in article_types:
                if type_key in self.ARTICLE_TYPES:
                    type_filters.append(f'"{self.ARTICLE_TYPES[type_key]}"[Publication Type]')
            if type_filters:
                parts.append(f'({" OR ".join(type_filters)})')
        
        # Exclude reviews if requested
        if exclude_reviews:
            parts.append('NOT "Review"[Publication Type]')
        
        # Human studies only
        if humans_only:
            parts.append('"humans"[MeSH Terms]')
        
        # English language only
        if english_only:
            parts.append('English[Language]')
        
        # Journal filter
        if journal_filter:
            journal_parts = [f'"{j}"[Journal]' for j in journal_filter[:5]]  # Limit to 5
            parts.append(f'({" OR ".join(journal_parts)})')
        
        final_query = " AND ".join(parts)
        logger.debug(f"Built query: {final_query[:100]}...")
        
        return final_query
    
    def build_author_query(
        self,
        author_name: str,
        affiliation: Optional[str] = None,
        date_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Build a query to find papers by author.
        
        Args:
            author_name: Author name (Last Name, First Initial)
            affiliation: Optional affiliation to filter by
            date_range: Optional date range
            
        Returns:
            PubMed query string
        """
        parts = [f'{author_name}[Author]']
        
        if affiliation:
            parts.append(f'{affiliation}[Affiliation]')
        
        if date_range:
            start_year, end_year = date_range
            parts.append(f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])')
        
        return " AND ".join(parts)
    
    def build_citation_query(
        self,
        pmids: List[str]
    ) -> str:
        """
        Build a query to fetch specific PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            PubMed query string
        """
        pmid_parts = [f'{pmid}[PMID]' for pmid in pmids[:100]]  # Limit to 100
        return " OR ".join(pmid_parts)
    
    def build_journal_query(
        self,
        journal_name: str,
        topic: Optional[str] = None,
        date_range: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Build a query for papers from a specific journal.
        
        Args:
            journal_name: Journal name or abbreviation
            topic: Optional topic to filter within journal
            date_range: Optional date range
            
        Returns:
            PubMed query string
        """
        parts = [f'"{journal_name}"[Journal]']
        
        if topic:
            parts.append(f'({topic}[Title/Abstract])')
        
        if date_range:
            start_year, end_year = date_range
            parts.append(f'("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])')
        
        return " AND ".join(parts)
    
    def build_eeg_research_query(
        self,
        topic: str,
        method: Optional[str] = None,
        application: Optional[str] = None,
        recent_only: bool = False
    ) -> str:
        """
        Build a specialized query for EEG research.
        
        Args:
            topic: Main research topic
            method: Specific method (e.g., "deep learning", "CNN")
            application: Application area (e.g., "epilepsy", "BCI")
            recent_only: Filter to last 5 years
            
        Returns:
            PubMed query string
        """
        # Base EEG query
        base = f'(({topic}) AND ("Electroencephalography"[MeSH Terms] OR EEG[Title/Abstract]))'
        parts = [base]
        
        if method:
            method_expanded = self.mesh_expander.expand_query(method)
            parts.append(f'({method_expanded})')
        
        if application:
            app_expanded = self.mesh_expander.expand_query(application)
            parts.append(f'({app_expanded})')
        
        if recent_only:
            from datetime import datetime
            current_year = datetime.now().year
            parts.append(f'("{current_year - 5}"[Date - Publication] : "{current_year}"[Date - Publication])')
        
        return " AND ".join(parts)
    
    def get_suggested_filters(self, query: str) -> dict:
        """
        Get suggested filters based on query analysis.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of suggested filters
        """
        query_lower = query.lower()
        suggestions = {
            "article_types": [],
            "use_mesh": True,
            "humans_only": False,
            "exclude_reviews": False
        }
        
        # Suggest systematic reviews for methodology queries
        if any(term in query_lower for term in ["method", "comparison", "benchmark", "evaluation"]):
            suggestions["article_types"].extend(["systematic_review", "meta_analysis", "comparative_study"])
        
        # Suggest human studies for clinical queries
        if any(term in query_lower for term in ["patient", "clinical", "diagnosis", "treatment", "therapy"]):
            suggestions["humans_only"] = True
        
        # Suggest excluding reviews for original research
        if any(term in query_lower for term in ["novel", "propose", "develop", "new method"]):
            suggestions["exclude_reviews"] = True
        
        # Suggest validation studies for accuracy/performance queries
        if any(term in query_lower for term in ["accuracy", "performance", "validation", "evaluation"]):
            suggestions["article_types"].append("validation_study")
        
        return suggestions
