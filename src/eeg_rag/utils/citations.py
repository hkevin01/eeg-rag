"""
Citation export utilities for EEG-RAG.

Supports BibTeX, RIS, and other citation formats.
"""

from typing import Dict, List, Optional
import re
from datetime import datetime


class CitationGenerator:
    """Generate citations in various formats."""
    
    @staticmethod
    def clean_string(s: str) -> str:
        """Clean string for citation formats."""
        if not s:
            return ""
        # Remove extra whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    @staticmethod
    def to_bibtex(paper: Dict, entry_type: str = "article") -> str:
        """
        Convert paper metadata to BibTeX format.
        
        Args:
            paper: Paper metadata dictionary
            entry_type: BibTeX entry type (article, inproceedings, etc.)
            
        Returns:
            BibTeX formatted string
        """
        # Extract fields with fallbacks
        title = paper.get('title') or paper.get('Title', 'Unknown Title')
        authors = paper.get('authors') or paper.get('Authors', 'Unknown')
        year = paper.get('year') or paper.get('Year', datetime.now().year)
        
        # Generate citation key: FirstAuthorYearFirstWord
        first_author = str(authors).split(',')[0].split()[0] if authors and authors != 'Unknown' else 'Unknown'
        first_word = title.split()[0] if title else 'paper'
        cite_key = f"{first_author}{year}{first_word}".replace(' ', '')
        
        # Start BibTeX entry
        bibtex = f"@{entry_type}{{{cite_key},\n"
        bibtex += f"  title = {{{CitationGenerator.clean_string(title)}}},\n"
        bibtex += f"  author = {{{CitationGenerator.clean_string(authors)}}},\n"
        bibtex += f"  year = {{{year}}},\n"
        
        # Add journal/venue if available
        journal = paper.get('journal') or paper.get('Journal / Origin', '')
        if journal and str(journal) != 'nan':
            bibtex += f"  journal = {{{CitationGenerator.clean_string(journal)}}},\n"
        
        # Add DOI if available
        doi = paper.get('doi') or paper.get('DOI', '')
        if doi and str(doi) != 'nan':
            bibtex += f"  doi = {{{doi}}},\n"
        
        # Add PMID if available
        pmid = paper.get('pmid') or paper.get('PMID', '')
        if pmid and str(pmid) != 'nan':
            bibtex += f"  pmid = {{{pmid}}},\n"
        
        # Add abstract if available
        abstract = paper.get('abstract') or paper.get('Abstract', '')
        if abstract and str(abstract) != 'nan' and len(str(abstract)) > 10:
            abstract_clean = CitationGenerator.clean_string(str(abstract))
            if len(abstract_clean) > 500:
                abstract_clean = abstract_clean[:500] + "..."
            bibtex += f"  abstract = {{{abstract_clean}}},\n"
        
        # Add URL if available
        url = paper.get('url') or paper.get('pubmed_url', '')
        if url and str(url) != 'nan':
            bibtex += f"  url = {{{url}}},\n"
        
        bibtex += "}\n"
        return bibtex
    
    @staticmethod
    def to_ris(paper: Dict) -> str:
        """
        Convert paper metadata to RIS format.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            RIS formatted string
        """
        # Extract fields
        title = paper.get('title') or paper.get('Title', 'Unknown Title')
        authors = paper.get('authors') or paper.get('Authors', '')
        year = paper.get('year') or paper.get('Year', '')
        journal = paper.get('journal') or paper.get('Journal / Origin', '')
        abstract = paper.get('abstract') or paper.get('Abstract', '')
        doi = paper.get('doi') or paper.get('DOI', '')
        pmid = paper.get('pmid') or paper.get('PMID', '')
        url = paper.get('url') or paper.get('pubmed_url', '')
        
        ris = "TY  - JOUR\n"  # Journal article
        ris += f"TI  - {CitationGenerator.clean_string(title)}\n"
        
        # Authors (one per line)
        if authors and str(authors) != 'nan':
            author_list = [a.strip() for a in str(authors).split(',')]
            for author in author_list:
                if author:
                    ris += f"AU  - {author}\n"
        
        if year and str(year) != 'nan':
            ris += f"PY  - {year}\n"
        
        if journal and str(journal) != 'nan':
            ris += f"JO  - {CitationGenerator.clean_string(journal)}\n"
        
        if abstract and str(abstract) != 'nan' and len(str(abstract)) > 10:
            ris += f"AB  - {CitationGenerator.clean_string(str(abstract))}\n"
        
        if doi and str(doi) != 'nan':
            ris += f"DO  - {doi}\n"
        
        if url and str(url) != 'nan':
            ris += f"UR  - {url}\n"
        
        if pmid and str(pmid) != 'nan':
            ris += f"M3  - PMID:{pmid}\n"
        
        ris += "ER  - \n\n"
        return ris
    
    @staticmethod
    def to_plain_text(paper: Dict) -> str:
        """
        Convert paper metadata to plain text citation.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Plain text citation
        """
        title = paper.get('title') or paper.get('Title', 'Unknown Title')
        authors = paper.get('authors') or paper.get('Authors', 'Unknown')
        year = paper.get('year') or paper.get('Year', 'n.d.')
        journal = paper.get('journal') or paper.get('Journal / Origin', '')
        
        citation = f"{authors} ({year}). {title}."
        if journal and str(journal) != 'nan':
            citation += f" {journal}."
        
        # Add identifiers
        doi = paper.get('doi') or paper.get('DOI', '')
        if doi and str(doi) != 'nan':
            citation += f" DOI: {doi}"
        
        pmid = paper.get('pmid') or paper.get('PMID', '')
        if pmid and str(pmid) != 'nan':
            citation += f" PMID: {pmid}"
        
        return citation


def generate_citations(papers: List[Dict], format: str = "bibtex") -> str:
    """
    Generate citations for multiple papers.
    
    Args:
        papers: List of paper metadata dictionaries
        format: Citation format (bibtex, ris, plain)
        
    Returns:
        Combined citations string
    """
    generator = CitationGenerator()
    citations = []
    
    for paper in papers:
        if format.lower() == "bibtex":
            citations.append(generator.to_bibtex(paper))
        elif format.lower() == "ris":
            citations.append(generator.to_ris(paper))
        elif format.lower() == "plain":
            citations.append(generator.to_plain_text(paper))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    return "\n".join(citations)
