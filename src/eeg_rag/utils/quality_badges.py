"""
Quality badge utilities for displaying paper metadata.

Provides badges for code availability, data availability, citations, etc.
"""

from typing import Dict, Optional


def get_code_badge(paper: Dict) -> Optional[str]:
    """Get badge for code availability."""
    code_available = paper.get('Code available') or paper.get('code_available', '')
    
    if not code_available or str(code_available).lower() in ['nan', 'no', 'none', '']:
        return None
    
    if str(code_available).lower() in ['yes', 'github', 'gitlab', 'bitbucket']:
        return "💻 Code Available"
    
    return None


def get_data_badge(paper: Dict) -> Optional[str]:
    """Get badge for data availability."""
    data_accessibility = paper.get('Dataset accessibility') or paper.get('data_available', '')
    
    if not data_accessibility or str(data_accessibility).lower() in ['nan', 'no', 'none', 'private', '']:
        return None
    
    if str(data_accessibility).lower() in ['yes', 'public', 'open', 'available']:
        return "📊 Public Data"
    
    return None


def get_reproducibility_badge(paper: Dict) -> Optional[str]:
    """Get badge for reproducibility (code + data)."""
    has_code = get_code_badge(paper) is not None
    has_data = get_data_badge(paper) is not None
    
    if has_code and has_data:
        return "✅ Fully Reproducible"
    elif has_code or has_data:
        return "🔄 Partially Reproducible"
    else:
        return None


def get_citation_count_badge(paper: Dict) -> Optional[str]:
    """Get badge for citation count."""
    citation_count = paper.get('citation_count') or paper.get('citations', 0)
    
    try:
        count = int(citation_count)
    except (ValueError, TypeError):
        return None
    
    if count == 0:
        return None
    elif count < 10:
        return f"📄 {count} citations"
    elif count < 50:
        return f"📚 {count} citations"
    elif count < 100:
        return f"🌟 {count} citations"
    else:
        return f"⭐ {count} citations"


def get_all_badges(paper: Dict) -> str:
    """Get all applicable badges as a formatted string."""
    badges = []
    
    # Reproducibility (highest priority)
    repro = get_reproducibility_badge(paper)
    if repro:
        badges.append(repro)
    else:
        # Show individual badges if not fully/partially reproducible
        code = get_code_badge(paper)
        if code:
            badges.append(code)
        
        data = get_data_badge(paper)
        if data:
            badges.append(data)
    
    # Citations
    citations = get_citation_count_badge(paper)
    if citations:
        badges.append(citations)
    
    return " · ".join(badges) if badges else ""


def render_badge_html(badge_text: str, color: str = "#007bff") -> str:
    """Render a badge as HTML."""
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-right:5px;">{badge_text}</span>'


def get_quality_score(paper: Dict) -> float:
    """
    Calculate a quality score (0-1) based on available metadata.
    
    Factors:
    - Code availability: +0.3
    - Data availability: +0.3
    - Has DOI/PMID: +0.2
    - Has citation count: +0.1
    - Has abstract: +0.1
    """
    score = 0.0
    
    # Code
    if get_code_badge(paper):
        score += 0.3
    
    # Data
    if get_data_badge(paper):
        score += 0.3
    
    # DOI/PMID
    doi = paper.get('doi') or paper.get('DOI', '')
    pmid = paper.get('pmid') or paper.get('PMID', '')
    if (doi and str(doi) != 'nan') or (pmid and str(pmid) != 'nan'):
        score += 0.2
    
    # Citations
    if get_citation_count_badge(paper):
        score += 0.1
    
    # Abstract
    abstract = paper.get('abstract') or paper.get('Abstract', '')
    if abstract and str(abstract) != 'nan' and len(str(abstract)) > 50:
        score += 0.1
    
    return min(score, 1.0)
