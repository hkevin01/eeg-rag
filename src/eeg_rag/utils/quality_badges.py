"""
Quality badge utilities for displaying paper metadata.

Provides badges for code availability, data availability, citations, etc.
"""

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_code_badge
# Requirement  : `get_code_badge` shall get badge for code availability
# Purpose      : Get badge for code availability
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : Optional[str]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def get_code_badge(paper: Dict) -> Optional[str]:
    """Get badge for code availability."""
    code_available = paper.get('Code available') or paper.get('code_available', '')
    
    if not code_available or str(code_available).lower() in ['nan', 'no', 'none', '']:
        return None
    
    if str(code_available).lower() in ['yes', 'github', 'gitlab', 'bitbucket']:
        return "💻 Code Available"
    
    return None


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_data_badge
# Requirement  : `get_data_badge` shall get badge for data availability
# Purpose      : Get badge for data availability
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : Optional[str]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def get_data_badge(paper: Dict) -> Optional[str]:
    """Get badge for data availability."""
    data_accessibility = paper.get('Dataset accessibility') or paper.get('data_available', '')
    
    if not data_accessibility or str(data_accessibility).lower() in ['nan', 'no', 'none', 'private', '']:
        return None
    
    if str(data_accessibility).lower() in ['yes', 'public', 'open', 'available']:
        return "📊 Public Data"
    
    return None


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_reproducibility_badge
# Requirement  : `get_reproducibility_badge` shall get badge for reproducibility (code + data)
# Purpose      : Get badge for reproducibility (code + data)
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : Optional[str]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_citation_count_badge
# Requirement  : `get_citation_count_badge` shall get badge for citation count
# Purpose      : Get badge for citation count
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : Optional[str]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_all_badges
# Requirement  : `get_all_badges` shall get all applicable badges as a formatted string
# Purpose      : Get all applicable badges as a formatted string
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : str
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.render_badge_html
# Requirement  : `render_badge_html` shall render a badge as HTML
# Purpose      : Render a badge as HTML
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : badge_text: str; color: str (default='#007bff')
# Outputs      : str
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def render_badge_html(badge_text: str, color: str = "#007bff") -> str:
    """Render a badge as HTML."""
    return f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-right:5px;">{badge_text}</span>'


# ---------------------------------------------------------------------------
# ID           : utils.quality_badges.get_quality_score
# Requirement  : `get_quality_score` shall calculate a quality score (0-1) based on available metadata
# Purpose      : Calculate a quality score (0-1) based on available metadata
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : paper: Dict
# Outputs      : float
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
