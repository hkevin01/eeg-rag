# eeg_rag/api/stats_routes.py
"""
FastAPI routes for statistics endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from eeg_rag.services.stats_service import get_stats_service, IndexStats


router = APIRouter(prefix="/api/stats", tags=["statistics"])


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.DisplayStatsResponse
# Requirement  : `DisplayStatsResponse` class shall be instantiable and expose the documented interface
# Purpose      : Response model for display statistics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate DisplayStatsResponse with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class DisplayStatsResponse(BaseModel):
    """Response model for display statistics."""

    papers_indexed: str  # Legacy - kept for compatibility
    papers_cached: str  # New - clearer terminology
    search_coverage: str  # Total papers searchable via APIs
    ai_agents: str
    citation_accuracy: str
    last_updated: str
    raw_count: int


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.FullStatsResponse
# Requirement  : `FullStatsResponse` class shall be instantiable and expose the documented interface
# Purpose      : Response model for full statistics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate FullStatsResponse with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FullStatsResponse(BaseModel):
    """Response model for full statistics."""

    total_papers: int
    papers_with_abstracts: int
    papers_with_embeddings: int
    papers_by_source: Dict[str, int]
    date_range: Dict[str, Optional[int]]
    index_health: Dict[str, Any]
    last_updated: str


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.VerificationResponse
# Requirement  : `VerificationResponse` class shall be instantiable and expose the documented interface
# Purpose      : Response model for verification report
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate VerificationResponse with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class VerificationResponse(BaseModel):
    """Response model for verification report."""

    verified_total: int
    display_total: str
    tables_found: list
    counts: Dict[str, int]
    inconsistencies: list
    recommendations: list


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.HealthResponse
# Requirement  : `HealthResponse` class shall be instantiable and expose the documented interface
# Purpose      : Response model for health status
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate HealthResponse with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Response model for health status."""

    status: str
    total_papers: int
    issues: list
    recommendations: list


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.get_display_stats
# Requirement  : `get_display_stats` shall get statistics formatted for homepage display
# Purpose      : Get statistics formatted for homepage display
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@router.get("/display", response_model=DisplayStatsResponse)
async def get_display_stats():
    """
    Get statistics formatted for homepage display.

    Returns the paper count formatted for display (e.g., "52,431"),
    along with other display-ready statistics.
    """
    service = get_stats_service()
    stats = service.get_display_stats()

    # Also get the raw count for debugging
    full_stats = service.get_full_stats()

    return DisplayStatsResponse(
        papers_indexed=stats.get("papers_indexed", "0"),  # Legacy
        papers_cached=stats.get("papers_cached", stats.get("papers_indexed", "0")),
        search_coverage=stats.get("search_coverage", "35M+ via PubMed"),
        ai_agents=stats["ai_agents"],
        citation_accuracy=stats["citation_accuracy"],
        last_updated=stats["last_updated"],
        raw_count=full_stats.total_papers,
    )


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.get_full_stats
# Requirement  : `get_full_stats` shall get comprehensive statistics about the paper index
# Purpose      : Get comprehensive statistics about the paper index
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@router.get("/full", response_model=FullStatsResponse)
async def get_full_stats():
    """
    Get comprehensive statistics about the paper index.

    Returns detailed breakdown including:
    - Total paper count
    - Papers with abstracts
    - Papers with embeddings
    - Breakdown by source (PubMed, arXiv, etc.)
    - Date range of indexed papers
    - Index health status
    """
    service = get_stats_service()
    stats = service.get_full_stats()

    return FullStatsResponse(
        total_papers=stats.total_papers,
        papers_with_abstracts=stats.papers_with_abstracts,
        papers_with_embeddings=stats.papers_with_embeddings,
        papers_by_source=stats.papers_by_source,
        date_range=stats.date_range,
        index_health=stats.index_health,
        last_updated=(
            stats.last_updated.isoformat()
            if stats.last_updated
            else datetime.now().isoformat()
        ),
    )


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.verify_stats
# Requirement  : `verify_stats` shall run verification on database statistics
# Purpose      : Run verification on database statistics
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@router.get("/verify", response_model=VerificationResponse)
async def verify_stats():
    """
    Run verification on database statistics.

    Checks for:
    - Table consistency
    - Duplicate papers
    - Missing data
    - Source inconsistencies

    Returns a detailed report with issues and recommendations.
    """
    service = get_stats_service()
    report = service.verify_counts()

    return VerificationResponse(
        verified_total=report.get("verified_total", 0),
        display_total=report.get("display_total", "0"),
        tables_found=report.get("tables_found", []),
        counts=report.get("counts", {}),
        inconsistencies=report.get("inconsistencies", []),
        recommendations=report.get("recommendations", []),
    )


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.refresh_stats
# Requirement  : `refresh_stats` shall force refresh of cached statistics
# Purpose      : Force refresh of cached statistics
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@router.post("/refresh")
async def refresh_stats():
    """
    Force refresh of cached statistics.

    Invalidates the cache and fetches fresh data from the database.
    Use this after bulk imports or data modifications.
    """
    service = get_stats_service()
    service.invalidate_cache()

    # Get fresh stats
    stats = service.get_full_stats(use_cache=False)

    return {
        "status": "refreshed",
        "total_papers": stats.total_papers,
        "last_updated": (
            stats.last_updated.isoformat()
            if stats.last_updated
            else datetime.now().isoformat()
        ),
    }


# ---------------------------------------------------------------------------
# ID           : api.stats_routes.get_health
# Requirement  : `get_health` shall get index health status
# Purpose      : Get index health status
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Implicitly None or see body
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Get index health status.

    Returns:
    - Overall health status (healthy/degraded/critical)
    - Current paper count
    - List of issues (if any)
    - Recommendations for improvement
    """
    service = get_stats_service()
    stats = service.get_full_stats()

    health = stats.index_health

    # Generate recommendations based on issues
    recommendations = []
    if health["status"] != "healthy":
        for issue in health.get("issues", []):
            if "embedding" in issue.lower():
                recommendations.append(
                    "Run embedding generation: `python -m eeg_rag.cli.embed generate`"
                )
            if "abstract" in issue.lower():
                recommendations.append(
                    "Fetch abstracts: `python -m eeg_rag.cli.fetch abstracts`"
                )
            if "papers" in issue.lower() and "low" in issue.lower():
                recommendations.append(
                    "Import more papers: `python -m eeg_rag.cli.ingest --source pubmed`"
                )

    return HealthResponse(
        status=health["status"],
        total_papers=stats.total_papers,
        issues=health.get("issues", []),
        recommendations=recommendations,
    )


# Optional: WebSocket endpoint for live updates
# from fastapi import WebSocket
#
# @router.websocket("/ws/live")
# async def stats_websocket(websocket: WebSocket):
#     """WebSocket for live statistics updates."""
#     await websocket.accept()
#     service = get_stats_service()
#
#     while True:
#         stats = service.get_display_stats()
#         await websocket.send_json(stats)
#         await asyncio.sleep(30)  # Update every 30 seconds
