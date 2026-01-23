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


class DisplayStatsResponse(BaseModel):
    """Response model for display statistics."""
    papers_indexed: str
    ai_agents: str
    citation_accuracy: str
    last_updated: str
    raw_count: int


class FullStatsResponse(BaseModel):
    """Response model for full statistics."""
    total_papers: int
    papers_with_abstracts: int
    papers_with_embeddings: int
    papers_by_source: Dict[str, int]
    date_range: Dict[str, Optional[int]]
    index_health: Dict[str, Any]
    last_updated: str


class VerificationResponse(BaseModel):
    """Response model for verification report."""
    verified_total: int
    display_total: str
    tables_found: list
    counts: Dict[str, int]
    inconsistencies: list
    recommendations: list


class HealthResponse(BaseModel):
    """Response model for health status."""
    status: str
    total_papers: int
    issues: list
    recommendations: list


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
        papers_indexed=stats['papers_indexed'],
        ai_agents=stats['ai_agents'],
        citation_accuracy=stats['citation_accuracy'],
        last_updated=stats['last_updated'],
        raw_count=full_stats.total_papers
    )


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
        last_updated=stats.last_updated.isoformat() if stats.last_updated else datetime.now().isoformat()
    )


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
        verified_total=report.get('verified_total', 0),
        display_total=report.get('display_total', '0'),
        tables_found=report.get('tables_found', []),
        counts=report.get('counts', {}),
        inconsistencies=report.get('inconsistencies', []),
        recommendations=report.get('recommendations', [])
    )


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
        "last_updated": stats.last_updated.isoformat() if stats.last_updated else datetime.now().isoformat()
    }


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
    if health['status'] != 'healthy':
        for issue in health.get('issues', []):
            if 'embedding' in issue.lower():
                recommendations.append("Run embedding generation: `python -m eeg_rag.cli.embed generate`")
            if 'abstract' in issue.lower():
                recommendations.append("Fetch abstracts: `python -m eeg_rag.cli.fetch abstracts`")
            if 'papers' in issue.lower() and 'low' in issue.lower():
                recommendations.append("Import more papers: `python -m eeg_rag.cli.ingest --source pubmed`")
    
    return HealthResponse(
        status=health['status'],
        total_papers=stats.total_papers,
        issues=health.get('issues', []),
        recommendations=recommendations
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
