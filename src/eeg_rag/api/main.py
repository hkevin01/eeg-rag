"""
FastAPI web service for EEG Literature RAG System.

Provides REST API and Server-Sent Events for the multi-agent orchestrator.
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import asyncio
import json
import logging
import os
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

from ..orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    QueryType,
    ExecutionStrategy
)
from ..agents.local_agent.local_data_agent import LocalDataAgent
from ..agents.pubmed_agent.pubmed_agent import PubMedAgent
from ..agents.semantic_scholar_agent.s2_agent import SemanticScholarAgent
from ..agents.synthesis_agent.synthesis_agent import SynthesisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
orchestrator: Optional[Orchestrator] = None
active_streams: Dict[str, asyncio.Queue] = {}


# ============== Lifecycle Management ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global orchestrator
    
    # Startup
    logger.info("Initializing EEG-RAG API...")
    
    config = {
        "pubmed_api_key": os.getenv("NCBI_API_KEY"),
        "s2_api_key": os.getenv("S2_API_KEY"),
        "email": os.getenv("RESEARCHER_EMAIL", "researcher@example.com"),
        "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
        "chroma_port": int(os.getenv("CHROMA_PORT", "8000"))
    }
    
    try:
        # Initialize agents
        local_agent = LocalDataAgent(
            collection_name="eeg_papers",
            chroma_host=config["chroma_host"],
            chroma_port=config["chroma_port"]
        )
        
        pubmed_agent = PubMedAgent(
            api_key=config["pubmed_api_key"],
            email=config["email"]
        )
        
        s2_agent = SemanticScholarAgent(
            api_key=config["s2_api_key"]
        )
        
        synthesis_agent = SynthesisAgent()
        
        # Create orchestrator
        orchestrator = Orchestrator(
            local_agent=local_agent,
            pubmed_agent=pubmed_agent,
            s2_agent=s2_agent,
            synthesis_agent=synthesis_agent,
            config=config
        )
        
        logger.info("✅ Orchestrator initialized successfully")
        logger.info(f"Available agents: {list(orchestrator.agents.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EEG-RAG API...")
    if orchestrator:
        await orchestrator.close()
    logger.info("✅ Shutdown complete")


# ============== FastAPI App ==============

app = FastAPI(
    title="EEG Literature RAG API",
    description="Multi-agent research assistant for EEG literature with intelligent synthesis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=3, max_length=500, description="Research query")
    max_results: int = Field(default=50, ge=1, le=200, description="Maximum results per source")
    sources: Optional[List[str]] = Field(default=None, description="Specific sources to use")
    date_range: Optional[List[int]] = Field(default=None, description="[start_year, end_year]")
    synthesize: bool = Field(default=True, description="Whether to synthesize results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "deep learning seizure detection EEG",
                "max_results": 50,
                "sources": ["local", "pubmed", "s2"],
                "date_range": [2020, 2025],
                "synthesize": True
            }
        }


class SearchResponse(BaseModel):
    """Search response model."""
    query_id: str
    success: bool
    papers: List[Dict[str, Any]]
    synthesis: Optional[Dict[str, Any]]
    total_found: int
    sources_used: List[str]
    execution_time_ms: float
    errors: List[str]
    metadata: Dict[str, Any] = {}


class PaperDetailsRequest(BaseModel):
    """Paper details request."""
    paper_id: str = Field(..., description="Paper identifier (PMID, DOI, or S2 ID)")
    source: str = Field(default="auto", description="Source: auto, pubmed, or s2")


class CitationRequest(BaseModel):
    """Citation network request."""
    paper_id: str = Field(..., description="Paper identifier")
    direction: str = Field(default="both", description="citing, cited_by, or both")
    source: str = Field(default="s2", description="Source: pubmed or s2")
    max_results: int = Field(default=50, ge=1, le=200)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    agents: List[str]
    uptime_seconds: float


# ============== API Endpoints ==============

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "EEG Literature RAG API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and available agents.
    """
    import time
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        agents=list(orchestrator.agents.keys()),
        uptime_seconds=time.monotonic()
    )


@app.get("/metrics")
async def get_metrics():
    """
    Get agent performance metrics.
    
    Returns detailed metrics for each agent including:
    - Request counts
    - Error rates
    - Average latency
    - Cache hit rates
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        metrics = orchestrator.get_agent_metrics()
        return {"metrics": metrics, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute a literature search query.
    
    Coordinates multiple agents (LocalData, PubMed, SemanticScholar) and synthesizes results.
    
    **Query Types Detected:**
    - Exploratory: Broad topic exploration
    - Comparative: Compare approaches/methods
    - Temporal: Evolution over time
    - Author-focused: Papers by specific authors
    - Dataset-focused: Papers using specific datasets
    - Citation network: Citation analysis
    
    **Returns:**
    - Merged and deduplicated papers from all sources
    - Research synthesis with themes, gaps, and evidence ranking
    - Execution statistics
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    date_range = None
    if request.date_range and len(request.date_range) == 2:
        date_range = tuple(request.date_range)
    
    try:
        logger.info(f"Search request: {request.query[:50]}... | sources={request.sources}")
        
        result = await orchestrator.search(
            query=request.query,
            max_results=request.max_results,
            sources=request.sources,
            date_range=date_range,
            synthesize=request.synthesize
        )
        
        logger.info(
            f"Search complete: {result.total_found} papers in {result.execution_time_ms:.0f}ms"
        )
        
        return SearchResponse(
            query_id=result.query_id,
            success=result.success,
            papers=result.papers,
            synthesis=result.synthesis,
            total_found=result.total_found,
            sources_used=result.sources_used,
            execution_time_ms=result.execution_time_ms,
            errors=result.errors,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/stream")
async def search_stream(request: SearchRequest):
    """
    Execute a search with streaming progress updates.
    
    Returns Server-Sent Events (SSE) with real-time progress and results.
    
    **Event Types:**
    - `progress`: Execution progress updates
    - `complete`: Final results
    - `error`: Error information
    - `heartbeat`: Keep-alive signals
    
    **Usage:**
    ```javascript
    const eventSource = new EventSource('/search/stream');
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        console.log(`${data.stage}: ${data.percent}%`);
      } else if (data.type === 'complete') {
        console.log('Results:', data.result);
      }
    };
    ```
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    query_id = str(uuid.uuid4())[:8]
    progress_queue: asyncio.Queue = asyncio.Queue()
    active_streams[query_id] = progress_queue
    
    async def generate_events():
        """Generate SSE events."""
        try:
            # Progress callback
            def on_progress(stage: str, percent: float):
                try:
                    asyncio.create_task(progress_queue.put({
                        "type": "progress",
                        "stage": stage,
                        "percent": round(percent * 100, 1),
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
            
            # Start search in background
            date_range = None
            if request.date_range and len(request.date_range) == 2:
                date_range = tuple(request.date_range)
            
            search_task = asyncio.create_task(
                orchestrator.search(
                    query=request.query,
                    max_results=request.max_results,
                    sources=request.sources,
                    date_range=date_range,
                    synthesize=request.synthesize,
                    progress_callback=on_progress
                )
            )
            
            # Stream progress events
            heartbeat_counter = 0
            while not search_task.done():
                try:
                    event = await asyncio.wait_for(
                        progress_queue.get(),
                        timeout=2.0
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                    heartbeat_counter = 0
                except asyncio.TimeoutError:
                    # Send heartbeat every 2 seconds
                    heartbeat_counter += 1
                    if heartbeat_counter % 15 == 0:  # Every 30 seconds
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            # Get final result
            result = await search_task
            
            # Send final result
            final_event = {
                "type": "complete",
                "result": {
                    "query_id": result.query_id,
                    "success": result.success,
                    "papers": result.papers,
                    "synthesis": result.synthesis,
                    "total_found": result.total_found,
                    "sources_used": result.sources_used,
                    "execution_time_ms": result.execution_time_ms,
                    "errors": result.errors,
                    "metadata": result.metadata
                }
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        
        finally:
            if query_id in active_streams:
                del active_streams[query_id]
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/paper/details")
async def get_paper_details(request: PaperDetailsRequest):
    """
    Get detailed information about a specific paper.
    
    Fetches complete metadata from the appropriate source (PubMed or Semantic Scholar).
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        paper = await orchestrator.get_paper_details(
            paper_id=request.paper_id,
            source=request.source
        )
        
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return {"paper": paper, "source": request.source}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper details error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper/citations")
async def get_paper_citations(request: CitationRequest):
    """
    Get citation network for a paper.
    
    Returns papers that cite this paper and/or papers cited by this paper.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        citations = await orchestrator.get_citations(
            paper_id=request.paper_id,
            direction=request.direction,
            source=request.source
        )
        
        return {
            "paper_id": request.paper_id,
            "direction": request.direction,
            "source": request.source,
            **citations
        }
        
    except Exception as e:
        logger.error(f"Citations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggest")
async def suggest_queries(
    prefix: str = Query(..., min_length=2, max_length=100, description="Query prefix")
):
    """
    Get query suggestions based on prefix.
    
    Returns common EEG research query patterns that match the prefix.
    """
    # Common EEG research query patterns
    suggestions = [
        "EEG seizure detection deep learning",
        "EEG emotion recognition CNN",
        "motor imagery BCI classification",
        "sleep stage classification EEG",
        "EEG artifact removal ICA",
        "transformer EEG signal processing",
        "cross-subject EEG transfer learning",
        "real-time EEG classification embedded",
        "EEG Alzheimer detection biomarkers",
        "attention mechanism EEG temporal",
        "graph neural network EEG connectivity",
        "EEG data augmentation techniques",
        "multi-channel EEG analysis methods",
        "EEG feature extraction wavelet",
        "LSTM EEG time series forecasting",
        "P300 speller BCI",
        "SSVEP frequency detection",
        "ERN error related negativity",
        "alpha beta gamma oscillations",
        "EEG preprocessing pipelines"
    ]
    
    prefix_lower = prefix.lower()
    matches = [s for s in suggestions if prefix_lower in s.lower()]
    
    return {"suggestions": matches[:10], "total": len(matches)}


@app.get("/query-types")
async def get_query_types():
    """
    Get available query types and their descriptions.
    
    Returns information about how different query types are handled.
    """
    query_types = {
        "exploratory": {
            "description": "Broad topic exploration across multiple sources",
            "strategy": "cascading",
            "example": "EEG deep learning methods"
        },
        "specific": {
            "description": "Specific method, technique, or implementation",
            "strategy": "parallel",
            "example": "how to implement EEGNet"
        },
        "comparative": {
            "description": "Compare approaches, methods, or techniques",
            "strategy": "parallel",
            "example": "CNN vs LSTM for EEG classification"
        },
        "temporal": {
            "description": "Evolution or trends over time",
            "strategy": "parallel",
            "example": "EEG seizure detection trends 2020-2025"
        },
        "author_focused": {
            "description": "Papers by specific authors or researchers",
            "strategy": "parallel",
            "example": "papers by Schomer on EEG"
        },
        "dataset_focused": {
            "description": "Papers using specific datasets or benchmarks",
            "strategy": "parallel",
            "example": "CHB-MIT seizure detection benchmark"
        },
        "citation_network": {
            "description": "Citation analysis and network traversal",
            "strategy": "cascading",
            "example": "influential papers on BCI systems"
        }
    }
    
    return {"query_types": query_types}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": str(exc.detail),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 handler."""
    logger.error(f"Internal error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============== Development Server ==============

def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    
    logger.info(f"Starting EEG-RAG API server on {host}:{port}")
    uvicorn.run(
        "eeg_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(reload=True)
