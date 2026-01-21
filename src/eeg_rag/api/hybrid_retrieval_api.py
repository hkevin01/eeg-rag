"""
FastAPI endpoint for hybrid retrieval

Provides REST API for the hybrid retrieval system with query expansion.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent
from eeg_rag.agents.base_agent import AgentQuery


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_api")

# Initialize FastAPI app
app = FastAPI(
    title="EEG-RAG Hybrid Retrieval API",
    description="Hybrid search API with BM25 + Dense + RRF fusion",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    use_query_expansion: bool = Field(True)
    use_reranking: bool = Field(False, description="Enable cross-encoder reranking for improved quality")


class SearchResponse(BaseModel):
    """Search response model"""
    success: bool
    query: str
    num_results: int
    search_time_ms: float
    results: List[Dict[str, Any]]
    reranking_enabled: bool = False


# Global agent instance
_agent: Optional[LocalDataAgent] = None


def get_agent(use_reranking: bool = False) -> LocalDataAgent:
    """Get or create agent instance"""
    global _agent
    
    # Recreate agent if reranking setting changed
    if _agent is not None and _agent.use_reranking != use_reranking:
        _agent = None
    
    if _agent is None:
        config = {
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "eeg_papers",
            "bm25_cache_dir": "data/bm25_cache",
            "use_query_expansion": True,
            "min_relevance_score": 0.01,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
        
        _agent = LocalDataAgent(
            config=config,
            use_hybrid_retrieval=True,
            use_reranking=use_reranking
        )
        logger.info(f"LocalDataAgent initialized (hybrid=True, reranking={use_reranking})")
    
    return _agent


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    logger.info("Starting EEG-RAG Hybrid Retrieval API...")
    get_agent()
    logger.info("API ready")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Hybrid search endpoint with optional reranking"""
    try:
        agent = get_agent(use_reranking=request.use_reranking)
        agent.top_k = request.top_k
        
        query = AgentQuery(text=request.query, intent="search")
        result = await agent.execute(query)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return SearchResponse(
            success=True,
            query=request.query,
            num_results=len(result.data['results']),
            search_time_ms=result.data['search_time_ms'],
            results=result.data['results'],
            reranking_enabled=request.use_reranking
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
