#!/usr/bin/env python3
"""
Web Interface for EEG-RAG Knowledge Graph Visualization

Provides a FastAPI-based web interface for interactive graph exploration:
- REST API endpoints for graph queries
- Real-time graph visualization
- Search and filtering capabilities
- Export functionality
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import graph components
from .graph_interface import Neo4jInterface
from .graph_visualizer import GraphVisualizer, VisualizationData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : knowledge_graph.web_interface.GraphQueryRequest
# Requirement  : `GraphQueryRequest` class shall be instantiable and expose the documented interface
# Purpose      : Request model for graph queries
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
# Verification : Instantiate GraphQueryRequest with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GraphQueryRequest(BaseModel):
    """Request model for graph queries."""
    query_terms: List[str]
    max_nodes: int = 100
    max_depth: int = 2
    node_types: Optional[List[str]] = None
    

# ---------------------------------------------------------------------------
# ID           : knowledge_graph.web_interface.CitationNetworkRequest
# Requirement  : `CitationNetworkRequest` class shall be instantiable and expose the documented interface
# Purpose      : Request model for citation network
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
# Verification : Instantiate CitationNetworkRequest with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CitationNetworkRequest(BaseModel):
    """Request model for citation network."""
    seed_pmids: List[str]
    max_depth: int = 2
    

# ---------------------------------------------------------------------------
# ID           : knowledge_graph.web_interface.EntityMapRequest
# Requirement  : `EntityMapRequest` class shall be instantiable and expose the documented interface
# Purpose      : Request model for entity relationship map
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
# Verification : Instantiate EntityMapRequest with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class EntityMapRequest(BaseModel):
    """Request model for entity relationship map."""
    entity_types: List[str]
    min_connection_strength: float = 0.1
    

# ---------------------------------------------------------------------------
# ID           : knowledge_graph.web_interface.TemporalAnalysisRequest
# Requirement  : `TemporalAnalysisRequest` class shall be instantiable and expose the documented interface
# Purpose      : Request model for temporal analysis
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
# Verification : Instantiate TemporalAnalysisRequest with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class TemporalAnalysisRequest(BaseModel):
    """Request model for temporal analysis."""
    start_year: int
    end_year: int
    granularity: str = 'year'
    

# ---------------------------------------------------------------------------
# ID           : knowledge_graph.web_interface.GraphWebInterface
# Requirement  : `GraphWebInterface` class shall be instantiable and expose the documented interface
# Purpose      : Web interface for knowledge graph visualization
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
# Verification : Instantiate GraphWebInterface with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GraphWebInterface:
    """Web interface for knowledge graph visualization."""
    
    # ---------------------------------------------------------------------------
    # ID           : knowledge_graph.web_interface.GraphWebInterface.__init__
    # Requirement  : `__init__` shall initialize web interface
    # Purpose      : Initialize web interface
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : neo4j_interface: Neo4jInterface; static_dir: Optional[Path] (default=None)
    # Outputs      : Implicitly None or see body
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
    def __init__(
        self,
        neo4j_interface: Neo4jInterface,
        static_dir: Optional[Path] = None
    ):
        """Initialize web interface.
        
        Args:
            neo4j_interface: Neo4j database interface.
            static_dir: Directory for static web files.
        """
        self.neo4j = neo4j_interface
        self.visualizer = GraphVisualizer(neo4j_interface)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="EEG-RAG Knowledge Graph Explorer",
            description="Interactive exploration of EEG research knowledge graph",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
        # Setup static files if directory provided
        if static_dir and static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # ---------------------------------------------------------------------------
    # ID           : knowledge_graph.web_interface.GraphWebInterface._setup_routes
    # Requirement  : `_setup_routes` shall setup API routes
    # Purpose      : Setup API routes
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
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
    def _setup_routes(self):
        """Setup API routes."""
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.root
        # Requirement  : `root` shall serve main application page
        # Purpose      : Serve main application page
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
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main application page."""
            return self._get_main_page_html()
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.health_check
        # Requirement  : `health_check` shall health check endpoint
        # Purpose      : Health check endpoint
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
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint."""
            try:
                # Test Neo4j connection
                result = await self.neo4j.execute_query("RETURN 1 as test")
                return {"status": "healthy", "database": "connected"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.create_research_network
        # Requirement  : `create_research_network` shall create research network visualization
        # Purpose      : Create research network visualization
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : request: GraphQueryRequest
        # Outputs      : JSONResponse
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
        @self.app.post("/api/graph/research-network")
        async def create_research_network(
            request: GraphQueryRequest
        ) -> JSONResponse:
            """Create research network visualization."""
            try:
                visualization_data = await self.visualizer.create_research_network_visualization(
                    query_terms=request.query_terms,
                    max_nodes=request.max_nodes,
                    max_depth=request.max_depth
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "data": self._visualization_data_to_dict(visualization_data)
                })
                
            except Exception as e:
                logger.error(f"Failed to create research network: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.create_citation_network
        # Requirement  : `create_citation_network` shall create citation network visualization
        # Purpose      : Create citation network visualization
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : request: CitationNetworkRequest
        # Outputs      : JSONResponse
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
        @self.app.post("/api/graph/citation-network")
        async def create_citation_network(
            request: CitationNetworkRequest
        ) -> JSONResponse:
            """Create citation network visualization."""
            try:
                visualization_data = await self.visualizer.create_citation_network_visualization(
                    seed_pmids=request.seed_pmids,
                    max_depth=request.max_depth
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "data": self._visualization_data_to_dict(visualization_data)
                })
                
            except Exception as e:
                logger.error(f"Failed to create citation network: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.create_entity_map
        # Requirement  : `create_entity_map` shall create entity relationship map
        # Purpose      : Create entity relationship map
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : request: EntityMapRequest
        # Outputs      : JSONResponse
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
        @self.app.post("/api/graph/entity-map")
        async def create_entity_map(
            request: EntityMapRequest
        ) -> JSONResponse:
            """Create entity relationship map."""
            try:
                visualization_data = await self.visualizer.create_entity_relationship_map(
                    entity_types=request.entity_types,
                    min_connection_strength=request.min_connection_strength
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "data": self._visualization_data_to_dict(visualization_data)
                })
                
            except Exception as e:
                logger.error(f"Failed to create entity map: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.create_temporal_analysis
        # Requirement  : `create_temporal_analysis` shall create temporal analysis visualization
        # Purpose      : Create temporal analysis visualization
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : request: TemporalAnalysisRequest
        # Outputs      : JSONResponse
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
        @self.app.post("/api/graph/temporal-analysis")
        async def create_temporal_analysis(
            request: TemporalAnalysisRequest
        ) -> JSONResponse:
            """Create temporal analysis visualization."""
            try:
                temporal_data = await self.visualizer.create_temporal_analysis_view(
                    start_year=request.start_year,
                    end_year=request.end_year,
                    time_granularity=request.granularity
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "data": temporal_data
                })
                
            except Exception as e:
                logger.error(f"Failed to create temporal analysis: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.search_papers
        # Requirement  : `search_papers` shall search papers by text query
        # Purpose      : Search papers by text query
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : q: str (default=Query(..., description='Search query')); limit: int (default=Query(20, description='Maximum results'))
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
        @self.app.get("/api/search/papers")
        async def search_papers(
            q: str = Query(..., description="Search query"),
            limit: int = Query(20, description="Maximum results")
        ):
            """Search papers by text query."""
            try:
                cypher_query = f"""
                MATCH (p:Paper)
                WHERE p.title CONTAINS '{q}' OR p.abstract CONTAINS '{q}'
                RETURN p.pmid, p.title, p.year, p.journal
                ORDER BY p.year DESC
                LIMIT {limit}
                """
                
                result = await self.neo4j.execute_query(cypher_query)
                
                papers = [
                    {
                        "pmid": record["p.pmid"],
                        "title": record["p.title"],
                        "year": record["p.year"],
                        "journal": record["p.journal"]
                    }
                    for record in result
                ]
                
                return {"papers": papers, "total": len(papers)}
                
            except Exception as e:
                logger.error(f"Paper search failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.search_entities
        # Requirement  : `search_entities` shall search entities by name
        # Purpose      : Search entities by name
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : q: str (default=Query(..., description='Search query')); entity_type: Optional[str] (default=Query(None, description='Entity type filter')); limit: int (default=Query(20, description='Maximum results'))
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
        @self.app.get("/api/search/entities")
        async def search_entities(
            q: str = Query(..., description="Search query"),
            entity_type: Optional[str] = Query(None, description="Entity type filter"),
            limit: int = Query(20, description="Maximum results")
        ):
            """Search entities by name."""
            try:
                # Build query based on entity type filter
                if entity_type:
                    cypher_query = f"""
                    MATCH (e:{entity_type.title()})
                    WHERE e.name CONTAINS '{q}' OR e.normalized_name CONTAINS '{q.lower()}'
                    RETURN e.name, labels(e)[0] as entity_type, 
                           size((e)--()) as connection_count
                    ORDER BY connection_count DESC
                    LIMIT {limit}
                    """
                else:
                    cypher_query = f"""
                    MATCH (e)
                    WHERE (e.name CONTAINS '{q}' OR e.normalized_name CONTAINS '{q.lower()}')
                    AND NOT e:Paper
                    RETURN e.name, labels(e)[0] as entity_type, 
                           size((e)--()) as connection_count
                    ORDER BY connection_count DESC
                    LIMIT {limit}
                    """
                
                result = await self.neo4j.execute_query(cypher_query)
                
                entities = [
                    {
                        "name": record["e.name"],
                        "type": record["entity_type"],
                        "connections": record["connection_count"]
                    }
                    for record in result
                ]
                
                return {"entities": entities, "total": len(entities)}
                
            except Exception as e:
                logger.error(f"Entity search failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.get_database_stats
        # Requirement  : `get_database_stats` shall get database overview statistics
        # Purpose      : Get database overview statistics
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
        @self.app.get("/api/stats/overview")
        async def get_database_stats():
            """Get database overview statistics."""
            try:
                stats_query = """
                MATCH (n)
                WITH labels(n)[0] as node_type, count(n) as count
                RETURN collect({type: node_type, count: count}) as node_stats
                """
                
                result = await self.neo4j.execute_query(stats_query)
                node_stats = result[0]["node_stats"] if result else []
                
                # Get relationship stats
                rel_query = """
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(r) as count
                RETURN collect({type: rel_type, count: count}) as rel_stats
                """
                
                result = await self.neo4j.execute_query(rel_query)
                rel_stats = result[0]["rel_stats"] if result else []
                
                total_nodes = sum(stat["count"] for stat in node_stats)
                total_relationships = sum(stat["count"] for stat in rel_stats)
                
                return {
                    "nodes": {
                        "total": total_nodes,
                        "by_type": node_stats
                    },
                    "relationships": {
                        "total": total_relationships,
                        "by_type": rel_stats
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get database stats: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ---------------------------------------------------------------------------
        # ID           : knowledge_graph.web_interface.GraphWebInterface.export_to_cytoscape
        # Requirement  : `export_to_cytoscape` shall export visualization to Cytoscape format
        # Purpose      : Export visualization to Cytoscape format
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : background_tasks: BackgroundTasks; request: GraphQueryRequest
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
        @self.app.post("/api/export/cytoscape")
        async def export_to_cytoscape(
            background_tasks: BackgroundTasks,
            request: GraphQueryRequest
        ):
            """Export visualization to Cytoscape format."""
            try:
                # Generate visualization
                visualization_data = await self.visualizer.create_research_network_visualization(
                    query_terms=request.query_terms,
                    max_nodes=request.max_nodes,
                    max_depth=request.max_depth
                )
                
                # Export to Cytoscape format
                output_path = Path(f"/tmp/eeg_graph_export_{int(asyncio.get_event_loop().time())}.json")
                self.visualizer.export_to_cytoscape(visualization_data, output_path)
                
                return {"export_path": str(output_path), "status": "completed"}
                
            except Exception as e:
                logger.error(f"Export failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # ---------------------------------------------------------------------------
    # ID           : knowledge_graph.web_interface.GraphWebInterface._visualization_data_to_dict
    # Requirement  : `_visualization_data_to_dict` shall convert visualization data to dictionary
    # Purpose      : Convert visualization data to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : vis_data: VisualizationData
    # Outputs      : Dict[str, Any]
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
    def _visualization_data_to_dict(self, vis_data: VisualizationData) -> Dict[str, Any]:
        """Convert visualization data to dictionary."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type,
                    "size": node.size,
                    "color": node.color,
                    "properties": node.properties
                }
                for node in vis_data.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "color": edge.color,
                    "properties": edge.properties
                }
                for edge in vis_data.edges
            ],
            "metadata": vis_data.metadata
        }
    
    # ---------------------------------------------------------------------------
    # ID           : knowledge_graph.web_interface.GraphWebInterface._get_main_page_html
    # Requirement  : `_get_main_page_html` shall generate main page HTML
    # Purpose      : Generate main page HTML
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def _get_main_page_html(self) -> str:
        """Generate main page HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EEG-RAG Knowledge Graph Explorer</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1 { 
                    color: #2c3e50; 
                    text-align: center;
                }
                .api-section {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .api-section h3 {
                    color: #34495e;
                    margin-top: 0;
                }
                .endpoint {
                    background: #ecf0f1;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    font-family: monospace;
                }
                .method {
                    display: inline-block;
                    padding: 2px 6px;
                    border-radius: 3px;
                    color: white;
                    font-size: 12px;
                    margin-right: 8px;
                }
                .get { background-color: #27ae60; }
                .post { background-color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 EEG-RAG Knowledge Graph Explorer</h1>
                <p>Interactive exploration and visualization of EEG research knowledge graph.</p>
                
                <div class="api-section">
                    <h3>📊 Graph Visualization APIs</h3>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>/api/graph/research-network
                        <br>Create research network visualization around query terms
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>/api/graph/citation-network
                        <br>Generate citation network from seed PMIDs
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>/api/graph/entity-map
                        <br>Create entity relationship map by type
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>/api/graph/temporal-analysis
                        <br>Generate temporal analysis of research trends
                    </div>
                </div>
                
                <div class="api-section">
                    <h3>🔍 Search APIs</h3>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>/api/search/papers?q=query&limit=20
                        <br>Search research papers by text query
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>/api/search/entities?q=query&entity_type=type&limit=20
                        <br>Search entities by name and type
                    </div>
                </div>
                
                <div class="api-section">
                    <h3>📈 Statistics & Export</h3>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>/api/stats/overview
                        <br>Get database overview statistics
                    </div>
                    
                    <div class="endpoint">
                        <span class="method post">POST</span>/api/export/cytoscape
                        <br>Export visualization to Cytoscape format
                    </div>
                </div>
                
                <div class="api-section">
                    <h3>🔧 System</h3>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>/api/health
                        <br>Health check and database connection status
                    </div>
                </div>
                
                <div style="margin-top: 30px; text-align: center; color: #7f8c8d;">
                    <p>Use the API documentation at <strong>/docs</strong> for interactive testing.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    # ---------------------------------------------------------------------------
    # ID           : knowledge_graph.web_interface.GraphWebInterface.start_server
    # Requirement  : `start_server` shall start the web server
    # Purpose      : Start the web server
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : host: str (default='0.0.0.0'); port: int (default=8000); reload: bool (default=False)
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
    async def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False
    ):
        """Start the web server."""
        import uvicorn
        
        logger.info(f"Starting EEG-RAG Graph Explorer at http://{host}:{port}")
        
        await uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload
        )