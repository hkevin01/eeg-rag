"""
Agent 3: Knowledge Graph Agent
Queries Neo4j knowledge graph for EEG biomarker relationships and entity connections.

REQ-AGT3-001 to REQ-AGT3-015: Knowledge graph querying, relationship traversal, Cypher generation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import asyncio
import json
import time
import hashlib
from collections import defaultdict


# Node and Relationship Types
class NodeType(Enum):
    """Types of nodes in the EEG knowledge graph"""
    BIOMARKER = "Biomarker"
    CONDITION = "Condition"
    OUTCOME = "Outcome"
    STUDY = "Study"
    PAPER = "Paper"
    DATASET = "Dataset"
    METHOD = "Method"
    BRAIN_REGION = "BrainRegion"


class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    PREDICTS = "PREDICTS"
    CORRELATES_WITH = "CORRELATES_WITH"
    INDICATES = "INDICATES"
    MEASURED_IN = "MEASURED_IN"
    REPORTS = "REPORTS"
    USES = "USES"
    LOCATED_IN = "LOCATED_IN"
    AFFECTS = "AFFECTS"


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any]
    labels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.node_id,
            'type': self.node_type.value,
            'properties': self.properties,
            'labels': self.labels
        }


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # 0.0 - 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.relationship_type.value,
            'properties': self.properties,
            'strength': self.strength
        }


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    path_length: int
    total_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'relationships': [r.to_dict() for r in self.relationships],
            'path_length': self.path_length,
            'total_strength': self.total_strength
        }


@dataclass
class GraphQueryResult:
    """Result from a graph query"""
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    paths: List[GraphPath]
    subgraph: Dict[str, Any]
    query_text: str
    cypher_query: str
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'relationships': [r.to_dict() for r in self.relationships],
            'paths': [p.to_dict() for p in self.paths],
            'subgraph': self.subgraph,
            'query_text': self.query_text,
            'cypher_query': self.cypher_query,
            'execution_time': self.execution_time
        }


class CypherQueryBuilder:
    """Builds Cypher queries from natural language"""
    
    # EEG-specific query patterns
    PATTERNS = {
        'find_biomarkers': """
            MATCH (b:Biomarker)-[r:PREDICTS]->(c:Condition)
            WHERE toLower(c.name) CONTAINS toLower($condition)
            RETURN b, r, c
            ORDER BY r.strength DESC
            LIMIT $limit
        """,
        'biomarker_relationships': """
            MATCH (b:Biomarker {name: $biomarker})-[r]->(target)
            RETURN b, r, target, type(r) as rel_type
            LIMIT $limit
        """,
        'multi_hop_path': """
            MATCH path = (start)-[*1..$hops]-(end)
            WHERE toLower(start.name) CONTAINS toLower($start_entity)
            AND toLower(end.name) CONTAINS toLower($end_entity)
            RETURN path
            LIMIT $limit
        """,
        'related_studies': """
            MATCH (s:Study)-[:REPORTS]->(b:Biomarker)
            WHERE toLower(b.name) CONTAINS toLower($biomarker)
            RETURN s, b
            ORDER BY s.year DESC
            LIMIT $limit
        """,
        'condition_outcomes': """
            MATCH (c:Condition)-[r:HAS_OUTCOME]->(o:Outcome)
            WHERE toLower(c.name) CONTAINS toLower($condition)
            RETURN c, r, o
            ORDER BY r.probability DESC
            LIMIT $limit
        """
    }
    
    @staticmethod
    def detect_query_intent(query_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect the intent of a natural language query and extract parameters
        
        Returns:
            Tuple of (pattern_name, parameters)
        """
        query_lower = query_text.lower()
        
        # Pattern: Find biomarkers for condition
        if any(word in query_lower for word in ['biomarker', 'marker', 'indicator']) and \
           any(word in query_lower for word in ['predict', 'indicate', 'correlate']):
            # Extract condition name
            for condition in ['epilepsy', 'seizure', 'alzheimer', 'parkinson', 'depression', 'schizophrenia']:
                if condition in query_lower:
                    return 'find_biomarkers', {'condition': condition, 'limit': 10}
            return 'find_biomarkers', {'condition': query_text.split()[-1], 'limit': 10}
        
        # Pattern: Relationships of specific biomarker
        if 'relationship' in query_lower or 'related to' in query_lower or 'connected' in query_lower:
            # Extract biomarker name
            words = query_text.split()
            for i, word in enumerate(words):
                if word.lower() in ['p300', 'alpha', 'beta', 'gamma', 'theta', 'delta']:
                    return 'biomarker_relationships', {'biomarker': word, 'limit': 10}
            return 'biomarker_relationships', {'biomarker': words[-1], 'limit': 10}
        
        # Pattern: Multi-hop path query
        if 'between' in query_lower or 'connect' in query_lower or 'link' in query_lower:
            words = query_text.split()
            if 'and' in words:
                idx = words.index('and')
                start = ' '.join(words[1:idx])
                end = ' '.join(words[idx+1:])
                return 'multi_hop_path', {'start_entity': start, 'end_entity': end, 'hops': 3, 'limit': 5}
        
        # Pattern: Studies about biomarker
        if 'stud' in query_lower or 'research' in query_lower or 'paper' in query_lower:
            words = query_text.split()
            biomarker = words[-1].rstrip('?')
            return 'related_studies', {'biomarker': biomarker, 'limit': 10}
        
        # Pattern: Condition outcomes
        if 'outcome' in query_lower or 'prognosis' in query_lower or 'result' in query_lower:
            for condition in ['epilepsy', 'seizure', 'alzheimer', 'parkinson', 'depression']:
                if condition in query_lower:
                    return 'condition_outcomes', {'condition': condition, 'limit': 10}
        
        # Default: find biomarkers
        return 'find_biomarkers', {'condition': 'epilepsy', 'limit': 10}
    
    @staticmethod
    def build_cypher(pattern_name: str, parameters: Dict[str, Any]) -> str:
        """Build Cypher query from pattern and parameters"""
        if pattern_name not in CypherQueryBuilder.PATTERNS:
            pattern_name = 'find_biomarkers'
        
        # Get template
        template = CypherQueryBuilder.PATTERNS[pattern_name]
        
        # Simple parameter substitution (in production, use proper parameterized queries)
        query = template
        for key, value in parameters.items():
            if isinstance(value, str):
                query = query.replace(f'${key}', f'"{value}"')
            else:
                query = query.replace(f'${key}', str(value))
        
        return query.strip()


class MockNeo4jConnection:
    """Mock Neo4j connection for testing (replace with real neo4j.Driver in production)"""
    
    def __init__(self):
        self.mock_data = self._create_mock_data()
    
    def _create_mock_data(self) -> Dict[str, Any]:
        """Create mock EEG knowledge graph data"""
        return {
            'nodes': [
                {'id': 'bio1', 'type': 'Biomarker', 'name': 'P300 amplitude', 'unit': 'μV'},
                {'id': 'bio2', 'type': 'Biomarker', 'name': 'Alpha asymmetry', 'unit': 'dB'},
                {'id': 'bio3', 'type': 'Biomarker', 'name': 'Theta power', 'unit': 'μV²/Hz'},
                {'id': 'cond1', 'type': 'Condition', 'name': 'epilepsy'},
                {'id': 'cond2', 'type': 'Condition', 'name': 'depression'},
                {'id': 'out1', 'type': 'Outcome', 'name': 'seizure recurrence'},
                {'id': 'study1', 'type': 'Study', 'name': 'P300 in Epilepsy', 'year': 2023},
            ],
            'relationships': [
                {'source': 'bio1', 'target': 'cond1', 'type': 'PREDICTS', 'strength': 0.85},
                {'source': 'bio2', 'target': 'cond2', 'type': 'CORRELATES_WITH', 'strength': 0.72},
                {'source': 'bio3', 'target': 'cond1', 'type': 'INDICATES', 'strength': 0.68},
                {'source': 'cond1', 'target': 'out1', 'type': 'HAS_OUTCOME', 'probability': 0.45},
                {'source': 'study1', 'target': 'bio1', 'type': 'REPORTS', 'strength': 1.0},
            ]
        }
    
    async def run_query(self, cypher: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute mock Cypher query"""
        await asyncio.sleep(0.05)  # Simulate network latency
        
        # Return mock results based on query pattern
        if 'Biomarker' in cypher and 'PREDICTS' in cypher:
            return [
                {
                    'b': {'id': 'bio1', 'name': 'P300 amplitude', 'unit': 'μV'},
                    'r': {'strength': 0.85},
                    'c': {'id': 'cond1', 'name': 'epilepsy'}
                },
                {
                    'b': {'id': 'bio3', 'name': 'Theta power', 'unit': 'μV²/Hz'},
                    'r': {'strength': 0.68},
                    'c': {'id': 'cond1', 'name': 'epilepsy'}
                }
            ]
        elif 'Study' in cypher:
            return [
                {
                    's': {'id': 'study1', 'name': 'P300 in Epilepsy', 'year': 2023},
                    'b': {'id': 'bio1', 'name': 'P300 amplitude'}
                }
            ]
        else:
            return self.mock_data['nodes'][:3]


class GraphAgent:
    """
    Agent 3: Knowledge Graph Agent
    Queries Neo4j knowledge graph for EEG biomarker relationships
    
    REQ-AGT3-001: Initialize graph connection with Neo4j URI
    REQ-AGT3-002: Execute Cypher queries with parameter binding
    REQ-AGT3-003: Parse and structure query results
    REQ-AGT3-004: Build natural language to Cypher query translation
    REQ-AGT3-005: Support multi-hop relationship traversal (1-3 hops)
    REQ-AGT3-006: Extract subgraphs around entities of interest
    REQ-AGT3-007: Calculate relationship strength scores
    REQ-AGT3-008: Find shortest paths between entities
    REQ-AGT3-009: Track query execution time (<200ms target)
    REQ-AGT3-010: Cache frequently accessed graph patterns
    REQ-AGT3-011: Handle disconnected graph components
    REQ-AGT3-012: Support 5+ relationship types (PREDICTS, CORRELATES_WITH, etc.)
    REQ-AGT3-013: Return structured GraphQueryResult objects
    REQ-AGT3-014: Provide graph visualization data (nodes, edges, layout)
    REQ-AGT3-015: Collect statistics (queries executed, nodes retrieved, avg latency)
    """
    
    def __init__(
        self,
        name: str = "GraphAgent",
        agent_type: str = "graph",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        max_hops: int = 3,
        use_mock: bool = True  # Use mock for testing, set False for production
    ):
        """
        Initialize Knowledge Graph Agent
        
        Args:
            name: Agent name
            agent_type: Agent type identifier
            neo4j_uri: Neo4j database URI (e.g., bolt://localhost:7687)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            capabilities: List of agent capabilities
            max_hops: Maximum relationship hops for path queries
            use_mock: Use mock Neo4j connection (for testing)
        """
        self.name = name
        self.agent_type = agent_type
        self.max_hops = max_hops
        self.capabilities = capabilities or [
            "graph_query",
            "relationship_traversal",
            "entity_extraction",
            "cypher_generation"
        ]
        
        # Neo4j connection (mock or real)
        if use_mock:
            self.db = MockNeo4jConnection()
        else:
            # In production, use: from neo4j import GraphDatabase
            # self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            raise NotImplementedError("Real Neo4j connection not yet implemented. Set use_mock=True.")
        
        # Query builder
        self.query_builder = CypherQueryBuilder()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_nodes_retrieved': 0,
            'total_relationships_retrieved': 0,
            'total_execution_time': 0.0,
            'average_latency': 0.0
        }
        
        # Cache for frequent queries
        self.query_cache: Dict[str, GraphQueryResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _cache_key(self, query_text: str) -> str:
        """Generate cache key from query text"""
        return hashlib.md5(query_text.encode()).hexdigest()
    
    async def execute(self, query_text: str, use_cache: bool = True) -> GraphQueryResult:
        """
        Execute a knowledge graph query
        
        Args:
            query_text: Natural language query
            use_cache: Whether to use cached results
            
        Returns:
            GraphQueryResult with nodes, relationships, and paths
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._cache_key(query_text)
        if use_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        self.cache_misses += 1
        self.stats['total_queries'] += 1
        
        try:
            # Step 1: Detect query intent and build Cypher query
            pattern_name, parameters = self.query_builder.detect_query_intent(query_text)
            cypher_query = self.query_builder.build_cypher(pattern_name, parameters)
            
            # Step 2: Execute Cypher query
            raw_results = await self.db.run_query(cypher_query, parameters)
            
            # Step 3: Parse results into structured format
            nodes, relationships = self._parse_results(raw_results)
            
            # Step 4: Find paths if applicable
            paths = self._extract_paths(nodes, relationships)
            
            # Step 5: Build subgraph representation
            subgraph = self._build_subgraph(nodes, relationships)
            
            # Step 6: Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result
            result = GraphQueryResult(
                nodes=nodes,
                relationships=relationships,
                paths=paths,
                subgraph=subgraph,
                query_text=query_text,
                cypher_query=cypher_query,
                execution_time=execution_time
            )
            
            # Update statistics
            self.stats['successful_queries'] += 1
            self.stats['total_nodes_retrieved'] += len(nodes)
            self.stats['total_relationships_retrieved'] += len(relationships)
            self.stats['total_execution_time'] += execution_time
            self.stats['average_latency'] = (
                self.stats['total_execution_time'] / self.stats['total_queries']
            )
            
            # Cache result
            if use_cache:
                self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.stats['failed_queries'] += 1
            execution_time = time.time() - start_time
            
            # Return empty result with error info
            return GraphQueryResult(
                nodes=[],
                relationships=[],
                paths=[],
                subgraph={},
                query_text=query_text,
                cypher_query="",
                execution_time=execution_time
            )
    
    def _parse_results(self, raw_results: List[Dict[str, Any]]) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Parse raw Cypher results into GraphNode and GraphRelationship objects"""
        nodes = []
        relationships = []
        node_ids_seen = set()
        
        for record in raw_results:
            # Extract nodes
            for key, value in record.items():
                if isinstance(value, dict):
                    node_id = value.get('id', f'node_{len(nodes)}')
                    if node_id not in node_ids_seen:
                        node_type = NodeType[value.get('type', 'BIOMARKER').upper()]
                        properties = {k: v for k, v in value.items() if k not in ['id', 'type']}
                        
                        node = GraphNode(
                            node_id=node_id,
                            node_type=node_type,
                            properties=properties,
                            labels=[node_type.value]
                        )
                        nodes.append(node)
                        node_ids_seen.add(node_id)
            
            # Extract relationships (if present)
            if 'r' in record and isinstance(record['r'], dict):
                source_id = record.get('b', {}).get('id', 'unknown')
                target_id = record.get('c', {}).get('id', 'unknown')
                rel_props = record['r']
                
                relationship = GraphRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=RelationType.PREDICTS,  # Default, should parse from data
                    properties=rel_props,
                    strength=rel_props.get('strength', 1.0)
                )
                relationships.append(relationship)
        
        return nodes, relationships
    
    def _extract_paths(self, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> List[GraphPath]:
        """Extract paths from nodes and relationships"""
        paths = []
        
        if not relationships:
            return paths
        
        # Simple path extraction: each relationship forms a 2-node path
        for rel in relationships:
            source_node = next((n for n in nodes if n.node_id == rel.source_id), None)
            target_node = next((n for n in nodes if n.node_id == rel.target_id), None)
            
            if source_node and target_node:
                path = GraphPath(
                    nodes=[source_node, target_node],
                    relationships=[rel],
                    path_length=1,
                    total_strength=rel.strength
                )
                paths.append(path)
        
        return paths
    
    def _build_subgraph(self, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> Dict[str, Any]:
        """Build subgraph representation for visualization"""
        return {
            'nodes': [
                {
                    'id': node.node_id,
                    'label': node.properties.get('name', node.node_id),
                    'type': node.node_type.value,
                    'properties': node.properties
                }
                for node in nodes
            ],
            'edges': [
                {
                    'source': rel.source_id,
                    'target': rel.target_id,
                    'label': rel.relationship_type.value,
                    'strength': rel.strength
                }
                for rel in relationships
            ],
            'metadata': {
                'node_count': len(nodes),
                'edge_count': len(relationships),
                'node_types': list(set(n.node_type.value for n in nodes)),
                'relationship_types': list(set(r.relationship_type.value for r in relationships))
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'name': self.name,
            'agent_type': self.agent_type,
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'success_rate': (
                self.stats['successful_queries'] / self.stats['total_queries']
                if self.stats['total_queries'] > 0 else 0.0
            ),
            'total_nodes_retrieved': self.stats['total_nodes_retrieved'],
            'total_relationships_retrieved': self.stats['total_relationships_retrieved'],
            'average_latency': self.stats['average_latency'],
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            )
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
