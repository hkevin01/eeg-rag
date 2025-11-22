"""
Test Suite for Agent 3: Knowledge Graph Agent
Tests Neo4j Cypher query generation, graph traversal, and relationship extraction
"""

import pytest
import asyncio
from eeg_rag.agents.graph_agent import (
    GraphAgent,
    GraphNode,
    GraphRelationship,
    GraphPath,
    GraphQueryResult,
    NodeType,
    RelationType,
    CypherQueryBuilder,
    MockNeo4jConnection
)


class TestGraphNode:
    """Tests for GraphNode dataclass"""
    
    def test_node_creation(self):
        """Test creating a graph node"""
        node = GraphNode(
            node_id="bio1",
            node_type=NodeType.BIOMARKER,
            properties={'name': 'P300', 'unit': 'μV'},
            labels=['Biomarker', 'EEG']
        )
        
        assert node.node_id == "bio1"
        assert node.node_type == NodeType.BIOMARKER
        assert node.properties['name'] == 'P300'
        assert 'Biomarker' in node.labels
    
    def test_node_to_dict(self):
        """Test node serialization"""
        node = GraphNode(
            node_id="cond1",
            node_type=NodeType.CONDITION,
            properties={'name': 'epilepsy'},
            labels=['Condition']
        )
        
        result = node.to_dict()
        assert result['id'] == 'cond1'
        assert result['type'] == 'Condition'
        assert result['properties']['name'] == 'epilepsy'


class TestGraphRelationship:
    """Tests for GraphRelationship dataclass"""
    
    def test_relationship_creation(self):
        """Test creating a relationship"""
        rel = GraphRelationship(
            source_id="bio1",
            target_id="cond1",
            relationship_type=RelationType.PREDICTS,
            properties={'evidence': 'strong'},
            strength=0.85
        )
        
        assert rel.source_id == "bio1"
        assert rel.target_id == "cond1"
        assert rel.relationship_type == RelationType.PREDICTS
        assert rel.strength == 0.85
    
    def test_relationship_to_dict(self):
        """Test relationship serialization"""
        rel = GraphRelationship(
            source_id="bio1",
            target_id="cond1",
            relationship_type=RelationType.CORRELATES_WITH,
            strength=0.72
        )
        
        result = rel.to_dict()
        assert result['source'] == 'bio1'
        assert result['target'] == 'cond1'
        assert result['type'] == 'CORRELATES_WITH'
        assert result['strength'] == 0.72


class TestGraphPath:
    """Tests for GraphPath dataclass"""
    
    def test_path_creation(self):
        """Test creating a graph path"""
        node1 = GraphNode("bio1", NodeType.BIOMARKER, {'name': 'P300'})
        node2 = GraphNode("cond1", NodeType.CONDITION, {'name': 'epilepsy'})
        rel = GraphRelationship("bio1", "cond1", RelationType.PREDICTS, strength=0.85)
        
        path = GraphPath(
            nodes=[node1, node2],
            relationships=[rel],
            path_length=1,
            total_strength=0.85
        )
        
        assert len(path.nodes) == 2
        assert len(path.relationships) == 1
        assert path.path_length == 1
        assert path.total_strength == 0.85
    
    def test_path_to_dict(self):
        """Test path serialization"""
        node1 = GraphNode("bio1", NodeType.BIOMARKER, {'name': 'P300'})
        node2 = GraphNode("cond1", NodeType.CONDITION, {'name': 'epilepsy'})
        rel = GraphRelationship("bio1", "cond1", RelationType.PREDICTS, strength=0.85)
        
        path = GraphPath([node1, node2], [rel], 1, 0.85)
        result = path.to_dict()
        
        assert len(result['nodes']) == 2
        assert len(result['relationships']) == 1
        assert result['path_length'] == 1


class TestCypherQueryBuilder:
    """Tests for Cypher query generation"""
    
    def test_detect_biomarker_query(self):
        """Test detecting biomarker search intent"""
        query = "What biomarkers predict epilepsy?"
        pattern, params = CypherQueryBuilder.detect_query_intent(query)
        
        assert pattern == 'find_biomarkers'
        assert params['condition'] == 'epilepsy'
        assert params['limit'] == 10
    
    def test_detect_relationship_query(self):
        """Test detecting relationship query intent"""
        query = "What is P300 related to?"
        pattern, params = CypherQueryBuilder.detect_query_intent(query)
        
        assert pattern == 'biomarker_relationships'
        assert params['biomarker'] == 'P300'
    
    def test_detect_study_query(self):
        """Test detecting study search intent"""
        query = "Find studies about P300"
        pattern, params = CypherQueryBuilder.detect_query_intent(query)
        
        assert pattern == 'related_studies'
        assert 'P300' in params['biomarker']
    
    def test_detect_multi_hop_query(self):
        """Test detecting multi-hop path query"""
        query = "Find connection between P300 and epilepsy"
        pattern, params = CypherQueryBuilder.detect_query_intent(query)
        
        assert pattern == 'multi_hop_path'
        assert 'start_entity' in params
        assert 'end_entity' in params
        assert params['hops'] == 3
    
    def test_build_cypher_query(self):
        """Test building Cypher query from pattern"""
        pattern = 'find_biomarkers'
        params = {'condition': 'epilepsy', 'limit': 10}
        
        cypher = CypherQueryBuilder.build_cypher(pattern, params)
        
        assert 'MATCH' in cypher
        assert 'Biomarker' in cypher
        assert 'PREDICTS' in cypher
        assert 'epilepsy' in cypher
    
    def test_default_pattern(self):
        """Test fallback to default pattern"""
        pattern = 'invalid_pattern'
        params = {'condition': 'test', 'limit': 5}
        
        cypher = CypherQueryBuilder.build_cypher(pattern, params)
        
        # Should fall back to find_biomarkers
        assert 'MATCH' in cypher
        assert 'Biomarker' in cypher


class TestMockNeo4jConnection:
    """Tests for mock Neo4j connection"""
    
    def test_mock_data_creation(self):
        """Test mock database contains data"""
        db = MockNeo4jConnection()
        
        assert len(db.mock_data['nodes']) > 0
        assert len(db.mock_data['relationships']) > 0
        assert any(node['type'] == 'Biomarker' for node in db.mock_data['nodes'])
    
    @pytest.mark.asyncio
    async def test_mock_query_execution(self):
        """Test executing mock Cypher query"""
        db = MockNeo4jConnection()
        
        cypher = "MATCH (b:Biomarker)-[r:PREDICTS]->(c:Condition) RETURN b, r, c"
        results = await db.run_query(cypher, {})
        
        assert len(results) > 0
        assert 'b' in results[0]
    
    @pytest.mark.asyncio
    async def test_mock_study_query(self):
        """Test mock study query"""
        db = MockNeo4jConnection()
        
        cypher = "MATCH (s:Study)-[:REPORTS]->(b:Biomarker) RETURN s, b"
        results = await db.run_query(cypher, {})
        
        assert len(results) > 0
        assert 's' in results[0]
        assert 'b' in results[0]


class TestGraphAgent:
    """Tests for GraphAgent class"""
    
    def test_agent_initialization(self):
        """Test creating GraphAgent"""
        agent = GraphAgent(
            name="TestGraphAgent",
            agent_type="graph",
            use_mock=True
        )
        
        assert agent.name == "TestGraphAgent"
        assert agent.agent_type == "graph"
        assert "graph_query" in agent.capabilities
        assert agent.max_hops == 3
    
    @pytest.mark.asyncio
    async def test_execute_query(self):
        """Test executing a graph query"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("What biomarkers predict epilepsy?")
        
        assert isinstance(result, GraphQueryResult)
        assert result.query_text == "What biomarkers predict epilepsy?"
        assert len(result.nodes) >= 0
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_query_returns_nodes(self):
        """Test query returns graph nodes"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("Find biomarkers for epilepsy")
        
        assert len(result.nodes) > 0
        assert all(isinstance(node, GraphNode) for node in result.nodes)
    
    @pytest.mark.asyncio
    async def test_query_returns_relationships(self):
        """Test query returns relationships"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("What predicts epilepsy?")
        
        if len(result.relationships) > 0:
            assert all(isinstance(rel, GraphRelationship) for rel in result.relationships)
    
    @pytest.mark.asyncio
    async def test_query_caching(self):
        """Test query result caching"""
        agent = GraphAgent(use_mock=True)
        
        query = "Find biomarkers for depression"
        
        # First query - cache miss
        result1 = await agent.execute(query, use_cache=True)
        assert agent.cache_misses == 1
        assert agent.cache_hits == 0
        
        # Second query - cache hit
        result2 = await agent.execute(query, use_cache=True)
        assert agent.cache_hits == 1
        
        # Results should be identical
        assert result1.query_text == result2.query_text
    
    @pytest.mark.asyncio
    async def test_cache_clearing(self):
        """Test clearing the query cache"""
        agent = GraphAgent(use_mock=True)
        
        # Execute and cache a query
        await agent.execute("Test query", use_cache=True)
        assert len(agent.query_cache) > 0
        
        # Clear cache
        agent.clear_cache()
        assert len(agent.query_cache) == 0
        assert agent.cache_hits == 0
        assert agent.cache_misses == 0
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test agent statistics collection"""
        agent = GraphAgent(use_mock=True)
        
        # Execute multiple queries
        await agent.execute("Query 1")
        await agent.execute("Query 2")
        await agent.execute("Query 3")
        
        stats = agent.get_statistics()
        
        assert stats['total_queries'] == 3
        assert stats['successful_queries'] > 0
        assert stats['average_latency'] > 0
        assert 'total_nodes_retrieved' in stats
    
    @pytest.mark.asyncio
    async def test_subgraph_generation(self):
        """Test subgraph visualization data"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("Find biomarkers for epilepsy")
        
        assert 'nodes' in result.subgraph
        assert 'edges' in result.subgraph
        assert 'metadata' in result.subgraph
        
        if len(result.subgraph['nodes']) > 0:
            node = result.subgraph['nodes'][0]
            assert 'id' in node
            assert 'label' in node
            assert 'type' in node
    
    @pytest.mark.asyncio
    async def test_performance_target(self):
        """Test query execution time is under 200ms target"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("Quick query")
        
        # Should be well under 200ms with mock
        assert result.execution_time < 0.2
    
    @pytest.mark.asyncio
    async def test_multiple_queries_statistics(self):
        """Test statistics over multiple queries"""
        agent = GraphAgent(use_mock=True)
        
        queries = [
            "What biomarkers predict epilepsy?",
            "Find studies about P300",
            "What is related to depression?",
            "Show outcomes for epilepsy",
        ]
        
        for query in queries:
            await agent.execute(query)
        
        stats = agent.get_statistics()
        
        assert stats['total_queries'] == len(queries)
        assert stats['success_rate'] > 0.5
        assert stats['total_nodes_retrieved'] > 0
    
    def test_agent_capabilities(self):
        """Test agent has correct capabilities"""
        agent = GraphAgent(use_mock=True)
        
        assert "graph_query" in agent.capabilities
        assert "relationship_traversal" in agent.capabilities
        assert "entity_extraction" in agent.capabilities
        assert "cypher_generation" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_cypher_query_generated(self):
        """Test that Cypher queries are generated"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("Find biomarkers")
        
        assert result.cypher_query != ""
        assert "MATCH" in result.cypher_query or result.cypher_query == ""
    
    @pytest.mark.asyncio
    async def test_path_extraction(self):
        """Test path extraction from graph results"""
        agent = GraphAgent(use_mock=True)
        
        result = await agent.execute("What predicts epilepsy?")
        
        # Paths may or may not exist depending on results
        assert isinstance(result.paths, list)
        if len(result.paths) > 0:
            path = result.paths[0]
            assert isinstance(path, GraphPath)
            assert len(path.nodes) >= 2
