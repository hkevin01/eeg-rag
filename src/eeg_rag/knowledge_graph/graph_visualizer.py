#!/usr/bin/env python3
"""
Knowledge Graph Visualization for EEG-RAG

Provides interactive visualization of the EEG knowledge graph including:
- Node and relationship exploration
- Research paper networks
- Citation graphs
- Entity relationship maps
- Temporal analysis views
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import colorsys
import math

# Import graph interface
from .graph_interface import Neo4jInterface, NodeType, RelationType

logger = logging.getLogger(__name__)


@dataclass
class VisualizationNode:
    """Node for visualization."""
    id: str
    label: str
    node_type: str
    size: float
    color: str
    properties: Dict[str, Any]
    

@dataclass
class VisualizationEdge:
    """Edge for visualization."""
    source: str
    target: str
    label: str
    edge_type: str
    weight: float
    color: str
    properties: Dict[str, Any]
    

@dataclass
class VisualizationData:
    """Complete visualization data."""
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any]
    

class GraphVisualizer:
    """Interactive knowledge graph visualizer."""
    
    def __init__(self, neo4j_interface: Neo4jInterface):
        """Initialize graph visualizer.
        
        Args:
            neo4j_interface: Neo4j database interface.
        """
        self.neo4j = neo4j_interface
        
        # Color schemes for different node types
        self.node_colors = {
            NodeType.PAPER.value: '#3498db',
            NodeType.AUTHOR.value: '#e74c3c',
            NodeType.JOURNAL.value: '#f39c12',
            NodeType.ELECTRODE.value: '#9b59b6',
            NodeType.FREQUENCY_BAND.value: '#2ecc71',
            NodeType.ERP_COMPONENT.value: '#1abc9c',
            NodeType.CONDITION.value: '#e67e22',
            NodeType.BRAIN_REGION.value: '#34495e',
            NodeType.METHODOLOGY.value: '#95a5a6',
            NodeType.DEVICE.value: '#f1c40f',
            NodeType.MEASUREMENT.value: '#8e44ad',
            NodeType.MESH_TERM.value: '#16a085'
        }
        
        # Edge colors for different relationship types
        self.edge_colors = {
            RelationType.AUTHORED.value: '#bdc3c7',
            RelationType.CITES.value: '#3498db',
            RelationType.PUBLISHED_IN.value: '#f39c12',
            RelationType.STUDIES.value: '#e74c3c',
            RelationType.USES_METHOD.value: '#2ecc71',
            RelationType.MENTIONS.value: '#9b59b6',
            RelationType.CLASSIFIED_AS.value: '#34495e'
        }
    
    async def create_research_network_visualization(
        self,
        query_terms: List[str],
        max_nodes: int = 100,
        max_depth: int = 2
    ) -> VisualizationData:
        """Create visualization for research network around query terms.
        
        Args:
            query_terms: Terms to center the visualization around.
            max_nodes: Maximum number of nodes to include.
            max_depth: Maximum relationship depth to explore.
            
        Returns:
            Visualization data ready for rendering.
        """
        logger.info(f"Creating research network visualization for terms: {query_terms}")
        
        # Find papers related to query terms
        papers = await self._find_papers_by_terms(query_terms, max_nodes // 2)
        
        # Get connected entities
        entities = await self._get_connected_entities(papers, max_depth)
        
        # Build visualization
        nodes, edges = await self._build_network_visualization(papers + entities)
        
        # Calculate layout properties
        self._calculate_layout_properties(nodes, edges)
        
        metadata = {
            'title': f"Research Network: {', '.join(query_terms)}",
            'node_count': len(nodes),
            'edge_count': len(edges),
            'query_terms': query_terms,
            'max_depth': max_depth
        }
        
        return VisualizationData(nodes, edges, metadata)
    
    async def create_citation_network_visualization(
        self,
        seed_pmids: List[str],
        max_depth: int = 2
    ) -> VisualizationData:
        """Create citation network visualization.
        
        Args:
            seed_pmids: PMIDs to start citation network from.
            max_depth: Maximum citation depth to explore.
            
        Returns:
            Visualization data for citation network.
        """
        logger.info(f"Creating citation network for PMIDs: {seed_pmids}")
        
        # Get citation network
        citation_nodes, citation_edges = await self._build_citation_network(
            seed_pmids, max_depth
        )
        
        # Convert to visualization format
        vis_nodes = []
        vis_edges = []
        
        # Process nodes
        for node_id, node_data in citation_nodes.items():
            size = self._calculate_citation_node_size(node_data)
            color = self._get_citation_node_color(node_data)
            
            vis_nodes.append(VisualizationNode(
                id=node_id,
                label=node_data.get('title', node_data.get('pmid', 'Unknown')),
                node_type='paper',
                size=size,
                color=color,
                properties=node_data
            ))
        
        # Process edges
        for edge in citation_edges:
            vis_edges.append(VisualizationEdge(
                source=edge['source'],
                target=edge['target'],
                label='cites',
                edge_type='citation',
                weight=1.0,
                color=self.edge_colors.get('cites', '#95a5a6'),
                properties=edge.get('properties', {})
            ))
        
        metadata = {
            'title': 'Citation Network',
            'node_count': len(vis_nodes),
            'edge_count': len(vis_edges),
            'seed_pmids': seed_pmids,
            'max_depth': max_depth
        }
        
        return VisualizationData(vis_nodes, vis_edges, metadata)
    
    async def create_entity_relationship_map(
        self,
        entity_types: List[str],
        min_connection_strength: float = 0.1
    ) -> VisualizationData:
        """Create entity relationship map.
        
        Args:
            entity_types: Types of entities to include.
            min_connection_strength: Minimum connection strength to include.
            
        Returns:
            Visualization data for entity relationships.
        """
        logger.info(f"Creating entity relationship map for types: {entity_types}")
        
        # Get entities and their relationships
        entities = await self._get_entities_by_types(entity_types)
        relationships = await self._get_entity_relationships(
            entities, min_connection_strength
        )
        
        # Build visualization
        vis_nodes = []
        vis_edges = []
        
        # Process entities
        for entity in entities:
            node_type = entity.get('type', 'unknown')
            size = self._calculate_entity_node_size(entity)
            color = self.node_colors.get(node_type, '#95a5a6')
            
            vis_nodes.append(VisualizationNode(
                id=entity['id'],
                label=entity.get('name', 'Unknown'),
                node_type=node_type,
                size=size,
                color=color,
                properties=entity
            ))
        
        # Process relationships
        for rel in relationships:
            weight = rel.get('strength', 1.0)
            edge_type = rel.get('type', 'unknown')
            
            vis_edges.append(VisualizationEdge(
                source=rel['source'],
                target=rel['target'],
                label=edge_type,
                edge_type=edge_type,
                weight=weight,
                color=self.edge_colors.get(edge_type, '#95a5a6'),
                properties=rel
            ))
        
        metadata = {
            'title': 'Entity Relationship Map',
            'node_count': len(vis_nodes),
            'edge_count': len(vis_edges),
            'entity_types': entity_types,
            'min_connection_strength': min_connection_strength
        }
        
        return VisualizationData(vis_nodes, vis_edges, metadata)
    
    async def create_temporal_analysis_view(
        self,
        start_year: int,
        end_year: int,
        time_granularity: str = 'year'
    ) -> Dict[str, Any]:
        """Create temporal analysis visualization data.
        
        Args:
            start_year: Start year for analysis.
            end_year: End year for analysis.
            time_granularity: Time granularity ('year', 'month').
            
        Returns:
            Temporal analysis data.
        """
        logger.info(f"Creating temporal analysis from {start_year} to {end_year}")
        
        # Get temporal data
        temporal_data = await self._get_temporal_research_data(
            start_year, end_year, time_granularity
        )
        
        # Process for visualization
        timeline_data = self._process_temporal_data(temporal_data, time_granularity)
        
        return {
            'timeline': timeline_data,
            'metadata': {
                'title': f'EEG Research Timeline ({start_year}-{end_year})',
                'start_year': start_year,
                'end_year': end_year,
                'granularity': time_granularity,
                'total_papers': sum(period['paper_count'] for period in timeline_data)
            }
        }
    
    async def _find_papers_by_terms(
        self,
        query_terms: List[str],
        max_papers: int
    ) -> List[Dict[str, Any]]:
        """Find papers containing query terms."""
        # Build Cypher query to find papers
        term_conditions = []
        for term in query_terms:
            term_conditions.append(
                f"(p.title CONTAINS '{term}' OR p.abstract CONTAINS '{term}')"
            )
        
        cypher_query = f"""
        MATCH (p:Paper)
        WHERE {' OR '.join(term_conditions)}
        RETURN p
        LIMIT {max_papers}
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        papers = []
        for record in result:
            paper_data = dict(record['p'])
            paper_data['id'] = record['p'].element_id
            paper_data['type'] = 'paper'
            papers.append(paper_data)
        
        return papers
    
    async def _get_connected_entities(
        self,
        papers: List[Dict[str, Any]],
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Get entities connected to papers."""
        if not papers:
            return []
        
        paper_ids = [paper['id'] for paper in papers]
        
        # Build query to get connected entities
        cypher_query = f"""
        MATCH (p:Paper)-[r*1..{max_depth}]-(e)
        WHERE p.element_id IN {paper_ids}
        AND NOT e:Paper
        RETURN DISTINCT e, labels(e)[0] as entity_type
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        entities = []
        for record in result:
            entity_data = dict(record['e'])
            entity_data['id'] = record['e'].element_id
            entity_data['type'] = record['entity_type'].lower()
            entities.append(entity_data)
        
        return entities
    
    async def _build_network_visualization(
        self,
        all_nodes: List[Dict[str, Any]]
    ) -> Tuple[List[VisualizationNode], List[VisualizationEdge]]:
        """Build network visualization from nodes."""
        if not all_nodes:
            return [], []
        
        node_ids = [node['id'] for node in all_nodes]
        
        # Get relationships between nodes
        cypher_query = f"""
        MATCH (a)-[r]->(b)
        WHERE a.element_id IN {node_ids} AND b.element_id IN {node_ids}
        RETURN a.element_id as source, b.element_id as target, 
               type(r) as rel_type, properties(r) as props
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        # Build visualization nodes
        vis_nodes = []
        for node in all_nodes:
            node_type = node.get('type', 'unknown')
            size = self._calculate_node_size(node)
            color = self.node_colors.get(node_type, '#95a5a6')
            label = node.get('name', node.get('title', 'Unknown'))
            
            vis_nodes.append(VisualizationNode(
                id=node['id'],
                label=label[:50] + '...' if len(label) > 50 else label,
                node_type=node_type,
                size=size,
                color=color,
                properties=node
            ))
        
        # Build visualization edges
        vis_edges = []
        for record in result:
            rel_type = record['rel_type'].lower()
            color = self.edge_colors.get(rel_type, '#95a5a6')
            
            vis_edges.append(VisualizationEdge(
                source=record['source'],
                target=record['target'],
                label=rel_type.replace('_', ' '),
                edge_type=rel_type,
                weight=1.0,
                color=color,
                properties=record['props'] or {}
            ))
        
        return vis_nodes, vis_edges
    
    async def _build_citation_network(
        self,
        seed_pmids: List[str],
        max_depth: int
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """Build citation network from seed PMIDs."""
        citation_nodes = {}
        citation_edges = []
        visited_pmids = set()
        
        # BFS to build citation network
        current_level = seed_pmids.copy()
        
        for depth in range(max_depth + 1):
            if not current_level:
                break
            
            next_level = set()
            
            # Get papers for current level
            pmid_conditions = [f"p.pmid = '{pmid}'" for pmid in current_level]
            
            cypher_query = f"""
            MATCH (p:Paper)
            WHERE {' OR '.join(pmid_conditions)}
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            RETURN p, collect(cited.pmid) as cited_pmids, 
                   collect(citing.pmid) as citing_pmids
            """
            
            result = await self.neo4j.execute_query(cypher_query)
            
            for record in result:
                paper = record['p']
                pmid = paper['pmid']
                
                if pmid not in visited_pmids:
                    visited_pmids.add(pmid)
                    paper_data = dict(paper)
                    paper_data['depth'] = depth
                    citation_nodes[pmid] = paper_data
                
                # Add citation edges
                for cited_pmid in record['cited_pmids']:
                    if cited_pmid:
                        citation_edges.append({
                            'source': pmid,
                            'target': cited_pmid,
                            'type': 'cites'
                        })
                        if depth < max_depth:
                            next_level.add(cited_pmid)
                
                for citing_pmid in record['citing_pmids']:
                    if citing_pmid:
                        citation_edges.append({
                            'source': citing_pmid,
                            'target': pmid,
                            'type': 'cites'
                        })
                        if depth < max_depth:
                            next_level.add(citing_pmid)
            
            current_level = list(next_level - visited_pmids)
        
        return citation_nodes, citation_edges
    
    async def _get_entities_by_types(
        self,
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Get entities by their types."""
        type_conditions = [f"'{etype.title()}' IN labels(e)" for etype in entity_types]
        
        cypher_query = f"""
        MATCH (e)
        WHERE {' OR '.join(type_conditions)}
        RETURN e, labels(e)[0] as entity_type
        LIMIT 1000
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        entities = []
        for record in result:
            entity_data = dict(record['e'])
            entity_data['id'] = record['e'].element_id
            entity_data['type'] = record['entity_type'].lower()
            entities.append(entity_data)
        
        return entities
    
    async def _get_entity_relationships(
        self,
        entities: List[Dict[str, Any]],
        min_strength: float
    ) -> List[Dict[str, Any]]:
        """Get relationships between entities."""
        if not entities:
            return []
        
        entity_ids = [entity['id'] for entity in entities]
        
        # Count co-occurrences in papers to determine relationship strength
        cypher_query = f"""
        MATCH (e1)-[:MENTIONED_IN|STUDIES|USES_METHOD]->(p:Paper)<-[:MENTIONED_IN|STUDIES|USES_METHOD]-(e2)
        WHERE e1.element_id IN {entity_ids} AND e2.element_id IN {entity_ids}
        AND e1 <> e2
        WITH e1, e2, count(p) as co_occurrence_count
        WHERE co_occurrence_count >= {int(min_strength * 10)}
        RETURN e1.element_id as source, e2.element_id as target, 
               co_occurrence_count,
               (co_occurrence_count * 1.0 / 10) as strength
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        relationships = []
        for record in result:
            relationships.append({
                'source': record['source'],
                'target': record['target'],
                'type': 'co_occurs',
                'strength': record['strength'],
                'co_occurrence_count': record['co_occurrence_count']
            })
        
        return relationships
    
    async def _get_temporal_research_data(
        self,
        start_year: int,
        end_year: int,
        granularity: str
    ) -> List[Dict[str, Any]]:
        """Get temporal research data."""
        cypher_query = f"""
        MATCH (p:Paper)
        WHERE p.year >= {start_year} AND p.year <= {end_year}
        RETURN p.year as year, count(p) as paper_count,
               collect(p.pmid)[0..10] as sample_pmids
        ORDER BY year
        """
        
        result = await self.neo4j.execute_query(cypher_query)
        
        temporal_data = []
        for record in result:
            temporal_data.append({
                'year': record['year'],
                'paper_count': record['paper_count'],
                'sample_pmids': record['sample_pmids']
            })
        
        return temporal_data
    
    def _process_temporal_data(
        self,
        temporal_data: List[Dict[str, Any]],
        granularity: str
    ) -> List[Dict[str, Any]]:
        """Process temporal data for visualization."""
        timeline_data = []
        
        for data_point in temporal_data:
            timeline_data.append({
                'period': str(data_point['year']),
                'paper_count': data_point['paper_count'],
                'sample_pmids': data_point['sample_pmids'],
                'normalized_count': data_point['paper_count'] / max(1, max(
                    dp['paper_count'] for dp in temporal_data
                ))
            })
        
        return timeline_data
    
    def _calculate_layout_properties(self, nodes: List[VisualizationNode], edges: List[VisualizationEdge]):
        """Calculate layout properties for nodes."""
        # Simple force-directed layout principles
        node_degrees = {}
        
        # Calculate node degrees
        for node in nodes:
            node_degrees[node.id] = 0
        
        for edge in edges:
            if edge.source in node_degrees:
                node_degrees[edge.source] += 1
            if edge.target in node_degrees:
                node_degrees[edge.target] += 1
        
        # Adjust node sizes based on connectivity
        max_degree = max(node_degrees.values()) if node_degrees.values() else 1
        
        for node in nodes:
            degree = node_degrees.get(node.id, 0)
            connectivity_factor = 1 + (degree / max_degree) * 0.5
            node.size *= connectivity_factor
    
    def _calculate_node_size(self, node: Dict[str, Any]) -> float:
        """Calculate node size based on properties."""
        base_size = 10.0
        
        # Size based on node type
        node_type = node.get('type', 'unknown')
        
        if node_type == 'paper':
            # Size based on citation count or year
            year = node.get('year', 2023)
            recency_factor = 1 + (year - 2000) / 23 * 0.3  # More recent = slightly larger
            return base_size * recency_factor
        
        elif node_type in ['author', 'journal']:
            return base_size * 1.2
        
        else:
            return base_size * 0.8
    
    def _calculate_citation_node_size(self, node_data: Dict[str, Any]) -> float:
        """Calculate node size for citation network."""
        base_size = 15.0
        depth = node_data.get('depth', 0)
        
        # Larger for seed nodes, smaller for distant nodes
        depth_factor = max(0.5, 1.0 - (depth * 0.2))
        
        return base_size * depth_factor
    
    def _get_citation_node_color(self, node_data: Dict[str, Any]) -> str:
        """Get color for citation network node based on depth."""
        depth = node_data.get('depth', 0)
        
        # Color gradient from blue (seed) to light gray (distant)
        if depth == 0:
            return '#2c3e50'  # Dark blue for seed
        elif depth == 1:
            return '#3498db'  # Blue for direct citations
        elif depth == 2:
            return '#85c1e9'  # Light blue
        else:
            return '#d5dbdb'  # Gray for distant
    
    def _calculate_entity_node_size(self, entity: Dict[str, Any]) -> float:
        """Calculate entity node size."""
        base_size = 12.0
        
        # Size based on how often the entity appears
        # This would require additional query to count occurrences
        # For now, use base size with slight variation
        
        entity_type = entity.get('type', 'unknown')
        
        if entity_type in ['frequency_band', 'brain_region']:
            return base_size * 1.3
        elif entity_type in ['condition', 'methodology']:
            return base_size * 1.1
        else:
            return base_size
    
    def export_to_json(self, visualization_data: VisualizationData, output_path: Path):
        """Export visualization data to JSON file."""
        export_data = {
            'nodes': [asdict(node) for node in visualization_data.nodes],
            'edges': [asdict(edge) for edge in visualization_data.edges],
            'metadata': visualization_data.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Visualization data exported to {output_path}")
    
    def export_to_cytoscape(
        self,
        visualization_data: VisualizationData,
        output_path: Path
    ):
        """Export visualization data in Cytoscape.js format."""
        cytoscape_data = {
            'elements': {
                'nodes': [
                    {
                        'data': {
                            'id': node.id,
                            'label': node.label,
                            'type': node.node_type
                        },
                        'style': {
                            'background-color': node.color,
                            'width': node.size,
                            'height': node.size
                        }
                    }
                    for node in visualization_data.nodes
                ],
                'edges': [
                    {
                        'data': {
                            'id': f"{edge.source}_{edge.target}",
                            'source': edge.source,
                            'target': edge.target,
                            'label': edge.label,
                            'type': edge.edge_type,
                            'weight': edge.weight
                        },
                        'style': {
                            'line-color': edge.color,
                            'width': edge.weight * 2
                        }
                    }
                    for edge in visualization_data.edges
                ]
            },
            'layout': {
                'name': 'cola',
                'animate': True,
                'randomize': False
            },
            'metadata': visualization_data.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cytoscape_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cytoscape visualization exported to {output_path}")