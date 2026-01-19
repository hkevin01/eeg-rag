#!/usr/bin/env python3
"""
Knowledge Graph Population Pipeline

Populates the Neo4j knowledge graph with:
- EEG research papers and metadata
- Citation relationships
- Biomarker-condition associations
- Author and institution networks
- Temporal relationships
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import re

from ..utils.logging_utils import log_time, setup_logging
from ..nlp.ner_eeg import EEGEntityRecognizer, Entity, EntityType
from .graph_interface import Neo4jInterface, NodeType, RelationType
from ..verification.citation_verifier import CitationVerifier

logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """Statistics from graph population."""
    papers_processed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    entities_extracted: int = 0
    citations_resolved: int = 0
    errors: int = 0
    processing_time_ms: float = 0.0


@dataclass
class PaperData:
    """Structured paper data for graph population."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str] = None
    mesh_terms: List[str] = None
    citations: List[str] = None
    full_text: Optional[str] = None


class GraphPopulator:
    """Populates EEG knowledge graph with research data."""
    
    def __init__(
        self,
        neo4j_interface: Neo4jInterface,
        batch_size: int = 100,
        enable_ner: bool = True,
        enable_citation_resolution: bool = True
    ):
        """Initialize graph populator.
        
        Args:
            neo4j_interface: Neo4j database interface.
            batch_size: Batch size for processing papers.
            enable_ner: Whether to extract named entities.
            enable_citation_resolution: Whether to resolve citations.
        """
        self.neo4j = neo4j_interface
        self.batch_size = batch_size
        self.enable_ner = enable_ner
        self.enable_citation_resolution = enable_citation_resolution
        
        # Initialize NER if enabled
        if enable_ner:
            self.ner = EEGEntityRecognizer(use_spacy=True)
        else:
            self.ner = None
        
        # Initialize citation verifier if enabled
        if enable_citation_resolution:
            self.citation_verifier = CitationVerifier(
                enable_medical_validation=True
            )
        else:
            self.citation_verifier = None
        
        # Statistics
        self.stats = PopulationStats()
        
        # Entity caches for deduplication
        self.entity_cache: Dict[str, str] = {}  # normalized_name -> node_id
        self.author_cache: Dict[str, str] = {}
        self.journal_cache: Dict[str, str] = {}
        
        # Relationship tracking
        self.processed_relationships: Set[Tuple[str, str, str]] = set()
    
    async def populate_from_corpus(
        self,
        corpus_path: Path,
        limit: Optional[int] = None
    ) -> PopulationStats:
        """Populate graph from EEG corpus.
        
        Args:
            corpus_path: Path to corpus JSONL file.
            limit: Optional limit on number of papers to process.
            
        Returns:
            Population statistics.
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting graph population from {corpus_path}")
        
        try:
            papers_data = self._load_corpus_data(corpus_path, limit)
            
            # Process papers in batches
            for i in range(0, len(papers_data), self.batch_size):
                batch = papers_data[i:i + self.batch_size]
                await self._process_paper_batch(batch)
                
                logger.info(
                    f"Processed batch {i//self.batch_size + 1}/{(len(papers_data)-1)//self.batch_size + 1} "
                    f"({len(batch)} papers)"
                )
            
            # Create cross-paper relationships
            await self._create_cross_paper_relationships(papers_data)
            
            # Calculate final stats
            end_time = asyncio.get_event_loop().time()
            self.stats.processing_time_ms = (end_time - start_time) * 1000
            
            logger.info(
                f"Graph population completed. "
                f"Papers: {self.stats.papers_processed}, "
                f"Nodes: {self.stats.nodes_created}, "
                f"Relationships: {self.stats.relationships_created}"
            )
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Graph population failed: {str(e)}")
            self.stats.errors += 1
            raise
    
    def _load_corpus_data(
        self,
        corpus_path: Path,
        limit: Optional[int] = None
    ) -> List[PaperData]:
        """Load and parse corpus data."""
        papers = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    paper = PaperData(
                        pmid=data.get('pmid', str(i)),
                        title=data.get('title', ''),
                        abstract=data.get('abstract', ''),
                        authors=data.get('authors', []),
                        journal=data.get('journal', ''),
                        year=data.get('year', 2023),
                        doi=data.get('doi'),
                        mesh_terms=data.get('mesh_terms', []),
                        citations=data.get('citations', []),
                        full_text=data.get('full_text')
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse paper at line {i+1}: {str(e)}")
                    self.stats.errors += 1
        
        logger.info(f"Loaded {len(papers)} papers from corpus")
        return papers
    
    async def _process_paper_batch(self, papers: List[PaperData]):
        """Process a batch of papers."""
        for paper in papers:
            try:
                await self._process_single_paper(paper)
                self.stats.papers_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process paper {paper.pmid}: {str(e)}")
                self.stats.errors += 1
    
    async def _process_single_paper(self, paper: PaperData):
        """Process a single paper and create graph nodes/relationships."""
        with log_time(logger, f"Processing paper {paper.pmid}"):
            # 1. Create paper node
            paper_node_id = await self._create_paper_node(paper)
            
            # 2. Create author nodes and relationships
            await self._create_author_relationships(paper, paper_node_id)
            
            # 3. Create journal node and relationship
            await self._create_journal_relationship(paper, paper_node_id)
            
            # 4. Extract and create entity nodes
            if self.enable_ner:
                await self._extract_and_create_entities(paper, paper_node_id)
            
            # 5. Create MeSH term relationships
            await self._create_mesh_relationships(paper, paper_node_id)
            
            # 6. Process citations
            if self.enable_citation_resolution:
                await self._process_citations(paper, paper_node_id)
    
    async def _create_paper_node(self, paper: PaperData) -> str:
        """Create paper node in graph."""
        properties = {
            'pmid': paper.pmid,
            'title': paper.title,
            'abstract': paper.abstract,
            'journal': paper.journal,
            'year': paper.year,
            'created_at': datetime.now().isoformat()
        }
        
        if paper.doi:
            properties['doi'] = paper.doi
        
        node_id = await self.neo4j.create_node(NodeType.PAPER, properties)
        self.stats.nodes_created += 1
        
        return node_id
    
    async def _create_author_relationships(self, paper: PaperData, paper_node_id: str):
        """Create author nodes and authorship relationships."""
        for i, author_name in enumerate(paper.authors):
            # Normalize author name
            normalized_name = self._normalize_author_name(author_name)
            
            # Get or create author node
            if normalized_name in self.author_cache:
                author_node_id = self.author_cache[normalized_name]
            else:
                author_properties = {
                    'name': author_name,
                    'normalized_name': normalized_name
                }
                author_node_id = await self.neo4j.create_node(
                    NodeType.AUTHOR, author_properties
                )
                self.author_cache[normalized_name] = author_node_id
                self.stats.nodes_created += 1
            
            # Create authorship relationship
            relationship_properties = {
                'position': i + 1,
                'is_first_author': i == 0,
                'is_last_author': i == len(paper.authors) - 1
            }
            
            await self.neo4j.create_relationship(
                author_node_id,
                paper_node_id,
                RelationType.AUTHORED,
                relationship_properties
            )
            self.stats.relationships_created += 1
    
    async def _create_journal_relationship(self, paper: PaperData, paper_node_id: str):
        """Create journal node and publication relationship."""
        if not paper.journal:
            return
        
        normalized_journal = paper.journal.strip().lower()
        
        # Get or create journal node
        if normalized_journal in self.journal_cache:
            journal_node_id = self.journal_cache[normalized_journal]
        else:
            journal_properties = {
                'name': paper.journal,
                'normalized_name': normalized_journal
            }
            journal_node_id = await self.neo4j.create_node(
                NodeType.JOURNAL, journal_properties
            )
            self.journal_cache[normalized_journal] = journal_node_id
            self.stats.nodes_created += 1
        
        # Create publication relationship
        relationship_properties = {'year': paper.year}
        
        await self.neo4j.create_relationship(
            paper_node_id,
            journal_node_id,
            RelationType.PUBLISHED_IN,
            relationship_properties
        )
        self.stats.relationships_created += 1
    
    async def _extract_and_create_entities(
        self,
        paper: PaperData,
        paper_node_id: str
    ):
        """Extract entities and create nodes/relationships."""
        # Combine title and abstract for entity extraction
        text_content = f"{paper.title}. {paper.abstract}"
        if paper.full_text:
            text_content += f" {paper.full_text[:2000]}"  # Limit full text
        
        # Extract entities
        entities = self.ner.extract_entities(text_content)
        self.stats.entities_extracted += len(entities)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Create nodes for each entity type
        for entity_type, entity_list in entities_by_type.items():
            await self._create_entity_nodes(
                entity_type, entity_list, paper_node_id
            )
    
    async def _create_entity_nodes(
        self,
        entity_type: EntityType,
        entities: List[Entity],
        paper_node_id: str
    ):
        """Create entity nodes for a specific type."""
        # Map EntityType to NodeType
        entity_to_node_mapping = {
            EntityType.ELECTRODE: NodeType.ELECTRODE,
            EntityType.FREQUENCY_BAND: NodeType.FREQUENCY_BAND,
            EntityType.ERP_COMPONENT: NodeType.ERP_COMPONENT,
            EntityType.CLINICAL_CONDITION: NodeType.CONDITION,
            EntityType.BRAIN_REGION: NodeType.BRAIN_REGION,
            EntityType.METHODOLOGY: NodeType.METHODOLOGY,
            EntityType.DEVICE: NodeType.DEVICE,
            EntityType.MEASUREMENT: NodeType.MEASUREMENT
        }
        
        node_type = entity_to_node_mapping.get(entity_type)
        if not node_type:
            return
        
        for entity in entities:
            # Use normalized form for deduplication
            normalized_name = entity.normalized_form or entity.text.lower().strip()
            cache_key = f"{entity_type.value}:{normalized_name}"
            
            # Get or create entity node
            if cache_key in self.entity_cache:
                entity_node_id = self.entity_cache[cache_key]
            else:
                entity_properties = {
                    'name': entity.text,
                    'normalized_name': normalized_name,
                    'confidence': entity.confidence
                }
                
                if entity.metadata:
                    entity_properties.update(entity.metadata)
                
                entity_node_id = await self.neo4j.create_node(
                    node_type, entity_properties
                )
                self.entity_cache[cache_key] = entity_node_id
                self.stats.nodes_created += 1
            
            # Create relationship based on entity type
            if entity_type == EntityType.CLINICAL_CONDITION:
                relation_type = RelationType.STUDIES
            elif entity_type == EntityType.METHODOLOGY:
                relation_type = RelationType.USES_METHOD
            else:
                relation_type = RelationType.MENTIONS
            
            relationship_key = (paper_node_id, entity_node_id, relation_type.value)
            if relationship_key not in self.processed_relationships:
                await self.neo4j.create_relationship(
                    paper_node_id,
                    entity_node_id,
                    relation_type,
                    {'confidence': entity.confidence}
                )
                self.processed_relationships.add(relationship_key)
                self.stats.relationships_created += 1
    
    async def _create_mesh_relationships(self, paper: PaperData, paper_node_id: str):
        """Create MeSH term nodes and relationships."""
        if not paper.mesh_terms:
            return
        
        for mesh_term in paper.mesh_terms:
            normalized_mesh = mesh_term.strip().lower()
            cache_key = f"mesh:{normalized_mesh}"
            
            # Get or create MeSH node
            if cache_key in self.entity_cache:
                mesh_node_id = self.entity_cache[cache_key]
            else:
                mesh_properties = {
                    'name': mesh_term,
                    'normalized_name': normalized_mesh
                }
                mesh_node_id = await self.neo4j.create_node(
                    NodeType.MESH_TERM, mesh_properties
                )
                self.entity_cache[cache_key] = mesh_node_id
                self.stats.nodes_created += 1
            
            # Create classification relationship
            await self.neo4j.create_relationship(
                paper_node_id,
                mesh_node_id,
                RelationType.CLASSIFIED_AS,
                {}
            )
            self.stats.relationships_created += 1
    
    async def _process_citations(self, paper: PaperData, paper_node_id: str):
        """Process and create citation relationships."""
        if not paper.citations:
            return
        
        for cited_pmid in paper.citations:
            # Validate PMID format
            if not re.match(r'^\d{7,8}$', cited_pmid.strip()):
                continue
            
            # Create cited paper node (will be placeholder until populated)
            cited_paper_properties = {
                'pmid': cited_pmid,
                'placeholder': True
            }
            
            cited_paper_id = await self.neo4j.create_node(
                NodeType.PAPER, cited_paper_properties
            )
            self.stats.nodes_created += 1
            
            # Create citation relationship
            await self.neo4j.create_relationship(
                paper_node_id,
                cited_paper_id,
                RelationType.CITES,
                {'pmid': cited_pmid}
            )
            self.stats.relationships_created += 1
            self.stats.citations_resolved += 1
    
    async def _create_cross_paper_relationships(self, papers: List[PaperData]):
        """Create relationships between papers based on shared entities."""
        logger.info("Creating cross-paper relationships...")
        
        # This would involve querying the graph for papers that share
        # significant entities and creating similarity relationships
        # Implementation would depend on specific requirements
        
        # Example: Papers studying the same condition
        # Example: Papers by same authors
        # Example: Papers using same methodology
        
        pass  # Placeholder for now
    
    def _normalize_author_name(self, name: str) -> str:
        """Normalize author name for deduplication."""
        # Remove extra whitespace and standardize format
        name = ' '.join(name.strip().split())
        
        # Convert to "Last, First Middle" format if possible
        if ',' not in name and ' ' in name:
            parts = name.split()
            if len(parts) >= 2:
                name = f"{parts[-1]}, {' '.join(parts[:-1])}"
        
        return name.lower()
    
    async def get_population_progress(self) -> Dict[str, Any]:
        """Get current population progress."""
        return {
            'papers_processed': self.stats.papers_processed,
            'nodes_created': self.stats.nodes_created,
            'relationships_created': self.stats.relationships_created,
            'entities_extracted': self.stats.entities_extracted,
            'citations_resolved': self.stats.citations_resolved,
            'errors': self.stats.errors,
            'cache_sizes': {
                'entities': len(self.entity_cache),
                'authors': len(self.author_cache),
                'journals': len(self.journal_cache)
            }
        }


async def populate_graph_from_corpus(
    corpus_path: Path,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    limit: Optional[int] = None,
    batch_size: int = 100
) -> PopulationStats:
    """Convenience function to populate graph from corpus."""
    neo4j_interface = Neo4jInterface(neo4j_uri, neo4j_user, neo4j_password)
    
    async with neo4j_interface:
        populator = GraphPopulator(
            neo4j_interface,
            batch_size=batch_size,
            enable_ner=True,
            enable_citation_resolution=True
        )
        
        return await populator.populate_from_corpus(corpus_path, limit)