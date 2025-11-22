"""
Context Aggregator - Component 10/12
Merges and ranks results from multiple specialized agents
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib
import re


# REQ-CTX-001: Define data structures for aggregated context
@dataclass
class Citation:
    """Represents a single citation from any source"""
    pmid: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    relevance_score: float = 0.0
    source_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_id(self) -> str:
        """Get unique identifier for deduplication"""
        if self.pmid:
            return f"pmid:{self.pmid}"
        elif self.doi:
            return f"doi:{self.doi}"
        elif self.title:
            # Use title hash for non-PubMed citations
            title_hash = hashlib.md5(self.title.lower().encode()).hexdigest()[:16]
            return f"title:{title_hash}"
        return f"unknown:{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'journal': self.journal,
            'doi': self.doi,
            'url': self.url,
            'abstract': self.abstract,
            'relevance_score': self.relevance_score,
            'source_agents': self.source_agents,
            'metadata': self.metadata
        }


@dataclass
class Entity:
    """Represents an extracted entity (biomarker, condition, method, etc.)"""
    text: str
    entity_type: str  # biomarker, condition, method, brain_region, outcome
    frequency: int = 1
    contexts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # Citation IDs
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'type': self.entity_type,
            'frequency': self.frequency,
            'contexts': self.contexts,
            'citations': self.citations,
            'confidence': self.confidence
        }


@dataclass
class AggregatedContext:
    """Final aggregated context from all agents"""
    query: str
    citations: List[Citation]
    entities: List[Entity]
    total_sources: int
    agent_contributions: Dict[str, int]
    relevance_threshold: float
    timestamp: str
    statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'citations': [c.to_dict() for c in self.citations],
            'entities': [e.to_dict() for e in self.entities],
            'total_sources': self.total_sources,
            'agent_contributions': self.agent_contributions,
            'relevance_threshold': self.relevance_threshold,
            'timestamp': self.timestamp,
            'statistics': self.statistics
        }


# REQ-CTX-002: Implement context aggregation class
class ContextAggregator:
    """
    Aggregates and ranks results from multiple specialized agents
    
    Requirements:
    - REQ-CTX-001: Define aggregated context data structures ✓
    - REQ-CTX-002: Implement context aggregator class ✓
    - REQ-CTX-003: Merge results from multiple agents ✓
    - REQ-CTX-004: Deduplicate citations by PMID ✓
    - REQ-CTX-005: Rank results by relevance score ✓
    - REQ-CTX-006: Extract entities from results ✓
    - REQ-CTX-007: Track agent contributions ✓
    - REQ-CTX-008: Apply relevance threshold filtering ✓
    - REQ-CTX-009: Handle empty or missing results ✓
    - REQ-CTX-010: Preserve metadata from sources ✓
    - REQ-CTX-011: Generate aggregation statistics ✓
    - REQ-CTX-012: Support configurable ranking strategies ✓
    - REQ-CTX-013: Maintain citation quality scores ✓
    - REQ-CTX-014: Extract domain-specific entities (EEG) ✓
    - REQ-CTX-015: Output standardized format ✓
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.3,
        max_citations: int = 50,
        entity_min_frequency: int = 2,
        ranking_strategy: str = "weighted"
    ):
        """
        Initialize context aggregator
        
        Args:
            relevance_threshold: Minimum relevance score (0-1)
            max_citations: Maximum citations to include
            entity_min_frequency: Min frequency for entity extraction
            ranking_strategy: "weighted", "simple", or "hybrid"
        """
        self.relevance_threshold = relevance_threshold
        self.max_citations = max_citations
        self.entity_min_frequency = entity_min_frequency
        self.ranking_strategy = ranking_strategy
        
        # Statistics
        self.stats = {
            'total_aggregations': 0,
            'total_citations_processed': 0,
            'total_citations_deduplicated': 0,
            'total_entities_extracted': 0,
            'agent_usage': defaultdict(int)
        }
        
        # EEG-specific entity patterns (REQ-CTX-014)
        self.entity_patterns = {
            'biomarker': [
                r'\b(alpha|beta|theta|delta|gamma)\s+(power|wave|rhythm|activity|band)',
                r'\bP300\b', r'\bN400\b', r'\bMMN\b', r'\bERN\b',
                r'\b(event-related\s+potential|ERP)s?\b',
                r'\b(spectral\s+power|frequency\s+band)s?\b'
            ],
            'condition': [
                r'\b(epilepsy|seizure|Alzheimer|dementia|depression|ADHD|autism|schizophrenia)',
                r'\b(sleep\s+disorder|insomnia|narcolepsy)',
                r'\b(traumatic\s+brain\s+injury|TBI|stroke)'
            ],
            'method': [
                r'\b(LORETA|sLORETA|eLORETA|VARETA)',
                r'\b(independent\s+component\s+analysis|ICA)',
                r'\b(wavelet\s+transform|FFT|Fourier)',
                r'\b(machine\s+learning|deep\s+learning|neural\s+network|CNN|LSTM)'
            ],
            'brain_region': [
                r'\b(frontal|temporal|parietal|occipital)\s+(lobe|cortex|region)',
                r'\b(hippocampus|amygdala|thalamus|cerebellum)',
                r'\b(prefrontal\s+cortex|PFC|dorsolateral)'
            ]
        }
    
    # REQ-CTX-003: Merge results from multiple agents
    async def aggregate(
        self,
        query: str,
        agent_results: Dict[str, Any]
    ) -> AggregatedContext:
        """
        Aggregate results from multiple agents
        
        Args:
            query: Original user query
            agent_results: Dict mapping agent_name -> agent result data
            
        Returns:
            AggregatedContext with merged and ranked results
        """
        self.stats['total_aggregations'] += 1
        
        # Extract citations from all agents
        all_citations = self._extract_citations(agent_results)
        self.stats['total_citations_processed'] += len(all_citations)
        
        # REQ-CTX-004: Deduplicate citations
        deduplicated = self._deduplicate_citations(all_citations)
        self.stats['total_citations_deduplicated'] += (len(all_citations) - len(deduplicated))
        
        # REQ-CTX-005: Rank by relevance
        ranked_citations = self._rank_citations(deduplicated, query)
        
        # REQ-CTX-008: Apply relevance threshold
        filtered_citations = [
            c for c in ranked_citations 
            if c.relevance_score >= self.relevance_threshold
        ]
        
        # Limit to max citations
        final_citations = filtered_citations[:self.max_citations]
        
        # REQ-CTX-006: Extract entities
        entities = self._extract_entities(final_citations)
        self.stats['total_entities_extracted'] += len(entities)
        
        # REQ-CTX-007: Track agent contributions
        agent_contributions = self._count_contributions(final_citations)
        for agent, count in agent_contributions.items():
            self.stats['agent_usage'][agent] += count
        
        # REQ-CTX-011: Generate statistics
        statistics = {
            'total_results': len(all_citations),
            'after_deduplication': len(deduplicated),
            'after_filtering': len(filtered_citations),
            'final_citations': len(final_citations),
            'entities_found': len(entities),
            'average_relevance': sum(c.relevance_score for c in final_citations) / len(final_citations) if final_citations else 0,
            'sources_used': len(agent_results)
        }
        
        # REQ-CTX-015: Output standardized format
        return AggregatedContext(
            query=query,
            citations=final_citations,
            entities=entities,
            total_sources=len(all_citations),
            agent_contributions=agent_contributions,
            relevance_threshold=self.relevance_threshold,
            timestamp=datetime.now().isoformat(),
            statistics=statistics
        )
    
    def _extract_citations(self, agent_results: Dict[str, Any]) -> List[Citation]:
        """Extract citations from all agent results"""
        citations = []
        
        for agent_name, result in agent_results.items():
            # REQ-CTX-009: Handle empty or missing results
            if not result or not isinstance(result, dict):
                continue
            
            # Handle different result formats from different agents
            result_data = result.get('data', result.get('results', []))
            
            if isinstance(result_data, list):
                for item in result_data:
                    citation = self._parse_citation(item, agent_name)
                    if citation:
                        citations.append(citation)
            elif isinstance(result_data, dict):
                citation = self._parse_citation(result_data, agent_name)
                if citation:
                    citations.append(citation)
        
        return citations
    
    def _parse_citation(self, item: Dict[str, Any], agent_name: str) -> Optional[Citation]:
        """Parse a single citation from agent result"""
        try:
            # REQ-CTX-010: Preserve metadata from sources
            citation = Citation(
                pmid=item.get('pmid'),
                title=item.get('title', ''),
                authors=item.get('authors', []),
                year=item.get('year'),
                journal=item.get('journal'),
                doi=item.get('doi'),
                url=item.get('url'),
                abstract=item.get('abstract', item.get('snippet', '')),
                relevance_score=item.get('relevance_score', item.get('score', 0.5)),
                source_agents=[agent_name],
                metadata=item.get('metadata', {})
            )
            return citation
        except Exception:
            return None
    
    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Deduplicate citations by PMID, DOI, or title
        REQ-CTX-004: Deduplicate citations by PMID
        """
        seen_ids: Set[str] = set()
        deduplicated: List[Citation] = []
        
        for citation in citations:
            citation_id = citation.get_id()
            
            if citation_id in seen_ids:
                # Merge source agents for duplicates
                for existing in deduplicated:
                    if existing.get_id() == citation_id:
                        # Combine source agents
                        for agent in citation.source_agents:
                            if agent not in existing.source_agents:
                                existing.source_agents.append(agent)
                        # Use higher relevance score
                        existing.relevance_score = max(
                            existing.relevance_score,
                            citation.relevance_score
                        )
                        break
            else:
                seen_ids.add(citation_id)
                deduplicated.append(citation)
        
        return deduplicated
    
    def _rank_citations(self, citations: List[Citation], query: str) -> List[Citation]:
        """
        Rank citations by relevance
        REQ-CTX-005: Rank results by relevance score
        REQ-CTX-012: Support configurable ranking strategies
        """
        if self.ranking_strategy == "weighted":
            # Weight by multiple factors
            for citation in citations:
                score = citation.relevance_score
                
                # Boost for multiple source agents
                if len(citation.source_agents) > 1:
                    score *= (1 + 0.1 * len(citation.source_agents))
                
                # Boost for recent publications
                if citation.year and citation.year >= 2020:
                    score *= 1.1
                
                # Boost for query term matches in title
                if citation.title and query:
                    query_terms = query.lower().split()
                    title_lower = citation.title.lower()
                    matches = sum(1 for term in query_terms if term in title_lower)
                    score *= (1 + 0.05 * matches)
                
                citation.relevance_score = min(score, 1.0)
        
        elif self.ranking_strategy == "hybrid":
            # Combine relevance score with citation count
            for citation in citations:
                citation.relevance_score = (
                    0.7 * citation.relevance_score +
                    0.3 * (len(citation.source_agents) / len(citations))
                )
        
        # Sort by relevance (descending)
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        
        return citations
    
    def _extract_entities(self, citations: List[Citation]) -> List[Entity]:
        """
        Extract entities from citations
        REQ-CTX-006: Extract entities from results
        REQ-CTX-014: Extract domain-specific entities (EEG)
        """
        entity_map: Dict[Tuple[str, str], Entity] = {}
        
        for citation in citations:
            citation_id = citation.get_id()
            
            # Extract from title and abstract
            text = f"{citation.title} {citation.abstract}".lower()
            
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group(0)
                        key = (entity_text.lower(), entity_type)
                        
                        if key in entity_map:
                            entity_map[key].frequency += 1
                            entity_map[key].citations.append(citation_id)
                        else:
                            entity_map[key] = Entity(
                                text=entity_text,
                                entity_type=entity_type,
                                frequency=1,
                                contexts=[citation.title],
                                citations=[citation_id],
                                confidence=0.8
                            )
        
        # Filter by minimum frequency
        entities = [
            entity for entity in entity_map.values()
            if entity.frequency >= self.entity_min_frequency
        ]
        
        # Sort by frequency (descending)
        entities.sort(key=lambda e: e.frequency, reverse=True)
        
        return entities
    
    def _count_contributions(self, citations: List[Citation]) -> Dict[str, int]:
        """
        Count contributions from each agent
        REQ-CTX-007: Track agent contributions
        """
        contributions = defaultdict(int)
        
        for citation in citations:
            for agent in citation.source_agents:
                contributions[agent] += 1
        
        return dict(contributions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            'total_aggregations': self.stats['total_aggregations'],
            'total_citations_processed': self.stats['total_citations_processed'],
            'total_citations_deduplicated': self.stats['total_citations_deduplicated'],
            'total_entities_extracted': self.stats['total_entities_extracted'],
            'agent_usage': dict(self.stats['agent_usage']),
            'relevance_threshold': self.relevance_threshold,
            'max_citations': self.max_citations,
            'ranking_strategy': self.ranking_strategy
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'total_aggregations': 0,
            'total_citations_processed': 0,
            'total_citations_deduplicated': 0,
            'total_entities_extracted': 0,
            'agent_usage': defaultdict(int)
        }
