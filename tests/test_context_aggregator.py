"""
Test Suite for Context Aggregator (Component 10/12)
Tests aggregation, deduplication, ranking, and entity extraction
"""

import pytest
from eeg_rag.ensemble import (
    ContextAggregator,
    Citation,
    Entity,
    AggregatedContext
)


class TestCitation:
    """Tests for Citation dataclass"""

    def test_citation_creation(self):
        """Test creating a citation"""
        citation = Citation(
            pmid="12345678",
            title="EEG Alpha Power in Alzheimer's Disease",
            authors=["Smith J", "Doe A"],
            year=2023,
            journal="Brain Research",
            relevance_score=0.85
        )

        assert citation.pmid == "12345678"
        assert citation.year == 2023
        assert len(citation.authors) == 2
        assert citation.relevance_score == 0.85

    def test_citation_get_id_pmid(self):
        """Test citation ID generation with PMID"""
        citation = Citation(pmid="12345678", title="Test")
        assert citation.get_id() == "pmid:12345678"

    def test_citation_get_id_doi(self):
        """Test citation ID generation with DOI"""
        citation = Citation(doi="10.1234/test", title="Test")
        assert citation.get_id() == "doi:10.1234/test"

    def test_citation_get_id_title(self):
        """Test citation ID generation with title hash"""
        citation = Citation(title="Test Article Title")
        citation_id = citation.get_id()
        assert citation_id.startswith("title:")

    def test_citation_to_dict(self):
        """Test citation serialization"""
        citation = Citation(
            pmid="12345678",
            title="Test",
            relevance_score=0.9,
            source_agents=["local", "web"]
        )

        result = citation.to_dict()
        assert result['pmid'] == "12345678"
        assert result['relevance_score'] == 0.9
        assert len(result['source_agents']) == 2


class TestEntity:
    """Tests for Entity dataclass"""

    def test_entity_creation(self):
        """Test creating an entity"""
        entity = Entity(
            text="alpha power",
            entity_type="biomarker",
            frequency=5,
            confidence=0.9
        )

        assert entity.text == "alpha power"
        assert entity.entity_type == "biomarker"
        assert entity.frequency == 5
        assert entity.confidence == 0.9

    def test_entity_to_dict(self):
        """Test entity serialization"""
        entity = Entity(
            text="epilepsy",
            entity_type="condition",
            frequency=3,
            citations=["pmid:123", "pmid:456"]
        )

        result = entity.to_dict()
        assert result['text'] == "epilepsy"
        assert result['type'] == "condition"
        assert len(result['citations']) == 2


class TestAggregatedContext:
    """Tests for AggregatedContext dataclass"""

    def test_aggregated_context_creation(self):
        """Test creating aggregated context"""
        citations = [
            Citation(pmid="123", title="Test 1", relevance_score=0.9),
            Citation(pmid="456", title="Test 2", relevance_score=0.8)
        ]

        entities = [
            Entity(text="alpha power", entity_type="biomarker", frequency=2)
        ]

        context = AggregatedContext(
            query="test query",
            citations=citations,
            entities=entities,
            total_sources=2,
            agent_contributions={"local": 1, "web": 1},
            relevance_threshold=0.3,
            timestamp="2024-01-01T00:00:00",
            statistics={'total_results': 2}
        )

        assert context.query == "test query"
        assert len(context.citations) == 2
        assert len(context.entities) == 1
        assert context.total_sources == 2

    def test_aggregated_context_to_dict(self):
        """Test aggregated context serialization"""
        context = AggregatedContext(
            query="test",
            citations=[Citation(title="Test", relevance_score=0.8)],
            entities=[],
            total_sources=1,
            agent_contributions={"local": 1},
            relevance_threshold=0.3,
            timestamp="2024-01-01",
            statistics={}
        )

        result = context.to_dict()
        assert 'query' in result
        assert 'citations' in result
        assert 'entities' in result


class TestContextAggregator:
    """Tests for ContextAggregator class"""

    def test_aggregator_initialization(self):
        """Test creating context aggregator"""
        aggregator = ContextAggregator(
            relevance_threshold=0.5,
            max_citations=20,
            entity_min_frequency=3
        )

        assert aggregator.relevance_threshold == 0.5
        assert aggregator.max_citations == 20
        assert aggregator.entity_min_frequency == 3

    @pytest.mark.asyncio
    async def test_aggregate_single_agent(self):
        """Test aggregating results from single agent"""
        aggregator = ContextAggregator(relevance_threshold=0.3)

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '12345678',
                        'title': 'Alpha power in Alzheimer disease',
                        'authors': ['Smith J'],
                        'year': 2023,
                        'relevance_score': 0.9
                    }
                ]
            }
        }

        result = await aggregator.aggregate("test query", agent_results)

        assert isinstance(result, AggregatedContext)
        assert len(result.citations) == 1
        assert result.citations[0].pmid == '12345678'

    @pytest.mark.asyncio
    async def test_aggregate_multiple_agents(self):
        """Test aggregating results from multiple agents"""
        aggregator = ContextAggregator(relevance_threshold=0.3)

        agent_results = {
            'local': {
                'data': [
                    {'pmid': '111', 'title': 'Study 1', 'relevance_score': 0.8}
                ]
            },
            'web': {
                'data': [
                    {'pmid': '222', 'title': 'Study 2', 'relevance_score': 0.7}
                ]
            }
        }

        result = await aggregator.aggregate("test query", agent_results)

        assert len(result.citations) == 2
        assert result.agent_contributions['local'] == 1
        assert result.agent_contributions['web'] == 1

    @pytest.mark.asyncio
    async def test_deduplication_by_pmid(self):
        """Test citation deduplication by PMID"""
        aggregator = ContextAggregator(
            relevance_threshold=0.0,
            ranking_strategy="simple"  # Use simple ranking to avoid boosting
        )

        agent_results = {
            'local': {
                'data': [
                    {'pmid': '12345', 'title': 'Test Study', 'relevance_score': 0.8}
                ]
            },
            'web': {
                'data': [
                    {'pmid': '12345', 'title': 'Test Study', 'relevance_score': 0.9}
                ]
            }
        }

        result = await aggregator.aggregate("test", agent_results)

        # Should deduplicate to 1 citation
        assert len(result.citations) == 1
        # Should merge source agents
        assert len(result.citations[0].source_agents) == 2
        assert 'local' in result.citations[0].source_agents
        assert 'web' in result.citations[0].source_agents
        # Should use higher relevance score (base score before ranking)
        assert result.citations[0].relevance_score >= 0.9

    @pytest.mark.asyncio
    async def test_relevance_threshold_filtering(self):
        """Test filtering by relevance threshold"""
        aggregator = ContextAggregator(relevance_threshold=0.7)

        agent_results = {
            'local': {
                'data': [
                    {'pmid': '111', 'title': 'High relevance', 'relevance_score': 0.9},
                    {'pmid': '222', 'title': 'Low relevance', 'relevance_score': 0.5}
                ]
            }
        }

        result = await aggregator.aggregate("test", agent_results)

        # Only high relevance citation should pass
        assert len(result.citations) == 1
        assert result.citations[0].pmid == '111'

    @pytest.mark.asyncio
    async def test_max_citations_limit(self):
        """Test max citations limit"""
        aggregator = ContextAggregator(
            relevance_threshold=0.0,
            max_citations=2
        )

        agent_results = {
            'local': {
                'data': [
                    {'pmid': f'{i}', 'title': f'Study {i}', 'relevance_score': 0.9}
                    for i in range(5)
                ]
            }
        }

        result = await aggregator.aggregate("test", agent_results)

        # Should limit to 2 citations
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Test entity extraction from citations"""
        aggregator = ContextAggregator(
            relevance_threshold=0.0,
            entity_min_frequency=1
        )

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'Alpha power and beta wave activity in epilepsy',
                        'abstract': 'Study of alpha power in frontal cortex',
                        'relevance_score': 0.9
                    }
                ]
            }
        }

        result = await aggregator.aggregate("test", agent_results)

        # Should extract entities
        assert len(result.entities) > 0
        entity_texts = [e.text.lower() for e in result.entities]
        # Should find biomarker patterns
        assert any('alpha' in text and 'power' in text for text in entity_texts)

    @pytest.mark.asyncio
    async def test_handle_empty_results(self):
        """Test handling empty agent results"""
        aggregator = ContextAggregator()

        agent_results = {}

        result = await aggregator.aggregate("test", agent_results)

        assert len(result.citations) == 0
        assert len(result.entities) == 0
        assert result.total_sources == 0

    @pytest.mark.asyncio
    async def test_handle_malformed_results(self):
        """Test handling malformed agent results"""
        aggregator = ContextAggregator()

        agent_results = {
            'local': None,
            'web': {'data': []},
            'graph': {'results': []}
        }

        result = await aggregator.aggregate("test", agent_results)

        # Should handle gracefully
        assert isinstance(result, AggregatedContext)

    @pytest.mark.asyncio
    async def test_weighted_ranking_strategy(self):
        """Test weighted ranking strategy"""
        aggregator = ContextAggregator(
            ranking_strategy="weighted",
            relevance_threshold=0.0
        )

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'test query result',  # Should boost for query match
                        'year': 2023,  # Recent year
                        'relevance_score': 0.7
                    },
                    {
                        'pmid': '222',
                        'title': 'other study',
                        'year': 2015,
                        'relevance_score': 0.8
                    }
                ]
            }
        }

        result = await aggregator.aggregate("test query", agent_results)

        # First result should be boosted due to query match and recent year
        assert result.citations[0].pmid == '111'

    def test_get_statistics(self):
        """Test getting aggregation statistics"""
        aggregator = ContextAggregator()

        stats = aggregator.get_statistics()

        assert 'total_aggregations' in stats
        assert 'total_citations_processed' in stats
        assert 'relevance_threshold' in stats

    def test_reset_statistics(self):
        """Test resetting statistics"""
        aggregator = ContextAggregator()
        aggregator.stats['total_aggregations'] = 10

        aggregator.reset_statistics()

        assert aggregator.stats['total_aggregations'] == 0
