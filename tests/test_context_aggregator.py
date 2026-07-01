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
    async def test_structured_concept_coverage(self):
        """Test structured concept-group coverage diagnostics."""
        aggregator = ContextAggregator(relevance_threshold=0.0, entity_min_frequency=1)

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '333',
                        'title': 'EEG biomarkers for epilepsy',
                        'abstract': 'Alpha power biomarkers in epilepsy monitoring',
                        'relevance_score': 0.9,
                    }
                ]
            }
        }

        result = await aggregator.aggregate(
            'EEG biomarkers and ICA methods for epilepsy',
            agent_results,
        )

        stats = result.statistics
        assert stats['query_concept_coverage_score'] < 1.0
        assert any('method' in item for item in stats['missing_query_concept_groups'])

    @pytest.mark.asyncio
    async def test_structured_concept_groups_include_outcome_and_design(self):
        """Outcome and design concept groups should be extracted and tracked."""
        aggregator = ContextAggregator(relevance_threshold=0.0, entity_min_frequency=1)

        query = (
            "EEG seizure detection sensitivity in longitudinal cohort studies"
        )
        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '444',
                        'title': 'Longitudinal cohort EEG seizure study',
                        'abstract': 'Sensitivity and specificity outcomes in seizure detection.',
                        'relevance_score': 0.9,
                    }
                ]
            }
        }

        result = await aggregator.aggregate(query, agent_results)
        stats = result.statistics

        assert 'outcome' in stats['query_concept_groups']
        assert 'experimental_design' in stats['query_concept_groups']
        assert 'outcome' in stats['covered_query_concept_groups']
        assert 'experimental_design' in stats['covered_query_concept_groups']

    @pytest.mark.asyncio
    async def test_structured_concept_coverage_affects_ranking(self):
        """Test that concept coverage improves ranking for richer citations."""
        aggregator = ContextAggregator(relevance_threshold=0.0, ranking_strategy="weighted")

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': 'a1',
                        'title': 'EEG biomarkers in epilepsy',
                        'abstract': 'Alpha power biomarkers in epilepsy monitoring',
                        'relevance_score': 0.8,
                    },
                    {
                        'pmid': 'a2',
                        'title': 'EEG biomarkers overview',
                        'abstract': 'Alpha power biomarkers overview',
                        'relevance_score': 0.8,
                    },
                ]
            }
        }

        result = await aggregator.aggregate(
            'EEG biomarkers and epilepsy methods',
            agent_results,
        )

        assert result.citations[0].pmid == 'a1'

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

    @pytest.mark.asyncio
    async def test_concept_aware_ranking_strategy(self):
        """Concept-aware strategy should favor concept-complete citations."""
        aggregator = ContextAggregator(
            ranking_strategy="concept_aware",
            relevance_threshold=0.0,
        )

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': 'c1',
                        'title': 'EEG biomarkers in epilepsy',
                        'abstract': 'Alpha biomarker evidence only.',
                        'relevance_score': 0.85,
                    },
                    {
                        'pmid': 'c2',
                        'title': 'Longitudinal EEG biomarkers and outcomes in epilepsy cohort',
                        'abstract': 'Sensitivity outcomes from longitudinal cohort design.',
                        'relevance_score': 0.82,
                    },
                ]
            }
        }

        result = await aggregator.aggregate(
            'EEG biomarkers outcomes in longitudinal epilepsy cohort',
            agent_results,
        )

        assert result.citations[0].pmid == 'c2'
        assert result.citations[0].metadata['concept_coverage_score'] > result.citations[1].metadata['concept_coverage_score']

    @pytest.mark.asyncio
    async def test_diversified_ranking_reduces_redundancy(self):
        """Test diversified ranking prefers a distinct second citation."""
        aggregator = ContextAggregator(
            ranking_strategy="diversified",
            relevance_threshold=0.0,
            max_citations=3,
            diversity_lambda=0.7,
        )

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'EEG seizure detection using convolutional neural networks',
                        'abstract': 'Deep learning model for EEG seizure detection with convolutional features.',
                        'relevance_score': 0.95,
                        'year': 2024,
                    },
                    {
                        'pmid': '222',
                        'title': 'EEG-based seizure detection using convolutional neural networks',
                        'abstract': 'Convolutional network approach for seizure detection in EEG recordings.',
                        'relevance_score': 0.93,
                        'year': 2023,
                    },
                    {
                        'pmid': '333',
                        'title': 'Graph biomarkers for epilepsy progression in longitudinal EEG',
                        'abstract': 'Longitudinal network biomarkers expose epileptic progression using EEG graphs.',
                        'relevance_score': 0.84,
                        'year': 2024,
                    },
                ]
            }
        }

        result = await aggregator.aggregate('eeg seizure detection', agent_results)

        assert result.citations[0].pmid == '111'
        assert result.citations[1].pmid == '333'

    @pytest.mark.asyncio
    async def test_diversified_ranking_uses_centrality_metadata(self):
        """Test diversified ranking boosts high-centrality papers."""
        aggregator = ContextAggregator(
            ranking_strategy="diversified",
            relevance_threshold=0.0,
            diversity_lambda=0.85,
        )

        agent_results = {
            'graph': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'Alpha power in epilepsy cohorts',
                        'abstract': 'Alpha power study in epilepsy cohorts.',
                        'relevance_score': 0.80,
                        'metadata': {'centrality_score': 0.90},
                    },
                    {
                        'pmid': '222',
                        'title': 'Alpha power in epilepsy case series',
                        'abstract': 'Alpha power findings in small epilepsy case series.',
                        'relevance_score': 0.80,
                        'metadata': {'centrality_score': 0.10},
                    },
                ]
            }
        }

        result = await aggregator.aggregate('alpha power epilepsy', agent_results)

        assert result.citations[0].pmid == '111'
        assert result.citations[0].metadata['utility_score'] >= result.citations[1].metadata['utility_score']

    @pytest.mark.asyncio
    async def test_diversified_ranking_uses_embedding_similarity(self):
        """Dense embeddings should drive redundancy estimates when available."""
        aggregator = ContextAggregator(
            ranking_strategy="diversified",
            relevance_threshold=0.0,
            max_citations=3,
        )

        agent_results = {
            'dense': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'Study A',
                        'abstract': 'Topic A',
                        'relevance_score': 0.95,
                        'metadata': {'embedding_vector': [1.0, 0.0, 0.0]},
                    },
                    {
                        'pmid': '222',
                        'title': 'Study B',
                        'abstract': 'Lexically different text',
                        'relevance_score': 0.94,
                        'metadata': {'embedding_vector': [0.99, 0.01, 0.0]},
                    },
                    {
                        'pmid': '333',
                        'title': 'Study C',
                        'abstract': 'Orthogonal evidence',
                        'relevance_score': 0.80,
                        'metadata': {'embedding_vector': [0.0, 1.0, 0.0]},
                    },
                ]
            }
        }

        result = await aggregator.aggregate('study topic', agent_results)

        assert result.citations[1].pmid == '333'
        assert result.statistics['redundancy_score'] > 0.0

    @pytest.mark.asyncio
    async def test_statistics_include_query_entity_coverage(self):
        """Aggregation statistics should report query-entity coverage gaps."""
        aggregator = ContextAggregator(
            ranking_strategy="diversified",
            relevance_threshold=0.0,
            entity_min_frequency=1,
        )

        agent_results = {
            'local': {
                'data': [
                    {
                        'pmid': '111',
                        'title': 'Alpha power biomarkers in epilepsy',
                        'abstract': 'Frontal cortex alpha power correlates with epilepsy severity.',
                        'relevance_score': 0.9,
                    }
                ]
            }
        }

        result = await aggregator.aggregate('alpha power and frontal cortex in epilepsy and sleep disorder', agent_results)

        assert result.statistics['query_entity_coverage_score'] < 1.0
        assert 'sleep disorder' in result.statistics['missing_query_entities']

    @pytest.mark.asyncio
    async def test_diversity_lambda_adapts_to_query_complexity(self):
        """Simple queries should favor precision more than complex queries."""
        aggregator = ContextAggregator(
            ranking_strategy="diversified",
            relevance_threshold=0.0,
        )
        simple_results = {
            'local': {'data': [{'pmid': '1', 'title': 'Alpha power study', 'relevance_score': 0.8}]}
        }
        complex_results = {
            'local': {'data': [{'pmid': '1', 'title': 'Alpha power study in sleep disorder and epilepsy', 'relevance_score': 0.8}]}
        }

        simple = await aggregator.aggregate('alpha power epilepsy', simple_results)
        complex_ = await aggregator.aggregate(
            'What longitudinal EEG biomarkers explain alpha power changes in epilepsy and sleep disorder cohorts?',
            complex_results,
        )

        assert simple.statistics['diversity_lambda'] > complex_.statistics['diversity_lambda']

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
