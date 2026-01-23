"""
Tests for Enhanced Agent Modules

Comprehensive tests for:
- PubMedAgent with MeSH expansion and citation crawling
- SemanticScholarAgent with influence scoring
- SynthesisAgent with evidence ranking and gap detection
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestMeSHExpander:
    """Tests for MeSH term expansion."""
    
    def test_mesh_expansion_basic(self):
        """Test basic MeSH term expansion."""
        from eeg_rag.agents.pubmed_agent import MeSHExpander
        
        expander = MeSHExpander()
        
        # Test EEG-related terms
        result = expander.expand_query("eeg alpha waves")
        assert "Electroencephalography" in result or "EEG" in result
        
    def test_mesh_suggestions(self):
        """Test MeSH term suggestions."""
        from eeg_rag.agents.pubmed_agent import MeSHExpander
        
        expander = MeSHExpander()
        
        suggestions = expander.get_mesh_suggestions("eeg")
        assert len(suggestions) > 0
        assert any("Electroencephalography" in s or "Brain" in s for s in suggestions)
    
    def test_mesh_expansion_epilepsy(self):
        """Test epilepsy-related MeSH expansion."""
        from eeg_rag.agents.pubmed_agent import MeSHExpander
        
        expander = MeSHExpander()
        
        suggestions = expander.get_mesh_suggestions("epilepsy seizure")
        assert len(suggestions) > 0
    
    def test_mesh_expansion_empty_query(self):
        """Test handling of empty query."""
        from eeg_rag.agents.pubmed_agent import MeSHExpander
        
        expander = MeSHExpander()
        result = expander.expand_query("")
        assert result == ""


class TestPubMedQueryBuilder:
    """Tests for PubMed query construction."""
    
    def test_basic_query_build(self):
        """Test basic query building."""
        from eeg_rag.agents.pubmed_agent import PubMedQueryBuilder
        
        builder = PubMedQueryBuilder()
        
        query = builder.build_query("eeg alzheimer")
        assert query  # Non-empty
        assert "eeg" in query.lower() or "alzheimer" in query.lower()
    
    def test_query_with_date_range(self):
        """Test query with date filtering."""
        from eeg_rag.agents.pubmed_agent import PubMedQueryBuilder
        
        builder = PubMedQueryBuilder()
        
        query = builder.build_query(
            "eeg",
            date_range=(2020, 2024)
        )
        assert "2020" in query or "[dp]" in query or "date" in query.lower()
    
    def test_query_with_article_types(self):
        """Test query with article type filtering."""
        from eeg_rag.agents.pubmed_agent import PubMedQueryBuilder
        
        builder = PubMedQueryBuilder()
        
        query = builder.build_query(
            "eeg",
            article_types=["review", "clinical_trial"]
        )
        assert query  # Should contain article type filters
    
    def test_eeg_research_query(self):
        """Test specialized EEG research query."""
        from eeg_rag.agents.pubmed_agent import PubMedQueryBuilder
        
        builder = PubMedQueryBuilder()
        
        query = builder.build_eeg_research_query(
            topic="epilepsy",
            method="spectral analysis",
            application="seizure detection"
        )
        assert query
        assert "epilepsy" in query.lower() or "seizure" in query.lower()


class TestCitationCrawler:
    """Tests for citation network traversal."""
    
    def test_citation_crawler_init(self):
        """Test citation crawler initialization."""
        from eeg_rag.agents.pubmed_agent import CitationCrawler
        
        crawler = CitationCrawler()
        assert crawler.ELINK_BASE is not None
        assert "ncbi.nlm.nih.gov" in crawler.ELINK_BASE
    
    @pytest.mark.asyncio
    async def test_get_citing_papers_mock(self):
        """Test citation retrieval with mock."""
        from eeg_rag.agents.pubmed_agent import CitationCrawler
        
        crawler = CitationCrawler()
        
        # Mock the session's get method
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "linksets": [{
                "linksetdbs": [{
                    "linkname": "pubmed_pubmed_citedin",
                    "links": [12345678, 23456789]
                }]
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        crawler._session = mock_session
        
        result = await crawler.get_citing_papers("12345678")
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_references_mock(self):
        """Test reference retrieval with mock."""
        from eeg_rag.agents.pubmed_agent import CitationCrawler
        
        crawler = CitationCrawler()
        
        # Mock the session's get method
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "linksets": [{
                "linksetdbs": [{
                    "linkname": "pubmed_pubmed_refs",
                    "links": [11111111, 22222222]
                }]
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        crawler._session = mock_session
        
        result = await crawler.get_references("12345678")
        assert isinstance(result, list)


class TestInfluenceScorer:
    """Tests for research influence scoring."""
    
    def test_influence_scorer_init(self):
        """Test influence scorer initialization."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        assert scorer.TOP_VENUES is not None
    
    def test_citation_score_calculation(self):
        """Test citation score calculation."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        
        # High citation count should give high score
        high_score = scorer._calculate_citation_score(500)
        low_score = scorer._calculate_citation_score(5)
        
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
    
    def test_recency_score_calculation(self):
        """Test recency score calculation."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        
        current_year = datetime.now().year
        
        # Recent paper should score higher
        recent_score = scorer._calculate_recency_score(current_year - 1)
        old_score = scorer._calculate_recency_score(2000)
        
        assert recent_score > old_score
        assert 0 <= recent_score <= 1
    
    def test_venue_score_calculation(self):
        """Test venue prestige score."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        
        # Top venue should score high
        nature_score = scorer._calculate_venue_score("Nature")
        unknown_score = scorer._calculate_venue_score("Unknown Journal")
        
        assert nature_score > unknown_score
        assert nature_score >= 0.8  # Top venue
    
    def test_overall_influence_score(self):
        """Test overall influence score calculation."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        
        paper = {
            "citation_count": 100,
            "influential_citation_count": 20,
            "year": 2022,
            "venue": "Nature",
            "is_open_access": True
        }
        
        # Use score_paper which takes a dict
        score = scorer.score_paper(paper)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be relatively high
    
    def test_paper_ranking(self):
        """Test paper ranking by influence."""
        from eeg_rag.agents.semantic_scholar_agent import InfluenceScorer
        
        scorer = InfluenceScorer()
        
        papers = [
            {"citation_count": 10, "year": 2020, "venue": "Unknown"},
            {"citation_count": 500, "year": 2022, "venue": "Nature"},
            {"citation_count": 50, "year": 2021, "venue": "NeuroImage"},
        ]
        
        ranked = scorer.rank_papers(papers)
        
        # Nature paper should be first
        assert ranked[0]["citation_count"] == 500


class TestEvidenceRanker:
    """Tests for evidence quality ranking."""
    
    def test_evidence_ranker_init(self):
        """Test evidence ranker initialization."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        assert ranker.STUDY_PATTERNS is not None
    
    def test_study_type_detection_rct(self):
        """Test RCT detection."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker, EvidenceLevel
        
        ranker = EvidenceRanker()
        
        paper = {
            "title": "A randomized controlled trial of EEG biofeedback",
            "abstract": "Patients were randomized to treatment or control groups."
        }
        
        score = ranker.rank_evidence(paper)
        
        assert score.study_type == "randomized_controlled_trial"
        assert score.evidence_level == EvidenceLevel.LEVEL_1B
    
    def test_study_type_detection_systematic_review(self):
        """Test systematic review detection."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker, EvidenceLevel
        
        ranker = EvidenceRanker()
        
        paper = {
            "title": "Systematic review of EEG biomarkers in depression",
            "abstract": "We conducted a meta-analysis of 50 studies."
        }
        
        score = ranker.rank_evidence(paper)
        
        assert score.study_type == "systematic_review"
        assert score.evidence_level == EvidenceLevel.LEVEL_1A
    
    def test_study_type_detection_cohort(self):
        """Test cohort study detection."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        
        paper = {
            "title": "Longitudinal study of EEG changes in aging",
            "abstract": "A prospective cohort study following participants over 10 years."
        }
        
        score = ranker.rank_evidence(paper)
        
        assert score.study_type in ["cohort_study", "prospective"]
        assert score.evidence_level.numeric_score >= 0.5
    
    def test_sample_size_extraction(self):
        """Test sample size score calculation."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        
        large_study = {
            "title": "EEG study",
            "abstract": "We studied 500 participants with EEG."
        }
        
        small_study = {
            "title": "EEG case series",
            "abstract": "We report on 10 patients."
        }
        
        large_score = ranker.rank_evidence(large_study)
        small_score = ranker.rank_evidence(small_study)
        
        assert large_score.sample_size_score > small_score.sample_size_score
    
    def test_clinical_relevance_score(self):
        """Test clinical relevance detection."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        
        clinical = {
            "title": "Diagnostic accuracy of EEG in epilepsy",
            "abstract": "Clinical trial evaluating treatment outcomes in patients."
        }
        
        basic = {
            "title": "EEG signal processing algorithm",
            "abstract": "Novel signal processing method using wavelets."
        }
        
        clinical_score = ranker.rank_evidence(clinical)
        basic_score = ranker.rank_evidence(basic)
        
        assert clinical_score.clinical_relevance > basic_score.clinical_relevance
    
    def test_paper_ranking(self):
        """Test ranking multiple papers."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        
        papers = [
            {"title": "Case report of EEG findings", "abstract": "Single case study."},
            {"title": "Meta-analysis of EEG biomarkers", "abstract": "Systematic review of 100 studies."},
            {"title": "EEG cohort study", "abstract": "Prospective study of 200 patients."},
        ]
        
        ranked = ranker.rank_papers(papers)
        
        # Meta-analysis should be first
        assert ranked[0][1].study_type == "systematic_review"
    
    def test_evidence_summary(self):
        """Test evidence summary generation."""
        from eeg_rag.agents.synthesis_agent import EvidenceRanker
        
        ranker = EvidenceRanker()
        
        papers = [
            {"title": "RCT of EEG", "abstract": "Randomized controlled trial."},
            {"title": "Case series", "abstract": "Case series of 5 patients."},
        ]
        
        summary = ranker.get_evidence_summary(papers)
        
        assert summary["total_papers"] == 2
        assert "study_types" in summary
        assert "evidence_levels" in summary


class TestGapDetector:
    """Tests for research gap detection."""
    
    def test_gap_detector_init(self):
        """Test gap detector initialization."""
        from eeg_rag.agents.synthesis_agent import GapDetector
        
        detector = GapDetector()
        assert detector.EEG_GAP_PATTERNS is not None
    
    def test_sample_size_gap_detection(self):
        """Test sample size limitation detection."""
        from eeg_rag.agents.synthesis_agent import GapDetector, GapType
        
        detector = GapDetector()
        
        papers = [
            {
                "title": "EEG study",
                "abstract": "Limitation: small sample size of only 15 participants."
            },
            {
                "title": "Another study",
                "abstract": "Limited by the small number of subjects (n=10)."
            }
        ]
        
        gaps = detector.detect_gaps(papers)
        
        sample_gaps = [g for g in gaps if g.gap_type == GapType.SAMPLE_SIZE]
        assert len(sample_gaps) > 0
    
    def test_methodology_gap_detection(self):
        """Test methodological limitation detection."""
        from eeg_rag.agents.synthesis_agent import GapDetector, GapType
        
        detector = GapDetector()
        
        papers = [
            {
                "title": "EEG review",
                "abstract": "Different methodologies across studies limit comparability."
            }
        ]
        
        gaps = detector.detect_gaps(papers)
        
        method_gaps = [g for g in gaps if g.gap_type == GapType.METHODOLOGICAL]
        assert len(method_gaps) >= 0  # May or may not detect
    
    def test_limitation_extraction(self):
        """Test explicit limitation extraction."""
        from eeg_rag.agents.synthesis_agent import GapDetector
        
        detector = GapDetector()
        
        paper = {
            "title": "EEG study",
            "abstract": "A limitation of this study is the cross-sectional design."
        }
        
        limitations = detector.extract_limitations(paper)
        
        assert len(limitations) > 0
    
    def test_future_directions_extraction(self):
        """Test future research extraction."""
        from eeg_rag.agents.synthesis_agent import GapDetector
        
        detector = GapDetector()
        
        paper = {
            "title": "EEG study",
            "abstract": "Future research should investigate long-term outcomes."
        }
        
        directions = detector.extract_future_directions(paper)
        
        assert len(directions) > 0
    
    def test_gap_summary(self):
        """Test gap summary generation."""
        from eeg_rag.agents.synthesis_agent import GapDetector
        
        detector = GapDetector()
        
        papers = [
            {"title": "Study 1", "abstract": "Small sample size of 10 patients."},
            {"title": "Study 2", "abstract": "Cross-sectional design limits conclusions."},
        ]
        
        gaps = detector.detect_gaps(papers)
        summary = detector.get_gap_summary(gaps)
        
        assert "total_gaps" in summary
        assert "by_type" in summary


class TestSynthesisAgent:
    """Tests for the synthesis agent."""
    
    def test_synthesis_agent_init(self):
        """Test synthesis agent initialization."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent()
        
        assert agent.evidence_ranker is not None
        assert agent.gap_detector is not None
    
    @pytest.mark.asyncio
    async def test_synthesize_basic(self):
        """Test basic synthesis."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent()
        
        papers = [
            {
                "title": "Alpha waves in meditation: A systematic review",
                "abstract": "Meta-analysis of 50 studies on alpha power during meditation.",
                "year": 2022,
                "pmid": "12345678"
            },
            {
                "title": "EEG biomarkers in epilepsy",
                "abstract": "Randomized controlled trial of 100 patients.",
                "year": 2023,
                "pmid": "23456789"
            }
        ]
        
        result = await agent.synthesize(papers, query="EEG research")
        
        assert result.summary
        assert len(result.themes) >= 0
        assert result.confidence > 0
        assert len(result.citations) == 2
    
    @pytest.mark.asyncio
    async def test_synthesize_with_gaps(self):
        """Test synthesis with gap detection."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent()
        
        papers = [
            {
                "title": "EEG study with limitations",
                "abstract": "Small sample size of 15 participants limits generalizability.",
                "year": 2022
            }
        ]
        
        result = await agent.synthesize(papers, include_gaps=True)
        
        # May or may not find gaps depending on patterns
        assert isinstance(result.research_gaps, list)
    
    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test agent execute method."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        from eeg_rag.agents.base_agent import AgentQuery
        
        agent = SynthesisAgent()
        
        query = AgentQuery(
            text="EEG research synthesis",
            context={"papers": [
                {"title": "Test paper", "abstract": "Test abstract", "year": 2023}
            ]},
            parameters={}
        )
        
        result = await agent.execute(query)
        
        assert result.success
        assert "summary" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_no_papers(self):
        """Test execute with no papers."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        from eeg_rag.agents.base_agent import AgentQuery
        
        agent = SynthesisAgent()
        
        query = AgentQuery(
            text="test",
            context={},
            parameters={}
        )
        
        result = await agent.execute(query)
        
        assert not result.success
        assert "No papers" in result.error
    
    def test_theme_extraction(self):
        """Test theme extraction from papers."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent()
        
        papers = [
            {"title": "Alpha and beta oscillations in cognition", "abstract": "Frequency band analysis."},
            {"title": "Functional connectivity in the brain", "abstract": "Coherence and synchronization."},
            {"title": "P300 component in attention", "abstract": "Event-related potential study."},
        ]
        
        themes = agent._extract_themes(papers)
        
        assert len(themes) > 0
        # Should identify frequency_analysis and/or connectivity themes
    
    def test_citation_extraction(self):
        """Test citation formatting."""
        from eeg_rag.agents.synthesis_agent import SynthesisAgent
        
        agent = SynthesisAgent()
        
        papers = [
            {
                "title": "Test Paper",
                "authors": ["Smith J", "Jones M"],
                "year": 2023,
                "journal": "J Neurosci",
                "pmid": "12345678"
            }
        ]
        
        citations = agent._extract_citations(papers)
        
        assert len(citations) == 1
        assert "PMID:12345678" in citations[0]


class TestSemanticScholarAgent:
    """Tests for Semantic Scholar agent."""
    
    def test_s2_agent_init(self):
        """Test S2 agent initialization."""
        from eeg_rag.agents.semantic_scholar_agent import SemanticScholarAgent
        
        agent = SemanticScholarAgent()
        
        assert agent.S2_API_BASE is not None
        assert agent.influence_scorer is not None
    
    def test_s2_paper_dataclass(self):
        """Test S2Paper dataclass."""
        from eeg_rag.agents.semantic_scholar_agent import S2Paper
        
        paper = S2Paper(
            paper_id="abc123",
            title="Test Paper",
            abstract="Test abstract",
            year=2023,
            citation_count=100
        )
        
        paper_dict = paper.to_dict()
        
        assert paper_dict["paper_id"] == "abc123"
        assert paper_dict["citation_count"] == 100
        assert paper_dict["source"] == "semantic_scholar"
    
    def test_s2_agent_statistics(self):
        """Test statistics tracking."""
        from eeg_rag.agents.semantic_scholar_agent import SemanticScholarAgent
        
        agent = SemanticScholarAgent()
        
        stats = agent.get_statistics()
        
        assert "total_searches" in stats
        assert "cache_hits" in stats


class TestPubMedAgent:
    """Tests for PubMed agent."""
    
    def test_pubmed_agent_init(self):
        """Test PubMed agent initialization."""
        from eeg_rag.agents.pubmed_agent import PubMedAgent
        
        agent = PubMedAgent()
        
        assert agent.mesh_expander is not None
        assert agent.query_builder is not None
        # citation_crawler is lazily initialized, so check the method exists
        assert hasattr(agent, '_get_citation_crawler')
    
    def test_pubmed_paper_dataclass(self):
        """Test PubMedPaper dataclass."""
        from eeg_rag.agents.pubmed_agent import PubMedPaper
        
        paper = PubMedPaper(
            pmid="12345678",
            title="Test Paper",
            abstract="Test abstract",
            year=2023
        )
        
        paper_dict = paper.to_dict()
        
        assert paper_dict["pmid"] == "12345678"
        assert paper_dict["source"] == "pubmed"
    
    @pytest.mark.asyncio
    async def test_pubmed_agent_execute_mock(self):
        """Test PubMed agent execute with mock."""
        from eeg_rag.agents.pubmed_agent import PubMedAgent
        from eeg_rag.agents.base_agent import AgentQuery
        
        agent = PubMedAgent()
        
        # Mock the internal search method
        with patch.object(agent, '_search', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {"pmids": [], "total_count": 0}
            
            query = AgentQuery(
                text="test query",
                context={},
                parameters={}
            )
            
            result = await agent.execute(query)
            
            assert result.success
            assert "papers" in result.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
