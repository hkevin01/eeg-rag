"""
Tests for RAG system improvements:
  - EEGQueryExpander.expand() word-order preservation
  - ResearchAgent._deduplicate() fuzzy near-duplicate detection
  - ResearchAgent._metadata_quality_score()
  - SufficiencyEvaluator._mean_score() normalization fix
"""

import pytest
from collections import Counter
from eeg_rag.retrieval.query_expander import EEGQueryExpander
from eeg_rag.agents.research_agent.research_agent import ResearchAgent
from eeg_rag.rag.agentic_rag import SufficiencyEvaluator
from eeg_rag.retrieval.hybrid_retriever import HybridResult


# ---------------------------------------------------------------------------
# EEGQueryExpander tests
# ---------------------------------------------------------------------------

class TestQueryExpanderWordOrder:
    """Verifies that expand() preserves original word order."""

    def setup_method(self):
        self.expander = EEGQueryExpander()

    def test_original_query_is_prefix(self):
        """Expanded query must start with the original query verbatim."""
        original = "CNN for seizure detection"
        result = self.expander.expand(original)
        assert result.startswith(original), (
            f"Expected prefix '{original}', got '{result}'"
        )

    def test_synonyms_are_appended_not_inserted(self):
        """Synonym tokens must appear after original tokens, not mixed in."""
        result = self.expander.expand("seizure detection eeg")
        tokens = result.split()
        orig_tokens = ["seizure", "detection", "eeg"]
        positions = [tokens.index(t) for t in orig_tokens if t in tokens]
        assert positions == sorted(positions), (
            f"Original tokens out of order in: {result}"
        )

    def test_no_expansion_when_no_match(self):
        """Query without known EEG terms returns unchanged or same prefix."""
        query = "general information retrieval benchmark"
        result = self.expander.expand(query)
        assert result.startswith(query)

    def test_expansion_adds_meaningful_synonyms(self):
        """Synonyms should be substantive terms from the EEG domain."""
        result = self.expander.expand("bci motor imagery")
        assert len(result) > len("bci motor imagery"), (
            "Expected synonyms to be appended"
        )

    def test_no_duplicate_tokens(self):
        """Expanded query must not repeat any token more than twice."""
        result = self.expander.expand("seizure epilepsy eeg detection")
        tokens = result.lower().split()
        counts = Counter(tokens)
        repeated = {t: c for t, c in counts.items() if c > 2}
        assert not repeated, f"Duplicate tokens found: {repeated}"


# ---------------------------------------------------------------------------
# ResearchAgent._deduplicate tests
# ---------------------------------------------------------------------------

class TestResearchAgentDeduplication:
    """Tests for exact and fuzzy title deduplication."""

    def _paper(self, title="", pmid="", doi="", abstract="", year=2020):
        return {
            "title": title, "pmid": pmid, "doi": doi,
            "abstract": abstract, "year": year,
        }

    def test_exact_pmid_dedup(self):
        papers = [
            self._paper(title="Paper A", pmid="12345678"),
            self._paper(title="Paper A variant", pmid="12345678"),
        ]
        result = ResearchAgent._deduplicate(papers)
        assert len(result) == 1

    def test_exact_doi_dedup(self):
        papers = [
            self._paper(title="Paper B", doi="10.1000/xyz"),
            self._paper(title="Paper B different title", doi="10.1000/xyz"),
        ]
        result = ResearchAgent._deduplicate(papers)
        assert len(result) == 1

    def test_exact_title_dedup(self):
        papers = [
            self._paper(title="EEG-based seizure detection using CNN"),
            self._paper(title="EEG-based seizure detection using CNN"),
        ]
        result = ResearchAgent._deduplicate(papers)
        assert len(result) == 1

    def test_fuzzy_title_near_duplicate_removed(self):
        """Papers with only minor title variations are deduped."""
        papers = [
            self._paper(
                title="EEG based seizure detection using convolutional neural networks"
            ),
            self._paper(
                title="EEG-based seizure detection using convolutional neural networks"
            ),
        ]
        result = ResearchAgent._deduplicate(papers, fuzzy_title_threshold=0.92)
        assert len(result) == 1, "Near-duplicate titles should be collapsed"

    def test_distinct_papers_kept(self):
        """Genuinely different papers must not be merged."""
        papers = [
            self._paper(title="Alpha band EEG during cognitive load tasks"),
            self._paper(title="Deep learning for epileptic seizure prediction"),
            self._paper(title="Motor imagery BCI using SSVEP paradigm"),
        ]
        result = ResearchAgent._deduplicate(papers)
        assert len(result) == 3

    def test_empty_input(self):
        assert ResearchAgent._deduplicate([]) == []


# ---------------------------------------------------------------------------
# ResearchAgent._metadata_quality_score tests
# ---------------------------------------------------------------------------

class TestMetadataQualityScore:

    def test_complete_metadata_scores_high(self):
        paper = {
            "abstract": "This study investigates EEG biomarkers for epilepsy. " * 4,
            "pmid": "12345678",
            "year": 2022,
            "citation_count": 45,
            "journal": "Journal of Neural Engineering",
        }
        score = ResearchAgent._metadata_quality_score(paper)
        assert score >= 0.9, f"Complete metadata should score >= 0.9, got {score}"

    def test_empty_metadata_scores_zero(self):
        score = ResearchAgent._metadata_quality_score({})
        assert score == 0.0

    def test_abstract_only_partial_score(self):
        paper = {"abstract": "Short abstract about EEG signals and theta band power."}
        score = ResearchAgent._metadata_quality_score(paper)
        assert 0.1 <= score <= 0.5

    def test_no_abstract_below_half(self):
        paper = {"pmid": "99999999", "year": 2021, "journal": "NeuroImage"}
        score = ResearchAgent._metadata_quality_score(paper)
        assert score < 0.6

    def test_score_capped_at_one(self):
        paper = {
            "abstract": "x" * 200,
            "pmid": "11111111",
            "year": 2023,
            "citation_count": 500,
            "journal": "Brain",
        }
        score = ResearchAgent._metadata_quality_score(paper)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# SufficiencyEvaluator._mean_score normalization tests
# ---------------------------------------------------------------------------

class TestSufficiencyMeanScore:
    """Tests that _mean_score uses the correct dual-ranker RRF maximum."""

    def _hr(self, rrf_score: float) -> HybridResult:
        return HybridResult(
            doc_id="d1", text="text", metadata={},
            bm25_score=0.0, dense_score=0.0, rrf_score=rrf_score,
        )

    def test_empty_results_zero(self):
        assert SufficiencyEvaluator._mean_score([]) == 0.0

    def test_dual_ranker_max_gives_one(self):
        """A doc first in both rankers: rrf = 2/61 => normalised = 1.0."""
        rrf_max = 2.0 / 61.0
        score = SufficiencyEvaluator._mean_score([self._hr(rrf_max)])
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_single_ranker_max_gives_half(self):
        """A doc first in ONE ranker: rrf = 1/61 => normalised ~0.5."""
        rrf_single = 1.0 / 61.0
        score = SufficiencyEvaluator._mean_score([self._hr(rrf_single)])
        assert score == pytest.approx(0.5, abs=1e-3)

    def test_score_capped_at_one(self):
        """Scores above maximum are capped at 1.0."""
        score = SufficiencyEvaluator._mean_score([self._hr(1.0)])
        assert score == 1.0

    def test_score_is_mean_not_sum(self):
        """Mean of two identical results equals single result score."""
        rrf = 1.0 / 61.0
        s1 = SufficiencyEvaluator._mean_score([self._hr(rrf)])
        s2 = SufficiencyEvaluator._mean_score([self._hr(rrf), self._hr(rrf)])
        assert s1 == pytest.approx(s2, abs=1e-9)
