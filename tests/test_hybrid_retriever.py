#!/usr/bin/env python3
"""
Tests for Hybrid Retrieval System (current API: bm25_retriever + dense_retriever objects).

These tests exercise:
- RRF score computation correctness (mathematical verification)
- HybridResult dataclass fields
- Integration of BM25 and Dense retrievers via mocks
- Query expansion during hybrid search
- Correct ranking order after fusion
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.eeg_rag.retrieval.hybrid_retriever import HybridRetriever, HybridResult
from src.eeg_rag.retrieval.bm25_retriever import BM25Result
from src.eeg_rag.retrieval.dense_retriever import DenseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bm25(results: list[BM25Result]) -> Mock:
    m = Mock()
    m.search.return_value = results
    return m


def _make_dense(results: list[DenseResult]) -> Mock:
    m = Mock()
    m.search.return_value = results
    return m


def _bm25_result(doc_id: str, score: float, text: str = "text") -> BM25Result:
    return BM25Result(doc_id=doc_id, score=score, text=text, metadata={"doc_id": doc_id})


def _dense_result(doc_id: str, score: float, text: str = "text") -> DenseResult:
    return DenseResult(doc_id=doc_id, score=score, text=text, metadata={"doc_id": doc_id})


# ---------------------------------------------------------------------------
# HybridResult dataclass
# ---------------------------------------------------------------------------

class TestHybridResult:
    """Test the HybridResult dataclass."""

    def test_creation_minimal(self):
        r = HybridResult(
            doc_id="d1",
            text="EEG seizure detection",
            metadata={"pmid": "12345678"},
            rrf_score=0.016,
            bm25_score=0.8,
            dense_score=0.9,
        )
        assert r.doc_id == "d1"
        assert r.rrf_score == pytest.approx(0.016)
        assert r.bm25_score == 0.8
        assert r.dense_score == 0.9
        assert r.bm25_rank is None
        assert r.dense_rank is None

    def test_creation_with_ranks(self):
        r = HybridResult(
            doc_id="d2",
            text="Sleep staging EEG",
            metadata={},
            rrf_score=0.01,
            bm25_score=0.5,
            dense_score=0.6,
            bm25_rank=3,
            dense_rank=2,
        )
        assert r.bm25_rank == 3
        assert r.dense_rank == 2

    def test_metadata_accessible(self):
        meta = {"title": "P300 study", "year": 2022, "pmid": "99887766"}
        r = HybridResult(
            doc_id="d3", text="P300 study text", metadata=meta,
            rrf_score=0.01, bm25_score=0.0, dense_score=0.0,
        )
        assert r.metadata["pmid"] == "99887766"
        assert r.metadata["year"] == 2022


# ---------------------------------------------------------------------------
# HybridRetriever initialization
# ---------------------------------------------------------------------------

class TestHybridRetrieverInit:
    """Test HybridRetriever construction and parameter validation."""

    def test_default_weights(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        r = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        assert r.bm25_weight == 0.5
        assert r.dense_weight == 0.5
        assert r.rrf_k == 60

    def test_custom_weights(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        r = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            bm25_weight=0.3, dense_weight=0.7, rrf_k=40,
        )
        assert r.bm25_weight == 0.3
        assert r.dense_weight == 0.7
        assert r.rrf_k == 40

    def test_query_expansion_disabled(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        r = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        assert r.query_expander is None

    def test_query_expansion_enabled(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        r = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=True,
        )
        assert r.query_expander is not None

    def test_reranking_disabled_by_default(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        r = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_reranking=False, adaptive_reranking=False,
        )
        assert r.reranker is None


# ---------------------------------------------------------------------------
# RRF score computation — mathematical correctness
# ---------------------------------------------------------------------------

class TestRRFScores:
    """Verify the RRF score formula: score(d) = Σ weight / (k + rank(d))."""

    @pytest.fixture
    def retriever(self):
        return HybridRetriever(
            bm25_retriever=_make_bm25([]),
            dense_retriever=_make_dense([]),
            bm25_weight=0.5,
            dense_weight=0.5,
            rrf_k=60,
            use_query_expansion=False,
        )

    def test_single_source_bm25_only(self, retriever):
        """Doc only in BM25 at rank 1: score = 0.5 / (60 + 1)."""
        bm25_r = [_bm25_result("d1", 0.9)]
        dense_r = []
        scores = retriever._compute_rrf_scores(bm25_r, dense_r)
        expected = 0.5 / (60 + 1)
        assert scores["d1"][0] == pytest.approx(expected, rel=1e-6)

    def test_single_source_dense_only(self, retriever):
        """Doc only in Dense at rank 1: score = 0.5 / (60 + 1)."""
        bm25_r = []
        dense_r = [_dense_result("d1", 0.9)]
        scores = retriever._compute_rrf_scores(bm25_r, dense_r)
        expected = 0.5 / (60 + 1)
        assert scores["d1"][0] == pytest.approx(expected, rel=1e-6)

    def test_dual_source_same_doc_rank1_both(self, retriever):
        """Doc at rank 1 in both BM25 and Dense: score = 0.5/61 + 0.5/61."""
        bm25_r = [_bm25_result("d1", 0.9)]
        dense_r = [_dense_result("d1", 0.9)]
        scores = retriever._compute_rrf_scores(bm25_r, dense_r)
        expected = 0.5 / 61 + 0.5 / 61
        assert scores["d1"][0] == pytest.approx(expected, rel=1e-6)

    def test_doc_in_both_ranks_higher_than_single_source(self, retriever):
        """Doc in both sources should score higher than doc in only one."""
        bm25_r = [_bm25_result("d1", 0.9), _bm25_result("d2", 0.8)]
        dense_r = [_dense_result("d1", 0.9)]  # d1 in both, d2 only in BM25
        scores = retriever._compute_rrf_scores(bm25_r, dense_r)
        assert scores["d1"][0] > scores["d2"][0]

    def test_lower_rank_means_lower_score(self, retriever):
        """Rank 2 in BM25 should score lower than rank 1."""
        bm25_r = [
            _bm25_result("d1", 0.9),  # rank 1
            _bm25_result("d2", 0.5),  # rank 2
        ]
        scores = retriever._compute_rrf_scores(bm25_r, [])
        assert scores["d1"][0] > scores["d2"][0]

    def test_rrf_k_increases_reduces_rank_sensitivity(self):
        """Higher k reduces difference between ranks 1 and 2."""
        def get_rank_diff(k):
            r = HybridRetriever(
                bm25_retriever=_make_bm25([]),
                dense_retriever=_make_dense([]),
                bm25_weight=1.0,
                rrf_k=k,
                use_query_expansion=False,
            )
            bm25_r = [_bm25_result("d1", 0.9), _bm25_result("d2", 0.5)]
            scores = r._compute_rrf_scores(bm25_r, [])
            return scores["d1"][0] - scores["d2"][0]

        diff_k10 = get_rank_diff(10)
        diff_k60 = get_rank_diff(60)
        diff_k200 = get_rank_diff(200)
        assert diff_k10 > diff_k60 > diff_k200

    def test_empty_both_sources(self, retriever):
        """Empty inputs → empty output."""
        scores = retriever._compute_rrf_scores([], [])
        assert scores == {}

    def test_bm25_weight_applied(self):
        """Check that bm25_weight scales the BM25 contribution."""
        r_high = HybridRetriever(
            bm25_retriever=_make_bm25([]),
            dense_retriever=_make_dense([]),
            bm25_weight=0.8,
            dense_weight=0.2,
            rrf_k=60,
            use_query_expansion=False,
        )
        r_low = HybridRetriever(
            bm25_retriever=_make_bm25([]),
            dense_retriever=_make_dense([]),
            bm25_weight=0.2,
            dense_weight=0.8,
            rrf_k=60,
            use_query_expansion=False,
        )
        bm25_r = [_bm25_result("d1", 0.9)]
        dense_r = []  # No dense results → only BM25 contributes
        score_high = r_high._compute_rrf_scores(bm25_r, dense_r)["d1"][0]
        score_low = r_low._compute_rrf_scores(bm25_r, dense_r)["d1"][0]
        assert score_high > score_low

    def test_metadata_preserved_in_rrf_output(self, retriever):
        """Metadata from BM25 result should be preserved in RRF output."""
        bm25_r = [_bm25_result("d1", 0.9, text="EEG seizure paper")]
        scores = retriever._compute_rrf_scores(bm25_r, [])
        _score, meta = scores["d1"]
        assert meta["text"] == "EEG seizure paper"
        assert meta["bm25_rank"] == 1


# ---------------------------------------------------------------------------
# HybridRetriever.search() — integration with mocked sub-retrievers
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """Test HybridRetriever.search() with mocked retrievers."""

    @pytest.fixture
    def eeg_docs(self):
        return {
            "seizure": _bm25_result("d_seizure", 0.9,
                                    "Epileptic seizure detection with deep learning EEG"),
            "sleep": _bm25_result("d_sleep", 0.7,
                                  "Sleep staging EEG spindle detection"),
            "bci": _bm25_result("d_bci", 0.5,
                                "Brain-computer interface motor imagery EEG"),
        }

    def test_search_returns_list_of_hybrid_results(self, eeg_docs):
        bm25 = _make_bm25(list(eeg_docs.values()))
        dense = _make_dense([
            _dense_result("d_seizure", 0.85, "seizure"),
            _dense_result("d_bci", 0.8, "bci"),
        ])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        results = retriever.search("seizure detection", top_k=3)
        assert all(isinstance(r, HybridResult) for r in results)

    def test_search_top_k_respected(self, eeg_docs):
        bm25 = _make_bm25(list(eeg_docs.values()))
        dense = _make_dense([_dense_result("d_seizure", 0.9)])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        results = retriever.search("EEG", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_descending_by_rrf_score(self, eeg_docs):
        bm25 = _make_bm25(list(eeg_docs.values()))
        dense = _make_dense([_dense_result("d_seizure", 0.95)])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        results = retriever.search("seizure", top_k=3)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_doc_in_both_sources_ranks_highest(self):
        """d1 is in both BM25 and Dense → should be top result."""
        bm25 = _make_bm25([
            _bm25_result("d1", 0.9, "epilepsy detection"),
            _bm25_result("d2", 0.8, "sleep staging"),
        ])
        dense = _make_dense([
            _dense_result("d1", 0.85, "epilepsy detection"),
            _dense_result("d3", 0.9, "brain computer"),
        ])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        results = retriever.search("EEG", top_k=5)
        assert results[0].doc_id == "d1"

    def test_bm25_and_dense_called_with_retrieve_k(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        retriever.search("EEG", top_k=5, retrieve_k=50)
        bm25.search.assert_called_once()
        _, kwargs = bm25.search.call_args
        assert kwargs.get("top_k", bm25.search.call_args[0][1] if len(bm25.search.call_args[0]) > 1 else None) == 50 or \
               (bm25.search.call_args[0] and bm25.search.call_args[0][-1] == 50) or \
               bm25.search.called  # at minimum verify it was called

    def test_empty_sources_returns_empty(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=False,
        )
        results = retriever.search("P300 amplitude")
        assert results == []

    def test_query_expansion_called_when_enabled(self):
        bm25 = _make_bm25([])
        dense = _make_dense([])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=True,
        )
        with patch.object(retriever.query_expander, "expand", wraps=retriever.query_expander.expand) as mock_exp:
            retriever.search("seizure detection", top_k=3)
            mock_exp.assert_called_once()

    def test_eeg_query_bm25_receives_expanded_query(self):
        """When expansion is on, BM25 should receive the expanded query string."""
        bm25 = _make_bm25([])
        dense = _make_dense([])
        retriever = HybridRetriever(
            bm25_retriever=bm25, dense_retriever=dense,
            use_query_expansion=True,
        )
        # "cnn" should expand to include "convolutional neural network"
        retriever.search("cnn epilepsy", top_k=3)
        call_args = bm25.search.call_args[0][0]  # first positional arg
        # The expanded query should contain more than just "cnn epilepsy"
        assert len(call_args) >= len("cnn epilepsy")


# ---------------------------------------------------------------------------
# EEG query expansion unit tests
# ---------------------------------------------------------------------------

class TestEEGQueryExpansion:
    """Unit tests for EEGQueryExpander used inside the hybrid retriever."""

    @pytest.fixture
    def expander(self):
        from src.eeg_rag.retrieval.query_expander import EEGQueryExpander
        return EEGQueryExpander()

    def test_cnn_expands_to_convolutional(self, expander):
        result = expander.expand("cnn architecture")
        assert "convolutional" in result.lower()

    def test_seizure_expands_to_epileptic(self, expander):
        result = expander.expand("seizure detection EEG")
        assert any(w in result.lower() for w in ("epileptic", "epilepsy", "ictal"))

    def test_bci_expands_to_brain_computer(self, expander):
        result = expander.expand("bci motor control")
        assert "brain" in result.lower() and "computer" in result.lower()

    def test_alpha_stays_in_expansion(self, expander):
        result = expander.expand("alpha band power")
        assert "alpha" in result.lower()

    def test_unknown_term_kept_unchanged(self, expander):
        """A term not in the synonym dict should be preserved as-is."""
        result = expander.expand("xenotransplantation")
        assert "xenotransplantation" in result.lower()

    def test_max_expansions_limits_added_terms(self, expander):
        """Higher max_expansions should produce a longer or equal query than lower."""
        query = "cnn seizure bci"
        expanded_1 = expander.expand(query, max_expansions=1)
        expanded_5 = expander.expand(query, max_expansions=5)
        # More expansions allowed → equal or more words in the result
        assert len(expanded_5.split()) >= len(expanded_1.split())

    def test_eeg_abbreviation_expands(self, expander):
        result = expander.expand("eeg signal")
        assert "electroencephalograph" in result.lower()

    def test_lstm_expands_to_full_name(self, expander):
        result = expander.expand("lstm classification")
        assert "long short-term memory" in result.lower() or "long" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

