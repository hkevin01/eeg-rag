"""Eligibility and candidate-depth contracts for hybrid retrieval."""

from types import SimpleNamespace

import pytest

from eeg_rag.retrieval.bm25_retriever import BM25Result
from eeg_rag.retrieval.hybrid_retriever import HybridRetriever


class SparseCandidates:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=100, filter_fn=None):
        eligible = [
            result
            for result in self.results
            if filter_fn is None or filter_fn({"metadata": result.metadata})
        ]
        return eligible[:top_k]


class EmptyDense:
    def search(self, query, **kwargs):
        return []


def test_sparse_only_paper_must_satisfy_year_filter():
    sparse = SparseCandidates(
        [
            BM25Result("old", 4.0, "EEG epilepsy", {"year": 2010}),
            BM25Result("eligible", 2.0, "EEG epilepsy", {"year": 2025}),
            BM25Result("unknown", 1.0, "EEG epilepsy", {}),
        ]
    )
    retriever = HybridRetriever(sparse, EmptyDense(), use_query_expansion=False)
    results = retriever.search("epilepsy", filters={"year": 2025}, top_k=1)
    assert [result.doc_id for result in results] == ["eligible"]


def test_range_boundaries_preserve_eligible_papers():
    sparse = SparseCandidates(
        [
            BM25Result(str(year), 1.0, "Sleep EEG", {"year": year})
            for year in (2019, 2020, 2025, 2026)
        ]
    )
    retriever = HybridRetriever(sparse, EmptyDense(), use_query_expansion=False)
    results = retriever.search("sleep", filters={"year": {"gte": 2020, "lte": 2025}})
    assert {result.doc_id for result in results} == {"2020", "2025"}


def test_reranker_can_promote_candidate_below_final_cutoff():
    sparse = SparseCandidates(
        [
            BM25Result("first", 3.0, "General EEG", {}),
            BM25Result("relevant", 2.0, "P300 latency evidence", {}),
        ]
    )
    retriever = HybridRetriever(sparse, EmptyDense(), use_query_expansion=False)

    class EvidenceReranker:
        def rerank(self, query, candidates):
            return [
                SimpleNamespace(**candidate, final_score=0.9)
                for candidate in reversed(candidates)
            ]

    retriever.use_reranking = True
    retriever.reranker = EvidenceReranker()
    results = retriever.search("P300", top_k=1, retrieve_k=10)
    assert [result.doc_id for result in results] == ["relevant"]
    assert results[0].rrf_score == pytest.approx(0.5 / 62)
    assert results[0].reranker_score == 0.9


@pytest.mark.parametrize("top_k,retrieve_k", [(0, 10), (-1, 10), (1, 0)])
def test_invalid_candidate_counts_are_rejected(top_k, retrieve_k):
    retriever = HybridRetriever(SparseCandidates([]), EmptyDense())
    with pytest.raises(ValueError):
        retriever.search("EEG", top_k=top_k, retrieve_k=retrieve_k)


def test_native_mode_uses_server_fusion_without_local_sparse_snapshot():
    from eeg_rag.storage.vector_db import SearchResult

    class NativeDatabase:
        def hybrid_search(self, query, **kwargs):
            return [
                SearchResult(
                    "eligible",
                    0.04,
                    {"text": "P300 evidence", "metadata": {"year": 2025}},
                )
            ]

    class ForbiddenSparse:
        def search(self, *args, **kwargs):
            raise AssertionError("Native mode must not query local BM25")

    dense = SimpleNamespace(vector_db=NativeDatabase())
    retriever = HybridRetriever(
        ForbiddenSparse(), dense, native_qdrant=True, use_query_expansion=False
    )
    result = retriever.search("P300", top_k=1, filters={"year": 2025})
    assert result[0].doc_id == "eligible"
    assert result[0].rrf_score == 0.04
