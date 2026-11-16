"""Honest retrieval evaluation contracts for the upgrade benchmark."""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_benchmark_penalizes_missing_relevant_documents():
    from eeg_rag.evaluation.upgrade_benchmark import benchmark_retrieval

    class Retriever:
        def search(self, query, **kwargs):
            return [SimpleNamespace(doc_id="first")]

    report = await benchmark_retrieval(
        Retriever(),
        [
            {
                "query": "P300 latency",
                "relevant_ids": ["first", "missed"],
            }
        ],
        rerank_counts=[30],
        repetitions=2,
    )
    assert report[0]["recall_at_50"] == 0.5
    assert report[0]["ndcg_at_10"] < 1
    assert report[0]["sample_count"] == 2
    assert report[0]["p95_ms"] >= 0


@pytest.mark.asyncio
async def test_benchmark_rejects_unlabeled_or_duplicate_query_rows():
    from eeg_rag.evaluation.upgrade_benchmark import benchmark_retrieval

    with pytest.raises(ValueError, match="relevance"):
        await benchmark_retrieval(None, [{"query": "Sleep EEG", "relevant_ids": []}])
