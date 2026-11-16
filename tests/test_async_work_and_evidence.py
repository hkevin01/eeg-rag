"""Deterministic concurrency and NLI evidence tests."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_rag.utils.async_work import BoundedWorker
from eeg_rag.utils.ncbi import NcbiRateLimiter, get_ncbi_limiter


@pytest.mark.asyncio
async def test_ncbi_clients_share_one_request_budget():
    assert get_ncbi_limiter() is get_ncbi_limiter()
    now = [10.0]
    delays = []

    async def advance(seconds):
        delays.append(seconds)
        now[0] += seconds

    limiter = NcbiRateLimiter(3, clock=lambda: now[0], delay=advance)
    await asyncio.gather(*(limiter.acquire() for _ in range(4)))
    assert delays == pytest.approx([1 / 3] * 3)
    assert now[0] == pytest.approx(11.0)


@pytest.mark.asyncio
async def test_cancelled_blocking_work_retains_capacity():
    worker = BoundedWorker(1)
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()

    def blocking():
        started.set()
        assert release.wait(timeout=5)

    first = asyncio.create_task(worker.run(blocking))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        second = asyncio.create_task(worker.run(second_started.set))
        await asyncio.sleep(0)
        assert not second_started.is_set()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert second_started.is_set()


@pytest.mark.parametrize(
    "scores,status",
    [
        ([0.01, 0.98, 0.01], "supported"),
        ([0.98, 0.01, 0.01], "unsupported"),
        ([0.01, 0.01, 0.98], "unknown"),
    ],
)
def test_nli_label_mapping_preserves_contradictions(scores, status):
    from eeg_rag.verification.entailment import NLIClaimEvaluator

    model = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([scores]))
    evaluator = NLIClaimEvaluator(model=model)
    result = evaluator("EEG does not detect spikes.", "EEG detects spikes.")
    assert result.status.value == status
    assert result.span == "EEG detects spikes."


def test_nli_rejects_invalid_probabilities():
    from eeg_rag.verification.entailment import NLIClaimEvaluator

    model = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([[np.nan, 0, 0]]))
    evaluator = NLIClaimEvaluator(model=model)
    assert evaluator("P300 claim", "P300 source").status.value == "unknown"


@pytest.mark.asyncio
async def test_local_retrieval_runs_outside_event_loop_thread():
    from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent

    main_thread = threading.get_ident()
    observed_threads = []

    def search(**kwargs):
        observed_threads.append(threading.get_ident())
        return []

    agent = LocalDataAgent(use_hybrid_retrieval=False)
    agent.hybrid_retriever = SimpleNamespace(search=search)
    agent.retrieve_k = 50
    assert await agent._hybrid_search("Sleep EEG") == []
    assert observed_threads and observed_threads[0] != main_thread
