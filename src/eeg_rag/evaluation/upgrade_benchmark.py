"""Repeatable candidate-depth and backend evaluation on independently labeled data."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from eeg_rag.evaluation.retrieval_metrics import RetrievalEvaluator
from eeg_rag.utils.async_work import BoundedWorker


async def benchmark_retrieval(
    retriever: Any,
    queries: list[dict[str, Any]],
    rerank_counts: Optional[list[int]] = None,
    repetitions: int = 3,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate quality and latency without tuning on the held-out relevance set.

    Args:
        retriever: Initialized retriever with the HybridRetriever search contract.
        queries: Unique query rows with independently assigned relevant_ids.
        rerank_counts: Candidate depths to compare, defaulting to 30, 50, 100.
        repetitions: Timed repetitions after one warm-up query per configuration.
        concurrency: Maximum concurrent blocking retrieval operations.

    Returns:
        Quality, p50/p95 latency, sample counts, and event-loop delay per depth.

    Raises:
        ValueError: Missing relevance labels or invalid benchmark settings.
    """
    counts = [30, 50, 100] if rerank_counts is None else rerank_counts
    if not queries or any(
        not row.get("query") or not row.get("relevant_ids") for row in queries
    ):
        raise ValueError("Every query needs independent relevance labels")
    if len({row["query"] for row in queries}) != len(queries):
        raise ValueError("Duplicate query rows are not permitted")
    if repetitions < 1 or not counts or any(count < 1 for count in counts):
        raise ValueError("Repetitions and reranking counts must be positive")
    worker = BoundedWorker(concurrency)
    evaluator = RetrievalEvaluator()
    reports = []
    for count in counts:
        await worker.run(
            retriever.search,
            queries[0]["query"],
            top_k=50,
            retrieve_k=max(100, count),
            rerank_k=count,
        )
        samples = []
        loop_delays = []
        finished = asyncio.Event()

        async def monitor(
            finished: asyncio.Event = finished, loop_delays: list = loop_delays
        ) -> None:
            loop = asyncio.get_running_loop()
            while not finished.is_set():
                expected = loop.time() + 0.01
                await asyncio.sleep(0.01)
                loop_delays.append(max(0, loop.time() - expected) * 1000)

        async def measure(
            row: dict[str, Any], count: int = count, samples: list = samples
        ) -> None:
            started = time.perf_counter()
            results = await worker.run(
                retriever.search,
                row["query"],
                top_k=50,
                retrieve_k=max(100, count),
                rerank_k=count,
                filters=row.get("filters"),
            )
            elapsed = (time.perf_counter() - started) * 1000
            ids = list(dict.fromkeys(result.doc_id for result in results))
            relevant = set(row["relevant_ids"])
            samples.append(
                {
                    "latency": elapsed,
                    "recall": evaluator.recall_at_k(relevant, ids, 50),
                    "ndcg": evaluator.ndcg_at_k(relevant, ids, 10),
                    "mrr": evaluator.mean_reciprocal_rank(relevant, ids[:10]),
                }
            )

        monitor_task = asyncio.create_task(monitor())
        try:
            for _ in range(repetitions):
                for start in range(0, len(queries), concurrency):
                    await asyncio.gather(
                        *(measure(row) for row in queries[start : start + concurrency])
                    )
        finally:
            finished.set()
            await monitor_task
        latencies = [sample["latency"] for sample in samples]
        reports.append(
            {
                "rerank_k": count,
                "sample_count": len(samples),
                "concurrency": concurrency,
                "recall_at_50": float(
                    np.mean([sample["recall"] for sample in samples])
                ),
                "ndcg_at_10": float(np.mean([sample["ndcg"] for sample in samples])),
                "mrr_at_10": float(np.mean([sample["mrr"] for sample in samples])),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "event_loop_delay_p95_ms": (
                    float(np.percentile(loop_delays, 95)) if loop_delays else 0.0
                ),
            }
        )
    return reports


def main() -> None:
    """Benchmark an existing corpus using JSON configuration and held-out queries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    async def run() -> None:
        from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent

        config = await asyncio.to_thread(lambda: json.loads(args.config.read_text()))
        queries = await asyncio.to_thread(lambda: json.loads(args.queries.read_text()))
        config["retrieval_concurrency"] = args.concurrency
        agent = await asyncio.to_thread(
            LocalDataAgent, config=config, use_reranking=True
        )
        try:
            report = await benchmark_retrieval(
                agent.hybrid_retriever, queries, concurrency=args.concurrency
            )
            await asyncio.to_thread(
                args.output.write_text, json.dumps(report, indent=2) + "\n"
            )
        finally:
            if hasattr(agent, "dense_retriever"):
                await asyncio.to_thread(agent.dense_retriever.vector_db.client.close)

    asyncio.run(run())


if __name__ == "__main__":
    main()
