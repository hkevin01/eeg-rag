"""Unit tests for ranking-quality formulas in benchmark aggregation strategy evaluation."""

import importlib
import json
import sys
import types
from types import SimpleNamespace


def _install_dependency_stubs() -> None:
    """Install lightweight module stubs for optional heavy dependencies."""
    bm25_mod = types.ModuleType("rank_bm25")
    setattr(bm25_mod, "BM25Okapi", object)
    sys.modules["rank_bm25"] = bm25_mod

    sent_mod = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return 384

    setattr(sent_mod, "SentenceTransformer", _DummySentenceTransformer)
    sys.modules["sentence_transformers"] = sent_mod


def _import_benchmark_module():
    """Import benchmarking with lightweight stubs for unrelated heavy modules."""
    _install_dependency_stubs()

    orchestrator_mod = types.ModuleType(
        "src.eeg_rag.agents.orchestrator.orchestrator_agent"
    )
    setattr(orchestrator_mod, "OrchestratorAgent", object)
    sys.modules[
        "src.eeg_rag.agents.orchestrator.orchestrator_agent"
    ] = orchestrator_mod

    local_mod = types.ModuleType("src.eeg_rag.agents.local_agent.local_data_agent")
    setattr(local_mod, "LocalDataAgent", object)
    sys.modules["src.eeg_rag.agents.local_agent.local_data_agent"] = local_mod

    web_mod = types.ModuleType("src.eeg_rag.agents.web_agent.web_search_agent")
    setattr(web_mod, "WebSearchAgent", object)
    sys.modules["src.eeg_rag.agents.web_agent.web_search_agent"] = web_mod

    return importlib.import_module("src.eeg_rag.evaluation.benchmarking")


def _benchmark_instance():
    """Create benchmark instance without invoking heavy constructor dependencies."""
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    return EEGRAGBenchmark.__new__(EEGRAGBenchmark)


def test_compute_ndcg_perfect_ranking_is_one() -> None:
    benchmark = _benchmark_instance()

    ndcg = benchmark._compute_ndcg([1.0, 0.8, 0.3], k=3)

    assert ndcg == 1.0


def test_compute_ndcg_degrades_for_suboptimal_order() -> None:
    benchmark = _benchmark_instance()

    perfect = benchmark._compute_ndcg([1.0, 0.8, 0.3], k=3)
    suboptimal = benchmark._compute_ndcg([0.3, 0.8, 1.0], k=3)

    assert perfect > suboptimal
    assert 0.0 <= suboptimal < 1.0


def test_citation_utility_rewards_novel_concept_coverage() -> None:
    benchmark = _benchmark_instance()

    query_concepts = {
        "method": ["ica"],
        "outcome": ["sensitivity"],
    }

    first_citation = SimpleNamespace(
        title="ICA improves EEG signal quality",
        abstract="Method-focused preprocessing report",
        metadata={"centrality_score": 1.0},
    )
    second_citation = SimpleNamespace(
        title="ICA replication in EEG cohort",
        abstract="Another method-only report",
        metadata={"centrality_score": 1.0},
    )

    first_utility, seen = benchmark._compute_citation_utility(
        citation=first_citation,
        query_concepts=query_concepts,
        max_centrality=1.0,
        seen_concept_groups=set(),
    )
    second_utility, _ = benchmark._compute_citation_utility(
        citation=second_citation,
        query_concepts=query_concepts,
        max_centrality=1.0,
        seen_concept_groups=seen,
    )

    assert first_utility > second_utility


def test_export_includes_ranking_ci_and_drift_fields(tmp_path) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    BenchmarkSuite = benchmarking_mod.BenchmarkSuite

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    out_path = tmp_path / "benchmark_export.json"

    suite = BenchmarkSuite(
        retrieval_results=[],
        generation_results=[],
        end_to_end_results=[],
        overall_score=0.81,
        retrieval_score=0.83,
        generation_score=0.79,
        avg_total_time_ms=420.0,
        avg_citation_accuracy=0.92,
        avg_response_quality=0.84,
        avg_redundancy_score=0.18,
        avg_diversity_score=0.72,
        avg_query_entity_coverage_score=0.71,
        avg_query_concept_coverage_score=0.74,
        avg_centrality_grounding_score=0.69,
        avg_grounding_quality=0.75,
        concept_aware_grounding_score=0.77,
        concept_aware_ranking_ndcg=0.82,
        ranking_strategy_comparison={
            "concept_aware": {
                "ranking_ndcg_ci_lower": 0.76,
                "ranking_ndcg_ci_upper": 0.87,
                "calibration_drift": {
                    "drift": 0.04,
                    "status": "ok",
                },
            }
        },
    )

    benchmark.export_benchmark_results(suite, out_path)

    payload = json.loads(out_path.read_text())
    summary = payload["summary"]
    assert summary["concept_aware_ranking_ndcg_ci"]["lower"] == 0.76
    assert summary["concept_aware_ranking_ndcg_ci"]["upper"] == 0.87
    assert summary["concept_aware_calibration_drift"]["status"] == "ok"
