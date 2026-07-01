"""Unit tests for ranking-quality formulas in benchmark aggregation strategy evaluation."""

import importlib
import sys
import types
from types import SimpleNamespace


def _import_benchmark_module():
    """Import benchmarking with lightweight stubs for unrelated heavy modules."""
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
