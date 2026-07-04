"""Unit tests for ranking-quality formulas in benchmark aggregation strategy evaluation."""

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


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

    sent_util_mod = types.ModuleType("sentence_transformers.util")

    def _cos_sim(*args, **kwargs):
        return 0.0

    setattr(sent_util_mod, "cos_sim", _cos_sim)
    sys.modules["sentence_transformers.util"] = sent_util_mod


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


def test_hard_archetype_uncertainty_guard_raises_on_regression() -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.hard_archetype_utility_margin = 0.02

    ranking_comparison = {
        "weighted": {
            "hard_archetype_uncertainty_adjusted_utility": 0.64,
        },
        "concept_aware": {
            "hard_archetype_uncertainty_adjusted_utility": 0.61,
        },
    }

    with pytest.raises(RuntimeError):
        benchmark._enforce_uncertainty_adjusted_utility_guard(ranking_comparison)


def test_monotonic_safety_response_validation_detects_noncontraction() -> None:
    benchmark = _benchmark_instance()

    profiles = {
        "clinical_hard": {
            "low_risk_max_step": 0.18,
            "medium_risk_max_step": 0.12,
            "high_risk_max_step": 0.08,
        },
        "method_hard": {
            "low_risk_max_step": 0.16,
            "medium_risk_max_step": 0.11,
            "high_risk_max_step": 0.09,
        },
    }
    validation = benchmark._validate_monotonic_safety_response(profiles)
    assert validation["valid"] is True

    profiles["bci_hard"] = {
        "low_risk_max_step": 0.10,
        "medium_risk_max_step": 0.12,
        "high_risk_max_step": 0.07,
    }
    failing = benchmark._validate_monotonic_safety_response(profiles)
    assert failing["valid"] is False
    assert any("bci_hard" in item for item in failing["failing"])


def test_temporal_forgetting_validation_requires_utility_gain_and_citation_floor() -> None:
    benchmark = _benchmark_instance()

    before = {
        "hard_archetype_uncertainty_adjusted_utility": 0.58,
        "citation_validity_proxy": 0.70,
    }
    after_good = {
        "hard_archetype_uncertainty_adjusted_utility": 0.63,
        "citation_validity_proxy": 0.69,
    }

    ok = benchmark._validate_temporal_forgetting_safety(before, after_good)
    assert ok["valid"] is True
    assert ok["hard_utility_delta"] > 0.0

    after_bad = {
        "hard_archetype_uncertainty_adjusted_utility": 0.64,
        "citation_validity_proxy": 0.55,
    }
    failed = benchmark._validate_temporal_forgetting_safety(before, after_bad)
    assert failed["valid"] is False


def test_ranking_guard_fails_on_monotonic_safety_regression() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_concept_aware_ranking_ndcg = 0.5
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.4,
    }
    benchmark.hard_archetype_utility_margin = 0.0

    ranking_comparison = {
        "weighted": {
            "hard_archetype_uncertainty_adjusted_utility": 0.60,
        },
        "concept_aware": {
            "ranking_ndcg": 0.72,
            "hard_archetype_uncertainty_adjusted_utility": 0.66,
            "per_archetype": {
                "clinical_hard": {
                    "difficulty": "hard",
                    "ranking_ndcg": 0.70,
                }
            },
            "monotonic_safety_response": {
                "valid": False,
                "failing": ["clinical_hard: low=0.08, medium=0.11, high=0.07"],
            },
            "temporal_forgetting_validation": {
                "valid": True,
            },
        },
    }

    with pytest.raises(RuntimeError, match="Monotonic safety response regression"):
        benchmark._enforce_ranking_regression_guard(ranking_comparison)


def test_ranking_guard_fails_on_temporal_forgetting_safety_regression() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_concept_aware_ranking_ndcg = 0.5
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.4,
    }
    benchmark.hard_archetype_utility_margin = 0.0

    ranking_comparison = {
        "weighted": {
            "hard_archetype_uncertainty_adjusted_utility": 0.60,
        },
        "concept_aware": {
            "ranking_ndcg": 0.72,
            "hard_archetype_uncertainty_adjusted_utility": 0.66,
            "per_archetype": {
                "clinical_hard": {
                    "difficulty": "hard",
                    "ranking_ndcg": 0.70,
                }
            },
            "monotonic_safety_response": {
                "valid": True,
                "failing": [],
            },
            "temporal_forgetting_validation": {
                "valid": False,
                "hard_utility_delta": -0.01,
                "citation_validity_delta": -0.04,
                "citation_validity_floor": 0.62,
            },
        },
    }

    with pytest.raises(RuntimeError, match="Temporal forgetting safety regression"):
        benchmark._enforce_ranking_regression_guard(ranking_comparison)


def test_hard_archetype_utility_guard_uses_confidence_interval_lower_bound() -> None:
    benchmark = _benchmark_instance()
    benchmark.hard_archetype_utility_margin = 0.02

    ranking_comparison = {
        "weighted": {
            "hard_archetype_uncertainty_adjusted_utility": 0.62,
        },
        "concept_aware": {
            "hard_archetype_uncertainty_adjusted_utility": 0.66,
            "hard_archetype_utility_delta_ci": {
                "mean": 0.04,
                "ci_lower": 0.01,
                "ci_upper": 0.07,
                "samples": 12,
            },
        },
    }

    with pytest.raises(RuntimeError, match="Confidence-bounded utility regression"):
        benchmark._enforce_uncertainty_adjusted_utility_guard(ranking_comparison)


def test_temporal_forgetting_validation_applies_stricter_category_floors() -> None:
    benchmark = _benchmark_instance()
    benchmark.category_adaptive_safety_floors = {
        "general": {
            "citation_validity_floor": 0.62,
            "hard_utility_margin": 0.0,
        },
        "clinical": {
            "citation_validity_floor": 0.72,
            "hard_utility_margin": 0.02,
        },
    }

    before = {
        "hard_archetype_uncertainty_adjusted_utility": 0.58,
        "citation_validity_proxy": 0.71,
    }
    after = {
        "hard_archetype_uncertainty_adjusted_utility": 0.60,
        "citation_validity_proxy": 0.70,
    }
    before_per = {
        "clinical_hard": {
            "category": "clinical",
            "difficulty": "hard",
            "uncertainty_adjusted_utility": 0.58,
            "citation_validity_proxy": 0.73,
        }
    }
    after_per = {
        "clinical_hard": {
            "category": "clinical",
            "difficulty": "hard",
            "uncertainty_adjusted_utility": 0.59,
            "citation_validity_proxy": 0.71,
        }
    }

    validation = benchmark._validate_temporal_forgetting_safety(
        before,
        after,
        before_per_archetype=before_per,
        after_per_archetype=after_per,
    )
    assert validation["valid"] is False
    assert validation["category_checks"]["clinical"]["valid"] is False


def test_export_includes_adaptive_safety_summary_fields(tmp_path) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    BenchmarkSuite = benchmarking_mod.BenchmarkSuite

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    out_path = tmp_path / "benchmark_export_adaptive_safety.json"

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
                "hard_archetype_utility_delta_ci": {
                    "mean": 0.03,
                    "ci_lower": 0.01,
                    "ci_upper": 0.05,
                    "samples": 10,
                },
                "monotonic_safety_response": {"valid": True},
                "temporal_forgetting_validation": {"valid": True},
                "risk_to_step_model": {"heldout_mse": 0.002},
                "hard_archetype_utility_delta_by_category": {
                    "clinical": {"mean": 0.04}
                },
            }
        },
    )

    benchmark.export_benchmark_results(suite, out_path)
    payload = json.loads(out_path.read_text())
    summary = payload["summary"]

    assert "adaptive_safety" in summary
    assert "concept_aware_hard_utility_delta_ci" in summary
    assert "risk_to_step_model" in summary["adaptive_safety"]


def test_end_to_end_strategy_comparison_triggers_guard_under_synthetic_drift(monkeypatch) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.min_concept_aware_ranking_ndcg = 0.45
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.35,
        "medium": 0.35,
        "hard": 0.35,
    }
    benchmark.hard_archetype_utility_margin = 0.0
    benchmark.hard_archetype_delta_ci_alpha = 0.05
    benchmark.bootstrap_samples = 200
    benchmark.risk_to_step_ridge_lambda = 0.15
    benchmark.category_adaptive_safety_floors = {
        "general": {"citation_validity_floor": 0.60, "hard_utility_margin": 0.0},
        "clinical": {"citation_validity_floor": 0.72, "hard_utility_margin": 0.02},
        "bci": {"citation_validity_floor": 0.70, "hard_utility_margin": 0.01},
    }
    benchmark._calibration_drift_state = {
        "baseline_mae": None,
        "last_mae": None,
        "last_relative_shift": 0.0,
        "drift_detected": False,
        "recalibration_recommended": False,
        "checked_at": None,
    }
    benchmark._utility_weights = {
        "concept": 0.50,
        "centrality": 0.30,
        "novelty": 0.20,
    }

    fixtures = [
        {
            "name": "clinical_hard",
            "category": "clinical",
            "difficulty": "hard",
            "query": "clinical hard eeg",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.20,
                            "query_concept_coverage_score": 0.82,
                        },
                        "centralities": [0.82, 0.75],
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.30,
                            "query_concept_coverage_score": 0.75,
                        },
                        "centralities": [0.70, 0.66],
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.55,
                            "query_concept_coverage_score": 0.58,
                        },
                        "centralities": [0.52, 0.49],
                    },
                }
            },
        },
        {
            "name": "bci_hard",
            "category": "bci",
            "difficulty": "hard",
            "query": "bci hard eeg",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.22,
                            "query_concept_coverage_score": 0.80,
                        },
                        "centralities": [0.78, 0.74],
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.33,
                            "query_concept_coverage_score": 0.73,
                        },
                        "centralities": [0.69, 0.65],
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.52,
                            "query_concept_coverage_score": 0.57,
                        },
                        "centralities": [0.50, 0.47],
                    },
                }
            },
        },
    ]

    monkeypatch.setattr(
        benchmark,
        "_create_archetype_fixture_bank",
        lambda: fixtures,
    )

    class _FakeAggregator:
        def __init__(self, relevance_threshold, max_citations, entity_min_frequency, ranking_strategy):
            _ = (relevance_threshold, max_citations, entity_min_frequency)
            self.ranking_strategy = ranking_strategy

        def _extract_query_concepts(self, query):
            _ = query
            return {"concept": ["eeg"]}

        async def aggregate(self, query, fixture_results):
            _ = query
            scenario = fixture_results["scenario"][self.ranking_strategy]
            centralities = scenario["centralities"]
            citations = [
                SimpleNamespace(
                    title="eeg concept citation",
                    abstract="eeg evidence",
                    pmid=f"p{idx}",
                    metadata={"centrality_score": value},
                )
                for idx, value in enumerate(centralities)
            ]
            return SimpleNamespace(citations=citations, statistics=scenario["stats"])

    monkeypatch.setattr(benchmarking_mod, "ContextAggregator", _FakeAggregator)

    ranking = asyncio.run(benchmark._benchmark_aggregation_strategies())
    with pytest.raises(RuntimeError):
        benchmark._enforce_ranking_regression_guard(ranking)


def test_end_to_end_strategy_comparison_passes_under_stable_conditions(monkeypatch) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.min_concept_aware_ranking_ndcg = 0.45
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.35,
        "medium": 0.35,
        "hard": 0.35,
    }
    benchmark.hard_archetype_utility_margin = -0.05
    benchmark.hard_archetype_delta_ci_alpha = 0.05
    benchmark.bootstrap_samples = 200
    benchmark.risk_to_step_ridge_lambda = 0.15
    benchmark.category_adaptive_safety_floors = {
        "general": {"citation_validity_floor": 0.60, "hard_utility_margin": -0.01},
        "clinical": {"citation_validity_floor": 0.66, "hard_utility_margin": -0.01},
        "bci": {"citation_validity_floor": 0.64, "hard_utility_margin": -0.01},
    }
    benchmark._calibration_drift_state = {
        "baseline_mae": None,
        "last_mae": None,
        "last_relative_shift": 0.0,
        "drift_detected": False,
        "recalibration_recommended": False,
        "checked_at": None,
    }
    benchmark._utility_weights = {
        "concept": 0.50,
        "centrality": 0.30,
        "novelty": 0.20,
    }

    fixtures = [
        {
            "name": "clinical_hard",
            "category": "clinical",
            "difficulty": "hard",
            "query": "clinical hard eeg",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.24,
                            "query_concept_coverage_score": 0.78,
                        },
                        "centralities": [0.76, 0.72],
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.28,
                            "query_concept_coverage_score": 0.76,
                        },
                        "centralities": [0.73, 0.70],
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.18,
                            "query_concept_coverage_score": 0.84,
                        },
                        "centralities": [0.85, 0.80],
                    },
                }
            },
        },
        {
            "name": "bci_hard",
            "category": "bci",
            "difficulty": "hard",
            "query": "bci hard eeg",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.26,
                            "query_concept_coverage_score": 0.76,
                        },
                        "centralities": [0.74, 0.70],
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.30,
                            "query_concept_coverage_score": 0.73,
                        },
                        "centralities": [0.70, 0.67],
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.19,
                            "query_concept_coverage_score": 0.82,
                        },
                        "centralities": [0.83, 0.79],
                    },
                }
            },
        },
    ]

    monkeypatch.setattr(
        benchmark,
        "_create_archetype_fixture_bank",
        lambda: fixtures,
    )

    class _FakeAggregator:
        def __init__(self, relevance_threshold, max_citations, entity_min_frequency, ranking_strategy):
            _ = (relevance_threshold, max_citations, entity_min_frequency)
            self.ranking_strategy = ranking_strategy

        def _extract_query_concepts(self, query):
            _ = query
            return {"concept": ["eeg"]}

        async def aggregate(self, query, fixture_results):
            _ = query
            scenario = fixture_results["scenario"][self.ranking_strategy]
            centralities = scenario["centralities"]
            citations = [
                SimpleNamespace(
                    title="eeg concept citation",
                    abstract="eeg evidence",
                    pmid=f"p{idx}",
                    metadata={"centrality_score": value},
                )
                for idx, value in enumerate(centralities)
            ]
            return SimpleNamespace(citations=citations, statistics=scenario["stats"])

    monkeypatch.setattr(benchmarking_mod, "ContextAggregator", _FakeAggregator)

    ranking = asyncio.run(benchmark._benchmark_aggregation_strategies())
    benchmark._enforce_ranking_regression_guard(ranking)
