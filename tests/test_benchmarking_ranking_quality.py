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


def test_export_includes_provider_generation_and_streaming_latency_summary(
    tmp_path,
) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    BenchmarkSuite = benchmarking_mod.BenchmarkSuite
    ProviderGenerationBenchmarkResult = (
        benchmarking_mod.ProviderGenerationBenchmarkResult
    )

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    out_path = tmp_path / "benchmark_export_provider_generation.json"

    suite = BenchmarkSuite(
        retrieval_results=[],
        generation_results=[],
        end_to_end_results=[],
        provider_generation_results=[
            ProviderGenerationBenchmarkResult(
                provider="openai",
                query="Explain alpha rhythm EEG findings",
                streaming_latency_ms=220.0,
                total_generation_time_ms=860.0,
                response_quality=0.84,
                citation_quality=0.81,
                response_length=512,
                citation_count=4,
                chunk_count=8,
                success=True,
                error_message=None,
            )
        ],
        avg_provider_streaming_latency_ms=220.0,
    )

    benchmark.export_benchmark_results(suite, out_path)

    payload = json.loads(out_path.read_text())
    summary = payload["summary"]
    details = payload["detailed_results"]
    provider_generation = details["provider_generation"]

    assert summary["avg_provider_streaming_latency_ms"] == 220.0
    assert len(provider_generation) == 1
    assert provider_generation[0]["provider"] == "openai"
    assert provider_generation[0]["streaming_latency_ms"] == 220.0


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
    benchmark.min_total_papers_per_strategy = 2
    benchmark.min_avg_papers_per_archetype = 1.0
    benchmark.min_metadata_completeness_rate = 0.40
    benchmark.min_papers_per_archetype = 1
    benchmark.min_metadata_completeness_per_archetype = 0.40
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


def test_end_to_end_strategy_comparison_large_body_and_metadata_completeness(
    monkeypatch,
) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.min_concept_aware_ranking_ndcg = 0.35
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.25,
        "medium": 0.25,
        "hard": 0.25,
    }
    benchmark.hard_archetype_utility_margin = -0.02
    benchmark.hard_archetype_delta_ci_alpha = 0.05
    benchmark.bootstrap_samples = 200
    benchmark.risk_to_step_ridge_lambda = 0.15
    benchmark.category_adaptive_safety_floors = {
        "general": {"citation_validity_floor": 0.55, "hard_utility_margin": -0.01},
        "clinical": {"citation_validity_floor": 0.60, "hard_utility_margin": -0.01},
        "bci": {"citation_validity_floor": 0.58, "hard_utility_margin": -0.01},
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

    categories = ["clinical", "bci", "method", "outcome", "longitudinal", "erp"]
    fixtures = []
    for idx in range(12):
        category = categories[idx % len(categories)]
        fixtures.append(
            {
                "name": f"{category}_archetype_{idx}",
                "category": category,
                "difficulty": "hard" if idx % 2 == 0 else "medium",
                "query": f"{category} eeg query {idx}",
                "results": {
                    "scenario": {
                        "weighted": {
                            "stats": {
                                "redundancy_score": 0.22,
                                "query_concept_coverage_score": 0.78,
                            },
                            "centrality_base": 0.72,
                            "paper_count": 28,
                        },
                        "diversified": {
                            "stats": {
                                "redundancy_score": 0.26,
                                "query_concept_coverage_score": 0.75,
                            },
                            "centrality_base": 0.70,
                            "paper_count": 28,
                        },
                        "concept_aware": {
                            "stats": {
                                "redundancy_score": 0.18,
                                "query_concept_coverage_score": 0.84,
                            },
                            "centrality_base": 0.80,
                            "paper_count": 28,
                        },
                    }
                },
            }
        )

    monkeypatch.setattr(
        benchmark,
        "_create_archetype_fixture_bank",
        lambda: fixtures,
    )

    class _FakeAggregator:
        def __init__(
            self,
            relevance_threshold,
            max_citations,
            entity_min_frequency,
            ranking_strategy,
        ):
            _ = (relevance_threshold, max_citations, entity_min_frequency)
            self.ranking_strategy = ranking_strategy

        def _extract_query_concepts(self, query):
            _ = query
            return {"clinical": ["eeg"], "method": ["artifact"]}

        async def aggregate(self, query, fixture_results):
            _ = query
            scenario = fixture_results["scenario"][self.ranking_strategy]
            paper_count = int(scenario["paper_count"])
            centrality_base = float(scenario["centrality_base"])

            citations = []
            for idx in range(paper_count):
                centrality = max(0.0, min(1.0, centrality_base - (0.01 * (idx % 5))))
                citations.append(
                    SimpleNamespace(
                        pmid=f"{self.ranking_strategy}-{idx}",
                        title=f"EEG study {idx}",
                        abstract="eeg cohort metadata rich evidence",
                        year=2020 + (idx % 5),
                        doi=f"10.1000/{self.ranking_strategy}.{idx}",
                        metadata={
                            "centrality_score": centrality,
                            "year": 2020 + (idx % 5),
                            "doi": f"10.1000/{self.ranking_strategy}.{idx}",
                            "source": "synthetic_corpus",
                        },
                    )
                )

            return SimpleNamespace(
                citations=citations,
                statistics=scenario["stats"],
            )

    monkeypatch.setattr(benchmarking_mod, "ContextAggregator", _FakeAggregator)

    ranking = asyncio.run(benchmark._benchmark_aggregation_strategies())
    benchmark._enforce_ranking_regression_guard(ranking)

    for strategy in ("weighted", "diversified", "concept_aware"):
        strategy_metrics = ranking[strategy]
        assert strategy_metrics["total_papers_evaluated"] >= 300
        assert strategy_metrics["avg_papers_per_archetype"] >= 25
        assert strategy_metrics["metadata_completeness_rate"] >= 0.99
        assert strategy_metrics["corpus_coverage_validation"]["valid"] is True
        for archetype_metrics in strategy_metrics["per_archetype"].values():
            assert archetype_metrics["citation_count"] >= 25
            assert archetype_metrics["metadata_completeness"] >= 0.99


def test_strategy_corpus_coverage_validation_detects_sparse_body_and_metadata() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_total_papers_per_strategy = 100
    benchmark.min_avg_papers_per_archetype = 8.0
    benchmark.min_metadata_completeness_rate = 0.8
    benchmark.min_papers_per_archetype = 5
    benchmark.min_metadata_completeness_per_archetype = 0.7

    payload = {
        "total_papers_evaluated": 18,
        "avg_papers_per_archetype": 3.0,
        "metadata_completeness_rate": 0.55,
        "per_archetype": {
            "clinical_hard": {
                "citation_count": 3,
                "metadata_completeness": 0.5,
            }
        },
    }
    validation = benchmark._validate_strategy_corpus_coverage("concept_aware", payload)
    assert validation["valid"] is False
    assert len(validation["failing"]) >= 3


def test_ranking_guard_fails_on_strategy_corpus_coverage_regression() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_concept_aware_ranking_ndcg = 0.5
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.4,
    }
    benchmark.hard_archetype_utility_margin = -0.1

    ranking_comparison = {
        "weighted": {
            "hard_archetype_uncertainty_adjusted_utility": 0.60,
            "corpus_coverage_validation": {"valid": True, "failing": []},
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
            "corpus_coverage_validation": {
                "valid": False,
                "failing": ["total_papers_evaluated 12 < 100"],
            },
            "monotonic_safety_response": {"valid": True, "failing": []},
            "temporal_forgetting_validation": {"valid": True},
        },
    }

    with pytest.raises(RuntimeError, match="Strategy corpus coverage regression"):
        benchmark._enforce_ranking_regression_guard(ranking_comparison)


def test_export_includes_corpus_coverage_validation_summary(tmp_path) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    BenchmarkSuite = benchmarking_mod.BenchmarkSuite

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    out_path = tmp_path / "benchmark_export_corpus_coverage.json"

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
            "weighted": {
                "corpus_coverage_validation": {"valid": True, "failing": []}
            },
            "concept_aware": {
                "ranking_ndcg_ci_lower": 0.76,
                "ranking_ndcg_ci_upper": 0.87,
                "calibration_drift": {"drift": 0.04, "status": "ok"},
                "corpus_coverage_validation": {"valid": True, "failing": []},
            },
        },
    )

    benchmark.export_benchmark_results(suite, out_path)
    payload = json.loads(out_path.read_text())
    adaptive = payload["summary"]["adaptive_safety"]
    assert "corpus_coverage_validation" in adaptive
    assert "concept_aware" in adaptive["corpus_coverage_validation"]


def test_end_to_end_strategy_comparison_triggers_guard_for_sparse_corpus_and_metadata(
    monkeypatch,
) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.min_concept_aware_ranking_ndcg = 0.35
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.25,
        "medium": 0.25,
        "hard": 0.25,
    }
    benchmark.hard_archetype_utility_margin = -0.05
    benchmark.hard_archetype_delta_ci_alpha = 0.05
    benchmark.bootstrap_samples = 200
    benchmark.risk_to_step_ridge_lambda = 0.15
    benchmark.min_total_papers_per_strategy = 100
    benchmark.min_avg_papers_per_archetype = 8.0
    benchmark.min_metadata_completeness_rate = 0.8
    benchmark.min_papers_per_archetype = 5
    benchmark.min_metadata_completeness_per_archetype = 0.7
    benchmark.category_adaptive_safety_floors = {
        "general": {"citation_validity_floor": 0.55, "hard_utility_margin": -0.01},
        "clinical": {"citation_validity_floor": 0.60, "hard_utility_margin": -0.01},
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
            "name": "clinical_sparse_0",
            "category": "clinical",
            "difficulty": "hard",
            "query": "clinical sparse 0",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.22,
                            "query_concept_coverage_score": 0.78,
                        },
                        "paper_count": 3,
                        "full_metadata": True,
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.27,
                            "query_concept_coverage_score": 0.74,
                        },
                        "paper_count": 3,
                        "full_metadata": False,
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.20,
                            "query_concept_coverage_score": 0.79,
                        },
                        "paper_count": 3,
                        "full_metadata": False,
                    },
                }
            },
        },
        {
            "name": "clinical_sparse_1",
            "category": "clinical",
            "difficulty": "medium",
            "query": "clinical sparse 1",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.24,
                            "query_concept_coverage_score": 0.76,
                        },
                        "paper_count": 3,
                        "full_metadata": True,
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.30,
                            "query_concept_coverage_score": 0.72,
                        },
                        "paper_count": 3,
                        "full_metadata": False,
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.21,
                            "query_concept_coverage_score": 0.77,
                        },
                        "paper_count": 3,
                        "full_metadata": False,
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

    class _SparseAggregator:
        def __init__(self, relevance_threshold, max_citations, entity_min_frequency, ranking_strategy):
            _ = (relevance_threshold, max_citations, entity_min_frequency)
            self.ranking_strategy = ranking_strategy

        def _extract_query_concepts(self, query):
            _ = query
            return {"concept": ["eeg"]}

        async def aggregate(self, query, fixture_results):
            _ = query
            scenario = fixture_results["scenario"][self.ranking_strategy]
            citations = []
            for idx in range(int(scenario["paper_count"])):
                if bool(scenario.get("full_metadata", False)):
                    citations.append(
                        SimpleNamespace(
                            pmid=f"p{idx}",
                            title=f"EEG paper {idx}",
                            abstract="eeg evidence",
                            year=2022,
                            doi=f"10.1000/sparse.{idx}",
                            metadata={
                                "centrality_score": 0.72,
                                "year": 2022,
                                "doi": f"10.1000/sparse.{idx}",
                            },
                        )
                    )
                else:
                    citations.append(
                        SimpleNamespace(
                            pmid=f"p{idx}",
                            title=f"EEG paper {idx}",
                            abstract="eeg evidence",
                            metadata={"centrality_score": 0.62},
                        )
                    )
            return SimpleNamespace(citations=citations, statistics=scenario["stats"])

    monkeypatch.setattr(benchmarking_mod, "ContextAggregator", _SparseAggregator)

    ranking = asyncio.run(benchmark._benchmark_aggregation_strategies())
    with pytest.raises(RuntimeError, match="Strategy corpus coverage regression"):
        benchmark._enforce_ranking_regression_guard(ranking)


def test_strategy_corpus_coverage_validation_uses_category_thresholds_and_cis() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_total_papers_per_strategy = 60
    benchmark.min_avg_papers_per_archetype = 8.0
    benchmark.min_metadata_completeness_rate = 0.80
    benchmark.min_papers_per_archetype = 5
    benchmark.min_metadata_completeness_per_archetype = 0.70
    benchmark.category_corpus_thresholds = {
        "general": {
            "min_total_papers": 60.0,
            "min_avg_papers_per_archetype": 8.0,
            "min_metadata_completeness_rate": 0.80,
        },
        "clinical": {
            "min_total_papers": 100.0,
            "min_avg_papers_per_archetype": 10.0,
            "min_metadata_completeness_rate": 0.86,
        },
    }

    payload = {
        "total_papers_evaluated": 90,
        "avg_papers_per_archetype": 9.0,
        "paper_volume_ci": {"ci_lower": 7.5, "ci_upper": 10.2},
        "metadata_completeness_rate": 0.85,
        "metadata_completeness_ci": {"ci_lower": 0.77, "ci_upper": 0.90},
        "per_archetype": {
            "clinical_hard": {
                "category": "clinical",
                "citation_count": 90,
                "metadata_completeness": 0.85,
                "metadata_missing_field_counts": {
                    "pmid": 0,
                    "title": 0,
                    "year": 5,
                    "doi": 9,
                },
                "unique_pmid_count": 88,
                "unique_pmid_ratio": 0.98,
                "distinct_sources": 3,
                "max_source_share": 0.52,
            }
        },
    }

    validation = benchmark._validate_strategy_corpus_coverage("concept_aware", payload)
    assert validation["valid"] is False
    assert any("paper_volume_ci.lower" in item for item in validation["failing"])
    assert any(
        "metadata_completeness_ci.lower" in item for item in validation["failing"]
    )
    assert any(
        "category[clinical] total_papers" in item for item in validation["failing"]
    )
    assert any(
        "category[clinical] metadata_rate" in item for item in validation["failing"]
    )


def test_strategy_corpus_coverage_validation_reports_failing_archetypes_and_missing_fields() -> None:
    benchmark = _benchmark_instance()
    benchmark.min_total_papers_per_strategy = 1
    benchmark.min_avg_papers_per_archetype = 1.0
    benchmark.min_metadata_completeness_rate = 0.0
    benchmark.min_papers_per_archetype = 2
    benchmark.min_metadata_completeness_per_archetype = 0.5
    benchmark.min_unique_pmids_per_archetype = 4
    benchmark.min_unique_pmid_ratio_per_archetype = 0.7
    benchmark.min_distinct_sources_per_archetype = 2
    benchmark.max_source_concentration_per_archetype = 0.7

    payload = {
        "total_papers_evaluated": 12,
        "avg_papers_per_archetype": 6.0,
        "metadata_completeness_rate": 0.9,
        "per_archetype": {
            "clinical_hard": {
                "category": "clinical",
                "citation_count": 6,
                "metadata_completeness": 0.90,
                "metadata_missing_field_counts": {
                    "pmid": 1,
                    "title": 0,
                    "year": 2,
                    "doi": 3,
                },
                "unique_pmid_count": 2,
                "unique_pmid_ratio": 0.33,
                "distinct_sources": 1,
                "max_source_share": 0.95,
            },
            "bci_hard": {
                "category": "bci",
                "citation_count": 6,
                "metadata_completeness": 0.92,
                "metadata_missing_field_counts": {
                    "pmid": 0,
                    "title": 1,
                    "year": 1,
                    "doi": 1,
                },
                "unique_pmid_count": 6,
                "unique_pmid_ratio": 1.0,
                "distinct_sources": 2,
                "max_source_share": 0.55,
            },
        },
    }

    validation = benchmark._validate_strategy_corpus_coverage("concept_aware", payload)
    assert validation["valid"] is False
    assert validation["failing_archetypes"]
    assert validation["failing_archetypes"][0]["id"] == "clinical_hard"
    assert validation["missing_field_breakdown"]["pmid"] == 1
    assert validation["missing_field_breakdown"]["title"] == 1
    assert validation["missing_field_breakdown"]["year"] == 3
    assert validation["missing_field_breakdown"]["doi"] == 4


def test_export_includes_failing_archetype_ids_and_missing_field_breakdown(tmp_path) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark
    BenchmarkSuite = benchmarking_mod.BenchmarkSuite

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    out_path = tmp_path / "benchmark_export_corpus_quality.json"

    suite = BenchmarkSuite(
        retrieval_results=[],
        generation_results=[],
        end_to_end_results=[],
        overall_score=0.80,
        retrieval_score=0.82,
        generation_score=0.78,
        avg_total_time_ms=400.0,
        avg_citation_accuracy=0.90,
        avg_response_quality=0.83,
        avg_redundancy_score=0.19,
        avg_diversity_score=0.70,
        avg_query_entity_coverage_score=0.70,
        avg_query_concept_coverage_score=0.73,
        avg_centrality_grounding_score=0.68,
        avg_grounding_quality=0.74,
        concept_aware_grounding_score=0.76,
        concept_aware_ranking_ndcg=0.81,
        ranking_strategy_comparison={
            "weighted": {
                "corpus_coverage_validation": {
                    "valid": True,
                    "failing": [],
                    "failing_archetypes": [],
                    "missing_field_breakdown": {
                        "pmid": 0,
                        "title": 0,
                        "year": 0,
                        "doi": 0,
                    },
                }
            },
            "concept_aware": {
                "ranking_ndcg_ci_lower": 0.76,
                "ranking_ndcg_ci_upper": 0.87,
                "calibration_drift": {"drift": 0.04, "status": "ok"},
                "corpus_coverage_validation": {
                    "valid": False,
                    "failing": ["clinical_hard: unique_pmid_ratio 0.330 < 0.700"],
                    "failing_archetypes": [
                        {"id": "clinical_hard", "issues": ["duplicate pmids"]}
                    ],
                    "missing_field_breakdown": {
                        "pmid": 2,
                        "title": 1,
                        "year": 4,
                        "doi": 5,
                    },
                },
                "monotonic_safety_response": {"valid": True},
                "temporal_forgetting_validation": {"valid": True},
                "risk_to_step_model": {"heldout_mse": 0.002},
                "hard_archetype_utility_delta_by_category": {
                    "clinical": {"mean": 0.04}
                },
            },
        },
    )

    benchmark.export_benchmark_results(suite, out_path)
    payload = json.loads(out_path.read_text())
    adaptive = payload["summary"]["adaptive_safety"]

    assert adaptive["failing_archetype_ids_by_strategy"]["concept_aware"] == [
        "clinical_hard"
    ]
    assert adaptive["missing_field_breakdown_by_strategy"]["concept_aware"]["doi"] == 5


def test_end_to_end_strategy_comparison_fails_on_mixed_source_skew_and_duplicate_pmids(
    monkeypatch,
) -> None:
    benchmarking_mod = _import_benchmark_module()
    EEGRAGBenchmark = benchmarking_mod.EEGRAGBenchmark

    benchmark = EEGRAGBenchmark.__new__(EEGRAGBenchmark)
    benchmark.min_concept_aware_ranking_ndcg = 0.35
    benchmark.min_archetype_ndcg_by_difficulty = {
        "easy": 0.25,
        "medium": 0.25,
        "hard": 0.25,
    }
    benchmark.hard_archetype_utility_margin = -0.05
    benchmark.hard_archetype_delta_ci_alpha = 0.05
    benchmark.bootstrap_samples = 200
    benchmark.risk_to_step_ridge_lambda = 0.15
    benchmark.min_total_papers_per_strategy = 1
    benchmark.min_avg_papers_per_archetype = 1.0
    benchmark.min_metadata_completeness_rate = 0.70
    benchmark.min_papers_per_archetype = 1
    benchmark.min_metadata_completeness_per_archetype = 0.70
    benchmark.min_unique_pmids_per_archetype = 8
    benchmark.min_unique_pmid_ratio_per_archetype = 0.65
    benchmark.min_distinct_sources_per_archetype = 2
    benchmark.max_source_concentration_per_archetype = 0.80
    benchmark.category_adaptive_safety_floors = {
        "general": {"citation_validity_floor": 0.55, "hard_utility_margin": -0.01},
        "clinical": {"citation_validity_floor": 0.60, "hard_utility_margin": -0.01},
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
            "name": "clinical_skew_0",
            "category": "clinical",
            "difficulty": "hard",
            "query": "clinical skewed source mix",
            "results": {
                "scenario": {
                    "weighted": {
                        "stats": {
                            "redundancy_score": 0.24,
                            "query_concept_coverage_score": 0.76,
                        },
                        "paper_count": 20,
                        "dominant_source": False,
                    },
                    "diversified": {
                        "stats": {
                            "redundancy_score": 0.26,
                            "query_concept_coverage_score": 0.74,
                        },
                        "paper_count": 20,
                        "dominant_source": False,
                    },
                    "concept_aware": {
                        "stats": {
                            "redundancy_score": 0.20,
                            "query_concept_coverage_score": 0.80,
                        },
                        "paper_count": 20,
                        "dominant_source": True,
                    },
                }
            },
        }
    ]

    monkeypatch.setattr(
        benchmark,
        "_create_archetype_fixture_bank",
        lambda: fixtures,
    )

    class _SkewedAggregator:
        def __init__(self, relevance_threshold, max_citations, entity_min_frequency, ranking_strategy):
            _ = (relevance_threshold, max_citations, entity_min_frequency)
            self.ranking_strategy = ranking_strategy

        def _extract_query_concepts(self, query):
            _ = query
            return {"concept": ["eeg"]}

        async def aggregate(self, query, fixture_results):
            _ = query
            scenario = fixture_results["scenario"][self.ranking_strategy]
            dominant_source = bool(scenario.get("dominant_source", False))
            citations = []
            for idx in range(int(scenario["paper_count"])):
                if dominant_source:
                    pmid = f"dup-{idx % 4}"
                    source = "pubmed"
                else:
                    pmid = f"u-{idx}"
                    source = "pubmed" if idx % 2 == 0 else "crossref"

                citations.append(
                    SimpleNamespace(
                        pmid=pmid,
                        title=f"EEG source-mix paper {idx}",
                        abstract="eeg evidence",
                        year=2021,
                        doi=f"10.1000/mix.{idx}",
                        metadata={
                            "centrality_score": 0.70,
                            "year": 2021,
                            "doi": f"10.1000/mix.{idx}",
                            "source": source,
                        },
                    )
                )
            return SimpleNamespace(citations=citations, statistics=scenario["stats"])

    monkeypatch.setattr(benchmarking_mod, "ContextAggregator", _SkewedAggregator)

    ranking = asyncio.run(benchmark._benchmark_aggregation_strategies())
    with pytest.raises(RuntimeError, match="Strategy corpus coverage regression"):
        benchmark._enforce_ranking_regression_guard(ranking)
