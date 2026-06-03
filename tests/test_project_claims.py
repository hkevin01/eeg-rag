"""Tests for README claim auditing and project truthfulness checks."""

from __future__ import annotations

import shutil
from pathlib import Path

from eeg_rag.verification.project_claims import (
    ClaimStatus,
    verify_project_claims,
)


def _copy_test_corpus(tmp_path: Path) -> Path:
    source_dir = Path("data/test_corpus")
    target_dir = tmp_path / "test_corpus"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("eeg_corpus_20251122.jsonl", "corpus_metadata.json"):
        shutil.copy2(source_dir / filename, target_dir / filename)

    return target_dir


def test_project_claim_audit_reports_corpus_and_retrieval(tmp_path):
    corpus_dir = _copy_test_corpus(tmp_path)

    report = verify_project_claims(corpus_dir=corpus_dir)
    report_dict = report.to_dict()

    assert report_dict["corpus_dir"] == str(corpus_dir)
    assert report_dict["corpus_stats"]["total_records"] == 3
    assert report_dict["corpus_stats"]["stats_service_verified_total"] == 3

    retrieval = report_dict["retrieval_smoke_test"]
    assert retrieval["top_doc_id"] == "sleep-stage"
    assert retrieval["top_has_pmid"] is True

    statuses = {check["name"]: check["status"] for check in report_dict["checks"]}
    assert statuses["Shipped corpus metadata is internally consistent"] == ClaimStatus.PASS.value
    assert statuses["Hybrid retrieval returns ranked metadata-bearing papers"] == ClaimStatus.PASS.value
    assert statuses["Large-scale latency claims require benchmark execution"] == ClaimStatus.VERIFY_AT_BENCHMARK.value


def test_project_claim_audit_exposes_recommendations(tmp_path):
    corpus_dir = _copy_test_corpus(tmp_path)

    report = verify_project_claims(corpus_dir=corpus_dir)
    checks = {check.name: check for check in report.checks}

    benchmark_check = checks["Large-scale latency claims require benchmark execution"]
    assert benchmark_check.recommendation
    assert "benchmark" in benchmark_check.recommendation.lower()
    assert "evaluate_reranking_improvements" in benchmark_check.details["recommended_command"]
