"""Project claim verification utilities.

This module provides a lightweight audit layer for the public claims made in
README.md. The goal is not to replace full benchmarks, but to make the most
important promises testable:

* the repository ships coherent corpus metadata
* retrieval returns ranked results with preserved metadata
* hybrid scoring produces deterministic top results on a known fixture

The audit is intentionally conservative. If a claim cannot be validated with a
local, reproducible check, it is reported as a benchmark requirement instead of
being treated as proven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_rag.retrieval.bm25_retriever import BM25Result
from eeg_rag.retrieval.dense_retriever import DenseResult
from eeg_rag.retrieval.hybrid_retriever import HybridRetriever
from eeg_rag.services.stats_service import StatsService


class ClaimStatus(str, Enum):
    """Status for a single README claim check."""

    PASS = "pass"
    FAIL = "fail"
    VERIFY_AT_BENCHMARK = "verify_at_benchmark"


@dataclass
class ClaimCheck:
    """One claim check with evidence and remediation guidance."""

    name: str
    status: ClaimStatus
    evidence: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None


@dataclass
class ClaimAuditReport:
    """Structured report describing which project claims are supported."""

    generated_at: str
    corpus_dir: str
    corpus_stats: Dict[str, Any]
    retrieval_smoke_test: Dict[str, Any]
    checks: List[ClaimCheck]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "corpus_dir": self.corpus_dir,
            "corpus_stats": self.corpus_stats,
            "retrieval_smoke_test": self.retrieval_smoke_test,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "evidence": check.evidence,
                    "details": check.details,
                    "recommendation": check.recommendation,
                }
                for check in self.checks
            ],
        }


class ProjectClaimAuditor:
    """Verify the README's operational claims against local repository data."""

    def __init__(self, corpus_dir: Optional[Path] = None) -> None:
        self.corpus_dir = corpus_dir or Path("data/test_corpus")

    def _load_jsonl_papers(self) -> List[Dict[str, Any]]:
        papers: List[Dict[str, Any]] = []
        for jsonl_path in sorted(self.corpus_dir.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        papers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return papers

    def _corpus_stats(self) -> Dict[str, Any]:
        papers = self._load_jsonl_papers()
        total = len(papers)
        pmid_count = sum(1 for paper in papers if paper.get("pmid"))
        doi_count = sum(1 for paper in papers if paper.get("doi"))
        title_count = sum(1 for paper in papers if paper.get("title"))
        abstract_count = sum(1 for paper in papers if paper.get("abstract"))

        service = StatsService(corpus_dir=self.corpus_dir)
        verified = service.verify_counts()

        return {
            "files": [str(path.name) for path in sorted(self.corpus_dir.glob("*.jsonl"))],
            "total_records": total,
            "pmid_coverage_pct": round((pmid_count / max(total, 1)) * 100, 1),
            "doi_coverage_pct": round((doi_count / max(total, 1)) * 100, 1),
            "title_coverage_pct": round((title_count / max(total, 1)) * 100, 1),
            "abstract_coverage_pct": round((abstract_count / max(total, 1)) * 100, 1),
            "stats_service_verified_total": verified.get("verified_total", 0),
            "stats_service_display_total": verified.get("display_total", "0"),
            "stats_service_inconsistencies": verified.get("inconsistencies", []),
        }

    def _retrieval_smoke_test(self) -> Dict[str, Any]:
        documents = [
            {
                "id": "sleep-stage",
                "text": "Sleep stage classification with EEG alpha and theta oscillations.",
                "metadata": {"pmid": "10000001", "title": "Sleep staging EEG"},
            },
            {
                "id": "motor-imagery",
                "text": "Motor imagery BCI classification using sensorimotor rhythms.",
                "metadata": {"pmid": "10000002", "title": "Motor imagery BCI"},
            },
            {
                "id": "epilepsy",
                "text": "EEG seizure detection for epilepsy with hybrid retrieval and reranking.",
                "metadata": {"pmid": "10000003", "title": "Epilepsy seizure detection"},
            },
        ]

        bm25_results = [
            BM25Result(
                doc_id="sleep-stage",
                score=1.0,
                text=documents[0]["text"],
                metadata=documents[0]["metadata"],
            ),
            BM25Result(
                doc_id="epilepsy",
                score=0.8,
                text=documents[2]["text"],
                metadata=documents[2]["metadata"],
            ),
        ]

        dense_results = [
            DenseResult(
                doc_id="motor-imagery",
                score=0.92,
                text=documents[1]["text"],
                metadata=documents[1]["metadata"],
            ),
            DenseResult(
                doc_id="sleep-stage",
                score=0.88,
                text=documents[0]["text"],
                metadata=documents[0]["metadata"],
            ),
        ]

        class _DenseMock:
            def search(
                self,
                query: str,
                top_k: int = 10,
                filters=None,
                include_vectors: bool = False,
            ):
                del query
                del filters
                del include_vectors
                return dense_results[:top_k]

        class _BM25Mock:
            def search(self, query: str, top_k: int = 10, min_score: float = 0.0):
                del query
                del min_score
                return bm25_results[:top_k]

        hybrid = HybridRetriever(
            bm25_retriever=_BM25Mock(),
            dense_retriever=_DenseMock(),
            use_query_expansion=False,
            use_reranking=False,
        )

        results = hybrid.search("sleep stage classification eeg", top_k=2, retrieve_k=3)
        top = results[0] if results else None

        return {
            "query": "sleep stage classification eeg",
            "top_doc_id": top.doc_id if top else None,
            "top_title": top.metadata.get("title") if top else None,
            "top_has_pmid": bool(top and top.metadata.get("pmid")),
            "result_count": len(results),
            "top_rrf_score": top.rrf_score if top else None,
        }

    def audit(self) -> ClaimAuditReport:
        corpus_stats = self._corpus_stats()
        retrieval_smoke = self._retrieval_smoke_test()

        checks = [
            ClaimCheck(
                name="Shipped corpus metadata is internally consistent",
                status=(
                    ClaimStatus.PASS
                    if corpus_stats["total_records"] > 0
                    and corpus_stats["stats_service_verified_total"] == corpus_stats["total_records"]
                    and corpus_stats["title_coverage_pct"] == 100.0
                    and corpus_stats["abstract_coverage_pct"] == 100.0
                    else ClaimStatus.FAIL
                ),
                evidence=(
                    f"{corpus_stats['total_records']} records, "
                    f"{corpus_stats['pmid_coverage_pct']}% PMID coverage, "
                    f"{corpus_stats['doi_coverage_pct']}% DOI coverage"
                ),
                details=corpus_stats,
                recommendation=(
                    "Fix corpus metadata or regenerate the shipped corpus files"
                    if corpus_stats["total_records"] == 0
                    or corpus_stats["stats_service_verified_total"] != corpus_stats["total_records"]
                    else None
                ),
            ),
            ClaimCheck(
                name="Hybrid retrieval returns ranked metadata-bearing papers",
                status=(
                    ClaimStatus.PASS
                    if retrieval_smoke["top_doc_id"] == "sleep-stage"
                    and retrieval_smoke["top_has_pmid"]
                    else ClaimStatus.FAIL
                ),
                evidence=(
                    f"Top doc={retrieval_smoke['top_doc_id']}, "
                    f"title={retrieval_smoke['top_title']}, "
                    f"pmid={retrieval_smoke['top_has_pmid']}"
                ),
                details=retrieval_smoke,
                recommendation=(
                    "Inspect BM25/dense fusion and metadata propagation"
                    if retrieval_smoke["top_doc_id"] != "sleep-stage"
                    or not retrieval_smoke["top_has_pmid"]
                    else None
                ),
            ),
            ClaimCheck(
                name="Large-scale latency claims require benchmark execution",
                status=ClaimStatus.VERIFY_AT_BENCHMARK,
                evidence=(
                    "Latency and recall claims are environment-dependent and cannot be "+
                    "proven from the small shipped corpus alone."
                ),
                details={
                    "recommended_command": "python examples/evaluate_reranking_improvements.py",
                    "follow_up": "Run on a release corpus and record benchmark output in CI artifacts.",
                },
                recommendation="Gate README performance claims on a benchmark report per release.",
            ),
        ]

        return ClaimAuditReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            corpus_dir=str(self.corpus_dir),
            corpus_stats=corpus_stats,
            retrieval_smoke_test=retrieval_smoke,
            checks=checks,
        )


def verify_project_claims(corpus_dir: Optional[Path] = None) -> ClaimAuditReport:
    """Convenience wrapper used by tests and future CLI integration."""
    auditor = ProjectClaimAuditor(corpus_dir=corpus_dir)
    return auditor.audit()
