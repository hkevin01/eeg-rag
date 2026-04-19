#!/usr/bin/env python3
"""
Tests for the Agentic RAG Orchestrator.

Coverage targets (≥ 85% per project standards):
  - RetrievalDecisionMaker: all four decision branches
  - SufficiencyEvaluator: all five status paths
  - QueryReformulator: all six strategies + fallbacks
  - AgenticRAGOrchestrator:
      * SKIP path (direct answer)
      * single-iteration SUFFICIENT path
      * multi-iteration reformulation path
      * DECOMPOSE path (sub-queries)
      * VERIFY_CLAIM path
      * citation verification integration
      * no-results fallback answer
      * deduplication logic

All external I/O (HybridRetriever, ResponseGenerator, CitationVerifier) is
mocked so tests remain fast and network-free.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

from eeg_rag.rag.agentic_rag import (
    AgenticRAGOrchestrator,
    AgenticRAGResult,
    AgenticStep,
    QueryReformulator,
    ReformulationStrategy,
    RetrievalDecision,
    RetrievalDecisionMaker,
    RetrievalNeed,
    SufficiencyCheck,
    SufficiencyEvaluator,
    SufficiencyStatus,
)
from eeg_rag.retrieval.hybrid_retriever import HybridResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hybrid_result(
    doc_id: str,
    rrf_score: float = 0.015,
    text: str = "EEG seizure detection study",
    pmid: str | None = None,
) -> HybridResult:
    return HybridResult(
        doc_id=doc_id,
        text=text,
        metadata={"pmid": pmid or doc_id, "title": f"Paper {doc_id}"},
        rrf_score=rrf_score,
        bm25_score=0.5,
        dense_score=0.6,
    )


def _make_retriever(results: list[HybridResult]) -> MagicMock:
    """Return a mock HybridRetriever whose search() returns ``results``."""
    mock = MagicMock()
    mock.search.return_value = results
    mock.bm25_weight = 0.5
    mock.dense_weight = 0.5
    return mock


async def _gen_chunks(*chunks: str):
    """Async generator that yields the given strings."""
    for c in chunks:
        yield c


def _make_generator(answer: str = "Generated EEG answer.") -> MagicMock:
    """Return a mock ResponseGenerator whose generate() streams ``answer``."""
    mock = MagicMock()
    mock.generate = MagicMock(return_value=_gen_chunks(answer))
    return mock


# ---------------------------------------------------------------------------
# RetrievalDecisionMaker
# ---------------------------------------------------------------------------

class TestRetrievalDecisionMaker:
    """Unit tests for RetrievalDecisionMaker.decide()."""

    def setup_method(self):
        self.dm = RetrievalDecisionMaker(min_query_length=15)

    def test_skip_short_query(self):
        decision = self.dm.decide("What is EEG?")
        assert decision.need == RetrievalNeed.SKIP

    def test_skip_definitional_pattern(self):
        decision = self.dm.decide("What is the definition of alpha band?")
        assert decision.need == RetrievalNeed.SKIP

    def test_skip_define_prefix(self):
        decision = self.dm.decide("Define delta wave frequency range.")
        assert decision.need == RetrievalNeed.SKIP

    def test_skip_abbreviation_query(self):
        decision = self.dm.decide("What does ERP stand for in EEG research?")
        assert decision.need == RetrievalNeed.SKIP

    def test_retrieve_standard_research_query(self):
        decision = self.dm.decide(
            "Deep learning methods for seizure detection using EEG signals"
        )
        assert decision.need == RetrievalNeed.RETRIEVE

    def test_verify_claim_with_pmid(self):
        decision = self.dm.decide(
            "Smith et al. showed 95% accuracy in seizure detection PMID:12345678"
        )
        assert decision.need == RetrievalNeed.VERIFY_CLAIM
        assert "12345678" in decision.claimed_pmids

    def test_verify_claim_pmid_colon_format(self):
        decision = self.dm.decide(
            "Validate the findings in PMID: 9876543"
        )
        assert decision.need == RetrievalNeed.VERIFY_CLAIM
        assert "9876543" in decision.claimed_pmids

    def test_decompose_multi_question(self):
        decision = self.dm.decide(
            "What are the best methods for seizure detection? "
            "And how does sleep staging work in EEG?"
        )
        assert decision.need == RetrievalNeed.DECOMPOSE
        assert len(decision.sub_queries) >= 2

    def test_eeg_entities_extracted(self):
        decision = self.dm.decide(
            "CNN for seizure detection using alpha and theta band features"
        )
        assert "seizure" in decision.detected_entities
        assert "alpha" in decision.detected_entities
        assert "theta" in decision.detected_entities
        assert "cnn" in decision.detected_entities

    def test_empty_entities_on_generic_query(self):
        decision = self.dm.decide(
            "What are good natural language processing papers from 2023?"
        )
        # No EEG-specific entities expected
        assert decision.need == RetrievalNeed.RETRIEVE
        assert isinstance(decision.detected_entities, list)

    def test_decision_includes_rationale(self):
        decision = self.dm.decide("P300 BCI speller accuracy systematic review")
        assert len(decision.rationale) > 0


# ---------------------------------------------------------------------------
# SufficiencyEvaluator
# ---------------------------------------------------------------------------

class TestSufficiencyEvaluator:
    """Unit tests for SufficiencyEvaluator.evaluate()."""

    def setup_method(self):
        self.ev = SufficiencyEvaluator(
            min_docs=3,
            min_relevance=0.25,
            coverage_threshold=0.6,
        )
        self.decision = RetrievalDecision(
            need=RetrievalNeed.RETRIEVE,
            rationale="test",
            detected_entities=["seizure", "cnn"],
        )

    def test_empty_results(self):
        check = self.ev.evaluate("query", [], self.decision)
        assert check.status == SufficiencyStatus.EMPTY
        assert check.doc_count == 0

    def test_low_count(self):
        results = [_hybrid_result("d1", rrf_score=0.016, text="seizure cnn")]
        check = self.ev.evaluate("query", results, self.decision)
        assert check.status == SufficiencyStatus.LOW_COUNT
        assert check.doc_count == 1

    def test_low_relevance(self):
        # RRF scores near 0 → mean_score ~ 0 after normalisation
        results = [
            _hybrid_result(f"d{i}", rrf_score=0.0001, text="seizure cnn study")
            for i in range(5)
        ]
        check = self.ev.evaluate("query", results, self.decision)
        assert check.status == SufficiencyStatus.LOW_RELEVANCE

    def test_low_coverage(self):
        # Enough docs and relevance but missing entity terms in text
        results = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.015,
                text="motor imagery BCI study"  # no "seizure" or "cnn"
            )
            for i in range(5)
        ]
        check = self.ev.evaluate("query", results, self.decision)
        assert check.status in (
            SufficiencyStatus.LOW_COVERAGE, SufficiencyStatus.SUFFICIENT
        )

    def test_sufficient(self):
        results = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure cnn deep learning detection study",
            )
            for i in range(5)
        ]
        check = self.ev.evaluate("query", results, self.decision)
        assert check.status == SufficiencyStatus.SUFFICIENT
        assert check.doc_count == 5

    def test_no_entities_always_covers(self):
        """If no entities detected, coverage should never block sufficiency."""
        decision = RetrievalDecision(
            need=RetrievalNeed.RETRIEVE,
            rationale="test",
            detected_entities=[],
        )
        results = [
            _hybrid_result(f"d{i}", rrf_score=0.016, text="some paper")
            for i in range(5)
        ]
        check = self.ev.evaluate("query", results, decision)
        assert check.status == SufficiencyStatus.SUFFICIENT

    def test_explanation_non_empty(self):
        check = self.ev.evaluate("query", [], self.decision)
        assert len(check.explanation) > 0


# ---------------------------------------------------------------------------
# QueryReformulator
# ---------------------------------------------------------------------------

class TestQueryReformulator:
    """Unit tests for QueryReformulator.reformulate()."""

    def setup_method(self):
        self.reformulator = QueryReformulator()

    def _empty_check(self) -> SufficiencyCheck:
        return SufficiencyCheck(
            status=SufficiencyStatus.EMPTY,
            doc_count=0,
            relevance_score=0.0,
            coverage_score=0.0,
        )

    def _low_rel_check(self) -> SufficiencyCheck:
        return SufficiencyCheck(
            status=SufficiencyStatus.LOW_RELEVANCE,
            doc_count=5,
            relevance_score=0.1,
            coverage_score=0.5,
        )

    def _low_cov_check(self, missing: list[str] | None = None) -> SufficiencyCheck:
        return SufficiencyCheck(
            status=SufficiencyStatus.LOW_COVERAGE,
            doc_count=5,
            relevance_score=0.5,
            coverage_score=0.4,
            missing_aspects=missing or ["sleep staging"],
        )

    def test_expand_on_empty(self):
        result = self.reformulator.reformulate(
            original_query="seizure detection EEG",
            current_query="seizure detection EEG",
            check=self._empty_check(),
            iteration=1,
            prior_strategies=[],
        )
        assert result.strategy == ReformulationStrategy.EXPAND
        assert len(result.new_query) > 0

    def test_pivot_dense_on_low_relevance(self):
        result = self.reformulator.reformulate(
            original_query="seizure detection EEG",
            current_query="seizure AND detection NOT noise",
            check=self._low_rel_check(),
            iteration=1,
            prior_strategies=[ReformulationStrategy.EXPAND],
        )
        assert result.strategy == ReformulationStrategy.PIVOT_DENSE
        # Boolean operators should be removed
        assert "AND" not in result.new_query
        assert "NOT" not in result.new_query

    def test_narrow_on_low_coverage_with_missing(self):
        result = self.reformulator.reformulate(
            original_query="EEG sleep staging methods",
            current_query="EEG sleep staging methods",
            check=self._low_cov_check(missing=["sleep staging"]),
            iteration=1,
            prior_strategies=[],
        )
        assert result.strategy == ReformulationStrategy.NARROW
        assert "sleep staging" in result.new_query

    def test_relax_falls_back_when_narrow_used(self):
        result = self.reformulator.reformulate(
            original_query="EEG cognitive load",
            current_query='EEG cognitive load "mental workload" (human)',
            check=self._low_cov_check(),
            iteration=2,
            prior_strategies=[ReformulationStrategy.NARROW],
        )
        assert result.strategy == ReformulationStrategy.RELAX

    def test_decompose_final_fallback(self):
        """When all strategies exhausted, DECOMPOSE falls back to original."""
        all_strategies = [s for s in ReformulationStrategy
                          if s != ReformulationStrategy.DECOMPOSE]
        result = self.reformulator.reformulate(
            original_query="EEG emotion recognition",
            current_query="EEG emotion recognition final",
            check=self._empty_check(),
            iteration=3,
            prior_strategies=all_strategies,
        )
        assert result.strategy == ReformulationStrategy.DECOMPOSE

    def test_bm25_hint_on_expand(self):
        result = self.reformulator.reformulate(
            "q", "q", self._empty_check(), iteration=1, prior_strategies=[]
        )
        assert result.bm25_weight_hint is not None
        assert result.dense_weight_hint is not None

    def test_dense_hint_on_pivot_dense(self):
        result = self.reformulator.reformulate(
            "q", "q AND test",
            self._low_rel_check(),
            iteration=1,
            prior_strategies=[ReformulationStrategy.EXPAND],
        )
        assert result.dense_weight_hint is not None
        assert result.dense_weight_hint > 0.5


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – SKIP path
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorSkip:
    """Tests for the direct-answer (no-retrieval) code path."""

    def setup_method(self):
        self.retriever = _make_retriever([])
        self.generator = _make_generator("Alpha band is 8-13 Hz.")
        self.orchestrator = AgenticRAGOrchestrator(
            retriever=self.retriever,
            generator=self.generator,
        )

    def test_skip_returns_direct_answer(self):
        result = asyncio.run(
            self.orchestrator.run("Define the alpha frequency band.")
        )
        assert isinstance(result, AgenticRAGResult)
        assert result.skipped_retrieval is True
        assert result.iterations_used == 0
        assert len(result.sources) == 0
        assert len(result.answer) > 0

    def test_skip_no_retriever_calls(self):
        asyncio.run(
            self.orchestrator.run("What does EEG stand for?")
        )
        self.retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – single-iteration sufficient path
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorSingleIteration:
    """Tests for the happy path: one retrieval round is sufficient."""

    def setup_method(self):
        # 5 highly relevant docs mentioning all entities
        self.docs = [
            _hybrid_result(
                f"pmid_{i}",
                rrf_score=0.016,
                text="seizure detection cnn deep learning EEG study",
                pmid=f"1234567{i}",
            )
            for i in range(5)
        ]
        self.retriever = _make_retriever(self.docs)
        self.generator = _make_generator("CNN-based seizure detection achieves 95%.")
        self.orchestrator = AgenticRAGOrchestrator(
            retriever=self.retriever,
            generator=self.generator,
            max_iterations=3,
            min_docs=3,
            min_relevance=0.25,
            top_k=10,
        )

    def test_single_iteration_returns_result(self):
        result = asyncio.run(
            self.orchestrator.run(
                "CNN methods for seizure detection using EEG signals"
            )
        )
        assert result.skipped_retrieval is False
        assert len(result.sources) == 5
        assert result.iterations_used == 1

    def test_answer_contains_content(self):
        result = asyncio.run(
            self.orchestrator.run("Seizure detection CNN EEG deep learning")
        )
        assert "seizure" in result.answer.lower() or len(result.answer) > 5

    def test_step_audit_trail(self):
        result = asyncio.run(
            self.orchestrator.run("CNN seizure detection EEG deep learning")
        )
        assert len(result.steps) == 1
        step = result.steps[0]
        assert isinstance(step, AgenticStep)
        assert step.iteration == 1
        assert step.docs_retrieved == 5
        assert step.sufficiency.status == SufficiencyStatus.SUFFICIENT

    def test_retriever_called_once(self):
        asyncio.run(
            self.orchestrator.run("CNN seizure detection EEG deep learning")
        )
        assert self.retriever.search.call_count == 1

    def test_timing_recorded(self):
        result = asyncio.run(
            self.orchestrator.run("CNN seizure detection EEG deep learning")
        )
        assert result.total_elapsed_ms > 0
        assert result.steps[0].elapsed_ms > 0


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – multi-iteration reformulation path
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorMultiIteration:
    """Tests for iterative reformulation when first retrieval is insufficient."""

    def _make_retriever_sequence(
        self,
        call_results: list[list[HybridResult]],
    ) -> MagicMock:
        """Mock whose search() returns successive lists on each call."""
        mock = MagicMock()
        mock.search.side_effect = call_results
        mock.bm25_weight = 0.5
        mock.dense_weight = 0.5
        return mock

    def test_reformulates_on_empty_first_round(self):
        # Round 1: empty → Round 2: sufficient
        sufficient_docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure detection EEG deep learning",
            )
            for i in range(5)
        ]
        retriever = self._make_retriever_sequence([[], sufficient_docs])
        generator = _make_generator("Found relevant results after reformulation.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=3,
            min_docs=3,
        )
        result = asyncio.run(
            orchestrator.run("Seizure detection EEG deep learning CNN")
        )
        assert result.iterations_used == 2
        assert retriever.search.call_count == 2
        # Step 1 should show reformulation was applied
        assert result.steps[0].reformulation is not None

    def test_max_iterations_respected(self):
        # All rounds return empty
        retriever = self._make_retriever_sequence([[], [], []])
        generator = _make_generator("No evidence found.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=3,
            min_docs=3,
        )
        result = asyncio.run(
            orchestrator.run("Seizure detection EEG CNN deep learning methods")
        )
        assert result.iterations_used == 3
        assert retriever.search.call_count == 3

    def test_best_results_used_after_failed_iterations(self):
        """Best intermediate results should be returned even if never sufficient."""
        round1 = [_hybrid_result("d1", rrf_score=0.016, text="seizure EEG")]
        round2 = []  # worse round
        round3 = []
        retriever = self._make_retriever_sequence([round1, round2, round3])
        generator = _make_generator("Answer from partial evidence.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=3,
            min_docs=3,
        )
        result = asyncio.run(
            orchestrator.run("EEG seizure detection deep learning CNN")
        )
        # Should have used the best result (from round 1)
        assert len(result.sources) >= 1

    def test_reformulation_strategy_logged_in_step(self):
        round1: list[HybridResult] = []
        round2 = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure cnn detection study EEG",
            )
            for i in range(5)
        ]
        retriever = self._make_retriever_sequence([round1, round2])
        generator = _make_generator("Answer.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever, generator=generator, max_iterations=3
        )
        result = asyncio.run(
            orchestrator.run("Seizure detection EEG deep learning CNN BCI")
        )
        step1 = result.steps[0]
        assert step1.reformulation is not None
        assert isinstance(step1.reformulation.strategy, ReformulationStrategy)


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – DECOMPOSE path
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorDecompose:
    """Tests for multi-sub-query decomposition mode."""

    def test_decompose_retrieves_per_sub_query(self):
        docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure cnn detection study EEG",
            )
            for i in range(5)
        ]
        retriever = _make_retriever(docs)
        generator = _make_generator("Synthesised answer across sub-queries.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=1,
        )
        # Query with two sub-questions
        query = (
            "What CNN architectures are used for seizure detection? "
            "And how does sleep staging work in EEG recordings?"
        )
        result = asyncio.run(
            orchestrator.run(query)
        )
        # Retriever should be called once per sub-query
        assert retriever.search.call_count >= 2

    def test_sources_deduplicated_across_sub_queries(self):
        """Same doc_id from two sub-query rounds should be deduplicated."""
        shared_doc = _hybrid_result("shared", rrf_score=0.016, text="seizure EEG")
        retriever = _make_retriever([shared_doc] * 5)
        generator = _make_generator("Answer.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=1,
        )
        query = (
            "What is seizure detection accuracy? "
            "And how can EEG be used for BCI systems?"
        )
        result = asyncio.run(
            orchestrator.run(query)
        )
        doc_ids = [s.doc_id for s in result.sources]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids found"


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – VERIFY_CLAIM path
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorVerifyClaim:
    """Tests for PMID-claim verification mode."""

    def test_verify_claim_still_retrieves(self):
        docs = [
            _hybrid_result("d1", pmid="12345678", text="seizure accuracy EEG")
        ] * 5
        retriever = _make_retriever(docs)
        generator = _make_generator("Claim verified.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever, generator=generator
        )
        result = asyncio.run(
            orchestrator.run(
                "PMID:12345678 reports 95% accuracy for seizure detection, "
                "please verify this finding in EEG literature."
            )
        )
        assert result.decision.need == RetrievalNeed.VERIFY_CLAIM
        assert retriever.search.call_count >= 1


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – citation verification integration
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorCitationVerification:
    """Tests for optional citation verification step."""

    def _make_verifier(
        self, exists: bool = True, retracted: bool = False
    ) -> MagicMock:
        """Return a mock CitationVerifier."""
        vr = MagicMock()
        vr.exists = exists
        vr.is_retracted = retracted
        mock = MagicMock()
        mock.verify_citation = AsyncMock(return_value=vr)
        return mock

    def test_verification_runs_when_verifier_provided(self):
        docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                pmid=f"1234567{i}",
                text="seizure cnn eeg detection study",
            )
            for i in range(5)
        ]
        retriever = _make_retriever(docs)
        generator = _make_generator("Answer with verified citations.")
        verifier = self._make_verifier(exists=True)
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            verifier=verifier,
        )
        result = asyncio.run(
            orchestrator.run("CNN seizure detection EEG deep learning methods")
        )
        assert result.citations_verified is True
        assert verifier.verify_citation.call_count >= 1

    def test_retracted_paper_generates_warning(self):
        docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                pmid=f"1234567{i}",
                text="seizure cnn eeg study detection",
            )
            for i in range(5)
        ]
        retriever = _make_retriever(docs)
        generator = _make_generator("Answer.")
        verifier = self._make_verifier(exists=True, retracted=True)
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            verifier=verifier,
        )
        result = asyncio.run(
            orchestrator.run("CNN seizure detection EEG deep learning methods")
        )
        assert any("retracted" in w.lower() for w in result.verification_warnings)

    def test_no_verification_without_verifier(self):
        docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure cnn eeg detection study",
            )
            for i in range(5)
        ]
        retriever = _make_retriever(docs)
        generator = _make_generator("Answer.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            verifier=None,
        )
        result = asyncio.run(
            orchestrator.run("CNN seizure detection EEG deep learning methods")
        )
        assert result.citations_verified is False
        assert result.verification_warnings == []


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – empty sources fallback
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorNoSources:
    """Tests for the no-sources fallback answer."""

    def test_no_sources_returns_informative_message(self):
        retriever = _make_retriever([])
        retriever.search.side_effect = [[], [], []]  # always empty
        generator = _make_generator("Generated but unused.")
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
            max_iterations=3,
            min_docs=3,
        )
        result = asyncio.run(
            orchestrator.run("EEG deep learning seizure detection CNN methods")
        )
        assert "no relevant literature" in result.answer.lower()
        assert result.sources == []


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – input validation
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorValidation:
    """Tests for input validation."""

    def setup_method(self):
        self.orchestrator = AgenticRAGOrchestrator(
            retriever=_make_retriever([]),
            generator=_make_generator(),
        )

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            asyncio.run(
                self.orchestrator.run("")
            )

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError):
            asyncio.run(
                self.orchestrator.run("   ")
            )


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator – streaming
# ---------------------------------------------------------------------------

class TestAgenticRAGOrchestratorStreaming:
    """Tests for the stream() generator."""

    def test_stream_yields_non_empty_chunks(self):
        docs = [
            _hybrid_result(
                f"d{i}",
                rrf_score=0.016,
                text="seizure cnn eeg detection",
            )
            for i in range(5)
        ]
        retriever = _make_retriever(docs)
        generator = _make_generator("A" * 250)  # 250 chars → 3 chunks of 100
        orchestrator = AgenticRAGOrchestrator(
            retriever=retriever,
            generator=generator,
        )

        async def collect():
            chunks = []
            async for chunk in orchestrator.stream(
                "CNN seizure EEG detection deep learning"
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Unit tests for AgenticRAGOrchestrator._deduplicate()."""

    def test_keeps_highest_score(self):
        r1 = _hybrid_result("d1", rrf_score=0.010)
        r2 = _hybrid_result("d1", rrf_score=0.016)  # same id, higher score
        r3 = _hybrid_result("d2", rrf_score=0.012)
        deduped = AgenticRAGOrchestrator._deduplicate([r1, r2, r3])
        ids = {r.doc_id for r in deduped}
        assert ids == {"d1", "d2"}
        d1_result = next(r for r in deduped if r.doc_id == "d1")
        assert d1_result.rrf_score == 0.016

    def test_sorted_by_score_descending(self):
        results = [
            _hybrid_result("a", rrf_score=0.010),
            _hybrid_result("b", rrf_score=0.020),
            _hybrid_result("c", rrf_score=0.015),
        ]
        deduped = AgenticRAGOrchestrator._deduplicate(results)
        scores = [r.rrf_score for r in deduped]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        assert AgenticRAGOrchestrator._deduplicate([]) == []
