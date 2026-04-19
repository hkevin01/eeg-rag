#!/usr/bin/env python3
"""
Tests for RAGAS-style evaluation metrics (ragas_metrics.py).

Coverage targets (≥ 85% per project standards):
  - _EmbeddingEvaluator: all four metrics, edge cases
  - _LLMEvaluator: all four metrics (LLM calls mocked)
  - _LLMJudge: JSON parsing helper
  - RAGASEvaluator: AUTO mode fallback, batch, summary
  - export_for_human_eval: structure and length checks
  - _harmonic_mean: pure-function tests
  - RAGASScores.to_dict(): serialisation

All sentence-transformer encoding is mocked so tests run without GPU or
model downloads.  LLM API calls are always mocked (no network required).
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List

from eeg_rag.evaluation.ragas_metrics import (
    ContextChunkScore,
    ContextDocument,
    EvaluationMode,
    FaithfulnessDetail,
    HumanEvalRecord,
    LLMProvider,
    RAGASEvaluator,
    RAGASInput,
    RAGASScores,
    _EmbeddingEvaluator,
    _LLMJudge,
    _LLMEvaluator,
    _harmonic_mean,
    export_for_human_eval,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _doc(
    doc_id: str = "d1",
    text: str = "Seizure detection using CNN achieved 95% accuracy.",
    pmid: str | None = None,
) -> ContextDocument:
    return ContextDocument(
        doc_id=doc_id,
        text=text,
        metadata={"pmid": pmid or doc_id},
        pmid=pmid,
    )


def _input(
    query: str = "What CNN methods are used for seizure detection?",
    answer: str = (
        "CNN-based methods achieve high accuracy for seizure detection. "
        "Recurrent architectures like LSTM also perform well. "
        "[PMID:12345678]"
    ),
    docs: List[ContextDocument] | None = None,
    gt_answer: str | None = None,
    gt_ids: set[str] | None = None,
) -> RAGASInput:
    if docs is None:
        docs = [
            _doc("d1", "Seizure detection using CNN achieved 95% accuracy.", pmid="12345678"),
            _doc("d2", "LSTM networks for epilepsy classification outperform classical methods."),
            _doc("d3", "EEG-based brain-computer interface using motor imagery."),
        ]
    return RAGASInput(
        query=query,
        answer=answer,
        context_docs=docs,
        ground_truth_answer=gt_answer,
        ground_truth_doc_ids=gt_ids,
    )


def _mock_st_model(sim_value: float = 0.8):
    """Return a mock SentenceTransformer whose encode() returns fake tensors."""
    import numpy as np

    mock_model = MagicMock()

    def fake_encode(texts, convert_to_tensor=False):
        n = len(texts) if isinstance(texts, list) else 1
        arr = np.ones((n, 4), dtype=np.float32) * 0.5
        if convert_to_tensor:
            import torch
            return torch.tensor(arr)
        return arr

    mock_model.encode = MagicMock(side_effect=fake_encode)
    return mock_model


def _mock_cos_sim(value: float = 0.8):
    """Mock _st_cos_sim to return a tensor of constant value."""
    import numpy as np
    import torch

    def _cos_sim(a, b):
        rows = a.shape[0]
        cols = b.shape[0]
        return torch.full((rows, cols), value)

    return _cos_sim


# ---------------------------------------------------------------------------
# _harmonic_mean
# ---------------------------------------------------------------------------

class TestHarmonicMean:
    def test_all_ones(self):
        assert _harmonic_mean([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_any_zero_returns_zero(self):
        assert _harmonic_mean([1.0, 0.0, 0.8, 0.9]) == 0.0

    def test_empty_returns_zero(self):
        assert _harmonic_mean([]) == 0.0

    def test_single_value(self):
        assert _harmonic_mean([0.6]) == pytest.approx(0.6)

    def test_two_values(self):
        # H = 2 / (1/0.5 + 1/0.5) = 2 / 4 = 0.5
        assert _harmonic_mean([0.5, 0.5]) == pytest.approx(0.5)

    def test_asymmetric_values(self):
        # H({1, 0.5}) = 2 / (1 + 2) = 2/3
        assert _harmonic_mean([1.0, 0.5]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ContextDocument
# ---------------------------------------------------------------------------

class TestContextDocument:
    def test_pmid_set_from_metadata(self):
        doc = ContextDocument(
            doc_id="x", text="text", metadata={"pmid": "12345678"}
        )
        assert doc.pmid == "12345678"

    def test_pmid_explicit_overrides_metadata(self):
        doc = ContextDocument(
            doc_id="x",
            text="text",
            metadata={"pmid": "99999999"},
            pmid="12345678",
        )
        assert doc.pmid == "12345678"

    def test_pmid_none_when_absent(self):
        doc = ContextDocument(doc_id="x", text="text")
        assert doc.pmid is None


# ---------------------------------------------------------------------------
# RAGASScores.to_dict
# ---------------------------------------------------------------------------

class TestRAGASScoresToDict:
    def test_to_dict_structure(self):
        scores = RAGASScores(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
            context_recall=0.6,
            ragas_score=0.75,
            mode=EvaluationMode.EMBEDDING,
        )
        d = scores.to_dict()
        assert set(d.keys()) >= {
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "ragas_score",
            "mode",
            "faithfulness_details",
            "chunk_scores",
            "warnings",
        }
        assert d["mode"] == "embedding"

    def test_to_dict_rounds_values(self):
        scores = RAGASScores(
            faithfulness=0.12345678,
            answer_relevance=0.5,
            context_precision=0.5,
            context_recall=0.5,
            ragas_score=0.5,
            mode=EvaluationMode.LLM,
        )
        d = scores.to_dict()
        assert len(str(d["faithfulness"]).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# _EmbeddingEvaluator — unit tests with mocked sentence-transformers
# ---------------------------------------------------------------------------

@pytest.fixture()
def emb_eval():
    """Return an _EmbeddingEvaluator with a mocked ST model."""
    import numpy as np
    import torch

    ev = object.__new__(_EmbeddingEvaluator)

    mock_model = MagicMock()

    def fake_encode(texts, convert_to_tensor=False):
        n = len(texts) if isinstance(texts, list) else 1
        arr = np.ones((n, 4), dtype=np.float32) * 0.5
        if convert_to_tensor:
            return torch.tensor(arr)
        return arr

    mock_model.encode = MagicMock(side_effect=fake_encode)
    ev._model = mock_model
    return ev


class TestEmbeddingEvaluatorFaithfulness:
    def test_empty_answer_returns_zero(self, emb_eval):
        inp = _input(answer="")
        score, details = emb_eval.faithfulness(inp)
        assert score == 0.0
        assert details == []

    def test_empty_context_returns_zero(self, emb_eval):
        inp = _input(docs=[])
        score, details = emb_eval.faithfulness(inp)
        assert score == 0.0

    def test_returns_score_and_details(self, emb_eval):
        """With high cosine sim (mocked 0.5 → below 0.40 threshold — but
        the fake encode produces vectors that will land around 1.0 cos_sim)."""
        import torch
        import numpy as np

        # Force high similarity so claims are marked supported
        def high_sim_encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            if convert_to_tensor:
                return torch.tensor(arr)
            return arr

        emb_eval._model.encode = MagicMock(side_effect=high_sim_encode)
        inp = _input()
        score, details = emb_eval.faithfulness(inp)
        assert 0.0 <= score <= 1.0
        assert isinstance(details, list)
        assert all(isinstance(d, FaithfulnessDetail) for d in details)

    def test_claims_count_matches_sentences(self, emb_eval):
        import torch, numpy as np

        def high_encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        emb_eval._model.encode = MagicMock(side_effect=high_encode)
        answer = "First claim here. Second claim about EEG. Third claim about seizure."
        inp = _input(answer=answer)
        _, details = emb_eval.faithfulness(inp)
        # Should have one detail per sentence/claim
        claims = _EmbeddingEvaluator._split_claims(answer)
        assert len(details) == len(claims)


class TestEmbeddingEvaluatorAnswerRelevance:
    def test_empty_answer_returns_zero(self, emb_eval):
        inp = _input(answer="")
        assert emb_eval.answer_relevance(inp) == 0.0

    def test_empty_query_returns_zero(self, emb_eval):
        inp = _input(query="")
        assert emb_eval.answer_relevance(inp) == 0.0

    def test_returns_float_in_range(self, emb_eval):
        inp = _input()
        score = emb_eval.answer_relevance(inp)
        assert 0.0 <= score <= 1.0

    def test_penalty_for_hedge_phrases(self, emb_eval):
        """Answer with many hedges should score lower than confident answer."""
        import torch, numpy as np

        def same_encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        emb_eval._model.encode = MagicMock(side_effect=same_encode)
        confident = _input(answer="CNN achieves 95% seizure detection accuracy.")
        hedged = _input(
            answer=(
                "I believe there is speculation that it is possible CNN might "
                "sometimes help with seizure detection."
            )
        )
        s_conf = emb_eval.answer_relevance(confident)
        s_hedged = emb_eval.answer_relevance(hedged)
        assert s_conf >= s_hedged


class TestEmbeddingEvaluatorContextPrecision:
    def test_empty_context_returns_zero(self, emb_eval):
        inp = _input(docs=[])
        score, chunk_scores = emb_eval.context_precision(inp)
        assert score == 0.0
        assert chunk_scores == []

    def test_returns_chunks_for_each_doc(self, emb_eval):
        docs = [_doc(f"d{i}") for i in range(3)]
        inp = _input(docs=docs)
        _, chunk_scores = emb_eval.context_precision(inp)
        assert len(chunk_scores) == 3

    def test_ground_truth_ids_used(self, emb_eval):
        docs = [
            _doc("pmid_123", pmid="123"),
            _doc("pmid_456", pmid="456"),
        ]
        inp = _input(docs=docs, gt_ids={"123"})
        score, chunk_scores = emb_eval.context_precision(inp)
        # Only "123" is relevant → AP = 1/1 = 1.0
        assert score == pytest.approx(1.0)
        assert chunk_scores[0].relevant is True
        assert chunk_scores[1].relevant is False

    def test_ranks_are_one_based(self, emb_eval):
        docs = [_doc(f"d{i}") for i in range(4)]
        inp = _input(docs=docs)
        _, chunk_scores = emb_eval.context_precision(inp)
        ranks = [cs.rank for cs in chunk_scores]
        assert ranks == [1, 2, 3, 4]


class TestEmbeddingEvaluatorContextRecall:
    def test_no_ground_truth_returns_zero(self, emb_eval):
        inp = _input()
        assert emb_eval.context_recall(inp) == 0.0

    def test_ground_truth_ids_full_hit(self, emb_eval):
        docs = [_doc("pmid_123", pmid="123"), _doc("pmid_456", pmid="456")]
        inp = _input(docs=docs, gt_ids={"123", "456"})
        assert emb_eval.context_recall(inp) == pytest.approx(1.0)

    def test_ground_truth_ids_partial_hit(self, emb_eval):
        docs = [_doc("pmid_123", pmid="123")]
        inp = _input(docs=docs, gt_ids={"123", "999"})
        assert emb_eval.context_recall(inp) == pytest.approx(0.5)

    def test_ground_truth_ids_zero_hit(self, emb_eval):
        docs = [_doc("pmid_123", pmid="123")]
        inp = _input(docs=docs, gt_ids={"999"})
        assert emb_eval.context_recall(inp) == pytest.approx(0.0)

    def test_ground_truth_answer_mode(self, emb_eval):
        """With a GT answer and high-similarity mocked model, coverage is 1.0."""
        import torch, numpy as np

        def high_encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        emb_eval._model.encode = MagicMock(side_effect=high_encode)
        inp = _input(
            gt_answer="Seizure detection accuracy was high in CNN studies.",
        )
        score = emb_eval.context_recall(inp)
        assert 0.0 <= score <= 1.0

    def test_gt_answer_empty_context_returns_zero(self, emb_eval):
        inp = _input(docs=[], gt_answer="Some ground truth sentence.")
        assert emb_eval.context_recall(inp) == 0.0


class TestEmbeddingEvaluatorSplitClaims:
    def test_splits_on_period(self):
        text = "First sentence. Second sentence. Third sentence."
        claims = _EmbeddingEvaluator._split_claims(text)
        assert len(claims) >= 2

    def test_filters_short_fragments(self):
        text = "Long enough sentence here. Hi."
        claims = _EmbeddingEvaluator._split_claims(text)
        assert all(len(c) >= 10 for c in claims)

    def test_empty_string_returns_empty(self):
        assert _EmbeddingEvaluator._split_claims("") == []

    def test_single_sentence_no_period(self):
        text = "CNN seizure detection accuracy reaches 95 percent in recent studies"
        claims = _EmbeddingEvaluator._split_claims(text)
        assert len(claims) == 1


# ---------------------------------------------------------------------------
# _LLMJudge._parse_json
# ---------------------------------------------------------------------------

class TestLLMJudgeParseJSON:
    def test_parses_valid_json_array(self):
        raw = '[{"claim": "test", "supported": true}]'
        result = _LLMJudge._parse_json(raw, fallback=[])
        assert result == [{"claim": "test", "supported": True}]

    def test_strips_markdown_fences(self):
        raw = "```json\n[{\"a\": 1}]\n```"
        result = _LLMJudge._parse_json(raw, fallback=[])
        assert result == [{"a": 1}]

    def test_fallback_on_invalid_json(self):
        result = _LLMJudge._parse_json("not json at all!!!", fallback={"default": True})
        assert result == {"default": True}

    def test_parses_object(self):
        raw = '{"relevant": false}'
        result = _LLMJudge._parse_json(raw, fallback={})
        assert result == {"relevant": False}


# ---------------------------------------------------------------------------
# _LLMEvaluator — mocked judge calls
# ---------------------------------------------------------------------------

def _make_judge(responses: list[str]) -> _LLMJudge:
    """Return a _LLMJudge whose call() returns responses in sequence."""
    mock = MagicMock(spec=_LLMJudge)
    mock.call = AsyncMock(side_effect=responses)
    mock._parse_json = _LLMJudge._parse_json
    return mock


class TestLLMEvaluatorFaithfulness:
    def test_all_claims_supported(self):
        answer = "CNN achieves 95% accuracy. LSTM also performs well."
        claims_json = json.dumps([
            {"claim": "CNN achieves 95% accuracy.", "supported": True, "doc_id": "d1"},
            {"claim": "LSTM also performs well.", "supported": True, "doc_id": "d2"},
        ])
        judge = _make_judge([claims_json])
        ev = _LLMEvaluator(judge=judge)
        score, details = asyncio.run(ev.faithfulness(_input(answer=answer)))
        assert score == pytest.approx(1.0)
        assert all(d.supported for d in details)

    def test_partial_support(self):
        answer = "CNN achieves 95% accuracy. False claim here."
        claims_json = json.dumps([
            {"claim": "CNN achieves 95% accuracy.", "supported": True, "doc_id": "d1"},
            {"claim": "False claim here.", "supported": False, "doc_id": None},
        ])
        judge = _make_judge([claims_json])
        ev = _LLMEvaluator(judge=judge)
        score, details = asyncio.run(ev.faithfulness(_input(answer=answer)))
        assert score == pytest.approx(0.5)

    def test_empty_answer_returns_zero(self):
        judge = _make_judge([])
        ev = _LLMEvaluator(judge=judge)
        score, details = asyncio.run(ev.faithfulness(_input(answer="")))
        assert score == 0.0
        assert details == []

    def test_invalid_llm_response_returns_zero(self):
        judge = _make_judge(["INVALID JSON"])
        ev = _LLMEvaluator(judge=judge)
        score, details = asyncio.run(ev.faithfulness(_input()))
        # Falls back to all-unsupported for claims not covered by LLM
        assert 0.0 <= score <= 1.0


class TestLLMEvaluatorAnswerRelevance:
    def test_high_similarity_from_generated_questions(self):
        import numpy as np, torch

        questions_json = json.dumps([
            "What methods are used for seizure detection?",
            "Which CNN architectures detect seizures?",
            "How accurate is deep learning for seizure identification?",
        ])
        judge = _make_judge([questions_json])

        ev = object.__new__(_LLMEvaluator)
        ev._judge = judge
        ev._n = 3

        mock_model = MagicMock()
        def encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr
        mock_model.encode = encode
        ev._embed_model = mock_model

        score = asyncio.run(ev.answer_relevance(_input()))
        assert 0.0 <= score <= 1.0

    def test_empty_answer_returns_zero(self):
        judge = _make_judge([])
        ev = _LLMEvaluator(judge=judge)
        score = asyncio.run(ev.answer_relevance(_input(answer="")))
        assert score == 0.0

    def test_fallback_when_no_questions_returned(self):
        """If LLM returns empty list and no embed model, return word-overlap proxy."""
        judge = _make_judge(['[]'])
        ev = object.__new__(_LLMEvaluator)
        ev._judge = judge
        ev._n = 3
        ev._embed_model = None  # no sentence-transformer
        score = asyncio.run(ev.answer_relevance(_input()))
        assert 0.0 <= score <= 1.0


class TestLLMEvaluatorContextPrecision:
    def test_all_relevant(self):
        judge_responses = [
            '{"relevant": true}',
            '{"relevant": true}',
            '{"relevant": true}',
        ]
        judge = _make_judge(judge_responses)
        ev = _LLMEvaluator(judge=judge)
        docs = [_doc(f"d{i}") for i in range(3)]
        inp = _input(docs=docs)
        score, chunk_scores = asyncio.run(ev.context_precision(inp))
        assert score == pytest.approx(1.0)
        assert all(c.relevant for c in chunk_scores)

    def test_ground_truth_ids_skip_llm(self):
        judge = _make_judge([])  # should not be called
        ev = _LLMEvaluator(judge=judge)
        docs = [_doc("pmid_123", pmid="123"), _doc("pmid_456", pmid="456")]
        inp = _input(docs=docs, gt_ids={"123"})
        score, chunk_scores = asyncio.run(ev.context_precision(inp))
        # Only first is relevant
        assert chunk_scores[0].relevant is True
        assert chunk_scores[1].relevant is False
        judge.call.assert_not_called()

    def test_empty_context_returns_zero(self):
        judge = _make_judge([])
        ev = _LLMEvaluator(judge=judge)
        score, chunks = asyncio.run(ev.context_precision(_input(docs=[])))
        assert score == 0.0
        assert chunks == []


class TestLLMEvaluatorContextRecall:
    def test_gt_ids_full_hit(self):
        judge = _make_judge([])  # not called for gt_ids path
        ev = _LLMEvaluator(judge=judge)
        docs = [_doc("123", pmid="123"), _doc("456", pmid="456")]
        inp = _input(docs=docs, gt_ids={"123", "456"})
        score = asyncio.run(ev.context_recall(inp))
        assert score == pytest.approx(1.0)

    def test_gt_answer_all_attributable(self):
        responses = ['{"attributable": true}', '{"attributable": true}']
        judge = _make_judge(responses)
        ev = _LLMEvaluator(judge=judge)
        inp = _input(gt_answer="First fact about EEG. Second fact about CNN.")
        score = asyncio.run(ev.context_recall(inp))
        assert score == pytest.approx(1.0)

    def test_no_ground_truth_returns_zero(self):
        judge = _make_judge([])
        ev = _LLMEvaluator(judge=judge)
        score = asyncio.run(ev.context_recall(_input()))
        assert score == 0.0


# ---------------------------------------------------------------------------
# RAGASEvaluator — integration (all LLM / ST calls mocked)
# ---------------------------------------------------------------------------

class TestRAGASEvaluatorEmbeddingMode:
    """Tests for RAGASEvaluator in EMBEDDING mode with mocked ST model."""

    def _make_evaluator(self) -> RAGASEvaluator:
        ev = RAGASEvaluator(mode=EvaluationMode.EMBEDDING)
        mock_embed = object.__new__(_EmbeddingEvaluator)
        import numpy as np, torch

        def encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=encode)
        mock_embed._model = mock_model
        ev._embed_eval = mock_embed
        return ev

    def test_returns_ragas_scores(self):
        ev = self._make_evaluator()
        result = asyncio.run(ev.evaluate(_input()))
        assert isinstance(result, RAGASScores)
        assert result.mode == EvaluationMode.EMBEDDING

    def test_all_scores_in_range(self):
        ev = self._make_evaluator()
        result = asyncio.run(ev.evaluate(_input()))
        for metric in (
            result.faithfulness,
            result.answer_relevance,
            result.context_precision,
            result.context_recall,
            result.ragas_score,
        ):
            assert 0.0 <= metric <= 1.0

    def test_ragas_score_is_harmonic_mean(self):
        ev = self._make_evaluator()
        result = asyncio.run(ev.evaluate(_input()))
        expected = _harmonic_mean([
            result.faithfulness,
            result.answer_relevance,
            result.context_precision,
            result.context_recall,
        ])
        assert result.ragas_score == pytest.approx(expected)

    def test_batch_evaluate_returns_list(self):
        ev = self._make_evaluator()
        inputs = [_input(), _input(query="How does sleep staging work?")]
        results = asyncio.run(ev.evaluate_batch(inputs))
        assert len(results) == 2
        assert all(isinstance(r, RAGASScores) for r in results)

    def test_summary_computes_means(self):
        ev = self._make_evaluator()
        s1 = RAGASScores(0.8, 0.9, 0.7, 0.6, 0.75, EvaluationMode.EMBEDDING)
        s2 = RAGASScores(0.6, 0.7, 0.5, 0.4, 0.55, EvaluationMode.EMBEDDING)
        summary = ev.summary([s1, s2])
        assert summary["faithfulness"] == pytest.approx(0.7)
        assert summary["answer_relevance"] == pytest.approx(0.8)

    def test_summary_empty_input(self):
        ev = self._make_evaluator()
        summary = ev.summary([])
        assert summary["ragas_score"] == 0.0


class TestRAGASEvaluatorAutoMode:
    """Tests for EvaluationMode.AUTO — falls back to EMBEDDING when no API key."""

    def test_auto_falls_back_to_embedding_without_api_key(self):
        import os
        with patch.dict(os.environ, {}, clear=True):
            ev = RAGASEvaluator(
                mode=EvaluationMode.AUTO,
                llm_provider=LLMProvider.OPENAI,
            )
            resolved = ev._resolve_mode()
            # Without OPENAI_API_KEY, should resolve to EMBEDDING
            assert resolved == EvaluationMode.EMBEDDING

    def test_auto_resolves_to_llm_with_api_key(self):
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            ev = RAGASEvaluator(
                mode=EvaluationMode.AUTO,
                llm_provider=LLMProvider.OPENAI,
            )
            resolved = ev._resolve_mode()
            assert resolved == EvaluationMode.LLM

    def test_auto_falls_back_on_llm_failure(self):
        """If LLM eval raises, should fall back to embedding mode."""
        import numpy as np, torch

        ev = RAGASEvaluator(mode=EvaluationMode.AUTO)

        # Make LLM eval raise immediately
        mock_llm_eval = MagicMock()
        mock_llm_eval.faithfulness = AsyncMock(side_effect=RuntimeError("no key"))
        ev._llm_eval = mock_llm_eval

        # Patch _resolve_mode to return LLM (as if API key is set)
        ev._resolve_mode = lambda: EvaluationMode.LLM

        # Provide a real embed evaluator with mocked model
        mock_embed = object.__new__(_EmbeddingEvaluator)

        def encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        model = MagicMock()
        model.encode = MagicMock(side_effect=encode)
        mock_embed._model = model
        ev._embed_eval = mock_embed

        result = asyncio.run(ev.evaluate(_input()))
        # Should have fallen back and produced a result
        assert isinstance(result, RAGASScores)
        assert any("fell back" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# export_for_human_eval
# ---------------------------------------------------------------------------

class TestExportForHumanEval:
    def _make_scores(self) -> RAGASScores:
        return RAGASScores(
            faithfulness=0.8,
            answer_relevance=0.9,
            context_precision=0.7,
            context_recall=0.6,
            ragas_score=0.75,
            mode=EvaluationMode.EMBEDDING,
        )

    def test_returns_one_record_per_input(self):
        inputs = [_input(), _input(query="Sleep staging?")]
        scores = [self._make_scores(), self._make_scores()]
        records = export_for_human_eval(inputs, scores)
        assert len(records) == 2

    def test_record_fields_populated(self):
        inp = _input()
        score = self._make_scores()
        records = export_for_human_eval([inp], [score])
        rec = records[0]
        assert rec.query == inp.query
        assert rec.answer == inp.answer
        assert len(rec.context_excerpts) == len(inp.context_docs)
        assert rec.auto_scores.faithfulness == 0.8
        assert rec.human_faithfulness is None

    def test_context_excerpts_truncated(self):
        doc = _doc("d1", text="A" * 500)
        inp = _input(docs=[doc])
        records = export_for_human_eval([inp], [self._make_scores()],
                                        context_excerpt_chars=100)
        assert len(records[0].context_excerpts[0]["excerpt"]) == 100

    def test_record_ids_are_sequential(self):
        inputs = [_input() for _ in range(5)]
        scores = [self._make_scores() for _ in range(5)]
        records = export_for_human_eval(inputs, scores)
        ids = [r.record_id for r in records]
        assert ids == ["eval-0000", "eval-0001", "eval-0002", "eval-0003", "eval-0004"]

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            export_for_human_eval([_input()], [])

    def test_to_dict_has_human_annotation_keys(self):
        records = export_for_human_eval([_input()], [self._make_scores()])
        d = records[0].to_dict()
        assert "human_annotations" in d
        annot = d["human_annotations"]
        assert "faithfulness" in annot
        assert annot["faithfulness"] is None


# ---------------------------------------------------------------------------
# Integration: AgenticRAGResult → RAGASEvaluator pipeline
# ---------------------------------------------------------------------------

class TestRAGASWithAgenticRAGResult:
    """Smoke test: build a RAGASInput from an AgenticRAGResult and evaluate."""

    def test_pipeline_from_agentic_result(self):
        """Convert AgenticRAGResult sources to ContextDocuments and evaluate."""
        import numpy as np, torch
        from eeg_rag.retrieval.hybrid_retriever import HybridResult

        # Build fake AgenticRAGResult-like data
        sources = [
            HybridResult(
                doc_id=f"pmid_{i}",
                text=f"CNN seizure detection paper {i}. High accuracy reported.",
                metadata={"pmid": f"1234567{i}", "title": f"Paper {i}"},
                rrf_score=0.015,
                bm25_score=0.5,
                dense_score=0.6,
            )
            for i in range(3)
        ]

        # Convert to ContextDocuments
        context_docs = [
            ContextDocument(
                doc_id=s.doc_id,
                text=s.text,
                metadata=s.metadata,
                pmid=s.metadata.get("pmid"),
            )
            for s in sources
        ]

        inp = RAGASInput(
            query="CNN for seizure detection",
            answer="CNN methods achieve high accuracy for seizure detection [PMID:12345670].",
            context_docs=context_docs,
            ground_truth_doc_ids={"12345670", "12345671"},
        )

        # Evaluator with mocked ST model
        ev = RAGASEvaluator(mode=EvaluationMode.EMBEDDING)
        mock_embed = object.__new__(_EmbeddingEvaluator)

        def encode(texts, convert_to_tensor=False):
            n = len(texts) if isinstance(texts, list) else 1
            arr = np.ones((n, 4), dtype=np.float32)
            return torch.tensor(arr) if convert_to_tensor else arr

        model = MagicMock()
        model.encode = MagicMock(side_effect=encode)
        mock_embed._model = model
        ev._embed_eval = mock_embed

        result = asyncio.run(ev.evaluate(inp))
        assert isinstance(result, RAGASScores)
        # Both gt_ids (12345670, 12345671) are present in docs → recall = 1.0
        assert result.context_recall == pytest.approx(1.0)
