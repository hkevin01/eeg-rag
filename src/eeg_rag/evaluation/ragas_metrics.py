"""
# =============================================================================
# ID:             MOD-EVAL-RAGAS-001
# Requirement:    REQ-EVAL-001 — Faithfulness: fraction of answer claims
#                                grounded in the retrieved context;
#                 REQ-EVAL-002 — Answer Relevance: degree to which the answer
#                                addresses the original question;
#                 REQ-EVAL-003 — Context Precision: fraction of retrieved
#                                chunks that are actually relevant (ranked
#                                quality of the retrieval set);
#                 REQ-EVAL-004 — Context Recall: fraction of ground-truth
#                                relevant evidence present in retrieved chunks.
# Purpose:        Provide RAGAS-style automated evaluation of the EEG-RAG
#                 pipeline.  Supports two evaluation modes:
#                 - LLM-as-judge (production): calls a configured LLM to
#                   decompose claims and verify them against context.
#                 - Embedding-based (fast/offline): uses sentence-transformers
#                   cosine similarity as a cheap proxy without LLM API calls.
#                 Both modes produce scores in [0, 1] with identical interfaces.
# Rationale:      RAGAS (Shahul et al. 2023) proposed metric definitions that
#                 isolate retrieval quality (precision/recall) from generation
#                 quality (faithfulness/relevance).  Implementing them inside
#                 the EEG-RAG repo allows CI/CD regression gating without an
#                 external RAGAS service.
# Inputs:         RAGASInput dataclass: query, answer, context_docs,
#                 ground_truth_answer (optional), ground_truth_doc_ids (opt).
# Outputs:        RAGASScores dataclass with per-metric floats (0–1) and
#                 a composite RAGAS score (harmonic mean of all four).
# Failure Modes:  LLM provider unavailable → falls back to embedding mode;
#                 empty context → Context Precision/Recall return 0;
#                 empty answer → Faithfulness/Relevance return 0.
# Constraints:    Embedding mode: all-MiniLM-L6-v2 (no GPU required);
#                 LLM mode: requires OPENAI_API_KEY or ANTHROPIC_API_KEY.
# Verification:   tests/test_ragas_metrics.py
# References:     Shahul et al. 2023 "RAGAS: Automated Evaluation of
#                 Retrieval Augmented Generation" (arXiv:2309.15217);
#                 REQ-EVAL-001–004.
# =============================================================================
RAGAS-style evaluation metrics for EEG-RAG.

Four core metrics — Faithfulness, Answer Relevance, Context Precision,
Context Recall — implemented in two modes:

* **Embedding mode** (``EvaluationMode.EMBEDDING``): fast, fully offline,
  uses ``sentence-transformers`` cosine similarity.
* **LLM-as-judge mode** (``EvaluationMode.LLM``): higher fidelity, calls
  an LLM to decompose claims and classify chunk relevance.

Usage::

    evaluator = RAGASEvaluator(mode=EvaluationMode.EMBEDDING)

    score = await evaluator.evaluate(
        RAGASInput(
            query="CNN methods for seizure detection",
            answer="CNN-based methods achieve 95% accuracy ... [PMID:12345678]",
            context_docs=[...],           # list of retrieved text chunks
            ground_truth_doc_ids={"12345678", "23456789"},   # optional
        )
    )
    print(score.faithfulness)          # 0.0 – 1.0
    print(score.context_precision)     # 0.0 – 1.0
    print(score.ragas_score)           # harmonic mean of all four
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import statistics
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful absence)
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim as _st_cos_sim
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed; embedding-based RAGAS metrics "
        "will return 0.0.  Install with: pip install sentence-transformers"
    )

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# LLM provider imports (optional)
try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PMID_RE = re.compile(r'\bPMID[:\s]*(\d{7,8})\b', re.IGNORECASE)
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
_CLAIM_SPLIT_RE = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|;)\s+'
)
# Sentences that are definitively unsupported signal words
_HALLUCINATION_HEDGES = frozenset({
    "i believe", "i think", "i would guess", "it is possible",
    "there is speculation", "no studies have",
})


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EvaluationMode(Enum):
    """Evaluation backend for RAGAS metrics.

    Values:
        EMBEDDING: Use sentence-transformer cosine similarity (fast, offline).
        LLM: Use LLM-as-judge calls for claim verification (higher fidelity).
        AUTO: Try LLM first; fall back to EMBEDDING if no provider available.
    """

    EMBEDDING = "embedding"
    LLM = "llm"
    AUTO = "auto"


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.LLMProvider
# Requirement  : `LLMProvider` class shall be instantiable and expose the documented interface
# Purpose      : LLM providers available for the judge step
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate LLMProvider with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class LLMProvider(Enum):
    """LLM providers available for the judge step."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContextDocument:
    """A single retrieved context chunk passed to the evaluator.

    Attributes:
        doc_id: Unique identifier (PMID, hash, etc.).
        text: Full text of the chunk.
        metadata: Arbitrary key-value pairs (title, authors, year, …).
        pmid: Convenience shortcut for PubMed ID if present.
    """

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pmid: Optional[str] = None

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.ContextDocument.__post_init__
    # Requirement  : `__post_init__` shall execute as specified
    # Purpose      :   post init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.pmid is None:
            self.pmid = (
                self.metadata.get("pmid")
                or self.metadata.get("PMID")
            )


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.RAGASInput
# Requirement  : `RAGASInput` class shall be instantiable and expose the documented interface
# Purpose      : All inputs required to compute the four RAGAS metrics
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RAGASInput with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class RAGASInput:
    """All inputs required to compute the four RAGAS metrics.

    Attributes:
        query: The original user question.
        answer: The generated answer text to evaluate.
        context_docs: Ordered list of retrieved context chunks (rank-ordered,
            first = highest retrieval score).
        ground_truth_answer: Reference answer used for Context Recall
            (optional; required only for Context Recall in LLM mode).
        ground_truth_doc_ids: Set of doc_ids that are known relevant
            (used for Context Precision / Recall ground-truth mode).
    """

    query: str
    answer: str
    context_docs: List[ContextDocument]
    ground_truth_answer: Optional[str] = None
    ground_truth_doc_ids: Optional[Set[str]] = None


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.FaithfulnessDetail
# Requirement  : `FaithfulnessDetail` class shall be instantiable and expose the documented interface
# Purpose      : Per-claim faithfulness result
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate FaithfulnessDetail with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class FaithfulnessDetail:
    """Per-claim faithfulness result.

    Attributes:
        claim: The atomic statement extracted from the answer.
        supported: True if the claim is grounded in the context.
        supporting_doc_ids: doc_ids of context docs that support the claim.
        confidence: Confidence in the classification (0–1).
    """

    claim: str
    supported: bool
    supporting_doc_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.ContextChunkScore
# Requirement  : `ContextChunkScore` class shall be instantiable and expose the documented interface
# Purpose      : Relevance verdict for a single retrieved context chunk
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ContextChunkScore with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ContextChunkScore:
    """Relevance verdict for a single retrieved context chunk.

    Attributes:
        doc_id: Identifier of the retrieved chunk.
        relevant: True if the chunk contributed relevant evidence.
        relevance_score: Continuous relevance score (0–1).
        rank: 1-based rank in retrieval order.
    """

    doc_id: str
    relevant: bool
    relevance_score: float
    rank: int


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.RAGASScores
# Requirement  : `RAGASScores` class shall be instantiable and expose the documented interface
# Purpose      : All four RAGAS metric scores plus a composite
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RAGASScores with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class RAGASScores:
    """All four RAGAS metric scores plus a composite.

    All values are in [0, 1]; higher is better.

    Attributes:
        faithfulness: Fraction of answer claims supported by context.
            A score of 1.0 means every claim in the answer is grounded.
        answer_relevance: How well the answer addresses the query.
            A score of 1.0 means the answer is perfectly on-topic.
        context_precision: Quality/ranking of retrieved chunks.
            Weighted precision — relevant chunks ranked higher increase the score.
        context_recall: Coverage of ground-truth evidence by retrieved chunks.
            A score of 1.0 means all relevant evidence was retrieved.
        ragas_score: Harmonic mean of all four metrics (composite).
        mode: Evaluation mode used to compute these scores.
        faithfulness_details: Per-claim breakdown (if LLM mode).
        chunk_scores: Per-chunk relevance breakdown (if available).
        warnings: Non-fatal issues encountered during evaluation.
    """

    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    ragas_score: float
    mode: EvaluationMode
    faithfulness_details: List[FaithfulnessDetail] = field(default_factory=list)
    chunk_scores: List[ContextChunkScore] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASScores.to_dict
    # Requirement  : `to_dict` shall serialise to a plain dictionary for logging or JSON export
    # Purpose      : Serialise to a plain dictionary for logging or JSON export
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary for logging or JSON export."""
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "ragas_score": round(self.ragas_score, 4),
            "mode": self.mode.value,
            "faithfulness_details": [
                {
                    "claim": d.claim,
                    "supported": d.supported,
                    "supporting_doc_ids": d.supporting_doc_ids,
                    "confidence": round(d.confidence, 4),
                }
                for d in self.faithfulness_details
            ],
            "chunk_scores": [
                {
                    "doc_id": c.doc_id,
                    "relevant": c.relevant,
                    "relevance_score": round(c.relevance_score, 4),
                    "rank": c.rank,
                }
                for c in self.chunk_scores
            ],
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Embedding-based sub-evaluators
# ---------------------------------------------------------------------------


class _EmbeddingEvaluator:
    """Computes all four RAGAS metrics using sentence-transformer similarity.

    This is the offline, LLM-free evaluation path.  It is deliberately
    conservative (the scores act as lower bounds on the true values) because
    cosine similarity over MiniLM embeddings cannot perform full logical
    entailment checking.

    Args:
        model_name: Sentence-transformer model to load.
    """

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._EmbeddingEvaluator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : model_name: str (default=_DEFAULT_EMBED_MODEL)
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, model_name: str = _DEFAULT_EMBED_MODEL) -> None:
        if not _ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required for EMBEDDING mode. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding evaluator loaded model: %s", model_name)

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    def faithfulness(self, inp: RAGASInput) -> tuple[float, list[FaithfulnessDetail]]:
        """Estimate faithfulness via claim→context cosine similarity.

        Each sentence in the answer is treated as an atomic claim.  A claim
        is considered "supported" if its maximum cosine similarity with any
        context sentence exceeds a threshold (0.40).

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Tuple of (score, per-claim details).
        """
        if not inp.answer.strip() or not inp.context_docs:
            return 0.0, []

        claims = self._split_claims(inp.answer)
        if not claims:
            return 0.0, []

        # Build context sentence pool
        context_sentences: list[str] = []
        sentence_to_doc: list[str] = []
        for doc in inp.context_docs:
            for sent in self._split_claims(doc.text):
                context_sentences.append(sent)
                sentence_to_doc.append(doc.doc_id)

        if not context_sentences:
            return 0.0, []

        claim_embs = self._model.encode(claims, convert_to_tensor=True)
        ctx_embs = self._model.encode(context_sentences, convert_to_tensor=True)

        # shape: [n_claims, n_ctx_sentences]
        sim_matrix = _st_cos_sim(claim_embs, ctx_embs).cpu().numpy()

        details: list[FaithfulnessDetail] = []
        threshold = 0.40
        for i, claim in enumerate(claims):
            max_sim_idx = int(np.argmax(sim_matrix[i]))
            max_sim = float(sim_matrix[i][max_sim_idx])
            supported = max_sim >= threshold
            details.append(
                FaithfulnessDetail(
                    claim=claim,
                    supported=supported,
                    supporting_doc_ids=(
                        [sentence_to_doc[max_sim_idx]] if supported else []
                    ),
                    confidence=max_sim,
                )
            )

        score = sum(d.supported for d in details) / len(details)
        return score, details

    # ------------------------------------------------------------------
    # Answer Relevance
    # ------------------------------------------------------------------

    def answer_relevance(self, inp: RAGASInput) -> float:
        """Estimate answer relevance via query–answer cosine similarity.

        Computes the cosine similarity between the embedded query and the
        embedded answer.  A short answer that barely addresses the query
        receives a low score; a comprehensive, on-topic answer scores high.

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Score in [0, 1].
        """
        if not inp.answer.strip() or not inp.query.strip():
            return 0.0

        query_emb = self._model.encode([inp.query], convert_to_tensor=True)
        answer_emb = self._model.encode([inp.answer], convert_to_tensor=True)
        sim = float(_st_cos_sim(query_emb, answer_emb)[0][0])

        # Penalise for non-committal phrasing
        penalty = 0.0
        ans_lower = inp.answer.lower()
        for hedge in _HALLUCINATION_HEDGES:
            if hedge in ans_lower:
                penalty += 0.05
        return max(0.0, min(1.0, sim - penalty))

    # ------------------------------------------------------------------
    # Context Precision
    # ------------------------------------------------------------------

    def context_precision(
        self,
        inp: RAGASInput,
    ) -> tuple[float, list[ContextChunkScore]]:
        """Compute weighted context precision (AP-style).

        A chunk is judged relevant if its cosine similarity with the query
        exceeds a threshold (0.35).  The final score is the average precision
        over the ranked list — identical to the RAGAS definition when
        ground-truth relevance is replaced by embedding-based proxy relevance.

        Weighted average precision formula:
            sum_k (precision@k * rel_k) / number_of_relevant_chunks

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Tuple of (weighted precision score, per-chunk scores).
        """
        if not inp.context_docs:
            return 0.0, []

        query_emb = self._model.encode([inp.query], convert_to_tensor=True)
        doc_texts = [d.text for d in inp.context_docs]
        doc_embs = self._model.encode(doc_texts, convert_to_tensor=True)
        sims = _st_cos_sim(query_emb, doc_embs)[0].cpu().numpy()

        threshold = 0.35
        chunk_scores: list[ContextChunkScore] = []
        for rank, (doc, sim) in enumerate(zip(inp.context_docs, sims), 1):
            # Override with ground-truth if available
            if inp.ground_truth_doc_ids is not None:
                relevant = doc.doc_id in inp.ground_truth_doc_ids or (
                    doc.pmid is not None
                    and doc.pmid in inp.ground_truth_doc_ids
                )
            else:
                relevant = float(sim) >= threshold
            chunk_scores.append(
                ContextChunkScore(
                    doc_id=doc.doc_id,
                    relevant=relevant,
                    relevance_score=float(sim),
                    rank=rank,
                )
            )

        # Weighted mean precision
        running_hits = 0
        precision_sum = 0.0
        total_relevant = sum(c.relevant for c in chunk_scores)

        for cs in chunk_scores:
            if cs.relevant:
                running_hits += 1
                precision_sum += running_hits / cs.rank

        score = precision_sum / total_relevant if total_relevant else 0.0
        return score, chunk_scores

    # ------------------------------------------------------------------
    # Context Recall
    # ------------------------------------------------------------------

    def context_recall(self, inp: RAGASInput) -> float:
        """Estimate context recall via ground-truth evidence coverage.

        If ground_truth_doc_ids are provided: recall = fraction of them
        present in the retrieved set.

        If a ground_truth_answer is provided: decompose it into sentences,
        then for each sentence find its max cosine similarity with any
        retrieved context sentence.  A sentence is "covered" if max sim ≥ 0.35.

        If neither is provided, returns 0.0 (cannot compute without ground truth).

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Score in [0, 1].
        """
        # --- Ground-truth doc_ids mode -----------------------------------
        if inp.ground_truth_doc_ids:
            retrieved_ids: set[str] = set()
            for doc in inp.context_docs:
                retrieved_ids.add(doc.doc_id)
                if doc.pmid:
                    retrieved_ids.add(doc.pmid)
            hits = inp.ground_truth_doc_ids & retrieved_ids
            return len(hits) / len(inp.ground_truth_doc_ids)

        # --- Ground-truth answer mode ------------------------------------
        if inp.ground_truth_answer:
            gt_sentences = self._split_claims(inp.ground_truth_answer)
            if not gt_sentences:
                return 0.0

            context_sentences = [
                sent
                for doc in inp.context_docs
                for sent in self._split_claims(doc.text)
            ]
            if not context_sentences:
                return 0.0

            gt_embs = self._model.encode(gt_sentences, convert_to_tensor=True)
            ctx_embs = self._model.encode(context_sentences, convert_to_tensor=True)
            sim_matrix = _st_cos_sim(gt_embs, ctx_embs).cpu().numpy()

            threshold = 0.35
            covered = sum(
                1
                for i in range(len(gt_sentences))
                if float(np.max(sim_matrix[i])) >= threshold
            )
            return covered / len(gt_sentences)

        # No ground truth → cannot compute recall
        logger.debug(
            "context_recall: no ground_truth_doc_ids or ground_truth_answer "
            "provided; returning 0.0"
        )
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_claims(text: str) -> list[str]:
        """Split text into atomic sentences suitable as claims.

        Splits on sentence-ending punctuation.  Filters out fragments shorter
        than 10 characters.

        Args:
            text: Input text to split.

        Returns:
            List of sentence strings.
        """
        raw = _CLAIM_SPLIT_RE.split(text.strip())
        return [s.strip() for s in raw if len(s.strip()) >= 10]


# ---------------------------------------------------------------------------
# LLM-as-judge sub-evaluator
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """\
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems
specialising in EEG and neuroscience research.

Your task: decide whether each claim extracted from an AI-generated answer is
SUPPORTED by the provided context documents.

Rules:
- A claim is SUPPORTED if it can be directly inferred from at least one context
  document, even if worded differently.
- A claim is NOT SUPPORTED if it contradicts the context or cannot be inferred.
- If a claim is a general scientific fact (e.g. "the brain generates electrical
  signals"), mark it SUPPORTED only if the context mentions it.
- Respond with a JSON array only, no markdown, no commentary.

Output format (JSON array, one object per claim):
[
  {"claim": "<exact claim text>", "supported": true, "doc_id": "<doc_id or null>"},
  ...
]
"""

_RELEVANCE_SYSTEM = """\
You are an evaluator for EEG research RAG systems.

Generate {n} independent questions that the following answer is trying to address.
Output only a JSON array of question strings, no markdown.

Example output:
["What CNN architecture is best for seizure detection?", ...]
"""

_PRECISION_SYSTEM = """\
You are an evaluator for EEG research RAG systems.

Given a user question and a retrieved document chunk, decide whether the chunk
is RELEVANT to answering the question.

A chunk is RELEVANT if it contains information that would help answer the question.
Output a single JSON object: {"relevant": true} or {"relevant": false}.
No markdown, no commentary.
"""

_RECALL_SYSTEM = """\
You are an evaluator for EEG research RAG systems.

Given a ground-truth statement and a set of retrieved context chunks, decide
whether the statement is ATTRIBUTABLE to (can be inferred from) any of the chunks.

Output a single JSON object: {"attributable": true, "doc_id": "<doc_id or null>"}
No markdown, no commentary.
"""


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics._LLMJudge
# Requirement  : `_LLMJudge` class shall be instantiable and expose the documented interface
# Purpose      : Calls an LLM provider to act as evaluation judge
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate _LLMJudge with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class _LLMJudge:
    """Calls an LLM provider to act as evaluation judge.

    Args:
        provider: Which LLM provider to use.
        model: Model identifier (provider-specific).
        temperature: Sampling temperature (low for deterministic eval).
        timeout: HTTP request timeout in seconds.
    """

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : provider: LLMProvider; model: Optional[str] (default=None); temperature: float (default=0.0); timeout: float (default=30.0)
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._timeout = timeout
        self._model = model or self._default_model(provider)

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge._default_model
    # Requirement  : `_default_model` shall execute as specified
    # Purpose      :  default model
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : provider: LLMProvider
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        defaults = {
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
            LLMProvider.OLLAMA: "llama3",
        }
        return defaults[provider]

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge.call
    # Requirement  : `call` shall call the LLM and return raw text response
    # Purpose      : Call the LLM and return raw text response
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : system: str; user: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def call(self, system: str, user: str) -> str:
        """Call the LLM and return raw text response.

        Args:
            system: System prompt.
            user: User message.

        Returns:
            Raw text from the LLM.

        Raises:
            RuntimeError: If the provider library is not installed or API key
                is missing.
        """
        if self._provider == LLMProvider.OPENAI:
            return await self._call_openai(system, user)
        if self._provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(system, user)
        if self._provider == LLMProvider.OLLAMA:
            return await self._call_ollama(system, user)
        raise RuntimeError(f"Unknown provider: {self._provider}")

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge._call_openai
    # Requirement  : `_call_openai` shall execute as specified
    # Purpose      :  call openai
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : system: str; user: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _call_openai(self, system: str, user: str) -> str:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        client = _openai.AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            timeout=self._timeout,
        )
        return resp.choices[0].message.content or ""

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge._call_anthropic
    # Requirement  : `_call_anthropic` shall execute as specified
    # Purpose      :  call anthropic
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : system: str; user: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _call_anthropic(self, system: str, user: str) -> str:
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic package not installed.")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        client = _anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text if msg.content else ""

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge._call_ollama
    # Requirement  : `_call_ollama` shall execute as specified
    # Purpose      :  call ollama
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : system: str; user: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _call_ollama(self, system: str, user: str) -> str:
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx package not installed.")
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        prompt = f"{system}\n\nUser: {user}\nAssistant:"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self._temperature},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMJudge._parse_json
    # Requirement  : `_parse_json` shall parse JSON from LLM output, stripping markdown fences
    # Purpose      : Parse JSON from LLM output, stripping markdown fences
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str; fallback: Any
    # Outputs      : Any
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    @staticmethod
    def _parse_json(text: str, fallback: Any) -> Any:
        """Parse JSON from LLM output, stripping markdown fences."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM output is not valid JSON: %r", text[:200])
            return fallback


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics._LLMEvaluator
# Requirement  : `_LLMEvaluator` class shall be instantiable and expose the documented interface
# Purpose      : Computes RAGAS metrics using an LLM as the verification judge
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate _LLMEvaluator with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class _LLMEvaluator:
    """Computes RAGAS metrics using an LLM as the verification judge.

    Args:
        judge: Configured :class:`_LLMJudge` instance.
        n_relevance_questions: Number of synthetic questions to generate for
            Answer Relevance scoring (default: 3).
        embed_model_name: Fallback sentence-transformer model for semantic
            similarity in the Answer Relevance step.
    """

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMEvaluator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : judge: _LLMJudge; n_relevance_questions: int (default=3); embed_model_name: str (default=_DEFAULT_EMBED_MODEL)
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        judge: _LLMJudge,
        n_relevance_questions: int = 3,
        embed_model_name: str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        self._judge = judge
        self._n = n_relevance_questions
        if _ST_AVAILABLE:
            self._embed_model = SentenceTransformer(embed_model_name)
        else:
            self._embed_model = None

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    async def faithfulness(
        self, inp: RAGASInput
    ) -> tuple[float, list[FaithfulnessDetail]]:
        """Compute LLM-verified faithfulness.

        Procedure:
        1. Split the answer into atomic claims (sentences).
        2. Build a context string with labelled doc sections.
        3. Ask the LLM to classify each claim as supported/unsupported.

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Tuple of (score, per-claim details).
        """
        if not inp.answer.strip() or not inp.context_docs:
            return 0.0, []

        claims = _EmbeddingEvaluator._split_claims(inp.answer)
        if not claims:
            return 0.0, []

        context_block = "\n\n".join(
            f"[{doc.doc_id}]\n{doc.text[:800]}"
            for doc in inp.context_docs[:10]  # cap at 10 docs for token budget
        )
        claims_text = json.dumps(claims, indent=2)

        user_msg = (
            f"# Context Documents\n{context_block}\n\n"
            f"# Claims to Verify\n{claims_text}"
        )

        raw = await self._judge.call(_FAITHFULNESS_SYSTEM, user_msg)
        parsed = self._judge._parse_json(raw, fallback=[])

        details: list[FaithfulnessDetail] = []
        for item in parsed if isinstance(parsed, list) else []:
            claim = item.get("claim", "")
            supported = bool(item.get("supported", False))
            doc_id = item.get("doc_id") or ""
            details.append(
                FaithfulnessDetail(
                    claim=claim,
                    supported=supported,
                    supporting_doc_ids=[doc_id] if doc_id else [],
                    confidence=1.0 if supported else 0.0,
                )
            )

        # If LLM returned fewer items than claims, append unsupported for rest
        if len(details) < len(claims):
            for claim in claims[len(details):]:
                details.append(
                    FaithfulnessDetail(
                        claim=claim,
                        supported=False,
                        confidence=0.0,
                    )
                )

        score = (
            sum(d.supported for d in details) / len(details)
            if details
            else 0.0
        )
        return score, details

    # ------------------------------------------------------------------
    # Answer Relevance
    # ------------------------------------------------------------------

    async def answer_relevance(self, inp: RAGASInput) -> float:
        """Compute LLM-based answer relevance.

        Procedure:
        1. Ask the LLM to generate ``n`` questions from the answer.
        2. Embed the original query and each generated question.
        3. Return the mean cosine similarity.

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Score in [0, 1].
        """
        if not inp.answer.strip():
            return 0.0

        system = _RELEVANCE_SYSTEM.format(n=self._n)
        user_msg = f"# Answer\n{inp.answer[:2000]}"

        raw = await self._judge.call(system, user_msg)
        generated_qs = self._judge._parse_json(raw, fallback=[])

        if not isinstance(generated_qs, list) or not generated_qs:
            logger.warning(
                "Answer relevance: LLM returned no questions; falling back "
                "to embedding similarity."
            )
            if self._embed_model is None:
                return 0.0
            q_emb = self._embed_model.encode(
                [inp.query], convert_to_tensor=True
            )
            a_emb = self._embed_model.encode(
                [inp.answer], convert_to_tensor=True
            )
            return max(0.0, float(_st_cos_sim(q_emb, a_emb)[0][0]))

        if self._embed_model is None:
            # No embedding model; return fraction of generated questions that
            # share at least one word with the original query (rough proxy)
            q_words = set(inp.query.lower().split())
            score = statistics.mean(
                len(set(q.lower().split()) & q_words) / max(len(q_words), 1)
                for q in generated_qs
                if isinstance(q, str)
            )
            return min(1.0, score)

        q_emb = self._embed_model.encode([inp.query], convert_to_tensor=True)
        gen_embs = self._embed_model.encode(
            generated_qs, convert_to_tensor=True
        )
        sims = _st_cos_sim(q_emb, gen_embs)[0].cpu().numpy()
        return float(np.mean(sims))

    # ------------------------------------------------------------------
    # Context Precision
    # ------------------------------------------------------------------

    async def context_precision(
        self, inp: RAGASInput
    ) -> tuple[float, list[ContextChunkScore]]:
        """Compute LLM-judged context precision.

        Each chunk is classified as relevant/irrelevant by the LLM.
        The final score is the weighted average precision (AP-style).

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Tuple of (weighted precision, per-chunk scores).
        """
        if not inp.context_docs:
            return 0.0, []

        # If ground truth doc_ids are available, skip LLM calls
        if inp.ground_truth_doc_ids is not None:
            chunk_scores = []
            for rank, doc in enumerate(inp.context_docs, 1):
                relevant = (
                    doc.doc_id in inp.ground_truth_doc_ids
                    or (
                        doc.pmid is not None
                        and doc.pmid in inp.ground_truth_doc_ids
                    )
                )
                chunk_scores.append(
                    ContextChunkScore(
                        doc_id=doc.doc_id,
                        relevant=relevant,
                        relevance_score=1.0 if relevant else 0.0,
                        rank=rank,
                    )
                )
        else:
            # LLM judge each chunk in parallel (up to 5 for cost control)
            tasks = [
                self._judge_chunk_relevance(inp.query, doc, rank)
                for rank, doc in enumerate(inp.context_docs[:5], 1)
            ]
            chunk_scores_partial = await asyncio.gather(*tasks)
            chunk_scores = list(chunk_scores_partial)

            # Remaining chunks (rank > 5) judged as irrelevant for cost
            for rank, doc in enumerate(inp.context_docs[5:], 6):
                chunk_scores.append(
                    ContextChunkScore(
                        doc_id=doc.doc_id,
                        relevant=False,
                        relevance_score=0.0,
                        rank=rank,
                    )
                )

        # Weighted average precision
        running_hits = 0
        precision_sum = 0.0
        total_relevant = sum(c.relevant for c in chunk_scores)

        for cs in chunk_scores:
            if cs.relevant:
                running_hits += 1
                precision_sum += running_hits / cs.rank

        score = precision_sum / total_relevant if total_relevant else 0.0
        return score, chunk_scores

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMEvaluator._judge_chunk_relevance
    # Requirement  : `_judge_chunk_relevance` shall ask the LLM whether a single chunk is relevant to the query
    # Purpose      : Ask the LLM whether a single chunk is relevant to the query
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; doc: ContextDocument; rank: int
    # Outputs      : ContextChunkScore
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _judge_chunk_relevance(
        self, query: str, doc: ContextDocument, rank: int
    ) -> ContextChunkScore:
        """Ask the LLM whether a single chunk is relevant to the query."""
        user_msg = (
            f"# Question\n{query}\n\n"
            f"# Document Chunk [{doc.doc_id}]\n{doc.text[:600]}"
        )
        raw = await self._judge.call(_PRECISION_SYSTEM, user_msg)
        parsed = self._judge._parse_json(raw, fallback={})
        relevant = bool(parsed.get("relevant", False))
        return ContextChunkScore(
            doc_id=doc.doc_id,
            relevant=relevant,
            relevance_score=1.0 if relevant else 0.0,
            rank=rank,
        )

    # ------------------------------------------------------------------
    # Context Recall
    # ------------------------------------------------------------------

    async def context_recall(self, inp: RAGASInput) -> float:
        """Compute LLM-judged context recall.

        For each sentence in the ground-truth answer (or ground_truth_doc_ids),
        determine if it can be attributed to the retrieved context.

        Args:
            inp: RAGAS evaluation input.

        Returns:
            Score in [0, 1]; 0.0 if no ground truth is available.
        """
        if inp.ground_truth_doc_ids:
            # Fast path: doc_id set overlap
            retrieved = {d.doc_id for d in inp.context_docs}
            retrieved |= {d.pmid for d in inp.context_docs if d.pmid}
            hits = inp.ground_truth_doc_ids & retrieved
            return len(hits) / len(inp.ground_truth_doc_ids)

        if not inp.ground_truth_answer:
            return 0.0

        gt_sentences = _EmbeddingEvaluator._split_claims(inp.ground_truth_answer)
        if not gt_sentences:
            return 0.0

        context_block = "\n\n".join(
            f"[{doc.doc_id}]\n{doc.text[:600]}"
            for doc in inp.context_docs[:10]
        )

        tasks = [
            self._judge_statement_attributable(stmt, context_block)
            for stmt in gt_sentences[:10]  # cap to control cost
        ]
        results = await asyncio.gather(*tasks)
        attributable_count = sum(results)
        return attributable_count / len(gt_sentences[:10])

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics._LLMEvaluator._judge_statement_attributable
    # Requirement  : `_judge_statement_attributable` shall ask LLM if a ground-truth statement is attributable to context
    # Purpose      : Ask LLM if a ground-truth statement is attributable to context
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : statement: str; context_block: str
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _judge_statement_attributable(
        self, statement: str, context_block: str
    ) -> bool:
        """Ask LLM if a ground-truth statement is attributable to context."""
        user_msg = (
            f"# Ground-truth Statement\n{statement}\n\n"
            f"# Context Documents\n{context_block}"
        )
        raw = await self._judge.call(_RECALL_SYSTEM, user_msg)
        parsed = self._judge._parse_json(raw, fallback={})
        return bool(parsed.get("attributable", False))


# ---------------------------------------------------------------------------
# Human Eval scaffolding
# ---------------------------------------------------------------------------


@dataclass
class HumanEvalRecord:
    """A single record exported for human annotation.

    Attributes:
        record_id: Unique identifier for the annotation record.
        query: The original user question.
        answer: The generated answer.
        context_excerpts: Short excerpts from each retrieved chunk.
        auto_scores: RAGASScores computed by the automated pipeline.
        human_faithfulness: Annotator's faithfulness rating (0–1 or None).
        human_relevance: Annotator's relevance rating (0–1 or None).
        human_context_precision: Annotator's context precision rating.
        human_context_recall: Annotator's context recall rating.
        annotator_notes: Free-text notes from the annotator.
    """

    record_id: str
    query: str
    answer: str
    context_excerpts: List[Dict[str, str]]
    auto_scores: RAGASScores
    human_faithfulness: Optional[float] = None
    human_relevance: Optional[float] = None
    human_context_precision: Optional[float] = None
    human_context_recall: Optional[float] = None
    annotator_notes: str = ""

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.HumanEvalRecord.to_dict
    # Requirement  : `to_dict` shall execute as specified
    # Purpose      : To dict
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "query": self.query,
            "answer": self.answer,
            "context_excerpts": self.context_excerpts,
            "auto_scores": self.auto_scores.to_dict(),
            "human_annotations": {
                "faithfulness": self.human_faithfulness,
                "answer_relevance": self.human_relevance,
                "context_precision": self.human_context_precision,
                "context_recall": self.human_context_recall,
                "notes": self.annotator_notes,
            },
        }


# ---------------------------------------------------------------------------
# ID           : evaluation.ragas_metrics.export_for_human_eval
# Requirement  : `export_for_human_eval` shall build :class:`HumanEvalRecord` list ready for annotation tooling
# Purpose      : Build :class:`HumanEvalRecord` list ready for annotation tooling
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : inputs: List[RAGASInput]; scores: List[RAGASScores]; context_excerpt_chars: int (default=300)
# Outputs      : List[HumanEvalRecord]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def export_for_human_eval(
    inputs: List[RAGASInput],
    scores: List[RAGASScores],
    context_excerpt_chars: int = 300,
) -> List[HumanEvalRecord]:
    """Build :class:`HumanEvalRecord` list ready for annotation tooling.

    The exported records contain auto-computed scores alongside blank
    ``human_*`` fields.  Feed these into a spreadsheet, Label Studio, or
    any annotation interface; merge the filled-in scores back and compute
    inter-annotator agreement.

    Args:
        inputs: List of :class:`RAGASInput` instances (one per query).
        scores: Corresponding list of :class:`RAGASScores` (same length).
        context_excerpt_chars: Maximum characters per context excerpt.

    Returns:
        List of :class:`HumanEvalRecord`, one per input.

    Raises:
        ValueError: If ``inputs`` and ``scores`` have different lengths.
    """
    if len(inputs) != len(scores):
        raise ValueError(
            f"inputs ({len(inputs)}) and scores ({len(scores)}) must match."
        )

    records: list[HumanEvalRecord] = []
    for idx, (inp, score) in enumerate(zip(inputs, scores)):
        excerpts = [
            {
                "doc_id": doc.doc_id,
                "pmid": doc.pmid or "",
                "excerpt": doc.text[:context_excerpt_chars],
            }
            for doc in inp.context_docs
        ]
        records.append(
            HumanEvalRecord(
                record_id=f"eval-{idx:04d}",
                query=inp.query,
                answer=inp.answer,
                context_excerpts=excerpts,
                auto_scores=score,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class RAGASEvaluator:
    """Orchestrates all four RAGAS metrics in a single evaluate() call.

    Supports three modes:

    * ``EvaluationMode.EMBEDDING`` — fully offline, fast, uses
      sentence-transformer cosine similarity.
    * ``EvaluationMode.LLM`` — LLM-as-judge, more faithful to the original
      RAGAS paper; requires a configured LLM provider.
    * ``EvaluationMode.AUTO`` — attempts LLM; falls back to EMBEDDING if
      no LLM credentials are available.

    Args:
        mode: Evaluation mode (default: AUTO).
        llm_provider: LLM provider for LLM/AUTO mode.
        llm_model: LLM model name override.
        embed_model: Sentence-transformer model name.
        n_relevance_questions: Questions to generate per query in LLM mode.

    Example::

        evaluator = RAGASEvaluator(mode=EvaluationMode.EMBEDDING)

        score = await evaluator.evaluate(
            RAGASInput(
                query="CNN for seizure detection",
                answer="Convolutional networks achieve 95% [PMID:12345678]",
                context_docs=[ContextDocument("12345678", "... paper text ...")],
                ground_truth_doc_ids={"12345678"},
            )
        )
        print(score.ragas_score)
    """

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : mode: EvaluationMode (default=EvaluationMode.AUTO); llm_provider: LLMProvider (default=LLMProvider.OPENAI); llm_model: Optional[str] (default=None); embed_model: str (default=_DEFAULT_EMBED_MODEL); n_relevance_questions: int (default=3)
    # Outputs      : None
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        mode: EvaluationMode = EvaluationMode.AUTO,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        llm_model: Optional[str] = None,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        n_relevance_questions: int = 3,
    ) -> None:
        self._mode = mode
        self._embed_model_name = embed_model
        self._n = n_relevance_questions
        self._llm_provider = llm_provider
        self._llm_model = llm_model

        self._embed_eval: Optional[_EmbeddingEvaluator] = None
        self._llm_eval: Optional[_LLMEvaluator] = None

    # ------------------------------------------------------------------
    # Lazy init helpers (avoid loading heavy models until first use)
    # ------------------------------------------------------------------

    def _get_embed_eval(self) -> _EmbeddingEvaluator:
        if self._embed_eval is None:
            self._embed_eval = _EmbeddingEvaluator(self._embed_model_name)
        return self._embed_eval

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator._get_llm_eval
    # Requirement  : `_get_llm_eval` shall execute as specified
    # Purpose      :  get llm eval
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : _LLMEvaluator
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _get_llm_eval(self) -> _LLMEvaluator:
        if self._llm_eval is None:
            judge = _LLMJudge(
                provider=self._llm_provider,
                model=self._llm_model,
            )
            self._llm_eval = _LLMEvaluator(
                judge=judge,
                n_relevance_questions=self._n,
                embed_model_name=self._embed_model_name,
            )
        return self._llm_eval

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator._resolve_mode
    # Requirement  : `_resolve_mode` shall resolve AUTO to a concrete mode based on available credentials
    # Purpose      : Resolve AUTO to a concrete mode based on available credentials
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : EvaluationMode
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _resolve_mode(self) -> EvaluationMode:
        """Resolve AUTO to a concrete mode based on available credentials."""
        if self._mode == EvaluationMode.AUTO:
            has_llm = (
                (self._llm_provider == LLMProvider.OPENAI and bool(os.getenv("OPENAI_API_KEY")))
                or (self._llm_provider == LLMProvider.ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY")))
                or (self._llm_provider == LLMProvider.OLLAMA)
            )
            return EvaluationMode.LLM if has_llm else EvaluationMode.EMBEDDING
        return self._mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, inp: RAGASInput) -> RAGASScores:
        """Compute all four RAGAS metrics for a single query–answer pair.

        Args:
            inp: :class:`RAGASInput` with query, answer, and context docs.

        Returns:
            :class:`RAGASScores` with all metrics and a composite score.
        """
        mode = self._resolve_mode()
        warnings: list[str] = []

        if mode == EvaluationMode.LLM:
            try:
                scores = await self._evaluate_llm(inp)
                scores.warnings.extend(warnings)
                return scores
            except Exception as exc:
                logger.warning(
                    "LLM evaluation failed (%s); falling back to embedding "
                    "mode.",
                    exc,
                )
                warnings.append(
                    f"LLM evaluation failed ({exc}); fell back to embedding mode."
                )
                mode = EvaluationMode.EMBEDDING

        scores = await self._evaluate_embedding(inp)
        scores.mode = mode
        scores.warnings.extend(warnings)
        return scores

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator.evaluate_batch
    # Requirement  : `evaluate_batch` shall evaluate a list of query–answer pairs concurrently
    # Purpose      : Evaluate a list of query–answer pairs concurrently
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : inputs: List[RAGASInput]
    # Outputs      : List[RAGASScores]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def evaluate_batch(
        self, inputs: List[RAGASInput]
    ) -> List[RAGASScores]:
        """Evaluate a list of query–answer pairs concurrently.

        Args:
            inputs: List of :class:`RAGASInput` instances.

        Returns:
            List of :class:`RAGASScores` in the same order.
        """
        tasks = [self.evaluate(inp) for inp in inputs]
        return list(await asyncio.gather(*tasks))

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator.summary
    # Requirement  : `summary` shall compute mean scores over a collection of :class:`RAGASScores`
    # Purpose      : Compute mean scores over a collection of :class:`RAGASScores`
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : scores: Sequence[RAGASScores]
    # Outputs      : Dict[str, float]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def summary(self, scores: Sequence[RAGASScores]) -> Dict[str, float]:
        """Compute mean scores over a collection of :class:`RAGASScores`.

        Args:
            scores: Any sequence of :class:`RAGASScores`.

        Returns:
            Dictionary with mean values for each metric.
        """
        if not scores:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "ragas_score": 0.0,
            }
        return {
            "faithfulness": statistics.mean(s.faithfulness for s in scores),
            "answer_relevance": statistics.mean(
                s.answer_relevance for s in scores
            ),
            "context_precision": statistics.mean(
                s.context_precision for s in scores
            ),
            "context_recall": statistics.mean(s.context_recall for s in scores),
            "ragas_score": statistics.mean(s.ragas_score for s in scores),
        }

    # ------------------------------------------------------------------
    # Internal evaluation paths
    # ------------------------------------------------------------------

    async def _evaluate_embedding(self, inp: RAGASInput) -> RAGASScores:
        """Run all four metrics in embedding mode (synchronous, run in executor)."""
        loop = asyncio.get_event_loop()
        ev = self._get_embed_eval()

        # Run CPU-bound encoding in thread pool to avoid blocking event loop
        faith_score, faith_details = await loop.run_in_executor(
            None, ev.faithfulness, inp
        )
        rel_score = await loop.run_in_executor(None, ev.answer_relevance, inp)
        prec_score, chunk_scores = await loop.run_in_executor(
            None, ev.context_precision, inp
        )
        recall_score = await loop.run_in_executor(None, ev.context_recall, inp)

        ragas = _harmonic_mean(
            [faith_score, rel_score, prec_score, recall_score]
        )

        return RAGASScores(
            faithfulness=faith_score,
            answer_relevance=rel_score,
            context_precision=prec_score,
            context_recall=recall_score,
            ragas_score=ragas,
            mode=EvaluationMode.EMBEDDING,
            faithfulness_details=faith_details,
            chunk_scores=chunk_scores,
        )

    # ---------------------------------------------------------------------------
    # ID           : evaluation.ragas_metrics.RAGASEvaluator._evaluate_llm
    # Requirement  : `_evaluate_llm` shall run all four metrics in LLM-as-judge mode (concurrent async calls)
    # Purpose      : Run all four metrics in LLM-as-judge mode (concurrent async calls)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : inp: RAGASInput
    # Outputs      : RAGASScores
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _evaluate_llm(self, inp: RAGASInput) -> RAGASScores:
        """Run all four metrics in LLM-as-judge mode (concurrent async calls)."""
        ev = self._get_llm_eval()

        (faith_score, faith_details), rel_score, (prec_score, chunk_scores), recall_score = (
            await asyncio.gather(
                ev.faithfulness(inp),
                ev.answer_relevance(inp),
                ev.context_precision(inp),
                ev.context_recall(inp),
            )
        )

        ragas = _harmonic_mean(
            [faith_score, rel_score, prec_score, recall_score]
        )

        return RAGASScores(
            faithfulness=faith_score,
            answer_relevance=rel_score,
            context_precision=prec_score,
            context_recall=recall_score,
            ragas_score=ragas,
            mode=EvaluationMode.LLM,
            faithfulness_details=faith_details,
            chunk_scores=chunk_scores,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _harmonic_mean(values: list[float]) -> float:
    """Compute the harmonic mean of non-negative values.

    Any zero value makes the harmonic mean 0 (consistent with RAGAS paper).

    Args:
        values: List of floats in [0, 1].

    Returns:
        Harmonic mean in [0, 1].
    """
    if not values or any(v == 0.0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)
