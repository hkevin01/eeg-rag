"""
# =============================================================================
# ID:             MOD-ARAG-001
# Requirement:    REQ-ARAG-001 — Conditional retrieval based on query analysis;
#                 REQ-ARAG-002 — Iterative query reformulation on insufficient
#                                results;
#                 REQ-ARAG-003 — Multi-source retrieval with per-source strategies;
#                 REQ-ARAG-004 — Reflection/verification before answer generation.
# Purpose:        Implement an Agentic RAG loop that moves beyond single-shot
#                 retrieval.  The orchestrator (1) decides WHETHER to retrieve,
#                 (2) can REFORMULATE the query when first results are
#                 insufficient, (3) queries MULTIPLE sources in successive
#                 rounds, and (4) VERIFIES retrieved evidence before generating
#                 the final answer.
# Rationale:      Basic RAG blindly retrieves on every query.  Agentic RAG adds
#                 deliberate planning: simple definitional questions are answered
#                 directly; complex or ambiguous queries trigger iterative loops
#                 that progressively narrow or expand search until sufficient
#                 evidence is gathered.  This mirrors how a skilled researcher
#                 would conduct a literature review.
# Inputs:         query (str) — natural language EEG research question;
#                 retriever (HybridRetriever) — initialised hybrid index;
#                 generator (ResponseGenerator) — LLM response generator;
#                 verifier (CitationVerifier | None) — optional citation check.
# Outputs:        AgenticRAGResult with answer text, cited sources, per-step
#                 audit trail, and quality metrics.
# Failure Modes:  All iterations produce empty results → answer generated with
#                 explicit "no evidence found" notice;
#                 LLM generation fails → re-raised after logging.
# Constraints:    max_iterations default 3; min_docs_threshold default 3;
#                 min_relevance_threshold default 0.25.
# Verification:   tests/test_agentic_rag.py
# References:     Yao et al. 2022 "ReAct: Synergizing Reasoning and Acting";
#                 Shinn et al. 2023 "Reflexion";
#                 Lewis et al. 2020 "Retrieval-Augmented Generation".
# =============================================================================
Agentic RAG Orchestrator for EEG-RAG.

Implements a four-capability loop on top of the existing hybrid retriever and
response generator:

1. **Decision** – Classifies whether retrieval is needed at all.
2. **Retrieval**  – Executes hybrid (BM25 + dense) search across sources.
3. **Reformulation** – Rewrites or decomposes the query when results are
   insufficient, then retrieves again (up to ``max_iterations``).
4. **Reflection** – Verifies citation integrity before final generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
import statistics
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from eeg_rag.ensemble.context_aggregator import ContextAggregator
from eeg_rag.retrieval.hybrid_retriever import HybridResult, HybridRetriever
from eeg_rag.retrieval.query_expander import EEGQueryExpander
from eeg_rag.generation.response_generator import Document, ResponseGenerator
from eeg_rag.services.history_manager import HistoryManager
from eeg_rag.verification.citation_verifier import CitationVerifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# EEG terms whose presence strongly signals a research query needing retrieval
_EEG_RESEARCH_SIGNALS: frozenset[str] = frozenset({
    "seizure", "epilep", "alpha", "beta", "delta", "theta", "gamma",
    "erp", "p300", "n400", "bci", "motor imagery", "sleep stage",
    "artifact", "electrode", "montage", "ica", "ssvep", "eog",
    "mismatch negativity", "mmn", "cognitive load", "neurofeedback",
    "deep learning", "cnn", "lstm", "transformer", "classification",
    "detection", "prediction", "biomarker", "connectivity",
    "synchrony", "coherence", "power spectral", "time-frequency",
    "wavelet", "hilbert", "source localisation", "beamforming",
    "dipole", "ica", "fastica", "eeglab", "brainstorm", "mne",
})

# Patterns that indicate direct-answer queries (definitions / maths)
_DIRECT_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*what\s+(is|are)\s+(the\s+)?definition", re.IGNORECASE),
    re.compile(r"^\s*define\s+\w", re.IGNORECASE),
    re.compile(r"^\s*what\s+(does|is)\s+\w+\s+stand\s+for", re.IGNORECASE),
    re.compile(r"^\s*how\s+many\s+(hz|hertz|channels?|electrodes?)", re.IGNORECASE),
]

# Minimum quality thresholds
_MIN_DOCS_DEFAULT = 3
_MIN_RELEVANCE_DEFAULT = 0.25
_MAX_ITERATIONS_DEFAULT = 3


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RetrievalNeed(Enum):
    """Decision on whether and how to retrieve evidence for a query.

    Values:
        SKIP: Query can be answered directly from parametric knowledge
              (simple definitions, EEG band frequencies, etc.).
        RETRIEVE: Standard hybrid retrieval required.
        VERIFY_CLAIM: Query asserts a specific finding (with PMID or author);
                      retrieve to verify/contradict the claim.
        DECOMPOSE: Query contains multiple independent sub-questions that
                   should be retrieved for separately before synthesis.
    """

    SKIP = "skip"
    RETRIEVE = "retrieve"
    VERIFY_CLAIM = "verify_claim"
    DECOMPOSE = "decompose"


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.ReformulationStrategy
# Requirement  : `ReformulationStrategy` class shall be instantiable and expose the documented interface
# Purpose      : Strategy applied when reformulating an insufficient query
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
# Verification : Instantiate ReformulationStrategy with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ReformulationStrategy(Enum):
    """Strategy applied when reformulating an insufficient query.

    Values:
        EXPAND: Append EEG-domain synonyms and related terms.
        PIVOT_DENSE: Switch emphasis to semantic (dense) retrieval by
                     removing overly specific jargon.
        PIVOT_BM25: Switch to exact-keyword mode by removing vague phrases.
        RELAX: Broaden scope by removing restrictive clauses.
        NARROW: Add specificity constraints (e.g. year, modality).
        DECOMPOSE: Break into independent sub-queries.
    """

    EXPAND = "expand"
    PIVOT_DENSE = "pivot_dense"
    PIVOT_BM25 = "pivot_bm25"
    RELAX = "relax"
    NARROW = "narrow"
    DECOMPOSE = "decompose"


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.SufficiencyStatus
# Requirement  : `SufficiencyStatus` class shall be instantiable and expose the documented interface
# Purpose      : Outcome of the sufficiency check on retrieved documents
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
# Verification : Instantiate SufficiencyStatus with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class SufficiencyStatus(Enum):
    """Outcome of the sufficiency check on retrieved documents.

    Values:
        SUFFICIENT: Enough relevant docs found; proceed to generation.
        LOW_COUNT: Too few documents returned.
        LOW_RELEVANCE: Docs returned but average relevance below threshold.
        LOW_COVERAGE: Docs found but do not cover all query aspects.
        LOW_DIVERSITY: Docs are relevant but too redundant to cover the space.
        EMPTY: Retriever returned no results at all.
    """

    SUFFICIENT = "sufficient"
    LOW_COUNT = "low_count"
    LOW_RELEVANCE = "low_relevance"
    LOW_COVERAGE = "low_coverage"
    LOW_DIVERSITY = "low_diversity"
    EMPTY = "empty"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievalDecision:
    """Output of the retrieval decision step.

    Attributes:
        need: Whether/how to retrieve.
        rationale: Human-readable explanation of the decision.
        detected_entities: EEG-domain entities extracted from the query
            (band names, components, methods, conditions).
        sub_queries: Non-empty only when need == DECOMPOSE; contains
            individual sub-questions to retrieve for independently.
        claimed_pmids: PMIDs mentioned in the query (for VERIFY_CLAIM).
    """

    need: RetrievalNeed
    rationale: str
    detected_entities: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    claimed_pmids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.SufficiencyCheck
# Requirement  : `SufficiencyCheck` class shall be instantiable and expose the documented interface
# Purpose      : Outcome of evaluating whether retrieved docs satisfy the query
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
# Verification : Instantiate SufficiencyCheck with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SufficiencyCheck:
    """Outcome of evaluating whether retrieved docs satisfy the query.

    Attributes:
        status: Overall verdict.
        doc_count: Number of documents retrieved.
        relevance_score: Mean RRF score of top-k results (0–1 range
            after normalisation; higher = more relevant).
        coverage_score: Fraction of detected query aspects covered by
            at least one retrieved document (0.0–1.0).
        missing_aspects: List of query aspects not covered.
        explanation: Human-readable summary of the verdict.
    """

    status: SufficiencyStatus
    doc_count: int
    relevance_score: float
    coverage_score: float
    redundancy_score: float = 0.0
    diversity_score: float = 1.0
    query_entity_coverage_score: float = 1.0
    aggregation_diagnostics: Dict[str, Any] = field(default_factory=dict)
    missing_aspects: List[str] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.ReformulationResult
# Requirement  : `ReformulationResult` class shall be instantiable and expose the documented interface
# Purpose      : A reformulated query produced when results are insufficient
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
# Verification : Instantiate ReformulationResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ReformulationResult:
    """A reformulated query produced when results are insufficient.

    Attributes:
        new_query: The reformulated query string to use for the next
            retrieval round.
        strategy: Which reformulation strategy was applied.
        rationale: Why this strategy was chosen.
        bm25_weight_hint: Suggested BM25 weight override for next round
            (None = keep current).
        dense_weight_hint: Suggested dense weight override for next round
            (None = keep current).
    """

    new_query: str
    strategy: ReformulationStrategy
    rationale: str
    bm25_weight_hint: Optional[float] = None
    dense_weight_hint: Optional[float] = None


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.AgenticStep
# Requirement  : `AgenticStep` class shall be instantiable and expose the documented interface
# Purpose      : Audit record for a single iteration of the agentic loop
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
# Verification : Instantiate AgenticStep with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class AgenticStep:
    """Audit record for a single iteration of the agentic loop.

    Attributes:
        iteration: 1-based iteration number.
        query_used: The query (original or reformulated) that was issued.
        strategy: Reformulation strategy applied before this round
            (None on the first iteration).
        docs_retrieved: Number of documents returned by the retriever.
        top_doc_ids: doc_id of the highest-scoring results (up to 5).
        sufficiency: Sufficiency verdict for this round.
        reformulation: Reformulation decision for the *next* round
            (None if this was the final round).
        elapsed_ms: Wall-clock time for this iteration in milliseconds.
    """

    iteration: int
    query_used: str
    strategy: Optional[ReformulationStrategy]
    docs_retrieved: int
    top_doc_ids: List[str]
    sufficiency: SufficiencyCheck
    reformulation: Optional[ReformulationResult]
    elapsed_ms: float
    aggregation_diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ID           : rag.agentic_rag.AgenticRAGResult
# Requirement  : `AgenticRAGResult` class shall be instantiable and expose the documented interface
# Purpose      : Final result from the Agentic RAG orchestrator
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
# Verification : Instantiate AgenticRAGResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class AgenticRAGResult:
    """Final result from the Agentic RAG orchestrator.

    Attributes:
        query: Original user query.
        answer: Generated answer text (may be streamed separately).
        sources: Deduplicated list of source documents used in the answer.
        steps: Per-iteration audit trail.
        decision: The initial retrieval-need decision.
        iterations_used: How many retrieval iterations were executed.
        total_elapsed_ms: End-to-end wall-clock time in milliseconds.
        citations_verified: Whether CitationVerifier was run.
        verification_warnings: Warnings from citation verification.
        skipped_retrieval: True when the query was answered directly.
        error: Non-None if the orchestrator encountered a fatal error.
    """

    query: str
    answer: str
    sources: List[HybridResult]
    steps: List[AgenticStep]
    decision: RetrievalDecision
    iterations_used: int
    total_elapsed_ms: float
    citations_verified: bool = False
    verification_warnings: List[str] = field(default_factory=list)
    skipped_retrieval: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# RetrievalDecisionMaker
# ---------------------------------------------------------------------------


class RetrievalDecisionMaker:
    """Decides whether and how to retrieve evidence for a given query.

    The decision is rule-based using three tiers:

    1. **Direct-answer check** – Regex patterns that match definitional or
       arithmetic questions about EEG that can be answered from parametric
       knowledge (e.g. "What is the frequency range of the alpha band?").
    2. **Claim verification** – Queries that contain a PMID or assert a
       specific finding ("Smith et al. showed …") are routed to
       ``VERIFY_CLAIM`` so the orchestrator can cross-check the assertion.
    3. **Decomposition** – Queries with multiple question marks or
       conjunctive sub-questions ("What are ... and how does ...") are
       flagged for ``DECOMPOSE`` so each part is retrieved independently.
    4. **Default** – All remaining queries receive ``RETRIEVE``.

    Args:
        min_query_length: Queries shorter than this character count are
            assumed to be lookup/definition queries and skipped.
    """

    _PMID_RE = re.compile(r'\bPMID[:\s]*(\d{7,8})\b', re.IGNORECASE)
    _MULTI_Q_RE = re.compile(r'\?.*\?', re.DOTALL)
    _CONJUNCTION_RE = re.compile(
        r'\b(and|also|additionally|furthermore|moreover|besides)\b.*\b'
        r'(what|how|why|when|which|who|where)\b',
        re.IGNORECASE,
    )

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.RetrievalDecisionMaker.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : min_query_length: int (default=15)
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
    def __init__(self, min_query_length: int = 15) -> None:
        self._min_length = min_query_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, query: str) -> RetrievalDecision:
        """Analyse ``query`` and return a :class:`RetrievalDecision`.

        Args:
            query: Raw user query string.

        Returns:
            RetrievalDecision describing the recommended retrieval mode.
        """
        query = query.strip()
        entities = self._extract_eeg_entities(query)
        pmids = self._PMID_RE.findall(query)

        # Tier 1: short or definitional – skip retrieval
        if len(query) < self._min_length or self._is_direct_answer(query):
            return RetrievalDecision(
                need=RetrievalNeed.SKIP,
                rationale=(
                    "Query is definitional or too short to require "
                    "literature retrieval."
                ),
                detected_entities=entities,
            )

        # Tier 2: specific PMID or claim verification
        if pmids:
            return RetrievalDecision(
                need=RetrievalNeed.VERIFY_CLAIM,
                rationale=(
                    f"Query references specific PMID(s) {pmids}; "
                    "retrieval will verify the claimed findings."
                ),
                detected_entities=entities,
                claimed_pmids=pmids,
            )

        # Tier 3: multi-part decomposition
        if self._needs_decomposition(query):
            sub_queries = self._decompose(query)
            return RetrievalDecision(
                need=RetrievalNeed.DECOMPOSE,
                rationale=(
                    "Query contains multiple independent sub-questions; "
                    "each will be retrieved for separately."
                ),
                detected_entities=entities,
                sub_queries=sub_queries,
            )

        # Tier 4: standard retrieval
        return RetrievalDecision(
            need=RetrievalNeed.RETRIEVE,
            rationale="Query requires literature retrieval.",
            detected_entities=entities,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_direct_answer(self, query: str) -> bool:
        """Return True if the query matches a direct-answer pattern."""
        return any(p.search(query) for p in _DIRECT_ANSWER_PATTERNS)

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.RetrievalDecisionMaker._needs_decomposition
    # Requirement  : `_needs_decomposition` shall return True if the query contains multiple independent questions
    # Purpose      : Return True if the query contains multiple independent questions
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : bool
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
    def _needs_decomposition(self, query: str) -> bool:
        """Return True if the query contains multiple independent questions."""
        return bool(
            self._MULTI_Q_RE.search(query)
            or self._CONJUNCTION_RE.search(query)
        )

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.RetrievalDecisionMaker._decompose
    # Requirement  : `_decompose` shall split a multi-part query into independent sub-queries
    # Purpose      : Split a multi-part query into independent sub-queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : List[str]
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
    def _decompose(self, query: str) -> List[str]:
        """Split a multi-part query into independent sub-queries.

        Uses sentence-boundary splitting followed by filtering out
        very short fragments.

        Args:
            query: Multi-part query string.

        Returns:
            List of sub-query strings (at least 2 elements).
        """
        # Split on question marks, "and", common conjunctions
        parts = re.split(r'\?|\band\b(?=\s+\b(?:what|how|why|when|which)\b)',
                         query, flags=re.IGNORECASE)
        sub_queries = [p.strip().rstrip('.,;') for p in parts if len(p.strip()) > 15]
        # Fallback: return original as single item
        return sub_queries if len(sub_queries) >= 2 else [query]

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.RetrievalDecisionMaker._extract_eeg_entities
    # Requirement  : `_extract_eeg_entities` shall extract EEG-domain entity mentions from the query
    # Purpose      : Extract EEG-domain entity mentions from the query
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : List[str]
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
    def _extract_eeg_entities(query: str) -> List[str]:
        """Extract EEG-domain entity mentions from the query.

        Args:
            query: Raw query string.

        Returns:
            Sorted list of matched EEG domain terms (lowercased).
        """
        q_lower = query.lower()
        return sorted({
            term for term in _EEG_RESEARCH_SIGNALS
            if term in q_lower
        })


# ---------------------------------------------------------------------------
# SufficiencyEvaluator
# ---------------------------------------------------------------------------


class SufficiencyEvaluator:
    """Evaluates whether a set of retrieved documents adequately answers a query.

    Checks four orthogonal dimensions:

    * **Count** – Are there at least ``min_docs`` results?
    * **Relevance** – Is the mean RRF score ≥ ``min_relevance``?
    * **Coverage** – Do the results collectively mention all EEG entities
      detected in the query?
    * **Completeness** – (VERIFY_CLAIM mode) Is the claimed PMID present?

    Args:
        min_docs: Minimum number of retrieved documents.
        min_relevance: Minimum mean RRF score (0–1 normalised).
        coverage_threshold: Fraction of query entities that must appear in
            retrieved text for coverage to pass (default 0.6).
    """

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.SufficiencyEvaluator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : min_docs: int (default=_MIN_DOCS_DEFAULT); min_relevance: float (default=_MIN_RELEVANCE_DEFAULT); coverage_threshold: float (default=0.6)
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
        min_docs: int = _MIN_DOCS_DEFAULT,
        min_relevance: float = _MIN_RELEVANCE_DEFAULT,
        coverage_threshold: float = 0.6,
    ) -> None:
        self._min_docs = min_docs
        self._min_relevance = min_relevance
        self._coverage_threshold = coverage_threshold

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.SufficiencyEvaluator.evaluate
    # Requirement  : `evaluate` shall assess whether ``results`` sufficiently cover ``query``
    # Purpose      : Assess whether ``results`` sufficiently cover ``query``
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; results: List[HybridResult]; decision: RetrievalDecision
    # Outputs      : SufficiencyCheck
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
    def evaluate(
        self,
        query: str,
        results: List[HybridResult],
        decision: RetrievalDecision,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> SufficiencyCheck:
        """Assess whether ``results`` sufficiently cover ``query``.

        Args:
            query: The query that was issued for this retrieval round.
            results: Documents returned by the retriever.
            decision: The initial retrieval decision (for entity list and
                claimed PMIDs).

        Returns:
            SufficiencyCheck with a verdict and supporting metrics.
        """
        doc_count = len(results)

        # --- Count check ------------------------------------------------
        if doc_count == 0:
            return SufficiencyCheck(
                status=SufficiencyStatus.EMPTY,
                doc_count=0,
                relevance_score=0.0,
                coverage_score=0.0,
                aggregation_diagnostics=diagnostics or {},
                explanation="Retriever returned no results.",
            )

        if doc_count < self._min_docs:
            return SufficiencyCheck(
                status=SufficiencyStatus.LOW_COUNT,
                doc_count=doc_count,
                relevance_score=self._mean_score(results),
                coverage_score=0.0,
                aggregation_diagnostics=diagnostics or {},
                explanation=(
                    f"Only {doc_count} document(s) retrieved "
                    f"(minimum required: {self._min_docs})."
                ),
            )

        # --- Relevance check -------------------------------------------
        mean_rel = self._mean_score(results)
        if mean_rel < self._min_relevance:
            return SufficiencyCheck(
                status=SufficiencyStatus.LOW_RELEVANCE,
                doc_count=doc_count,
                relevance_score=mean_rel,
                coverage_score=0.0,
                aggregation_diagnostics=diagnostics or {},
                explanation=(
                    f"Mean relevance {mean_rel:.3f} below threshold "
                    f"{self._min_relevance}."
                ),
            )

        # --- Coverage check --------------------------------------------
        coverage_score, missing = self._compute_coverage(
            results, decision.detected_entities
        )
        redundancy_score = 0.0
        diversity_score = 1.0
        query_entity_coverage_score = coverage_score
        query_concept_coverage_score = coverage_score
        if diagnostics:
            redundancy_score = float(diagnostics.get("redundancy_score", 0.0))
            diversity_score = float(diagnostics.get("diversity_score", 1.0))
            query_entity_coverage_score = float(
                diagnostics.get("query_entity_coverage_score", coverage_score)
            )
            query_concept_coverage_score = float(
                diagnostics.get(
                    "query_concept_coverage_score",
                    query_entity_coverage_score,
                )
            )
            missing = diagnostics.get(
                "missing_query_concept_groups",
                diagnostics.get("missing_query_entities", missing),
            )
            coverage_score = max(
                coverage_score,
                query_entity_coverage_score,
                query_concept_coverage_score,
            )
        if (
            decision.detected_entities
            and coverage_score < self._coverage_threshold
        ):
            return SufficiencyCheck(
                status=SufficiencyStatus.LOW_COVERAGE,
                doc_count=doc_count,
                relevance_score=mean_rel,
                coverage_score=coverage_score,
                redundancy_score=redundancy_score,
                diversity_score=diversity_score,
                query_entity_coverage_score=query_entity_coverage_score,
                aggregation_diagnostics=diagnostics or {},
                missing_aspects=missing,
                explanation=(
                    f"Coverage {coverage_score:.2f} below threshold "
                    f"{self._coverage_threshold}. Missing: {missing}."
                ),
            )

        if redundancy_score >= 0.90 and diversity_score <= 0.10:
            return SufficiencyCheck(
                status=SufficiencyStatus.LOW_DIVERSITY,
                doc_count=doc_count,
                relevance_score=mean_rel,
                coverage_score=coverage_score,
                redundancy_score=redundancy_score,
                diversity_score=diversity_score,
                query_entity_coverage_score=query_entity_coverage_score,
                aggregation_diagnostics=diagnostics or {},
                missing_aspects=missing,
                explanation=(
                    f"Results are relevant but redundant "
                    f"(redundancy={redundancy_score:.2f}, diversity={diversity_score:.2f})."
                ),
            )

        # --- All checks passed ----------------------------------------
        return SufficiencyCheck(
            status=SufficiencyStatus.SUFFICIENT,
            doc_count=doc_count,
            relevance_score=mean_rel,
            coverage_score=coverage_score,
            redundancy_score=redundancy_score,
            diversity_score=diversity_score,
            query_entity_coverage_score=query_entity_coverage_score,
            aggregation_diagnostics=diagnostics or {},
            explanation=(
                f"{doc_count} documents retrieved with mean relevance "
                f"{mean_rel:.3f}, coverage {coverage_score:.2f}, "
                f"diversity {diversity_score:.2f}."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_score(results: List[HybridResult]) -> float:
        """Compute mean RRF score, normalised to 0-1 range.

        The theoretical maximum for a document that ranks first in BOTH the
        BM25 and the dense ranker is 2 * 1/(k+1) where k=60, giving
        2/61 ~= 0.0328.  We use that as the normalisation denominator so
        that a perfectly ranked document across both sources yields 1.0.
        """
        # ---------------------------------------------------------------------------
        # ID:          rag.agentic_rag.SufficiencyEvaluator._mean_score
        # Requirement: Normalised score must reflect dual-ranker maximum correctly.
        # Rationale:   Previous constant 0.02 was below the dual-ranker max (0.0328),
        #              causing high-quality results to be clipped and distorting the
        #              sufficiency verdict.
        # ---------------------------------------------------------------------------
        if not results:
            return 0.0
        raw = sum(r.rrf_score for r in results) / len(results)
        # k=60 for both BM25 and dense; dual-ranker max = 2/(60+1)
        _RRF_DUAL_MAX = 2.0 / 61.0  # ~0.0328
        return min(raw / _RRF_DUAL_MAX, 1.0)

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.SufficiencyEvaluator._compute_coverage
    # Requirement  : `_compute_coverage` shall measure what fraction of query entities appear in retrieved text
    # Purpose      : Measure what fraction of query entities appear in retrieved text
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : results: List[HybridResult]; entities: List[str]
    # Outputs      : Tuple[float, List[str]]
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
    def _compute_coverage(
        results: List[HybridResult],
        entities: List[str],
    ) -> Tuple[float, List[str]]:
        """Measure what fraction of query entities appear in retrieved text.

        Args:
            results: Retrieved documents.
            entities: Entity terms to check.

        Returns:
            Tuple of (coverage_fraction, list_of_missing_entities).
        """
        if not entities:
            return 1.0, []

        combined_text = " ".join(r.text.lower() for r in results)
        missing = [e for e in entities if e not in combined_text]
        coverage = 1.0 - len(missing) / len(entities)
        return coverage, missing


# ---------------------------------------------------------------------------
# QueryReformulator
# ---------------------------------------------------------------------------


class QueryReformulator:
    """Generates a reformulated query when retrieval results are insufficient.

    Selects the reformulation strategy based on the insufficiency type and
    iteration history to avoid repeating the same strategy.

    Args:
        expander: EEGQueryExpander instance (or None to disable expansion).
    """

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.QueryReformulator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : expander: Optional[EEGQueryExpander] (default=None)
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
    def __init__(self, expander: Optional[EEGQueryExpander] = None) -> None:
        self._expander = expander or EEGQueryExpander()

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.QueryReformulator.reformulate
    # Requirement  : `reformulate` shall generate a new query based on the insufficiency signal
    # Purpose      : Generate a new query based on the insufficiency signal
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : original_query: str; current_query: str; check: SufficiencyCheck; iteration: int; prior_strategies: List[ReformulationStrategy]
    # Outputs      : ReformulationResult
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
    def reformulate(
        self,
        original_query: str,
        current_query: str,
        check: SufficiencyCheck,
        iteration: int,
        prior_strategies: List[ReformulationStrategy],
    ) -> ReformulationResult:
        """Generate a new query based on the insufficiency signal.

        Strategy selection priority (each strategy used at most once):

        1. EMPTY / LOW_COUNT → EXPAND (add synonyms)
        2. LOW_RELEVANCE → PIVOT_DENSE (rely more on semantic similarity)
        3. LOW_COVERAGE → RELAX (broaden by removing restrictive terms)
        4. 2nd iteration fallback → NARROW (add specificity)
        5. 3rd iteration fallback → DECOMPOSE (last resort)

        Args:
            original_query: The very first query the user provided.
            current_query: The query used in the most recent retrieval round.
            check: Sufficiency check result from the last round.
            iteration: Current iteration number (1-based).
            prior_strategies: Strategies already applied in previous rounds.

        Returns:
            ReformulationResult with the new query and strategy metadata.
        """
        available = [s for s in ReformulationStrategy
                     if s not in prior_strategies]

        strategy = self._pick_strategy(check, available)

        if strategy == ReformulationStrategy.EXPAND:
            new_query = self._apply_expand(current_query)
            rationale = (
                "Low/no results; expanding with EEG-domain synonyms "
                "to improve recall."
            )
            return ReformulationResult(
                new_query=new_query,
                strategy=strategy,
                rationale=rationale,
                bm25_weight_hint=0.6,
                dense_weight_hint=0.4,
            )

        if strategy == ReformulationStrategy.PIVOT_DENSE:
            new_query = self._apply_pivot_dense(current_query)
            rationale = (
                "Low relevance with keyword search; shifting weight "
                "toward semantic (dense) retrieval."
            )
            return ReformulationResult(
                new_query=new_query,
                strategy=strategy,
                rationale=rationale,
                bm25_weight_hint=0.3,
                dense_weight_hint=0.7,
            )

        if strategy == ReformulationStrategy.RELAX:
            new_query = self._apply_relax(current_query, original_query)
            if check.status == SufficiencyStatus.LOW_DIVERSITY:
                rationale = (
                    "Results are relevant but overly redundant; relaxing "
                    "constraints to broaden evidence coverage."
                )
            else:
                rationale = (
                    "Missing aspects in coverage; relaxing constraints "
                    "to broaden search scope."
                )
            return ReformulationResult(
                new_query=new_query,
                strategy=strategy,
                rationale=rationale,
            )

        if strategy == ReformulationStrategy.NARROW:
            new_query = self._apply_narrow(current_query, check.missing_aspects)
            rationale = (
                "Refining query with missing aspect terms to improve "
                "topical precision."
            )
            return ReformulationResult(
                new_query=new_query,
                strategy=strategy,
                rationale=rationale,
                bm25_weight_hint=0.5,
                dense_weight_hint=0.5,
            )

        # Final fallback: DECOMPOSE (use original to avoid drift)
        new_query = original_query
        return ReformulationResult(
            new_query=new_query,
            strategy=ReformulationStrategy.DECOMPOSE,
            rationale=(
                "Multiple strategies exhausted; reverting to original "
                "query for a final broad retrieval pass."
            ),
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _apply_expand(self, query: str) -> str:
        """Add EEG-domain synonyms to ``query``."""
        expanded = self._expander.expand(query, max_expansions=3)
        # Ensure we don't repeat the exact same string
        return expanded if expanded.lower() != query.lower() else f"{query} EEG"

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.QueryReformulator._apply_pivot_dense
    # Requirement  : `_apply_pivot_dense` shall remove overly specific jargon that hinders dense retrieval
    # Purpose      : Remove overly specific jargon that hinders dense retrieval
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
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
    def _apply_pivot_dense(query: str) -> str:
        """Remove overly specific jargon that hinders dense retrieval."""
        # Strip boolean-operator style terms that harm dense embeddings
        cleaned = re.sub(r'\b(AND|OR|NOT)\b', ' ', query)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        return cleaned or query

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.QueryReformulator._apply_relax
    # Requirement  : `_apply_relax` shall broaden scope by falling back toward the original query terms
    # Purpose      : Broaden scope by falling back toward the original query terms
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : current_query: str; original_query: str
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
    def _apply_relax(current_query: str, original_query: str) -> str:
        """Broaden scope by falling back toward the original query terms."""
        # Remove quoted phrases and parenthetical constraints
        relaxed = re.sub(r'"[^"]+"', '', current_query)
        relaxed = re.sub(r'\([^)]+\)', '', relaxed)
        relaxed = re.sub(r'\s{2,}', ' ', relaxed).strip()
        return relaxed if len(relaxed) > 10 else original_query

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.QueryReformulator._apply_narrow
    # Requirement  : `_apply_narrow` shall append missing aspect terms to focus retrieval
    # Purpose      : Append missing aspect terms to focus retrieval
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : current_query: str; missing: List[str]
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
    def _apply_narrow(current_query: str, missing: List[str]) -> str:
        """Append missing aspect terms to focus retrieval."""
        if missing:
            narrowed_terms: List[str] = []
            for aspect in missing[:2]:
                term = aspect.split(":", 1)[-1].strip() if ":" in aspect else aspect
                if term:
                    narrowed_terms.append(term)
            if narrowed_terms:
                return f"{current_query} {' '.join(narrowed_terms)}"
        return current_query

    # ------------------------------------------------------------------
    # Strategy picker
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_strategy(
        check: SufficiencyCheck,
        available: List[ReformulationStrategy],
    ) -> ReformulationStrategy:
        """Choose the best available strategy for the given insufficiency."""
        preference_map: Dict[SufficiencyStatus, List[ReformulationStrategy]] = {
            SufficiencyStatus.EMPTY: [
                ReformulationStrategy.EXPAND,
                ReformulationStrategy.RELAX,
                ReformulationStrategy.PIVOT_DENSE,
            ],
            SufficiencyStatus.LOW_COUNT: [
                ReformulationStrategy.EXPAND,
                ReformulationStrategy.RELAX,
                ReformulationStrategy.PIVOT_DENSE,
            ],
            SufficiencyStatus.LOW_RELEVANCE: [
                ReformulationStrategy.PIVOT_DENSE,
                ReformulationStrategy.EXPAND,
                ReformulationStrategy.RELAX,
            ],
            SufficiencyStatus.LOW_COVERAGE: [
                ReformulationStrategy.NARROW,
                ReformulationStrategy.RELAX,
                ReformulationStrategy.EXPAND,
            ],
            SufficiencyStatus.LOW_DIVERSITY: [
                ReformulationStrategy.RELAX,
                ReformulationStrategy.EXPAND,
                ReformulationStrategy.PIVOT_DENSE,
            ],
        }
        preferred = preference_map.get(check.status, [])
        for strategy in preferred:
            if strategy in available:
                return strategy
        # Final fallback
        return (
            available[0]
            if available
            else ReformulationStrategy.DECOMPOSE
        )


# ---------------------------------------------------------------------------
# AgenticRAGOrchestrator
# ---------------------------------------------------------------------------


class AgenticRAGOrchestrator:
    """Orchestrates the full Agentic RAG loop for EEG research queries.

    The loop proceeds as follows:

    .. code-block:: text

        query
          └─► RetrievalDecisionMaker.decide()
                ├─► SKIP      ─► direct_answer() ─► AgenticRAGResult
                └─► RETRIEVE / VERIFY_CLAIM / DECOMPOSE
                      └─► for each sub-query (or original):
                            iteration 1..max_iterations:
                              retrieve(current_query)
                              SufficiencyEvaluator.evaluate()
                              if SUFFICIENT → break
                              else → QueryReformulator.reformulate()
                                      current_query = reformulated
                      └─► CitationVerifier.verify_batch() [optional]
                      └─► ResponseGenerator.generate() ─► answer

    Args:
        retriever: Fully initialised :class:`HybridRetriever`.
        generator: Fully initialised :class:`ResponseGenerator`.
        verifier: Optional :class:`CitationVerifier` for citation
            integrity checks before generation.
        max_iterations: Maximum retrieval + reformulation cycles per
            sub-query (default: 3).
        min_docs: Passed to :class:`SufficiencyEvaluator` (default: 3).
        min_relevance: Passed to :class:`SufficiencyEvaluator`
            (default: 0.25).
        top_k: Number of results to request from retriever each round.
    """

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.AgenticRAGOrchestrator.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : retriever: HybridRetriever; generator: ResponseGenerator; verifier: Optional[CitationVerifier] (default=None); max_iterations: int (default=_MAX_ITERATIONS_DEFAULT); min_docs: int (default=_MIN_DOCS_DEFAULT); min_relevance: float (default=_MIN_RELEVANCE_DEFAULT); top_k: int (default=10)
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
        retriever: HybridRetriever,
        generator: ResponseGenerator,
        verifier: Optional[CitationVerifier] = None,
        max_iterations: int = _MAX_ITERATIONS_DEFAULT,
        min_docs: int = _MIN_DOCS_DEFAULT,
        min_relevance: float = _MIN_RELEVANCE_DEFAULT,
        top_k: int = 10,
        fusion_outcome_log_path: Optional[Path] = None,
        max_fusion_outcomes: int = 250,
        fusion_decay_half_life_sec: float = 604800.0,
        huber_delta: float = 1.5,
        history_manager: Optional[HistoryManager] = None,
        exploration_alpha: float = 0.08,
        prediction_uncertainty_guard: float = 0.12,
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._verifier = verifier
        self._max_iterations = max_iterations
        self._top_k = top_k

        self._decision_maker = RetrievalDecisionMaker()
        self._sufficiency = SufficiencyEvaluator(
            min_docs=min_docs,
            min_relevance=min_relevance,
        )
        self._reformulator = QueryReformulator()
        self._context_aggregator = ContextAggregator(
            relevance_threshold=0.0,
            max_citations=top_k,
            entity_min_frequency=1,
            ranking_strategy="diversified",
        )
        self._persistent_bm25_weight = retriever.bm25_weight
        self._persistent_dense_weight = retriever.dense_weight
        self._fusion_outcome_log: List[Dict[str, float]] = []
        self._response_surface_coeffs: Optional[np.ndarray] = None
        self._response_surface_global_model: Optional[Dict[str, Any]] = None
        self._response_surface_coeffs_by_segment: Dict[str, Dict[str, Any]] = {}
        self._response_surface_min_samples = 12
        self._segment_response_surface_min_samples = 8
        self._max_fusion_outcomes = max(50, int(max_fusion_outcomes))
        self._fusion_decay_half_life_sec = max(3600.0, float(fusion_decay_half_life_sec))
        self._huber_delta = max(0.25, float(huber_delta))
        self._exploration_alpha = max(0.0, float(exploration_alpha))
        self._prediction_uncertainty_guard = max(
            0.02,
            float(prediction_uncertainty_guard),
        )
        default_objective_weights = {
            "utility": 0.60,
            "citation_validity": 0.25,
            "latency": 0.15,
        }
        self._objective_weights = dict(default_objective_weights)
        if objective_weights:
            self._objective_weights.update(objective_weights)
        weight_total = max(1e-6, sum(max(0.0, float(v)) for v in self._objective_weights.values()))
        self._objective_weights = {
            k: max(0.0, float(v)) / weight_total
            for k, v in self._objective_weights.items()
        }
        self._objective_weights_by_segment: Dict[str, Dict[str, float]] = {}
        self._uncertainty_calibration_by_segment: Dict[str, Dict[str, float]] = {}
        self._calibration_error_decomposition_by_segment: Dict[str, Dict[str, float]] = {}
        self._segment_decay_state: Dict[str, Dict[str, float]] = {}
        self._counterfactual_policy_eval: Dict[str, Any] = {}
        self._history_manager = history_manager
        self._fusion_outcome_log_path = (
            Path("data/search_history/fusion_outcomes.jsonl")
            if fusion_outcome_log_path is None
            else Path(fusion_outcome_log_path)
        )
        self._restore_fusion_outcomes_from_log()

        logger.info(
            "AgenticRAGOrchestrator ready "
            f"(max_iterations={max_iterations}, top_k={top_k})"
        )

    def _restore_fusion_outcomes_from_log(self) -> None:
        """Warm-load past fusion outcomes to initialize response-surface state."""
        if not self._fusion_outcome_log_path.exists():
            return

        try:
            lines = self._fusion_outcome_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError as exc:
            logger.warning(
                "Could not read fusion outcome log %s: %s",
                self._fusion_outcome_log_path,
                exc,
            )
            return

        loaded: List[Dict[str, float]] = []
        for raw_line in lines[-self._max_fusion_outcomes :]:
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            entry = {
                "bm25_weight": float(payload.get("bm25_weight", 0.5)),
                "utility": float(payload.get("utility", 0.0)),
                "concept": float(payload.get("concept", 0.0)),
                "diversity": float(payload.get("diversity", 1.0)),
                "redundancy": float(payload.get("redundancy", 0.0)),
                "centrality": float(payload.get("centrality", 0.0)),
                "timestamp": float(payload.get("timestamp", time.time())),
                "query_category": str(payload.get("query_category", "general")),
                "query_difficulty": str(payload.get("query_difficulty", "medium")),
                "feedback_score": float(payload.get("feedback_score", 0.0)),
                "click_through_rate": float(payload.get("click_through_rate", 0.0)),
            }
            loaded.append(
                {
                    "bm25_weight": max(0.0, min(1.0, entry["bm25_weight"])),
                    "utility": max(0.0, min(1.0, entry["utility"])),
                    "concept": max(0.0, min(1.0, entry["concept"])),
                    "diversity": max(0.0, min(1.0, entry["diversity"])),
                    "redundancy": max(0.0, min(1.0, entry["redundancy"])),
                    "centrality": max(0.0, min(1.0, entry["centrality"])),
                    "timestamp": entry["timestamp"],
                    "query_category": entry["query_category"],
                    "query_difficulty": entry["query_difficulty"],
                    "feedback_score": max(-1.0, min(1.0, entry["feedback_score"])),
                    "click_through_rate": max(0.0, min(1.0, entry["click_through_rate"])),
                }
            )

        self._fusion_outcome_log = loaded[-self._max_fusion_outcomes :]
        if self._fusion_outcome_log:
            self._fit_response_surface()
            logger.info(
                "Loaded %d historical fusion outcomes from %s",
                len(self._fusion_outcome_log),
                self._fusion_outcome_log_path,
            )

    @staticmethod
    def _segment_key(query_category: Optional[str], query_difficulty: Optional[str]) -> str:
        """Create a stable segment key for archetype-specific response surfaces."""
        category = (query_category or "general").strip().lower()
        difficulty = (query_difficulty or "medium").strip().lower()
        return f"{category}|{difficulty}"

    def _infer_archetype_context(
        self,
        query_text: str,
        detected_entities: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Infer archetype category and difficulty from query semantics."""
        text = query_text.lower()
        entities = " ".join(detected_entities or []).lower()
        combined = f"{text} {entities}".strip()

        category = "general"
        if any(token in combined for token in ("epilep", "clinical", "cohort", "patient")):
            category = "clinical"
        elif any(token in combined for token in ("ica", "preprocess", "artifact", "pipeline")):
            category = "preprocessing"
        elif any(token in combined for token in ("motor imagery", "bci", "ssvep")):
            category = "bci"
        elif any(token in combined for token in ("p300", "n400", "p600", "mmn", "erp")):
            category = "erp"
        elif any(token in combined for token in ("sensitivity", "specificity", "auc", "outcome")):
            category = "outcome"
        elif any(token in combined for token in ("longitudinal", "follow-up", "trajectory")):
            category = "longitudinal"
        elif any(token in combined for token in ("connectivity", "graph", "coherence", "method")):
            category = "method"

        difficulty = "medium"
        hard_markers = (
            "systematic review",
            "meta-analysis",
            "longitudinal",
            "multimodal",
            "causal",
            "connectivity",
            "graph",
        )
        easy_markers = (
            "what is",
            "define",
            "frequency band",
            "basic",
        )
        if any(marker in combined for marker in hard_markers) or len(query_text.split()) >= 14:
            difficulty = "hard"
        elif any(marker in combined for marker in easy_markers) or len(query_text.split()) <= 6:
            difficulty = "easy"

        return category, difficulty

    def _feedback_signal_from_history(self, query_text: str) -> Dict[str, float]:
        """Estimate utility correction from user feedback and click-through behavior."""
        if self._history_manager is None:
            try:
                self._history_manager = HistoryManager.get_instance()
            except Exception:
                return {"feedback_score": 0.0, "click_through_rate": 0.0}

        try:
            history_items = self._history_manager.search_in_history(query_text, limit=8)
        except Exception:
            return {"feedback_score": 0.0, "click_through_rate": 0.0}

        if not history_items:
            return {"feedback_score": 0.0, "click_through_rate": 0.0}

        normalized_query = query_text.strip().lower()
        best = None
        for item in history_items:
            if item.query_text.strip().lower() == normalized_query:
                best = item
                break
        if best is None:
            best = history_items[0]

        ctr = min(1.0, len(best.clicked_results) / max(1, best.result_count))
        feedback_map = {
            "helpful": 0.08,
            "not_helpful": -0.12,
        }
        feedback_label = (best.user_feedback or "").strip().lower()
        feedback_component = feedback_map.get(feedback_label, 0.0)
        click_component = 0.10 * (ctr - 0.2)
        feedback_score = max(-0.25, min(0.25, feedback_component + click_component))
        return {
            "feedback_score": feedback_score,
            "click_through_rate": ctr,
        }

    @staticmethod
    def _solve_weighted_ridge(
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        ridge_lambda: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve weighted ridge regression for response-surface coefficients."""
        sqrt_w = np.sqrt(np.clip(weights, 1e-6, None))
        xw = x * sqrt_w[:, None]
        yw = y * sqrt_w
        ridge = ridge_lambda * np.eye(x.shape[1], dtype=float)
        xtwx = xw.T @ xw + ridge
        coeffs = np.linalg.solve(xtwx, xw.T @ yw)
        cov_inv = np.linalg.pinv(xtwx)
        return coeffs, cov_inv

    @staticmethod
    def _huber_reweights(residuals: np.ndarray, delta: float) -> np.ndarray:
        """Return Huber IRLS weights for robust regression."""
        abs_residuals = np.abs(residuals)
        weights = np.ones_like(abs_residuals)
        mask = abs_residuals > delta
        weights[mask] = delta / np.maximum(abs_residuals[mask], 1e-6)
        return weights

    @staticmethod
    def _prediction_uncertainty(
        features: np.ndarray,
        model: Dict[str, Any],
    ) -> float:
        """Estimate predictive uncertainty from residual variance and covariance."""
        cov_inv = model.get("cov_inv")
        residual_var = float(model.get("residual_var", 0.0))
        if cov_inv is None:
            return max(0.05, min(0.5, residual_var))

        variance = float(features @ cov_inv @ features)
        pred_var = max(0.0, variance * max(1e-6, residual_var))
        return float(max(0.0, min(0.5, math.sqrt(pred_var))))

    def _adaptive_uncertainty_guard(self, segment_key: str) -> float:
        """Return current guard threshold for a segment from Bayesian calibration."""
        state = self._uncertainty_calibration_by_segment.get(segment_key)
        if not state:
            return self._prediction_uncertainty_guard
        return float(state.get("adaptive_guard", self._prediction_uncertainty_guard))

    def _bayesian_update_uncertainty_calibration(
        self,
        segment_key: str,
        residuals: np.ndarray,
    ) -> None:
        """Online Bayesian update for segment uncertainty and guard threshold."""
        state = self._uncertainty_calibration_by_segment.get(segment_key)
        if state is None:
            state = {
                "alpha": 2.0,
                "beta": 0.02,
                "adaptive_guard": self._prediction_uncertainty_guard,
                "updates": 0.0,
            }

        alpha = float(state["alpha"])
        beta = float(state["beta"])
        for residual in residuals.tolist():
            alpha += 0.5
            beta += 0.5 * float(residual) * float(residual)

        posterior_var_mean = beta / max(1e-6, alpha - 1.0)
        posterior_std = math.sqrt(max(0.0, posterior_var_mean))
        adaptive_guard = max(
            0.05,
            min(0.45, self._prediction_uncertainty_guard + (1.28 * posterior_std)),
        )

        state.update(
            {
                "alpha": alpha,
                "beta": beta,
                "adaptive_guard": adaptive_guard,
                "updates": float(state.get("updates", 0.0) + len(residuals)),
                "posterior_std": posterior_std,
            }
        )
        self._uncertainty_calibration_by_segment[segment_key] = state

    @staticmethod
    def _pareto_front_indices(candidates: List[Dict[str, float]]) -> List[int]:
        """Return indices for Pareto-efficient candidates across 3 objectives."""
        efficient: List[int] = []
        for i, cand_i in enumerate(candidates):
            dominated = False
            for j, cand_j in enumerate(candidates):
                if i == j:
                    continue
                dominates = (
                    cand_j["utility"] >= cand_i["utility"]
                    and cand_j["citation_validity"] >= cand_i["citation_validity"]
                    and cand_j["latency_ms"] <= cand_i["latency_ms"]
                    and (
                        cand_j["utility"] > cand_i["utility"]
                        or cand_j["citation_validity"] > cand_i["citation_validity"]
                        or cand_j["latency_ms"] < cand_i["latency_ms"]
                    )
                )
                if dominates:
                    dominated = True
                    break
            if not dominated:
                efficient.append(i)
        return efficient

    def _pareto_weighted_score(
        self,
        candidate: Dict[str, float],
        all_candidates: List[Dict[str, float]],
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute explicit Pareto-weighted score across utility/validity/latency."""
        util_values = [c["utility"] for c in all_candidates]
        valid_values = [c["citation_validity"] for c in all_candidates]
        lat_values = [c["latency_ms"] for c in all_candidates]
        weights = self._objective_weights if objective_weights is None else objective_weights

        def _normalize(value: float, values: List[float], maximize: bool = True) -> float:
            v_min = min(values)
            v_max = max(values)
            if abs(v_max - v_min) < 1e-6:
                return 0.5
            normalized = (value - v_min) / (v_max - v_min)
            return normalized if maximize else (1.0 - normalized)

        util_n = _normalize(candidate["utility"], util_values, maximize=True)
        valid_n = _normalize(candidate["citation_validity"], valid_values, maximize=True)
        lat_n = _normalize(candidate["latency_ms"], lat_values, maximize=False)

        return (
            weights.get("utility", 0.0) * util_n
            + weights.get("citation_validity", 0.0) * valid_n
            + weights.get("latency", 0.0) * lat_n
        )

    @staticmethod
    def _project_to_simplex_with_floor(
        weights: np.ndarray,
        floor: float = 0.05,
    ) -> np.ndarray:
        """Project a weight vector onto simplex with per-dimension lower bound."""
        dim = len(weights)
        floor = max(0.0, min(1.0 / max(1, dim), floor))
        adjusted = np.maximum(weights, floor)
        adjusted_sum = float(np.sum(adjusted))
        if adjusted_sum <= 1e-9:
            return np.full(dim, 1.0 / dim, dtype=float)
        adjusted = adjusted / adjusted_sum
        min_mask = adjusted < floor
        if not np.any(min_mask):
            return adjusted

        adjusted[min_mask] = floor
        residual = 1.0 - float(np.sum(adjusted[min_mask]))
        free_mask = ~min_mask
        if not np.any(free_mask):
            return np.full(dim, 1.0 / dim, dtype=float)
        free_values = adjusted[free_mask]
        free_sum = float(np.sum(free_values))
        if free_sum <= 1e-9:
            adjusted[free_mask] = residual / float(np.sum(free_mask))
        else:
            adjusted[free_mask] = residual * (free_values / free_sum)
        return adjusted

    def _objective_weights_for_segment(self, segment_key: str) -> Dict[str, float]:
        """Return learned objective weights for a segment or global defaults."""
        return dict(
            self._objective_weights_by_segment.get(segment_key, self._objective_weights)
        )

    def _learn_objective_weights_by_segment(self) -> None:
        """Learn constrained per-segment objective weights from logged outcomes."""
        if len(self._fusion_outcome_log) < self._segment_response_surface_min_samples:
            return

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._fusion_outcome_log:
            segment_key = self._segment_key(
                entry.get("query_category"),
                entry.get("query_difficulty"),
            )
            grouped.setdefault(segment_key, []).append(entry)

        learned: Dict[str, Dict[str, float]] = {}
        base_vec = np.array(
            [
                float(self._objective_weights.get("utility", 0.6)),
                float(self._objective_weights.get("citation_validity", 0.25)),
                float(self._objective_weights.get("latency", 0.15)),
            ],
            dtype=float,
        )

        for segment_key, entries in grouped.items():
            if len(entries) < self._segment_response_surface_min_samples:
                continue

            latest_ts = max(float(item.get("timestamp", 0.0)) for item in entries)
            feature_rows: List[np.ndarray] = []
            targets: List[float] = []
            sample_weights: List[float] = []

            for item in entries:
                concept = float(item.get("concept", 0.0))
                diversity = float(item.get("diversity", 1.0))
                redundancy = float(item.get("redundancy", 0.0))
                centrality = float(item.get("centrality", 0.0))
                bm25 = float(item.get("bm25_weight", 0.5))

                citation_validity = max(
                    0.0,
                    min(
                        1.0,
                        0.45
                        + (0.30 * concept)
                        + (0.15 * diversity)
                        - (0.20 * redundancy)
                        + (0.05 * (1.0 - abs(0.55 - bm25))),
                    ),
                )
                latency_ms = max(
                    25.0,
                    180.0
                    + (90.0 * (1.0 - bm25))
                    + (35.0 * max(0.0, 0.7 - concept))
                    + (20.0 * max(0.0, redundancy - 0.4)),
                )
                latency_utility = max(0.0, min(1.0, 1.0 - ((latency_ms - 25.0) / 280.0)))
                feature_rows.append(
                    np.asarray(
                        [
                            float(item.get("utility", 0.0)),
                            citation_validity,
                            latency_utility,
                        ],
                        dtype=float,
                    )
                )
                targets.append(float(item.get("utility", 0.0)))
                age = max(0.0, latest_ts - float(item.get("timestamp", latest_ts)))
                sample_weights.append(float(np.power(0.5, age / self._fusion_decay_half_life_sec)))

            x = np.vstack(feature_rows)
            y = np.asarray(targets, dtype=float)
            w = np.asarray(sample_weights, dtype=float)
            w = np.maximum(1e-6, w)

            params = np.copy(base_vec)
            learning_rate = 0.18
            reg_lambda = 0.05
            for _ in range(80):
                pred = x @ params
                residual = pred - y
                grad = (x.T @ (w * residual)) / float(np.sum(w))
                grad += reg_lambda * (params - base_vec)
                params = params - (learning_rate * grad)
                params = self._project_to_simplex_with_floor(params, floor=0.05)

            learned[segment_key] = {
                "utility": float(params[0]),
                "citation_validity": float(params[1]),
                "latency": float(params[2]),
            }

        self._objective_weights_by_segment = learned

    def _update_calibration_error_decomposition(
        self,
        segment_key: str,
        model: Dict[str, Any],
        residuals: np.ndarray,
    ) -> None:
        """Estimate aleatoric/epistemic calibration components per segment."""
        if residuals.size == 0:
            return

        aleatoric = float(np.mean(np.square(residuals)))
        cov_inv = model.get("cov_inv")
        if cov_inv is None:
            epistemic = aleatoric * 0.5
        else:
            trace_proxy = float(np.trace(np.asarray(cov_inv, dtype=float)))
            epistemic = max(0.0, min(0.5, aleatoric * min(2.0, math.sqrt(max(0.0, trace_proxy)))))

        total = max(1e-9, aleatoric + epistemic)
        state = {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "aleatoric_share": aleatoric / total,
            "epistemic_share": epistemic / total,
            "updated_at": float(time.time()),
        }
        self._calibration_error_decomposition_by_segment[segment_key] = state

    def _uncertainty_decomposition(self, segment_key: str) -> Dict[str, float]:
        """Return decomposition and exploration gate for segment uncertainty."""
        state = self._calibration_error_decomposition_by_segment.get(segment_key)
        if not state:
            return {
                "aleatoric": 0.03,
                "epistemic": 0.03,
                "aleatoric_share": 0.5,
                "epistemic_share": 0.5,
                "exploration_gate": 1.0,
            }

        epistemic_norm = min(1.0, float(state.get("epistemic", 0.0)) / 0.20)
        aleatoric_norm = min(1.0, float(state.get("aleatoric", 0.0)) / 0.20)
        exploration_gate = max(
            0.10,
            min(1.80, 1.0 + (0.60 * epistemic_norm) - (0.50 * aleatoric_norm)),
        )
        return {
            "aleatoric": float(state.get("aleatoric", 0.0)),
            "epistemic": float(state.get("epistemic", 0.0)),
            "aleatoric_share": float(state.get("aleatoric_share", 0.5)),
            "epistemic_share": float(state.get("epistemic_share", 0.5)),
            "exploration_gate": exploration_gate,
        }

    def _expected_objectives(
        self,
        diagnostics: Dict[str, Any],
        bm25_weight: float,
        query_category: Optional[str] = None,
        query_difficulty: Optional[str] = None,
    ) -> Dict[str, float]:
        """Predict multi-objective outcomes for candidate fusion weights."""
        utility, uncertainty = self._expected_citation_utility_with_uncertainty(
            diagnostics,
            bm25_weight,
            query_category=query_category,
            query_difficulty=query_difficulty,
        )

        concept = float(
            diagnostics.get(
                "query_concept_coverage_score",
                diagnostics.get("query_entity_coverage_score", 0.0),
            )
        )
        redundancy = float(diagnostics.get("redundancy_score", 0.0))
        diversity = float(diagnostics.get("diversity_score", 1.0))

        citation_validity = max(
            0.0,
            min(
                1.0,
                0.45
                + (0.30 * concept)
                + (0.15 * diversity)
                - (0.20 * redundancy)
                + (0.05 * (1.0 - abs(0.55 - bm25_weight))),
            ),
        )

        latency_ms = max(
            25.0,
            180.0
            + (90.0 * (1.0 - bm25_weight))
            + (35.0 * max(0.0, 0.7 - concept))
            + (20.0 * max(0.0, redundancy - 0.4)),
        )

        return {
            "utility": utility,
            "citation_validity": citation_validity,
            "latency_ms": latency_ms,
            "uncertainty": uncertainty,
        }

    def _evaluate_counterfactual_policies(
        self,
        candidate_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Replay logged outcomes with doubly robust per-segment value estimation."""
        if not self._fusion_outcome_log:
            return {
                "overall_regret": 0.0,
                "samples": 0,
                "by_segment": {},
            }

        weights = candidate_weights or [round(w, 2) for w in np.arange(0.2, 0.81, 0.05)]
        regret_by_segment: Dict[str, List[float]] = {}

        segment_counts: Dict[str, Dict[float, int]] = {}
        for entry in self._fusion_outcome_log:
            category = str(entry.get("query_category", "general"))
            difficulty = str(entry.get("query_difficulty", "medium"))
            segment_key = self._segment_key(category, difficulty)
            nearest = min(weights, key=lambda w: abs(w - float(entry.get("bm25_weight", 0.5))))
            bucket = segment_counts.setdefault(segment_key, {})
            bucket[nearest] = bucket.get(nearest, 0) + 1

        dr_values_by_segment: Dict[str, List[float]] = {}
        ips_values_by_segment: Dict[str, List[float]] = {}
        dm_values_by_segment: Dict[str, List[float]] = {}
        value_gap_by_segment: Dict[str, List[float]] = {}

        kernel_bw = 0.08

        for entry in self._fusion_outcome_log:
            diagnostics = {
                "query_concept_coverage_score": entry["concept"],
                "diversity_score": entry["diversity"],
                "redundancy_score": entry["redundancy"],
                "centrality_grounding_score": entry["centrality"],
            }
            category = str(entry.get("query_category", "general"))
            difficulty = str(entry.get("query_difficulty", "medium"))
            segment_key = self._segment_key(category, difficulty)
            reward = float(entry.get("utility", 0.0))
            logged_weight = float(entry.get("bm25_weight", 0.5))

            nearest_logged = min(weights, key=lambda w: abs(w - logged_weight))
            counts = segment_counts.get(segment_key, {})
            total_counts = max(1, sum(counts.values()))
            behavior_prob = (
                float(counts.get(nearest_logged, 0)) + 1.0
            ) / (float(total_counts) + float(len(weights)))

            observed_policy_utility = reward
            q_logged = self._expected_citation_utility(
                diagnostics,
                logged_weight,
                query_category=category,
                query_difficulty=difficulty,
            )
            best_alt = observed_policy_utility
            for candidate in weights:
                q_candidate = self._expected_citation_utility(
                    diagnostics,
                    candidate,
                    query_category=category,
                    query_difficulty=difficulty,
                )

                target_kernel = np.exp(
                    -((logged_weight - candidate) ** 2) / (2.0 * (kernel_bw ** 2))
                )
                target_norm = np.sum(
                    np.exp(
                        -((np.asarray(weights, dtype=float) - candidate) ** 2)
                        / (2.0 * (kernel_bw ** 2))
                    )
                )
                target_prob = float(target_kernel / max(1e-6, target_norm))
                importance = min(5.0, target_prob / max(1e-6, behavior_prob))

                ips_est = importance * reward
                dr_est = q_candidate + (importance * (reward - q_logged))
                dr_est = max(0.0, min(1.0, dr_est))

                dr_values_by_segment.setdefault(segment_key, []).append(dr_est)
                ips_values_by_segment.setdefault(segment_key, []).append(ips_est)
                dm_values_by_segment.setdefault(segment_key, []).append(q_candidate)
                value_gap_by_segment.setdefault(segment_key, []).append(
                    abs(dr_est - q_candidate)
                )
                if dr_est > best_alt:
                    best_alt = dr_est

            regret = max(0.0, best_alt - observed_policy_utility)
            regret_by_segment.setdefault(segment_key, []).append(regret)

        by_segment = {
            key: {
                "mean_regret": float(statistics.mean(values)),
                "p95_regret": float(np.percentile(values, 95)),
                "samples": len(values),
                "dr_value": float(statistics.mean(dr_values_by_segment.get(key, [0.0]))),
                "ips_value": float(statistics.mean(ips_values_by_segment.get(key, [0.0]))),
                "dm_value": float(statistics.mean(dm_values_by_segment.get(key, [0.0]))),
                "dr_model_gap": float(statistics.mean(value_gap_by_segment.get(key, [0.0]))),
            }
            for key, values in regret_by_segment.items()
        }

        all_regrets = [r for values in regret_by_segment.values() for r in values]
        return {
            "overall_regret": float(statistics.mean(all_regrets)) if all_regrets else 0.0,
            "samples": len(all_regrets),
            "by_segment": by_segment,
        }

    def _policy_risk_adjustment(self, segment_key: str) -> Dict[str, float]:
        """Compute risk-aware policy controls from regret and drift signals."""
        segment_eval = self._counterfactual_policy_eval.get("by_segment", {}).get(
            segment_key,
            {},
        )
        decay_state = self._segment_decay_state.get(segment_key, {})

        mean_regret = float(segment_eval.get("mean_regret", 0.0))
        p95_regret = float(segment_eval.get("p95_regret", 0.0))
        dr_model_gap = float(segment_eval.get("dr_model_gap", 0.0))
        sample_count = float(segment_eval.get("samples", 0.0))
        relative_shift = max(0.0, float(decay_state.get("last_relative_shift", 0.0)))

        regret_norm = min(1.0, mean_regret / 0.10)
        tail_norm = min(1.0, p95_regret / 0.20)
        drift_norm = min(1.0, relative_shift / 0.50)
        gap_norm = min(1.0, dr_model_gap / 0.15)
        sample_confidence = min(1.0, sample_count / 25.0)

        risk_score = sample_confidence * (
            (0.40 * regret_norm)
            + (0.25 * tail_norm)
            + (0.20 * drift_norm)
            + (0.15 * gap_norm)
        )

        return {
            "risk_score": max(0.0, min(1.0, risk_score)),
            "exploration_multiplier": max(0.30, 1.0 - (0.70 * risk_score)),
            "penalty_multiplier": min(2.0, 1.0 + (0.80 * risk_score)),
            "guard_multiplier": max(0.40, 1.0 - (0.60 * risk_score)),
            "max_step": max(0.05, 0.20 - (0.10 * risk_score)),
        }

    def _persist_fusion_outcome(self, entry: Dict[str, float]) -> None:
        """Append a single outcome observation to the persistent JSONL log."""
        try:
            self._fusion_outcome_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._fusion_outcome_log_path.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.warning(
                "Could not persist fusion outcome to %s: %s",
                self._fusion_outcome_log_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> AgenticRAGResult:
        """Execute the full agentic RAG loop for ``query``.

        Args:
            query: Natural language EEG research question.

        Returns:
            :class:`AgenticRAGResult` with answer, sources, and audit trail.

        Raises:
            ValueError: If ``query`` is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        t_start = time.monotonic()
        query = query.strip()

        # Step 1: Decide whether / how to retrieve
        decision = self._decision_maker.decide(query)
        logger.info(
            "Retrieval decision: %s — %s",
            decision.need.value,
            decision.rationale,
        )

        # Step 2: Short-circuit for skip
        if decision.need == RetrievalNeed.SKIP:
            answer = await self._direct_answer(query)
            elapsed = (time.monotonic() - t_start) * 1000
            return AgenticRAGResult(
                query=query,
                answer=answer,
                sources=[],
                steps=[],
                decision=decision,
                iterations_used=0,
                total_elapsed_ms=elapsed,
                skipped_retrieval=True,
            )

        # Step 3: Determine the set of queries to process
        queries_to_process: List[str] = (
            decision.sub_queries
            if decision.need == RetrievalNeed.DECOMPOSE
            else [query]
        )

        # Step 4: Run the agentic retrieval loop for each sub-query
        all_sources: List[HybridResult] = []
        all_steps: List[AgenticStep] = []
        total_iterations = 0

        for sub_query in queries_to_process:
            sources, steps = await self._retrieval_loop(
                original_query=query,
                sub_query=sub_query,
                decision=decision,
            )
            all_sources.extend(sources)
            all_steps.extend(steps)
            total_iterations += len(steps)

        # Deduplicate sources by doc_id (keep highest RRF score)
        all_sources = self._deduplicate(all_sources)

        # Step 5: Verify citations (optional)
        warnings: List[str] = []
        citations_verified = False
        if self._verifier and all_sources:
            warnings = await self._verify_citations(all_sources)
            citations_verified = True

        # Step 6: Generate answer
        answer = await self._generate_answer(query, all_sources)

        elapsed = (time.monotonic() - t_start) * 1000
        return AgenticRAGResult(
            query=query,
            answer=answer,
            sources=all_sources,
            steps=all_steps,
            decision=decision,
            iterations_used=total_iterations,
            total_elapsed_ms=elapsed,
            citations_verified=citations_verified,
            verification_warnings=warnings,
        )

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.AgenticRAGOrchestrator.stream
    # Requirement  : `stream` shall stream the agentic RAG response token-by-token
    # Purpose      : Stream the agentic RAG response token-by-token
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : AsyncGenerator[str, None]
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
    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream the agentic RAG response token-by-token.

        Retrieval and reformulation run synchronously before the generator
        begins streaming; only the final LLM generation is streamed.

        Args:
            query: Natural language EEG research question.

        Yields:
            Text chunks as they are produced by the LLM provider.
        """
        result = await self.run(query)
        # The answer is already complete; yield it in chunks for API parity
        chunk_size = 100
        for i in range(0, len(result.answer), chunk_size):
            yield result.answer[i: i + chunk_size]
            await asyncio.sleep(0)  # allow event-loop turn

    # ------------------------------------------------------------------
    # Internal retrieval loop
    # ------------------------------------------------------------------

    async def _retrieval_loop(
        self,
        original_query: str,
        sub_query: str,
        decision: RetrievalDecision,
    ) -> Tuple[List[HybridResult], List[AgenticStep]]:
        """Run up to ``max_iterations`` of retrieve → evaluate → reformulate.

        Args:
            original_query: The user's original query (used for fallback).
            sub_query: The query (or sub-query) to retrieve for.
            decision: Initial retrieval decision for entity/PMID info.

        Returns:
            Tuple of (collected sources, per-iteration steps).
        """
        current_query = sub_query
        bm25_weight: Optional[float] = self._persistent_bm25_weight
        dense_weight: Optional[float] = self._persistent_dense_weight

        steps: List[AgenticStep] = []
        prior_strategies: List[ReformulationStrategy] = []
        best_results: List[HybridResult] = []

        for iteration in range(1, self._max_iterations + 1):
            t_iter = time.monotonic()
            logger.info(
                "Agentic iteration %d/%d — query: %r",
                iteration,
                self._max_iterations,
                current_query[:80],
            )

            # Retrieve
            results = await self._retrieve(
                current_query,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
            results, diagnostics = await self._apply_aggregation_diagnostics(
                current_query,
                results,
            )

            # Keep the best-seen results even if later rounds are worse
            if len(results) > len(best_results):
                best_results = results

            # Evaluate sufficiency
            check = self._sufficiency.evaluate(
                current_query,
                results,
                decision,
                diagnostics=diagnostics,
            )
            logger.info(
                "Sufficiency: %s — %s",
                check.status.value,
                check.explanation,
            )

            current_bm25 = (
                self._retriever.bm25_weight
                if bm25_weight is None
                else bm25_weight
            )
            query_category, query_difficulty = self._infer_archetype_context(
                current_query,
                decision.detected_entities,
            )
            observed_utility = self._observed_citation_utility(diagnostics)
            feedback_signal = self._feedback_signal_from_history(current_query)
            adjusted_utility = max(
                0.0,
                min(1.0, observed_utility + feedback_signal["feedback_score"]),
            )
            self._record_fusion_outcome(
                bm25_weight=current_bm25,
                diagnostics=diagnostics,
                utility=adjusted_utility,
                query_text=current_query,
                query_category=query_category,
                query_difficulty=query_difficulty,
                feedback_score=feedback_signal["feedback_score"],
                click_through_rate=feedback_signal["click_through_rate"],
            )

            adaptive_bm25, adaptive_dense = self._derive_fusion_weights(
                check=check,
                diagnostics=diagnostics,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
            adaptive_bm25, adaptive_dense = self._optimize_fusion_for_expected_utility(
                bm25_weight=adaptive_bm25,
                dense_weight=adaptive_dense,
                diagnostics=diagnostics,
                query_category=query_category,
                query_difficulty=query_difficulty,
            )
            self._set_persistent_fusion_weights(adaptive_bm25, adaptive_dense)

            # Decide on reformulation for next round (if any)
            reformulation: Optional[ReformulationResult] = None
            if check.status != SufficiencyStatus.SUFFICIENT:
                if iteration < self._max_iterations:
                    reformulation = self._reformulator.reformulate(
                        original_query=original_query,
                        current_query=current_query,
                        check=check,
                        iteration=iteration,
                        prior_strategies=prior_strategies,
                    )
                    logger.info(
                        "Reformulating [%s]: %r → %r",
                        reformulation.strategy.value,
                        current_query[:60],
                        reformulation.new_query[:60],
                    )
                else:
                    logger.info(
                        "Max iterations reached; proceeding with best "
                        "results (%d docs).",
                        len(best_results),
                    )

            elapsed_ms = (time.monotonic() - t_iter) * 1000
            steps.append(
                AgenticStep(
                    iteration=iteration,
                    query_used=current_query,
                    strategy=prior_strategies[-1] if prior_strategies else None,
                    docs_retrieved=len(results),
                    top_doc_ids=[r.doc_id for r in results[:5]],
                    sufficiency=check,
                    reformulation=reformulation,
                    elapsed_ms=elapsed_ms,
                    aggregation_diagnostics=diagnostics,
                )
            )

            # Break early if sufficient
            if check.status == SufficiencyStatus.SUFFICIENT:
                best_results = results  # Use this round's results
                break

            # Apply reformulation for next round
            if reformulation:
                prior_strategies.append(reformulation.strategy)
                current_query = reformulation.new_query
                bm25_weight = (
                    reformulation.bm25_weight_hint
                    if reformulation.bm25_weight_hint is not None
                    else adaptive_bm25
                )
                dense_weight = (
                    reformulation.dense_weight_hint
                    if reformulation.dense_weight_hint is not None
                    else adaptive_dense
                )
                self._set_persistent_fusion_weights(bm25_weight, dense_weight)

        return best_results, steps

    async def _apply_aggregation_diagnostics(
        self,
        query: str,
        results: List[HybridResult],
    ) -> Tuple[List[HybridResult], Dict[str, Any]]:
        """Run aggregation diagnostics and reorder results by diversified context."""
        if not results:
            return results, {}

        agent_results = {
            "hybrid": {
                "data": [
                    {
                        "pmid": r.metadata.get("pmid") or r.metadata.get("PMID"),
                        "title": r.metadata.get("title", r.doc_id),
                        "abstract": r.text,
                        "year": r.metadata.get("year"),
                        "doi": r.metadata.get("doi"),
                        "relevance_score": min(r.rrf_score / (2.0 / 61.0), 1.0),
                        "metadata": dict(r.metadata),
                    }
                    for r in results
                ]
            }
        }
        aggregated = await self._context_aggregator.aggregate(query, agent_results)

        by_pmid = {
            str(r.metadata.get("pmid") or r.metadata.get("PMID") or ""): r
            for r in results
            if r.metadata.get("pmid") or r.metadata.get("PMID")
        }
        by_title = {
            str(r.metadata.get("title") or r.doc_id).lower(): r
            for r in results
        }

        reordered: List[HybridResult] = []
        seen: set[str] = set()
        for citation in aggregated.citations:
            result = None
            if citation.pmid:
                result = by_pmid.get(str(citation.pmid))
            if result is None and citation.title:
                result = by_title.get(citation.title.lower())
            if result is None or result.doc_id in seen:
                continue
            result.metadata.update(citation.metadata)
            reordered.append(result)
            seen.add(result.doc_id)

        for result in results:
            if result.doc_id not in seen:
                reordered.append(result)

        return reordered, dict(aggregated.statistics)

    @staticmethod
    def _derive_fusion_weights(
        check: SufficiencyCheck,
        diagnostics: Dict[str, Any],
        bm25_weight: Optional[float],
        dense_weight: Optional[float],
    ) -> Tuple[float, float]:
        """Adapt BM25/dense fusion weights from aggregation diagnostics."""
        base_bm25 = 0.5 if bm25_weight is None else bm25_weight
        redundancy = float(diagnostics.get("redundancy_score", 0.0))
        diversity = float(diagnostics.get("diversity_score", 1.0))
        concept_coverage = float(
            diagnostics.get(
                "query_concept_coverage_score",
                diagnostics.get("query_entity_coverage_score", 1.0),
            )
        )

        sparse_bias = 0.0
        if redundancy >= 0.75 or diversity <= 0.35:
            sparse_bias += min(
                0.25,
                max(0.0, (redundancy - 0.75) * 0.8)
                + max(0.0, (0.35 - diversity) * 0.5),
            )
        if concept_coverage < 0.8:
            sparse_bias += 0.10
        if check.status == SufficiencyStatus.LOW_COVERAGE:
            sparse_bias += 0.10
        elif check.status == SufficiencyStatus.LOW_DIVERSITY:
            sparse_bias += 0.15
        elif check.status == SufficiencyStatus.LOW_RELEVANCE:
            sparse_bias -= 0.10

        next_bm25 = max(0.2, min(0.8, base_bm25 + sparse_bias))
        next_dense = 1.0 - next_bm25
        return round(next_bm25, 3), round(next_dense, 3)

    def _set_persistent_fusion_weights(self, bm25_weight: float, dense_weight: float) -> None:
        """Persist tuned fusion weights for subsequent retrieval retries."""
        total = bm25_weight + dense_weight
        if total <= 0:
            return
        normalized_bm25 = round(bm25_weight / total, 3)
        normalized_dense = round(dense_weight / total, 3)
        self._persistent_bm25_weight = normalized_bm25
        self._persistent_dense_weight = normalized_dense
        self._retriever.bm25_weight = normalized_bm25
        self._retriever.dense_weight = normalized_dense

    @staticmethod
    def _observed_citation_utility(diagnostics: Dict[str, Any]) -> float:
        """Compute observed citation utility from current retrieval diagnostics."""
        concept_coverage = float(
            diagnostics.get(
                "query_concept_coverage_score",
                diagnostics.get("query_entity_coverage_score", 0.0),
            )
        )
        redundancy = float(diagnostics.get("redundancy_score", 0.0))
        diversity = float(diagnostics.get("diversity_score", 1.0))
        centrality = float(
            diagnostics.get(
                "centrality_grounding_score",
                diagnostics.get("mean_centrality_score", 0.0),
            )
        )
        utility = (
            0.45 * concept_coverage
            + 0.25 * diversity
            + 0.20 * centrality
            + 0.10 * (1.0 - redundancy)
        )
        return max(0.0, min(1.0, utility))

    @staticmethod
    def _response_surface_features(
        bm25_weight: float,
        diagnostics: Dict[str, Any],
    ) -> np.ndarray:
        """Build polynomial feature vector for utility response-surface fitting."""
        concept = float(
            diagnostics.get(
                "query_concept_coverage_score",
                diagnostics.get("query_entity_coverage_score", 0.0),
            )
        )
        diversity = float(diagnostics.get("diversity_score", 1.0))
        redundancy = float(diagnostics.get("redundancy_score", 0.0))
        centrality = float(
            diagnostics.get(
                "centrality_grounding_score",
                diagnostics.get("mean_centrality_score", 0.0),
            )
        )
        b = float(bm25_weight)
        return np.array(
            [
                1.0,
                b,
                b * b,
                concept,
                diversity,
                redundancy,
                centrality,
                b * concept,
                b * diversity,
                b * redundancy,
                b * centrality,
            ],
            dtype=float,
        )

    @staticmethod
    def _response_surface_features_compact(bm25_weight: float) -> np.ndarray:
        """Compact feature basis for segment-local fitting with sparse samples."""
        b = float(bm25_weight)
        return np.array([1.0, b, b * b], dtype=float)

    def _record_fusion_outcome(
        self,
        bm25_weight: float,
        diagnostics: Dict[str, Any],
        utility: float,
        query_text: str = "",
        query_category: str = "general",
        query_difficulty: str = "medium",
        feedback_score: float = 0.0,
        click_through_rate: float = 0.0,
    ) -> None:
        """Log retrieval outcomes and update learned response surface."""
        entry = {
            "bm25_weight": float(max(0.0, min(1.0, bm25_weight))),
            "utility": float(max(0.0, min(1.0, utility))),
            "concept": float(
                max(
                    0.0,
                    min(
                        1.0,
                        diagnostics.get(
                            "query_concept_coverage_score",
                            diagnostics.get("query_entity_coverage_score", 0.0),
                        ),
                    ),
                )
            ),
            "diversity": float(
                max(0.0, min(1.0, diagnostics.get("diversity_score", 1.0)))
            ),
            "redundancy": float(
                max(0.0, min(1.0, diagnostics.get("redundancy_score", 0.0)))
            ),
            "centrality": float(
                max(
                    0.0,
                    min(
                        1.0,
                        diagnostics.get(
                            "centrality_grounding_score",
                            diagnostics.get("mean_centrality_score", 0.0),
                        ),
                    ),
                )
            ),
            "timestamp": float(time.time()),
            "query_hash": hashlib.md5(query_text.strip().lower().encode("utf-8")).hexdigest()
            if query_text
            else "",
            "query_category": str(query_category).strip().lower() or "general",
            "query_difficulty": str(query_difficulty).strip().lower() or "medium",
            "feedback_score": float(max(-1.0, min(1.0, feedback_score))),
            "click_through_rate": float(max(0.0, min(1.0, click_through_rate))),
        }
        segment_key = self._segment_key(query_category, query_difficulty)
        self._segment_decay_state.setdefault(
            segment_key,
            {
                "half_life_multiplier": 1.0,
                "baseline_residual_var": None,
                "drift_strikes": 0.0,
                "last_residual_var": None,
                "last_relative_shift": 0.0,
            },
        )
        self._fusion_outcome_log.append(entry)
        if len(self._fusion_outcome_log) > self._max_fusion_outcomes:
            self._fusion_outcome_log = self._fusion_outcome_log[
                -self._max_fusion_outcomes :
            ]
        self._persist_fusion_outcome(entry)
        self._fit_response_surface()

    def _fit_response_surface(self) -> None:
        """Fit global and segmented robust response surfaces over logged outcomes."""
        if len(self._fusion_outcome_log) < self._response_surface_min_samples:
            self._response_surface_coeffs = None
            self._response_surface_coeffs_by_segment = {}
            self._response_surface_global_model = None
            return

        def _fit_from_entries(
            entries: List[Dict[str, Any]],
            min_samples: int,
            compact: bool = False,
            half_life_override_sec: Optional[float] = None,
        ) -> Optional[Dict[str, Any]]:
            if len(entries) < min_samples:
                return None

            design_rows: List[np.ndarray] = []
            targets: List[float] = []
            timestamps: List[float] = []
            for item in entries:
                diagnostics = {
                    "query_concept_coverage_score": item["concept"],
                    "diversity_score": item["diversity"],
                    "redundancy_score": item["redundancy"],
                    "centrality_grounding_score": item["centrality"],
                }
                if compact:
                    design_rows.append(
                        self._response_surface_features_compact(item["bm25_weight"])
                    )
                else:
                    design_rows.append(
                        self._response_surface_features(item["bm25_weight"], diagnostics)
                    )
                targets.append(float(item["utility"]))
                timestamps.append(float(item.get("timestamp", time.time())))

            x = np.vstack(design_rows)
            y = np.asarray(targets, dtype=float)
            ts = np.asarray(timestamps, dtype=float)
            newest = float(np.max(ts))
            ages = np.maximum(0.0, newest - ts)
            half_life = (
                self._fusion_decay_half_life_sec
                if half_life_override_sec is None
                else max(3600.0, float(half_life_override_sec))
            )
            decay_weights = np.power(0.5, ages / half_life)

            beta, cov_inv = self._solve_weighted_ridge(x, y, decay_weights)
            robust_weights = decay_weights
            for _ in range(3):
                residuals = y - (x @ beta)
                scale = float(np.median(np.abs(residuals))) + 1e-6
                delta = self._huber_delta * scale
                robust_component = self._huber_reweights(residuals, delta)
                robust_weights = decay_weights * robust_component
                beta, cov_inv = self._solve_weighted_ridge(x, y, robust_weights)
            final_residuals = y - (x @ beta)
            residual_var = float(np.average(final_residuals ** 2, weights=robust_weights))
            return {
                "coeffs": beta,
                "cov_inv": cov_inv,
                "residual_var": residual_var,
                "residuals": final_residuals,
                "sample_count": len(entries),
                "compact": compact,
            }

        global_model = _fit_from_entries(
            self._fusion_outcome_log,
            self._response_surface_min_samples,
        )
        self._response_surface_coeffs = (
            None if global_model is None else global_model["coeffs"]
        )
        self._response_surface_global_model = global_model

        segments: Dict[str, List[Dict[str, Any]]] = {}
        for item in self._fusion_outcome_log:
            key = self._segment_key(
                item.get("query_category"),
                item.get("query_difficulty"),
            )
            segments.setdefault(key, []).append(item)

        self._response_surface_coeffs_by_segment = {}
        for key, entries in segments.items():
            if len(entries) < self._segment_response_surface_min_samples:
                continue

            decay_state = self._segment_decay_state.get(key, {
                "half_life_multiplier": 1.0,
                "baseline_residual_var": None,
                "drift_strikes": 0.0,
            })
            segment_entries = entries
            if float(decay_state.get("drift_strikes", 0.0)) >= 2.0:
                keep = max(self._segment_response_surface_min_samples * 2, 20)
                segment_entries = entries[-keep:]

            coeffs = _fit_from_entries(
                segment_entries,
                self._segment_response_surface_min_samples,
                compact=True,
                half_life_override_sec=(
                    self._fusion_decay_half_life_sec
                    * max(0.2, float(decay_state.get("half_life_multiplier", 1.0)))
                ),
            )
            if coeffs is not None:
                self._response_surface_coeffs_by_segment[key] = coeffs
                residual_var = float(coeffs.get("residual_var", 0.0))
                baseline = decay_state.get("baseline_residual_var")
                if baseline is None:
                    baseline = residual_var
                relative_shift = (residual_var - baseline) / max(1e-6, baseline)
                if relative_shift > 0.35:
                    decay_state["drift_strikes"] = float(decay_state.get("drift_strikes", 0.0) + 1.0)
                    decay_state["half_life_multiplier"] = max(
                        0.2,
                        float(decay_state.get("half_life_multiplier", 1.0)) * 0.7,
                    )
                else:
                    decay_state["drift_strikes"] = max(
                        0.0,
                        float(decay_state.get("drift_strikes", 0.0)) - 1.0,
                    )
                    decay_state["half_life_multiplier"] = min(
                        1.0,
                        float(decay_state.get("half_life_multiplier", 1.0)) + 0.05,
                    )
                    baseline = (0.95 * float(baseline)) + (0.05 * residual_var)

                decay_state["baseline_residual_var"] = float(baseline)
                decay_state["last_residual_var"] = residual_var
                decay_state["last_relative_shift"] = float(relative_shift)
                self._segment_decay_state[key] = decay_state

                residuals = np.asarray(coeffs.get("residuals", []), dtype=float)
                if residuals.size > 0:
                    self._bayesian_update_uncertainty_calibration(key, residuals)
                    self._update_calibration_error_decomposition(key, coeffs, residuals)

        self._counterfactual_policy_eval = self._evaluate_counterfactual_policies()
        self._learn_objective_weights_by_segment()

    def _expected_citation_utility_with_uncertainty(
        self,
        diagnostics: Dict[str, Any],
        bm25_weight: float,
        query_category: Optional[str] = None,
        query_difficulty: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Predict utility and uncertainty for confidence-aware fusion optimization."""
        segment_key = self._segment_key(query_category, query_difficulty)
        model = self._response_surface_coeffs_by_segment.get(segment_key)
        if model is None:
            model = getattr(self, "_response_surface_global_model", None)

        if model is not None:
            features = (
                self._response_surface_features_compact(bm25_weight)
                if model.get("compact", False)
                else self._response_surface_features(bm25_weight, diagnostics)
            )
            predicted = float(features @ model["coeffs"])
            uncertainty = self._prediction_uncertainty(features, model)
            return max(0.0, min(1.0, predicted)), uncertainty

        utility = self._expected_citation_utility(
            diagnostics,
            bm25_weight,
            query_category=query_category,
            query_difficulty=query_difficulty,
        )
        return utility, 0.2

    def _expected_citation_utility(
        self,
        diagnostics: Dict[str, Any],
        bm25_weight: float,
        query_category: Optional[str] = None,
        query_difficulty: Optional[str] = None,
    ) -> float:
        """Predict expected utility using learned response surfaces from logs."""
        segment_key = self._segment_key(query_category, query_difficulty)
        segment_model = self._response_surface_coeffs_by_segment.get(segment_key)
        if segment_model is not None:
            coeffs = segment_model["coeffs"]
            features = (
                self._response_surface_features_compact(bm25_weight)
                if segment_model.get("compact", False)
                else self._response_surface_features(bm25_weight, diagnostics)
            )
            predicted = float(features @ coeffs)
            return max(0.0, min(1.0, predicted))

        if self._response_surface_coeffs is not None:
            features = self._response_surface_features(bm25_weight, diagnostics)
            predicted = float(features @ self._response_surface_coeffs)
            return max(0.0, min(1.0, predicted))

        if self._fusion_outcome_log:
            weighted_sum = 0.0
            total_weight = 0.0
            concept = float(
                diagnostics.get(
                    "query_concept_coverage_score",
                    diagnostics.get("query_entity_coverage_score", 0.0),
                )
            )
            diversity = float(diagnostics.get("diversity_score", 1.0))
            redundancy = float(diagnostics.get("redundancy_score", 0.0))
            centrality = float(
                diagnostics.get(
                    "centrality_grounding_score",
                    diagnostics.get("mean_centrality_score", 0.0),
                )
            )
            for entry in self._fusion_outcome_log:
                if query_category and str(entry.get("query_category", "")).lower() != query_category.lower():
                    continue
                if query_difficulty and str(entry.get("query_difficulty", "")).lower() != query_difficulty.lower():
                    continue
                distance = (
                    abs(entry["bm25_weight"] - bm25_weight)
                    + 0.5 * abs(entry["concept"] - concept)
                    + 0.3 * abs(entry["diversity"] - diversity)
                    + 0.3 * abs(entry["redundancy"] - redundancy)
                    + 0.3 * abs(entry["centrality"] - centrality)
                )
                weight = 1.0 / (1.0 + distance)
                weighted_sum += weight * entry["utility"]
                total_weight += weight
            if total_weight > 0.0:
                return max(0.0, min(1.0, weighted_sum / total_weight))

        return self._observed_citation_utility(diagnostics)

    def _optimize_fusion_for_expected_utility(
        self,
        bm25_weight: float,
        dense_weight: float,
        diagnostics: Dict[str, Any],
        query_category: Optional[str] = None,
        query_difficulty: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Choose BM25/dense mix via Pareto-weighted multi-objective scoring."""
        _ = dense_weight
        candidate_weights = [round(w, 2) for w in np.arange(0.2, 0.81, 0.05)]
        segment_key = self._segment_key(query_category, query_difficulty)
        risk_controls = self._policy_risk_adjustment(segment_key)
        uncertainty_parts = self._uncertainty_decomposition(segment_key)
        segment_objective_weights = self._objective_weights_for_segment(segment_key)
        effective_exploration_alpha = (
            self._exploration_alpha
            * risk_controls["exploration_multiplier"]
            * uncertainty_parts["exploration_gate"]
        )

        candidate_metrics: List[Dict[str, float]] = []
        for candidate in candidate_weights:
            obj = self._expected_objectives(
                diagnostics,
                candidate,
                query_category=query_category,
                query_difficulty=query_difficulty,
            )
            candidate_metrics.append(
                {
                    "bm25_weight": candidate,
                    "utility": obj["utility"],
                    "citation_validity": obj["citation_validity"],
                    "latency_ms": obj["latency_ms"],
                    "uncertainty": obj["uncertainty"],
                }
            )

        pareto_indices = set(self._pareto_front_indices(candidate_metrics))

        best_bm25 = bm25_weight
        best_score = -1e9
        best_uncertainty = 1.0
        for idx, metrics in enumerate(candidate_metrics):
            pareto_score = self._pareto_weighted_score(
                metrics,
                candidate_metrics,
                objective_weights=segment_objective_weights,
            )
            confidence_penalty = metrics["uncertainty"] * abs(
                metrics["bm25_weight"] - bm25_weight
            )
            score = (
                pareto_score
                + (effective_exploration_alpha * metrics["uncertainty"])
                - (confidence_penalty * risk_controls["penalty_multiplier"])
            )
            if idx not in pareto_indices:
                score -= 0.05
            if score > best_score:
                best_score = score
                best_bm25 = float(metrics["bm25_weight"])
                best_uncertainty = float(metrics["uncertainty"])

        guard_threshold = self._adaptive_uncertainty_guard(segment_key)
        guard_threshold = max(
            0.05,
            guard_threshold * risk_controls["guard_multiplier"],
        )
        if (
            abs(best_bm25 - bm25_weight) > 0.10
            and best_uncertainty > guard_threshold
        ):
            best_bm25 = bm25_weight

        max_step = risk_controls["max_step"]
        step = best_bm25 - bm25_weight
        if abs(step) > max_step:
            best_bm25 = bm25_weight + (max_step if step > 0.0 else -max_step)

        best_dense = round(1.0 - best_bm25, 3)
        return round(best_bm25, 3), best_dense

    # ------------------------------------------------------------------
    # Retrieval helper (runs sync retriever in executor)
    # ------------------------------------------------------------------

    async def _retrieve(
        self,
        query: str,
        bm25_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
    ) -> List[HybridResult]:
        """Call the synchronous :class:`HybridRetriever` in a thread.

        Args:
            query: Query string to retrieve for.
            bm25_weight: Override BM25 fusion weight (None = keep default).
            dense_weight: Override dense fusion weight (None = keep default).

        Returns:
            Sorted list of :class:`HybridResult` objects.
        """
        loop = asyncio.get_event_loop()
        if bm25_weight is not None and dense_weight is not None:
            self._set_persistent_fusion_weights(bm25_weight, dense_weight)

        return await loop.run_in_executor(
            None,
            lambda: self._retriever.search(query, top_k=self._top_k),
        )

    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    async def _verify_citations(
        self, sources: List[HybridResult]
    ) -> List[str]:
        """Run citation verification on retrieved sources.

        Args:
            sources: Deduplicated retrieved documents.

        Returns:
            List of human-readable warning strings for any issues found.
        """
        assert self._verifier is not None
        warnings: List[str] = []

        for source in sources:
            pmid = source.metadata.get("pmid") or source.metadata.get("PMID")
            if not pmid:
                continue
            try:
                vr = await self._verifier.verify_citation(str(pmid))
                if not vr.exists:
                    warnings.append(
                        f"PMID {pmid} could not be verified in PubMed."
                    )
                if vr.is_retracted:
                    warnings.append(
                        f"PMID {pmid} has been retracted — "
                        "exclude from clinical recommendations."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Citation verification error for %s: %s", pmid, exc)
                warnings.append(
                    f"Could not verify PMID {pmid}: {exc}"
                )

        return warnings

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    async def _generate_answer(
        self,
        query: str,
        sources: List[HybridResult],
    ) -> str:
        """Convert retrieved documents to an LLM-generated answer.

        Args:
            query: Original user query.
            sources: Evidence documents to ground the answer in.

        Returns:
            Generated answer string.
        """
        if not sources:
            return (
                "No relevant literature was found for this query. "
                "This may indicate the topic is outside the indexed corpus "
                "or requires more specific search terms."
            )

        documents = [
            Document(
                content=r.text,
                metadata=r.metadata,
                pmid=r.metadata.get("pmid") or r.metadata.get("PMID"),
                title=r.metadata.get("title"),
                authors=r.metadata.get("authors"),
                year=r.metadata.get("year"),
            )
            for r in sources
        ]

        chunks: List[str] = []
        async for chunk in self._generator.generate(
            query=query,
            context=documents,
        ):
            chunks.append(chunk)
        return "".join(chunks)

    # ---------------------------------------------------------------------------
    # ID           : rag.agentic_rag.AgenticRAGOrchestrator._direct_answer
    # Requirement  : `_direct_answer` shall generate a direct (no-retrieval) answer using the LLM
    # Purpose      : Generate a direct (no-retrieval) answer using the LLM
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
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
    async def _direct_answer(self, query: str) -> str:
        """Generate a direct (no-retrieval) answer using the LLM.

        Args:
            query: Query deemed answerable from parametric knowledge.

        Returns:
            Generated answer string.
        """
        chunks: List[str] = []
        async for chunk in self._generator.generate(
            query=query,
            context=[],
        ):
            chunks.append(chunk)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: List[HybridResult]) -> List[HybridResult]:
        """Deduplicate by doc_id, keeping the entry with the highest score.

        Args:
            results: Potentially duplicated results across sub-query rounds.

        Returns:
            Deduplicated list sorted by RRF score descending.
        """
        best: Dict[str, HybridResult] = {}
        for r in results:
            if r.doc_id not in best or r.rrf_score > best[r.doc_id].rrf_score:
                best[r.doc_id] = r
        return sorted(best.values(), key=lambda r: r.rrf_score, reverse=True)
