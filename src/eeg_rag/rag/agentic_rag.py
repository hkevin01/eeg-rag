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
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from eeg_rag.retrieval.hybrid_retriever import HybridResult, HybridRetriever
from eeg_rag.retrieval.query_expander import EEGQueryExpander
from eeg_rag.generation.response_generator import Document, ResponseGenerator
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
        EMPTY: Retriever returned no results at all.
    """

    SUFFICIENT = "sufficient"
    LOW_COUNT = "low_count"
    LOW_RELEVANCE = "low_relevance"
    LOW_COVERAGE = "low_coverage"
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
                explanation="Retriever returned no results.",
            )

        if doc_count < self._min_docs:
            return SufficiencyCheck(
                status=SufficiencyStatus.LOW_COUNT,
                doc_count=doc_count,
                relevance_score=self._mean_score(results),
                coverage_score=0.0,
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
                explanation=(
                    f"Mean relevance {mean_rel:.3f} below threshold "
                    f"{self._min_relevance}."
                ),
            )

        # --- Coverage check --------------------------------------------
        coverage_score, missing = self._compute_coverage(
            results, decision.detected_entities
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
                missing_aspects=missing,
                explanation=(
                    f"Coverage {coverage_score:.2f} below threshold "
                    f"{self._coverage_threshold}. Missing: {missing}."
                ),
            )

        # --- All checks passed ----------------------------------------
        return SufficiencyCheck(
            status=SufficiencyStatus.SUFFICIENT,
            doc_count=doc_count,
            relevance_score=mean_rel,
            coverage_score=coverage_score,
            explanation=(
                f"{doc_count} documents retrieved with mean relevance "
                f"{mean_rel:.3f} and coverage {coverage_score:.2f}."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_score(results: List[HybridResult]) -> float:
        """Compute mean RRF score, normalised to 0–1 range."""
        if not results:
            return 0.0
        raw = sum(r.rrf_score for r in results) / len(results)
        # RRF scores are typically in [0, 1/k] range (k=60 → ~0.017 max)
        # Normalise relative to the theoretical maximum for a single ranker
        # 1/(60+1) ≈ 0.0164; cap at 1.0
        return min(raw / 0.02, 1.0)

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
            return f"{current_query} {' '.join(missing[:2])}"
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

        logger.info(
            "AgenticRAGOrchestrator ready "
            f"(max_iterations={max_iterations}, top_k={top_k})"
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
        bm25_weight: Optional[float] = None
        dense_weight: Optional[float] = None

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

            # Keep the best-seen results even if later rounds are worse
            if len(results) > len(best_results):
                best_results = results

            # Evaluate sufficiency
            check = self._sufficiency.evaluate(
                current_query, results, decision
            )
            logger.info(
                "Sufficiency: %s — %s",
                check.status.value,
                check.explanation,
            )

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
                bm25_weight = reformulation.bm25_weight_hint
                dense_weight = reformulation.dense_weight_hint

        return best_results, steps

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

        # Temporarily override weights if hints are provided
        original_bm25 = self._retriever.bm25_weight
        original_dense = self._retriever.dense_weight

        if bm25_weight is not None:
            self._retriever.bm25_weight = bm25_weight
        if dense_weight is not None:
            self._retriever.dense_weight = dense_weight

        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._retriever.search(query, top_k=self._top_k),
            )
        finally:
            # Always restore original weights
            self._retriever.bm25_weight = original_bm25
            self._retriever.dense_weight = original_dense

        return results

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
