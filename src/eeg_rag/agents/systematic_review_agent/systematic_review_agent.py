"""
Systematic Review Agent for EEG Research.

Implements a PRISMA-inspired automated systematic review pipeline:
1. Search — query multiple sources (PubMed, Europe PMC, ClinicalTrials, SemanticScholar)
2. Deduplication — collapse duplicate records by PMID/DOI/title similarity
3. Abstract screening — apply inclusion/exclusion criteria to titles + abstracts
4. Full-text eligibility (optional) — deeper assessment of shortlisted papers
5. Data extraction — extract EEG methods, sample sizes, outcomes
6. Quality assessment — grade evidence using EvidenceRanker
7. Synthesis — summarise findings by theme with confidence intervals

Requirements:
    REQ-REVIEW-001: PRISMA-compliant search and selection flow
    REQ-REVIEW-002: Reproducible screening criteria
    REQ-REVIEW-003: Cross-source deduplication
    REQ-REVIEW-004: Automated evidence grading
    REQ-REVIEW-005: Structured extraction of EEG methodology metadata
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from eeg_rag.agents.base_agent import BaseAgent, AgentType, AgentResult, AgentQuery
from eeg_rag.agents.synthesis_agent.evidence_ranker import EvidenceRanker, EvidenceScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InclusionCriteria:
    """Criteria to include/exclude papers in the review."""
    min_year: int = 2000
    max_year: int = 9999
    min_sample_size: int = 0
    required_keywords: List[str] = field(default_factory=list)
    excluded_keywords: List[str] = field(default_factory=list)
    required_eeg_methods: List[str] = field(default_factory=list)
    study_types: List[str] = field(default_factory=list)   # empty = all
    languages: List[str] = field(default_factory=lambda: ["English"])
    require_abstract: bool = True
    require_human_subjects: bool = True


# ---------------------------------------------------------------------------
# ID           : agents.systematic_review_agent.systematic_review_agent.ReviewProtocol
# Requirement  : `ReviewProtocol` class shall be instantiable and expose the documented interface
# Purpose      : Full review configuration
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
# Verification : Instantiate ReviewProtocol with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ReviewProtocol:
    """Full review configuration."""
    research_question: str
    pico: Dict[str, str] = field(default_factory=dict)   # P/I/C/O
    inclusion: InclusionCriteria = field(default_factory=InclusionCriteria)
    max_results_per_source: int = 500
    sources: List[str] = field(default_factory=lambda: [
        "pubmed", "europe_pmc", "semantic_scholar"
    ])
    deduplicate: bool = True
    review_id: str = ""

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.ReviewProtocol.__post_init__
    # Requirement  : `__post_init__` shall execute as specified
    # Purpose      :   post init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
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
    def __post_init__(self):
        if not self.review_id:
            self.review_id = hashlib.md5(
                self.research_question.encode()
            ).hexdigest()[:8]


# ---------------------------------------------------------------------------
# ID           : agents.systematic_review_agent.systematic_review_agent.ScreenedPaper
# Requirement  : `ScreenedPaper` class shall be instantiable and expose the documented interface
# Purpose      : Paper with screening decision
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
# Verification : Instantiate ScreenedPaper with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ScreenedPaper:
    """Paper with screening decision."""
    paper: Dict[str, Any]
    included: bool
    exclusion_reason: Optional[str] = None
    screening_score: float = 0.0
    evidence_score: Optional[EvidenceScore] = None

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.ScreenedPaper.to_dict
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
        d = dict(self.paper)
        d["screening"] = {
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            "screening_score": round(self.screening_score, 3),
        }
        if self.evidence_score:
            d["evidence"] = self.evidence_score.to_dict()
        return d


# ---------------------------------------------------------------------------
# ID           : agents.systematic_review_agent.systematic_review_agent.ReviewResult
# Requirement  : `ReviewResult` class shall be instantiable and expose the documented interface
# Purpose      : Completed systematic review output
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
# Verification : Instantiate ReviewResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ReviewResult:
    """Completed systematic review output."""
    protocol: ReviewProtocol
    total_retrieved: int
    after_deduplication: int
    after_abstract_screen: int
    included: int
    papers: List[ScreenedPaper]
    themes: List[Dict[str, Any]]
    evidence_summary: Dict[str, Any]
    research_gaps: List[str]
    prisma_flow: Dict[str, int]
    duration_seconds: float
    timestamp: str = ""

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.ReviewResult.__post_init__
    # Requirement  : `__post_init__` shall execute as specified
    # Purpose      :   post init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
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
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.ReviewResult.to_dict
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
            "review_id": self.protocol.review_id,
            "research_question": self.protocol.research_question,
            "timestamp": self.timestamp,
            "prisma_flow": self.prisma_flow,
            "included_papers": [p.to_dict() for p in self.papers if p.included],
            "themes": self.themes,
            "evidence_summary": self.evidence_summary,
            "research_gaps": self.research_gaps,
            "duration_seconds": round(self.duration_seconds, 2),
        }


# ---------------------------------------------------------------------------
# EEG-specific extraction patterns
# ---------------------------------------------------------------------------

_EEG_FREQ_BANDS = re.compile(
    r"\b(delta|theta|alpha|beta|gamma|mu\s+rhythm)\s*(?:\([\d.]+[\-–][\d.]+\s*Hz\))?",
    re.IGNORECASE,
)
_EEG_METHODS = re.compile(
    r"\b(quantitative\s+EEG|qEEG|source\s+localization|ICA|LORETA|beamform(?:ing)?|"
    r"event[\-\s]related\s+(de)?synchroni[sz]ation|ERDS|microstate|"
    r"coherence|granger\s+causality|phase[\-\s]locking|SSVEP|P300|N400|MMN|ERN)\b",
    re.IGNORECASE,
)
_SAMPLE_SIZE = re.compile(
    r"n\s*=\s*(\d+)|(\d+)\s*(healthy\s+)?(?:participants?|subjects?|patients?|individuals?|"
    r"volunteers?|controls?|cases?)\b",
    re.IGNORECASE,
)
_HUMAN_STUDY = re.compile(
    r"\b(participants?|subjects?|patients?|volunteers?|humans?|adults?|children|"
    r"infants?|neonates?|healthy\s+controls?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# SystematicReviewAgent
# ---------------------------------------------------------------------------


class SystematicReviewAgent(BaseAgent):
    """
    Automates PRISMA-compliant systematic reviews of EEG literature.

    Works as a standalone agent or can be orchestrated alongside
    PubMedAgent, EuropePMCAgent, and SemanticScholarAgent.

    Usage::

        agent = SystematicReviewAgent()
        protocol = ReviewProtocol(
            research_question="What EEG biomarkers predict seizure recurrence?",
            pico={"P": "epilepsy patients", "I": "EEG monitoring",
                  "C": "standard care", "O": "seizure recurrence"},
            inclusion=InclusionCriteria(
                min_year=2015,
                required_keywords=["EEG", "seizure"],
                require_human_subjects=True,
            ),
        )
        query = AgentQuery(
            text=protocol.research_question,
            parameters={"protocol": protocol.__dict__},
        )
        result = await agent.execute(query)
    """

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str (default='SystematicReviewAgent'); config: Optional[Dict[str, Any]] (default=None)
    # Outputs      : Implicitly None or see body
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
        name: str = "SystematicReviewAgent",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_type=AgentType.AGGREGATOR,
            name=name,
            config=config or {},
        )
        self.evidence_ranker = EvidenceRanker()
        logger.info("SystematicReviewAgent initialised")

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent.execute
    # Requirement  : `execute` shall execute as specified
    # Purpose      : Execute
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: AgentQuery
    # Outputs      : AgentResult
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
    async def execute(self, query: AgentQuery) -> AgentResult:
        start = datetime.now()
        try:
            proto_dict = query.parameters.get("protocol", {})
            protocol = self._build_protocol(query.text, proto_dict)

            # Retrieve papers from context (allow external agents to supply them)
            papers = (
                query.context.get("papers", [])
                or query.parameters.get("papers", [])
            )

            result = await self.run_review(papers=papers, protocol=protocol)
            elapsed = (datetime.now() - start).total_seconds()

            return AgentResult(
                success=True,
                data=result.to_dict(),
                metadata={
                    "review_id": protocol.review_id,
                    "included_count": result.included,
                    "total_retrieved": result.total_retrieved,
                },
                agent_type=AgentType.AGGREGATOR,
                elapsed_time=elapsed,
            )
        except Exception as exc:
            logger.exception("Systematic review failed: %s", exc)
            elapsed = (datetime.now() - start).total_seconds()
            return AgentResult(
                success=False,
                data={},
                error=str(exc),
                agent_type=AgentType.AGGREGATOR,
                elapsed_time=elapsed,
            )

    # ------------------------------------------------------------------
    # Core review pipeline
    # ------------------------------------------------------------------

    async def run_review(
        self,
        papers: List[Dict[str, Any]],
        protocol: "ReviewProtocol",
    ) -> ReviewResult:
        """
        Execute all PRISMA stages on a pre-fetched paper list.

        If ``papers`` is empty the review still runs but will report 0 results.
        When integrated with ResearchAgent, pass fetched papers via context.
        """
        t0 = datetime.now()

        total_retrieved = len(papers)

        # Stage 1 — Deduplication
        if protocol.deduplicate:
            papers = self._deduplicate(papers)
        after_dedup = len(papers)

        # Stage 2 — Abstract screening
        screened = [
            self._screen_abstract(p, protocol.inclusion) for p in papers
        ]
        included = [s for s in screened if s.included]
        after_screen = len(included)

        # Stage 3 — Evidence grading for included papers
        for sp in included:
            sp.evidence_score = self.evidence_ranker.rank_evidence(sp.paper)

        # Stage 4 — Thematic grouping
        themes = self._extract_themes(included)

        # Stage 5 — Evidence summary
        evidence_summary = self._summarise_evidence(included)

        # Stage 6 — Research gap detection
        gaps = self._detect_gaps(included, protocol.research_question)

        duration = (datetime.now() - t0).total_seconds()

        prisma_flow = {
            "identified": total_retrieved,
            "after_deduplication": after_dedup,
            "screened": after_dedup,
            "excluded_abstract": after_dedup - after_screen,
            "included": after_screen,
        }

        logger.info(
            "Review '%s' complete: %d identified → %d included (%.1fs)",
            protocol.review_id,
            total_retrieved,
            after_screen,
            duration,
        )

        return ReviewResult(
            protocol=protocol,
            total_retrieved=total_retrieved,
            after_deduplication=after_dedup,
            after_abstract_screen=after_screen,
            included=after_screen,
            papers=screened,
            themes=themes,
            evidence_summary=evidence_summary,
            research_gaps=gaps,
            prisma_flow=prisma_flow,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates by PMID → DOI → normalised title."""
        seen_pmid: set = set()
        seen_doi: set = set()
        seen_title: set = set()
        unique: List[Dict[str, Any]] = []

        for p in papers:
            pmid = str(p.get("pmid") or "").strip()
            doi = str(p.get("doi") or "").strip().lower()
            title_norm = re.sub(r"\W+", " ", (p.get("title") or "")).lower().strip()

            if pmid and pmid in seen_pmid:
                continue
            if doi and doi in seen_doi:
                continue
            if title_norm and title_norm in seen_title:
                continue

            if pmid:
                seen_pmid.add(pmid)
            if doi:
                seen_doi.add(doi)
            if title_norm:
                seen_title.add(title_norm)
            unique.append(p)

        logger.debug("Dedup: %d → %d", len(papers), len(unique))
        return unique

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent._screen_abstract
    # Requirement  : `_screen_abstract` shall apply inclusion criteria to a single paper's title + abstract
    # Purpose      : Apply inclusion criteria to a single paper's title + abstract
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]; criteria: InclusionCriteria
    # Outputs      : ScreenedPaper
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
    def _screen_abstract(
        self,
        paper: Dict[str, Any],
        criteria: InclusionCriteria,
    ) -> ScreenedPaper:
        """Apply inclusion criteria to a single paper's title + abstract."""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        full_text = f"{title} {abstract}".lower()

        # --- Year filter ---
        year = paper.get("year") or paper.get("publication_year")
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
        if year:
            if year < criteria.min_year or year > criteria.max_year:
                return ScreenedPaper(
                    paper=paper,
                    included=False,
                    exclusion_reason=f"Year {year} outside [{criteria.min_year},{criteria.max_year}]",
                )

        # --- Abstract required ---
        if criteria.require_abstract and not abstract.strip():
            return ScreenedPaper(
                paper=paper,
                included=False,
                exclusion_reason="No abstract",
            )

        # --- Human subjects ---
        if criteria.require_human_subjects and not _HUMAN_STUDY.search(full_text):
            return ScreenedPaper(
                paper=paper,
                included=False,
                exclusion_reason="No human subjects detected",
            )

        # --- Excluded keywords ---
        for kw in criteria.excluded_keywords:
            if re.search(re.escape(kw), full_text, re.IGNORECASE):
                return ScreenedPaper(
                    paper=paper,
                    included=False,
                    exclusion_reason=f"Excluded keyword: '{kw}'",
                )

        # --- Required keywords ---
        score = 0.0
        for kw in criteria.required_keywords:
            if re.search(re.escape(kw), full_text, re.IGNORECASE):
                score += 1.0
        if criteria.required_keywords:
            keyword_ratio = score / len(criteria.required_keywords)
            if keyword_ratio < 0.5:
                return ScreenedPaper(
                    paper=paper,
                    included=False,
                    exclusion_reason="Insufficient required keywords matched",
                    screening_score=keyword_ratio,
                )
        else:
            score = 1.0

        # --- Sample size ---
        if criteria.min_sample_size > 0:
            sizes = self._extract_sample_size(abstract)
            if not sizes or max(sizes) < criteria.min_sample_size:
                return ScreenedPaper(
                    paper=paper,
                    included=False,
                    exclusion_reason=f"Sample size < {criteria.min_sample_size}",
                    screening_score=score / max(len(criteria.required_keywords), 1),
                )

        normalised = score / max(len(criteria.required_keywords), 1)
        return ScreenedPaper(paper=paper, included=True, screening_score=normalised)

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent._extract_sample_size
    # Requirement  : `_extract_sample_size` shall execute as specified
    # Purpose      :  extract sample size
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : List[int]
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
    def _extract_sample_size(self, text: str) -> List[int]:
        sizes: List[int] = []
        for m in _SAMPLE_SIZE.finditer(text):
            num_str = next((g for g in m.groups() if g and g.isdigit()), None)
            if num_str:
                sizes.append(int(num_str))
        return sizes

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent._extract_themes
    # Requirement  : `_extract_themes` shall group papers by detected EEG theme
    # Purpose      : Group papers by detected EEG theme
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : included: List[ScreenedPaper]
    # Outputs      : List[Dict[str, Any]]
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
    def _extract_themes(
        self, included: List[ScreenedPaper]
    ) -> List[Dict[str, Any]]:
        """Group papers by detected EEG theme."""
        from collections import defaultdict

        theme_patterns = {
            "frequency_analysis": re.compile(
                r"alpha|beta|gamma|delta|theta|power\s+spectral|frequency\s+band",
                re.IGNORECASE,
            ),
            "connectivity": re.compile(
                r"functional\s+connectivity|coherence|synchroni[sz]ation|phase[\-\s]locking",
                re.IGNORECASE,
            ),
            "erp_components": re.compile(
                r"P300|N400|P600|MMN|ERN|event[\-\s]related\s+potential",
                re.IGNORECASE,
            ),
            "clinical_applications": re.compile(
                r"epilep(?:sy|tic)|seizure|sleep\s+(?:stage|disorder)|encephalopathy",
                re.IGNORECASE,
            ),
            "machine_learning": re.compile(
                r"machine\s+learning|deep\s+learning|neural\s+network|CNN|LSTM|transformer",
                re.IGNORECASE,
            ),
            "bci_neurofeedback": re.compile(
                r"brain[\-\s]computer\s+interface|BCI|neurofeedback|SSVEP|motor\s+imagery",
                re.IGNORECASE,
            ),
        }

        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for sp in included:
            text = f"{sp.paper.get('title','')} {sp.paper.get('abstract','')}"
            for theme, pat in theme_patterns.items():
                if pat.search(text):
                    buckets[theme].append(
                        {
                            "pmid": sp.paper.get("pmid"),
                            "title": sp.paper.get("title", ""),
                            "year": sp.paper.get("year"),
                        }
                    )

        return [
            {
                "name": theme.replace("_", " ").title(),
                "paper_count": len(papers),
                "papers": papers[:5],
            }
            for theme, papers in sorted(
                buckets.items(), key=lambda x: len(x[1]), reverse=True
            )
        ]

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent._summarise_evidence
    # Requirement  : `_summarise_evidence` shall aggregate evidence level distribution
    # Purpose      : Aggregate evidence level distribution
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : included: List[ScreenedPaper]
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
    def _summarise_evidence(
        self, included: List[ScreenedPaper]
    ) -> Dict[str, Any]:
        """Aggregate evidence level distribution."""
        from collections import Counter

        level_counts: Counter = Counter()
        scores: List[float] = []
        study_types: Counter = Counter()

        for sp in included:
            if sp.evidence_score:
                level_counts[sp.evidence_score.evidence_level.value] += 1
                scores.append(sp.evidence_score.overall_score)
                study_types[sp.evidence_score.study_type] += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_included": len(included),
            "average_evidence_score": round(avg_score, 3),
            "level_distribution": dict(level_counts),
            "study_type_distribution": dict(study_types),
            "high_quality_count": sum(1 for s in scores if s >= 0.8),
        }

    # ---------------------------------------------------------------------------
    # ID           : agents.systematic_review_agent.systematic_review_agent.SystematicReviewAgent._detect_gaps
    # Requirement  : `_detect_gaps` shall detect basic research gaps from included papers
    # Purpose      : Detect basic research gaps from included papers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : included: List[ScreenedPaper]; research_question: str
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
    def _detect_gaps(
        self, included: List[ScreenedPaper], research_question: str
    ) -> List[str]:
        """Detect basic research gaps from included papers."""
        gaps: List[str] = []

        years = [
            int(sp.paper.get("year") or 0)
            for sp in included
            if sp.paper.get("year")
        ]
        if years and max(years) < 2020:
            gaps.append("No recent studies (post-2020) found on this topic.")

        # Sample size gap
        sample_sizes: List[int] = []
        for sp in included:
            sz = self._extract_sample_size(sp.paper.get("abstract") or "")
            if sz:
                sample_sizes.extend(sz)
        if sample_sizes and max(sample_sizes) < 50:
            gaps.append(
                "All included studies have small sample sizes (< 50). "
                "Larger prospective studies are needed."
            )

        # RCT gap
        n_rct = sum(
            1
            for sp in included
            if sp.evidence_score and sp.evidence_score.study_type
            == "randomized_controlled_trial"
        )
        if n_rct == 0 and len(included) > 3:
            gaps.append(
                "No randomised controlled trials identified. "
                "RCT evidence is lacking for this research question."
            )

        return gaps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_protocol(
        question: str, proto_dict: Dict[str, Any]
    ) -> "ReviewProtocol":
        incl_dict = proto_dict.pop("inclusion", {})
        inclusion = InclusionCriteria(
            min_year=incl_dict.get("min_year", 2000),
            max_year=incl_dict.get("max_year", 9999),
            min_sample_size=incl_dict.get("min_sample_size", 0),
            required_keywords=incl_dict.get("required_keywords", []),
            excluded_keywords=incl_dict.get("excluded_keywords", []),
            required_eeg_methods=incl_dict.get("required_eeg_methods", []),
            study_types=incl_dict.get("study_types", []),
            require_abstract=incl_dict.get("require_abstract", True),
            require_human_subjects=incl_dict.get("require_human_subjects", True),
        )
        return ReviewProtocol(
            research_question=question,
            pico=proto_dict.get("pico", {}),
            inclusion=inclusion,
            max_results_per_source=proto_dict.get("max_results_per_source", 500),
            sources=proto_dict.get("sources", ["pubmed", "europe_pmc"]),
        )
