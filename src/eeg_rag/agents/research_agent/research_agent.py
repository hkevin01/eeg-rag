"""
ResearchAgent — Multi-source EEG Literature Coordinator.

Coordinates parallel searches across:
  • PubMedAgent     — peer-reviewed biomedical literature
  • SemanticScholarAgent — citation-graph enriched results
  • LocalDataAgent  — ingested local corpus

After retrieval:
  1. Deduplicates by PMID / DOI / normalised title
  2. Ranks by evidence quality (EvidenceRanker)
  3. Optionally feeds into SystematicReviewAgent or SynthesisAgent

Requirements:
    REQ-RESEARCH-001: Parallel multi-source literature search
    REQ-RESEARCH-002: Cross-source deduplication
    REQ-RESEARCH-003: Evidence-ranked result fusion
    REQ-RESEARCH-004: EEG-domain query expansion
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from eeg_rag.agents.base_agent import BaseAgent, AgentType, AgentResult, AgentQuery
from eeg_rag.agents.synthesis_agent.evidence_ranker import EvidenceRanker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EEG-specific query expansion for research queries
# ---------------------------------------------------------------------------

_EEG_SYNONYMS: Dict[str, List[str]] = {
    "eeg": ["electroencephalography", "electroencephalogram"],
    "bci": ["brain-computer interface", "brain machine interface"],
    "erp": ["event-related potential", "evoked potential"],
    "seizure": ["epileptic seizure", "ictus", "ictal activity"],
    "epilepsy": ["epileptic disorder", "seizure disorder"],
    "sleep": ["polysomnography", "sleep architecture"],
    "p300": ["P300 component", "P3 potential"],
    "ssvep": ["steady-state visually evoked potential"],
    "motor imagery": ["imagined movement", "kinesthetic imagery"],
    "neurofeedback": ["EEG biofeedback", "brain wave biofeedback"],
    "connectivity": ["functional connectivity", "effective connectivity", "brain network"],
    "source localization": ["EEG source reconstruction", "LORETA", "dipole fitting"],
    "artifact removal": ["artifact rejection", "ICA denoising", "blind source separation"],
}


# ---------------------------------------------------------------------------
# ID           : agents.research_agent.research_agent._expand_eeg_query
# Requirement  : `_expand_eeg_query` shall expand EEG query with domain synonyms
# Purpose      : Expand EEG query with domain synonyms
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : query: str; max_expansions: int (default=3)
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
def _expand_eeg_query(query: str, max_expansions: int = 3) -> str:
    """Expand EEG query with domain synonyms."""
    words = query.lower().split()
    added: List[str] = []
    for phrase, synonyms in _EEG_SYNONYMS.items():
        if phrase in query.lower() and len(added) < max_expansions:
            added.extend(synonyms[: max_expansions - len(added)])
            if len(added) >= max_expansions:
                break
    if added:
        return query + " " + " ".join(added)
    return query


# ---------------------------------------------------------------------------
# Result data model
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """Aggregated multi-source research result."""
    query: str
    papers: List[Dict[str, Any]]
    total_retrieved: int
    after_dedup: int
    sources_queried: List[str]
    top_evidence_level: Optional[str]
    duration_seconds: float
    timestamp: str = ""

    # ---------------------------------------------------------------------------
    # ID           : agents.research_agent.research_agent.ResearchResult.__post_init__
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
    # ID           : agents.research_agent.research_agent.ResearchResult.to_dict
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
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "after_dedup": self.after_dedup,
            "returned": len(self.papers),
            "sources_queried": self.sources_queried,
            "top_evidence_level": self.top_evidence_level,
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
            "papers": self.papers,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """
    Orchestrates parallel EEG literature search across multiple sources.

    Internally calls registered sub-agents concurrently, deduplicates
    results, and ranks by evidence quality.

    Sub-agents are injected at construction time so the ResearchAgent
    remains testable without live network calls.

    Example::

        from eeg_rag.agents.pubmed_agent import PubMedAgent
        from eeg_rag.agents.semantic_scholar_agent.s2_agent import SemanticScholarAgent

        research = ResearchAgent(
            pubmed_agent=PubMedAgent(),
            semantic_scholar_agent=SemanticScholarAgent(),
        )
        query = AgentQuery(
            text="EEG biomarkers for Alzheimer's disease",
            parameters={"max_results": 100, "min_year": 2015},
        )
        result = await research.execute(query)
    """

    # ---------------------------------------------------------------------------
    # ID           : agents.research_agent.research_agent.ResearchAgent.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str (default='ResearchAgent'); pubmed_agent: Optional[BaseAgent] (default=None); semantic_scholar_agent: Optional[BaseAgent] (default=None); local_agent: Optional[BaseAgent] (default=None); config: Optional[Dict[str, Any]] (default=None); max_results_per_source: int (default=200); use_query_expansion: bool (default=True); deduplicate: bool (default=True)
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
        name: str = "ResearchAgent",
        pubmed_agent: Optional[BaseAgent] = None,
        semantic_scholar_agent: Optional[BaseAgent] = None,
        local_agent: Optional[BaseAgent] = None,
        config: Optional[Dict[str, Any]] = None,
        max_results_per_source: int = 200,
        use_query_expansion: bool = True,
        deduplicate: bool = True,
    ):
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            name=name,
            config=config or {},
        )
        self.pubmed_agent = pubmed_agent
        self.semantic_scholar_agent = semantic_scholar_agent
        self.local_agent = local_agent
        self.max_results_per_source = max_results_per_source
        self.use_query_expansion = use_query_expansion
        self.deduplicate = deduplicate
        self.evidence_ranker = EvidenceRanker()

        active = [
            n
            for n, a in [
                ("PubMed", pubmed_agent),
                ("SemanticScholar", semantic_scholar_agent),
                ("Local", local_agent),
            ]
            if a is not None
        ]
        logger.info(
            "ResearchAgent initialised with sources: %s",
            active if active else ["none — provide papers via context"],
        )

    # ---------------------------------------------------------------------------
    # ID           : agents.research_agent.research_agent.ResearchAgent.execute
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
        t0 = datetime.now()
        try:
            result = await self.search(
                query=query.text,
                parameters=query.parameters or {},
                context_papers=query.context.get("papers", []),
            )
            elapsed = (datetime.now() - t0).total_seconds()
            return AgentResult(
                success=True,
                data=result.to_dict(),
                metadata={
                    "sources": result.sources_queried,
                    "paper_count": len(result.papers),
                },
                agent_type=AgentType.ORCHESTRATOR,
                elapsed_time=elapsed,
            )
        except Exception as exc:
            logger.exception("ResearchAgent.execute failed: %s", exc)
            elapsed = (datetime.now() - t0).total_seconds()
            return AgentResult(
                success=False,
                data={},
                error=str(exc),
                agent_type=AgentType.ORCHESTRATOR,
                elapsed_time=elapsed,
            )

    # ---------------------------------------------------------------------------
    # ID           : agents.research_agent.research_agent.ResearchAgent.search
    # Requirement  : `search` shall execute the multi-source search pipeline
    # Purpose      : Execute the multi-source search pipeline
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; parameters: Dict[str, Any]; context_papers: Optional[List[Dict[str, Any]]] (default=None)
    # Outputs      : ResearchResult
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
    async def search(
        self,
        query: str,
        parameters: Dict[str, Any],
        context_papers: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchResult:
        """
        Execute the multi-source search pipeline.

        Args:
            query: Natural language EEG research question.
            parameters: Dict with optional keys:
                max_results (int), min_year (int), max_year (int),
                sources (List[str]: "pubmed", "semantic_scholar", "local")
            context_papers: Pre-fetched papers to merge (bypass network calls).

        Returns:
            ResearchResult with deduplicated, evidence-ranked papers.
        """
        t0 = datetime.now()
        max_results = parameters.get("max_results", self.max_results_per_source)
        expanded_query = (
            _expand_eeg_query(query, max_expansions=2)
            if self.use_query_expansion
            else query
        )
        sources_queried: List[str] = []

        # --- Parallel sub-agent queries ---
        all_papers: List[Dict[str, Any]] = list(context_papers or [])

        tasks: List[Tuple[str, asyncio.Task]] = []

        if self.pubmed_agent:
            tasks.append(
                (
                    "pubmed",
                    asyncio.create_task(
                        self._query_agent(
                            self.pubmed_agent,
                            expanded_query,
                            {"max_results": max_results, **parameters},
                        )
                    ),
                )
            )

        if self.semantic_scholar_agent:
            tasks.append(
                (
                    "semantic_scholar",
                    asyncio.create_task(
                        self._query_agent(
                            self.semantic_scholar_agent,
                            expanded_query,
                            {"max_results": max_results, **parameters},
                        )
                    ),
                )
            )

        if self.local_agent:
            tasks.append(
                (
                    "local",
                    asyncio.create_task(
                        self._query_agent(
                            self.local_agent,
                            query,
                            {"top_k": max_results},
                        )
                    ),
                )
            )

        # Gather with error isolation
        if tasks:
            done = await asyncio.gather(
                *[t for _, t in tasks], return_exceptions=True
            )
            for (source_name, _), papers_or_exc in zip(tasks, done):
                if isinstance(papers_or_exc, Exception):
                    logger.warning(
                        "Source '%s' failed: %s", source_name, papers_or_exc
                    )
                else:
                    all_papers.extend(papers_or_exc)
                    sources_queried.append(source_name)

        total_retrieved = len(all_papers)

        # --- Deduplication ---
        if self.deduplicate:
            all_papers = self._deduplicate(all_papers)
        after_dedup = len(all_papers)

        # --- Evidence ranking ---
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for paper in all_papers:
            try:
                ev = self.evidence_ranker.rank_evidence(paper)
                paper["evidence_score"] = ev.overall_score
                paper["evidence_level"] = ev.evidence_level.value
                paper["study_type"] = ev.study_type
                scored.append((ev.overall_score, paper))
            except Exception:
                scored.append((0.0, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [p for _, p in scored]

        top_level = ranked[0].get("evidence_level") if ranked else None

        logger.info(
            "ResearchAgent '%s': %d retrieved, %d after dedup, sources=%s",
            query[:50],
            total_retrieved,
            after_dedup,
            sources_queried,
        )

        return ResearchResult(
            query=query,
            papers=ranked,
            total_retrieved=total_retrieved,
            after_dedup=after_dedup,
            sources_queried=sources_queried,
            top_evidence_level=top_level,
            duration_seconds=(datetime.now() - t0).total_seconds(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _query_agent(
        self,
        agent: BaseAgent,
        query: str,
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Call a sub-agent and normalise its result to a list of paper dicts."""
        sub_query = AgentQuery(
            text=query,
            parameters=parameters,
            context={},
        )
        result = await agent.execute(sub_query)
        if not result.success:
            logger.warning(
                "Sub-agent %s reported failure: %s",
                agent.__class__.__name__,
                result.error,
            )
            return []

        # Sub-agents return papers under various keys; try common ones
        for key in ("papers", "results", "documents"):
            papers = result.data.get(key)
            if isinstance(papers, list):
                return papers
        # Fallback: if data is a list itself
        if isinstance(result.data, list):
            return result.data
        return []

    # ---------------------------------------------------------------------------
    # ID           : agents.research_agent.research_agent.ResearchAgent._deduplicate
    # Requirement  : `_deduplicate` shall execute as specified
    # Purpose      :  deduplicate
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : papers: List[Dict[str, Any]]
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
    @staticmethod
    def _deduplicate(
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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

        return unique
