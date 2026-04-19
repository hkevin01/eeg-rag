"""
ClinicalMatchingAgent — EEG Pattern to Clinical Diagnosis Matching.

Maps observed EEG characteristics to:
- Clinical diagnoses and differential diagnoses
- Standard discharge patterns (ACNS terminology)
- Drug effects on EEG
- Age-appropriate normal variant identification
- Relevant literature citations for each match

This agent is decision-support only. All output must be reviewed
by a qualified clinical neurophysiologist.

Requirements:
    REQ-CLINICAL-001: Evidence-based EEG pattern matching
    REQ-CLINICAL-002: Differential diagnosis ranking
    REQ-CLINICAL-003: Drug effect mapping
    REQ-CLINICAL-004: Age-specific normal variant detection
    REQ-CLINICAL-005: Citation-backed recommendations
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eeg_rag.agents.base_agent import BaseAgent, AgentType, AgentResult, AgentQuery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EEG pattern vocabulary  (ACNS standardised terminology 2021)
# ---------------------------------------------------------------------------

@dataclass
class EEGPattern:
    """Observed EEG pattern described by the query."""
    morphology: List[str] = field(default_factory=list)   # spike, sharp-wave, LRDA, GRDA …
    frequency_bands: List[str] = field(default_factory=list)  # delta, theta, alpha …
    distribution: List[str] = field(default_factory=list)  # focal, generalised, hemispheric …
    clinical_context: str = ""
    patient_age_group: Optional[str] = None   # neonate, infant, child, adult, elderly
    medications: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.EEGPattern.to_dict
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
            "morphology": self.morphology,
            "frequency_bands": self.frequency_bands,
            "distribution": self.distribution,
            "clinical_context": self.clinical_context,
            "patient_age_group": self.patient_age_group,
            "medications": self.medications,
        }


# ---------------------------------------------------------------------------
# ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatch
# Requirement  : `ClinicalMatch` class shall be instantiable and expose the documented interface
# Purpose      : Single matched clinical entity with evidence
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
# Verification : Instantiate ClinicalMatch with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ClinicalMatch:
    """Single matched clinical entity with evidence."""
    name: str
    category: str        # diagnosis, discharge_pattern, drug_effect, normal_variant
    confidence: float    # 0–1
    supporting_features: List[str]
    differential_rank: int = 1
    icd10_codes: List[str] = field(default_factory=list)
    acns_terminology: Optional[str] = None
    evidence_pmids: List[str] = field(default_factory=list)
    evidence_text: str = ""
    clinical_action: str = ""

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatch.to_dict
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
            "name": self.name,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "supporting_features": self.supporting_features,
            "differential_rank": self.differential_rank,
            "icd10_codes": self.icd10_codes,
            "acns_terminology": self.acns_terminology,
            "evidence_pmids": self.evidence_pmids,
            "evidence_text": self.evidence_text,
            "clinical_action": self.clinical_action,
        }


# ---------------------------------------------------------------------------
# Knowledge base — EEG patterns → clinical entities
# ---------------------------------------------------------------------------

# Each entry: (pattern_triggers, ClinicalMatch template)
# Confidence is the base score; increases with number of matching features.
_PATTERN_KB: List[Tuple[List[str], Dict[str, Any]]] = [
    # ── Epileptiform / ictal ──────────────────────────────────────
    (
        ["3 hz spike", "3hz spike", "3-hz spike-wave", "absence", "generalised spike-wave"],
        dict(
            name="Childhood Absence Epilepsy / Typical Absence Seizures",
            category="diagnosis",
            confidence=0.90,
            supporting_features=["3 Hz generalised spike-wave bursts"],
            icd10_codes=["G40.3"],
            acns_terminology="Generalised spike-wave discharge",
            evidence_pmids=["28546536", "24812626"],
            evidence_text=(
                "Classical 3 Hz generalised spike-wave bursts are the hallmark of "
                "childhood absence epilepsy. Duration ≥3s suggests clinical absence."
            ),
            clinical_action=(
                "Confirm with prolonged EEG including hyperventilation. "
                "Consider ethosuximide or valproate as first-line treatment."
            ),
        ),
    ),
    (
        ["hypsarrhythmia", "chaotic high-amplitude", "infantile spasm"],
        dict(
            name="West Syndrome / Infantile Spasms",
            category="diagnosis",
            confidence=0.92,
            supporting_features=["Hypsarrhythmia pattern", "age < 2 years"],
            icd10_codes=["G40.42"],
            acns_terminology="Hypsarrhythmia",
            evidence_pmids=["25936199", "30270014"],
            evidence_text=(
                "Hypsarrhythmia — high-amplitude, chaotic slow activity with "
                "multifocal spikes — is pathognomonic for West syndrome when "
                "combined with infantile spasms and developmental regression."
            ),
            clinical_action=(
                "Urgent neurological referral. ACTH or vigabatrin first-line. "
                "MRI brain to identify treatable aetiology."
            ),
        ),
    ),
    (
        ["periodic discharges", "LRDA", "lateralised rhythmic delta"],
        dict(
            name="Acute Focal CNS Injury / Ictal-Interictal Continuum",
            category="discharge_pattern",
            confidence=0.75,
            supporting_features=["Lateralised rhythmic delta activity"],
            icd10_codes=["G93.4"],
            acns_terminology="LRDA — Lateralised Rhythmic Delta Activity",
            evidence_pmids=["28986005", "32888273"],
            evidence_text=(
                "LRDA correlates with underlying structural injury and may represent "
                "ictal activity. Associated with seizure risk in ~40% of cases."
            ),
            clinical_action=(
                "Continuous EEG monitoring recommended. "
                "Neurology consult for antiseizure medication decision."
            ),
        ),
    ),
    (
        ["GRDA", "generalised rhythmic delta", "FIRDA", "frontal rhythmic delta"],
        dict(
            name="Diffuse Encephalopathy",
            category="discharge_pattern",
            confidence=0.70,
            supporting_features=["Generalised rhythmic delta activity"],
            icd10_codes=["G93.40"],
            acns_terminology="GRDA — Generalised Rhythmic Delta Activity",
            evidence_pmids=["28986005"],
            evidence_text=(
                "GRDA is a non-specific marker of diffuse cerebral dysfunction. "
                "Common in metabolic encephalopathy, sepsis, and post-anoxic states."
            ),
            clinical_action=(
                "Identify and treat underlying metabolic/toxic cause. "
                "Serial EEG to monitor trajectory."
            ),
        ),
    ),
    (
        ["3 per second", "LPD", "lateralised periodic discharges", "PLEDs", "PLDs"],
        dict(
            name="Lateralised Periodic Discharges (LPD / PLEDs)",
            category="discharge_pattern",
            confidence=0.82,
            supporting_features=["Lateralised periodic discharges"],
            icd10_codes=["G40.89"],
            acns_terminology="LPD — Lateralised Periodic Discharges",
            evidence_pmids=["28986005", "30195378"],
            evidence_text=(
                "LPDs indicate focal cortical irritation. Strongly associated with "
                "herpes simplex encephalitis, stroke, and neoplasms."
            ),
            clinical_action=(
                "Rule out herpes simplex encephalitis (CSF PCR + MRI). "
                "Monitor for clinical seizures."
            ),
        ),
    ),
    # ── Sleep patterns ─────────────────────────────────────────────
    (
        ["sleep spindle", "k-complex", "sigma", "stage 2", "N2"],
        dict(
            name="NREM Stage 2 Sleep (Normal)",
            category="normal_variant",
            confidence=0.95,
            supporting_features=["Sleep spindles 12–15 Hz", "K-complexes"],
            icd10_codes=[],
            acns_terminology="Sleep spindle / K-complex",
            evidence_pmids=["33285940"],
            evidence_text=(
                "Sleep spindles and K-complexes are defining features of NREM Stage 2. "
                "Spindle frequency 12–15 Hz; centroparietal maximum."
            ),
            clinical_action="No intervention needed for isolated normal sleep architecture.",
        ),
    ),
    (
        ["slow wave sleep", "delta sleep", "stage 3", "N3", "deep sleep"],
        dict(
            name="NREM Stage 3 / Slow-Wave Sleep (Normal)",
            category="normal_variant",
            confidence=0.95,
            supporting_features=["High-amplitude delta (0.5–4 Hz) >20% of epoch"],
            icd10_codes=[],
            acns_terminology="Slow wave sleep",
            evidence_pmids=["33285940"],
            evidence_text=(
                "N3 is characterised by ≥20% high-amplitude (<75 µV) delta waves. "
                "Reduced in ageing and depression."
            ),
            clinical_action="Normal finding. Note if reduced for age; consider sleep disorder work-up.",
        ),
    ),
    (
        ["REM", "saw-tooth wave", "rapid eye movement", "REM sleep"],
        dict(
            name="REM Sleep (Normal)",
            category="normal_variant",
            confidence=0.93,
            supporting_features=["Mixed frequency EEG", "REMs", "saw-tooth waves"],
            icd10_codes=[],
            acns_terminology="REM sleep",
            evidence_pmids=["33285940"],
            evidence_text=(
                "REM is characterised by mixed-frequency low-amplitude EEG, "
                "rapid eye movements, and atonia. Saw-tooth waves at 2–6 Hz."
            ),
            clinical_action=(
                "Normal. Increased REM in first cycle can indicate prior REM deprivation. "
                "REM without atonia → consider REM sleep behaviour disorder."
            ),
        ),
    ),
    # ── Drug effects ───────────────────────────────────────────────
    (
        ["benzodiazepine", "benzo", "beta frequency", "excess fast activity", "drug-induced"],
        dict(
            name="Benzodiazepine / Sedative Drug Effect",
            category="drug_effect",
            confidence=0.85,
            supporting_features=["Excess beta (13–25 Hz) activity", "diffuse distribution"],
            icd10_codes=[],
            acns_terminology="Drug-induced beta activity",
            evidence_pmids=["24917549"],
            evidence_text=(
                "Benzodiazepines, barbiturates, and some hypnotics cause diffuse beta "
                "augmentation (\"drug spindles\") and can suppress background activity at "
                "high doses."
            ),
            clinical_action=(
                "Correlate with medication history. Reduce sedatives if clinically appropriate "
                "before interpreting for encephalopathy."
            ),
        ),
    ),
    (
        ["propofol", "burst suppression", "BURST SUPPRESSION"],
        dict(
            name="Propofol-Induced Burst Suppression",
            category="drug_effect",
            confidence=0.88,
            supporting_features=["Burst-suppression pattern", "ICU context"],
            icd10_codes=[],
            acns_terminology="Burst suppression",
            evidence_pmids=["20686372"],
            evidence_text=(
                "Propofol produces dose-dependent EEG suppression progressing to burst "
                "suppression and isoelectric at anaesthetic doses."
            ),
            clinical_action=(
                "Confirm dose and target level. Distinguish from post-anoxic "
                "burst suppression — clinical context is essential."
            ),
        ),
    ),
    # ── Normal variants ────────────────────────────────────────────
    (
        ["small sharp spikes", "BETS", "benign epileptiform transients of sleep"],
        dict(
            name="Small Sharp Spikes / BETS (Benign Normal Variant)",
            category="normal_variant",
            confidence=0.88,
            supporting_features=["Small amplitude (<50 µV)", "brief duration", "drowsiness/Stage 1"],
            icd10_codes=[],
            acns_terminology="Small sharp spikes (BETS)",
            evidence_pmids=["22975469"],
            evidence_text=(
                "BETS are benign normal variants seen in drowsiness and light sleep. "
                "They can be mistaken for interictal spikes but lack after-going slow wave "
                "and have broad field."
            ),
            clinical_action="No clinical significance. Document as normal variant.",
        ),
    ),
    (
        ["14 and 6", "14 hz positive spike", "positive occipital sharp transients", "POSTS"],
        dict(
            name="POSTS / 14&6 Hz Positive Spikes (Benign Normal Variant)",
            category="normal_variant",
            confidence=0.85,
            supporting_features=["Occipital distribution", "drowsiness", "adolescent age group"],
            icd10_codes=[],
            acns_terminology="POSTS / 14 and 6 Hz positive spikes",
            evidence_pmids=["22975469"],
            evidence_text=(
                "POSTS are saw-tooth waves maximally at O1/O2, seen in drowsiness. "
                "14 and 6 Hz positive spikes are comb-like, maximal at posterior temporal regions."
            ),
            clinical_action="Normal variant. No further investigation needed.",
        ),
    ),
    # ── BCI / neurofeedback ────────────────────────────────────────
    (
        ["mu rhythm", "event-related desynchronization", "ERD", "motor imagery"],
        dict(
            name="Mu Rhythm with Motor Imagery ERD (BCI application)",
            category="normal_variant",
            confidence=0.90,
            supporting_features=["8–12 Hz rolandic (C3/C4)", "contralateral ERD on movement"],
            icd10_codes=[],
            acns_terminology="Mu rhythm",
            evidence_pmids=["11751997", "17172988"],
            evidence_text=(
                "The mu rhythm (8–12 Hz, rolandic) undergoes event-related desynchronisation "
                "during motor imagery and execution. Core signal for motor imagery BCI."
            ),
            clinical_action=(
                "Functional neuroimaging or EEG source analysis for pre-surgical motor "
                "mapping. BCI calibration protocol recommended."
            ),
        ),
    ),
]

# Simple drug-frequency lookup for supplementary drug-effect matching
_DRUG_EEG_EFFECTS: Dict[str, str] = {
    "phenytoin": "Dose-dependent increases in beta activity; toxicity causes theta slowing.",
    "carbamazepine": "Mild background slowing at therapeutic doses; toxic levels cause diffuse delta.",
    "valproate": "Mild background slowing; can cause tremor artifact.",
    "lithium": "Diffuse slowing; high levels can cause triphasic waves.",
    "clozapine": "High incidence of delta slowing and epileptiform discharges.",
    "tricyclic": "Excess frontal theta; lowers seizure threshold.",
    "opioid": "Generalised delta slowing; high doses → burst suppression.",
}


# ---------------------------------------------------------------------------
# ClinicalMatchingAgent
# ---------------------------------------------------------------------------


class ClinicalMatchingAgent(BaseAgent):
    """
    Maps EEG patterns to clinical diagnoses, discharge patterns,
    drug effects, and normal variants.

    This is a rule-based knowledge agent backed by evidence citations.
    It is NOT a replacement for clinical interpretation.

    Query format::

        AgentQuery(
            text="3 Hz spike-wave generalised absence seizures",
            parameters={
                "age_group": "child",      # neonate/infant/child/adult/elderly
                "medications": ["valproate"],
                "clinical_context": "staring spells school age child",
            }
        )
    """

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatchingAgent.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str (default='ClinicalMatchingAgent'); config: Optional[Dict[str, Any]] (default=None)
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
        name: str = "ClinicalMatchingAgent",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_type=AgentType.LOCAL_DATA,
            name=name,
            config=config or {},
        )
        logger.info("ClinicalMatchingAgent initialised (%d KB entries)", len(_PATTERN_KB))

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatchingAgent.execute
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
        from datetime import datetime

        start = datetime.now()
        try:
            pattern = self._parse_pattern_from_query(query)
            matches = self.match_pattern(pattern)
            drug_effects = self._check_drug_effects(pattern.medications)

            elapsed = (datetime.now() - start).total_seconds()
            return AgentResult(
                success=True,
                data={
                    "pattern": pattern.to_dict(),
                    "matches": [m.to_dict() for m in matches],
                    "drug_effects": drug_effects,
                    "disclaimer": (
                        "DECISION SUPPORT ONLY. All EEG interpretations must be "
                        "reviewed by a qualified clinical neurophysiologist."
                    ),
                },
                metadata={
                    "match_count": len(matches),
                    "query": query.text,
                },
                agent_type=AgentType.LOCAL_DATA,
                elapsed_time=elapsed,
            )
        except Exception as exc:
            logger.exception("ClinicalMatchingAgent error: %s", exc)
            from datetime import datetime as dt

            elapsed = (dt.now() - start).total_seconds()
            return AgentResult(
                success=False,
                data={},
                error=str(exc),
                agent_type=AgentType.LOCAL_DATA,
                elapsed_time=elapsed,
            )

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def match_pattern(
        self,
        pattern: EEGPattern,
        top_k: int = 5,
    ) -> List[ClinicalMatch]:
        """
        Return ranked ClinicalMatch list for the given EEG pattern.

        Args:
            pattern: Structured EEG pattern description.
            top_k: Maximum matches to return.

        Returns:
            List of ClinicalMatch sorted by confidence descending.
        """
        query_text = self._pattern_to_text(pattern).lower()
        candidates: List[Tuple[float, ClinicalMatch]] = []

        for triggers, template in _PATTERN_KB:
            matched_features: List[str] = []
            for trig in triggers:
                if re.search(re.escape(trig), query_text, re.IGNORECASE):
                    matched_features.append(trig)

            if not matched_features:
                continue

            # Scale confidence by overlap ratio
            score = template["confidence"] * (len(matched_features) / len(triggers))
            score = min(score + 0.05 * (len(matched_features) - 1), 1.0)

            # Age modifier
            if pattern.patient_age_group:
                score = self._apply_age_modifier(
                    score, template["name"], pattern.patient_age_group
                )

            match = ClinicalMatch(
                name=template["name"],
                category=template["category"],
                confidence=score,
                supporting_features=matched_features + template.get("supporting_features", []),
                icd10_codes=template.get("icd10_codes", []),
                acns_terminology=template.get("acns_terminology"),
                evidence_pmids=template.get("evidence_pmids", []),
                evidence_text=template.get("evidence_text", ""),
                clinical_action=template.get("clinical_action", ""),
            )
            candidates.append((score, match))

        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in candidates[:top_k]]

        # Assign differential ranks
        for i, m in enumerate(results, 1):
            m.differential_rank = i

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_pattern_from_query(self, query: AgentQuery) -> EEGPattern:
        text = query.text.lower()
        params = query.parameters or {}

        # Extract distribution signals from free text
        dist: List[str] = []
        if re.search(r"\bfocal\b|\bunilateral\b|\blaterali[sz]ed\b", text):
            dist.append("focal")
        if re.search(r"\bgenerali[sz]ed\b|\bdiffuse\b|\bbilateral\b", text):
            dist.append("generalised")

        # Extract frequency bands
        freq: List[str] = []
        for band in ["delta", "theta", "alpha", "beta", "gamma", "mu"]:
            if band in text:
                freq.append(band)

        return EEGPattern(
            morphology=[],  # could extract with NER; left for future
            frequency_bands=freq,
            distribution=dist,
            clinical_context=params.get("clinical_context", ""),
            patient_age_group=params.get("age_group"),
            medications=params.get("medications", []),
        )

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatchingAgent._pattern_to_text
    # Requirement  : `_pattern_to_text` shall execute as specified
    # Purpose      :  pattern to text
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pattern: EEGPattern
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
    def _pattern_to_text(pattern: EEGPattern) -> str:
        parts = (
            pattern.morphology
            + pattern.frequency_bands
            + pattern.distribution
            + ([pattern.clinical_context] if pattern.clinical_context else [])
        )
        return " ".join(parts)

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatchingAgent._apply_age_modifier
    # Requirement  : `_apply_age_modifier` shall boost or penalise based on age appropriateness
    # Purpose      : Boost or penalise based on age appropriateness
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : score: float; match_name: str; age_group: str
    # Outputs      : float
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
    def _apply_age_modifier(
        score: float, match_name: str, age_group: str
    ) -> float:
        """Boost or penalise based on age appropriateness."""
        paediatric_patterns = {"west syndrome", "childhood absence", "benign infantile"}
        adult_patterns = {"lrda", "lpd", "grda", "encephalopathy"}
        name_lower = match_name.lower()
        if age_group in ("neonate", "infant", "child"):
            for p in paediatric_patterns:
                if p in name_lower:
                    return min(score + 0.05, 1.0)
        if age_group in ("adult", "elderly"):
            for p in adult_patterns:
                if p in name_lower:
                    return min(score + 0.05, 1.0)
        return score

    # ---------------------------------------------------------------------------
    # ID           : agents.clinical_matching_agent.clinical_matching_agent.ClinicalMatchingAgent._check_drug_effects
    # Requirement  : `_check_drug_effects` shall execute as specified
    # Purpose      :  check drug effects
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : medications: List[str]
    # Outputs      : List[Dict[str, str]]
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
    def _check_drug_effects(medications: List[str]) -> List[Dict[str, str]]:
        effects: List[Dict[str, str]] = []
        for drug in medications:
            for key, description in _DRUG_EEG_EFFECTS.items():
                if key in drug.lower():
                    effects.append({"drug": drug, "eeg_effect": description})
        return effects
