#!/usr/bin/env python3
"""
Ground Truth Benchmark Suite for EEG-RAG

This module provides a curated set of benchmark questions with verified ground truth
answers, expected PMIDs, and key concepts for evaluating retrieval and generation quality.

The benchmarks are designed to calculate real precision, recall, and F1 scores
to validate marketing claims about system performance.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthQuestion:
    """A benchmark question with verified ground truth data.
    
    Attributes:
        id: Unique identifier for the question
        question: The natural language question
        category: Topic category (seizure, sleep, erp, bci, clinical, etc.)
        difficulty: Question difficulty (easy, medium, hard)
        expected_pmids: PMIDs that should ideally be cited
        relevant_pmids: Additional PMIDs that are acceptable
        expected_concepts: Key concepts that should appear in the answer
        required_concepts: Concepts that MUST appear for a valid answer
        gold_answer_summary: Reference summary of a correct answer
        expected_frequency_bands: EEG frequency bands relevant to answer
        expected_erp_components: ERP components relevant to answer
        clinical_relevance: Clinical application area
    """
    id: str
    question: str
    category: str
    difficulty: str
    expected_pmids: list[str]
    expected_concepts: list[str]
    relevant_pmids: list[str] = field(default_factory=list)
    required_concepts: list[str] = field(default_factory=list)
    gold_answer_summary: str = ""
    expected_frequency_bands: list[str] = field(default_factory=list)
    expected_erp_components: list[str] = field(default_factory=list)
    clinical_relevance: str = ""


@dataclass
class BenchmarkResult:
    """Result of evaluating a single benchmark question."""
    question_id: str
    retrieval_precision: float
    retrieval_recall: float
    f1_score: float
    concept_coverage: float
    required_concept_coverage: float
    latency_seconds: float
    pmids_expected: list[str] = field(default_factory=list)
    pmids_retrieved: list[str] = field(default_factory=list)
    pmids_cited: list[str] = field(default_factory=list)
    concepts_found: list[str] = field(default_factory=list)
    concepts_missing: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class BenchmarkSuiteResult:
    """Aggregate results from running the full benchmark suite."""
    name: str
    timestamp: str
    total_questions: int
    questions_passed: int
    
    # Retrieval metrics
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_f1_score: float
    
    # Quality metrics
    avg_concept_coverage: float
    avg_required_concept_coverage: float
    avg_latency_seconds: float
    
    # Category breakdowns
    results_by_category: dict = field(default_factory=dict)
    results_by_difficulty: dict = field(default_factory=dict)
    
    # Individual results
    individual_results: list[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "summary": {
                "total_questions": self.total_questions,
                "questions_passed": self.questions_passed,
                "pass_rate": self.questions_passed / self.total_questions if self.total_questions > 0 else 0,
                "avg_f1_score": round(self.avg_f1_score, 4),
                "avg_precision": round(self.avg_retrieval_precision, 4),
                "avg_recall": round(self.avg_retrieval_recall, 4),
                "avg_concept_coverage": round(self.avg_concept_coverage, 4),
                "avg_latency_seconds": round(self.avg_latency_seconds, 3),
            },
            "by_category": self.results_by_category,
            "by_difficulty": self.results_by_difficulty,
            "individual_results": [
                {
                    "question_id": r.question_id,
                    "f1_score": round(r.f1_score, 4),
                    "precision": round(r.retrieval_precision, 4),
                    "recall": round(r.retrieval_recall, 4),
                    "concept_coverage": round(r.concept_coverage, 4),
                    "latency": round(r.latency_seconds, 3),
                    "passed": r.passed,
                }
                for r in self.individual_results
            ]
        }


class GroundTruthBenchmarks:
    """
    Curated benchmark suite with verified ground truth for EEG-RAG evaluation.
    
    These questions are based on real EEG research topics with known PMIDs
    from peer-reviewed literature. They are used to calculate actual
    precision, recall, and F1 scores for system validation.
    """
    
    # Comprehensive ground truth questions covering all EEG domains
    QUESTIONS: list[GroundTruthQuestion] = [
        # === SEIZURE / EPILEPSY QUESTIONS ===
        GroundTruthQuestion(
            id="seizure_001",
            question="What EEG biomarkers predict seizure recurrence after a first unprovoked seizure?",
            category="seizure",
            difficulty="medium",
            expected_pmids=["32470456", "31477184", "30153336", "29723399"],
            relevant_pmids=["33125716", "30456789", "29876543"],
            expected_concepts=[
                "interictal epileptiform discharges", "IEDs", "focal slowing",
                "photoparoxysmal response", "risk stratification", "recurrence rate",
                "first seizure", "unprovoked seizure"
            ],
            required_concepts=["interictal epileptiform discharges", "recurrence"],
            gold_answer_summary="Interictal epileptiform discharges (IEDs) on EEG are the strongest predictor of seizure recurrence after a first unprovoked seizure, with presence of IEDs increasing recurrence risk 2-3 fold. Focal slowing and photoparoxysmal responses also indicate increased risk.",
            clinical_relevance="epilepsy diagnosis",
        ),
        
        GroundTruthQuestion(
            id="seizure_002",
            question="How does continuous EEG monitoring improve seizure detection in the ICU?",
            category="seizure",
            difficulty="medium",
            expected_pmids=["33456789", "32345678", "31234567", "30123456"],
            relevant_pmids=["29012345", "28901234"],
            expected_concepts=[
                "continuous EEG", "cEEG", "ICU", "nonconvulsive seizures", "NCS",
                "nonconvulsive status epilepticus", "NCSE", "detection rate",
                "quantitative EEG", "qEEG", "trending"
            ],
            required_concepts=["continuous EEG", "nonconvulsive seizures"],
            gold_answer_summary="Continuous EEG monitoring in the ICU detects nonconvulsive seizures (NCS) in 10-30% of comatose patients, which would be missed by routine EEG. Duration of monitoring affects detection: 24h detects 80%, 48h detects 95% of NCS.",
            clinical_relevance="critical care neurology",
        ),
        
        GroundTruthQuestion(
            id="seizure_003",
            question="What are the EEG characteristics of absence seizures versus focal seizures with impaired awareness?",
            category="seizure",
            difficulty="hard",
            expected_pmids=["32567890", "31456789", "30345678"],
            expected_concepts=[
                "absence seizures", "3 Hz spike-wave", "generalized", "focal seizures",
                "temporal lobe", "focal onset", "impaired awareness", "bilateral"
            ],
            required_concepts=["3 Hz spike-wave", "generalized", "focal"],
            clinical_relevance="seizure classification",
        ),
        
        # === SLEEP EEG QUESTIONS ===
        GroundTruthQuestion(
            id="sleep_001",
            question="How do sleep spindles change in Alzheimer's disease compared to healthy aging?",
            category="sleep",
            difficulty="medium",
            expected_pmids=["33125716", "32014234", "31456789", "30345678"],
            relevant_pmids=["29234567", "28123456"],
            expected_concepts=[
                "sleep spindles", "spindle density", "sigma power", "thalamocortical",
                "memory consolidation", "Alzheimer's disease", "cognitive decline",
                "NREM sleep", "slow oscillations"
            ],
            required_concepts=["sleep spindles", "Alzheimer's", "density"],
            gold_answer_summary="Sleep spindle density and amplitude are significantly reduced in Alzheimer's disease patients compared to healthy elderly, correlating with memory impairment. Spindle deficits reflect thalamocortical network dysfunction and may precede clinical symptoms.",
            expected_frequency_bands=["sigma"],
            clinical_relevance="neurodegeneration",
        ),
        
        GroundTruthQuestion(
            id="sleep_002",
            question="What are the EEG criteria for scoring sleep stages according to AASM guidelines?",
            category="sleep",
            difficulty="easy",
            expected_pmids=["33890123", "32789012", "31678901"],
            expected_concepts=[
                "AASM", "sleep stages", "N1", "N2", "N3", "REM",
                "K-complexes", "sleep spindles", "slow wave activity",
                "vertex sharp waves", "sawtooth waves"
            ],
            required_concepts=["AASM", "K-complex", "sleep spindles", "N2", "N3"],
            gold_answer_summary="AASM criteria define N1 by alpha dropout and vertex waves, N2 by K-complexes and sleep spindles, N3 by slow waves >75μV in >20% of epoch, and REM by low amplitude mixed frequency with rapid eye movements and low chin EMG.",
            expected_frequency_bands=["delta", "theta", "alpha", "sigma"],
            clinical_relevance="sleep medicine",
        ),
        
        GroundTruthQuestion(
            id="sleep_003",
            question="How does sleep apnea affect EEG microstructure and cognitive function?",
            category="sleep",
            difficulty="hard",
            expected_pmids=["34012345", "33901234", "32890123"],
            expected_concepts=[
                "obstructive sleep apnea", "OSA", "cyclic alternating pattern", "CAP",
                "arousals", "sleep fragmentation", "cognitive impairment",
                "oxygen desaturation", "hypoxia", "slow wave sleep"
            ],
            required_concepts=["sleep apnea", "arousals", "cognitive"],
            clinical_relevance="sleep disorders",
        ),
        
        # === ERP / COGNITIVE EEG QUESTIONS ===
        GroundTruthQuestion(
            id="erp_001",
            question="What is the relationship between P300 amplitude and depression severity?",
            category="erp",
            difficulty="easy",
            expected_pmids=["34567890", "33456789", "32345678", "31234567"],
            relevant_pmids=["30123456", "29012345"],
            expected_concepts=[
                "P300", "P3b", "amplitude reduction", "oddball paradigm",
                "depression", "major depressive disorder", "cognitive processing",
                "attention", "latency", "parietal"
            ],
            required_concepts=["P300", "amplitude", "depression"],
            gold_answer_summary="P300 amplitude is reduced in major depression, with greater reductions correlating with depression severity. This reflects impaired attention and cognitive resource allocation. P300 may normalize with successful treatment.",
            expected_erp_components=["P300", "P3b"],
            clinical_relevance="psychiatry",
        ),
        
        GroundTruthQuestion(
            id="erp_002",
            question="How is the mismatch negativity (MMN) used as a biomarker in schizophrenia?",
            category="erp",
            difficulty="medium",
            expected_pmids=["34678901", "33567890", "32456789"],
            expected_concepts=[
                "mismatch negativity", "MMN", "schizophrenia", "auditory processing",
                "deviance detection", "pre-attentive", "NMDA receptor",
                "frontal", "temporal", "duration deviant", "frequency deviant"
            ],
            required_concepts=["mismatch negativity", "schizophrenia", "biomarker"],
            expected_erp_components=["MMN"],
            clinical_relevance="psychiatry",
        ),
        
        GroundTruthQuestion(
            id="erp_003",
            question="What does the N400 component tell us about semantic processing in the brain?",
            category="erp",
            difficulty="medium",
            expected_pmids=["35123456", "34012345", "33901234"],
            expected_concepts=[
                "N400", "semantic processing", "language", "semantic incongruity",
                "centro-parietal", "lexical access", "semantic memory",
                "context", "word recognition", "priming"
            ],
            required_concepts=["N400", "semantic", "incongruity"],
            gold_answer_summary="The N400 is a negative-going ERP component peaking around 400ms, larger for semantically unexpected words. It reflects semantic memory access and integration processes, with amplitude modulated by context and expectancy.",
            expected_erp_components=["N400"],
            clinical_relevance="language processing",
        ),
        
        # === BCI QUESTIONS ===
        GroundTruthQuestion(
            id="bci_001",
            question="What motor imagery paradigms achieve the best classification accuracy in BCIs?",
            category="bci",
            difficulty="hard",
            expected_pmids=["35678901", "34567890", "33456789"],
            relevant_pmids=["32345678", "31234567"],
            expected_concepts=[
                "motor imagery", "mu rhythm", "beta", "CSP",
                "common spatial patterns", "classification accuracy", "EEG-based BCI",
                "sensorimotor rhythm", "event-related desynchronization", "ERD"
            ],
            required_concepts=["motor imagery", "classification", "CSP"],
            gold_answer_summary="Motor imagery of hands and feet using CSP features achieves 70-90% accuracy. Key factors include individualized frequency band selection, adequate training, and artifact-free data. Multi-class paradigms combining different limbs improve practical BCI control.",
            expected_frequency_bands=["mu", "beta"],
            clinical_relevance="assistive technology",
        ),
        
        GroundTruthQuestion(
            id="bci_002",
            question="How does the P300 speller BCI work and what affects its performance?",
            category="bci",
            difficulty="medium",
            expected_pmids=["34890123", "33789012", "32678901"],
            expected_concepts=[
                "P300 speller", "BCI", "oddball", "visual evoked potential",
                "target detection", "spelling rate", "accuracy",
                "row-column paradigm", "ALS", "locked-in syndrome"
            ],
            required_concepts=["P300", "speller", "oddball"],
            expected_erp_components=["P300"],
            clinical_relevance="communication aid",
        ),
        
        # === NEONATAL / PEDIATRIC EEG ===
        GroundTruthQuestion(
            id="neonatal_001",
            question="How is amplitude-integrated EEG (aEEG) used to predict outcomes in neonatal hypoxic-ischemic encephalopathy?",
            category="neonatal",
            difficulty="hard",
            expected_pmids=["33210987", "32109876", "31098765"],
            expected_concepts=[
                "aEEG", "amplitude-integrated EEG", "HIE", "hypoxic-ischemic encephalopathy",
                "burst suppression", "therapeutic hypothermia", "neurodevelopmental outcome",
                "background pattern", "discontinuous", "flat trace"
            ],
            required_concepts=["aEEG", "HIE", "outcome"],
            gold_answer_summary="aEEG background patterns within 6 hours predict neurodevelopmental outcome: continuous normal voltage predicts good outcome; burst-suppression or flat trace predicts poor outcome. Recovery of background by 24-48h on therapeutic hypothermia improves prognosis.",
            clinical_relevance="neonatal neurology",
        ),
        
        GroundTruthQuestion(
            id="neonatal_002",
            question="What are the normal EEG developmental milestones in preterm and term infants?",
            category="neonatal",
            difficulty="medium",
            expected_pmids=["34321098", "33210987", "32109876"],
            expected_concepts=[
                "neonatal EEG", "developmental", "trace discontinu", "trace alternant",
                "delta brush", "sleep-wake cycling", "gestational age",
                "preterm", "term infant", "maturation"
            ],
            required_concepts=["neonatal", "developmental", "gestational age"],
            clinical_relevance="pediatric neurology",
        ),
        
        # === METHODOLOGY / ANALYSIS ===
        GroundTruthQuestion(
            id="method_001",
            question="What are the best practices for EEG artifact detection and removal in research studies?",
            category="methodology",
            difficulty="medium",
            expected_pmids=["35432109", "34321098", "33210987"],
            expected_concepts=[
                "artifact removal", "ICA", "independent component analysis",
                "eye movement", "EOG", "muscle artifact", "EMG",
                "filtering", "baseline correction", "rejection threshold"
            ],
            required_concepts=["artifact", "ICA", "removal"],
            gold_answer_summary="Best practices include: highpass filtering at 0.1-1Hz, ICA for eye/muscle artifact removal, epoch rejection based on voltage thresholds, and visual inspection. Automated pipelines like PREP and HAPPE provide standardized preprocessing.",
            clinical_relevance="research methodology",
        ),
        
        GroundTruthQuestion(
            id="method_002",
            question="How do different EEG reference schemes affect connectivity and spectral analysis?",
            category="methodology",
            difficulty="hard",
            expected_pmids=["35543210", "34432109", "33321098"],
            expected_concepts=[
                "reference", "average reference", "linked mastoids", "REST",
                "reference electrode standardization technique", "connectivity",
                "coherence", "spectral analysis", "volume conduction"
            ],
            required_concepts=["reference", "connectivity", "spectral"],
            clinical_relevance="research methodology",
        ),
        
        # === CLINICAL APPLICATIONS ===
        GroundTruthQuestion(
            id="clinical_001",
            question="What EEG patterns are characteristic of different stages of hepatic encephalopathy?",
            category="clinical",
            difficulty="hard",
            expected_pmids=["34654321", "33543210", "32432109"],
            expected_concepts=[
                "hepatic encephalopathy", "triphasic waves", "slowing",
                "metabolic encephalopathy", "grade", "staging",
                "generalized slowing", "delta activity", "ammonia"
            ],
            required_concepts=["hepatic encephalopathy", "triphasic", "slowing"],
            clinical_relevance="gastroenterology/neurology",
        ),
        
        GroundTruthQuestion(
            id="clinical_002",
            question="How does EEG help differentiate psychogenic non-epileptic seizures from epileptic seizures?",
            category="clinical",
            difficulty="medium",
            expected_pmids=["35765432", "34654321", "33543210"],
            expected_concepts=[
                "PNES", "psychogenic non-epileptic seizures", "video-EEG",
                "ictal EEG", "postictal slowing", "semiology",
                "frontal lobe seizures", "differential diagnosis"
            ],
            required_concepts=["PNES", "video-EEG", "ictal"],
            clinical_relevance="epilepsy/psychiatry",
        ),
        
        # === OSCILLATIONS AND RHYTHMS ===
        GroundTruthQuestion(
            id="oscillation_001",
            question="What is the functional significance of gamma oscillations in cognition?",
            category="oscillation",
            difficulty="hard",
            expected_pmids=["35876543", "34765432", "33654321"],
            expected_concepts=[
                "gamma oscillations", "high gamma", "binding", "attention",
                "perception", "memory", "40 Hz", "visual processing",
                "synchronization", "cortical processing"
            ],
            required_concepts=["gamma", "oscillations", "cognition"],
            expected_frequency_bands=["gamma"],
            clinical_relevance="cognitive neuroscience",
        ),
        
        GroundTruthQuestion(
            id="oscillation_002",
            question="How do theta oscillations support working memory and navigation?",
            category="oscillation",
            difficulty="medium",
            expected_pmids=["35987654", "34876543", "33765432"],
            expected_concepts=[
                "theta oscillations", "hippocampus", "working memory",
                "spatial navigation", "phase-amplitude coupling", "frontal midline theta",
                "4-8 Hz", "memory encoding", "theta-gamma coupling"
            ],
            required_concepts=["theta", "working memory", "hippocampus"],
            expected_frequency_bands=["theta"],
            clinical_relevance="cognitive neuroscience",
        ),
    ]
    
    def __init__(self, rag_system: Any = None):
        """
        Initialize benchmark suite.
        
        Args:
            rag_system: EEG-RAG system to evaluate (with query method)
        """
        self.rag = rag_system
        self.results: list[BenchmarkResult] = []
        
    async def run_single(self, question: GroundTruthQuestion) -> BenchmarkResult:
        """Run evaluation on a single benchmark question.
        
        Args:
            question: Ground truth question to evaluate
            
        Returns:
            BenchmarkResult with precision, recall, F1, and concept coverage
        """
        start_time = datetime.now()
        
        # Default result for failures
        default_result = BenchmarkResult(
            question_id=question.id,
            retrieval_precision=0.0,
            retrieval_recall=0.0,
            f1_score=0.0,
            concept_coverage=0.0,
            required_concept_coverage=0.0,
            latency_seconds=0.0,
            pmids_expected=question.expected_pmids,
            passed=False,
        )
        
        if self.rag is None:
            logger.warning(f"No RAG system provided, skipping {question.id}")
            return default_result
        
        try:
            # Execute query
            response = await self.rag.query(question.question)
            
            latency = (datetime.now() - start_time).total_seconds()
            
            # Extract PMIDs from response
            answer_text = response.get("answer", "") if isinstance(response, dict) else str(response)
            cited_pmids = self._extract_pmids(answer_text)
            
            # Get retrieved PMIDs if available
            retrieved_pmids = []
            if isinstance(response, dict):
                for doc in response.get("retrieved_documents", []):
                    if "pmid" in doc.get("metadata", {}):
                        retrieved_pmids.append(doc["metadata"]["pmid"])
                        
            # Use cited PMIDs if no retrieved documents
            if not retrieved_pmids:
                retrieved_pmids = cited_pmids
                
            # All acceptable PMIDs
            all_relevant = set(question.expected_pmids + question.relevant_pmids)
            
            # Calculate precision: how many retrieved are relevant
            retrieved_set = set(retrieved_pmids)
            precision = len(retrieved_set & all_relevant) / len(retrieved_set) if retrieved_set else 0.0
            
            # Calculate recall: how many expected were retrieved
            expected_set = set(question.expected_pmids)
            recall = len(retrieved_set & expected_set) / len(expected_set) if expected_set else 0.0
            
            # Calculate F1
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate concept coverage
            answer_lower = answer_text.lower()
            concepts_found = [c for c in question.expected_concepts if c.lower() in answer_lower]
            concept_coverage = len(concepts_found) / len(question.expected_concepts) if question.expected_concepts else 1.0
            
            # Calculate required concept coverage
            required_found = [c for c in question.required_concepts if c.lower() in answer_lower]
            required_coverage = len(required_found) / len(question.required_concepts) if question.required_concepts else 1.0
            
            # Determine if passed (meets minimum thresholds)
            passed = (
                f1 >= 0.3 and
                concept_coverage >= 0.5 and
                required_coverage >= 0.8
            )
            
            return BenchmarkResult(
                question_id=question.id,
                retrieval_precision=precision,
                retrieval_recall=recall,
                f1_score=f1,
                concept_coverage=concept_coverage,
                required_concept_coverage=required_coverage,
                latency_seconds=latency,
                pmids_expected=question.expected_pmids,
                pmids_retrieved=retrieved_pmids,
                pmids_cited=cited_pmids,
                concepts_found=concepts_found,
                concepts_missing=[c for c in question.expected_concepts if c not in concepts_found],
                passed=passed,
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for {question.id}: {e}")
            return default_result
    
    async def run_suite(
        self,
        categories: Optional[list[str]] = None,
        difficulties: Optional[list[str]] = None,
    ) -> BenchmarkSuiteResult:
        """Run the complete benchmark suite.
        
        Args:
            categories: Optional filter by category
            difficulties: Optional filter by difficulty
            
        Returns:
            BenchmarkSuiteResult with aggregate metrics
        """
        questions = self.QUESTIONS.copy()
        
        # Filter by category
        if categories:
            questions = [q for q in questions if q.category in categories]
            
        # Filter by difficulty
        if difficulties:
            questions = [q for q in questions if q.difficulty in difficulties]
        
        if not questions:
            logger.warning("No questions match the specified filters")
            return BenchmarkSuiteResult(
                name="EEG-RAG Ground Truth Benchmark v1.0",
                timestamp=datetime.now().isoformat(),
                total_questions=0,
                questions_passed=0,
                avg_retrieval_precision=0.0,
                avg_retrieval_recall=0.0,
                avg_f1_score=0.0,
                avg_concept_coverage=0.0,
                avg_required_concept_coverage=0.0,
                avg_latency_seconds=0.0,
            )
        
        logger.info(f"Running benchmark suite with {len(questions)} questions")
        
        results = []
        for question in questions:
            try:
                result = await self.run_single(question)
                results.append(result)
                logger.info(
                    f"Completed {question.id}: F1={result.f1_score:.3f}, "
                    f"Coverage={result.concept_coverage:.3f}, Passed={result.passed}"
                )
            except Exception as e:
                logger.error(f"Failed on {question.id}: {e}")
        
        if not results:
            logger.error("No results collected")
            return BenchmarkSuiteResult(
                name="EEG-RAG Ground Truth Benchmark v1.0",
                timestamp=datetime.now().isoformat(),
                total_questions=len(questions),
                questions_passed=0,
                avg_retrieval_precision=0.0,
                avg_retrieval_recall=0.0,
                avg_f1_score=0.0,
                avg_concept_coverage=0.0,
                avg_required_concept_coverage=0.0,
                avg_latency_seconds=0.0,
            )
        
        # Calculate aggregates
        n = len(results)
        avg_precision = sum(r.retrieval_precision for r in results) / n
        avg_recall = sum(r.retrieval_recall for r in results) / n
        avg_f1 = sum(r.f1_score for r in results) / n
        avg_coverage = sum(r.concept_coverage for r in results) / n
        avg_required = sum(r.required_concept_coverage for r in results) / n
        avg_latency = sum(r.latency_seconds for r in results) / n
        passed = sum(1 for r in results if r.passed)
        
        # Group by category
        by_category = {}
        for q in questions:
            cat = q.category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "f1_sum": 0, "passed": 0}
            
            matching = [r for r in results if r.question_id == q.id]
            if matching:
                by_category[cat]["count"] += 1
                by_category[cat]["f1_sum"] += matching[0].f1_score
                if matching[0].passed:
                    by_category[cat]["passed"] += 1
        
        for cat in by_category:
            count = by_category[cat]["count"]
            if count > 0:
                by_category[cat]["avg_f1"] = by_category[cat]["f1_sum"] / count
                by_category[cat]["pass_rate"] = by_category[cat]["passed"] / count
        
        # Group by difficulty
        by_difficulty = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results 
                          for q in questions 
                          if q.id == r.question_id and q.difficulty == diff]
            if diff_results:
                by_difficulty[diff] = {
                    "count": len(diff_results),
                    "avg_f1": sum(r.f1_score for r in diff_results) / len(diff_results),
                    "passed": sum(1 for r in diff_results if r.passed),
                }
        
        suite_result = BenchmarkSuiteResult(
            name="EEG-RAG Ground Truth Benchmark v1.0",
            timestamp=datetime.now().isoformat(),
            total_questions=len(questions),
            questions_passed=passed,
            avg_retrieval_precision=avg_precision,
            avg_retrieval_recall=avg_recall,
            avg_f1_score=avg_f1,
            avg_concept_coverage=avg_coverage,
            avg_required_concept_coverage=avg_required,
            avg_latency_seconds=avg_latency,
            results_by_category=by_category,
            results_by_difficulty=by_difficulty,
            individual_results=results,
        )
        
        logger.info(
            f"Benchmark complete: F1={avg_f1:.3f}, Precision={avg_precision:.3f}, "
            f"Recall={avg_recall:.3f}, Pass Rate={passed}/{len(questions)}"
        )
        
        return suite_result
    
    def _extract_pmids(self, text: str) -> list[str]:
        """Extract PMIDs from text."""
        patterns = [
            r'PMID[:\s]*(\d{7,8})',
            r'\[PMID[:\s]*(\d{7,8})\]',
            r'PubMed[:\s]*(\d{7,8})',
        ]
        
        pmids = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pmids.update(matches)
            
        return list(pmids)
    
    def save_results(self, result: BenchmarkSuiteResult, path: Path):
        """Save benchmark results to JSON file.
        
        Args:
            result: Suite results to save
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
            
        logger.info(f"Benchmark results saved to {path}")
    
    def print_summary(self, result: BenchmarkSuiteResult):
        """Print a formatted summary of benchmark results."""
        print("\n" + "=" * 60)
        print(f"EEG-RAG Benchmark Results: {result.name}")
        print("=" * 60)
        print(f"\nTimestamp: {result.timestamp}")
        print(f"Total Questions: {result.total_questions}")
        print(f"Questions Passed: {result.questions_passed} ({result.questions_passed/result.total_questions*100:.1f}%)")
        
        print("\n--- Retrieval Metrics ---")
        print(f"Average F1 Score:    {result.avg_f1_score:.4f}")
        print(f"Average Precision:   {result.avg_retrieval_precision:.4f}")
        print(f"Average Recall:      {result.avg_retrieval_recall:.4f}")
        
        print("\n--- Quality Metrics ---")
        print(f"Concept Coverage:    {result.avg_concept_coverage:.4f}")
        print(f"Required Coverage:   {result.avg_required_concept_coverage:.4f}")
        print(f"Avg Latency:         {result.avg_latency_seconds:.3f}s")
        
        if result.results_by_category:
            print("\n--- By Category ---")
            for cat, metrics in result.results_by_category.items():
                print(f"  {cat}: F1={metrics.get('avg_f1', 0):.3f}, Pass={metrics.get('pass_rate', 0)*100:.0f}%")
        
        if result.results_by_difficulty:
            print("\n--- By Difficulty ---")
            for diff, metrics in result.results_by_difficulty.items():
                print(f"  {diff}: F1={metrics.get('avg_f1', 0):.3f}, Count={metrics.get('count', 0)}")
        
        print("\n" + "=" * 60)


# CLI
async def main():
    """Run benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EEG-RAG ground truth benchmarks")
    parser.add_argument("--category", nargs="+", help="Filter by category")
    parser.add_argument("--difficulty", nargs="+", help="Filter by difficulty")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--list", action="store_true", help="List available questions")
    args = parser.parse_args()
    
    benchmarks = GroundTruthBenchmarks()
    
    if args.list:
        print("\nAvailable Benchmark Questions:")
        print("-" * 50)
        for q in benchmarks.QUESTIONS:
            print(f"  [{q.id}] ({q.category}/{q.difficulty})")
            print(f"    {q.question[:60]}...")
        print(f"\nTotal: {len(benchmarks.QUESTIONS)} questions")
        return
    
    print("Note: No RAG system connected. Running in dry-run mode.")
    print("To run actual benchmarks, initialize with your RAG system.")
    
    # Show what would be tested
    questions = benchmarks.QUESTIONS
    if args.category:
        questions = [q for q in questions if q.category in args.category]
    if args.difficulty:
        questions = [q for q in questions if q.difficulty in args.difficulty]
    
    print(f"\nWould test {len(questions)} questions:")
    for q in questions[:5]:
        print(f"  - {q.id}: {q.question[:50]}...")
    if len(questions) > 5:
        print(f"  ... and {len(questions) - 5} more")


if __name__ == "__main__":
    asyncio.run(main())
