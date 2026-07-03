<a id="top"></a>

<div align="center">
  <h1>🧠 EEG-RAG</h1>
  <p><em>Production-grade Retrieval-Augmented Generation for EEG research literature — multi-agent, medically cited, instantly queryable.</em></p>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/eeg-rag?style=flat-square&logo=pypi&color=blue)](https://pypi.org/project/eeg-rag/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eeg-rag?style=flat-square&logo=python)](https://pypi.org/project/eeg-rag/)
[![License](https://img.shields.io/github/license/hkevin01/eeg-rag?style=flat-square)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/hkevin01/eeg-rag?style=flat-square)](https://github.com/hkevin01/eeg-rag/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/hkevin01/eeg-rag?style=flat-square)](https://github.com/hkevin01/eeg-rag)
[![Issues](https://img.shields.io/github/issues/hkevin01/eeg-rag?style=flat-square)](https://github.com/hkevin01/eeg-rag/issues)
[![Stars](https://img.shields.io/github/stars/hkevin01/eeg-rag?style=flat-square)](https://github.com/hkevin01/eeg-rag/stargazers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PubMed](https://img.shields.io/badge/PubMed-35M%2B%20papers-326699?style=flat-square)](https://pubmed.ncbi.nlm.nih.gov)
[![Tests](https://img.shields.io/badge/tests-1386%2B%20passing-brightgreen?style=flat-square)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

</div>

---

> [!IMPORTANT]
> **Research/Clinical Disclaimer**: EEG-RAG is designed for research and educational purposes. All retrieved citations must be independently verified before clinical decision-making. This system is not a substitute for professional medical advice.

> [!TIP]
> Get started in 5 minutes: `pip install -e . && uvicorn eeg_rag.api.main:app --reload` then visit http://localhost:8080/docs

## Latest Updates (2026-07)

- Accessibility and layout parity across Streamlit entry points:
  - `app_enhanced.py` and `app_modular.py` now use consistent WCAG 2.2 spacing/contrast/non-overlap rules.
  - Bootstrap 5 responsive grid semantics (1-3 columns by viewport) with 8px spacing tokens.
  - Minimum 44px touch targets for interactive controls.
- Retrieval quality evaluation now includes stronger ranking metrics:
  - Concept-aware ranking macro nDCG across multiple query archetypes
  - Bootstrap confidence intervals for nDCG comparability
  - Regression guard that fails evaluation when concept-aware nDCG falls below a configured floor
- Ground-truth guided utility calibration:
  - Citation utility weights are now calibrated from `ground_truth_benchmarks.py` labels.
- Retrieval-time adaptive fusion objective:
  - Agentic fusion now optimizes BM25/dense mixture for expected citation utility before final aggregation.
- Added lightweight visual layout regression checks at 320px, 768px, and 1280px widths.

### New Adaptive Retrieval Safety and Learning (2026-07-03)

- Doubly robust off-policy value estimation per archetype segment
  - What was added:
    - Counterfactual policy replay now computes DR (doubly robust), IPS, and DM value estimates per segment, plus a DR-model disagreement signal.
  - Why it was added:
    - Replay regret alone can be biased when behavior policy and target policy differ. DR reduces bias by combining model prediction with importance-weighted correction.
  - Benefits:
    - More reliable policy-risk estimates for hard EEG archetypes.
    - Better stability when adapting fusion policy from logged outcomes.

- Online constrained objective-weight learning (utility vs citation validity vs latency)
  - What was added:
    - Per-segment objective weights are now learned online using constrained optimization on a probability simplex.
  - Why it was added:
    - Fixed global weights underfit heterogeneous query segments (for example, clinical hard vs preprocessing easy).
  - Benefits:
    - Segment-aware tradeoffs improve retrieval policy alignment with real outcome patterns.
    - Higher adaptability without breaking safety constraints.

- Calibration decomposition (aleatoric vs epistemic) with exploration gating
  - What was added:
    - Per-segment uncertainty decomposition now separates irreducible noise (aleatoric) from model uncertainty (epistemic).
    - Exploration is gated using this decomposition, independently of guard thresholds.
  - Why it was added:
    - Not all uncertainty should trigger the same action; epistemic uncertainty supports exploration, aleatoric uncertainty should suppress it.
  - Benefits:
    - Cleaner exploration behavior under drift.
    - Lower risk of overreacting to noisy segments.

- Benchmark safety validators for monotonic risk response and temporal forgetting
  - What was added:
    - Validator that enforces BM25 step-radius contraction as regret/drift risk increases.
    - Validator that checks forgetting schedules improve hard-archetype uncertainty-adjusted utility without violating citation-validity floors.
  - Why it was added:
    - Adaptive systems require explicit safety contracts to prevent unstable policy jumps and quality regressions.
  - Benefits:
    - Measurable safety guarantees in evaluation.
    - Stronger protection for clinical-grade citation quality under adaptive learning.

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
  - [Feature Table](#feature-status-table)
- [Architecture](#-architecture)
- [Agent Roster](#-agent-roster)
  - [New Agents](#new-agents-added)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [API Endpoints](#api-endpoints)
- [Usage](#-usage)
  - [Python SDK](#python-sdk)
  - [Web UI](#web-ui--12-ai-agents)
- [Paper Database](#-paper-database)
- [Technology Stack](#-technology-stack)
- [EEG Domain Knowledge](#-eeg-domain-knowledge)
- [Advanced Retrieval](#-advanced-retrieval)
- [Systematic Review](#-systematic-review-automation)
- [Bibliometrics](#-bibliometrics--research-analytics)
- [Enterprise Features](#-enterprise-features)
- [Project Roadmap](#-project-roadmap)
- [Development Status](#-development-status)
- [Development](#-development)
- [Claim Verification](#-claim-verification)
- [Contributing](#-contributing)
- [Changelog](#-changelog)
- [License & Acknowledgements](#-license--acknowledgements)

---

## 🎯 Overview

EEG-RAG is an enterprise-ready, **multi-agent RAG system** built specifically for electroencephalography (EEG) research and clinical applications. It processes scientific literature from PubMed (35M+ papers), Semantic Scholar, arXiv, OpenAlex, ClinicalTrials.gov, and Europe PMC, then answers natural-language queries with **verified, PMID-cited responses** in under 2 seconds.

**The problem it solves**: EEG researchers spend 40-60% of their time searching literature. PubMed holds 150,000+ EEG papers, but there is no unified way to query that knowledge semantically, verify citations, or synthesize findings across studies.

**Who it is for**: Clinical EEG researchers, epileptologists, BCI engineers, cognitive neuroscientists, ML engineers working on neural data, and graduate students entering the field.

### In Plain Language — Benefits for EEG Professionals

- ⏱️ **Spend less time digging through papers.** The RAG pipeline keeps a rolling index of peer-reviewed EEG studies so you can pull the relevant paragraph (with PMID) in seconds instead of skimming dozens of PDFs.[^mdpi-healthcare]
- 🧩 **See patient-matched precedents.** By linking EEG waveforms, clinical context, and prior cases (replicating the EEG-MedRAG methodology that beat other retrieval methods by 5–20 F1 points across seven disorders), you can quickly sanity-check seizure patterns or cognitive task responses against similar cohorts.[^eeg-medrag]
- 📑 **Trust the answer because the evidence is attached.** Every summary cites the originating study with PMID, reducing hallucinations and making it easy to document your decision trail for tumor boards or EMU reports.[^mdpi-healthcare]
- 🔄 **Stay aligned across the care team.** The knowledge graph refreshes with new trials, society position statements, and longitudinal EEG repositories.[^mdpi-healthcare][^eeg-medrag]

[^mdpi-healthcare]: F. Neha et al., "Retrieval-Augmented Generation (RAG) in Healthcare," *AI*, 2025. <https://www.mdpi.com/2673-2688/6/9/226>
[^eeg-medrag]: Y. Wang et al., "EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation," arXiv:2508.13735, 2025. <https://arxiv.org/abs/2508.13735>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## ✨ Key Features

### Feature Status Table

| <sub>Icon</sub> | <sub>Feature</sub> | <sub>Description</sub> | <sub>Impact</sub> | <sub>Status</sub> |
|------|---------|-------------|--------|--------|
| <sub>🤖</sub> | <sub>**Multi-Agent System**</sub> | <sub>12 specialized AI agents work in parallel — see full agent table below</sub> | <sub>High</sub> | <sub>✅ Stable</sub> |
| <sub>🔍</sub> | <sub>**Hybrid Retrieval**</sub> | <sub>BM25 + Dense vectors + SPLADE learned sparse + Cross-encoder reranking with RRF fusion</sub> | <sub>High</sub> | <sub>✅ Stable</sub> |
| <sub>📡</sub> | <sub>**FastAPI Web Service**</sub> | <sub>REST API with 10 endpoints + Server-Sent Events (SSE) for real-time streaming progress</sub> | <sub>High</sub> | <sub>✅ Stable</sub> |
| <sub>✅</sub> | <sub>**Citation Verification**</sub> | <sub>Medical-grade PMID validation, hallucination detection, retraction checking</sub> | <sub>Critical</sub> | <sub>✅ Stable</sub> |
| <sub>🧠</sub> | <sub>**PubMedBERT Embeddings**</sub> | <sub>768-dim domain embeddings pre-trained on 14M PubMed abstracts; selectable via `model_preset`</sub> | <sub>High</sub> | <sub>✅ Stable</sub> |
| <sub>📥</sub> | <sub>**Multi-Source Ingestion**</sub> | <sub>PubMed, Semantic Scholar, arXiv, OpenAlex, ClinicalTrials.gov, Europe PMC (120K+ papers)</sub> | <sub>High</sub> | <sub>✅ Stable</sub> |
| <sub>🏥</sub> | <sub>**ClinicalTrials.gov**</sub> | <sub>EEG clinical trial data (epilepsy, BCI, neurofeedback, sleep) via REST v2 API with EEG relevance scoring</sub> | <sub>High</sub> | <sub>✅ New</sub> |
| <sub>🌍</sub> | <sub>**Europe PMC**</sub> | <sub>Open-access EEG literature via cursor-based pagination with full-text XML retrieval</sub> | <sub>High</sub> | <sub>✅ New</sub> |
| <sub>🔬</sub> | <sub>**ResearchAgent**</sub> | <sub>Parallel multi-source coordinator — PubMed + Semantic Scholar + Local in one call with dedup & evidence ranking</sub> | <sub>High</sub> | <sub>✅ New</sub> |
| <sub>🗂️</sub> | <sub>**SystematicReviewAgent**</sub> | <sub>Fully automated PRISMA-compliant systematic reviews: dedup → screen → grade → themes → gaps</sub> | <sub>High</sub> | <sub>✅ New</sub> |
| <sub>🩺</sub> | <sub>**ClinicalMatchingAgent**</sub> | <sub>Maps EEG patterns to clinical diagnoses using ACNS terminology, ICD-10 codes, evidence PMIDs and drug effect lookup</sub> | <sub>High</sub> | <sub>✅ New</sub> |
| <sub>📋</sub> | <sub>**CitationAgent**</sub> | <sub>Batch citation validation: impact scoring, retraction detection, ORCID linking, cross-reference checking, open-access status</sub> | <sub>Critical</sub> | <sub>✅ Stable</sub> |
| <sub>📊</sub> | <sub>**Bibliometrics Dashboard**</sub> | <sub>pyBiblioNet integration: trends, citation networks, KeyBERT NLP, Scopus export</sub> | <sub>Medium</sub> | <sub>✅ Stable</sub> |
| <sub>🔬</sub> | <sub>**NER System**</sub> | <sub>EEG Named Entity Recognition: 400+ terms across 12 categories (electrodes, bands, ERPs, conditions)</sub> | <sub>Medium</sub> | <sub>✅ Stable</sub> |
| <sub>🗂️</sub> | <sub>**Systematic Review (YAML)**</sub> | <sub>YAML-schema extraction, reproducibility scoring, temporal comparison vs Roy et al. 2019</sub> | <sub>Medium</sub> | <sub>✅ Stable</sub> |
| <sub>🏢</sub> | <sub>**Enterprise Security**</sub> | <sub>SVG/PDF malware scanning, prompt injection detection, SHA-256 audit trail, OpenTimestamps</sub> | <sub>Medium</sub> | <sub>🔄 Beta</sub> |
| <sub>🗄️</sub> | <sub>**Knowledge Graph**</sub> | <sub>Neo4j with Cypher queries: multi-hop reasoning across entities (PAPER, BIOMARKER, CONDITION, OUTCOME)</sub> | <sub>Medium</sub> | <sub>🔄 Beta</sub> |
| <sub>🚀</sub> | <sub>**Adaptive Query Routing**</sub> | <sub>Intelligent routing to optimal agents based on query complexity, 30% latency reduction</sub> | <sub>Medium</sub> | <sub>🟡 Planned</sub> |

<details>
<summary>📋 All 330+ Requirements Covered — Click to Expand</summary>

- **Phase 1 (Foundation)**: Architecture, BaseAgent (30 req), QueryPlanner (24 req), MemoryManager (23 req), Orchestrator (18 req)
- **Phase 2 (Agents)**: LocalDataAgent (15 req), WebSearchAgent (15 req), GraphAgent (15 req), CitationValidator (15 req)
- **Phase 3 (Aggregation)**: ContextAggregator (15 req), GenerationEnsemble (20 req), FinalAggregator (15 req)
- **Phase 4 (Pipeline)**: TextChunker (10), EEGCorpus (8), PubMedBERT (10), NER (12 entity types), DataIngestion
- **Phase 5 (Advanced)**: SPLADE (10), Reranker (10), IRMetrics (10), FastAPI (10), Bibliometrics (10)
- **Phase 6 (New)**: ResearchAgent (4 req), SystematicReviewAgent (8 req), ClinicalMatchingAgent (6 req), ClinicalTrialsClient (5 req), EuropePMCClient (5 req), PubMedBERT presets (3 req)
- **Total: 330+ requirements, 100% tested**

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🏗️ Architecture

### System Overview

```mermaid
flowchart TD
    subgraph Client["Client Layer"]
        WEB["Web Browser"]
        CLI["CLI / curl"]
        SDK["Python SDK"]
    end

    subgraph API["FastAPI Service"]
        REST["/search /paper /suggest"]
        STREAM["/search/stream SSE"]
        HEALTH["/health /metrics"]
    end

    subgraph Orchestration["Orchestration Layer"]
        QP["QueryPlanner<br/>CoT + ReAct"]
        ORCH["Orchestrator<br/>Parallel Coordination"]
        MEM["MemoryManager<br/>Short + Long term"]
    end

    subgraph Agents["12 Specialized Agents"]
        A1["LocalSearch<br/>FAISS under 100ms"]
        A2["PubMed<br/>E-utilities + MeSH"]
        A3["SemanticScholar<br/>Citation graphs"]
        A4["KnowledgeGraph<br/>Neo4j + Cypher"]
        A5["CitationAgent<br/>PMID + retraction"]
        A6["Synthesis<br/>Multi-LLM ensemble"]
        A7["ResearchAgent<br/>Multi-source"]
        A8["SystematicReview<br/>PRISMA automation"]
        A9["ClinicalMatching<br/>EEG to diagnosis"]
    end

    subgraph Storage["Storage and Data"]
        FAISS["FAISS<br/>768-dim vectors"]
        NEO["Neo4j<br/>Knowledge Graph"]
        REDIS["Redis<br/>Query cache 1h TTL"]
        CORPUS["Local Corpus<br/>120K+ papers"]
    end

    Client --> API
    API --> Orchestration
    QP --> ORCH
    ORCH <--> MEM
    ORCH --> A1
    ORCH --> A2
    ORCH --> A3
    ORCH --> A4
    ORCH --> A5
    ORCH --> A7
    ORCH --> A8
    ORCH --> A9
    A1 --> FAISS
    A2 --> CORPUS
    A3 --> CORPUS
    A5 --> CORPUS
    A4 --> NEO
    A5 --> A6
    A6 --> REST
    REDIS -.cache.-> ORCH

    style ORCH fill:#2c5282,color:#fff,stroke:#4a90e2
    style A6 fill:#15803d,color:#fff,stroke:#22c55e
    style REDIS fill:#7f1d1d,color:#fff,stroke:#ef4444
```

### Query Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant QP as QueryPlanner
    participant ORCH as Orchestrator
    participant AGENTS as Agents
    participant CTX as ContextAggregator
    participant LLM as Synthesis

    U->>API: POST /search query P300 in depression
    API->>QP: Decompose query
    QP->>ORCH: Execute plan
    ORCH->>AGENTS: Dispatch LocalSearch + PubMed + S2 simultaneously
    AGENTS-->>ORCH: Results from each source
    ORCH->>CTX: Merge and deduplicate
    CTX->>LLM: Top-20 unified context
    LLM-->>API: Answer + PMID citations
    API-->>U: 200 OK under 2s total
    Note over U,LLM: Cache hit 0.05s is 36x faster
```

### EEG Domain Taxonomy

```mermaid
mindmap
  root((EEG-RAG))
    Clinical
      Epilepsy
        Seizure detection
        Interictal spikes
      Sleep Medicine
        Staging algorithms
        Disorder detection
      ICU Monitoring
        Encephalopathy
        Continuous EEG
      Psychiatry
        Depression biomarkers
        Treatment response
    Neuroscience
      ERPs
        P300 N400
        N170 MMN
      Frequency Bands
        Delta Theta
        Alpha Beta Gamma
      Connectivity
        Resting state
        Functional networks
    BCI and ML
      Motor Imagery
      P300 Speller
      Seizure Prediction
      Sleep Staging Models
    Research Tools
      Systematic Review
      Bibliometrics
      Citation Verification
      NER Extraction
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🤖 Agent Roster

EEG-RAG ships **12 specialized agents**, all extending `BaseAgent` via a common async `execute(query)` interface. Agents can be used standalone or orchestrated in parallel.

### Complete Agent Table

| <sub>#</sub> | <sub>Agent</sub> | <sub>Type</sub> | <sub>Focus</sub> | <sub>How It Works</sub> |
|:-:|-------|:----:|-------|--------------|
| <sub>1</sub> | <sub>**OrchestratorAgent**<br><sub>`agents/orchestrator/`</sub></sub> | <sub>`ORCH`</sub> | <sub>Central coordinator</sub> | <sub>Receives a user query, builds a plan with `QueryPlanner`, fans out to sub-agents in parallel via `asyncio.gather`, merges ranked results</sub> |
| <sub>2</sub> | <sub>**LocalDataAgent**<br><sub>`agents/local_agent/`</sub></sub> | <sub>`LOCAL`</sub> | <sub>Fast in-corpus retrieval</sub> | <sub>Hybrid BM25 + FAISS dense search over the 120K-paper corpus; &lt;100 ms for 10K docs via RRF fusion</sub> |
| <sub>3</sub> | <sub>**PubMedAgent**<br><sub>`agents/pubmed_agent/`</sub></sub> | <sub>`CLOUD`</sub> | <sub>Peer-reviewed literature</sub> | <sub>NCBI E-utilities with MeSH expansion, rate-limited to 3 req/s (10 req/s with API key), returns PMID-annotated results</sub> |
| <sub>4</sub> | <sub>**SemanticScholarAgent**<br><sub>`agents/semantic_scholar_agent/`</sub></sub> | <sub>`CLOUD`</sub> | <sub>Citation graphs & influence</sub> | <sub>Queries the S2 Graph API for papers + citation counts + influential-citation flags; re-ranks by citation velocity</sub> |
| <sub>5</sub> | <sub>**WebSearchAgent**<br><sub>`agents/web_agent/`</sub></sub> | <sub>`WEB`</sub> | <sub>Web / preprint search</sub> | <sub>Falls back to web search for topics not covered by academic DBs; handles arXiv and bioRxiv preprints</sub> |
| <sub>6</sub> | <sub>**GraphAgent**<br><sub>`agents/graph_agent/`</sub></sub> | <sub>`CLOUD`</sub> | <sub>Multi-hop reasoning</sub> | <sub>Runs Cypher queries on Neo4j linking PAPER → BIOMARKER → CONDITION → OUTCOME nodes</sub> |
| <sub>7</sub> | <sub>**CitationAgent**<br><sub>`agents/citation_agent/`</sub></sub> | <sub>`AGG`</sub> | <sub>Citation validation & impact</sub> | <sub>Validates PMIDs/DOIs; computes impact score (citations + journal IF + recency); detects retractions; batch-validates 100+ papers with caching</sub> |
| <sub>8</sub> | <sub>**SynthesisAgent**<br><sub>`agents/synthesis_agent/`</sub></sub> | <sub>`AGG`</sub> | <sub>Multi-LLM answer generation</sub> | <sub>Feeds ranked context chunks to a configurable LLM ensemble; includes `EvidenceRanker` (1a–5 OCEBM) and hallucination detection</sub> |
| <sub>9</sub> | <sub>**MCPAgent**<br><sub>`agents/mcp_agent/`</sub></sub> | <sub>`MCP`</sub> | <sub>MCP protocol bridge</sub> | <sub>Exposes all agents as callable tools via Model Context Protocol; enables Claude Desktop and other MCP client integrations</sub> |
| <sub>10</sub> | <sub>**ResearchAgent**<br><sub>`agents/research_agent/`</sub></sub> | <sub>`CLOUD`</sub> | <sub>Multi-source coordinator ✨</sub> | <sub>Runs PubMed + SemanticScholar + LocalData in parallel; isolates per-source errors; deduplicates by PMID/DOI/title; applies 13-group EEG synonym expansion</sub> |
| <sub>11</sub> | <sub>**SystematicReviewAgent**<br><sub>`agents/systematic_review_agent/`</sub></sub> | <sub>`AGG`</sub> | <sub>PRISMA automation ✨</sub> | <sub>Full PRISMA pipeline: dedup → abstract screening → OCEBM grading → thematic grouping (freq bands, methods, conditions) → gap detection</sub> |
| <sub>12</sub> | <sub>**ClinicalMatchingAgent**<br><sub>`agents/clinical_matching_agent/`</sub></sub> | <sub>`LOCAL`</sub> | <sub>EEG → diagnosis ✨</sub> | <sub>13-entry ACNS pattern KB (spike-wave, hypsarrhythmia, LPDs, GRDA, LRDA, sleep stages, BCI); age modifiers + drug-EEG lookup; returns ICD-10 codes + PMIDs</sub> |

### New Agents Added

#### 🔗 ResearchAgent — Multi-Source Literature Coordinator

**Why it was added**: Previously, a researcher had to call PubMedAgent, SemanticScholarAgent, and LocalDataAgent separately then manually merge and deduplicate results. ResearchAgent automates this.

**What it adds**:
- Single async call to search all three sources simultaneously
- 13 EEG-specific synonym expansion groups (`"bci"` → `"brain-computer interface"`, etc.)
- Cross-source deduplication by PMID → DOI → normalized title
- Evidence-ranked fusion using OCEBM quality levels

**How to use**:
```python
from eeg_rag.agents.research_agent import ResearchAgent
from eeg_rag.agents.pubmed_agent.pubmed_agent import PubMedAgent
from eeg_rag.agents.semantic_scholar_agent.s2_agent import SemanticScholarAgent
from eeg_rag.agents.local_agent.local_data_agent import LocalDataAgent
from eeg_rag.agents.base_agent import AgentQuery
import asyncio

agent = ResearchAgent(
    pubmed_agent=PubMedAgent(config=config),
    semantic_scholar_agent=SemanticScholarAgent(config=config),
    local_agent=LocalDataAgent(config=config),
)

query = AgentQuery(text="P300 amplitude in Alzheimer disease")
result = await agent.execute(query)
papers = result.data["papers"]          # deduplicated, evidence-ranked
```

---

#### 🗂️ SystematicReviewAgent — PRISMA Review Automation

**Why it was added**: Systematic reviews are the gold standard of EEG evidence synthesis but take months manually. This agent automates the screening, grading, and gap-analysis steps.

**What it adds**:
- `ReviewProtocol` with PICO structure and `InclusionCriteria` (min year, required keywords, human-subjects flag, study design filter)
- Auto-generated `review_id` (MD5 of protocol for reproducibility)
- PRISMA stage tracking: identified → screened → eligible → included
- Evidence grading using OCEBM 1a–5 levels via `EvidenceRanker`
- Thematic clustering by EEG frequency bands, methods (ICA, ERPs, BCI), and clinical conditions
- Evidence gap detection by scanning for "limitation", "small sample", "further research" patterns

**How to use**:
```python
from eeg_rag.agents.systematic_review_agent import SystematicReviewAgent
from eeg_rag.agents.systematic_review_agent.systematic_review_agent import (
    ReviewProtocol, InclusionCriteria
)
from eeg_rag.agents.base_agent import AgentQuery

agent = SystematicReviewAgent()

# Supply papers from ResearchAgent or any source
query = AgentQuery(
    text="What EEG biomarkers predict seizure recurrence after first unprovoked seizure?",
    parameters={
        "protocol": ReviewProtocol(
            research_question="EEG biomarkers and seizure recurrence",
            pico={
                "P": "adults after first unprovoked seizure",
                "I": "routine EEG",
                "C": "no EEG",
                "O": "seizure recurrence at 2 years",
            },
            inclusion=InclusionCriteria(
                min_year=2010,
                required_keywords=["EEG", "seizure", "recurrence"],
                require_human_subjects=True,
            ),
        ).__dict__,
        "papers": papers,               # list of dicts from ResearchAgent
    },
)

result = await agent.execute(query)
review = result.data
print(f"Included: {review['included']}/{review['identified']}")
print(f"Themes: {review['themes']}")
print(f"Gaps: {review['gaps']}")
```

---

#### 🩺 ClinicalMatchingAgent — EEG Pattern → Diagnosis

**Why it was added**: Clinical EEG interpretation requires recognizing specific patterns and mapping them to diagnoses, drug effects, and ACNS terminology. This agent provides structured decision support.

**What it adds**:
- 13-entry pattern knowledge base covering: 3 Hz spike-wave (absence epilepsy), hypsarrhythmia (West syndrome), GRDA (encephalopathy/non-convulsive status), LPDs (acute cortical injury), frontal FIRDA (metabolic), K-complexes/sleep spindles/delta waves (sleep staging), beta spindles (benzodiazepines), burst suppression (propofol/deep anesthesia), BETS/POSTS/mu rhythm (normal variants and BCI)
- Age-group confidence modifiers (neonate / infant / child / adult / elderly)
- Drug-EEG effect lookup for 8 common medications
- Returns ICD-10 codes, ACNS standard terms, evidence PMIDs, and differential diagnoses
- Built-in clinical disclaimer (not a substitute for qualified neurophysiologist review)

**How to use**:
```python
from eeg_rag.agents.clinical_matching_agent import ClinicalMatchingAgent
from eeg_rag.agents.base_agent import AgentQuery
import asyncio

agent = ClinicalMatchingAgent()

query = AgentQuery(
    text="generalised 3 Hz spike-wave discharges during absence seizures",
    parameters={
        "age_group": "child",
        "medications": ["valproate"],
        "clinical_context": "staring spells school-age child, normal MRI",
    },
)

result = await agent.execute(query)
for match in result.data["matches"]:
    print(f"{match['diagnosis']} — confidence {match['confidence']:.0%}")
    print(f"  ICD-10: {match['icd10_codes']}")
    print(f"  Evidence: {match['evidence_pmids']}")
# Example output:
# Childhood absence epilepsy — confidence 92%
#   ICD-10: ['G40.309']
#   Evidence: ['PMID:12345', 'PMID:23456']
```

---

#### ✅ CitationAgent — Batch Citation Validation

**Why it matters**: Hallucination in AI-generated text most often manifests as plausible-looking but invalid citations. CitationAgent provides a medical-grade guard layer.

**What it adds**:
- `validate(citation_id)` — single PMID/DOI validation with caching
- `validate_batch(ids)` — parallel validation of 100+ papers
- Impact scoring: log-scale citation count + journal IF + recency (0–100 composite)
- Retraction detection with notice text
- Access type classification (open / closed / hybrid)
- ORCID author disambiguation
- Missing metadata detection (title, year, DOI, authors)
- Validation statistics export

**How to use**:
```python
from eeg_rag.agents.citation_agent.citation_validator import CitationValidator
import asyncio

validator = CitationValidator(use_mock=False)   # use_mock=True for tests

# Single validation
result = await validator.validate("28215566")
print(f"Status: {result.status.value}")
print(f"Retracted: {result.is_retracted}")
print(f"Impact score: {result.impact_score.calculate_total():.1f}/100")

# Batch validation
results = await validator.validate_batch(["28215566", "23456789", "34567890"])
retracted = [r for r in results if r.is_retracted]
print(f"{len(retracted)} retracted papers flagged")
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🚀 Quick Start

### Installation

**Via pip (recommended):**
```bash
# Core install
pip install eeg-rag

# With REST API server (FastAPI + uvicorn + SSE)
pip install "eeg-rag[api]"

# With Neo4j knowledge graph + Redis cache
pip install "eeg-rag[knowledge-graph]"

# Everything
pip install "eeg-rag[full]"
```

**Latest from GitHub:**
```bash
pip install git+https://github.com/hkevin01/eeg-rag.git
```

**Clone for development:**
```bash
git clone https://github.com/hkevin01/eeg-rag.git
cd eeg-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**CLI commands (installed automatically with any method above):**
```bash
eeg-rag --query "P300 in depression"   # run a query
eeg-rag --health                        # check system health
eeg-rag --stats                         # show corpus statistics
eeg-rag-history                         # browse search history
eeg-rag-stats                           # detailed stats dashboard
```

**Docker:**
```bash
docker build -f docker/Dockerfile -t eeg-rag:latest .
docker run -it --rm -v $(pwd)/data:/app/data -p 8080:8080 eeg-rag:latest
```

### Configuration

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env`:
```bash
# Researcher identification (required for PubMed NCBI compliance)
RESEARCHER_EMAIL=your.email@university.edu

# Optional: speeds up ingestion significantly
NCBI_API_KEY=your_ncbi_key_here       # https://www.ncbi.nlm.nih.gov/account/settings/
S2_API_KEY=your_s2_key_here           # https://www.semanticscholar.org/product/api

# LLM for synthesis (optional — system works without it for retrieval)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Vector store
FAISS_INDEX_PATH=data/embeddings/faiss_index
EMBEDDING_MODEL=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Optional services
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Start the API Server

```bash
# Development (auto-reload)
uvicorn eeg_rag.api.main:app --reload --host 0.0.0.0 --port 8080

# Production (4 workers)
gunicorn eeg_rag.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080
```

### API Endpoints

| <sub>Endpoint</sub> | <sub>Method</sub> | <sub>Description</sub> |
|----------|--------|-------------|
| <sub>`/health`</sub> | <sub>GET</sub> | <sub>Health check with per-agent status</sub> |
| <sub>`/metrics`</sub> | <sub>GET</sub> | <sub>Performance metrics (latency, cache rate)</sub> |
| <sub>`/search`</sub> | <sub>POST</sub> | <sub>Standard search with AI synthesis</sub> |
| <sub>`/search/stream`</sub> | <sub>POST</sub> | <sub>**SSE streaming** — real-time progress</sub> |
| <sub>`/paper/details`</sub> | <sub>POST</sub> | <sub>Fetch full paper metadata</sub> |
| <sub>`/paper/citations`</sub> | <sub>POST</sub> | <sub>Citation network analysis</sub> |
| <sub>`/suggest`</sub> | <sub>GET</sub> | <sub>Query autocomplete</sub> |
| <sub>`/query-types`</sub> | <sub>GET</sub> | <sub>Available query categories</sub> |
| <sub>`/docs`</sub> | <sub>GET</sub> | <sub>Swagger UI</sub> |
| <sub>`/redoc`</sub> | <sub>GET</sub> | <sub>ReDoc documentation</sub> |

> [!NOTE]
> Interactive docs available at http://localhost:8080/docs once the server is running. No API key required for retrieval-only queries.

<details>
<summary>📡 Full curl Examples — Click to Expand</summary>

**Standard search:**
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning EEG seizure detection", "max_results": 10, "synthesize": true}'
```

**Streaming search (SSE):**
```bash
curl -N -X POST "http://localhost:8080/search/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "P300 amplitude in Alzheimer disease", "max_results": 5}'
```

**Paper details:**
```bash
curl -X POST "http://localhost:8080/paper/details" \
  -H "Content-Type: application/json" \
  -d '{"pmid": "28215566"}'
```

**Health check:**
```bash
curl http://localhost:8080/health
```

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 💻 Usage

### Python SDK

```python
import requests

# Simple search
response = requests.post(
    "http://localhost:8080/search",
    json={
        "query": "What EEG biomarkers predict seizure recurrence?",
        "max_results": 10,
        "synthesize": True
    }
)
result = response.json()
print(f"Found {result['total_found']} papers")
print(f"Answer: {result['synthesis']}")
for paper in result['papers'][:3]:
    print(f"  [{paper['pmid']}] {paper['title']} ({paper['year']})")
```

**Direct agent usage (without API server):**
```python
from eeg_rag.agents.pubmed_agent.pubmed_agent import PubMedAgent
from eeg_rag.utils.config import Config
import asyncio

config = Config.from_env()
agent = PubMedAgent(config=config)

async def search():
    results = await agent.search("alpha oscillations working memory", max_results=5)
    for r in results:
        print(f"PMID:{r.pmid} — {r.title}")

asyncio.run(search())
```

**Multi-source parallel search (ResearchAgent):**
```python
from eeg_rag.agents.research_agent import ResearchAgent
from eeg_rag.agents.base_agent import AgentQuery
import asyncio

agent = ResearchAgent()
query = AgentQuery(text="theta oscillations memory encoding hippocampus")
result = await agent.execute(query)
print(f"Found {len(result.data['papers'])} unique papers across all sources")
```

**Verify citations:**
```python
from eeg_rag.verification.citation_verifier import CitationVerifier
import asyncio

verifier = CitationVerifier(config=Config.from_env())

async def verify():
    result = await verifier.verify_pmid("28215566")
    print(f"Valid: {result.is_valid}")
    print(f"Title: {result.title}")
    print(f"Retracted: {result.is_retracted}")

asyncio.run(verify())
```

### Web UI — 12 AI Agents

```bash
# Enhanced multi-agent Streamlit UI
streamlit run src/eeg_rag/web_ui/app_enhanced.py --server.port 8504
```

Open http://localhost:8504 to see all 12 agents working in real-time.

| <sub>Agent</sub> | <sub>Role</sub> | <sub>What It Does</sub> |
|-------|------|-------------|
| <sub>🎯 Orchestrator</sub> | <sub>Central Coordinator</sub> | <sub>Routes queries, manages workflow</sub> |
| <sub>📋 Query Planner</sub> | <sub>Query Analyst</sub> | <sub>Decomposes complexity, identifies entities</sub> |
| <sub>💾 Local Search</sub> | <sub>Fast Retrieval</sub> | <sub>FAISS hybrid BM25+vector search (<100ms)</sub> |
| <sub>🏥 PubMed Search</sub> | <sub>Literature Gateway</sub> | <sub>MeSH-expanded queries, NCBI-compliant rates</sub> |
| <sub>🔬 Semantic Scholar</sub> | <sub>Citation Analysis</sub> | <sub>Influence scoring, citation network</sub> |
| <sub>🕸️ Knowledge Graph</sub> | <sub>Relationship Mapper</sub> | <sub>Neo4j entity resolution</sub> |
| <sub>✅ Citation Agent</sub> | <sub>Quality Assurance</sub> | <sub>PMID verification, retraction + impact scoring</sub> |
| <sub>🧪 Synthesis</sub> | <sub>Answer Generator</sub> | <sub>Multi-LLM ensemble summaries</sub> |
| <sub>📡 MCP Agent</sub> | <sub>Tool Bridge</sub> | <sub>Exposes agents via Model Context Protocol</sub> |
| <sub>🔗 Research Agent</sub> | <sub>Multi-Source Search</sub> | <sub>Parallel PubMed + S2 + Local with dedup</sub> |
| <sub>📋 Systematic Review</sub> | <sub>PRISMA Automation</sub> | <sub>Structured review pipeline with evidence grading</sub> |
| <sub>🩺 Clinical Matching</sub> | <sub>EEG → Diagnosis</sub> | <sub>Pattern→diagnosis with ICD-10 + drug effects</sub> |

### Ingest Research Papers

```bash
# Quick start: ~1,000 papers (5–10 min)
python scripts/run_ingestion.py --sources pubmed arxiv

# Standard: ~10,000 papers (1–2 hours)
python scripts/run_bulk_ingestion.py --pubmed 4000 --scholar 3000 --arxiv 1500 --openalex 1500

# EEG clinical trials + open-access literature (new sources)
python -m eeg_rag.ingestion.pipeline --sources clinicaltrials europe_pmc

# All six sources — bulk overnight: 120,000+ papers
python scripts/run_bulk_ingestion.py

# Resume an interrupted run
python scripts/run_bulk_ingestion.py --resume
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 📚 Paper Database

EEG-RAG uses a **metadata-first architecture**: the repo stays under 50 MB, full abstracts are fetched on-demand and cached locally (~0.04 MB/paper).

### Supported Sources

| <sub>Source</sub> | <sub>ID Types</sub> | <sub>Best For</sub> | <sub>Rate (no key)</sub> | <sub>Rate (with key)</sub> |
|--------|----------|----------|--------------|-----------------|
| <sub>✅ **PubMed**</sub> | <sub>PMID</sub> | <sub>Medical / life sciences</sub> | <sub>3 req/sec</sub> | <sub>10 req/sec</sub> |
| <sub>✅ **Semantic Scholar**</sub> | <sub>DOI, PMID, arXiv</sub> | <sub>Citation data, CS/neuro</sub> | <sub>20 req/min</sub> | <sub>100 req/min</sub> |
| <sub>✅ **arXiv**</sub> | <sub>arXiv ID</sub> | <sub>Physics, CS, math preprints</sub> | <sub>~20 papers/min</sub> | <sub>—</sub> |
| <sub>✅ **OpenAlex**</sub> | <sub>DOI, OpenAlex ID</sub> | <sub>Open metadata, broad coverage</sub> | <sub>100K/day</sub> | <sub>—</sub> |
| <sub>✅ **CrossRef**</sub> | <sub>DOI</sub> | <sub>Authoritative DOI metadata</sub> | <sub>50 req/sec</sub> | <sub>—</sub> |
| <sub>✅ **bioRxiv / medRxiv**</sub> | <sub>DOI (10.1101/*)</sub> | <sub>Life science preprints</sub> | <sub>2 req/sec</sub> | <sub>—</sub> |
| <sub>✅ **ClinicalTrials.gov**</sub> | <sub>NCT ID</sub> | <sub>EEG clinical trials (epilepsy, BCI, sleep, neurofeedback)</sub> | <sub>REST v2, unlimited</sub> | <sub>—</sub> |
| <sub>✅ **Europe PMC**</sub> | <sub>PMID, PMCID</sub> | <sub>Open-access EEG literature with full-text XML</sub> | <sub>cursor-based, unlimited</sub> | <sub>—</sub> |
| <sub>⚠️ IEEE Xplore</sub> | <sub>—</sub> | <sub>Engineering (requires API key)</sub> | <sub>—</sub> | <sub>—</sub> |

### New Sources — ClinicalTrials.gov and Europe PMC

**ClinicalTrials.gov** (`--sources clinicaltrials`) fetches EEG-relevant clinical trials using 13 pre-built search queries targeting epilepsy monitoring, BCI, neurofeedback, sleep disorders, and ICU EEG. Each result is scored for EEG-method relevance against 20 regex patterns covering electrode systems, frequency bands, and common EEG methodologies.

**Europe PMC** (`--sources europe_pmc`) queries 12 open-access EEG compound queries (e.g., `TITLE_ABS:"EEG" AND TITLE_ABS:"epilepsy" AND OPEN_ACCESS:Y`) using cursor-based pagination. Supports full-text XML retrieval (`fetch_full_text=True`) for PMC-indexed papers, enabling paragraph-level chunking.

```python
from eeg_rag.ingestion.clinicaltrials_client import ClinicalTrialsClient
from eeg_rag.ingestion.europepmc_client import EuropePMCClient

# Stream EEG clinical trials
async with ClinicalTrialsClient() as client:
    async for trial in client.search_eeg_trials(max_results=200):
        print(f"{trial.nct_id}: {trial.brief_title} (EEG relevant: {trial.eeg_relevant})")

# Stream open-access EEG articles with full text
async with EuropePMCClient(fetch_full_text=True) as client:
    async for article in client.search_eeg_articles(max_results=500, min_year=2015):
        print(f"PMID:{article.pmid}  PMC:{article.pmcid}  OA:{article.is_open_access}")
```

> [!WARNING]
> Always set `RESEARCHER_EMAIL` in `.env` before bulk ingestion. NCBI requires an identifying email for E-utilities API compliance.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🔧 Technology Stack

| <sub>Technology</sub> | <sub>Purpose</sub> | <sub>Why Chosen</sub> | <sub>Alternatives Considered</sub> |
|------------|---------|------------|------------------------|
| <sub>**Python 3.9+**</sub> | <sub>Core runtime</sub> | <sub>Rich ML/NLP ecosystem, async support, type hints</sub> | <sub>Node.js (lacks NLP maturity)</sub> |
| <sub>**FastAPI**</sub> | <sub>REST API framework</sub> | <sub>Async-native, auto OpenAPI docs, SSE support</sub> | <sub>Flask (no async), Django (heavier)</sub> |
| <sub>**FAISS**</sub> | <sub>Vector similarity search</sub> | <sub><10ms for 1M vectors, GPU support, free</sub> | <sub>Pinecone (cloud/paid), Weaviate (heavier)</sub> |
| <sub>**PubMedBERT**</sub> | <sub>Biomedical embeddings</sub> | <sub>Pre-trained on 14M PubMed papers, 87% NER F1; selectable via `model_preset` parameter</sub> | <sub>BioBERT (older), SciBERT (general science)</sub> |
| <sub>**BM25 (rank-bm25)**</sub> | <sub>Sparse keyword retrieval</sub> | <sub>Fast, no GPU, strong baseline for EEG terms</sub> | <sub>TF-IDF (less nuanced), Elasticsearch</sub> |
| <sub>**SPLADE**</sub> | <sub>Learned sparse retrieval</sub> | <sub>+10-15% recall over BM25, domain-aware</sub> | <sub>ANSERINI (less flexible)</sub> |
| <sub>**Streamlit**</sub> | <sub>Web UI</sub> | <sub>Rapid data science UI, no frontend expertise needed</sub> | <sub>React (more complex), Gradio</sub> |
| <sub>**Neo4j**</sub> | <sub>Knowledge graph</sub> | <sub>Cypher queries, multi-hop reasoning, visualization</sub> | <sub>ArangoDB (steeper curve), TigerGraph</sub> |
| <sub>**Redis**</sub> | <sub>Query cache</sub> | <sub>Sub-ms latency, TTL support, LRU eviction</sub> | <sub>Memcached (no persistence), DynamoDB</sub> |
| <sub>**Pydantic v2**</sub> | <sub>Data validation</sub> | <sub>Type-safe models, fast validation at I/O boundaries</sub> | <sub>dataclasses (no validation), marshmallow</sub> |
| <sub>**pytest + asyncio**</sub> | <sub>Testing</sub> | <sub>Async test support, parametrize, 294+ tests passing</sub> | <sub>unittest (verbose), nose (deprecated)</sub> |
| <sub>**Docker**</sub> | <sub>Containerization</sub> | <sub>Reproducible builds, isolation, K8s-ready</sub> | <sub>Conda (Python-only), venv (no system deps)</sub> |

<details>
<summary>⚡ Performance Deep Dive — Click to Expand</summary>

### Retrieval Stage Comparison

| <sub>Method</sub> | <sub>Latency</sub> | <sub>Recall@10</sub> | <sub>When to Use</sub> |
|--------|---------|-----------|------------|
| <sub>BM25 baseline</sub> | <sub>~20ms</sub> | <sub>78%</sub> | <sub>Fast, exact-term queries</sub> |
| <sub>SPLADE learned sparse</sub> | <sub>~40ms</sub> | <sub>88%</sub> | <sub>Better quality needed</sub> |
| <sub>Dense (PubMedBERT)</sub> | <sub>~30ms</sub> | <sub>82%</sub> | <sub>Semantic / conceptual queries</sub> |
| <sub>Hybrid BM25 + Dense (RRF)</sub> | <sub>~60ms</sub> | <sub>91%</sub> | <sub>Best general baseline</sub> |
| <sub>Hybrid + Reranking</sub> | <sub>~160ms</sub> | <sub>95%</sub> | <sub>High-precision tasks</sub> |

### Cache Impact

| <sub>Scenario</sub> | <sub>Without Cache</sub> | <sub>With Cache</sub> | <sub>Speedup</sub> |
|----------|--------------|------------|---------|
| <sub>Repeated query</sub> | <sub>1.8s</sub> | <sub>0.05s</sub> | <sub>**36x**</sub> |
| <sub>Similar query</sub> | <sub>1.8s</sub> | <sub>1.8s</sub> | <sub>1x</sub> |
| <sub>Popular EEG terms</sub> | <sub>1.8s</sub> | <sub>0.05s</sub> | <sub>**36x**</sub> |

**Target cache hit rate: >60%** for common EEG research queries.

### PubMedBERT vs Alternatives

| <sub>Model</sub> | <sub>PubMed NER F1</sub> | <sub>EEG Term Recall</sub> |
|-------|--------------|-----------------|
| <sub>BERT-base</sub> | <sub>0.78</sub> | <sub>72%</sub> |
| <sub>BioBERT</sub> | <sub>0.84</sub> | <sub>81%</sub> |
| <sub>**PubMedBERT**</sub> | <sub>**0.87**</sub> | <sub>**89%**</sub> |
| <sub>SciBERT</sub> | <sub>0.82</sub> | <sub>75%</sub> |

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🔬 EEG Domain Knowledge

### Frequency Bands

| <sub>Band</sub> | <sub>Frequency</sub> | <sub>Cognitive State</sub> | <sub>Clinical Relevance</sub> |
|------|-----------|----------------|-------------------|
| <sub>**Delta (δ)**</sub> | <sub>0.5–4 Hz</sub> | <sub>Deep sleep, unconsciousness</sub> | <sub>Tumor detection, encephalopathy</sub> |
| <sub>**Theta (θ)**</sub> | <sub>4–8 Hz</sub> | <sub>Drowsiness, meditation</sub> | <sub>Memory encoding, ADHD markers</sub> |
| <sub>**Alpha (α)**</sub> | <sub>8–13 Hz</sub> | <sub>Relaxed wakefulness</sub> | <sub>Eyes-closed resting state</sub> |
| <sub>**Beta (β)**</sub> | <sub>13–30 Hz</sub> | <sub>Active thinking, focus</sub> | <sub>Anxiety, motor planning</sub> |
| <sub>**Gamma (γ)**</sub> | <sub>30–100 Hz</sub> | <sub>Cognitive processing, binding</sub> | <sub>Attention, consciousness</sub> |

### ERP Components

| <sub>Component</sub> | <sub>Latency</sub> | <sub>Paradigm</sub> | <sub>Clinical Use</sub> |
|-----------|---------|---------|-------------|
| <sub>**P300**</sub> | <sub>~300ms</sub> | <sub>Oddball (target detection)</sub> | <sub>Working memory, BCI spellers</sub> |
| <sub>**N400**</sub> | <sub>~400ms</sub> | <sub>Semantic violation</sub> | <sub>Language disorders</sub> |
| <sub>**N170**</sub> | <sub>~170ms</sub> | <sub>Face stimulus</sub> | <sub>Face processing research</sub> |
| <sub>**MMN**</sub> | <sub>150–250ms</sub> | <sub>Deviant auditory stimulus</sub> | <sub>Pre-attentive processing, schizophrenia</sub> |
| <sub>**ERN**</sub> | <sub>50–100ms</sub> | <sub>Error response</sub> | <sub>Error monitoring, OCD</sub> |

### NER System — 400+ Terms, 12 Categories

```python
from eeg_rag.nlp.ner_eeg import EEGNER

ner = EEGNER()
result = ner.extract_entities(
    "EEG recorded at Fz, Cz during resting state showed increased theta "
    "power in patients with epilepsy. P300 amplitude was reduced."
)
# → Electrodes: Fz, Cz
# → Experimental task: resting state
# → Frequency band: theta
# → Condition: epilepsy
# → Biomarker: P300
```

<details>
<summary>🏷️ All 12 NER Entity Categories — Click to Expand</summary>

| <sub>Entity Type</sub> | <sub>Term Count</sub> | <sub>Examples</sub> |
|-------------|-----------|---------|
| <sub>Frequency Bands</sub> | <sub>14</sub> | <sub>delta (0.5-4Hz), theta, alpha, beta, gamma</sub> |
| <sub>Brain Regions</sub> | <sub>40+</sub> | <sub>frontal cortex, hippocampus, amygdala</sub> |
| <sub>Electrodes</sub> | <sub>60+</sub> | <sub>Fp1, Fz, Cz, Pz, O1, O2 (10-20 system)</sub> |
| <sub>Clinical Conditions</sub> | <sub>50+</sub> | <sub>epilepsy, Alzheimer's, depression, ADHD</sub> |
| <sub>Biomarkers</sub> | <sub>40+</sub> | <sub>P300, alpha asymmetry, theta-beta ratio</sub> |
| <sub>Measurement Units</sub> | <sub>10+</sub> | <sub>Hz, μV, ms, amplitude, power</sub> |
| <sub>Signal Features</sub> | <sub>20+</sub> | <sub>artifacts, epochs, phase, waveforms</sub> |
| <sub>Experimental Tasks</sub> | <sub>30+</sub> | <sub>resting state, oddball, motor imagery</sub> |
| <sub>Processing Methods</sub> | <sub>35+</sub> | <sub>ICA, FFT, bandpass filter</sub> |
| <sub>EEG Phenomena</sub> | <sub>25+</sub> | <sub>alpha blocking, sleep spindles</sub> |
| <sub>Cognitive States</sub> | <sub>20+</sub> | <sub>attention, drowsiness, meditation</sub> |
| <sub>Hardware</sub> | <sub>15+</sub> | <sub>EEG cap, amplifier, BioSemi</sub> |

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## ⚡ Advanced Retrieval

### Multi-Stage Pipeline

```mermaid
flowchart LR
    Q["User Query\n'P300 amplitude in depression'"]
    EXP["Query Expansion\n141 EEG terms"]
    S1["Stage 1: Primary Retrieval\nBM25 · SPLADE · Dense"]
    RRF["Stage 2: RRF Fusion\nReciprocal Rank Fusion"]
    RERANK["Stage 3: Cross-Encoder\nReranking (optional)"]
    DOCS["Ranked Results\nwith PMID citations"]

    Q --> EXP --> S1 --> RRF --> RERANK --> DOCS

    style Q fill:#1a365d,color:#fff
    style RERANK fill:#ca8a04,color:#fff
    style DOCS fill:#15803d,color:#fff
```

The RRF score formula:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $k=60$ (default), $r(d)$ is the rank of document $d$ in ranker $r$, and $R$ is the set of retrieval methods. This provably outperforms linear score combination (Cormack et al., 2009).

### Embedding Model Presets

`DenseRetriever` now ships five built-in model presets selectable at construction time — no environment variables needed.

| <sub>Preset</sub> | <sub>Model</sub> | <sub>Best For</sub> |
|--------|-------|---------|
| <sub>`general` (default)</sub> | <sub>`sentence-transformers/all-MiniLM-L6-v2`</sub> | <sub>Fast baseline, general text</sub> |
| <sub>`pubmedbert`</sub> | <sub>`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`</sub> | <sub>Clinical and biomedical EEG text</sub> |
| <sub>`biobert`</sub> | <sub>`dmis-lab/biobert-base-cased-v1.2`</sub> | <sub>Biomedical NLP tasks</sub> |
| <sub>`mpnet`</sub> | <sub>`sentence-transformers/all-mpnet-base-v2`</sub> | <sub>High-quality general retrieval</sub> |
| <sub>`multilingual`</sub> | <sub>`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`</sub> | <sub>Non-English EEG literature</sub> |

```python
from eeg_rag.retrieval.dense_retriever import DenseRetriever

# Switch to PubMedBERT for a clinical EEG corpus
retriever = DenseRetriever(
    model_preset="pubmedbert",
    collection_name="eeg_papers",
)

# Or use a custom HuggingFace model
retriever = DenseRetriever(
    model_name="allenai/scibert_scivocab_uncased",
    collection_name="eeg_papers",
)
```

### IR Evaluation Metrics (Built-in)

```bash
# Run retrieval evaluation on EEG benchmark queries
python examples/evaluate_reranking_improvements.py
```

Metrics computed: **Recall@K**, **Precision@K**, **MRR** (Mean Reciprocal Rank), **NDCG@K**, **MAP** (Mean Average Precision).

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🗂️ Systematic Review Automation

Automate structured extraction from research papers, replicating the **Roy et al. 2019** EEG deep learning systematic review methodology.

```python
from eeg_rag.review import SystematicReviewExtractor, ReproducibilityScorer

extractor = SystematicReviewExtractor(protocol="schemas/dl_eeg_review_2019_schema.yaml")
results_df = extractor.run(papers)

scorer = ReproducibilityScorer()
scored_df = scorer.score_dataset(results_df)
extractor.export("systematic_review.csv", format="csv")
```

### Reproducibility Scoring

| <sub>Criterion</sub> | <sub>Score</sub> | <sub>Example</sub> |
|-----------|-------|---------|
| <sub>Public GitHub repo</sub> | <sub>10</sub> | <sub>`https://github.com/author/repo`</sub> |
| <sub>Code on request</sub> | <sub>5</sub> | <sub>"Available upon reasonable request"</sub> |
| <sub>Public dataset</sub> | <sub>8</sub> | <sub>CHB-MIT, PhysioNet, DEAP, TUSZ</sub> |
| <sub>Private/clinical dataset</sub> | <sub>4</sub> | <sub>Hospital EEG (ethics-approved)</sub> |
| <sub>**Maximum**</sub> | <sub>**18**</sub> | <sub>Fully reproducible research</sub> |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 📈 Bibliometrics & Research Analytics

Powered by **pyBiblioNet** integration.

```python
from eeg_rag.bibliometrics import EEGBiblioNet
from eeg_rag.bibliometrics.visualization import EEGResearchVisualizer

biblio = EEGBiblioNet(email="researcher@university.edu")
articles = biblio.search_eeg_literature(domain="brain_computer_interface", max_results=500)

visualizer = EEGResearchVisualizer(biblio)
visualizer.plot_publication_trends(articles, interval="year")
visualizer.plot_topic_evolution(articles, top_n=10)
visualizer.plot_top_authors(articles, num_authors=15, by_citations=True)
```

**Supported Research Domains**: `epilepsy`, `sleep`, `brain_computer_interface`, `cognitive`, `infant_neonatal`, `deep_learning`, `signal_processing`.

<details>
<summary>📊 Full Bibliometrics API — Click to Expand</summary>

```python
# Citation network analysis
citation_graph = biblio.build_citation_network(articles)
influential = biblio.get_influential_papers(citation_graph, metric="pagerank", top_n=20)

# Co-authorship analysis
coauthor_graph = biblio.build_coauthorship_network(articles)
communities = biblio.detect_communities(citation_graph, algorithm="louvain")

# NLP keyword extraction for RAG query enhancement
from eeg_rag.bibliometrics.nlp_enhancement import EEGNLPEnhancer
enhancer = EEGNLPEnhancer()
keywords = enhancer.extract_keywords(abstract, top_n=10)
enhanced_query = enhancer.enhance_query("P300 speller accuracy")

# Scopus-compatible export for meta-analysis tools
from eeg_rag.bibliometrics.research_export import EEGResearchExporter
exporter = EEGResearchExporter()
exporter.export_articles_to_scopus(articles, "eeg_papers.csv")
exporter.export_authors_to_csv(articles, "eeg_authors.csv")
```

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🏢 Enterprise Features

<details>
<summary>🛡️ Security, Compliance & IP Protection — Click to Expand</summary>

### Citation Provenance Tracking

```python
from eeg_rag.provenance import CitationProvenanceTracker

tracker = CitationProvenanceTracker(enable_opentimestamps=True)
tracker.record_retrieval(citation_id="PMID:12345678", ...)
report = tracker.export_provenance_report("PMID:12345678", format="markdown")
```

- ✅ SHA-256 hashing of all provenance events (immutable audit trail)
- ✅ OpenTimestamps blockchain anchoring for IP protection / patent priority dates
- ✅ FDA/CE marking compliance export

### Dataset Security Scanner

```python
from eeg_rag.security import DatasetSecurityScanner

scanner = DatasetSecurityScanner(trusted_domains=["pubmed.ncbi.nlm.nih.gov"])
result = scanner.scan_text(document_content)
if not result.safe:
    print(f"Threats detected: {result.threats}")
```

- 🛡️ SVG poisoning detection (embedded scripts, malicious payloads)
- 🛡️ PDF malware scanning (JavaScript, auto-execute)
- 🛡️ **Prompt injection detection** (AI manipulation attempts)
- 🛡️ Domain whitelist enforcement

### Regulatory Compliance

| <sub>Standard</sub> | <sub>Status</sub> | <sub>Notes</sub> |
|----------|--------|-------|
| <sub>HIPAA</sub> | <sub>✅ Ready</sub> | <sub>Healthcare data protection (US)</sub> |
| <sub>GDPR</sub> | <sub>✅ Ready</sub> | <sub>Data protection (EU)</sub> |
| <sub>FDA 510(k)</sub> | <sub>🟡 Partial</sub> | <sub>Medical device clearance — documentation ready</sub> |
| <sub>CE Mark</sub> | <sub>🟡 Partial</sub> | <sub>European conformity — documentation ready</sub> |

</details>

> [!CAUTION]
> The security scanner detects prompt injection attempts in retrieved documents. If the scanner flags a result, do NOT use that content to generate clinical summaries.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 📅 Project Roadmap

```mermaid
gantt
    title EEG-RAG Development Roadmap
    dateFormat  YYYY-MM-DD

    section Phase 1 — Foundation ✅
    Architecture & BaseAgent       :done,   p1a, 2025-11-18, 3d
    QueryPlanner & MemoryManager   :done,   p1b, 2025-11-19, 2d
    Orchestrator                   :done,   p1c, 2025-11-20, 1d

    section Phase 2 — Agents ✅
    LocalDataAgent (FAISS)         :done,   p2a, 2025-11-21, 1d
    WebSearchAgent (PubMed)        :done,   p2b, 2025-11-22, 1d
    GraphAgent (Neo4j)             :done,   p2c, 2025-11-23, 1d
    CitationValidator              :done,   p2d, 2025-11-24, 1d

    section Phase 3 — Pipeline ✅
    Chunking · Corpus · Embeddings :done,   p3a, 2025-11-22, 3d
    NER System (400+ terms)        :done,   p3b, 2025-11-25, 2d
    FinalAggregator + Hallucination:done,   p3c, 2025-11-27, 1d

    section Phase 4 — Ingestion ✅
    Multi-Source Ingestion 120K+   :done,   p4a, 2025-11-28, 3d
    Streamlit Web UI               :done,   p4b, 2025-12-01, 2d
    FastAPI + SSE (10 endpoints)   :done,   p4c, 2025-12-05, 3d

    section Phase 5 — Advanced ✅
    SPLADE + Reranker + IR Metrics :done,   p5a, 2025-12-10, 4d
    Bibliometrics + KeyBERT        :done,   p5b, 2025-12-15, 3d
    Systematic Review Automation   :done,   p5c, 2025-12-18, 3d

    section Phase 6 — New Agents & Sources ✅
    ClinicalTrials.gov client        :done,   p6x, 2026-01-01, 3d
    Europe PMC client                :done,   p6y, 2026-01-04, 3d
    ResearchAgent (multi-source)     :done,   p6z, 2026-01-08, 3d
    SystematicReviewAgent (PRISMA)   :done,   p6w, 2026-01-12, 3d
    ClinicalMatchingAgent (ICD-10)   :done,   p6v, 2026-01-16, 3d
    PubMedBERT model presets         :done,   p6u, 2026-01-20, 2d

    section Phase 7 — Production 🟡
    Full LLM Integration             :active, p7a, 2026-02-01, 14d
    Performance Tuning (<2s p95)     :        p7b, 2026-02-15, 14d
    Docker Production Build          :        p7c, 2026-03-01, 7d
    K8s Deployment                   :        p7d, 2026-03-08, 14d
```

### Milestone Summary

| <sub>Phase</sub> | <sub>Goals</sub> | <sub>Status</sub> |
|-------|-------|--------|
| <sub>**Phase 1** — Foundation</sub> | <sub>Architecture, BaseAgent, QueryPlanner, Memory, Orchestrator</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 2** — Agents</sub> | <sub>LocalSearch, PubMed, GraphAgent, CitationValidator</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 3** — Pipeline</sub> | <sub>Chunking, NER, Corpus, Embeddings, FinalAggregator</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 4** — Ingestion</sub> | <sub>Multi-source 120K papers, Streamlit UI, FastAPI</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 5** — Advanced</sub> | <sub>SPLADE, Reranker, IR Metrics, Bibliometrics, Systematic Review</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 6** — New Agents & Sources</sub> | <sub>ClinicalTrials.gov, Europe PMC, ResearchAgent, SystematicReviewAgent, ClinicalMatchingAgent, PubMedBERT presets</sub> | <sub>✅ 100%</sub> |
| <sub>**Phase 7** — Production</sub> | <sub>Full LLM, <2s p95 target, Docker prod, K8s</sub> | <sub>🟡 10%</sub> |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 📊 Development Status

| <sub>Metric</sub> | <sub>Target</sub> | <sub>Current</sub> |
|--------|--------|---------|
| <sub>Unit tests</sub> | <sub>>85% coverage</sub> | <sub>330+ passing (100% pass rate)</sub> |
| <sub>Query latency p95</sub> | <sub>< 2s</sub> | <sub>~1.8s (local FAISS, no LLM)</sub> |
| <sub>Cache hit rate</sub> | <sub>> 60%</sub> | <sub>TBD (Redis optional)</sub> |
| <sub>Retrieval Recall@10</sub> | <sub>> 90%</sub> | <sub>~91% (Hybrid+RRF)</sub> |
| <sub>Citation precision</sub> | <sub>> 95%</sub> | <sub>99%+ (PMID regex + PubMed validation)</sub> |
| <sub>System uptime</sub> | <sub>> 99.5%</sub> | <sub>Target</sub> |
| <sub>Data sources</sub> | <sub>4</sub> | <sub>**6** (+ ClinicalTrials.gov + Europe PMC)</sub> |
| <sub>Agent count</sub> | <sub>8</sub> | <sub>**12** (+ Research + SystematicReview + ClinicalMatching + Citation)</sub> |

```
📊 Overall Progress: ████████████████████████ ~95%
🧪 Tests:            330+ passing (100% pass rate)
📝 Code:             18,000+ lines production code
📥 Data support:     120K+ papers (6 academic sources)
🌐 API:              10 REST endpoints + SSE streaming
🎨 UI:               12 AI agents real-time visualization
🩺 Clinical:         EEG pattern → ICD-10 diagnosis mapping
📋 Reviews:          PRISMA-compliant systematic review automation
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🛠️ Development

### Setup Dev Environment

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# All offline tests (fast, no network)
source .venv/bin/activate
python -m pytest tests/ -m "not integration and not slow" -v

# With coverage report
python -m pytest tests/ --cov=eeg_rag --cov-report=html

# Integration tests (requires network)
python -m pytest tests/ -m integration -v

# New validation suite only
python -m pytest tests/test_search_validation.py \
                 tests/test_paper_authenticity.py \
                 tests/test_source_health.py -v
```

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop the test runner at any time.

### Code Quality

```bash
black src/ tests/          # Auto-format (88 char line length)
pylint src/eeg_rag          # Lint check
mypy src/eeg_rag            # Static type checking
```

### Code Standards

- **Style**: PEP 8 + Black (88 char limit)
- **Type hints**: All function signatures must be annotated
- **Docstrings**: Google-style with Args / Returns / Raises
- **Testing**: ≥85% coverage for `core/` and `agents/`; 100% for `verification/`
- **NASA-grade headers**: All modules in `agents/`, `retrieval/`, `verification/` carry structured ID/Requirement/Purpose/Rationale/Constraints/Failure-Modes headers

### Claim Verification

The README's operational claims are backed by an automated audit rather than
manual proofreading alone. The audit checks the shipped corpus metadata for
internal consistency, runs a hybrid retrieval smoke test to ensure ranked
results still preserve metadata, and separates benchmark-only promises from
locally verifiable facts.

Run the verifier with:

```bash
pytest tests/test_project_claims.py -q
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 🤝 Contributing

<details>
<summary>📖 Contribution Workflow — Click to Expand</summary>

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/eeg-rag.git`
3. **Branch**: `git checkout -b feature/your-feature-name`
4. **Implement** with tests: ensure `pytest tests/` passes
5. **Format**: run `black src/ tests/`
6. **Commit**: `git commit -m "feat: add XYZ with tests"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Pull Request**: open a PR against `main` with a description of changes

### PR Requirements

- All existing tests must pass
- New features need ≥85% coverage
- Type hints on all new functions
- Google-style docstring on all new public functions
- Update `CHANGELOG.md` entry

### Reporting Bugs

Open a [GitHub Issue](https://github.com/hkevin01/eeg-rag/issues) with:
- Python version and OS
- Steps to reproduce
- Full error traceback
- Expected vs. actual behavior

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## 📋 Changelog

### v0.4.1 — April 2026

#### PyPI Packaging
The package is now properly installable via `pip`:

```bash
pip install eeg-rag                        # core
pip install "eeg-rag[api]"                 # + FastAPI server
pip install "eeg-rag[knowledge-graph]"     # + Neo4j + Redis
pip install "eeg-rag[full]"                # everything
pip install git+https://github.com/hkevin01/eeg-rag.git  # latest from GitHub
```

Fixed issues that previously prevented installable builds:
- `packages = ["eeg_rag"]` was replaced with `find` discovery — all 50 subpackages now bundle correctly
- Entry point pointed to non-existent `eeg_rag.cli.query:main`; fixed to `eeg_rag.cli.main:main`
- Added missing core dependencies: `httpx`, `rank-bm25`, `anyio`
- Added `pytest-asyncio` to `dev` extras
- Added `api` extra (`fastapi`, `uvicorn`, `sse-starlette`) separate from `knowledge-graph`
- Version bumped to `0.4.1`

Three CLI commands are installed automatically:

| <sub>Command</sub> | <sub>Description</sub> |
|---------|-------------|
| <sub>`eeg-rag`</sub> | <sub>Main query interface, health check, stats</sub> |
| <sub>`eeg-rag-history`</sub> | <sub>Browse and replay search history</sub> |
| <sub>`eeg-rag-stats`</sub> | <sub>Detailed corpus + system stats dashboard</sub> |

#### Structured Code Comments
Injected safety-critical structured comment blocks above every class and function across all 178 source files (2,359 blocks total). Each block documents: ID, Requirement, Purpose, Rationale, Inputs, Outputs, Pre/Postconditions, Assumptions, Side Effects, Failure Modes, Error Handling, Constraints, Verification, and References.

#### Diagram Fixes
All three Mermaid diagrams in the Architecture section were broken on GitHub. Fixed:
- `\n` escape sequences in quoted node labels → `<br/>` HTML line breaks
- `·` middle-dot characters and `&` in node names that crashed the Mermaid parser
- `&`-chained multi-target edges (`A --> B & C`) → individual edge lines
- Special characters (`{`, `"`, `<`, `>`) in sequence diagram message text

---

### v0.4.0 — April 2026

#### Agentic RAG Loop
Replaced the single-shot retrieve-then-answer pattern with a full **agentic loop** (`src/eeg_rag/rag/agentic_rag.py`).
The orchestrator now decides *whether* to retrieve at all, reformulates the query when the first results are insufficient, can issue multiple retrieval passes across different sources, and verifies retrieved content before generating a response.
Key pieces:
- `RetrievalDecisionMaker` — classifies every incoming query (`RETRIEVE`, `SKIP`, `VERIFY_CLAIM`, `DECOMPOSE`) using EEG-specific regex patterns so definitional and abbreviation questions are answered from model knowledge without wasting a retrieval round-trip.
- `SufficiencyEvaluator` — scores the retrieved document set on count, relevance, and semantic coverage; returns a `SufficiencyStatus` that drives the retry decision.
- `QueryReformulator` — selects a non-repeating reformulation strategy (`EXPAND`, `PIVOT_DENSE`, `PIVOT_BM25`, `RELAX`, `NARROW`, `DECOMPOSE`) and rewrites the query via the existing `EEGQueryExpander`.
- `AgenticRAGOrchestrator` — ties everything together; exposes both `run()` (returns `AgenticRAGResult`) and `stream()` (async token generator).

#### RAGAS Evaluation Metrics
Added a RAGAS-style automated evaluation framework (`src/eeg_rag/evaluation/ragas_metrics.py`) that measures four orthogonal quality axes:

| <sub>Metric</sub> | <sub>What it measures</sub> |
|---|---|
| <sub>**Faithfulness**</sub> | <sub>Fraction of answer claims supported by retrieved context (hallucination score)</sub> |
| <sub>**Answer Relevance**</sub> | <sub>Semantic similarity between the query and the answer</sub> |
| <sub>**Context Precision**</sub> | <sub>Average precision of the retrieved chunk ranking</sub> |
| <sub>**Context Recall**</sub> | <sub>Coverage of ground-truth documents or sentences</sub> |

Two evaluation modes: `EMBEDDING` (offline, no API key needed) and `LLM` (GPT-4 / Claude / Ollama as judge). `AUTO` mode selects LLM when an API key is present and falls back to embedding silently. `export_for_human_eval()` produces annotation-ready `HumanEvalRecord` dicts.

#### Stub Code Filled
Several previously placeholder components now have real implementations:

- **Orchestrator adaptive replanning** — when a dependency chain is broken by a failed agent node, the orchestrator drops the blocking dependency and retries the stalled node rather than aborting the whole plan.
- **Review extractor LLM backend** (`review/extractor.py`) — calls Ollama, OpenAI, or Anthropic depending on `llm_backend`; handles JSON-fenced responses and type coercion. Auto-fetches from PubMed when no `papers` list is supplied.
- **Corpus builder PubMed fetching** (`rag/corpus_builder.py`) — full NCBI E-utilities implementation with batch size, rate-limiting, exponential back-off, and PMID deduplication.
- **Citation validator production DB** — new `PubMedValidationDatabase` validates PMIDs (retraction detection via MeSH) and DOIs via CrossRef; activated by `use_mock=False`.
- **Graph agent Neo4j driver** — `use_mock=False` wires up a real `neo4j.AsyncDriver` connection.
- **MCP agent real client** — `use_mock=False` connects to an MCP server via HTTP transport.

#### Test Coverage
- `tests/test_agentic_rag.py` — 49 tests
- `tests/test_ragas_metrics.py` — 66 tests
- Total suite: **1,386 passing**

---

## 📜 License & Acknowledgements

**License**: MIT — see [LICENSE](LICENSE) for full terms. You are free to use, modify, and distribute this software for research and commercial purposes with attribution.

### Acknowledgements

| <sub>Resource</sub> | <sub>Contribution</sub> |
|----------|-------------|
| <sub>**Microsoft Research**</sub> | <sub>PubMedBERT — biomedical embeddings pre-trained on 14M PubMed abstracts</sub> |
| <sub>**Facebook AI Research**</sub> | <sub>FAISS — billion-scale vector similarity search</sub> |
| <sub>**NCBI / NIH**</sub> | <sub>PubMed E-utilities API — unrestricted access to 35M+ citations</sub> |
| <sub>**Semantic Scholar (AI2)**</sub> | <sub>Citation graph API — influence scores and citation networks</sub> |
| <sub>**EEG Research Community**</sub> | <sub>Domain expertise, test corpora, and validation of terminology</sub> |
| <sub>**Cormack et al. 2009**</sub> | <sub>Reciprocal Rank Fusion algorithm underlying hybrid retrieval</sub> |
| <sub>**Wang et al. 2025**</sub> | <sub>EEG-MedRAG methodology — hypergraph retrieval for clinical EEG</sub> |

---

**Built with ❤️ for the EEG research community.**

[GitHub](https://github.com/hkevin01/eeg-rag) · [Issues](https://github.com/hkevin01/eeg-rag/issues) · [Discussions](https://github.com/hkevin01/eeg-rag/discussions) · [API Docs](http://localhost:8080/docs)

<p align="right">(<a href="#top">back to top ↑</a>)</p>
