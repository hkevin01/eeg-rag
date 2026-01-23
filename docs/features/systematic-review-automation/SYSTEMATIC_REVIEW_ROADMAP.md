# Systematic Review Automation Roadmap
**Transforming EEG-RAG into a Production-Grade Research Synthesis Tool**

*Created: January 2026*  
*Status: Planning Phase*

## Executive Summary

This document outlines a comprehensive plan to transform EEG-RAG from a basic paper search tool into a production-grade systematic review automation platform that can replicate and update Roy et al. 2019's seminal work on deep learning for EEG analysis.

**Current State**: Basic keyword search with LLM response generation  
**Target State**: Comprehensive systematic review automation with advanced analytics  
**Timeline**: 4 phases over 8-12 weeks  
**Priority**: High - enables real research impact

---

## 1. Roy et al. 2019 Analysis

### What They Did Manually
Roy et al. performed a comprehensive systematic review of 154 deep learning papers for EEG (2010-2018). They extracted **78 structured fields** including:

#### Origin & Context (9 fields)
- Title, Year, Authors, Journal/Origin
- Type of paper, Institution, Country
- Pages, Citation

#### Rationale (10 fields)  
- Domain 1-4 (Epilepsy, Sleep, BCI, etc.)
- High-level Goal, Practical Goal
- Task/Paradigm, Motivation for DL

#### EEG Data (15 fields)
- Hardware, Neural response pattern
- Dataset name, Accessibility
- Number of samples, time, subjects
- Channels, Sampling rate
- Offline/Online

#### Preprocessing (8 fields)
- Preprocessing steps (raw/clean)
- Artifact handling (raw/clean)
- Features (raw/clean)
- Normalization

#### DL Methodology (32 fields)
- Software, Architecture (raw/clean)
- Design peculiarities, EEG-specific design
- Network schema, Input format
- Layers (number/clean), Activation function
- Regularization, Classes, Output format
- Parameters, Training procedure
- Optimizer, Hyperparameters
- Minibatch size, Data augmentation
- Loss function, Cross-validation
- Data split, Performance metrics

#### Results & Analysis (4 fields)
- Results, Benchmarks
- Baseline comparisons
- Statistical analysis
- Code availability

### Key Findings
- 68% used CNNs, 30% RNNs
- 50% used publicly available data
- Shift from intra- to inter-subject approaches
- Median accuracy gain: 5.4% over baselines
- **MAJOR ISSUE**: Poor reproducibility - most papers lack code/data

### What Can Be Automated
✅ **Easy Automation** (Phase 1-2):
- Paper search and filtering
- Basic metadata extraction (title, authors, year)
- Citation network analysis
- Domain classification
- Availability tracking (code/data)
- Trend analysis over time

🟡 **Moderate Automation** (Phase 2-3):
- Architecture type extraction
- Dataset identification
- Performance metric extraction
- Preprocessing step detection
- Hyperparameter extraction
- Comparative analysis

🔴 **Hard Automation** (Phase 3-4):
- Full methodology extraction
- Quality assessment
- Bias detection
- Research gap identification
- Novel insight generation

---

## 2. Current vs. Required Capabilities

### Current EEG-RAG System

#### ✅ What Works
1. **Multi-Source Ingestion**
   - PubMed, Semantic Scholar, arXiv, OpenAlex
   - JSONL output format
   - Metadata extraction (title, authors, abstract, PMID, DOI)

2. **LLM Integration**
   - Mistral 7B via Ollama
   - Local inference (no API costs)
   - Related query suggestions

3. **Basic RAG Pipeline**
   - Keyword-based retrieval
   - Context formatting
   - Response generation with citations

4. **Streamlit UI**
   - Query interface
   - Paper explorer
   - Inline details display

#### ❌ Critical Gaps

1. **Retrieval Quality**
   - ❌ No semantic search (just keyword matching)
   - ❌ No vector embeddings
   - ❌ No query expansion
   - ❌ No re-ranking
   - ❌ Simple TF scoring (no BM25, no hybrid)
   - ❌ No citation-aware chunking

2. **Structured Extraction**
   - ❌ No field extraction templates
   - ❌ No NER for methodologies
   - ❌ No schema validation
   - ❌ No confidence scoring

3. **Analysis & Insights**
   - ❌ No citation network graphs
   - ❌ No temporal trend analysis
   - ❌ No research gap detection
   - ❌ No methodology comparison
   - ❌ No quality assessment
   - ❌ No statistical aggregation

4. **Systematic Review Workflow**
   - ❌ No PRISMA flow tracking
   - ❌ No inclusion/exclusion screening
   - ❌ No duplicate detection
   - ❌ No reviewer collaboration
   - ❌ No version tracking
   - ❌ No export to LaTeX/Word

5. **Data Management**
   - ❌ No structured database
   - ❌ No ground truth validation
   - ❌ No annotation UI
   - ❌ No quality metrics

### Required "Real RAG" Features

#### Core RAG Components
1. **Hybrid Retrieval**
   - Dense: sentence-transformers embeddings (all-MiniLM-L6-v2)
   - Sparse: BM25 keyword matching
   - Fusion: RRF (Reciprocal Rank Fusion)
   - Vector DB: Qdrant or Chroma

2. **Advanced Query Processing**
   - Query expansion with synonyms
   - Multi-hop reasoning
   - Contextual re-ranking
   - Source diversity

3. **Intelligent Chunking**
   - Section-aware splitting (Abstract, Methods, Results)
   - Citation preservation
   - Overlap with context windows

4. **Response Quality**
   - Multi-document synthesis
   - Claim verification
   - Contradiction detection
   - Confidence scores

#### Research-Specific Features
1. **Field Extraction Engine**
   - 78-field schema matching Roy et al.
   - NER for architectures, datasets, metrics
   - Regex patterns for hyperparameters
   - Validation against ontologies

2. **Citation Network**
   - Build citation graphs (NetworkX)
   - Identify influential papers
   - Track methodology evolution
   - Detect research clusters

3. **Trend Analysis**
   - Architecture popularity over time
   - Dataset usage patterns
   - Performance metric evolution
   - Geographic distribution
   - Publication venue analysis

4. **Quality Assessment**
   - Reproducibility scoring (code/data availability)
   - Methodology rigor
   - Statistical analysis quality
   - Sample size adequacy
   - Bias indicators

5. **Comparative Analysis**
   - Benchmark table generation
   - Performance ranking
   - Methodology comparison
   - Dataset comparison
   - Statistical significance testing

---

## 3. Implementation Roadmap

### Phase 1: Advanced Retrieval (Weeks 1-3)
**Goal**: Replace keyword search with production-grade hybrid retrieval

#### 1.1 Vector Database Setup
```python
# Install dependencies
- qdrant-client
- sentence-transformers
- rank-bm25

# Architecture
- Embedding model: all-MiniLM-L6-v2 (384 dims, fast, good quality)
- Vector DB: Qdrant (local mode for development)
- Chunking: 512 tokens with 128 overlap
- Metadata: title, authors, year, PMID, section, chunk_id
```

**Tasks**:
- [  ] Install Qdrant and sentence-transformers
- [  ] Create embedding pipeline for ingested papers
- [  ] Implement section-aware chunking (Abstract, Methods, Results, Discussion)
- [  ] Build hybrid retriever (BM25 + dense)
- [  ] Add reciprocal rank fusion
- [  ] Benchmark retrieval quality (recall@k, MRR)

**Deliverables**:
- `/src/eeg_rag/retrieval/hybrid_retriever_v2.py`
- `/src/eeg_rag/chunking/citation_aware_chunker.py`
- `/tests/test_retrieval_quality.py`

#### 1.2 Query Enhancement
**Tasks**:
- [  ] Query expansion with EEG-specific synonyms
- [  ] Multi-vector queries for complex questions
- [  ] Contextual re-ranking with cross-encoder
- [  ] Query router for routing to best strategy

**Deliverables**:
- `/src/eeg_rag/query/query_expander.py`
- `/src/eeg_rag/query/reranker.py`

#### 1.3 Response Quality
**Tasks**:
- [  ] Multi-document synthesis
- [  ] Citation tracking with source chunks
- [  ] Confidence scoring
- [  ] Contradiction detection

**Success Metrics**:
- Retrieval recall@10 > 90% on test queries
- Average retrieval time < 200ms
- User-rated response quality > 4/5

---

### Phase 2: Structured Extraction (Weeks 3-5)
**Goal**: Automatically extract Roy et al.'s 78 fields from papers

#### 2.1 Extraction Schema
**Tasks**:
- [  ] Define Pydantic models for all 78 fields
- [  ] Create field-specific extraction prompts
- [  ] Build validation rules
- [  ] Add confidence scoring

**Deliverables**:
- `/src/eeg_rag/extraction/roy_schema.py`
- `/src/eeg_rag/extraction/field_extractor.py`

#### 2.2 NER & Pattern Matching
**Tasks**:
- [  ] NER for architectures (CNN, RNN, LSTM, GRU, Transformer)
- [  ] NER for datasets (BCI Competition, TUH EEG, etc.)
- [  ] Regex for hyperparameters (learning rate, batch size, layers)
- [  ] Regex for performance metrics (accuracy, F1, AUC)
- [  ] Ontology validation (compare against known values)

**Deliverables**:
- `/src/eeg_rag/extraction/ner_models.py`
- `/src/eeg_rag/extraction/patterns.py`
- `/data/ontologies/architectures.json`
- `/data/ontologies/datasets.json`

#### 2.3 LLM-Based Extraction
**Tasks**:
- [  ] Prompt engineering for complex fields
- [  ] Few-shot examples from ground truth
- [  ] Structured output with JSON schema
- [  ] Validation and correction loop

**Deliverables**:
- `/src/eeg_rag/extraction/llm_extractor.py`
- `/prompts/field_extraction/` (78 field-specific prompts)

#### 2.4 Validation Pipeline
**Tasks**:
- [  ] Compare against Roy et al. ground truth
- [  ] Calculate extraction accuracy per field
- [  ] Active learning for low-confidence extractions
- [  ] Human-in-the-loop annotation UI

**Success Metrics**:
- Extraction accuracy > 85% on Roy et al. dataset
- Processing speed: < 30 seconds per paper
- Coverage: > 95% of papers have at least 50/78 fields

---

### Phase 3: Analytics & Insights (Weeks 5-8)
**Goal**: Generate research insights automatically

#### 3.1 Citation Network Analysis
**Tasks**:
- [  ] Build citation graph with NetworkX
- [  ] Calculate metrics (PageRank, betweenness centrality)
- [  ] Identify influential papers
- [  ] Detect research communities (Louvain clustering)
- [  ] Track methodology propagation

**Deliverables**:
- `/src/eeg_rag/analytics/citation_network.py`
- `/src/eeg_rag/analytics/influence_metrics.py`
- Interactive network visualization (Plotly)

#### 3.2 Trend Analysis
**Tasks**:
- [  ] Architecture popularity over time
- [  ] Dataset usage patterns
- [  ] Performance improvements by year
- [  ] Geographic distribution
- [  ] Venue analysis (conferences vs journals)

**Deliverables**:
- `/src/eeg_rag/analytics/trend_analyzer.py`
- Dashboard with temporal visualizations

#### 3.3 Comparative Analysis
**Tasks**:
- [  ] Benchmark table generation
- [  ] Performance ranking by domain
- [  ] Statistical significance testing
- [  ] Methodology comparison matrices
- [  ] Dataset comparison

**Deliverables**:
- `/src/eeg_rag/analytics/comparative_analyzer.py`
- Auto-generated LaTeX tables

#### 3.4 Research Gap Detection
**Tasks**:
- [  ] Identify under-studied domains
- [  ] Detect methodological gaps
- [  ] Find under-utilized datasets
- [  ] Suggest future research directions

**Deliverables**:
- `/src/eeg_rag/analytics/gap_detector.py`
- Automated research suggestions

#### 3.5 Quality Assessment
**Tasks**:
- [  ] Reproducibility scoring (code/data availability)
- [  ] Sample size adequacy
- [  ] Statistical analysis quality
- [  ] Methodology rigor scoring
- [  ] Bias detection

**Success Metrics**:
- Generate publication-quality figures
- Replicate Roy et al. findings with 2019 data
- Identify 10+ novel insights from 2019-2026 papers

---

### Phase 4: Systematic Review Workflow (Weeks 8-12)
**Goal**: Full PRISMA-compliant systematic review automation

#### 4.1 Screening Workflow
**Tasks**:
- [  ] Title/abstract screening with inclusion/exclusion criteria
- [  ] Full-text screening
- [  ] Duplicate detection across sources
- [  ] Conflict resolution between reviewers
- [  ] Screening decision tracking

**Deliverables**:
- `/src/eeg_rag/workflow/screening.py`
- Multi-reviewer UI in Streamlit

#### 4.2 PRISMA Flow
**Tasks**:
- [  ] Track papers at each stage (identification, screening, eligibility, inclusion)
- [  ] Exclusion reason categorization
- [  ] Auto-generate PRISMA diagram
- [  ] Export to publication formats

**Deliverables**:
- `/src/eeg_rag/workflow/prisma.py`
- PRISMA diagram generator (Graphviz)

#### 4.3 Data Synthesis
**Tasks**:
- [  ] Meta-analysis support (effect sizes, forest plots)
- [  ] Qualitative synthesis
- [  ] Risk of bias assessment
- [  ] GRADE evidence quality

**Deliverables**:
- `/src/eeg_rag/synthesis/meta_analysis.py`
- Forest plot generation

#### 4.4 Export & Reporting
**Tasks**:
- [  ] LaTeX table generation
- [  ] Word document export
- [  ] Citation export (BibTeX, RIS, EndNote)
- [  ] Supplementary materials
- [  ] Review protocol versioning

**Deliverables**:
- `/src/eeg_rag/export/` module
- Templates for major journals

#### 4.5 Collaboration Features
**Tasks**:
- [  ] Multi-user authentication
- [  ] Review assignments
- [  ] Comment system
- [  ] Version control for review protocol
- [  ] Change tracking

**Success Metrics**:
- Complete systematic review in 1/10th the time
- PRISMA compliance: 100%
- Inter-rater agreement > 90%
- Reproducible results

---

## 4. Architecture Refactoring

### Current Monolith Problem
All code is in a single 2,888-line `app.py` file mixing:
- UI logic
- RAG engine
- Data ingestion
- Benchmarking
- Query routing

### Proposed Clean Architecture

```
src/eeg_rag/
├── core/
│   ├── config.py          # Configuration management
│   ├── types.py           # Shared data models
│   └── exceptions.py      # Custom exceptions
│
├── ingestion/             # ✅ Already modular
│   ├── pubmed.py
│   ├── semantic_scholar.py
│   ├── arxiv.py
│   └── openalex.py
│
├── storage/               # NEW: Data persistence
│   ├── vector_db.py       # Qdrant client
│   ├── document_db.py     # SQLite/PostgreSQL for metadata
│   └── cache.py           # Redis for query cache
│
├── retrieval/             # REFACTOR: Hybrid retrieval
│   ├── embeddings.py      # Sentence-transformers
│   ├── bm25.py            # Sparse retrieval
│   ├── hybrid.py          # Fusion strategy
│   └── reranker.py        # Cross-encoder
│
├── extraction/            # NEW: Structured extraction
│   ├── schema.py          # Roy et al. 78 fields
│   ├── ner_models.py      # Named entity recognition
│   ├── patterns.py        # Regex extractors
│   ├── llm_extractor.py   # LLM-based extraction
│   └── validator.py       # Schema validation
│
├── analytics/             # NEW: Research insights
│   ├── citation_network.py
│   ├── trend_analyzer.py
│   ├── comparative_analyzer.py
│   ├── gap_detector.py
│   └── quality_assessment.py
│
├── workflow/              # NEW: Systematic review
│   ├── screening.py       # Inclusion/exclusion
│   ├── prisma.py          # PRISMA flow
│   └── collaboration.py   # Multi-reviewer
│
├── synthesis/             # NEW: Meta-analysis
│   ├── meta_analysis.py
│   └── qualitative.py
│
├── export/                # NEW: Export formats
│   ├── latex.py
│   ├── word.py
│   └── citations.py
│
├── web_ui/                # REFACTOR: Thin UI layer
│   ├── app.py             # Main Streamlit app (< 500 lines)
│   ├── pages/
│   │   ├── query.py
│   │   ├── explorer.py
│   │   ├── extraction.py
│   │   ├── analytics.py
│   │   ├── screening.py
│   │   └── export.py
│   └── components/        # Reusable UI components
│       ├── paper_card.py
│       ├── citation_graph.py
│       └── metric_charts.py
│
└── evaluation/            # ✅ Already exists
    ├── benchmarks.py
    └── metrics.py
```

### Migration Strategy
1. **Phase 1**: Extract retrieval logic from app.py
2. **Phase 2**: Create extraction module
3. **Phase 3**: Build analytics module
4. **Phase 4**: Add workflow/synthesis modules
5. **Phase 5**: Refactor UI to thin layer

---

## 5. Technology Stack

### Core Infrastructure
- **Vector DB**: Qdrant (local mode → cloud for production)
- **Document DB**: PostgreSQL (metadata, annotations, users)
- **Cache**: Redis (query cache, embeddings)
- **Job Queue**: Celery (long-running extractions)

### ML/NLP
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Mistral 7B (already installed) + GPT-4 for hard extractions
- **NER**: spaCy + custom EEG entity recognizer
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2

### Analytics
- **Graphs**: NetworkX (citation networks)
- **Viz**: Plotly (interactive), Matplotlib (static)
- **Stats**: scipy, statsmodels (meta-analysis)

### UI
- **Framework**: Streamlit (already using)
- **Auth**: streamlit-authenticator
- **Viz**: Plotly, Cytoscape.js (network graphs)

### DevOps
- **Container**: Docker + docker-compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: structlog + ELK stack

---

## 6. Success Metrics

### Retrieval Quality
- Recall@10 > 90%
- MRR > 0.8
- Retrieval time < 200ms

### Extraction Accuracy
- Per-field accuracy > 85% (vs Roy et al. ground truth)
- Coverage > 95% of papers
- Processing speed < 30 seconds/paper

### Research Insights
- Replicate Roy et al. 2019 findings with 2019 data
- Identify 10+ novel insights from 2019-2026 papers
- Generate publication-quality figures

### Workflow Efficiency
- Complete systematic review in 1/10th the time
- PRISMA compliance: 100%
- Inter-rater agreement > 90%

### User Experience
- User-rated quality > 4/5
- Task completion rate > 80%
- Time to first insight < 5 minutes

---

## 7. Immediate Next Steps

### Week 1 Actions
1. **Setup Vector DB**
   ```bash
   # Install dependencies
   pip install qdrant-client sentence-transformers rank-bm25
   
   # Start Qdrant locally
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Create Extraction Schema**
   - Define Pydantic models for 78 Roy et al. fields
   - Document field definitions and validation rules

3. **Benchmark Current System**
   - Measure retrieval quality on test queries
   - Measure response quality (manual evaluation)
   - Document baseline metrics

4. **Refactor App.py**
   - Extract RAGQueryEngine to separate module
   - Create clean API boundaries
   - Add tests

### Quick Wins
These can be implemented immediately while planning longer-term work:

1. **Duplicate Detection** (2 hours)
   - Add DOI/PMID deduplication during ingestion
   - Fuzzy title matching for preprints

2. **Citation Formatting** (1 hour)
   - Generate BibTeX from metadata
   - Add "Export Citations" button

3. **Search Filters** (3 hours)
   - Filter by year range
   - Filter by domain (Epilepsy, Sleep, BCI)
   - Filter by code availability

4. **Paper Quality Indicators** (2 hours)
   - Badge for papers with code
   - Badge for papers with public data
   - Citation count display

---

## 8. Resource Requirements

### Development Time
- **Phase 1**: 3 weeks (1 senior dev)
- **Phase 2**: 2 weeks (1 senior dev + 1 ML eng)
- **Phase 3**: 3 weeks (1 senior dev + 1 data scientist)
- **Phase 4**: 4 weeks (1 senior dev + 1 UX designer)
- **Total**: 8-12 weeks (2-3 person-months)

### Infrastructure
- **Development**: Local machine + free Qdrant/PostgreSQL
- **Production**: ~$50/month (cloud DB, storage)

### Data
- Roy et al. 2019 CSV (164 papers) - ✅ Already have
- Test set for validation (20 papers) - Need to annotate
- Evaluation queries (50 queries) - Need to create

---

## 9. Risks & Mitigation

### Technical Risks
1. **LLM Extraction Accuracy**
   - *Risk*: Mistral 7B may not be accurate enough for 78-field extraction
   - *Mitigation*: Hybrid approach (NER + patterns + LLM), use GPT-4 for hard cases

2. **Vector DB Performance**
   - *Risk*: Slow retrieval with growing corpus
   - *Mitigation*: Start with Qdrant (fast), optimize indexing, add caching

3. **UI Complexity**
   - *Risk*: Too many features makes UI confusing
   - *Mitigation*: Progressive disclosure, user testing, clear workflows

### Research Risks
1. **Ground Truth Quality**
   - *Risk*: Roy et al. may have annotation errors
   - *Mitigation*: Spot-check samples, use inter-annotator agreement

2. **Generalization**
   - *Risk*: System trained on DL papers may not work for other EEG research
   - *Mitigation*: Design modular extractors, support custom schemas

### Project Risks
1. **Scope Creep**
   - *Risk*: Too many features delay delivery
   - *Mitigation*: Strict phase boundaries, MVP mindset

2. **User Adoption**
   - *Risk*: Researchers prefer manual review
   - *Mitigation*: Focus on time savings, generate publication-quality outputs

---

## 10. Conclusion

**Current EEG-RAG is a proof-of-concept**. It demonstrates basic RAG capabilities but lacks the depth required for serious research use.

**The proposed system will**:
1. **Replicate Roy et al. 2019 work automatically** - extract 78 fields from 154 papers
2. **Update to present** - analyze 2019-2026 papers (est. 300+ new papers)
3. **Provide deep insights** - citation networks, trends, gaps, quality assessment
4. **Support all EEG research** - not just deep learning papers
5. **Enable rapid systematic reviews** - PRISMA-compliant workflow in 1/10th the time

**This is a real RAG system** with:
- Hybrid retrieval (not just keyword matching)
- Structured extraction (not just free-text responses)
- Research insights (not just paper search)
- Systematic workflow (not just Q&A)
- Publication-ready outputs (not just chat responses)

**Researchers will get**:
- Comprehensive literature analysis in hours, not months
- Reproducible, version-controlled reviews
- Publication-quality figures and tables
- Novel research insights from automated analysis
- A growing knowledge base that improves over time

This is the path from "demo" to "indispensable research tool."

---

## Appendix A: Roy et al. 78 Fields Reference

<details>
<summary>Click to expand full field list</summary>

### Origin & Context (9 fields)
1. Title
2. Year
3. Authors
4. Journal / Origin
5. Preprint first
6. Type of paper
7. Lab / School / Company
8. Country
9. Citation

### Rationale (10 fields)
10. Domain 1
11. Domain 2
12. Domain 3
13. Domain 4
14. High-level Goal
15. Practical Goal
16. Task/Paradigm
17. Motivation for DL
18. Pages (optional)

### Data (15 fields)
19. EEG Hardware
20. Neural response pattern
21. Dataset name
22. Dataset accessibility
23. Data description
24. Data - samples
25. Data - time
26. Data - subjects
27. Nb Channels
28. Sampling rate
29. Offline / Online

### Preprocessing (8 fields)
30. Preprocessing (raw)
31. Preprocessing (clean)
32. Artefact handling (raw)
33. Artefact handling (clean)
34. Features (raw)
35. Features (clean)
36. Normalization

### Methodology (32 fields)
37. Software
38. Architecture (raw)
39. Architecture (clean)
40. Design peculiarities
41. EEG-specific design
42. Network Schema
43. Input format
44. Layers (number)
45. Layers (clean)
46. Activation function
47. Regularization (raw)
48. Regularization (clean)
49. Nb Classes
50. Classes
51. Output format
52. Nb Parameters
53. Training procedure (raw)
54. Training procedure (clean)
55. Optimizer (raw)
56. Optimizer (clean)
57. Optim parameters
58. Minibatch size
59. Hyperparameter optim (raw)
60. Hyperparameter optim (clean)
61. Data augmentation
62. Loss
63. Intra/Inter subject
64. Cross validation (raw)
65. Cross validation (clean)
66. Data split
67. Performance metrics (raw)
68. Performance metrics (clean)

### Results & Discussion (10 fields)
69. Training hardware
70. Training time
71. Results
72. Benchmarks
73. Baseline model type
74. Statistical analysis of performance
75. Analysis of learned parameters
76. Discussion
77. Limitations
78. Code available

</details>

---

## Appendix B: Example Queries

<details>
<summary>Research questions the system should answer</summary>

### Descriptive Queries
- "What deep learning architectures are most popular for epilepsy detection?"
- "Which EEG datasets are most frequently used in BCI research?"
- "How has the performance of sleep stage classification improved from 2010 to 2025?"

### Comparative Queries
- "Compare CNN vs RNN performance for seizure detection"
- "What's the best preprocessing pipeline for motor imagery classification?"
- "Which optimizer works best for EEG-based emotion recognition?"

### Gap Analysis
- "What EEG domains are under-studied?"
- "Which datasets lack public accessibility?"
- "What methodological approaches haven't been tried for cognitive workload?"

### Methodological Queries
- "Extract all papers using transfer learning for cross-subject BCI"
- "Find papers with sample sizes < 10 subjects"
- "Show me papers without statistical significance testing"

### Quality Queries
- "Which papers provide code and data?"
- "What's the average reproducibility score by publication venue?"
- "Find highly-cited papers with poor methodology"

### Trend Queries
- "How has attention mechanism adoption changed over time?"
- "Show the geographic distribution of EEG deep learning research"
- "Track the evolution of EEGNet citations and derivatives"

</details>

---

**Next**: Discuss priorities and get approval to proceed with Phase 1
