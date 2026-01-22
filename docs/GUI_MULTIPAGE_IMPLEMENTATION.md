# Multi-Page GUI Implementation

## Overview

Restructured the EEG-RAG Streamlit GUI from a monolithic single-file application into a modular multi-page architecture with dedicated pages for testing and benchmarking.

## Architecture

### File Structure

```
src/eeg_rag/web_ui/
├── app.py (original 2,938 lines - deprecated)
├── app_multipage.py (new entry point)
└── pages/
    ├── __init__.py
    ├── systematic_review_page.py (330 lines)
    ├── 1_Testing_Suite.py (270 lines)
    └── 2_Benchmarking.py (420 lines)
```

### Pages Implemented

#### 1. Systematic Review Page (`systematic_review_page.py`)

**Features:**
- **Schema Setup Tab**: Load YAML extraction schemas from `schemas/` directory
- **Extract from Papers Tab**: Manual entry, JSON upload, or corpus selection
- **View Results Tab**: Interactive results viewer with:
  - Summary metrics (papers extracted, fields, confidence, success rate)
  - Field-by-field analysis with distribution charts
  - Full results table
- **Analysis & Export Tab**:
  - Reproducibility scoring with 18-point rubric
  - Category distribution visualization
  - Export to CSV, JSON, Excel

**Integration:**
- Uses `SystematicReviewExtractor` from `eeg_rag.review`
- Uses `ReproducibilityScorer` for quality assessment
- Session state management for results persistence

**Demo Papers:**
- CNN for seizure detection (demo_2023_1)
- Transformer for motor imagery BCI (demo_2024_1)

#### 2. Testing Suite Page (`1_Testing_Suite.py`)

**Features:**
- **Test Selection**: Organized by category
  - 🔍 Retrieval Tests (hybrid_retriever, local_agent)
  - 🤖 Agent Tests (enhanced_agents, graph_agent, mcp_agent, etc.)
  - 📚 Core Tests (query_router, semantic_chunker, common_utils)
  - ✅ Citation & Verification (citation_verifier, context_aggregator)
  - 🔬 Systematic Review (test_systematic_review)
  - 📊 Evaluation (evaluation_comprehensive)
  - 🔄 Integration (integration tests)

- **Test Options**:
  - Verbose output (`-v`)
  - Coverage reporting (`--cov`)
  - Stop on first failure (`-x`)
  - Parallel execution (`-n auto`)

- **Results Display**:
  - Metrics: Passed, Failed, Errors, Skipped, Duration
  - Full test output with color-coding
  - Success/failure indicators

- **Test History**:
  - Last 100 test runs stored in `test_history.json`
  - Historical view of test results

**Running:**
```bash
streamlit run src/eeg_rag/web_ui/pages/1_Testing_Suite.py
```

#### 3. Benchmarking Page (`2_Benchmarking.py`)

**Features:**

**Retrieval Benchmarks Tab:**
- Benchmark types: BM25 vs Dense, Reranking Impact, SPLADE vs Traditional
- IR metrics: MRR, NDCG, MAP, Recall@K
- Dataset selection: Demo, Full Corpus, Custom Queries
- Integration with `scripts/evaluate_reranking_improvements.py`

**Generation Quality Tab:**
- Metrics:
  - Citation Accuracy: PMID reference correctness
  - Factual Consistency: Agreement with sources
  - Completeness: Query aspect coverage
  - Hallucination Rate: False information detection
- LLM backend selection (GPT-4, Claude, Local LLaMA)

**Latency & Performance Tab:**
- Component benchmarks:
  - Retrieval latency: < 100ms target
  - Reranking latency: < 500ms target
  - Generation latency: < 1.5s target
  - End-to-End (p95): < 2s target
- Concurrent user testing (1-50 users)
- p50, p95, p99 percentile tracking

**Historical Trends Tab:**
- Time-series visualization of metrics
- Retrieval quality trends (MRR, NDCG, MAP over time)
- Latency trends (p50, p95, p99 over time)
- Benchmark history stored in `benchmark_history.json`

**Running:**
```bash
streamlit run src/eeg_rag/web_ui/pages/2_Benchmarking.py
```

## Usage

### Starting the Application

**Main Multi-Page App:**
```bash
streamlit run src/eeg_rag/web_ui/app_multipage.py
```

**Individual Pages:**
```bash
# Testing Suite
streamlit run src/eeg_rag/web_ui/pages/1_Testing_Suite.py

# Benchmarking
streamlit run src/eeg_rag/web_ui/pages/2_Benchmarking.py
```

### Streamlit Multi-Page App Conventions

Streamlit automatically creates a multi-page app when:
1. Files are placed in a `pages/` directory
2. Files are prefixed with numbers for ordering (e.g., `1_Testing_Suite.py`, `2_Benchmarking.py`)
3. Each page file has `st.set_page_config()` at the top

Pages appear in the sidebar automatically when running the main app.

## Integration Points

### Systematic Review Module
- `SystematicReviewExtractor`: YAML-based extraction engine
- `ReproducibilityScorer`: 18-point scoring rubric
- `SystematicReviewComparator`: Temporal comparison

### Testing Infrastructure
- pytest integration via `subprocess.run()`
- Test result parsing from pytest output
- Coverage reporting with `pytest-cov`

### Benchmarking System
- IR metrics calculation (MRR, NDCG, MAP)
- Latency measurement with percentile tracking
- Historical trend analysis with Plotly visualizations

## Data Persistence

### Test History (`test_history.json`)
```json
{
  "timestamp": "2024-01-21T16:00:00",
  "tests": ["test_systematic_review.py"],
  "passed": 7,
  "failed": 0,
  "skipped": 0,
  "errors": 0,
  "duration": 12.3,
  "command": "python -m pytest tests/test_systematic_review.py -v"
}
```

### Benchmark History (`benchmark_history.json`)
```json
{
  "timestamp": "2024-01-21T16:00:00",
  "type": "retrieval",
  "bm25_mrr": 0.42,
  "dense_mrr": 0.58,
  "reranked_mrr": 0.73,
  "splade_mrr": 0.68,
  "bm25_ndcg": 0.38,
  "dense_ndcg": 0.54,
  "reranked_ndcg": 0.69,
  "splade_ndcg": 0.61
}
```

## Future Enhancements

### Planned Pages
- **Query System**: Main RAG query interface
- **Data Ingestion**: Paper ingestion and corpus management
- **Corpus Explorer**: Browse and search document collection
- **Settings**: System configuration and API keys

### Enhancements
- **Authentication**: User authentication for multi-user deployment
- **Real-time Updates**: WebSocket integration for live test/benchmark updates
- **Custom Dashboards**: User-defined metric dashboards
- **Export Reports**: PDF/HTML report generation
- **Scheduled Benchmarks**: Automated benchmark runs with alerts

## Dependencies

### Python Packages
- `streamlit>=1.28.0` - Web framework
- `pandas>=1.3.0` - Data manipulation
- `plotly>=5.0.0` - Interactive visualizations
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting

### Internal Modules
- `eeg_rag.review` - Systematic review extraction
- `eeg_rag.agents` - Multi-agent RAG system
- `eeg_rag.retrieval` - Hybrid retrieval
- `eeg_rag.evaluation` - Evaluation metrics

## Maintenance

### Adding New Test Categories
Edit `1_Testing_Suite.py`:
```python
test_categories = {
    "🔍 Retrieval Tests": [...],
    "🆕 New Category": [
        "test_new_module.py",
        "test_another_module.py"
    ]
}
```

### Adding New Benchmark Types
Edit `2_Benchmarking.py`:
```python
benchmark_types = [
    "BM25 vs Dense",
    "Reranking Impact",
    "🆕 New Benchmark Type"
]
```

### Updating Schema Files
Place YAML schema files in `schemas/` directory:
```yaml
name: "DL-EEG Systematic Review 2019"
schema_version: "1.0"
baseline_study: "Roy et al. 2019"
fields:
  - name: architecture_type
    type: enum
    values: [CNN, RNN, LSTM, Transformer]
```

## Performance Considerations

- **Session State**: Uses Streamlit session state for result caching
- **Lazy Loading**: Pages load modules only when needed
- **Background Processes**: Long-running tests/benchmarks run in subprocesses
- **History Limits**: Test/benchmark history limited to last 100 runs

## Deployment

### Local Development
```bash
streamlit run src/eeg_rag/web_ui/app_multipage.py
```

### Production (Docker)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/eeg_rag/web_ui/app_multipage.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo, select `app_multipage.py`
- **AWS/GCP**: Use Docker container with load balancer
- **Kubernetes**: Deploy with Helm chart for scalability

## Summary

✅ **Completed:**
- Multi-page app structure created
- Systematic Review page fully functional
- Testing Suite page with pytest integration
- Benchmarking page with IR metrics and trends
- Modular architecture for easy extension

🔄 **In Progress:**
- Additional core pages (Query, Ingestion, Corpus, Settings)
- Real-time benchmark updates
- Enhanced visualizations

📋 **Todo:**
- Authentication system
- Advanced filtering and search
- Custom report generation
- Scheduled benchmark runs
