# Multi-Agent Orchestrator Implementation

## Overview

The **Enhanced Multi-Agent Orchestrator** coordinates PubMed, Semantic Scholar, Local Data, and Synthesis agents to provide comprehensive EEG literature search capabilities with intelligent query planning and parallel execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Orchestrator                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │             Query Analyzer & Planner                             │   │
│  │  • Detect query type (comparative, temporal, exploratory, etc.)  │   │
│  │  • Select execution strategy (parallel vs cascading)             │   │
│  │  • Create agent task plan                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                 │                                       │
│         ┌───────────────────────┼───────────────────────┐              │
│         ▼                       ▼                       ▼              │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐        │
│  │LocalAgent   │        │PubMedAgent  │        │  S2Agent    │        │
│  │• Query Exp  │        │• MeSH Exp   │        │• Influence  │        │
│  │• Reranking  │        │• Citations  │        │• Citations  │        │
│  │• Entities   │        │• Related    │        │• Authors    │        │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘        │
│         └───────────────────────┼───────────────────────┘              │
│                                 ▼                                       │
│                        ┌─────────────────┐                             │
│                        │ SynthesisAgent  │                             │
│                        │ • Merge/Dedup   │                             │
│                        │ • Evidence Rank │                             │
│                        │ • Gap Detection │                             │
│                        └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Query Analyzer

Detects query characteristics and determines optimal execution strategy:

- **Query Types**:
  - `EXPLORATORY` - Broad topic exploration
  - `SPECIFIC` - Specific methods/techniques
  - `COMPARATIVE` - Compare approaches
  - `TEMPORAL` - Evolution over time
  - `AUTHOR_FOCUSED` - Works by specific authors
  - `DATASET_FOCUSED` - Papers using datasets
  - `CITATION_NETWORK` - Citation analysis

### 2. Execution Strategies

**Cascading Strategy**:
1. Search local database first
2. If insufficient results (< 80% of requested), expand to external sources
3. Merge and deduplicate results

**Parallel Strategy**:
1. Search all sources simultaneously
2. Wait for all agents to complete
3. Merge results from all sources

### 3. Result Fusion

- **Deduplication**: By title similarity
- **Normalization**: Common paper format across sources
- **Scoring**: Combined relevance scores
- **URL Generation**: DOI, PMID, or S2 paper ID links

## Usage

### Basic Search

```python
from eeg_rag.orchestrator import Orchestrator

orchestrator = Orchestrator(config={
    "email": "researcher@example.com",
    "pubmed_api_key": "your-key",
    "s2_api_key": "your-key"
})

result = await orchestrator.search(
    query="deep learning EEG seizure detection",
    max_results=50,
    synthesize=True
)

print(f"Found {result.total_found} papers from {result.sources_used}")
for paper in result.papers[:5]:
    print(f"- {paper['title']} ({paper['year']})")
```

### Progress Tracking

```python
def on_progress(stage: str, percent: float):
    print(f"{stage}: {percent*100:.1f}%")

result = await orchestrator.search(
    query="motor imagery BCI",
    progress_callback=on_progress
)
```

### Source Selection

```python
# Only search PubMed and Semantic Scholar
result = await orchestrator.search(
    query="EEG analysis",
    sources=["pubmed", "s2"],
    date_range=(2020, 2024)
)
```

### Quick Search (one-off)

```python
from eeg_rag.orchestrator import quick_search

result = await quick_search("EEG classification CNN")
```

## Synthesis Output

When `synthesize=True`, the orchestrator provides:

```python
{
    "summary": "Analysis of 47 papers on...",
    "key_themes": ["Deep Learning", "Signal Processing", ...],
    "research_gaps": [
        "Limited work on graph neural networks...",
        "Cross-subject generalization remains challenging..."
    ],
    "top_papers": [
        {"title": "...", "citations": 1250, "year": 2018},
        ...
    ],
    "evidence_levels": {
        "LEVEL_1A": 5,  # Systematic reviews
        "LEVEL_2": 12,   # RCTs
        ...
    }
}
```

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Query Analysis | <5ms | Pattern matching |
| Local Search | 100-200ms | Vector + BM25 |
| PubMed Search | 500-1000ms | Network latency |
| S2 Search | 400-800ms | Network latency |
| Parallel (all 3) | ~1-2s | Limited by slowest |
| Cascading (local only) | <500ms | If local sufficient |
| Synthesis | 200-500ms | No LLM |

## Error Handling

The orchestrator gracefully handles agent failures:

```python
result = await orchestrator.search(query="test")

if not result.success:
    print(f"Errors: {result.errors}")
else:
    # Partial results still available
    print(f"Got {result.total_found} papers despite {len(result.errors)} errors")
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from eeg_rag.orchestrator import Orchestrator

app = FastAPI()
orchestrator = Orchestrator()

@app.post("/search")
async def search(query: str):
    result = await orchestrator.search(query)
    return {
        "papers": result.papers,
        "synthesis": result.synthesis,
        "metadata": result.metadata
    }
```

### With Streaming

```python
async def stream_search(query: str):
    progress_queue = asyncio.Queue()
    
    def on_progress(stage: str, percent: float):
        asyncio.create_task(progress_queue.put({
            "type": "progress",
            "stage": stage,
            "percent": percent
        }))
    
    # Start search in background
    search_task = asyncio.create_task(
        orchestrator.search(query, progress_callback=on_progress)
    )
    
    # Stream progress updates
    while not search_task.done():
        try:
            event = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            yield event
        except asyncio.TimeoutError:
            pass
    
    # Yield final result
    result = await search_task
    yield {"type": "complete", "result": result}
```

## Configuration

### Environment Variables

```bash
# Optional API keys for higher rate limits
export NCBI_API_KEY="your-pubmed-api-key"
export S2_API_KEY="your-semantic-scholar-api-key"
export RESEARCHER_EMAIL="your@email.com"

# Vector database
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

### Programmatic Config

```python
orchestrator = Orchestrator(config={
    "pubmed_api_key": "...",
    "s2_api_key": "...",
    "email": "researcher@example.com",
    "qdrant_host": "localhost",
    "qdrant_port": 6333
})
```

## Testing

Run the orchestrator demo:

```bash
python examples/demo_orchestrator.py
```

Run orchestrator tests:

```bash
pytest tests/test_orchestrator.py -v
```

## Future Enhancements

- [ ] LLM-powered query rewriting
- [ ] Citation network visualization
- [ ] Cross-source author disambiguation
- [ ] Automatic systematic review generation
- [ ] Real-time collaborative filtering
- [ ] Multi-language support
- [ ] PDF full-text extraction
- [ ] Citation context extraction

## Demo Output Example

```
============================================================
QUERY ANALYSIS DEMO
============================================================

Query: EEG seizure detection deep learning
  Type: exploratory
  Strategy: cascading
  Agents: ['local', 'pubmed', 's2']
  Estimated time: 2100ms

Query: Compare CNN vs LSTM for EEG classification
  Type: comparative
  Strategy: parallel
  Agents: ['local', 'pubmed', 's2']
  Estimated time: 2100ms

============================================================
MOCK SEARCH DEMO
============================================================

Query: deep learning EEG classification
[████████████████████████████████████████] complete     100.0%

Result:
  Success: True
  Papers found: 10
  Sources used: ['pubmed']
  Execution time: 1981.6ms
  Errors: None

  Sample papers:
    1. Deep learning for electroencephalogram (EEG) classification tasks...
       Year: 2019, Citations: 0
```

## API Reference

See [orchestrator.py](../src/eeg_rag/orchestrator/orchestrator.py) for complete API documentation.

## Related Documentation

- [Agent Enhancement Summary](../docs/AGENT_ENHANCEMENTS.md)
- [PubMed Agent](../docs/PUBMED_AGENT.md)
- [Semantic Scholar Agent](../docs/S2_AGENT.md)
- [Synthesis Agent](../docs/SYNTHESIS_AGENT.md)
