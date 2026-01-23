# Quick Start: Multi-Agent Orchestrator

## Basic Usage

```python
import asyncio
from eeg_rag.orchestrator import Orchestrator

async def main():
    # Initialize
    orchestrator = Orchestrator()
    
    # Search
    result = await orchestrator.search(
        query="deep learning EEG classification",
        max_results=50,
        synthesize=True
    )
    
    # Results
    print(f"Found {result.total_found} papers")
    print(f"Sources: {result.sources_used}")
    print(f"Time: {result.execution_time_ms:.1f}ms")
    
    # Top papers
    for i, paper in enumerate(result.papers[:5], 1):
        print(f"{i}. {paper['title']} ({paper['year']})")
    
    await orchestrator.close()

asyncio.run(main())
```

## With Progress Tracking

```python
def on_progress(stage: str, percent: float):
    print(f"{stage}: {percent*100:.1f}%")

result = await orchestrator.search(
    query="motor imagery BCI",
    progress_callback=on_progress
)
```

## Source Selection

```python
# Only PubMed and Semantic Scholar
result = await orchestrator.search(
    query="EEG seizure detection",
    sources=["pubmed", "s2"],
    date_range=(2020, 2024)
)
```

## Quick Search (one-off)

```python
from eeg_rag.orchestrator import quick_search

result = await quick_search("EEG classification CNN")
```

## Synthesis Results

```python
if result.synthesis:
    print("Key Themes:", result.synthesis['key_themes'])
    print("Research Gaps:", result.synthesis['research_gaps'])
    print("Top Papers:", result.synthesis['top_papers'])
```

## Run Demo

```bash
python examples/demo_orchestrator.py
```

## Run Tests

```bash
pytest tests/test_enhanced_agents_v2.py -v
```

## Query Types Auto-Detected

- **Exploratory**: "EEG signal processing"
- **Comparative**: "Compare CNN vs LSTM"
- **Temporal**: "Recent advances 2023"
- **Author**: "Papers by Yann LeCun"
- **Citation**: "Influential EEG papers"

## Documentation

- [Orchestrator Implementation](docs/ORCHESTRATOR_IMPLEMENTATION.md)
- [Complete Enhancement Summary](docs/AGENT_ENHANCEMENTS_COMPLETE.md)
