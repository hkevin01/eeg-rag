# GitHub Copilot Instructions for EEG-RAG

## Project Context

You are working on **EEG-RAG**, a Retrieval-Augmented Generation system for EEG research literature. This project helps researchers query scientific papers about electroencephalography with natural language and get evidence-based answers with citations.

## Key Architecture

- **RAG Pipeline**: FAISS vector store + PubMedBERT embeddings + LLM generation
- **Knowledge Graph**: Neo4j for entities (PAPER, EEG_BIOMARKER, CONDITION, OUTCOME)
- **Data Sources**: PubMed, arXiv, bioRxiv
- **Tech Stack**: Python 3.9+, FAISS, transformers, Neo4j, Docker

## Coding Standards

### Naming Conventions
- Classes: `PascalCase` (e.g., `EEGRAG`, `VectorStore`)
- Functions/Methods: `snake_case` (e.g., `query_rag`, `build_index`)
- Variables: `snake_case` (e.g., `chunk_size`, `embedding_dim`)
- Constants: `UPPER_CASE` (e.g., `MAX_TOKENS`, `DEFAULT_TOP_K`)
- Files: `snake_case.py` (e.g., `vector_store.py`)

### Code Style
- **Type hints**: Always use type hints
- **Docstrings**: Google-style docstrings for all public functions
- **Line length**: 88 characters (Black default)
- **Imports**: Organize with isort

### Example Function

```python
def retrieve_chunks(
    query: str, top_k: int = 10, min_score: float = 0.5
) -> List[RetrievalResult]:
    """Retrieve relevant chunks for a given query.

    Args:
        query: The search query
        top_k: Number of results to return
        min_score: Minimum similarity score threshold

    Returns:
        List of RetrievalResult objects sorted by score

    Raises:
        ValueError: If query is empty or top_k is invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    # Implementation here
    pass
```

## Error Handling

- Use specific exceptions (ValueError, TypeError, etc.)
- Implement graceful degradation for optional components (Neo4j, Redis)
- Log errors with context
- Add try-except blocks for external API calls with retries

## Testing

- Write tests for all new functions
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures for common setup
- Aim for >80% code coverage

## Performance Considerations

- Add timing measurements for slow operations
- Use batch processing for embeddings
- Implement caching for repeated queries
- Profile memory usage for large datasets

## EEG-Specific Context

### Common Terms
- **ERP Components**: P300, N400, N170, MMN (Mismatch Negativity)
- **Frequency Bands**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (>30 Hz)
- **Conditions**: Epilepsy, sleep disorders, coma, encephalopathy
- **Tasks**: Oddball paradigm, motor imagery, working memory, resting state

### Dataset References
- Sleep-EDF, DEAP, BCI Competition datasets, TUH EEG Corpus

## When Suggesting Code

1. **Prioritize correctness**: Ensure type safety and proper error handling
2. **Be modular**: Break complex functions into smaller, testable pieces
3. **Add docstrings**: Include parameter types and descriptions
4. **Consider edge cases**: Empty inputs, None values, boundary conditions
5. **Think about scale**: Will this work with 10,000+ papers?

## Project Structure

```
src/eeg_rag/
  ├── rag/          # Core RAG system (retriever, generator, vector_store)
  ├── nlp/          # NLP processing (embeddings, NER, chunking)
  ├── knowledge_graph/  # Neo4j client and schema
  ├── biomarkers/   # EEG biomarker analysis
  ├── utils/        # Config, logging, PubMed client
  └── cli/          # Command-line interface
```

## Common Patterns

### Config Management
```python
from eeg_rag.utils.config import Config

config = Config.from_env()
openai_key = config.openai_api_key
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Starting query processing")
```

### Timing
```python
import time

start_time = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - start_time
logger.info(f"Operation completed in {elapsed:.2f}s")
```

## Avoid

- ❌ Magic numbers (use named constants)
- ❌ Printing to stdout (use logging)
- ❌ Hardcoded paths (use config)
- ❌ Catching broad exceptions without reraising
- ❌ Mutable default arguments
- ❌ Not handling None values

## Prefer

- ✅ Type hints and dataclasses
- ✅ Descriptive variable names
- ✅ Early returns for error conditions
- ✅ Context managers for resources
- ✅ List comprehensions over loops (when clear)
- ✅ f-strings for formatting

## When in Doubt

- Check existing code patterns in the repository
- Follow PEP 8 and PEP 257
- Refer to `docs/project-plan.md` for architecture decisions
- Consult `.pylintrc` for project-specific rules

## Remember

This project has real-world impact on EEG research. Prioritize:
1. **Scientific accuracy**: Citations must be correct
2. **Robustness**: Handle errors gracefully
3. **Performance**: Researchers expect fast responses
4. **Maintainability**: Others will work on this code
