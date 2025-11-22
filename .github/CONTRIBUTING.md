# Contributing to EEG-RAG

Thank you for your interest in contributing to EEG-RAG! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a positive environment for all contributors

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/eeg-rag.git
cd eeg-rag
```

### Set Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to verify setup
pytest tests/
```

## Development Workflow

### Branching Strategy

- `main`: Stable production code
- `develop`: Integration branch for features
- `feature/your-feature-name`: New features
- `bugfix/issue-number`: Bug fixes
- `hotfix/critical-fix`: Urgent production fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

## Coding Standards

### Python Style Guide

- **Follow PEP 8** for Python code style
- **Line length**: 88 characters (Black default)
- **Type hints**: Use type hints for all functions
- **Docstrings**: Google-style docstrings

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `EEGRAG`, `VectorStore`)
- **Functions/Methods**: `snake_case` (e.g., `query_rag`, `build_index`)
- **Variables**: `snake_case` (e.g., `chunk_size`, `embedding_dim`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_TOKENS`, `DEFAULT_TOP_K`)
- **Files**: `snake_case.py` (e.g., `vector_store.py`, `pubmed_client.py`)

### Code Example

```python
"""Module for handling EEG data retrieval."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a single retrieval result.

    Attributes:
        text: The retrieved text chunk
        score: Similarity score
        pmid: PubMed ID of source paper
    """

    text: str
    score: float
    pmid: str


def retrieve_chunks(
    query: str, top_k: int = 10, min_score: float = 0.5
) -> List[RetrievalResult]:
    """Retrieve relevant chunks for a given query.

    Args:
        query: The search query
        top_k: Number of results to return
        min_score: Minimum similarity score threshold

    Returns:
        List of RetrievalResult objects

    Raises:
        ValueError: If query is empty or top_k is invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Implementation here
    pass
```

### Code Quality Tools

Run these before committing:

```bash
# Format code
black src/ tests/

# Lint code
pylint src/eeg_rag

# Type check
mypy src/eeg_rag

# Run tests
pytest tests/
```

## Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures for common setup
- Aim for >80% code coverage

### Test Example

```python
import pytest
from eeg_rag.rag.retriever import retrieve_chunks


def test_retrieve_chunks_valid_query():
    """Test retrieval with valid query."""
    results = retrieve_chunks("EEG biomarkers epilepsy", top_k=5)
    assert len(results) <= 5
    assert all(hasattr(r, "score") for r in results)


def test_retrieve_chunks_empty_query():
    """Test retrieval with empty query raises error."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_chunks("", top_k=5)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=eeg_rag --cov-report=html

# Run specific test file
pytest tests/unit/test_retriever.py

# Run tests matching pattern
pytest -k "test_retrieve"
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed. Can span multiple lines and include
    details about the function's behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer

    Example:
        >>> function_name("test", 5)
        True
    """
```

### Updating Documentation

- Update docstrings when changing function signatures
- Update `README.md` for user-facing changes
- Update `docs/project-plan.md` for architectural changes
- Update `memory-bank/change-log.md` for all changes

## Submitting Changes

### Before Submitting

1. **Update your branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout feature/your-feature
   git rebase develop
   ```

2. **Run all checks**
   ```bash
   black src/ tests/
   pylint src/eeg_rag
   mypy src/eeg_rag
   pytest tests/ --cov=eeg_rag
   ```

3. **Update documentation**
   - Add docstrings to new functions
   - Update README if needed
   - Update CHANGELOG

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: Add EEG biomarker extraction

   - Implement NER for EEG terms
   - Add tests for biomarker extraction
   - Update documentation

   Closes #123"
   ```

### Commit Message Format

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### Creating a Pull Request

1. Push your branch to your fork
   ```bash
   git push origin feature/your-feature
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template completely

4. Link related issues

5. Request review from maintainers

### PR Review Process

- Address all review comments
- Keep discussions focused and respectful
- Make requested changes in new commits
- Once approved, a maintainer will merge your PR

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Read the documentation in `docs/`
- Ask in a new issue with the `question` label

Thank you for contributing to EEG-RAG!
