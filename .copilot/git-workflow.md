# Git Workflow Standards

## Branch Naming
- feature/short-description (e.g., feature/hybrid-search)
- fix/issue-description (e.g., fix/pmid-validation)
- refactor/component-name (e.g., refactor/chunker)
- test/what-is-tested (e.g., test/citation-verifier)

## Commit Messages
Follow conventional commits:
- feat: add hybrid BM25+dense retrieval
- fix: correct PMID extraction regex
- test: add evaluation benchmark suite
- docs: update architecture diagram
- refactor: simplify agent orchestration
- perf: optimize embedding batch processing

## Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring
- [ ] Documentation
- [ ] Tests

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## EEG Domain Considerations
- [ ] Handles EEG terminology correctly
- [ ] Citation accuracy verified
- [ ] No hallucination risks introduced

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## Pre-commit Hooks
Always run before committing:
```bash
# Format
black src/ tests/

# Lint
pylint src/eeg_rag --fail-under=8.0

# Type check
mypy src/eeg_rag

# Tests
pytest tests/unit -x --tb=short
```