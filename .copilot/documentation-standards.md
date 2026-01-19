# Documentation Standards

## Code Documentation
Every module should have:
- Module-level docstring explaining purpose
- Class docstrings with usage examples
- Method docstrings with Args, Returns, Raises
- Inline comments for complex logic only

## README Updates
When adding features, update:
- Feature list in main README
- Architecture diagrams if structure changes
- Quick start examples
- Configuration reference

## Architecture Decision Records
For significant changes, create ADR in docs/adr/:
```markdown
# ADR-XXX: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult because of this change?
```

## API Documentation
Use consistent format for all public interfaces:
```python
def search_documents(
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None,
    include_metadata: bool = True
) -> list[RetrievalResult]:
    """Search indexed documents using semantic similarity.
    
    Performs vector similarity search against the FAISS index using
    PubMedBERT embeddings. Supports filtering by publication year,
    journal, and MeSH terms.
    
    Args:
        query: Natural language search query. Will be embedded using
            PubMedBERT before searching.
        top_k: Maximum number of results to return. Defaults to 10.
        filters: Optional dict of filters. Supported keys:
            - year_min: Minimum publication year (int)
            - year_max: Maximum publication year (int)
            - journals: List of journal names to include
            - mesh_terms: List of MeSH terms to filter by
        include_metadata: Whether to include full metadata in results.
            Set to False for faster retrieval.
    
    Returns:
        List of RetrievalResult objects sorted by relevance score
        (highest first). Each result contains:
            - document_id: Unique identifier
            - content: Text content of the chunk
            - score: Similarity score (0-1)
            - pmid: PubMed ID if available
            - metadata: Additional metadata if requested
    
    Raises:
        IndexNotFoundError: If FAISS index hasn't been built
        EmbeddingError: If query embedding fails
        
    Example:
        >>> retriever = DocumentRetriever()
        >>> results = retriever.search_documents(
        ...     "P300 amplitude in depression",
        ...     top_k=5,
        ...     filters={"year_min": 2020}
        ... )
        >>> for r in results:
        ...     print(f"{r.pmid}: {r.score:.3f}")
    """
```