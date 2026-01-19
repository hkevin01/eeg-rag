#!/usr/bin/env python3
"""
EEG-RAG: Enterprise-Grade RAG System for EEG Research

A comprehensive RAG system specifically designed for electroencephalography (EEG) research,
featuring multi-agent orchestration, memory management, and domain-specific optimizations.

## Core Features

### Multi-Agent Architecture
- **Local Agent**: Optimized for corpus-based queries with local embeddings
- **Web Agent**: Real-time web search and recent literature retrieval  
- **Graph Agent**: Knowledge graph construction and relationship mapping
- **Cloud Agent**: Scalable cloud processing for large datasets
- **MCP Agent**: Model Context Protocol integration for external tools

### Advanced Memory System
- Episodic memory for query-response patterns
- Semantic memory for domain knowledge
- Working memory for multi-turn conversations
- Persistent storage with automatic cleanup

### Enterprise Features
- HIPAA-compliant medical data handling
- Audit logging and compliance reporting
- Multi-tenant architecture support
- Role-based access control

### Production-Ready Enhancements
- **RAG Evaluation Framework**: Domain-specific benchmarks and metrics
- **Citation Verification**: PubMed integration and hallucination detection
- **Hybrid Retrieval**: BM25 + dense retrieval for improved recall
- **Query Routing**: Intelligent agent selection based on question type
- **Semantic Chunking**: Boundary-aware chunking for better context

### EEG-Specific Optimizations
- Domain-specific embedding models
- EEG terminology normalization
- Specialized chunking for research papers
- Citation verification and hallucination detection

## Quick Start

```python
from eeg_rag.verification import CitationVerifier
from eeg_rag.retrieval import HybridRetriever
from eeg_rag.core import QueryRouter, SemanticChunker

# Add citation verification
verifier = CitationVerifier(email="your@email.com")

# Set up hybrid retrieval
retriever = HybridRetriever(alpha=0.6)

# Add query routing
router = QueryRouter()

# Add semantic chunking
chunker = SemanticChunker()

# Process documents
chunks = chunker.chunk_text("EEG research content...")
retriever.add_documents([c.text for c in chunks])

# Route and search
routing = router.route_query("What are EEG biomarkers for epilepsy?")
results = retriever.search("EEG epilepsy biomarkers")

# Verify citations
verification = await verifier.verify_citation("12345678")
```

## Installation

```bash
pip install eeg-rag
```
"""

from .__version__ import (
    __author__,
    __description__,
    __license__,
    __version__,
)

# Production-ready components
try:
    from .verification import CitationVerifier, HallucinationDetector
    from .retrieval import HybridRetriever
    from .core import QueryRouter, SemanticChunker
    PRODUCTION_COMPONENTS_AVAILABLE = True
except ImportError:
    PRODUCTION_COMPONENTS_AVAILABLE = False

# Utility modules
try:
    from .utils import get_logger
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]

# Add production components if available
if PRODUCTION_COMPONENTS_AVAILABLE:
    __all__.extend([
        'CitationVerifier',
        'HallucinationDetector',
        'HybridRetriever',
        'QueryRouter',
        'SemanticChunker',
    ])

# Add utilities if available
if UTILS_AVAILABLE:
    __all__.append('get_logger')
