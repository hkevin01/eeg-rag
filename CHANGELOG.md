# Changelog

All notable changes to EEG-RAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2026-02-14
- Bibliometric RAG integration module (`src/eeg_rag/bibliometrics/rag_integration.py`)
  - `BibliometricEnhancer` class for integrating visualizations into RAG results
  - Automatic chart generation for search results (trends, citations, collaborations)
  - Base64 chart encoding for direct embedding in responses
  - Lazy-loading of bibliometric components for performance
  - `BibliometricInsight` and `EnhancedRAGResult` dataclasses
- Advanced bibliometrics demo script (`examples/demo_bibliometrics_advanced.py`)
  - Comprehensive showcase of visualization features
  - NLP enhancement demonstrations (keyword extraction, topic categorization)
  - Research export metrics and collaboration networks
  - Integrated workflow example combining all components
- Bibliometrics module exports (`src/eeg_rag/bibliometrics/__init__.py`)
  - Export `EEGVisualization`, `ChartResult` from visualization module
  - Export `EEGNLPEnhancer`, `ExtractedKeywords`, `TopicCluster` from NLP module
  - Export `EEGResearchExporter`, `VenueMetrics`, `InstitutionMetrics`, `AuthorProductivity` from export module
  - Export `BibliometricEnhancer`, `BibliometricInsight`, `EnhancedRAGResult` from RAG integration

### Fixed - 2026-02-14
- Test collection errors across multiple modules (7 files total):
  - `cross_encoder_reranker.py`: Changed `log_time` to `PerformanceTimer` context manager
  - `benchmarking.py`: Fixed `WebResearchAgent` import to `WebSearchAgent`
  - `base_agent.py`: Re-exported `QueryComplexity` for backward compatibility
  - `test_semantic_chunker.py`: Removed non-existent `chunk_eeg_paper` import
  - `test_hybrid_retriever.py`: Added `RetrievalResult` alias for `HybridResult`
  - `test_related_searches_integration.py`: Added conditional streamlit import with skipif marker
  - `pyproject.toml`: Added "benchmark" marker definition
- All 1358 tests now collect successfully with 0 errors

### Added
- Comprehensive requirements specification (`docs/requirements/REQUIREMENTS.md`)
  - 40+ requirements with rationale statements
  - Traceability matrix linking requirements to tests and code
  - Coverage of functional, performance, security, and reliability requirements
- Testing requirements document (`docs/testing/TESTING_REQUIREMENTS.md`)
  - Nominal and off-nominal test patterns
  - Boundary condition testing guidelines
  - Integration and performance testing requirements
- Time utilities module (`src/eeg_rag/utils/time_utils.py`)
  - Standardized time units (REQ-TIME-001)
  - High-precision Timer class (REQ-TIME-002)
  - TimingStats for performance tracking (REQ-PERF-002)
  - Time conversion and formatting utilities
  - Latency threshold monitoring
- Error handling module (`src/eeg_rag/utils/error_handling.py`)
  - Standardized error codes (REQ-ERR-002)
  - Custom exception hierarchy (REQ-ERR-001)
  - Safe execution wrappers (REQ-REL-002)
  - Retry decorators with exponential backoff
  - Input validation utilities
- Memory management module (`src/eeg_rag/utils/memory_utils.py`)
  - Memory usage monitoring (REQ-MEM-001)
  - Memory leak detection (REQ-MEM-002)
  - Garbage collection optimization (REQ-MEM-003)
  - Object pooling for resource efficiency
  - Memory health checks
- Persistence utilities module (`src/eeg_rag/utils/persistence_utils.py`)
  - Data persistence operations with retry logic (REQ-DAT-001)
  - Database connection pooling (REQ-DAT-002)
  - Backup and recovery mechanisms (REQ-DAT-003)
  - Data integrity verification with checksums (REQ-REL-003)
  - Atomic file writes
  - Write-ahead logging for crash recovery
  - JSON persistence with automatic backups
- Resilience utilities module (`src/eeg_rag/utils/resilience_utils.py`)
  - Circuit breaker pattern implementation (REQ-REL-001)
  - Health check monitoring system (REQ-REL-002)
  - Token bucket rate limiting (REQ-REL-004)
  - Graceful degradation with feature gates (REQ-REL-005)
  - Thread-safe implementations for all components
  - State change callbacks for monitoring
  - Decorator-based APIs for easy integration
- Bibliometrics integration module (`src/eeg_rag/bibliometrics/`)
  - pyBiblioNet integration for network-based bibliometric analysis (REQ-BIB-001)
  - OpenAlex API integration for EEG article retrieval
  - Citation network construction and analysis (REQ-BIB-002)
  - Co-authorship network analysis (REQ-BIB-003)
  - Centrality metrics (PageRank, betweenness, eigenvector) (REQ-BIB-004)
  - Community detection for research cluster identification
  - Pre-defined EEG research domain query patterns (epilepsy, sleep, BCI, cognitive, clinical, signal processing)
  - RAG integration with filtering and influence scoring (REQ-BIB-005)
  - Article caching for efficient repeated queries
- CI/CD workflow (`.github/workflows/ci.yml`)
  - Automated testing with Python 3.9-3.12
  - Code quality checks (Ruff, Black, mypy)
  - Security scanning (Bandit, Safety)
  - Coverage reporting
  - Build and package automation
- Comprehensive test suites for new modules
  - `tests/test_time_utils.py` - 63 unit tests
  - `tests/test_error_handling.py` - 63 unit tests
  - `tests/test_memory_utils.py` - 39 unit tests
  - `tests/test_persistence_utils.py` - 44 unit tests
  - `tests/test_resilience_utils.py` - 41 unit tests
  - `tests/test_bibliometrics.py` - 44 unit tests
  - Total: 294 tests for utility and bibliometrics modules
  - Tests follow requirement ID conventions

### Changed
- Updated documentation folder structure
  - Added `docs/requirements/` for requirements specifications
  - Added `docs/design/` for design documents
  - Added `docs/testing/` for testing plans and reports

### Documentation
- Added requirement ID comments pattern for code traceability
- Created CHANGELOG.md following Keep a Changelog format

---

## [0.9.0] - 2024-11-24

### Added
- Production readiness features
  - Health check endpoints
  - Metrics collection
  - Graceful shutdown handling
- Enterprise features
  - Multi-tenant support
  - Audit logging
  - Role-based access control
- NER (Named Entity Recognition) for EEG terminology
- Citation verification against PubMed
- Knowledge graph integration

### Changed
- Improved error handling across all agents
- Enhanced logging with structured format
- Optimized retrieval latency

### Fixed
- Memory leaks in embedding cache
- Race conditions in async agent coordination
- Citation extraction regex edge cases

---

## [0.8.0] - 2024-11-20

### Added
- Multi-agent orchestration system
  - Local corpus agent
  - PubMed agent
  - Semantic Scholar agent
  - Graph agent
  - Synthesis agent
- Hybrid retrieval (BM25 + dense vectors)
- Query routing with complexity assessment
- Semantic chunking for documents

### Changed
- Migrated to async/await throughout
- Improved FAISS index management
- Enhanced query expansion for EEG domain

---

## [0.7.0] - 2024-11-15

### Added
- Initial RAG pipeline
- Document ingestion from PDF and text
- BM25 retrieval baseline
- Dense retrieval with sentence-transformers
- Basic citation extraction

### Changed
- Restructured project layout
- Added comprehensive type hints

---

## Version History Legend

| Status       | Description                       |
| ------------ | --------------------------------- |
| 🟢 Added      | New features                      |
| 🔵 Changed    | Changes to existing functionality |
| 🟡 Deprecated | Features to be removed in future  |
| 🔴 Removed    | Features removed in this version  |
| 🟠 Fixed      | Bug fixes                         |
| 🔒 Security   | Security vulnerability patches    |
