# Changelog

All notable changes to EEG-RAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - Total: 165 tests for utility modules
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
