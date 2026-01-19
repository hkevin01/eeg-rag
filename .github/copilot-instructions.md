# EEG-RAG Copilot Development Instructions

This document provides comprehensive instructions for GitHub Copilot to effectively contribute to the EEG-RAG project.

## Project Context

EEG-RAG is a production-grade RAG (Retrieval-Augmented Generation) system specifically designed for electroencephalography (EEG) research and clinical applications. The system processes scientific literature, clinical guidelines, and research data to provide accurate, cited responses to EEG-related queries.

## Core Principles

1. **Medical-Grade Quality**: All code must meet healthcare/research standards with comprehensive error handling and validation
2. **Citation Accuracy**: Every response must be traceable to verified sources with PMID validation
3. **EEG Domain Expertise**: Understanding of EEG terminology, clinical context, and research methodologies is essential
4. **Production Readiness**: Code must be enterprise-ready with proper logging, monitoring, and scalability

## Quick Reference

- **Primary Language**: Python 3.9+ with async/await patterns
- **Key Frameworks**: FastAPI, Pydantic, asyncio, FAISS, sentence-transformers
- **Architecture**: Multi-agent RAG system with hybrid retrieval
- **Data Sources**: PubMed, local EEG datasets, knowledge graphs
- **Quality Standards**: 85%+ test coverage, type hints required, comprehensive logging

## Development Guidelines

### Code Standards
- Use type hints for all function signatures
- Follow Google-style docstrings with Args, Returns, Raises sections
- Maximum line length 88 characters (Black formatter)
- Prefer dataclasses/Pydantic models over plain dicts
- All I/O operations must be async

### Testing Requirements
- Minimum 85% coverage for core/ and agents/ directories
- 100% coverage for verification and citation modules
- EEG-specific test cases for terminology and clinical scenarios
- Mock all external APIs in unit tests

### Performance Targets
- Local retrieval: < 100ms for 10K documents
- End-to-end query: < 2 seconds (p95)
- Cache embeddings indefinitely, responses for 1 hour

## Domain Knowledge

### EEG Terminology
- Frequency bands: Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-100Hz)
- Electrode systems: 10-20, 10-10, high-density arrays
- ERP components: P300, N400, P600, MMN with typical latencies
- Clinical conditions: Epilepsy, sleep disorders, cognitive states

### Citation Standards
- Extract PMIDs using pattern: `PMID[:\s]*(\d{7,8})`
- Validate all PMIDs against PubMed before inclusion
- Use format `[PMID:XXXXXXXX]` in responses
- Track source chunks for each citation

## File Organization

```
src/eeg_rag/
├── agents/           # Multi-agent system components
├── core/            # Core functionality (routing, chunking)
├── retrieval/       # Hybrid search and indexing
├── verification/    # Citation and hallucination detection
├── evaluation/      # Testing and benchmarking
├── knowledge_graph/ # Graph storage and reasoning
└── utils/           # Shared utilities
```

For detailed instructions on specific aspects (testing, performance, security, etc.), see the individual instruction files in the `.copilot/` directory.