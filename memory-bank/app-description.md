# EEG-RAG: Retrieval-Augmented Generation for EEG Research

## Project Overview

EEG-RAG is an intelligent Retrieval-Augmented Generation (RAG) system designed to transform EEG (Electroencephalography) research literature into a queryable knowledge base. The system enables researchers, clinicians, and data scientists to ask natural-language questions about EEG research and receive evidence-based answers with proper citations.

## Core Features

### 1. Literature Ingestion & Processing
- **Multi-source data collection**: PubMed, arXiv, bioRxiv, clinical guidelines
- **EEG-focused corpus**: Epilepsy, sleep staging, BCIs, cognitive neuroscience
- **Intelligent chunking**: Section-aware text processing with ~512 tokens per chunk
- **Metadata extraction**: Authors, PMIDs, publication dates, MeSH terms

### 2. Advanced NLP Pipeline
- **Biomedical embeddings**: PubMedBERT or specialized neuroscience models
- **Entity recognition**: EEG biomarkers, ERP components, frequency bands, conditions
- **Relation extraction**: Biomarker-condition-outcome mappings
- **Vector storage**: FAISS for efficient similarity search

### 3. Knowledge Graph
- **Neo4j backend**: Stores entities and relationships
- **Entity types**: PAPER, STUDY, EEG_BIOMARKER, CONDITION, TASK, DATASET, OUTCOME
- **Multi-hop reasoning**: Connect related concepts across studies

### 4. RAG Pipeline
- **Query understanding**: Classify question types (clinical, experimental, methods)
- **Semantic retrieval**: FAISS-based vector search
- **Reranking**: Optional cross-encoder for precision
- **Graph expansion**: Enrich context with knowledge graph connections
- **LLM generation**: Synthesize answers with explicit citations

## Target Users

### Clinical Researchers
- Epileptologists studying seizure prediction
- Sleep medicine specialists
- ICU neurologists analyzing continuous EEG monitoring

### Experimental Neuroscientists
- Cognitive researchers studying ERPs (P300, N400, MMN)
- BCI developers working on motor imagery
- Oscillation researchers analyzing frequency bands

### Machine Learning Engineers
- Building seizure detection algorithms
- Developing sleep staging models
- Creating BCI classification systems

## Technical Stack

### Core Technologies
- **Language**: Python 3.9+
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: PubMedBERT / Sentence Transformers
- **Graph Database**: Neo4j (optional)
- **Caching**: Redis (optional)
- **LLM**: OpenAI API / Local models (LLaMA, Mistral)

### Development Tools
- **Testing**: pytest, coverage
- **Code Quality**: black, pylint, mypy
- **Documentation**: Sphinx
- **CI/CD**: GitHub Actions
- **Containerization**: Docker

## Project Goals

### Short-term Goals (Phase 1-2)
1. Build minimal working RAG system with sample EEG corpus
2. Implement FAISS vector search with PubMedBERT embeddings
3. Create CLI for querying the system
4. Develop Python API for programmatic access
5. Establish CI/CD pipeline with automated testing

### Medium-term Goals (Phase 3-4)
1. Expand corpus to 10,000+ EEG-related papers
2. Implement EEG-specific knowledge graph
3. Add cross-encoder reranking for improved precision
4. Create web interface for non-technical users
5. Integrate with PubMed for real-time updates

### Long-term Goals (Phase 5-6)
1. Multi-modal support (EEG signals + literature)
2. Meta-analysis capabilities for biomarkers
3. Clinical decision support integration
4. Dataset recommendation system
5. Automated literature review generation

## Architecture Principles

### Scientific Rigor
- Always provide explicit citations (PMIDs, DOIs)
- Preserve source context and metadata
- Validate claims against multiple sources
- Track confidence scores for answers

### Scalability
- Modular design for easy component replacement
- Efficient vector indexing for large corpora
- Distributed processing capability
- Caching strategies for common queries

### Maintainability
- Type-hinted Python code
- Comprehensive documentation
- Unit and integration tests
- Clear separation of concerns

### Flexibility
- Config-driven architecture
- Optional components (Neo4j, Redis)
- Multiple LLM backend support
- Pluggable embedding models

## Success Metrics

1. **Retrieval Quality**: Top-10 recall > 85% for EEG queries
2. **Answer Accuracy**: Expert validation > 90% agreement
3. **Citation Precision**: All claims backed by cited sources
4. **Performance**: Query latency < 2 seconds
5. **Coverage**: Support 95% of common EEG research questions

## Project Status

**Current Phase**: Foundation (Phase 1)
- ✅ Repository structure established
- 🟡 Core RAG pipeline in development
- 🔴 Knowledge graph planned
- 🔴 Web interface planned

See `docs/project-plan.md` for detailed roadmap.
