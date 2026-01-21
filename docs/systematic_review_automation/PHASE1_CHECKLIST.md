# Phase 1: Advanced Retrieval - Implementation Checklist

**Timeline**: Weeks 1-3  
**Goal**: Replace keyword search with production-grade hybrid retrieval  
**Status**: 🟡 Ready to Start

---

## Week 1: Vector Database Setup

### 1.1 Install Dependencies
```bash
- [ ] Install Qdrant client: `pip install qdrant-client`
- [ ] Install sentence-transformers: `pip install sentence-transformers`
- [ ] Install BM25: `pip install rank-bm25`
- [ ] Start Qdrant locally: `docker run -p 6333:6333 qdrant/qdrant`
- [ ] Test connection to Qdrant
```

### 1.2 Create Embedding Pipeline
```bash
- [ ] Create `/src/eeg_rag/storage/vector_db.py`
- [ ] Initialize sentence-transformers model (all-MiniLM-L6-v2)
- [ ] Create document embedding function
- [ ] Test embedding generation on sample papers
```

### 1.3 Implement Section-Aware Chunking
```bash
- [ ] Create `/src/eeg_rag/chunking/citation_aware_chunker.py`
- [ ] Detect sections: Abstract, Introduction, Methods, Results, Discussion
- [ ] Split with 512 token chunks, 128 overlap
- [ ] Preserve citation context across chunks
- [ ] Test on sample papers with citations
```

### 1.4 Build Qdrant Collection
```bash
- [ ] Design collection schema (vector + metadata)
- [ ] Metadata: title, authors, year, PMID, DOI, section, chunk_id
- [ ] Create collection creation script
- [ ] Index existing papers (Roy CSV + ingested JSONL)
- [ ] Verify vector count matches expected
```

---

## Week 2: Hybrid Retrieval

### 2.1 Implement BM25 Retriever
```bash
- [ ] Create `/src/eeg_rag/retrieval/bm25_retriever.py`
- [ ] Build BM25 index from paper corpus
- [ ] Implement sparse retrieval function
- [ ] Test BM25 retrieval on sample queries
```

### 2.2 Implement Dense Retriever
```bash
- [ ] Create `/src/eeg_rag/retrieval/dense_retriever.py`
- [ ] Query embedding function
- [ ] Vector similarity search in Qdrant
- [ ] Return top-k results with scores
- [ ] Test dense retrieval on sample queries
```

### 2.3 Implement Hybrid Fusion
```bash
- [ ] Create `/src/eeg_rag/retrieval/hybrid_retriever.py`
- [ ] Implement Reciprocal Rank Fusion (RRF)
- [ ] Combine BM25 + dense results
- [ ] Tune fusion weights (start with 50/50)
- [ ] Test hybrid retrieval on sample queries
```

### 2.4 Benchmark Retrieval Quality
```bash
- [ ] Create `/tests/test_retrieval_quality.py`
- [ ] Define 20 test queries with ground truth papers
- [ ] Measure Recall@5, Recall@10, MRR
- [ ] Compare: keyword-only vs BM25 vs dense vs hybrid
- [ ] Document baseline metrics
```

---

## Week 3: Query Enhancement & Integration

### 3.1 Query Expansion
```bash
- [ ] Create `/src/eeg_rag/query/query_expander.py`
- [ ] Build EEG-specific synonym dictionary
  - CNN → convolutional neural network
  - RNN → recurrent neural network, LSTM, GRU
  - BCI → brain-computer interface
  - etc.
- [ ] Expand queries with synonyms
- [ ] Test expanded queries improve recall
```

### 3.2 Contextual Re-ranking
```bash
- [ ] Create `/src/eeg_rag/query/reranker.py`
- [ ] Install cross-encoder: `pip install sentence-transformers`
- [ ] Load cross-encoder model (ms-marco-MiniLM-L-6-v2)
- [ ] Re-rank top-100 results to top-10
- [ ] Benchmark re-ranking impact on quality
```

### 3.3 Refactor RAGQueryEngine
```bash
- [ ] Extract retrieval logic from `app.py`
- [ ] Create clean API: `retrieve(query, top_k=10)`
- [ ] Replace keyword search with hybrid retriever
- [ ] Add retrieval confidence scores
- [ ] Maintain backward compatibility
```

### 3.4 Integration Testing
```bash
- [ ] Test end-to-end query flow
- [ ] Verify retrieval speed < 200ms
- [ ] Verify response quality improved
- [ ] Test with 10+ diverse queries
- [ ] Document performance improvements
```

---

## Success Metrics

### Quantitative
- [ ] Recall@10 > 90% (vs keyword baseline: ~60%)
- [ ] MRR > 0.8 (vs keyword baseline: ~0.5)
- [ ] Retrieval time < 200ms (vs keyword: ~50ms)
- [ ] Index size documented
- [ ] All tests passing

### Qualitative
- [ ] User-perceivable improvement in relevance
- [ ] Related papers surface correctly
- [ ] Semantic queries work (e.g., "methods for seizure detection" finds CNN/RNN papers)

---

## Deliverables

```
✅ Completed Files:
- /src/eeg_rag/storage/vector_db.py
- /src/eeg_rag/chunking/citation_aware_chunker.py
- /src/eeg_rag/retrieval/bm25_retriever.py
- /src/eeg_rag/retrieval/dense_retriever.py
- /src/eeg_rag/retrieval/hybrid_retriever.py
- /src/eeg_rag/query/query_expander.py
- /src/eeg_rag/query/reranker.py
- /tests/test_retrieval_quality.py
- /docs/retrieval_benchmarks.md

✅ Infrastructure:
- Qdrant running locally
- Papers indexed in vector DB
- BM25 index built

✅ Documentation:
- Benchmark results
- API documentation
- Usage examples
```

---

## Quick Start Commands

```bash
# Install dependencies
pip install qdrant-client sentence-transformers rank-bm25

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Create vector DB module
mkdir -p src/eeg_rag/storage
touch src/eeg_rag/storage/__init__.py
touch src/eeg_rag/storage/vector_db.py

# Create chunking module
mkdir -p src/eeg_rag/chunking
touch src/eeg_rag/chunking/__init__.py
touch src/eeg_rag/chunking/citation_aware_chunker.py

# Create query enhancement module
mkdir -p src/eeg_rag/query
touch src/eeg_rag/query/__init__.py
touch src/eeg_rag/query/query_expander.py
touch src/eeg_rag/query/reranker.py

# Run tests
pytest tests/test_retrieval_quality.py -v
```

---

## Notes

### Important Decisions
- **Embedding Model**: all-MiniLM-L6-v2 (384 dims, fast, good quality)
  - Alternative: all-mpnet-base-v2 (768 dims, better quality, slower)
- **Chunk Size**: 512 tokens with 128 overlap
  - Adjust if needed based on performance
- **Fusion Method**: Reciprocal Rank Fusion
  - Alternative: Linear combination with learned weights

### Potential Issues
1. **Memory Usage**: Qdrant in-memory mode uses ~1GB for 10K documents
   - Monitor and switch to persistent mode if needed
2. **Slow Indexing**: Embedding 10K papers takes ~10 minutes
   - Cache embeddings to avoid re-computation
3. **Query Latency**: Cross-encoder re-ranking adds ~50ms
   - Only re-rank top-100, not all results

### Future Enhancements (Phase 2+)
- Multi-vector queries for complex questions
- Query routing based on intent
- Federated search across multiple corpora
- Real-time indexing for new papers
- GPU acceleration for embeddings

---

**Ready?** Start with Week 1, Task 1.1 - Install dependencies! 🚀
