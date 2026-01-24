# EEG-RAG Search Architecture

## Overview

EEG-RAG uses a **hybrid search architecture** that combines local cached metadata with real-time remote searches across multiple external sources.

## Key Concepts

### 1. Papers Cached Locally (Metadata Only)
- **What**: Metadata database stored in SQLite/JSONL
- **Content**: Title, abstract, authors, PMID/DOI, keywords, MeSH terms
- **Purpose**: Fast retrieval, offline access, citation tracking
- **Size**: User-controlled (typically 10K-500K papers)
- **Storage**: ~1-2MB per 1000 papers (metadata only, NO full PDFs)

### 2. Search Coverage (Remote Sources)
- **What**: Real-time API access to external databases
- **Sources**:
  - **PubMed/PubMed Central**: 35M+ biomedical papers
  - **arXiv**: 2M+ preprints (including neuro/bio sections)
  - **Semantic Scholar**: 200M+ papers across disciplines
  - **Europe PMC**: 40M+ life sciences papers
- **Purpose**: Access to ALL available literature, not just local cache
- **Latency**: 1-3 seconds per query (with caching)

## How Search Works

### Query Flow

```
User Query: "EEG alpha waves in epilepsy"
    ↓
Orchestrator → Query Planner
    ↓
┌─────────────────────────────────────┐
│  Parallel Agent Execution:          │
├─────────────────────────────────────┤
│  1. Local Agent                     │
│     → Searches cached metadata      │
│     → Returns in ~100ms             │
│                                     │
│  2. Web Search Agent (PubMed)       │
│     → E-utilities API query         │
│     → Returns latest papers         │
│     → ~1-2 seconds                  │
│                                     │
│  3. Graph Agent                     │
│     → Knowledge graph traversal     │
│     → Entity relationships          │
│                                     │
│  4. Citation Agent                  │
│     → Validates PMIDs               │
│     → Checks retractions            │
└─────────────────────────────────────┘
    ↓
Context Aggregator
    ↓
Generation Ensemble (Multi-LLM)
    ↓
Final Answer with Citations
```

### Agent Responsibilities

#### Local Agent
- **Searches**: Cached metadata database only
- **Speed**: <100ms (FAISS vector search)
- **Coverage**: Papers explicitly ingested/cached
- **Best for**: Known papers, offline use, specific studies

#### Web Search Agent  
- **Searches**: PubMed via E-utilities API
- **Speed**: 1-2 seconds (rate-limited)
- **Coverage**: ALL 35M+ PubMed papers
- **Best for**: Latest research, comprehensive coverage, specific queries
- **Caching**: Results cached to avoid duplicate API calls

#### Graph Agent
- **Searches**: Knowledge graph relationships
- **Coverage**: Entities, concepts, cross-references
- **Best for**: Conceptual connections, research networks

#### Citation Agent
- **Validates**: PMIDs, DOIs, retractions
- **Coverage**: All cited papers regardless of source
- **Best for**: Citation verification, impact assessment

## Indexing Strategy

### What Gets "Indexed" (Cached Locally)

The local cache should contain:
1. **High-priority papers**: Frequently accessed, foundational studies
2. **Recent papers**: Last 2-3 years of relevant research
3. **User-specific corpus**: Papers relevant to your research domain
4. **Citation chains**: Papers cited by/citing your key papers

### What Stays Remote

Everything else! The Web Search Agent provides access to:
- Millions of papers not in local cache
- Real-time updates (papers published today)
- Niche/specialized topics
- Historical literature

## Best Practices

### For Users

1. **Don't try to index everything**: Cache metadata for ~10K-100K papers max
2. **Let remote search work**: Most queries will use PubMed API automatically
3. **Cache strategically**: Index papers you'll reference repeatedly
4. **Trust the orchestrator**: It decides local vs remote automatically

### For Developers

1. **Never download full PDFs** without explicit user request
2. **Cache metadata only**: Title, abstract, identifiers
3. **Use remote APIs**: Let PubMed/arXiv host the content
4. **Parallel execution**: Query local + remote simultaneously
5. **Aggregate results**: Merge local + remote findings seamlessly

## Performance Characteristics

### Local Search
- Latency: 50-100ms
- Throughput: 100+ queries/sec
- Cost: Storage only (~GB)
- Limitation: Only cached papers

### Remote Search (PubMed)
- Latency: 1-2 seconds
- Throughput: 3-10 req/sec (rate limited)
- Cost: Free (NCBI provides public API)
- Limitation: API rate limits

### Hybrid (Recommended)
- Latency: Max of parallel operations (~1-2 seconds)
- Coverage: Local cache + 35M+ remote papers
- Cost: Minimal (free APIs)
- Best of both worlds: Speed + Coverage

## Statistics Display

The UI should show:

```
Papers Cached Locally:    15,432
Search Coverage:          35M+ via PubMed
AI Agents Active:         12
Last Cache Update:        2026-01-24 08:00
```

**NOT** "Papers Indexed" implying that's all you can search.

## Common Misconceptions

❌ **Wrong**: "We need to index all 35M PubMed papers locally"
✅ **Right**: "We cache frequently-used papers, search PubMed for everything else"

❌ **Wrong**: "Users can only search papers we've downloaded"  
✅ **Right**: "Users can search ALL available papers via Web Search Agent"

❌ **Wrong**: "We need to download PDFs to search them"
✅ **Right**: "We search metadata (title/abstract/keywords), link to sources"

## Future Enhancements

1. **Multi-source federated search**: Query PubMed + arXiv + Semantic Scholar simultaneously
2. **Smart cache warming**: Automatically cache papers likely to be queried
3. **Distributed caching**: Share metadata across users/instances
4. **Citation network crawling**: Auto-expand cache based on citation graphs
5. **Full-text search**: OCR PDFs for users who explicitly download them

## Implementation Status

✅ Web Search Agent (PubMed E-utilities)  
✅ Local Agent (FAISS + SQLite)  
✅ Parallel orchestration  
✅ Result aggregation  
✅ Citation validation  
🟡 Multi-source federation (PubMed complete, arXiv/S2 pending)  
🟡 Smart cache warming (basic auto-refresh implemented)  
⭕ Full-text PDF search (not planned - metadata sufficient)

---

**Summary**: Think of EEG-RAG as a **search aggregator** not a **document database**. We cache metadata for speed, but search the entire scientific literature through APIs.
