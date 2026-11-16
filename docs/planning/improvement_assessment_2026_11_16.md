# EEG-RAG Improvement and Upgrade Assessment

Assessment date: 2026-11-16. Recent-upgrade window: 2026-08-16 through
2026-11-16, inclusive. This is a targeted source review, not a whole-project
audit or a measured performance benchmark. No application changes or dependency
upgrades were made. Declared constraints are not evidence of installed versions.

## Five Improvement Opportunities

### 1. Make Citation Verification Fail Closed

**Priority: critical.**

Evidence: [CitationVerifier](../../src/eeg_rag/verification/citation_verifier.py#L412)
initializes claim support to true and skips inference when the model is missing.
[_check_claim_support](../../src/eeg_rag/verification/citation_verifier.py#L623)
returns `(True, 0.0)` when checking is unavailable or raises an exception.
[_check_retraction_status](../../src/eeg_rag/verification/citation_verifier.py#L807)
returns false even if its external checks fail. The existing
[no-model test](../../tests/test_citation_verifier.py#L100) explicitly expects
the fail-open claim behavior.

Introduce explicit supported, unsupported, and unknown states, and checked,
retracted, and unknown retraction states. Distinguish a valid PMID without an
abstract from a nonexistent PMID and a transport failure. Require evidence before
an answer can describe a claim as verified. Cosine similarity is topical evidence,
not proof of entailment: add contradiction-aware claim-to-source evaluation and
retain source chunk IDs and evidence spans.

Validation: model unavailable, inference exception, HTTP timeout, missing abstract,
retracted publication, and a negated EEG claim must not become supported evidence.
Use independently labeled epilepsy, sleep-stage, and ERP examples to calibrate
claim decisions rather than assuming the current 0.5 similarity threshold is safe.

### 2. Apply Inclusion Filters to Both Retrieval Branches

**Priority: high.**

Evidence: [HybridRetriever.search](../../src/eeg_rag/retrieval/hybrid_retriever.py#L335)
passes filters to dense retrieval but calls BM25 without a filter. Yet
[BM25Retriever.search](../../src/eeg_rag/retrieval/bm25_retriever.py#L204)
already accepts a `filter_fn`. RRF can therefore include a BM25-only document
that does not satisfy the dense branch's year or other eligibility constraint.
The hybrid method currently documents filters as dense-only, so broaden that
contract deliberately rather than silently changing it.

Define one eligibility policy, translate it for both backends, and validate
eligibility before fusion. Preserve enough eligible candidates before truncation.
This protects systematic-review inclusion criteria; these filters should not be
treated as authorization controls without a separate security review.

Validation: place a highly ranked out-of-range paper in BM25 only and confirm it
cannot enter filtered hybrid results. Test missing metadata, empty results, range
boundaries, and equivalent backend predicates.

### 3. Rerank a Larger Candidate Pool Before Final Truncation

**Priority: high.**

Evidence: [hybrid truncation](../../src/eeg_rag/retrieval/hybrid_retriever.py#L402)
cuts fused results to `top_k` before invoking the cross-encoder. A relevant paper
at fused rank 11 cannot be promoted when `top_k=10`, even if it was retrieved
among the 100 candidates from each branch.

Separate first-stage retrieval count, reranking count, and final result count.
Benchmark reranking 30, 50, and 100 fused candidates, then return only `top_k`.
Retain the original RRF score separately from the reranker score instead of
overwriting `rrf_score` with a different kind of score. Keep adaptive reranking
and give it an explicit latency budget.

Validation: the best cross-encoder match starts below the original final cutoff
and must reach the returned results. Compare Recall@50, nDCG@10, MRR@10, and p95
latency on a held-out EEG corpus. A larger pool is not a guaranteed net win.

### 4. Remove Blocking Work and Unbounded Citation Fan-Out

**Priority: high.**

Evidence: the asynchronous
[LocalDataAgent._hybrid_search](../../src/eeg_rag/agents/local_agent/local_data_agent.py#L917)
directly calls synchronous hybrid retrieval. That search executes BM25 and dense
retrieval sequentially. Citation verification synchronously encodes a claim and
abstract inside an async request; its
[verify_multiple](../../src/eeg_rag/verification/citation_verifier.py#L495)
creates all verification tasks at once, while metadata and retraction helpers
create separate HTTP clients per operation. `zip(pmids, claims)` also silently
drops PMIDs when a nonempty claim list is too short.

Use a bounded inference worker or executor for blocking retrieval and embedding
work, with controlled model concurrency. Reuse lifecycle-managed HTTP clients,
deduplicate PMIDs, batch EFetch metadata, and apply a shared NCBI request limiter
across verification and ingestion. A semaphore limits concurrent requests, not
requests per second; implement both where needed. Validate claim-list length and
return a typed result for every requested PMID. Cache abstract embeddings by
model revision and content hash; retraction status needs a separate freshness
policy. Validate disabled-cache behavior in the older abstract helper as well.

Validation: under concurrent queries measure event-loop delay, p50/p95 latency,
maximum in-flight work, outbound requests, and cache hits. Use fake clocks and
mock external services. Confirm cancellation and timeout cleanup, duplicate
PMIDs, mismatched lists, and `enable_cache=False` behavior.

### 5. Unify Dependency and Python Compatibility Contracts

**Priority: high; prerequisite for the upgrades below.**

Evidence: [requirements.txt](../../requirements.txt) caps Sentence Transformers
below 3 and Transformers below 5, whereas [pyproject.toml](../../pyproject.toml#L39)
declares only lower bounds. The latter advertises Python 3.9+, and its dependency
lists do not declare `qdrant-client` or `aiohttp`, which are used by the reviewed
retrieval and PubMed modules. Installation methods can select substantially
different stacks.

Choose authoritative package metadata with explicit retrieval/ingestion extras,
generate a reproducible deployment lock or constraints file, and test the
supported Python matrix in clean environments. Raise the Python floor only as
an explicit compatibility decision. Record embedding model revision, tokenizer,
pooling, normalization, and vector dimensions with every index; reject mismatches
and preserve a rollback collection when changing encoders.

Validation: wheel and editable installations of each documented extra must
import and run the corresponding offline smoke tests without relying on packages
already present on a developer machine. Resolve both installation paths against
the same intended stack and run the held-out retrieval evaluation before rollout.

## Two Recent Upgrade Opportunities

### Upgrade A: Modernize Embedding and Reranking Inference

**Candidate: Sentence Transformers 6.3.0, released 2026-11-11.** Its 6.0 release
on 2026-08-18 and 6.2 release on 2026-10-21 also fall inside the window.
Primary release metadata: [official releases API](https://api.github.com/repos/huggingface/sentence-transformers/releases?per_page=5).

Target [VectorDB](../../src/eeg_rag/storage/vector_db.py),
[DenseRetriever](../../src/eeg_rag/retrieval/dense_retriever.py), and
[citation inference](../../src/eeg_rag/verification/citation_verifier.py#L623).

Start by preserving the existing model and comparing optimized ONNX CPU inference
with PyTorch, using the [official efficiency guide](https://sbert.net/docs/sentence_transformer/usage/efficiency.html).
The September 6.1 release refreshed benchmarks and documented ONNX CPU thread
tuning; ONNX itself is older, not a newly invented optimization. On supported
GPUs, benchmark FP16/BF16 and applicable attention kernels while keeping ranking
scores in FP32. The August release corrected low-precision scoring issues. The
October release reduced redundant Hub requests; its reported MiniLM constructor
time fell from 6.39 to 2.66 seconds in a warm-cache, online benchmark. This is a
startup benefit, not a per-query latency measurement.

As a second-stage experiment within this upgrade, evaluate late-interaction
reranking before committing to a multi-vector corpus migration. November's
[pooling-aware training](https://huggingface.co/blog/token-pooling) reduced average
document vectors from 193.22 to 48.68, retaining 97.8% of the trained model's
unpooled NanoBEIR nDCG@10. These are general benchmark results, not EEG results,
and do not apply directly to the current single-vector embeddings. A multi-vector
representation can still be much larger than the existing dense index.

Compatibility: the 6.x stack requires Python 3.10+, Transformers 5+, and Torch
2.2+ according to the official release notes. It conflicts with current text
requirements and needs a coordinated migration, not a one-line version bump.
Audit checkpoint compatibility and model trust settings. Pin model revisions,
retain source/citation mappings, and rebuild indexes if encoding semantics change.

Acceptance: no material held-out EEG recall or citation-support regression, lower
measured latency or memory on deployment hardware, and successful rollback.

### Upgrade B: Consolidate Hybrid Search in Qdrant 1.20.x

**New capabilities: Qdrant 1.20.0, released 2026-10-22.** Patch 1.20.1 was
released 2026-11-13; the release feed also reports 1.20.2 on 2026-11-16.
Primary metadata: [official releases API](https://api.github.com/repos/qdrant/qdrant/releases?per_page=5).
The deployed server version was not inspected, so these are candidates rather
than a claim that the running database is outdated.

Target [VectorDB](../../src/eeg_rag/storage/vector_db.py) and
[HybridRetriever](../../src/eeg_rag/retrieval/hybrid_retriever.py#L335).
Version 1.20 introduced [BM25 scoring on full-text payload indexes](https://qdrant.tech/documentation/search/text-search/full-text-search/).
If chunk text is already stored in Qdrant, create a scoring text index and evaluate
server-side BM25/dense prefetch and fusion. This can remove the separate local
BM25 snapshot and make filtering and updates easier to keep consistent. Preserve
the offline/local retrieval mode if it remains a product requirement.

Version 1.20 also introduced [turbo8 and turbo16 vector datatypes](https://qdrant.tech/documentation/manage-data/vectors/#turboquant-datatypes),
approximately one-quarter and one-half of float32 vector storage respectively.
These ratios are for vector representation, not total database memory or disk.
Datatype compression replaces the original stored vectors, unlike an additional
quantization index; keep float32 originals outside the compressed collection for
rollback and exact-reference evaluation.

The [1.20.1 patch](https://github.com/qdrant/qdrant/releases/tag/v1.20.1) improves
cold filtered reads and fixes a many-core HNSW-build slowdown introduced in 1.20.0.
Evaluate a patched 1.20.x release rather than deploying 1.20.0 just for its features.
The 1.20.2 notes restore the classic default HNSW build algorithm, so verify graph
quality and `ef_construct` behavior when choosing between these patches.

Risks: shard-local BM25 statistics and tokenizer differences can change ranking;
test EEG abbreviations, hyphens, frequency notation, electrode names, and ERP
terms. Current package constraints do not prove that the installed client supports
new query/index schemas. Verify server/client compatibility, authentication,
snapshot restoration, and resource usage in staging before cutover.

Acceptance: compare float32 and compressed retrieval against identical relevance
labels, test filter correctness during ingestion, measure p95 under concurrent
reads/writes, and demonstrate snapshot recovery and collection rollback.

## Recommended Order

1. Correct citation states and filter consistency.
2. Fix reranking candidate depth and asynchronous work boundaries.
3. Unify packaging and collect an EEG-specific performance/quality baseline.
4. Trial optimized inference without changing the embedding model.
5. Trial Qdrant consolidation and compression; keep multi-vector retrieval optional.

The <100 ms local-retrieval and <2 s end-to-end p95 figures in repository guidance
are targets, not measured outcomes of this assessment. Separate cached/local
queries from live literature discovery and provider generation when reporting
latency, so network and generation costs are visible.
