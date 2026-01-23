# EEG-RAG Metadata Index

This directory contains the lightweight paper metadata index that ships with the repository.

## Files

- **index.db.gz** - Compressed SQLite index (~7-10 MB)
- **index.db** - Extracted index (created on first run)

## What's in the Index?

The index contains **paper references only** - identifiers and titles for 500K+ EEG papers:

| Field | Description | Example |
|-------|-------------|---------|
| pmid | PubMed ID | 34520953 |
| doi | Digital Object Identifier | 10.1016/j.yebeh.2021.108296 |
| openalex_id | OpenAlex Work ID | W3196842 |
| title | Paper title | "EEG-based P300 in mesial temporal lobe epilepsy..." |
| year | Publication year | 2021 |
| keywords | Search keywords | ["EEG", "P300", "epilepsy"] |

## What's NOT in the Index?

Full content is fetched on-demand from APIs:
- ❌ Abstracts
- ❌ Authors
- ❌ Full text
- ❌ Citations

This keeps the repository lightweight while still providing access to 500K+ papers.

## How It Works

1. User searches → Metadata index returns matching paper IDs (instant)
2. Top results → Full content fetched from PubMed/OpenAlex
3. Fetched papers → Cached locally in `~/.eeg_rag/cache/`

## Rebuilding the Index

For maintainers only:

```bash
# Full build (500K papers, ~30 min)
python scripts/build_metadata_index.py --target 500000

# Quick build (50K papers, ~5 min)
python scripts/build_metadata_index.py --quick
```

## User Setup

Users run this after cloning:

```bash
python scripts/setup_user.py
```

This extracts the compressed index and initializes the local cache.
