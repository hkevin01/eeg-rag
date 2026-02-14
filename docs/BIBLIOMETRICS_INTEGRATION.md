# EEG Bibliometrics Integration

## Overview

The EEG-RAG bibliometrics module provides network-based bibliometric analysis capabilities for EEG research literature using the [pyBiblioNet](https://github.com/giorgioavena/pybiblionet) library.

This integration enables:
- **OpenAlex Integration**: Retrieve EEG research articles from the OpenAlex database
- **Citation Network Analysis**: Build and analyze citation relationships between papers
- **Co-authorship Network Analysis**: Map research collaboration networks
- **Influence Metrics**: Identify influential papers and researchers using centrality measures
- **RAG Enhancement**: Boost retrieval quality with bibliometric-informed article selection

## Requirements

```bash
pip install pybiblionet networkx
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Usage

```python
from eeg_rag.bibliometrics import EEGBiblioNet, EEGArticle
from eeg_rag.bibliometrics.eeg_biblionet import EEGResearchDomain

# Initialize with your email (required for OpenAlex polite pool)
biblio = EEGBiblioNet(email="your.email@example.com")

# Search for EEG epilepsy research
articles = biblio.search_eeg_literature(
    domain=EEGResearchDomain.EPILEPSY,
    from_date="2023-01-01",
    max_results=500,
)

# Build citation network
citation_graph = biblio.build_citation_network()

# Get influential papers
top_papers = biblio.get_influential_papers(top_n=10)
for paper in top_papers:
    print(f"{paper.title}: centrality={paper.centrality_score:.4f}")
```

### Integration with RAG Pipeline

```python
# Filter and prepare articles for RAG ingestion
rag_documents = biblio.get_articles_for_rag(
    min_citations=10,      # Only well-cited papers
    min_centrality=0.01,   # High network influence
    topics=["Epilepsy"],   # Topic filter
)

# Each document is formatted for RAG:
# {
#     "id": "openalex_id",
#     "doi": "...",
#     "pmid": "...",
#     "title": "...",
#     "content": "abstract text",
#     "metadata": {
#         "source": "OpenAlex",
#         "authors": [...],
#         "cited_by_count": 42,
#         "centrality_score": 0.0856,
#         ...
#     }
# }

# Ingest into your RAG system
for doc in rag_documents:
    rag_system.ingest(doc)
```

## Research Domains

Pre-defined query patterns for EEG research domains:

| Domain | Description | Example Query Pattern |
|--------|-------------|----------------------|
| `GENERAL` | All EEG research | `(EEG\|electroencephalography)` |
| `EPILEPSY` | Epilepsy & seizure detection | `(EEG)( )(epilepsy\|seizure\|ictal)` |
| `SLEEP` | Sleep staging & disorders | `(EEG)( )(sleep\|polysomnography)` |
| `BCI` | Brain-computer interfaces | `(EEG)( )(BCI\|brain-computer interface)` |
| `COGNITIVE` | Cognitive neuroscience & ERPs | `(EEG)( )(ERP\|P300\|N400)` |
| `CLINICAL` | Clinical neurophysiology | `(EEG)( )(clinical\|neurophysiology)` |
| `SIGNAL_PROCESSING` | Signal processing methods | `(EEG)( )(filtering\|artifact\|ICA)` |

## API Reference

### EEGBiblioNet

Main class for bibliometric analysis.

```python
class EEGBiblioNet:
    def __init__(
        self,
        email: str,                    # Required for OpenAlex API
        cache_dir: Optional[Path],     # Cache directory for results
        use_cache: bool = True,        # Enable caching
    )
```

#### Methods

| Method | Description |
|--------|-------------|
| `search_eeg_literature()` | Search OpenAlex for EEG articles |
| `build_citation_network()` | Build NetworkX citation graph |
| `build_coauthorship_network()` | Build NetworkX co-authorship graph |
| `compute_citation_centrality()` | Compute PageRank/betweenness/eigenvector |
| `get_influential_papers()` | Get top papers by centrality |
| `get_influential_authors()` | Get top authors by collaboration centrality |
| `get_network_metrics()` | Get network structure metrics |
| `detect_communities()` | Detect research communities |
| `get_articles_for_rag()` | Format articles for RAG ingestion |

### EEGArticle

Data class representing an EEG research article.

```python
@dataclass
class EEGArticle:
    openalex_id: str
    doi: Optional[str]
    pmid: Optional[str]
    title: str
    abstract: str
    authors: List[str]
    publication_date: Optional[str]
    cited_by_count: int
    venue: str
    topics: List[str]
    referenced_works: List[str]
    centrality_score: float
```

### NetworkMetrics

Data class for network-level statistics.

```python
@dataclass
class NetworkMetrics:
    node_count: int
    edge_count: int
    density: float
    avg_clustering: float
    num_components: int
    avg_degree: float
    max_degree: int
    diameter: int
```

## Centrality Methods

| Method | Use Case | Description |
|--------|----------|-------------|
| `pagerank` | Citation influence | Google's PageRank algorithm |
| `betweenness` | Information bridges | Nodes connecting communities |
| `eigenvector` | Prestige | Connection to influential nodes |
| `degree` | Productivity | Raw connection count |

## Example: Finding Key EEG Researchers

```python
# Build co-authorship network
coauthorship = biblio.build_coauthorship_network()

# Find researchers who bridge different communities
top_authors = biblio.get_influential_authors(
    top_n=20,
    method="betweenness"  # High betweenness = bridge builders
)

for author_name, score in top_authors:
    print(f"{author_name}: betweenness={score:.4f}")

# Detect research communities
communities = biblio.detect_communities(
    network_type="coauthorship",
    method="louvain"
)

# Group authors by community
from collections import defaultdict
community_members = defaultdict(list)
for author, comm_id in communities.items():
    community_members[comm_id].append(author)
```

## Caching

Results are cached in the specified `cache_dir` (defaults to temp directory):
- Article data: JSON files by query hash
- Networks: GML files for graph persistence

To clear cache:
```python
import shutil
shutil.rmtree(biblio.cache_dir)
```

## Performance Tips

1. **Start with smaller datasets**: Use `max_results=100` for testing
2. **Use date ranges**: Narrow `from_date` and `to_date` for faster queries
3. **Enable caching**: Set `use_cache=True` (default) for repeated queries
4. **Batch processing**: Process large result sets in chunks

## Requirements IDs

- **REQ-BIB-001**: OpenAlex integration for article retrieval
- **REQ-BIB-002**: Citation network analysis
- **REQ-BIB-003**: Co-authorship network analysis
- **REQ-BIB-004**: Centrality metrics computation
- **REQ-BIB-005**: RAG enhancement with bibliometric data

## Related Documentation

- [pyBiblioNet Documentation](https://github.com/giorgioavena/pybiblionet)
- [OpenAlex API Documentation](https://docs.openalex.org/)
- [NetworkX Documentation](https://networkx.org/documentation/)
