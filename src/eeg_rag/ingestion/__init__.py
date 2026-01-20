"""
EEG-RAG Data Ingestion Module.

Provides comprehensive data collection from multiple academic sources:
- PubMed/PMC for peer-reviewed biomedical literature
- Semantic Scholar for cross-disciplinary coverage
- arXiv for preprints and cutting-edge research
- OpenAlex for open access metadata

Usage:
    # Standard ingestion
    from eeg_rag.ingestion import IngestionPipeline
    
    pipeline = IngestionPipeline(output_dir="data/raw")
    stats = await pipeline.run_full_ingestion()
    
    # Bulk ingestion (100K+ papers with checkpointing)
    from eeg_rag.ingestion import BulkIngestionManager
    
    manager = BulkIngestionManager()
    stats = await manager.run_bulk_ingestion()
"""

from .pubmed_client import PubMedClient, PubMedArticle
from .scholar_client import SemanticScholarClient, ScholarArticle, GoogleScholarScraper
from .arxiv_client import ArxivClient, ArxivPaper
from .openalex_client import OpenAlexClient, OpenAlexWork
from .pipeline import IngestionPipeline, UnifiedDocument
from .chunker import EEGDocumentChunker, DocumentChunk, ChunkType
from .bulk_ingestion import BulkIngestionManager, BulkIngestionConfig, IngestionCheckpoint

__all__ = [
    # Clients
    "PubMedClient",
    "SemanticScholarClient", 
    "GoogleScholarScraper",
    "ArxivClient",
    "OpenAlexClient",
    
    # Data models
    "PubMedArticle",
    "ScholarArticle",
    "ArxivPaper",
    "OpenAlexWork",
    "UnifiedDocument",
    "DocumentChunk",
    "ChunkType",
    
    # Pipeline
    "IngestionPipeline",
    "EEGDocumentChunker",
    
    # Bulk ingestion
    "BulkIngestionManager",
    "BulkIngestionConfig",
    "IngestionCheckpoint",
]
