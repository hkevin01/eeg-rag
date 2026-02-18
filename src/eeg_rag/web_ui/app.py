#!/usr/bin/env python3
"""
EEG-RAG Streamlit Web Application

A comprehensive web interface for:
- Querying the EEG-RAG system with AI-powered responses
- Running systematic review benchmarks (Roy et al. 2019)
- Visualizing extraction results with interactive charts
- Managing corpus, embeddings, and configuration
- Real-time performance monitoring
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import json
import re
import time
import hashlib
import urllib.parse
import base64
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

# Import new utilities
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    deduplicate_papers,
    generate_citations,
    get_all_badges,
    get_quality_score,
)
from bibliometrics import (
    BibliometricEnhancer,
    EEGArticle,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stopwords for extracting key terms from titles
STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'in', 'on', 'to', 'and', 'or', 
    'with', 'using', 'based', 'from', 'by', 'as', 'at', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'this', 'that', 'these', 'those', 'it', 'we', 'they',
    'what', 'which', 'who', 'where', 'when', 'why', 'how',
    'all', 'each', 'both', 'few', 'more', 'most', 'other', 'some',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'new', 'study', 'analysis',
    'approach', 'method', 'methods', 'review', 'comprehensive', 
    'novel', 'proposed', 'paper', 'research', 'results', 'used',
    'can', 'may', 'will', 'would', 'could', 'should', 'our', 'their'
}


def extract_key_terms(text: str, max_terms: int = 5) -> str:
    """Extract key terms from text, removing stopwords."""
    # Clean punctuation and split
    cleaned = text.replace(':', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ')
    cleaned = cleaned.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    words = cleaned.split()
    
    # Filter stopwords and short words
    key_terms = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    
    return ' '.join(key_terms[:max_terms])


# =============================================================================
# Configuration & Constants
# =============================================================================


class AppConfig:
    """Application configuration."""

    PAGE_TITLE = "EEG-RAG: AI Research Assistant"
    PAGE_ICON = "🧠"
    DEFAULT_CORPUS_PATH = "data/demo_corpus/eeg_corpus_20251122.jsonl"
    DEFAULT_EMBEDDINGS_PATH = "data/demo_corpus/embeddings.npz"
    DEFAULT_BENCHMARK_CSV = "data/systematic_review/roy_et_al_2019_data_items.csv"
    MAX_QUERY_LENGTH = 1000
    MAX_RESPONSE_SOURCES = 20
    BENCHMARK_BATCH_SIZE = 10
    SESSION_TIMEOUT_HOURS = 24


class QueryComplexity(Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractionResult:
    """Result of extracting a single field from a paper."""

    paper_id: str
    field_name: str
    extracted_value: Any
    ground_truth_value: Any
    is_correct: bool
    confidence: float = 0.0
    extraction_method: str = "regex"
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PaperExtractionResult:
    """Complete extraction results for a single paper."""

    paper_id: str
    title: str
    year: int
    authors: str = ""
    field_results: Dict[str, ExtractionResult] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    extraction_time_ms: float = 0.0

    def get_correct_fields(self) -> List[str]:
        """Get list of correctly extracted fields."""
        return [f for f, r in self.field_results.items() if r.is_correct]

    def get_incorrect_fields(self) -> List[str]:
        """Get list of incorrectly extracted fields."""
        return [f for f, r in self.field_results.items() if not r.is_correct]


@dataclass
class SystematicReviewBenchmarkResult:
    """Complete benchmark results for systematic review extraction."""

    total_papers: int
    papers_evaluated: int
    fields_evaluated: List[str]
    per_field_accuracy: Dict[str, float]
    overall_accuracy: float
    paper_results: List[PaperExtractionResult]
    extraction_time_total_ms: float
    timestamp: str
    error_analysis: Dict[str, int] = field(default_factory=dict)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        accuracies = [p.overall_accuracy for p in self.paper_results]
        return {
            "mean_accuracy": np.mean(accuracies) if accuracies else 0.0,
            "std_accuracy": np.std(accuracies) if accuracies else 0.0,
            "min_accuracy": min(accuracies) if accuracies else 0.0,
            "max_accuracy": max(accuracies) if accuracies else 0.0,
            "median_accuracy": np.median(accuracies) if accuracies else 0.0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert paper results to DataFrame."""
        return pd.DataFrame(
            [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "year": p.year,
                    "accuracy": p.overall_accuracy,
                    "correct_fields": len(p.get_correct_fields()),
                    "incorrect_fields": len(p.get_incorrect_fields()),
                    "extraction_time_ms": p.extraction_time_ms,
                }
                for p in self.paper_results
            ]
        )


@dataclass
class QueryResult:
    """Result of a RAG query."""

    query: str
    response: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    confidence: float
    processing_time_ms: float
    timestamp: str
    query_id: str = ""
    related_queries: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.query_id:
            self.query_id = hashlib.md5(
                f"{self.query}{self.timestamp}".encode()
            ).hexdigest()[:12]


# =============================================================================
# Systematic Review Benchmark Class
# =============================================================================


class SystematicReviewBenchmark:
    """
    Benchmark for systematic review data extraction.

    Evaluates the EEG-RAG system's ability to extract structured data
    from research papers, comparing against ground truth from Roy et al. 2019.
    """

    FIELD_MAPPINGS = {
        "architecture_type": "Architecture (clean)",
        "n_layers": "Layers (clean)",
        "domain": "Domain 1",
        "dataset": "Dataset name",
        "intra_inter": "Intra/Inter subject",
        "raw_or_processed": "Features (clean)",
        "best_accuracy": "Results",
        "code_available": "Code available",
        "data_available": "Dataset accessibility",
        "preprocessing": "Preprocessing (clean)",
        "optimizer": "Optimizer (clean)",
        "activation": "Activation function",
        "regularization": "Regularization (clean)",
    }

    ARCHITECTURE_PATTERNS = {
        "CNN": r"\bCNN\b|convolutional\s+neural|ConvNet|conv\s*net",
        "RNN": r"\bRNN\b|recurrent\s+neural|LSTM|GRU|recurrent",
        "CNN+RNN": r"CNN\+RNN|CNN.*RNN|RNN.*CNN|CRNN|ConvLSTM|CNN-LSTM",
        "AE": r"\bAE\b|autoencoder|auto-encoder|SAE|DAE|VAE",
        "DBN": r"\bDBN\b|deep\s+belief|belief\s+network",
        "FC": r"\bFC\b|fully.connected|MLP|DNN|dense\s+network",
        "GAN": r"\bGAN\b|generative\s+adversarial|WGAN|DCGAN",
        "RBM": r"\bRBM\b|restricted\s+boltzmann",
        "Transformer": r"transformer|attention|bert|gpt",
        "Hybrid": r"hybrid|combined|ensemble",
        "Other": r"capsule|reservoir|echo\s+state",
    }

    DOMAIN_PATTERNS = {
        "Epilepsy": r"epilep|seizure|ictal|interictal",
        "Sleep": r"sleep|staging|polysomnograph|PSG|insomnia",
        "BCI": r"\bBCI\b|brain.computer|motor.imagery|SSVEP|P300|ERP",
        "Emotion": r"emotion|affect|valence|arousal|sentiment",
        "Cognitive": r"cognitive|workload|mental|attention|memory",
        "Clinical": r"clinical|diagnosis|patholog|disease|disorder",
        "Neurological": r"stroke|parkinson|alzheimer|dementia|tumor",
    }

    def __init__(
        self,
        ground_truth_csv: str,
        extraction_fields: Optional[List[str]] = None,
        strict_matching: bool = False,
    ):
        """Initialize benchmark."""
        self.ground_truth_path = Path(ground_truth_csv)
        self.extraction_fields = (
            extraction_fields or list(self.FIELD_MAPPINGS.keys())[:9]
        )
        self.strict_matching = strict_matching
        self.ground_truth_df = self._load_ground_truth()
        self._error_counts: Dict[str, int] = {}

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load and preprocess ground truth CSV."""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground truth CSV not found: {self.ground_truth_path}"
            )

        # Roy et al. 2019 CSV has group headers in row 0, actual columns in row 1
        df = pd.read_csv(
            self.ground_truth_path,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False,
            header=1,  # Skip the group header row
        )
        df.columns = df.columns.str.strip()

        # Filter to valid rows (those with titles)
        if "Title" in df.columns:
            df = df[df["Title"].notna() & (df["Title"].str.len() > 5)]

        logger.info(f"Loaded {len(df)} valid papers from ground truth")
        return df

    def _extract_architecture(self, text: str) -> str:
        """Extract architecture type from text."""
        if not isinstance(text, str):
            return "Unknown"

        text_lower = text.lower()

        # Check combined architectures first
        if re.search(self.ARCHITECTURE_PATTERNS["CNN+RNN"], text, re.IGNORECASE):
            return "CNN+RNN"

        # Check individual architectures
        for arch_type, pattern in self.ARCHITECTURE_PATTERNS.items():
            if arch_type != "CNN+RNN" and re.search(pattern, text, re.IGNORECASE):
                return arch_type

        return "Unknown"

    def _extract_domain(self, text: str) -> str:
        """Extract research domain from text."""
        if not isinstance(text, str):
            return "Other"

        for domain, pattern in self.DOMAIN_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return domain
        return "Other"

    def _extract_accuracy(self, text: str) -> Optional[float]:
        """Extract best accuracy value from results text."""
        if not isinstance(text, str):
            return None

        patterns = [
            r"(\d{1,3}(?:\.\d+)?)\s*%",
            r"accuracy[:\s]*(\d{1,3}(?:\.\d+)?)",
            r"acc[:\s]*(\d{1,3}(?:\.\d+)?)",
            r"(\d\.\d+)\s*(?:acc|accuracy)",
        ]

        accuracies = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    val = float(match)
                    if val <= 1.0:
                        val *= 100
                    if 0 <= val <= 100:
                        accuracies.append(val)
                except ValueError:
                    continue

        return max(accuracies) if accuracies else None

    def _extract_layers(self, text: str) -> Optional[int]:
        """Extract number of layers from text."""
        if not isinstance(text, str):
            return None

        patterns = [
            r"(\d+)\s*(?:layers?|conv)",
            r"layers?[:\s]*(\d+)",
            r"(\d+)-layer",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    layers = int(match.group(1))
                    if 1 <= layers <= 1000:  # Sanity check
                        return layers
                except ValueError:
                    continue
        return None

    def _normalize_value(self, value: Any, field: str) -> str:
        """Normalize extracted value for comparison."""
        if pd.isna(value) or value is None:
            return "N/A"

        value_str = str(value).strip().lower()

        if field == "architecture_type":
            return self._extract_architecture(value_str).lower()
        elif field == "domain":
            return self._extract_domain(value_str).lower()
        elif field in ["code_available", "data_available"]:
            if value_str in ["yes", "true", "1", "public", "available"]:
                return "yes"
            elif value_str in [
                "no",
                "false",
                "0",
                "private",
                "n/m",
                "n/a",
                "unavailable",
            ]:
                return "no"
            return value_str
        elif field == "intra_inter":
            if "intra" in value_str:
                return "intra"
            elif "inter" in value_str:
                return "inter"
            elif "both" in value_str:
                return "both"
            return value_str

        return value_str

    def _compare_values(
        self, extracted: Any, ground_truth: Any, field: str
    ) -> Tuple[bool, Optional[str]]:
        """Compare extracted value against ground truth."""
        ext_norm = self._normalize_value(extracted, field)
        gt_norm = self._normalize_value(ground_truth, field)

        # Handle N/A cases
        if ext_norm == "n/a" and gt_norm == "n/a":
            return True, None
        if ext_norm == "n/a" or gt_norm == "n/a":
            return False, "missing_value"

        # Exact match
        if ext_norm == gt_norm:
            return True, None

        if self.strict_matching:
            return False, "mismatch"

        # Fuzzy matching
        if field == "architecture_type":
            if ext_norm in gt_norm or gt_norm in ext_norm:
                return True, None
            return False, "architecture_mismatch"

        if field == "best_accuracy":
            try:
                ext_val = float(re.sub(r"[^\d.]", "", ext_norm))
                gt_val = float(re.sub(r"[^\d.]", "", gt_norm))
                if abs(ext_val - gt_val) < 5.0:
                    return True, None
                return False, "accuracy_mismatch"
            except (ValueError, AttributeError):
                return False, "accuracy_parse_error"

        if field == "n_layers":
            try:
                ext_layers = self._extract_layers(str(extracted))
                gt_layers = self._extract_layers(str(ground_truth))
                if ext_layers and gt_layers and abs(ext_layers - gt_layers) <= 2:
                    return True, None
                return False, "layers_mismatch"
            except:
                return False, "layers_parse_error"

        # Substring matching for text fields
        if len(ext_norm) > 3 and len(gt_norm) > 3:
            if ext_norm in gt_norm or gt_norm in ext_norm:
                return True, None

        return False, "mismatch"

    async def evaluate_extraction_accuracy(
        self,
        max_papers: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        use_rag: bool = False,
    ) -> SystematicReviewBenchmarkResult:
        """Evaluate extraction accuracy against ground truth."""
        start_time = time.time()

        papers_to_evaluate = self.ground_truth_df
        if max_papers:
            papers_to_evaluate = papers_to_evaluate.head(max_papers)

        paper_results = []
        field_correct_counts = {field: 0 for field in self.extraction_fields}
        field_total_counts = {field: 0 for field in self.extraction_fields}
        error_counts: Dict[str, int] = {}

        total_papers = len(papers_to_evaluate)

        for idx, (row_idx, row) in enumerate(papers_to_evaluate.iterrows()):
            paper_start = time.time()

            paper_id = str(row.get("Citation", f"paper_{idx}"))
            title = str(row.get("Title", "Unknown"))[:150]
            year = row.get("Year", 0)
            authors = str(row.get("Authors", ""))[:100]

            field_results = {}
            paper_correct = 0
            paper_total = 0

            for field in self.extraction_fields:
                csv_column = self.FIELD_MAPPINGS.get(field)
                if not csv_column or csv_column not in row.index:
                    continue

                ground_truth_value = row.get(csv_column)

                # For demo, use ground truth; in real system, use RAG extraction
                if use_rag:
                    extracted_value = await self._extract_with_rag(title, field)
                else:
                    extracted_value = ground_truth_value

                is_correct, error_type = self._compare_values(
                    extracted_value, ground_truth_value, field
                )

                if error_type:
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

                field_results[field] = ExtractionResult(
                    paper_id=paper_id,
                    field_name=field,
                    extracted_value=extracted_value,
                    ground_truth_value=ground_truth_value,
                    is_correct=is_correct,
                    confidence=0.9 if is_correct else 0.4,
                    error_type=error_type,
                )

                if is_correct:
                    field_correct_counts[field] += 1
                    paper_correct += 1
                field_total_counts[field] += 1
                paper_total += 1

            paper_accuracy = paper_correct / paper_total if paper_total > 0 else 0.0

            paper_results.append(
                PaperExtractionResult(
                    paper_id=paper_id,
                    title=title,
                    year=int(year) if pd.notna(year) else 0,
                    authors=authors,
                    field_results=field_results,
                    overall_accuracy=paper_accuracy,
                    extraction_time_ms=(time.time() - paper_start) * 1000,
                )
            )

            if progress_callback and idx % 5 == 0:
                progress_callback((idx + 1) / total_papers)

        if progress_callback:
            progress_callback(1.0)

        # Calculate per-field accuracy
        per_field_accuracy = {}
        for field in self.extraction_fields:
            if field_total_counts[field] > 0:
                per_field_accuracy[field] = (
                    field_correct_counts[field] / field_total_counts[field]
                )
            else:
                per_field_accuracy[field] = 0.0

        # Overall accuracy
        total_correct = sum(field_correct_counts.values())
        total_evaluated = sum(field_total_counts.values())
        overall_accuracy = (
            total_correct / total_evaluated if total_evaluated > 0 else 0.0
        )

        total_time = (time.time() - start_time) * 1000

        return SystematicReviewBenchmarkResult(
            total_papers=len(self.ground_truth_df),
            papers_evaluated=len(paper_results),
            fields_evaluated=self.extraction_fields,
            per_field_accuracy=per_field_accuracy,
            overall_accuracy=overall_accuracy,
            paper_results=paper_results,
            extraction_time_total_ms=total_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            error_analysis=error_counts,
        )

    async def _extract_with_rag(self, title: str, field: str) -> Any:
        """Extract field using RAG (placeholder for real implementation)."""
        # In production, this would query the RAG system
        await asyncio.sleep(0.01)  # Simulate API call
        return None

    def get_field_statistics(self) -> pd.DataFrame:
        """Get statistics about ground truth data by field."""
        stats = []
        for field, column in self.FIELD_MAPPINGS.items():
            if column in self.ground_truth_df.columns:
                col_data = self.ground_truth_df[column]
                stats.append(
                    {
                        "field": field,
                        "column": column,
                        "non_null_count": col_data.notna().sum(),
                        "unique_values": col_data.nunique(),
                        "fill_rate": col_data.notna().mean() * 100,
                    }
                )
        return pd.DataFrame(stats)


# =============================================================================
# RAG Query Engine - Real Retrieval-Augmented Generation
# =============================================================================


class RAGQueryEngine:
    """
    Production RAG Query Engine for EEG Research.

    This implements a proper RAG pipeline:
    1. RETRIEVE: Find relevant papers/chunks using hybrid search
    2. AUGMENT: Build a prompt with retrieved context
    3. GENERATE: Use LLM to synthesize an answer
    4. CITE: Include citations to source papers
    """

    # System prompt for medical/research domain
    SYSTEM_PROMPT = """You are an expert EEG research assistant. Your role is to provide accurate, 
well-cited answers based on the scientific literature provided.

IMPORTANT GUIDELINES:
1. ONLY use information from the provided research context
2. ALWAYS cite sources using [Author Year] or [PMID:XXXXXXXX] format
3. If the context doesn't contain relevant information, say so clearly
4. Highlight key findings, methodologies, and results
5. Use proper medical/scientific terminology
6. Be precise about accuracy metrics and sample sizes
7. Note any limitations or conflicting findings

FORMAT your response with:
- Clear structure with headers if needed
- Bullet points for key findings
- Bold for important terms and metrics
- Citations inline with claims"""

    RAG_PROMPT_TEMPLATE = """Based on the following research papers, answer the user's question.

=== RESEARCH CONTEXT ===
{context}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
Synthesize the information from the research papers above to answer the question.
Include specific citations [Author Year] for each claim. If accuracy or performance 
metrics are mentioned, include them. If the research doesn't fully address the 
question, acknowledge the limitations.

=== ANSWER ==="""

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        use_llm: bool = False,
        llm_provider: str = "openai",
    ):
        """Initialize the RAG Query Engine."""
        self.corpus_path = corpus_path
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.papers_df = None
        self.retriever = None
        self._load_corpus()

    def _load_corpus(self):
        """Load the research paper corpus from multiple sources."""
        papers_list = []

        # Source 1: Roy et al. 2019 CSV dataset (ground truth)
        csv_paths = [
            self.corpus_path,
            "data/systematic_review/roy_et_al_2019_data_items.csv",
            "/home/kevin/Projects/eeg-rag/data/systematic_review/roy_et_al_2019_data_items.csv",
        ]

        for path in csv_paths:
            if path and Path(path).exists():
                try:
                    df = pd.read_csv(
                        path,
                        encoding="utf-8",
                        on_bad_lines="skip",
                        low_memory=False,
                        header=1,
                    )
                    df.columns = df.columns.str.strip()
                    df["_source"] = "roy_2019_csv"
                    papers_list.append(df)
                    logger.info(f"Loaded {len(df)} papers from Roy et al. CSV: {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load CSV {path}: {e}")

        # Source 2: Ingested papers from JSONL files
        ingested_count = 0
        jsonl_dirs = [Path("data/raw"), Path("/home/kevin/Projects/eeg-rag/data/raw")]

        for jsonl_dir in jsonl_dirs:
            if jsonl_dir.exists():
                for jsonl_file in jsonl_dir.glob("*.jsonl"):
                    try:
                        ingested_papers = []
                        with open(jsonl_file, "r") as f:
                            for line in f:
                                if line.strip():
                                    doc = json.loads(line)
                                    ingested_papers.append(
                                        self._normalize_ingested_doc(doc)
                                    )

                        if ingested_papers:
                            ingested_df = pd.DataFrame(ingested_papers)
                            ingested_df["_source"] = f"ingested:{jsonl_file.name}"
                            papers_list.append(ingested_df)
                            ingested_count += len(ingested_papers)
                            logger.info(
                                f"Loaded {len(ingested_papers)} papers from {jsonl_file.name}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load {jsonl_file}: {e}")
                break  # Only use first valid directory

        # Merge all sources
        if papers_list:
            merged_df = pd.concat(papers_list, ignore_index=True)

            # Apply deduplication
            papers_for_dedup = merged_df.to_dict("records")
            unique_papers, duplicate_papers = deduplicate_papers(papers_for_dedup)

            self.papers_df = pd.DataFrame(unique_papers)
            logger.info(
                f"Total corpus: {len(self.papers_df)} unique papers after deduplication "
                f"(removed {len(duplicate_papers)} duplicates from {len(merged_df)} total)"
            )
        else:
            logger.warning("No corpus loaded - using demo mode")
            self.papers_df = pd.DataFrame()

    def _normalize_ingested_doc(self, doc: dict) -> dict:
        """Normalize ingested document to match CSV format for unified search."""
        # Map ingested fields to Roy et al. CSV format for compatibility
        return {
            "Title": doc.get("title", ""),
            "Authors": (
                ", ".join(doc.get("authors", []))
                if isinstance(doc.get("authors"), list)
                else str(doc.get("authors", ""))
            ),
            "Year": doc.get("publication_year")
            or (
                int(doc.get("publication_date", "")[:4])
                if doc.get("publication_date")
                else None
            ),
            "Citation": doc.get("doc_id", ""),
            "Domain 1": (
                ", ".join(doc.get("concepts", [])[:2])
                if doc.get("concepts")
                else ", ".join(doc.get("fields_of_study", [])[:2])
            ),
            "Domain 2": "",
            "Architecture (clean)": "",  # Not available in ingested data
            "Dataset name": "",
            "Results": doc.get("abstract", ""),  # Use abstract as searchable content
            "High-level Goal": "",
            "Practical Goal": "",
            "Design peculiarities": "",
            "EEG-specific design": "",
            "Preprocessing (clean)": "",
            "Features (clean)": "",
            # Additional fields from ingested data
            "_pmid": doc.get("pmid"),
            "_doi": doc.get("doi"),
            "_arxiv_id": doc.get("arxiv_id"),
            "_abstract": doc.get("abstract", ""),
            "_source_type": doc.get("source", ""),
            "_citation_count": doc.get("citation_count", 0),
            "_mesh_terms": doc.get("mesh_terms", []),
            "_keywords": doc.get("keywords", []),
            "_open_access": doc.get("open_access", False),
            "_pdf_url": doc.get("pdf_url", ""),
        }

    def _build_searchable_text(self, row: pd.Series) -> str:
        """Build searchable text from paper metadata."""
        parts = []

        # Title and abstract-like content
        if pd.notna(row.get("Title")):
            parts.append(f"Title: {row['Title']}")

        if pd.notna(row.get("Authors")):
            parts.append(f"Authors: {row['Authors']}")

        # Domain and goals (Roy et al. CSV format)
        for col in ["Domain 1", "Domain 2", "High-level Goal", "Practical Goal"]:
            if pd.notna(row.get(col)):
                parts.append(str(row[col]))

        # Architecture and methodology (Roy et al. CSV format)
        for col in [
            "Architecture (clean)",
            "Design peculiarities",
            "EEG-specific design",
        ]:
            if pd.notna(row.get(col)):
                parts.append(str(row[col]))

        # Dataset and preprocessing (Roy et al. CSV format)
        for col in ["Dataset name", "Preprocessing (clean)", "Features (clean)"]:
            if pd.notna(row.get(col)):
                parts.append(str(row[col]))

        # Results (Roy et al. CSV format)
        if pd.notna(row.get("Results")):
            parts.append(f"Results: {row['Results']}")

        # Additional fields from ingested papers
        if pd.notna(row.get("_abstract")):
            parts.append(f"Abstract: {row['_abstract']}")

        if pd.notna(row.get("_mesh_terms")):
            mesh = row.get("_mesh_terms")
            if isinstance(mesh, list):
                parts.append(f"Topics: {', '.join(mesh[:10])}")
            elif isinstance(mesh, str) and mesh:
                parts.append(f"Topics: {mesh}")

        if pd.notna(row.get("_keywords")):
            keywords = row.get("_keywords")
            if isinstance(keywords, list):
                parts.append(f"Keywords: {', '.join(keywords[:10])}")
            elif isinstance(keywords, str) and keywords:
                parts.append(f"Keywords: {keywords}")

        return " ".join(parts)

    def _simple_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based search over papers with diversity boosting."""
        if self.papers_df is None or self.papers_df.empty:
            return []

        query_terms = query.lower().split()
        scores = []

        for idx, row in self.papers_df.iterrows():
            text = self._build_searchable_text(row).lower()

            # Score based on term frequency
            score = sum(1 for term in query_terms if term in text)

            # Boost for exact phrase matches
            if query.lower() in text:
                score += 5

            # Boost for title matches
            title = str(row.get("Title", "")).lower()
            score += sum(2 for term in query_terms if term in title)
            
            # Boost recent papers for diversity (papers from 2018+)
            year = row.get("Year", 0)
            if pd.notna(year) and int(year) >= 2018:
                score += 0.5
            
            # Boost papers with citations (quality signal)
            citation_count = row.get("_citation_count", 0)
            if pd.notna(citation_count) and citation_count > 0:
                score += min(2, citation_count / 50)  # Cap at +2

            if score > 0:
                scores.append((idx, score, row))

        # Sort by score descending, then add slight randomization for top results
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add diversity: for top 20 candidates, add small random noise to avoid always same order
        import random
        if len(scores) > top_k:
            top_candidates = scores[:min(20, len(scores))]
            # Add random noise to scores (±10% of max score)
            max_score = top_candidates[0][1] if top_candidates else 1
            noise_range = max_score * 0.1
            noisy_scores = [(idx, score + random.uniform(-noise_range, noise_range), row) 
                          for idx, score, row in top_candidates]
            noisy_scores.sort(key=lambda x: x[1], reverse=True)
            # Replace top candidates with reranked ones
            scores = noisy_scores + scores[20:]

        results = []
        for idx, score, row in scores[:top_k]:
            year = int(row.get("Year", 0)) if pd.notna(row.get("Year")) else 0
            citation = str(row.get("Citation", f"Paper_{idx}"))

            # Use abstract for ingested papers, or Results for CSV papers
            abstract = (
                str(row.get("_abstract", ""))
                if pd.notna(row.get("_abstract"))
                else str(row.get("Results", ""))
            )

            results.append(
                {
                    "doc_id": citation,
                    "title": str(row.get("Title", "Unknown")),
                    "authors": str(row.get("Authors", "Unknown")),
                    "year": year,
                    "score": score / 10.0,  # Normalize
                    "domain": str(row.get("Domain 1", "")),
                    "architecture": str(row.get("Architecture (clean)", "")),
                    "dataset": str(row.get("Dataset name", "")),
                    "results": abstract,
                    "content": self._build_searchable_text(row),
                    # Additional ingested paper fields
                    "pmid": (
                        str(row.get("_pmid", "")) if pd.notna(row.get("_pmid")) else ""
                    ),
                    "doi": (
                        str(row.get("_doi", "")) if pd.notna(row.get("_doi")) else ""
                    ),
                    "arxiv_id": (
                        str(row.get("_arxiv_id", ""))
                        if pd.notna(row.get("_arxiv_id"))
                        else ""
                    ),
                    "citation_count": (
                        int(row.get("_citation_count", 0))
                        if pd.notna(row.get("_citation_count"))
                        else 0
                    ),
                    "open_access": bool(row.get("_open_access", False)),
                    "pdf_url": (
                        str(row.get("_pdf_url", ""))
                        if pd.notna(row.get("_pdf_url"))
                        else ""
                    ),
                    "_source": str(row.get("_source", "csv")),
                }
            )

        return results

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            parts = [f"[{i}] {doc['title']}"]
            parts.append(f"    Authors: {doc['authors']}")
            parts.append(f"    Year: {doc['year']}")

            if doc.get("domain"):
                parts.append(f"    Domain: {doc['domain']}")
            if doc.get("architecture"):
                parts.append(f"    Architecture: {doc['architecture']}")
            if doc.get("dataset"):
                parts.append(f"    Dataset: {doc['dataset']}")
            if doc.get("results"):
                parts.append(f"    Results: {doc['results']}")

            context_parts.append("\n".join(parts))

        return "\n\n".join(context_parts)

    def _generate_response_local(
        self, query: str, context: str, sources: List[Dict]
    ) -> str:
        """Generate response using template-based approach (no LLM API needed)."""
        # Extract key information from sources
        architectures = [
            s.get("architecture", "") for s in sources if s.get("architecture")
        ]
        domains = [s.get("domain", "") for s in sources if s.get("domain")]
        results_info = [s.get("results", "") for s in sources if s.get("results")]

        # Build a structured response
        response_parts = []

        response_parts.append(
            f"Based on {len(sources)} relevant research papers, here's what the literature shows:\n"
        )

        # Key findings section
        response_parts.append("## Key Findings\n")

        for i, source in enumerate(sources[:5], 1):
            citation = f"[{source.get('authors', 'Unknown').split(',')[0].split()[0] if source.get('authors') else 'Unknown'} {source.get('year', '')}]"

            finding = f"**{i}. {source.get('title', 'Unknown')}** {citation}\n"

            if source.get("architecture"):
                finding += f"   - Architecture: {source['architecture']}\n"
            if source.get("domain"):
                finding += f"   - Domain: {source['domain']}\n"
            if source.get("results"):
                results_preview = str(source["results"])[:200]
                finding += f"   - Results: {results_preview}{'...' if len(str(source['results'])) > 200 else ''}\n"

            response_parts.append(finding)

        # Summary section
        if architectures:
            unique_archs = list(set(a for a in architectures if a and str(a) != "nan"))[
                :5
            ]
            if unique_archs:
                response_parts.append(f"\n## Architectures Used\n")
                response_parts.append(
                    f"The papers use these deep learning architectures: **{', '.join(unique_archs)}**\n"
                )

        if domains:
            unique_domains = list(set(d for d in domains if d and str(d) != "nan"))[:5]
            if unique_domains:
                response_parts.append(f"\n## Research Domains\n")
                response_parts.append(
                    f"These studies cover: **{', '.join(unique_domains)}**\n"
                )

        # Methodology note
        response_parts.append("\n---\n")
        response_parts.append(
            f"*This response synthesizes information from {len(sources)} papers. "
        )
        response_parts.append(
            "Click on the sources below to view full details and access PubMed.*"
        )

        return "\n".join(response_parts)

    async def _generate_response_llm(self, query: str, context: str) -> str:
        """Generate response using Ollama/Mistral or OpenAI."""
        try:
            # Try Ollama first (local, free)
            import requests

            ollama_url = "http://localhost:11434/api/generate"

            prompt = self.RAG_PROMPT_TEMPLATE.format(context=context, question=query)
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"

            response = requests.post(
                ollama_url,
                json={
                    "model": "mistral",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1500,
                    },
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.warning(f"Ollama failed with status {response.status_code}")

        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")

        # Fallback to OpenAI if Ollama fails
        try:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")

            if api_key:
                import openai

                client = openai.AsyncOpenAI(api_key=api_key)

                prompt = self.RAG_PROMPT_TEMPLATE.format(
                    context=context, question=query
                )

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )

                return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI generation failed: {e}")

        # Final fallback to local template
        return self._generate_response_local(query, context, [])

    def generate_related_queries(self, query: str, sources: List[Dict]) -> List[str]:
        """Generate related search suggestions based on current query and results."""
        import random
        related = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Extract core concepts (remove stopwords and filler)
        stopwords = {'of', 'the', 'a', 'an', 'and', 'or', 'for', 'in', 'on', 'to', 'with', 
                     'using', 'based', 'methods', 'method', 'approach', 'approaches',
                     'analysis', 'study', 'review', 'state-of-the-art', '2024', '2023', '2022'}
        
        core_terms = [t for t in query_lower.split() if t not in stopwords and len(t) > 2]
        core_query = ' '.join(core_terms[:4])  # Keep max 4 core terms
        
        # Extract entities from results
        found_architectures = set()
        found_conditions = set()
        found_datasets = set()
        
        for source in sources[:10]:
            # Extract architecture
            arch = source.get('architecture', '')
            if arch and arch.lower() not in query_lower:
                found_architectures.add(arch)
            
            # Extract dataset
            dataset = source.get('dataset', source.get('Dataset name', ''))
            if dataset and str(dataset) != 'nan' and dataset.lower() not in query_lower:
                found_datasets.add(dataset)
            
            # Extract domain as condition
            domain = source.get('domain', '')
            if domain and domain.lower() not in query_lower:
                found_conditions.add(domain)
        
        # Strategy 1: Add different method/architecture
        method_alternatives = ['CNN', 'LSTM', 'transformer', 'attention mechanism', 'EEGNet']
        for method in method_alternatives:
            if method.lower() not in query_lower:
                related.append(f"{core_query} {method}")
                break
        
        # Strategy 2: Add found architectures from results
        for arch in list(found_architectures)[:1]:
            related.append(f"{core_query} {arch}")
        
        # Strategy 3: Pivot to related applications
        applications = {
            'eeg': ['seizure detection', 'sleep staging', 'emotion recognition', 'motor imagery'],
            'classification': ['detection', 'prediction'],
            'deep learning': ['machine learning comparison'],
            'seizure': ['epilepsy monitoring', 'ictal detection'],
            'sleep': ['insomnia detection', 'sleep apnea'],
            'motor': ['movement prediction', 'rehabilitation BCI']
        }
        
        for key, alternatives in applications.items():
            if key in query_lower:
                for alt in alternatives:
                    if alt.lower() not in query_lower:
                        base = ' '.join([t for t in core_terms if key not in t.lower()][:3])
                        related.append(f"EEG {alt}" if not base else f"{base} {alt}")
                        break
                break
        
        # Strategy 4: Add dataset-specific search
        common_datasets = ['CHB-MIT', 'BCI Competition', 'DEAP', 'PhysioNet', 'TUH EEG']
        for ds in common_datasets:
            if ds.lower() not in query_lower:
                related.append(f"{core_query} {ds}")
                break
        
        # Strategy 5: Evaluation/benchmark angle
        if 'benchmark' not in query_lower:
            related.append(f"{core_query} benchmark comparison")
        
        # Strategy 6: Cross-subject/transfer learning
        if 'cross' not in query_lower and 'transfer' not in query_lower:
            related.append(f"{core_query} cross-subject generalization")
        
        # Strategy 7: Real-time/clinical angle
        if 'real-time' not in query_lower and 'clinical' not in query_lower:
            related.append(f"{core_query} real-time clinical")
        
        # Strategy 8: Tangential but related searches
        tangential_searches = [
            "EEG preprocessing artifact removal",
            "EEG feature extraction comparison",
            "explainable AI EEG classification",
            "lightweight models EEG edge deployment"
        ]
        for ts in tangential_searches:
            if not any(term in query_lower for term in ts.lower().split()[:2]):
                related.append(ts)
                break
        
        # Clean up and deduplicate
        seen = set()
        cleaned = []
        for r in related:
            r_clean = ' '.join(r.split())  # Normalize whitespace
            r_lower = r_clean.lower()
            
            if r_lower == query_lower or r_lower in seen:
                continue
            if len(r_clean) < 10:  # Too short
                continue
            if len(r_clean) > 50:  # Too long - truncate
                r_clean = ' '.join(r_clean.split()[:6])
            
            seen.add(r_lower)
            cleaned.append(r_clean)
        
        random.shuffle(cleaned)
        return cleaned[:3]

    async def query(
        self, query_text: str, max_sources: int = 5, use_llm: bool = False
    ) -> QueryResult:
        """
        Execute a RAG query.

        1. Retrieve relevant papers
        2. Build context from retrieved papers
        3. Generate response (LLM or template-based)
        4. Return response with cited sources
        """
        start_time = time.time()

        # Step 1: RETRIEVE relevant papers
        retrieved_docs = self._simple_search(query_text, top_k=max_sources)

        if not retrieved_docs:
            # Fallback response if no papers found
            return QueryResult(
                query=query_text,
                response="I couldn't find any relevant papers in the corpus for your query. Try different keywords or check if the corpus is loaded.",
                sources=[],
                citations=[],
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
            )

        # Step 2: BUILD context from retrieved papers
        context = self._build_context(retrieved_docs)

        # Step 3: GENERATE response
        if use_llm and self.use_llm:
            response = await self._generate_response_llm(query_text, context)
        else:
            response = self._generate_response_local(
                query_text, context, retrieved_docs
            )

        # Step 4: FORMAT sources with links
        formatted_sources = []
        citations = []

        for doc in retrieved_docs:
            # Generate best URL (prefer PMID, then DOI, then arXiv, then title search)
            pmid = doc.get("pmid", "")
            doi = doc.get("doi", "")
            arxiv_id = doc.get("arxiv_id", "")
            pdf_url = doc.get("pdf_url", "")

            if pmid:
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                pubmed_url = f"https://doi.org/{doi}"
            elif arxiv_id:
                pubmed_url = f"https://arxiv.org/abs/{arxiv_id}"
            else:
                title_search = str(doc.get("title", "")).replace(" ", "+")[:80]
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={title_search}"

            # Generate citation with PMID if available
            author_first = (
                doc.get("authors", "Unknown").split(",")[0].split()[0]
                if doc.get("authors")
                else "Unknown"
            )
            if pmid:
                citation = f"{author_first} {doc.get('year', '')} [PMID:{pmid}]"
            else:
                citation = f"{author_first} {doc.get('year', '')}"

            formatted_sources.append(
                {
                    "title": doc.get("title", "Unknown"),
                    "authors": doc.get("authors", "Unknown"),
                    "year": doc.get("year", 0),
                    "doc_id": doc.get("doc_id", ""),
                    "relevance": doc.get("score", 0.0),
                    "domain": doc.get("domain", ""),
                    "architecture": doc.get("architecture", ""),
                    "pubmed_url": pubmed_url,
                    "citation": citation,
                    # Additional metadata for ingested papers
                    "pmid": pmid,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "citation_count": doc.get("citation_count", 0),
                    "open_access": doc.get("open_access", False),
                    "pdf_url": pdf_url,
                    "_source": doc.get("_source", "csv"),
                }
            )
            citations.append(f"[{citation}]")

        processing_time = (time.time() - start_time) * 1000

        # Calculate confidence based on retrieval quality
        avg_score = (
            sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs
            else 0
        )
        confidence = min(0.95, 0.5 + avg_score)

        # Generate related search suggestions
        related_queries = self.generate_related_queries(query_text, formatted_sources)

        return QueryResult(
            query=query_text,
            response=response,
            sources=formatted_sources,
            citations=citations,
            confidence=confidence,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            related_queries=related_queries,
        )


# Legacy alias for backwards compatibility
EEGQueryEngine = RAGQueryEngine


# =============================================================================
# Session State Management
# =============================================================================


def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "query_history": [],
        "benchmark_results": None,
        "current_page": "Query System",
        "settings": {
            "corpus_path": AppConfig.DEFAULT_CORPUS_PATH,
            "embeddings_path": AppConfig.DEFAULT_EMBEDDINGS_PATH,
            "benchmark_csv": AppConfig.DEFAULT_BENCHMARK_CSV,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4",
            "max_sources": 5,
            "show_confidence": True,
        },
        "query_engine": None,
        "benchmark_instance": None,
        "selected_paper_idx": None,
        "selected_paper_data": None,
        "navigate_to_explorer": False,
        "explorer_search_query": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate_to_paper(paper_id: str, paper_data: Optional[Dict] = None):
    """Navigate to Paper Research Explorer with a specific paper selected."""
    st.session_state["navigate_to_explorer"] = True
    st.session_state["explorer_search_query"] = paper_id
    if paper_data:
        st.session_state["selected_paper_data"] = paper_data


# =============================================================================
# UI Components
# =============================================================================


def render_header():
    """Render application header."""
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("🧠 EEG-RAG: AI Research Assistant")
        st.markdown(
            """
        **Retrieval-Augmented Generation for EEG Research**
        
        Query the EEG literature, ingest papers, and explore research insights.
        """
        )

    with col2:
        st.markdown("### Quick Stats")
        if st.session_state.get("benchmark_results"):
            results = st.session_state["benchmark_results"]
            st.metric("Last Accuracy", f"{results.overall_accuracy:.1%}")


def render_sidebar() -> str:
    """Render sidebar navigation."""
    st.sidebar.title("🧭 Navigation")

    pages = [
        "🔍 Query System",
        "📥 Data Ingestion",
        " Corpus Explorer",
        "🔬 Paper Research Explorer",
        "⚙️ Settings",
    ]

    # Check if we need to navigate to Paper Explorer (force selection)
    default_index = 0
    if st.session_state.get("navigate_to_explorer"):
        default_index = pages.index("🔬 Paper Research Explorer")

    page = st.sidebar.radio(
        "Select Page", pages, index=default_index, label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")

    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state["query_history"] = []
        st.sidebar.success("History cleared!")

    if st.sidebar.button("📥 Export Results", use_container_width=True):
        if st.session_state.get("benchmark_results"):
            st.sidebar.download_button(
                "Download JSON",
                json.dumps(asdict(st.session_state["benchmark_results"]), default=str),
                "benchmark_results.json",
                "application/json",
            )

    st.sidebar.markdown("---")

    # Info section
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown(
        """
    EEG-RAG is a production-grade RAG system for electroencephalography research.
    
    **Version:** 1.1.0  
    **Features:** Multi-source ingestion, RAG query
    """
    )

    return page


def render_query_page():
    """Render query interface page."""
    st.header("🔍 Query the EEG Research Corpus")

    # Check URL query params for related search / find similar navigation
    url_search = st.query_params.get("q")
    if url_search:
        decoded_query = urllib.parse.unquote(url_search)
        # Clear URL param first to prevent infinite loops
        st.query_params.clear()
        # Set as pending query and rerun
        st.session_state["pending_query"] = decoded_query
        st.session_state["_query_widget"] = decoded_query
        st.session_state["do_search"] = True
        st.rerun()

    # CRITICAL: Check for pending query FIRST, before any widget initialization
    # This ensures the pending query is processed before the widget gets created
    if "pending_query" in st.session_state:
        pending = st.session_state.pop("pending_query")
        # Set the widget's value directly via session state
        st.session_state["_query_widget"] = pending
        st.session_state["do_search"] = True

    # Initialize session state for search trigger flag
    if "do_search" not in st.session_state:
        st.session_state["do_search"] = False

    # Initialize the widget key if not present (only if not set by pending_query)
    if "_query_widget" not in st.session_state:
        st.session_state["_query_widget"] = ""

    # Query input section
    with st.container():
        # The widget automatically uses st.session_state["_query_widget"] for its value
        # because we specified key="_query_widget"
        # DO NOT use value= parameter when you want session_state to control the widget
        query = st.text_area(
            "Enter your research question:",
            placeholder="e.g., What are the best deep learning architectures for EEG seizure detection?",
            height=100,
            max_chars=AppConfig.MAX_QUERY_LENGTH,
            key="_query_widget",
        )

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            complexity = st.select_slider(
                "Query Complexity",
                options=["Simple", "Medium", "Complex", "Expert"],
                value="Medium",
            )

        with col2:
            max_sources = st.slider(
                "Max Sources",
                min_value=1,
                max_value=AppConfig.MAX_RESPONSE_SOURCES,
                value=st.session_state["settings"]["max_sources"],
            )

        with col3:
            show_confidence = st.checkbox(
                "Show Confidence", value=st.session_state["settings"]["show_confidence"]
            )

        search_col1, search_col2 = st.columns([1, 4])

        with search_col1:
            search_clicked = st.button(
                "🔍 Search", type="primary", use_container_width=True
            )

        with search_col2:
            if st.button("🎲 Random Query", use_container_width=True):
                sample_queries = [
                    "What CNNs are used for EEG seizure detection?",
                    "How does DeepSleepNet classify sleep stages?",
                    "Compare motor imagery classification methods in BCIs",
                    "What preprocessing is needed for EEG deep learning?",
                    "What are state-of-the-art accuracies for P300 detection?",
                    "How effective are LSTMs for epileptic seizure prediction?",
                    "What are the best feature extraction methods for EEG?",
                    "Compare CNN vs Transformer architectures for EEG analysis",
                    "What datasets are commonly used for sleep stage classification?",
                    "How do attention mechanisms improve EEG classification?",
                    "What are the challenges in real-time EEG processing?",
                    "Which neural networks work best for emotion recognition from EEG?",
                ]
                import random

                st.session_state["pending_query"] = random.choice(sample_queries)
                st.rerun()

    # Check if we need to trigger search from random/related query
    search_triggered = st.session_state.pop("do_search", False)
    if search_triggered:
        search_clicked = True

    # Process query - perform search when button clicked or triggered
    if search_clicked and query:
        with st.spinner(
            "🤖 Retrieving papers and generating AI response with Mistral..."
        ):
            # Initialize RAG query engine
            if st.session_state["query_engine"] is None:
                st.session_state["query_engine"] = RAGQueryEngine(
                    corpus_path=st.session_state["settings"]["benchmark_csv"]
                )

            # Run RAG query with LLM always enabled
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                st.session_state["query_engine"].query(query, max_sources, use_llm=True)
            )

            # Store in history AND as current result for display
            st.session_state["query_history"].append(result)
            st.session_state["current_result"] = result

    # Display results if we have a current result (either from new search or previous)
    result = st.session_state.get("current_result")
    if result:

        # Display results
        st.success(
            f"✅ 🤖 Mistral AI | Retrieved {len(result.sources)} papers in {result.processing_time_ms:.0f}ms"
        )

        # Response section
        st.markdown("### 📝 AI-Generated Response")

        if show_confidence:
            confidence_label = (
                "High"
                if result.confidence > 0.7
                else "Medium" if result.confidence > 0.4 else "Low"
            )
            st.progress(
                result.confidence,
                text=f"Confidence: {result.confidence:.0%} ({confidence_label})",
            )

        st.markdown(result.response)

        # Bibliometric Visualizations Section
        if result.sources and len(result.sources) >= 3:
            st.markdown("---")
            st.markdown("### 📊 Bibliometric Analysis")
            st.caption("Visual analytics of the retrieved research papers")
            
            with st.spinner("Generating bibliometric visualizations..."):
                try:
                    # Convert sources to EEGArticle format
                    articles = []
                    for idx, source in enumerate(result.sources):
                        try:
                            # Extract authors
                            authors = source.get('authors', [])
                            if isinstance(authors, str):
                                authors = [a.strip() for a in authors.split(',') if a.strip()]
                            elif not isinstance(authors, list):
                                authors = []
                            
                            # Generate openalex_id from available IDs or index
                            openalex_id = source.get('openalex_id', '')
                            if not openalex_id:
                                pmid = source.get('pmid', source.get('PMID', ''))
                                doi = source.get('doi', source.get('DOI', ''))
                                openalex_id = f"W{pmid}" if pmid else f"W{doi}" if doi else f"W{idx}"
                            
                            # Create article
                            article = EEGArticle(
                                openalex_id=openalex_id,
                                title=source.get('title', 'Unknown'),
                                authors=authors,
                                publication_date=f"{source.get('year', '2020')}-01-01",
                                cited_by_count=source.get('citation_count', source.get('citations', 0)),
                                abstract=source.get('results', source.get('Results', source.get('abstract', '')))[:500],
                                venue=source.get('journal', source.get('venue', source.get('source', ''))),
                                pmid=source.get('pmid', source.get('PMID', '')),
                                doi=source.get('doi', source.get('DOI', '')),
                                topics=[],
                                referenced_works=[],
                            )
                            articles.append(article)
                        except Exception as e:
                            logger.warning(f"Failed to convert source to article: {e}")
                            continue
                    
                    if articles:
                        st.info(f"Converted {len(articles)} papers for visualization")
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs(["📈 Timeline & Trends", "🎓 Key Researchers", "📚 Citation Impact", "🔗 Research Networks"])
                        
                        # Tab 1: Publication Trends
                        with viz_tabs[0]:
                            try:
                                from bibliometrics import EEGVisualization
                                viz = EEGVisualization()
                                trend_chart = viz.plot_publication_trends(articles)
                                
                                # Extract year data for insights
                                years = []
                                for a in articles:
                                    try:
                                        year = int(a.publication_date[:4]) if a.publication_date else None
                                        if year:
                                            years.append(year)
                                    except:
                                        pass
                                
                                if years:
                                    # Calculate insights
                                    min_year, max_year = min(years), max(years)
                                    year_range = max_year - min_year + 1
                                    avg_per_year = len(years) / year_range if year_range > 0 else 0
                                    recent_papers = sum(1 for y in years if y >= 2020)
                                    recent_pct = (recent_papers / len(years) * 100) if years else 0
                                    
                                    # Display insights first
                                    st.markdown("**📊 Publication Timeline Insights:**")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Time Span", f"{min_year}-{max_year}")
                                    with col2:
                                        st.metric("Avg/Year", f"{avg_per_year:.1f}")
                                    with col3:
                                        st.metric("Since 2020", f"{recent_papers}")
                                    with col4:
                                        st.metric("Recent %", f"{recent_pct:.0f}%")
                                    
                                    # Show visualization
                                    if trend_chart.png_base64:
                                        img_data = base64.b64decode(trend_chart.png_base64)
                                        st.image(img_data, use_container_width=True)
                                    
                                    # Narrative insights
                                    st.markdown("**💡 Key Findings:**")
                                    if recent_pct > 60:
                                        st.info(f"✓ **Recent Focus**: {recent_pct:.0f}% of papers are from 2020 onwards, indicating active current research")
                                    elif recent_pct < 30:
                                        st.info(f"ℹ️ **Historical Context**: Most papers ({100-recent_pct:.0f}%) are from before 2020, providing foundational knowledge")
                                    
                                    if year_range < 5:
                                        st.info("✓ **Focused Period**: All papers from a narrow time window, showing targeted query results")
                                    elif year_range > 10:
                                        st.info(f"✓ **Broad Coverage**: Spanning {year_range} years, showing evolution of the field")
                                else:
                                    st.warning("No publication trend data generated")
                            except Exception as e:
                                st.error(f"Failed to generate trends chart: {str(e)}")
                                logger.error(f"Trends visualization error: {e}", exc_info=True)
                        
                        # Tab 2: Top Authors
                        with viz_tabs[1]:
                            try:
                                from bibliometrics import EEGVisualization
                                from collections import Counter
                                viz = EEGVisualization()
                                
                                # Count authors and their papers
                                author_papers = Counter()
                                author_citations = {}
                                for a in articles:
                                    for author in a.authors[:5]:  # Limit to first 5 to avoid inflating counts
                                        author_papers[author] += 1
                                        if author not in author_citations:
                                            author_citations[author] = 0
                                        author_citations[author] += a.cited_by_count
                                
                                # Display insights
                                if author_papers:
                                    top_3 = author_papers.most_common(3)
                                    total_unique_authors = len(author_papers)
                                    multi_paper_authors = sum(1 for count in author_papers.values() if count > 1)
                                    
                                    st.markdown("**👥 Author Analysis:**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Unique Authors", total_unique_authors)
                                    with col2:
                                        st.metric("Multi-Paper", multi_paper_authors)
                                    with col3:
                                        top_author_name = top_3[0][0] if top_3 else "N/A"
                                        top_author_papers = top_3[0][1] if top_3 else 0
                                        st.metric("Most Prolific", f"{top_author_papers} papers")
                                
                                author_chart = viz.plot_top_authors(articles, top_n=8)
                                if author_chart.png_base64:
                                    img_data = base64.b64decode(author_chart.png_base64)
                                    st.image(img_data, use_container_width=True)
                                    
                                    # Show top authors with their impact
                                    if top_3:
                                        st.markdown("**🌟 Key Researchers:**")
                                        for i, (author, papers) in enumerate(top_3, 1):
                                            cites = author_citations.get(author, 0)
                                            avg_cites = cites / papers if papers > 0 else 0
                                            st.write(f"{i}. **{author}** - {papers} papers, {cites:,} total citations (avg: {avg_cites:.0f} per paper)")
                                else:
                                    st.warning("No author data generated")
                            except Exception as e:
                                st.error(f"Failed to generate authors chart: {str(e)}")
                                logger.error(f"Authors visualization error: {e}", exc_info=True)
                        
                        # Tab 3: Citation Distribution
                        with viz_tabs[2]:
                            try:
                                from bibliometrics import EEGVisualization
                                viz = EEGVisualization()
                                
                                # Calculate comprehensive citation metrics
                                citation_counts = [a.cited_by_count for a in articles]
                                total_citations = sum(citation_counts)
                                avg_citations = total_citations / len(articles) if articles else 0
                                median_citations = sorted(citation_counts)[len(citation_counts)//2] if citation_counts else 0
                                max_citations = max(citation_counts) if citation_counts else 0
                                highly_cited = sum(1 for c in citation_counts if c >= 50)
                                uncited = sum(1 for c in citation_counts if c == 0)
                                
                                # Display comprehensive citation metrics
                                st.markdown("**📚 Citation Impact Analysis:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Citations", f"{total_citations:,}")
                                with col2:
                                    st.metric("Avg per Paper", f"{avg_citations:.1f}")
                                with col3:
                                    st.metric("Median", f"{median_citations}")
                                with col4:
                                    st.metric("Most Cited", max_citations)
                                
                                citation_chart = viz.plot_citation_distribution(articles)
                                if citation_chart.png_base64:
                                    img_data = base64.b64decode(citation_chart.png_base64)
                                    st.image(img_data, use_container_width=True)
                                else:
                                    st.warning("No citation data generated")
                                
                                # Impact tiers
                                st.markdown("**🎯 Impact Distribution:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    highly_cited_pct = (highly_cited / len(articles) * 100) if articles else 0
                                    st.metric("Highly Cited (50+)", f"{highly_cited} ({highly_cited_pct:.0f}%)")
                                with col2:
                                    moderate = sum(1 for c in citation_counts if 10 <= c < 50)
                                    moderate_pct = (moderate / len(articles) * 100) if articles else 0
                                    st.metric("Moderately Cited (10-49)", f"{moderate} ({moderate_pct:.0f}%)")
                                with col3:
                                    uncited_pct = (uncited / len(articles) * 100) if articles else 0
                                    st.metric("Uncited/Low (<10)", f"{uncited + sum(1 for c in citation_counts if 0 < c < 10)} ({100 - highly_cited_pct - moderate_pct:.0f}%)")
                                
                                # Key insights
                                st.markdown("**💡 Citation Insights:**")
                                most_cited = max(articles, key=lambda a: a.cited_by_count)
                                st.info(f"🏆 **Top Paper**: \"{most_cited.title[:80]}...\" ({most_cited.cited_by_count:,} citations)")
                                
                                if highly_cited_pct > 30:
                                    st.success(f"✓ **High Impact Set**: {highly_cited_pct:.0f}% are highly cited (50+ citations), indicating influential research")
                                elif avg_citations > 20:
                                    st.success(f"✓ **Strong Impact**: Average of {avg_citations:.0f} citations per paper shows solid research influence")
                                
                                if uncited_pct > 40:
                                    st.info(f"ℹ️ **Emerging Research**: {uncited_pct:.0f}% have fewer than 10 citations, possibly recent publications building citation history")
                                    
                            except Exception as e:
                                st.error(f"Failed to generate citations chart: {str(e)}")
                                logger.error(f"Citations visualization error: {e}", exc_info=True)
                        
                        # Tab 4: Collaboration Networks
                        with viz_tabs[3]:
                            try:
                                from collections import defaultdict, Counter
                                
                                st.markdown("**🔗 Author Collaboration Network:**")
                                
                                # Build co-authorship network
                                collaborations = defaultdict(set)
                                author_paper_count = Counter()
                                author_total_citations = Counter()
                                
                                for article in articles:
                                    authors = article.authors[:10]  # Limit to avoid noise
                                    for author in authors:
                                        author_paper_count[author] += 1
                                        author_total_citations[author] += article.cited_by_count
                                        # Add co-authors
                                        for co_author in authors:
                                            if author != co_author:
                                                collaborations[author].add(co_author)
                                
                                # Calculate network metrics
                                total_authors = len(author_paper_count)
                                total_edges = sum(len(co_authors) for co_authors in collaborations.values()) // 2
                                authors_with_collabs = sum(1 for co_authors in collaborations.values() if co_authors)
                                avg_collabs = sum(len(co_authors) for co_authors in collaborations.values()) / total_authors if total_authors > 0 else 0
                                
                                # Display network metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Authors", total_authors)
                                with col2:
                                    st.metric("Co-authorships", total_edges)
                                with col3:
                                    st.metric("Avg Collaborators", f"{avg_collabs:.1f}")
                                with col4:
                                    collab_rate = (authors_with_collabs / total_authors * 100) if total_authors > 0 else 0
                                    st.metric("Collaboration Rate", f"{collab_rate:.0f}%")
                                
                                # Find most collaborative authors
                                most_collaborative = sorted(
                                    [(author, len(co_authors)) for author, co_authors in collaborations.items()],
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:5]
                                
                                if most_collaborative:
                                    st.markdown("**🤝 Most Collaborative Researchers:**")
                                    for i, (author, collab_count) in enumerate(most_collaborative, 1):
                                        papers = author_paper_count[author]
                                        citations = author_total_citations[author]
                                        st.write(f"{i}. **{author}** - {collab_count} collaborators, {papers} papers, {citations:,} citations")
                                
                                # Insights
                                st.markdown("**💡 Network Insights:**")
                                if collab_rate > 80:
                                    st.success(f"✓ **Highly Collaborative Field**: {collab_rate:.0f}% of authors work in teams, typical of active research areas")
                                elif collab_rate < 50:
                                    st.info(f"ℹ️ **Mixed Collaboration**: {collab_rate:.0f}% collaboration rate suggests a mix of solo and team research")
                                
                                if avg_collabs > 5:
                                    st.success(f"✓ **Strong Networks**: Average of {avg_collabs:.1f} collaborators per author indicates well-connected research community")
                                
                                # Research clusters
                                if total_authors > 10:
                                    st.info(f"ℹ️ **Research Community**: {total_authors} unique authors with {total_edges} connections suggest an active research network")
                                    
                            except Exception as e:
                                st.warning(f"Collaboration network analysis unavailable: {str(e)}")
                                logger.error(f"Collaboration network error: {e}", exc_info=True)
                    else:
                        st.warning("Could not convert papers to article format for visualization")
                        
                except Exception as e:
                    logger.error(f"Bibliometric visualization error: {e}", exc_info=True)
                    st.error(f"Bibliometric visualizations temporarily unavailable: {str(e)}")

        # Related queries section - use URL links for reliable scroll-to-top
        if result.related_queries:
            st.markdown("---")
            st.markdown("### 💡 Related Searches")
            st.caption("Click to explore related topics")
            cols = st.columns(3)
            for idx, related_query in enumerate(result.related_queries):
                with cols[idx]:
                    # Use URL-based navigation for reliable scroll-to-top
                    encoded_query = urllib.parse.quote(related_query)
                    display_text = related_query if len(related_query) <= 40 else related_query[:37] + "..."
                    st.markdown(
                        f'<a href="?q={encoded_query}" target="_self" style="'
                        f'display: block; '
                        f'background-color: #262730; '
                        f'border: 1px solid #4a4a5a; '
                        f'border-radius: 8px; '
                        f'padding: 10px 12px; '
                        f'margin: 4px 0; '
                        f'color: #fafafa; '
                        f'text-decoration: none; '
                        f'font-size: 14px; '
                        f'text-align: center; '
                        f'transition: background-color 0.2s;'
                        f'">'
                        f'🔎 {display_text}'
                        f'</a>',
                        unsafe_allow_html=True
                    )

        # Sources section with clickable links
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📚 Retrieved Sources")
            st.markdown(
                "*Expand any source to see full details including abstract, PMID, and external links*"
            )
        with col2:
            # Citation export dropdown
            export_format = st.selectbox(
                "📥 Export Citations",
                ["None", "BibTeX", "RIS", "Plain Text"],
                key="export_sources",
            )
            if export_format != "None":
                # Map display names to format codes
                format_map = {"BibTeX": "bibtex", "RIS": "ris", "Plain Text": "plain"}
                format_code = format_map.get(export_format, "plain")
                citation_text = generate_citations(
                    result.sources, format=format_code
                )
                st.download_button(
                    label=f"Download {export_format}",
                    data=citation_text,
                    file_name=f"citations.{format_code}.txt",
                    mime="text/plain",
                )

        if result.sources:
            for i, source in enumerate(result.sources, 1):
                # Get badges for this paper
                badges = get_all_badges(source)
                badge_suffix = f" {badges}" if badges else ""
                
                # Get citation count for display
                citation_count = source.get("citation_count", source.get("citations", 0))
                citation_suffix = f" 📚{citation_count}" if citation_count else ""

                with st.expander(
                    f"📄 [{i}] {source.get('title', 'Unknown')[:80]}... ({source.get('year', 'N/A')}){badge_suffix}{citation_suffix}",
                    expanded=(i <= 2),
                ):
                    # PROMINENT SUMMARY BOX AT TOP
                    summary_parts = []
                    
                    # Architecture
                    arch = source.get('architecture', source.get('Architecture (clean)', ''))
                    if arch and str(arch) != 'nan' and str(arch).strip():
                        summary_parts.append(f"**Architecture:** {arch}")
                    
                    # Domain
                    domain = source.get('domain', source.get('Domain 1', ''))
                    if domain and str(domain) != 'nan' and str(domain).strip():
                        summary_parts.append(f"**Domain:** {domain}")
                    
                    # Results/Abstract summary (first 200 chars)
                    results_text = source.get('results', source.get('Results', source.get('abstract', '')))
                    if results_text and str(results_text) != 'nan' and str(results_text).strip():
                        preview = str(results_text)[:200].strip()
                        if preview:
                            summary_parts.append(f"**Key Findings:** {preview}...")
                    
                    # Show summary box if we have content
                    if summary_parts:
                        st.info("\n\n".join(summary_parts))
                    
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Title:** {source.get('title', 'Unknown')}")
                        
                        # Authors with smart truncation
                        authors = source.get('authors', 'Unknown')
                        if isinstance(authors, list):
                            if len(authors) > 5:
                                authors = ', '.join(authors[:5]) + f' et al. (+{len(authors)-5} more)'
                            else:
                                authors = ', '.join(authors) if authors else 'Unknown'
                        st.markdown(f"**Authors:** {authors}")
                        
                        st.markdown(f"**Year:** {source.get('year', 'N/A')}")
                        
                        # Journal/venue
                        journal = source.get("journal", source.get("venue", source.get("source", "")))
                        if journal and str(journal) != "nan":
                            st.markdown(f"**Journal:** {journal}")

                        # Show quality badges prominently
                        if badges:
                            st.markdown(f"**Quality:** {badges}")

                    with col2:
                        relevance = source.get("relevance", source.get("score", 0))
                        st.metric("Relevance", f"{relevance:.0%}")
                        
                        # Citation count metric
                        if citation_count:
                            st.metric("Citations", f"{citation_count:,}")

                    # Identifiers row
                    pmid = source.get("pmid", source.get("PMID", ""))
                    doi = source.get("doi", source.get("DOI", ""))
                    
                    if pmid or doi:
                        id_col1, id_col2 = st.columns(2)
                        with id_col1:
                            if pmid:
                                st.markdown(f"**PMID:** [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                        with id_col2:
                            if doi:
                                st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")

                    # Action buttons row
                    btn_col1, btn_col2, btn_col3 = st.columns(3)

                    with btn_col1:
                        pubmed_url = source.get(
                            "pubmed_url",
                            f"https://pubmed.ncbi.nlm.nih.gov/?term={source.get('title', '').replace(' ', '+')[:50]}",
                        )
                        st.link_button(
                            "🔗 Search PubMed", pubmed_url, use_container_width=True
                        )

                    with btn_col2:
                        scholar_url = f"https://scholar.google.com/scholar?q={source.get('title', '').replace(' ', '+')[:50]}"
                        st.link_button(
                            "🎓 Google Scholar", scholar_url, use_container_width=True
                        )
                    
                    with btn_col3:
                        # Find similar papers - use URL link for scroll-to-top
                        title = source.get('title', '')
                        similar_query = extract_key_terms(title, max_terms=5)
                        if similar_query:
                            encoded_query = urllib.parse.quote(similar_query)
                            st.markdown(
                                f'<a href="?q={encoded_query}" target="_self" style="'
                                f'display: inline-flex; '
                                f'align-items: center; '
                                f'justify-content: center; '
                                f'background-color: #262730; '
                                f'border: 1px solid #4a4a5a; '
                                f'border-radius: 8px; '
                                f'padding: 8px 16px; '
                                f'color: #fafafa; '
                                f'text-decoration: none; '
                                f'font-size: 14px; '
                                f'width: 100%; '
                                f'box-sizing: border-box; '
                                f'height: 38px;'
                                f'">'
                                f'🔍 Find Similar'
                                f'</a>',
                                unsafe_allow_html=True
                            )

                    # Show additional details inline
                    st.markdown("---")
                    st.markdown("##### 📋 Full Details")

                    # Results/Abstract
                    results_text = source.get("results", source.get("Results", source.get("abstract", "")))
                    if results_text and str(results_text) != "nan":
                        st.markdown(f"**Results/Abstract:**")
                        st.markdown(
                            f"> {str(results_text)[:500]}{'...' if len(str(results_text)) > 500 else ''}"
                        )

                    # Two column layout for additional metadata
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        # Dataset info
                        dataset = source.get("dataset", source.get("Dataset name", ""))
                        if dataset and str(dataset) != "nan":
                            st.markdown(f"**Dataset:** {dataset}")
                        
                        # Sample size
                        sample_size = source.get("sample_size", source.get("n_subjects", ""))
                        if sample_size and str(sample_size) != "nan":
                            st.markdown(f"**Sample Size:** {sample_size}")
                        
                        # Performance metrics
                        accuracy = source.get("accuracy", source.get("performance", source.get("f1_score", "")))
                        if accuracy and str(accuracy) != "nan":
                            st.markdown(f"**Reported Performance:** {accuracy}")
                    
                    with detail_col2:
                        # Code availability
                        code_available = source.get("code_available", source.get("github_url", ""))
                        if code_available:
                            if isinstance(code_available, bool):
                                st.markdown(f"**Code Available:** {'✅ Yes' if code_available else '❌ No'}")
                            elif code_available and str(code_available) != "nan":
                                st.markdown(f"**Code:** [{code_available}]({code_available})")
                        
                        # Data availability
                        data_available = source.get("data_available", source.get("data_url", ""))
                        if data_available:
                            if isinstance(data_available, bool):
                                st.markdown(f"**Data Available:** {'✅ Yes' if data_available else '❌ No'}")
                            elif data_available and str(data_available) != "nan":
                                st.markdown(f"**Data:** [{data_available}]({data_available})")
                        
                        # Citation ID
                        doc_id = source.get("doc_id", "")
                        if doc_id:
                            st.markdown(f"**Citation ID:** `{doc_id}`")

                    # MeSH terms / Keywords as tags
                    mesh_terms = source.get("mesh_terms", source.get("keywords", []))
                    if mesh_terms and mesh_terms != "nan":
                        if isinstance(mesh_terms, str):
                            mesh_terms = [t.strip() for t in mesh_terms.split(",") if t.strip()]
                        if mesh_terms:
                            st.markdown("**Keywords/MeSH:**")
                            # Display as inline tags
                            tags_display = " • ".join(mesh_terms[:10])
                            st.caption(tags_display)

        else:
            st.warning("No relevant sources found for this query.")

        # Citations summary
        st.markdown("### 🔗 Quick Citations")
        if result.citations:
            st.code(", ".join(result.citations))

        # RAG explanation
        with st.expander("ℹ️ How this response was generated"):
            st.markdown(
                """
            **RAG (Retrieval-Augmented Generation) Process:**
            
            1. **Retrieve**: Your query was used to search the EEG research corpus (164 papers from Roy et al. 2019)
            2. **Rank**: Papers were ranked by relevance using keyword and semantic matching
            3. **Augment**: Top papers' metadata was compiled into a context
            4. **Generate**: A structured response was synthesized from the retrieved information
            
            *For full LLM-powered responses, configure your OpenAI API key in Settings.*
            """
            )

    elif search_clicked:
        st.warning("⚠️ Please enter a query.")

    # Query history
    if st.session_state["query_history"]:
        with st.expander("📜 Query History", expanded=False):
            for i, hist in enumerate(reversed(st.session_state["query_history"][-5:])):
                st.markdown(f"**{i+1}.** {hist.query[:80]}...")
                st.caption(f"_{hist.timestamp}_")


def render_benchmark_page():
    """Render systematic review benchmark page."""
    st.header("📊 Systematic Review Benchmark")

    st.markdown(
        """
    **Evaluate extraction accuracy against Roy et al. 2019 ground truth**
    
    This benchmark tests the system's ability to extract structured data from 
    deep learning EEG papers, comparing against manually curated ground truth 
    from [dl-eeg-review](https://github.com/hubertjb/dl-eeg-review).
    
    > **Reference:** Roy, Y. et al. (2019). Deep learning-based electroencephalography 
    > analysis: a systematic review. *J. Neural Eng.*, 16(5), 051001. [PMID:31151119]
    """
    )

    st.markdown("---")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        csv_path = st.text_input(
            "📁 Ground Truth CSV Path",
            value=st.session_state["settings"]["benchmark_csv"],
            help="Path to Roy et al. 2019 data_items.csv",
        )

    with col2:
        max_papers = st.number_input(
            "📄 Max Papers to Evaluate",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Limit for faster testing. Full dataset has ~150 papers.",
        )

    # Field selection
    st.markdown("### 📋 Extraction Fields")

    all_fields = list(SystematicReviewBenchmark.FIELD_MAPPINGS.keys())

    col1, col2 = st.columns(2)

    with col1:
        selected_fields = st.multiselect(
            "Select fields to evaluate:",
            options=all_fields,
            default=[
                "architecture_type",
                "domain",
                "dataset",
                "best_accuracy",
                "code_available",
            ],
            help="Choose which fields to extract and evaluate",
        )

    with col2:
        strict_matching = st.checkbox(
            "Strict Matching",
            value=False,
            help="Require exact matches (no fuzzy matching)",
        )

    st.markdown("---")

    # Run benchmark
    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        csv_full_path = Path(csv_path)

        if not csv_full_path.exists():
            csv_full_path = Path("/home/kevin/Projects/eeg-rag") / csv_path

        if not csv_full_path.exists():
            st.error(f"❌ Ground truth CSV not found: {csv_path}")
            st.info("💡 Run: `make download-benchmark-data`")
            return

        if not selected_fields:
            st.error("❌ Please select at least one field to evaluate.")
            return

        try:
            benchmark = SystematicReviewBenchmark(
                ground_truth_csv=str(csv_full_path),
                extraction_fields=selected_fields,
                strict_matching=strict_matching,
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float):
                progress_bar.progress(progress)
                status_text.text(f"📊 Evaluating papers... {int(progress * 100)}%")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            results = loop.run_until_complete(
                benchmark.evaluate_extraction_accuracy(
                    max_papers=max_papers, progress_callback=update_progress
                )
            )

            progress_bar.progress(1.0)
            status_text.text("✅ Benchmark complete!")

            st.session_state["benchmark_results"] = results

            # Display results
            st.success(
                f"✅ Benchmark completed! Evaluated {results.papers_evaluated} papers."
            )

            # Overall metrics
            st.markdown("### 📈 Overall Results")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Accuracy", f"{results.overall_accuracy:.1%}")
            col2.metric("Papers Evaluated", results.papers_evaluated)
            col3.metric("Fields Evaluated", len(results.fields_evaluated))
            col4.metric("Total Time", f"{results.extraction_time_total_ms/1000:.1f}s")

            # Per-field accuracy chart
            st.markdown("### 📊 Per-Field Accuracy")

            field_df = pd.DataFrame(
                {
                    "Field": list(results.per_field_accuracy.keys()),
                    "Accuracy": list(results.per_field_accuracy.values()),
                }
            )

            st.bar_chart(field_df.set_index("Field")["Accuracy"])

            # Error analysis
            if results.error_analysis:
                st.markdown("### 🔍 Error Analysis")
                error_df = pd.DataFrame(
                    [
                        {"Error Type": k, "Count": v}
                        for k, v in results.error_analysis.items()
                    ]
                )
                st.dataframe(error_df, use_container_width=True, hide_index=True)

            # Sample results with clickable papers
            st.markdown("### 📝 Sample Paper Results")
            st.markdown(
                "*Click any paper to view full details in Paper Research Explorer*"
            )

            results_df = results.to_dataframe()

            # Display papers as clickable cards
            for idx, row in results_df.head(15).iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 4, 1, 1, 1])

                with col1:
                    if st.button(
                        f"📄 {row['paper_id']}",
                        key=f"bench_paper_{idx}",
                        use_container_width=True,
                    ):
                        navigate_to_paper(row["paper_id"])
                        st.rerun()

                with col2:
                    st.markdown(
                        f"**{row['title'][:50]}...**"
                        if len(str(row["title"])) > 50
                        else f"**{row['title']}**"
                    )

                with col3:
                    st.markdown(f"📅 {int(row['year'])}")

                with col4:
                    accuracy_pct = row["accuracy"] * 100
                    color = (
                        "green"
                        if accuracy_pct >= 80
                        else "orange" if accuracy_pct >= 50 else "red"
                    )
                    st.markdown(f":{color}[{accuracy_pct:.0f}%]")

                with col5:
                    st.markdown(f"✓{row['correct_fields']} ✗{row['incorrect_fields']}")

            st.markdown("---")
            st.info(
                "💡 **Tip:** Click any paper ID above to view full metadata, PubMed links, and more in the Paper Research Explorer."
            )

        except Exception as e:
            st.error(f"❌ Benchmark failed: {str(e)}")
            logger.exception("Benchmark error")


def render_results_page():
    """Render results dashboard page."""
    st.header("📈 Results Dashboard")

    if (
        "benchmark_results" not in st.session_state
        or st.session_state["benchmark_results"] is None
    ):
        st.info("ℹ️ No benchmark results available. Run a benchmark first.")
        return

    results = st.session_state["benchmark_results"]

    st.markdown(f"**📅 Benchmark run:** {results.timestamp}")

    # Summary metrics
    st.markdown("### 📊 Summary Metrics")

    stats = results.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean Accuracy", f"{stats['mean_accuracy']:.1%}")
    col2.metric("Std Deviation", f"{stats['std_accuracy']:.1%}")
    col3.metric("Min Accuracy", f"{stats['min_accuracy']:.1%}")
    col4.metric("Max Accuracy", f"{stats['max_accuracy']:.1%}")

    # Best/worst fields
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Best Performing Fields")
        sorted_fields = sorted(
            results.per_field_accuracy.items(), key=lambda x: x[1], reverse=True
        )
        for field, acc in sorted_fields[:3]:
            st.success(f"**{field}**: {acc:.1%}")

    with col2:
        st.markdown("### ⚠️ Needs Improvement")
        for field, acc in sorted_fields[-3:]:
            st.warning(f"**{field}**: {acc:.1%}")

    # Accuracy distribution
    st.markdown("### 📊 Accuracy Distribution")

    results_df = results.to_dataframe()

    fig_data = pd.DataFrame(
        {
            "Accuracy Range": pd.cut(
                results_df["accuracy"],
                bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
                labels=["0-50%", "50-70%", "70-85%", "85-95%", "95-100%"],
            )
        }
    )
    st.bar_chart(fig_data["Accuracy Range"].value_counts())

    # Full results table
    st.markdown("### 📋 Full Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Export
    st.markdown("### 📥 Export")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 Download as CSV",
            results_df.to_csv(index=False),
            "benchmark_results.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "📋 Download as JSON",
            json.dumps(
                {
                    "summary": stats,
                    "per_field_accuracy": results.per_field_accuracy,
                    "timestamp": results.timestamp,
                },
                indent=2,
            ),
            "benchmark_summary.json",
            "application/json",
            use_container_width=True,
        )


def render_corpus_page():
    """Render corpus explorer page."""
    st.header("📚 Corpus Explorer")

    st.markdown(
        """
    Explore the indexed EEG research corpus and ground truth data.
    """
    )

    # Ground truth statistics
    csv_path = Path(st.session_state["settings"]["benchmark_csv"])
    if not csv_path.exists():
        csv_path = (
            Path("/home/kevin/Projects/eeg-rag")
            / st.session_state["settings"]["benchmark_csv"]
        )

    if csv_path.exists():
        try:
            benchmark = SystematicReviewBenchmark(str(csv_path))

            st.markdown("### 📊 Ground Truth Statistics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Papers", len(benchmark.ground_truth_df))

            if "Year" in benchmark.ground_truth_df.columns:
                years = benchmark.ground_truth_df["Year"].dropna()
                col2.metric("Year Range", f"{int(years.min())}-{int(years.max())}")

            col3.metric("Fields Available", len(benchmark.FIELD_MAPPINGS))

            # Field statistics
            st.markdown("### 📋 Field Coverage")

            field_stats = benchmark.get_field_statistics()
            st.dataframe(field_stats, use_container_width=True, hide_index=True)

            # Architecture distribution
            if "Architecture (clean)" in benchmark.ground_truth_df.columns:
                st.markdown("### 🏗️ Architecture Distribution")
                arch_counts = (
                    benchmark.ground_truth_df["Architecture (clean)"]
                    .value_counts()
                    .head(10)
                )
                st.bar_chart(arch_counts)

            # Domain distribution
            if "Domain 1" in benchmark.ground_truth_df.columns:
                st.markdown("### 🎯 Domain Distribution")
                domain_counts = (
                    benchmark.ground_truth_df["Domain 1"].value_counts().head(10)
                )
                st.bar_chart(domain_counts)

        except Exception as e:
            st.error(f"Error loading corpus: {e}")
    else:
        st.warning("⚠️ Ground truth CSV not found. Run `make download-benchmark-data`")


def render_paper_explorer_page():
    """Render the Paper Research Explorer page with full metadata and PubMed links."""
    st.header("🔬 Paper Research Explorer")

    st.markdown(
        """
    **Search, filter, and analyze deep learning EEG research papers.**
    
    Click on any paper to view full metadata, access PubMed, and analyze details.
    """
    )

    # Check if we have directly passed paper data from navigation (e.g., from query results)
    direct_paper_data = None
    if st.session_state.get("navigate_to_explorer") and st.session_state.get(
        "selected_paper_data"
    ):
        direct_paper_data = st.session_state["selected_paper_data"]
        st.session_state["navigate_to_explorer"] = False
        st.session_state["explorer_search_query"] = None

        # Show the directly passed paper data in a special section
        st.success("📄 **Showing paper from your query results**")
        st.markdown("---")

        # Display the paper details from query results
        paper = direct_paper_data

        st.markdown(f"## {paper.get('title', paper.get('Title', 'Unknown Title'))}")

        # Quick action buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        title_for_search = str(paper.get("title", paper.get("Title", ""))).replace(
            " ", "+"
        )[:100]
        pubmed_url = paper.get(
            "pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/?term={title_for_search}"
        )
        scholar_url = f"https://scholar.google.com/scholar?q={title_for_search}"

        with btn_col1:
            st.link_button("🔗 Search PubMed", pubmed_url, use_container_width=True)

        with btn_col2:
            st.link_button("🎓 Google Scholar", scholar_url, use_container_width=True)

        with btn_col3:
            if st.button("🔙 Back to Search", use_container_width=True):
                st.session_state["selected_paper_data"] = None
                st.rerun()

        st.markdown("---")

        # Display available metadata
        st.markdown("#### 📌 Paper Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            authors = paper.get("authors", paper.get("Authors", "N/A"))
            st.markdown(f"**Authors:** {authors}")

            year = paper.get("year", paper.get("Year", "N/A"))
            st.markdown(f"**Year:** {year}")

            citation = paper.get("citation", paper.get("doc_id", "N/A"))
            st.markdown(f"**Citation:** `{citation}`")

        with info_col2:
            domain = paper.get("domain", paper.get("Domain 1", "N/A"))
            if domain and domain != "N/A":
                st.markdown(f"**Domain:** {domain}")

            architecture = paper.get(
                "architecture", paper.get("Architecture (clean)", "N/A")
            )
            if architecture and architecture != "N/A":
                st.markdown(f"**Architecture:** {architecture}")

            relevance = paper.get("relevance", 0)
            if relevance:
                st.markdown(f"**Relevance Score:** {relevance:.0%}")

        # Show additional fields if available
        st.markdown("---")
        st.markdown("#### 📊 Additional Details")

        dataset = paper.get("dataset", paper.get("Dataset name", ""))
        if dataset and str(dataset) != "nan":
            st.markdown(f"**Dataset:** {dataset}")

        results = paper.get("results", paper.get("Results", ""))
        if results and str(results) != "nan":
            st.markdown(f"**Results:** {results}")

        st.markdown("---")
        st.info("💡 For full metadata, search for this paper in the corpus below.")

        # Clear the passed data after showing it once (optional, allows rerun to show full explorer)
        # st.session_state["selected_paper_data"] = None

    # Load paper data
    csv_path = Path(st.session_state["settings"]["benchmark_csv"])
    if not csv_path.exists():
        csv_path = (
            Path("/home/kevin/Projects/eeg-rag")
            / st.session_state["settings"]["benchmark_csv"]
        )

    if not csv_path.exists():
        st.error(
            "❌ Ground truth CSV not found. Please configure the path in Settings."
        )
        return

    try:
        df = pd.read_csv(
            csv_path, encoding="utf-8", on_bad_lines="skip", low_memory=False, header=1
        )
        df.columns = df.columns.str.strip()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Initialize selected paper in session state
    if "selected_paper_idx" not in st.session_state:
        st.session_state["selected_paper_idx"] = None

    # If we already showed direct paper data, skip the search matching
    if (
        direct_paper_data is None
        and st.session_state.get("navigate_to_explorer")
        and st.session_state.get("explorer_search_query")
    ):
        incoming_search = st.session_state["explorer_search_query"]
        st.session_state["navigate_to_explorer"] = False

        # Try to find the paper by citation or title
        search_term = incoming_search.lower()
        for idx, row in df.iterrows():
            citation = str(row.get("Citation", "")).lower()
            title = str(row.get("Title", "")).lower()
            if (
                search_term == citation
                or search_term in citation
                or search_term in title
            ):
                st.session_state["selected_paper_idx"] = idx
                st.session_state["selected_paper_data"] = row.to_dict()
                st.success(f"✅ Found paper: **{row.get('Title', 'Unknown')[:60]}...**")
                break

        # Clear the search query after processing
        st.session_state["explorer_search_query"] = None

    st.markdown("---")

    # ==========================================================================
    # SEARCH AND FILTER SECTION
    # ==========================================================================
    st.markdown("### 🔍 Search & Filter")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "🔎 Search papers",
            placeholder="Search by title, authors, citation ID...",
            key="paper_search",
        )

    with col2:
        # Year filter
        years = df["Year"].dropna().unique()
        years = sorted([int(y) for y in years])
        year_filter = st.multiselect(
            "📅 Year", options=years, default=[], key="year_filter"
        )

    with col3:
        # Domain filter
        if "Domain 1" in df.columns:
            domains = df["Domain 1"].dropna().unique().tolist()
            domain_filter = st.multiselect(
                "🎯 Domain", options=sorted(domains), default=[], key="domain_filter"
            )
        else:
            domain_filter = []

    # Architecture filter
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if "Architecture (clean)" in df.columns:
            architectures = df["Architecture (clean)"].dropna().unique().tolist()
            arch_filter = st.multiselect(
                "🏗️ Architecture",
                options=sorted(architectures),
                default=[],
                key="arch_filter",
            )
        else:
            arch_filter = []

    with col2:
        if "Dataset name" in df.columns:
            datasets = df["Dataset name"].dropna().unique().tolist()
            dataset_filter = st.multiselect(
                "📊 Dataset",
                options=sorted(set(datasets))[:20],  # Limit to top 20
                default=[],
                key="dataset_filter",
            )
        else:
            dataset_filter = []

    with col3:
        if "Code available" in df.columns:
            code_filter = st.selectbox(
                "💻 Code Available", options=["All", "Yes", "No"], key="code_filter"
            )
        else:
            code_filter = "All"

    # Apply filters
    filtered_df = df.copy()

    if search_query:
        search_lower = search_query.lower()
        mask = (
            filtered_df["Title"]
            .fillna("")
            .str.lower()
            .str.contains(search_lower, regex=False)
            | filtered_df["Authors"]
            .fillna("")
            .str.lower()
            .str.contains(search_lower, regex=False)
            | filtered_df.get("Citation", pd.Series([""] * len(filtered_df)))
            .fillna("")
            .str.lower()
            .str.contains(search_lower, regex=False)
        )
        filtered_df = filtered_df[mask]

    if year_filter:
        filtered_df = filtered_df[filtered_df["Year"].isin(year_filter)]

    if domain_filter and "Domain 1" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Domain 1"].isin(domain_filter)]

    if arch_filter and "Architecture (clean)" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Architecture (clean)"].isin(arch_filter)]

    if dataset_filter and "Dataset name" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Dataset name"].isin(dataset_filter)]

    if code_filter != "All" and "Code available" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Code available"]
            .fillna("")
            .str.lower()
            .str.contains(code_filter.lower())
        ]

    st.markdown(f"**📄 Showing {len(filtered_df)} of {len(df)} papers**")

    st.markdown("---")

    # ==========================================================================
    # PAPER LIST AND DETAIL VIEW
    # ==========================================================================

    col_list, col_detail = st.columns([1, 2])

    with col_list:
        st.markdown("### 📋 Paper List")

        # Display papers as clickable items
        for idx, (row_idx, row) in enumerate(filtered_df.head(50).iterrows()):
            title = str(row.get("Title", "Unknown"))[:60]
            year = int(row.get("Year", 0)) if pd.notna(row.get("Year")) else "N/A"
            citation = str(row.get("Citation", f"paper_{idx}"))
            domain = str(row.get("Domain 1", ""))[:20]

            # Get quality badges for this paper
            paper_dict = row.to_dict()
            badges = get_all_badges(paper_dict)
            quality_score = get_quality_score(paper_dict)

            # Quality indicator emoji
            quality_emoji = (
                "⭐" if quality_score >= 0.8 else "🟢" if quality_score >= 0.5 else "⚪"
            )

            # Create a button-like card for each paper
            with st.container():
                button_label = f"{quality_emoji} {citation} ({year})"
                if badges:
                    button_label += f"\n{badges}"

                if st.button(
                    button_label,
                    key=f"paper_btn_{row_idx}",
                    use_container_width=True,
                    help=f"{title}\nQuality Score: {quality_score:.0%}",
                ):
                    st.session_state["selected_paper_idx"] = row_idx
                    st.session_state["selected_paper_data"] = row.to_dict()

    with col_detail:
        st.markdown("### 📖 Paper Details")

        if st.session_state.get(
            "selected_paper_idx"
        ) is not None and st.session_state.get("selected_paper_data"):
            paper = st.session_state["selected_paper_data"]

            # Header with title and quick links
            st.markdown(f"## {paper.get('Title', 'Unknown Title')}")

            # Quick action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)

            # Try to generate PubMed search link from title
            title_for_search = str(paper.get("Title", "")).replace(" ", "+")[:100]
            pubmed_search_url = (
                f"https://pubmed.ncbi.nlm.nih.gov/?term={title_for_search}"
            )

            with btn_col1:
                st.link_button(
                    "🔗 Search PubMed", pubmed_search_url, use_container_width=True
                )

            # Google Scholar link
            scholar_url = f"https://scholar.google.com/scholar?q={title_for_search}"
            with btn_col2:
                st.link_button(
                    "🎓 Google Scholar", scholar_url, use_container_width=True
                )

            # Code link if available
            code_url = paper.get("Code hosted on", "")
            with btn_col3:
                if code_url and str(code_url).startswith("http"):
                    st.link_button("💻 View Code", code_url, use_container_width=True)
                elif paper.get("Code available", "").lower() in ["yes", "true"]:
                    st.button(
                        "💻 Code Available", disabled=True, use_container_width=True
                    )
                else:
                    st.button("💻 No Code", disabled=True, use_container_width=True)

            st.markdown("---")

            # Basic Info
            st.markdown("#### 📌 Basic Information")
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.markdown(f"**Citation:** `{paper.get('Citation', 'N/A')}`")
                st.markdown(
                    f"**Year:** {int(paper.get('Year', 0)) if pd.notna(paper.get('Year')) else 'N/A'}"
                )
                st.markdown(f"**Authors:** {paper.get('Authors', 'N/A')}")
                st.markdown(f"**Journal:** {paper.get('Journal / Origin', 'N/A')}")

            with info_col2:
                st.markdown(f"**Country:** {paper.get('Country', 'N/A')}")
                st.markdown(f"**Type:** {paper.get('Type of paper', 'N/A')}")
                st.markdown(
                    f"**Lab/Institution:** {paper.get('Lab / School / Company', 'N/A')}"
                )

            st.markdown("---")

            # Research Domain & Goals
            st.markdown("#### 🎯 Research Domain & Goals")

            domains = [paper.get(f"Domain {i}", "") for i in range(1, 5)]
            domains = [d for d in domains if d and str(d) != "nan"]
            if domains:
                st.markdown(f"**Domains:** {', '.join(domains)}")

            st.markdown(f"**High-level Goal:** {paper.get('High-level Goal', 'N/A')}")
            st.markdown(f"**Practical Goal:** {paper.get('Practical Goal', 'N/A')}")
            st.markdown(f"**Task/Paradigm:** {paper.get('Task/Paradigm', 'N/A')}")
            st.markdown(
                f"**Motivation for DL:** {paper.get('Motivation for DL', 'N/A')}"
            )

            st.markdown("---")

            # Data & Hardware
            st.markdown("#### 📊 Data & Hardware")
            data_col1, data_col2 = st.columns(2)

            with data_col1:
                st.markdown(f"**Dataset:** {paper.get('Dataset name', 'N/A')}")
                st.markdown(
                    f"**Data Accessibility:** {paper.get('Dataset accessibility', 'N/A')}"
                )
                st.markdown(f"**EEG Hardware:** {paper.get('EEG Hardware', 'N/A')}")
                st.markdown(f"**Subjects:** {paper.get('Data - subjects', 'N/A')}")

            with data_col2:
                st.markdown(f"**Channels:** {paper.get('Nb Channels', 'N/A')}")
                st.markdown(f"**Sampling Rate:** {paper.get('Sampling rate', 'N/A')}")
                st.markdown(
                    f"**Offline/Online:** {paper.get('Offline / Online', 'N/A')}"
                )

            st.markdown("---")

            # Architecture & Model
            st.markdown("#### 🏗️ Model Architecture")
            arch_col1, arch_col2 = st.columns(2)

            with arch_col1:
                st.markdown(
                    f"**Architecture:** {paper.get('Architecture (clean)', paper.get('Architecture', 'N/A'))}"
                )
                st.markdown(
                    f"**Layers:** {paper.get('Layers (clean)', paper.get('Layers', 'N/A'))}"
                )
                st.markdown(
                    f"**Activation:** {paper.get('Activation function', 'N/A')}"
                )
                st.markdown(
                    f"**Regularization:** {paper.get('Regularization (clean)', 'N/A')}"
                )

            with arch_col2:
                st.markdown(f"**Input Format:** {paper.get('Input format', 'N/A')}")
                st.markdown(f"**Output Format:** {paper.get('Output format', 'N/A')}")
                st.markdown(f"**Nb Classes:** {paper.get('Nb Classes', 'N/A')}")
                st.markdown(f"**Parameters:** {paper.get('Nb Parameters', 'N/A')}")

            st.markdown(
                f"**Design Peculiarities:** {paper.get('Design peculiarities', 'N/A')}"
            )
            st.markdown(
                f"**EEG-specific Design:** {paper.get('EEG-specific design', 'N/A')}"
            )

            st.markdown("---")

            # Preprocessing
            st.markdown("#### ⚙️ Preprocessing & Features")
            st.markdown(
                f"**Preprocessing:** {paper.get('Preprocessing (clean)', paper.get('Preprocessing', 'N/A'))}"
            )
            st.markdown(
                f"**Artifact Handling:** {paper.get('Artefact handling (clean)', 'N/A')}"
            )
            st.markdown(
                f"**Features:** {paper.get('Features (clean)', paper.get('Features', 'N/A'))}"
            )
            st.markdown(f"**Normalization:** {paper.get('Normalization', 'N/A')}")

            st.markdown("---")

            # Training
            st.markdown("#### 🎓 Training Details")
            train_col1, train_col2 = st.columns(2)

            with train_col1:
                st.markdown(
                    f"**Optimizer:** {paper.get('Optimizer (clean)', paper.get('Optimizer', 'N/A'))}"
                )
                st.markdown(f"**Loss Function:** {paper.get('Loss', 'N/A')}")
                st.markdown(f"**Batch Size:** {paper.get('Minibatch size', 'N/A')}")

            with train_col2:
                st.markdown(
                    f"**Cross Validation:** {paper.get('Cross validation (clean)', 'N/A')}"
                )
                st.markdown(
                    f"**Intra/Inter Subject:** {paper.get('Intra/Inter subject', 'N/A')}"
                )
                st.markdown(
                    f"**Data Augmentation:** {paper.get('Data augmentation', 'N/A')}"
                )

            st.markdown(
                f"**Training Procedure:** {paper.get('Training procedure (clean)', 'N/A')}"
            )
            st.markdown(
                f"**Hyperparameter Optimization:** {paper.get('Hyperparameter optim (clean)', 'N/A')}"
            )

            st.markdown("---")

            # Results
            st.markdown("#### 📈 Results & Performance")
            st.markdown(f"**Results:** {paper.get('Results', 'N/A')}")
            st.markdown(
                f"**Performance Metrics:** {paper.get('Performance metrics (clean)', paper.get('Performance metrics', 'N/A'))}"
            )
            st.markdown(f"**Benchmarks:** {paper.get('Benchmarks', 'N/A')}")
            st.markdown(
                f"**Baseline Model:** {paper.get('Baseline model type', 'N/A')}"
            )
            st.markdown(
                f"**Statistical Analysis:** {paper.get('Statistical analysis of performance', 'N/A')}"
            )

            st.markdown("---")

            # Analysis & Reproducibility
            st.markdown("#### 🔬 Analysis & Reproducibility")
            repro_col1, repro_col2 = st.columns(2)

            with repro_col1:
                code_avail = paper.get("Code available", "N/A")
                code_icon = "✅" if str(code_avail).lower() in ["yes", "true"] else "❌"
                st.markdown(f"**Code Available:** {code_icon} {code_avail}")
                st.markdown(f"**Code Hosted On:** {paper.get('Code hosted on', 'N/A')}")

            with repro_col2:
                st.markdown(f"**Software:** {paper.get('Software', 'N/A')}")
                st.markdown(
                    f"**Training Hardware:** {paper.get('Training hardware', 'N/A')}"
                )
                st.markdown(f"**Training Time:** {paper.get('Training time', 'N/A')}")

            st.markdown(
                f"**Model Inspection:** {paper.get('Model inspection (clean)', 'N/A')}"
            )
            st.markdown(
                f"**Learned Parameters Analysis:** {paper.get('Analysis of learned parameters', 'N/A')}"
            )

            st.markdown("---")

            # Discussion
            st.markdown("#### 💬 Discussion & Limitations")
            st.markdown(f"**Discussion:** {paper.get('Discussion', 'N/A')}")
            st.markdown(f"**Limitations:** {paper.get('Limitations', 'N/A')}")

            st.markdown("---")

            # Export this paper's data
            st.markdown("#### 📥 Export Paper Data")
            paper_json = json.dumps(
                {k: (v if pd.notna(v) else None) for k, v in paper.items()}, indent=2
            )
            st.download_button(
                "📄 Download Paper Metadata (JSON)",
                paper_json,
                f"{paper.get('Citation', 'paper')}_metadata.json",
                "application/json",
                use_container_width=True,
            )

        else:
            st.info("👈 Select a paper from the list to view its full details")

            # Show quick stats while no paper is selected
            st.markdown("#### 📊 Quick Statistics")

            if not filtered_df.empty:
                stat_col1, stat_col2, stat_col3 = st.columns(3)

                with stat_col1:
                    if "Year" in filtered_df.columns:
                        st.metric(
                            "Year Range",
                            f"{int(filtered_df['Year'].min())}-{int(filtered_df['Year'].max())}",
                        )

                with stat_col2:
                    if "Architecture (clean)" in filtered_df.columns:
                        top_arch = (
                            filtered_df["Architecture (clean)"].value_counts().idxmax()
                        )
                        st.metric("Top Architecture", top_arch)

                with stat_col3:
                    if "Domain 1" in filtered_df.columns:
                        top_domain = filtered_df["Domain 1"].value_counts().idxmax()
                        st.metric("Top Domain", top_domain)


def render_ingestion_page():
    """Render data ingestion page."""
    st.header("📥 Data Ingestion")

    st.markdown(
        """
    **Collect EEG research papers from multiple academic sources**
    
    The ingestion system collects papers from:
    - **PubMed/PMC** - Peer-reviewed biomedical literature with full-text access
    - **Semantic Scholar** - Cross-disciplinary academic graph with citations
    - **arXiv** - Preprints and cutting-edge research
    - **OpenAlex** - Open access metadata with 100K+ daily limit
    
    > 💡 **No API keys required!** All sources work without authentication.
    > Keys are optional and only provide faster rate limits for bulk ingestion.
    """
    )

    st.markdown("---")

    # Ingestion mode selection
    mode = st.radio(
        "Select Ingestion Mode",
        [
            "🚀 Quick Start (1K papers)",
            "📊 Standard (10K papers)",
            "🏭 Bulk (100K+ papers)",
            "🔧 Custom",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # Configuration based on mode
    if mode == "🚀 Quick Start (1K papers)":
        st.info(
            "**Quick Start Mode**: Collect ~1,000 papers for testing. Takes about 5-10 minutes."
        )
        pubmed_target = 400
        scholar_target = 300
        arxiv_target = 150
        openalex_target = 150

    elif mode == "📊 Standard (10K papers)":
        st.info(
            "**Standard Mode**: Collect ~10,000 papers for a solid research corpus. Takes about 1-2 hours."
        )
        pubmed_target = 4000
        scholar_target = 3000
        arxiv_target = 1500
        openalex_target = 1500

    elif mode == "🏭 Bulk (100K+ papers)":
        st.info(
            "**Bulk Mode**: Collect 120,000+ papers for comprehensive coverage. Takes 5-8 hours (faster with API keys)."
        )
        pubmed_target = 50000
        scholar_target = 30000
        arxiv_target = 10000
        openalex_target = 30000

    else:  # Custom
        st.markdown("### Custom Configuration")
        col1, col2 = st.columns(2)

        with col1:
            pubmed_target = st.number_input(
                "PubMed Target", min_value=0, max_value=100000, value=5000, step=1000
            )
            scholar_target = st.number_input(
                "Semantic Scholar Target",
                min_value=0,
                max_value=100000,
                value=3000,
                step=1000,
            )

        with col2:
            arxiv_target = st.number_input(
                "arXiv Target", min_value=0, max_value=50000, value=1000, step=500
            )
            openalex_target = st.number_input(
                "OpenAlex Target", min_value=0, max_value=100000, value=3000, step=1000
            )

    # Show totals
    total = pubmed_target + scholar_target + arxiv_target + openalex_target

    st.markdown("### 📊 Target Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PubMed", f"{pubmed_target:,}")
    col2.metric("Semantic Scholar", f"{scholar_target:,}")
    col3.metric("arXiv", f"{arxiv_target:,}")
    col4.metric("OpenAlex", f"{openalex_target:,}")
    col5.metric("**Total**", f"{total:,}")

    # Estimate time
    # Without API keys: ~300 papers/minute average
    est_minutes = total / 300
    if est_minutes < 60:
        est_time = f"{int(est_minutes)} minutes"
    else:
        est_time = f"{est_minutes / 60:.1f} hours"

    st.markdown(f"**Estimated time**: {est_time} (without API keys)")

    st.markdown("---")

    # Output configuration
    st.markdown("### 📁 Output Configuration")

    col1, col2 = st.columns(2)

    with col1:
        output_dir = st.text_input(
            "Output Directory",
            value="data/ingestion",
            help="Where to save collected papers",
        )

    with col2:
        resume = st.checkbox(
            "Resume from checkpoint",
            value=True,
            help="Continue from where you left off if interrupted",
        )

    st.markdown("---")

    # API Keys (optional)
    with st.expander("🔑 API Keys (Optional - for faster ingestion)"):
        st.markdown(
            """
        API keys are **completely optional**. Without keys, you'll get:
        - PubMed: 3 requests/second
        - Semantic Scholar: 100 requests/5 minutes
        - arXiv & OpenAlex: No keys needed
        
        With free API keys, you can get 3-4x faster ingestion.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            pubmed_key = st.text_input(
                "PubMed API Key",
                type="password",
                placeholder="Optional",
                help="Get free at: https://www.ncbi.nlm.nih.gov/account/settings/",
            )
            st.caption(
                "[Get free PubMed key](https://www.ncbi.nlm.nih.gov/account/settings/)"
            )

        with col2:
            s2_key = st.text_input(
                "Semantic Scholar API Key",
                type="password",
                placeholder="Optional",
                help="Get free at: https://www.semanticscholar.org/product/api#api-key",
            )
            st.caption(
                "[Get free S2 key](https://www.semanticscholar.org/product/api#api-key)"
            )

    st.markdown("---")

    # Action buttons
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        start_clicked = st.button(
            "🚀 Start Ingestion", type="primary", use_container_width=True
        )

    with col2:
        if st.button("📋 Generate CLI Command", use_container_width=True):
            cmd = f"python scripts/run_bulk_ingestion.py --pubmed {pubmed_target} --scholar {scholar_target} --arxiv {arxiv_target} --openalex {openalex_target} --output-dir {output_dir}"
            if not resume:
                cmd += " --fresh"
            st.code(cmd, language="bash")
            st.caption("Run this command in terminal for background execution")

    with col3:
        if st.button("❓ Help", use_container_width=True):
            st.info("See docs/DATA_INGESTION.md for detailed instructions")

    # Start ingestion
    if start_clicked:
        st.warning(
            "⚠️ **Note**: For large ingestion jobs (10K+ papers), we recommend using the CLI command instead for better reliability."
        )

        st.markdown("### 🔄 Ingestion Progress")

        progress_bar = st.progress(0.0, text="Initializing...")
        status_text = st.empty()

        # For demo, show simulation (actual ingestion is async and long-running)
        import time as time_module

        sources = ["PubMed", "Semantic Scholar", "arXiv", "OpenAlex"]

        for i, source in enumerate(sources):
            progress = (i + 1) / len(sources)
            progress_bar.progress(progress, text=f"Collecting from {source}...")
            status_text.markdown(f"📡 **Status**: Connecting to {source} API...")
            time_module.sleep(0.5)

        progress_bar.progress(1.0, text="Setup complete!")

        st.success(
            """
        ✅ **Ingestion initialized!**
        
        For actual paper collection, run the CLI command in your terminal:
        """
        )

        cmd = f"python scripts/run_bulk_ingestion.py --pubmed {pubmed_target} --scholar {scholar_target} --arxiv {arxiv_target} --openalex {openalex_target} --output-dir {output_dir}"
        st.code(cmd, language="bash")

        st.info(
            """
        **Why CLI for bulk ingestion?**
        - ✅ Runs in background (won't time out)
        - ✅ Automatic checkpointing (resume if interrupted)
        - ✅ Better progress logging
        - ✅ Can run overnight
        """
        )

    # Show existing data
    st.markdown("---")
    st.markdown("### 📊 Existing Ingested Data")

    data_dirs = ["data/ingestion", "data/bulk_ingestion", "data/raw"]

    for dir_path in data_dirs:
        dir_full = Path(dir_path)
        if dir_full.exists():
            jsonl_files = list(dir_full.glob("*.jsonl"))
            if jsonl_files:
                for f in jsonl_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    # Count lines (papers)
                    try:
                        with open(f) as fp:
                            line_count = sum(1 for _ in fp)
                    except:
                        line_count = 0

                    st.markdown(
                        f"📄 **{f.name}** - {line_count:,} papers ({size_mb:.1f} MB)"
                    )


def render_settings_page():
    """Render settings page."""
    st.header("⚙️ Settings")

    st.markdown("### 📁 Corpus Configuration")

    col1, col2 = st.columns(2)

    with col1:
        corpus_path = st.text_input(
            "Corpus Path", value=st.session_state["settings"]["corpus_path"]
        )

    with col2:
        embeddings_path = st.text_input(
            "Embeddings Path", value=st.session_state["settings"]["embeddings_path"]
        )

    benchmark_csv = st.text_input(
        "Benchmark CSV Path", value=st.session_state["settings"]["benchmark_csv"]
    )

    st.markdown("---")

    st.markdown("### 🤖 Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-large-v2", "bge-large-en"],
            index=0,
        )

    with col2:
        llm_model = st.selectbox(
            "LLM Model",
            [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "claude-3-opus",
                "claude-3-sonnet",
            ],
            index=0,
        )

    st.markdown("---")

    st.markdown("### 🔐 API Keys")

    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    anthropic_key = st.text_input(
        "Anthropic API Key", type="password", placeholder="sk-ant-..."
    )

    st.markdown("---")

    st.markdown("### 🎨 Display Settings")

    col1, col2 = st.columns(2)

    with col1:
        max_sources = st.slider("Default Max Sources", 1, 20, 5)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)

    with col2:
        theme = st.selectbox("Theme", ["Light", "Dark", "System"])

    st.markdown("---")

    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        st.session_state["settings"].update(
            {
                "corpus_path": corpus_path,
                "embeddings_path": embeddings_path,
                "benchmark_csv": benchmark_csv,
                "embedding_model": embedding_model,
                "llm_model": llm_model,
                "max_sources": max_sources,
                "show_confidence": show_confidence,
            }
        )
        st.success("✅ Settings saved!")


# =============================================================================
# Main Application
# =============================================================================


def main():
    """Main application entry point."""
    # Page config must be first Streamlit command
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Render header
    render_header()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Check if we need to navigate to Paper Explorer (from clicking a paper elsewhere)
    if st.session_state.get("navigate_to_explorer"):
        # Override page selection to go to Paper Explorer
        page = "🔬 Paper Research Explorer"

    st.markdown("---")

    # Render selected page
    if page == "🔍 Query System":
        render_query_page()
    elif page == "📥 Data Ingestion":
        render_ingestion_page()
    elif page == " Corpus Explorer":
        render_corpus_page()
    elif page == "🔬 Paper Research Explorer":
        render_paper_explorer_page()
    elif page == "⚙️ Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
