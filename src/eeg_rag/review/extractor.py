"""
Systematic Review Extractor for EEG-RAG.

Automates structured data extraction from papers for systematic reviews,
replicating methodologies like Roy et al. 2019.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExtractionField:
    """Definition of a field to extract from papers."""
    
    name: str
    description: str
    field_type: str  # "string", "number", "boolean", "list", "enum"
    enum_values: Optional[List[str]] = None
    required: bool = False
    extraction_prompt: Optional[str] = None


@dataclass
class ExtractedData:
    """Container for extracted data from a single paper."""
    
    paper_id: str
    title: str
    authors: List[str]
    year: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    
    # Extracted fields
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    extraction_notes: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_success: bool = True
    extraction_errors: List[str] = field(default_factory=list)


class SystematicReviewExtractor:
    """
    Extracts structured data from papers for systematic reviews.
    
    Example:
        extractor = SystematicReviewExtractor(
            protocol="dl_eeg_review_schema.yaml",
            date_range=("2018-01-01", "2026-01-01"),
            query="deep learning EEG classification"
        )
        results = extractor.run(max_papers=500)
        results.to_csv("dl_eeg_papers.csv")
    """
    
    def __init__(
        self,
        protocol: Union[str, Path, Dict],
        date_range: Optional[Tuple[str, str]] = None,
        query: Optional[str] = None,
        llm_backend: str = "ollama",
        model_name: str = "mistral",
        confidence_threshold: float = 0.7
    ):
        """
        Initialize systematic review extractor.
        
        Args:
            protocol: Path to YAML schema file or dict defining extraction fields
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            query: Search query to retrieve papers
            llm_backend: LLM backend to use ("ollama", "openai", "anthropic")
            model_name: Model name for extraction
            confidence_threshold: Minimum confidence for extraction
        """
        self.date_range = date_range
        self.query = query
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Load protocol
        if isinstance(protocol, (str, Path)):
            with open(protocol) as f:
                self.protocol = yaml.safe_load(f)
        else:
            self.protocol = protocol
        
        self.fields = self._parse_protocol()
        self.results: List[ExtractedData] = []
        
        logger.info(f"Initialized SystematicReviewExtractor with {len(self.fields)} fields")
    
    def _parse_protocol(self) -> List[ExtractionField]:
        """Parse protocol definition into ExtractionField objects."""
        fields = []
        for field_def in self.protocol.get("fields", []):
            fields.append(ExtractionField(
                name=field_def["name"],
                description=field_def["description"],
                field_type=field_def["type"],
                enum_values=field_def.get("enum_values"),
                required=field_def.get("required", False),
                extraction_prompt=field_def.get("extraction_prompt")
            ))
        return fields
    
    def _build_extraction_prompt(
        self,
        paper: Dict[str, Any],
        field: ExtractionField
    ) -> str:
        """Build extraction prompt for a specific field."""
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        
        base_prompt = f"""Extract the following information from this research paper:

Title: {title}

Abstract: {abstract}

Field to extract: {field.name}
Description: {field.description}
Type: {field.field_type}
"""
        
        if field.enum_values:
            base_prompt += f"\nAllowed values: {', '.join(field.enum_values)}"
        
        if field.extraction_prompt:
            base_prompt += f"\n\n{field.extraction_prompt}"
        
        base_prompt += """

Provide your answer in JSON format with:
{
  "value": <extracted value or null if not found>,
  "confidence": <0.0-1.0 confidence score>,
  "reasoning": <brief explanation>,
  "quote": <exact quote from text if applicable>
}

JSON:"""
        
        return base_prompt
    
    def _extract_field_llm(
        self,
        paper: Dict[str, Any],
        field: ExtractionField
    ) -> Tuple[Any, float, str]:
        """
        Extract a single field using LLM.
        
        Returns:
            (value, confidence, note)
        """
        prompt = self._build_extraction_prompt(paper, field)
        
        try:
            # TODO: Integrate with actual LLM backend
            # For now, return rule-based extraction
            value, confidence, note = self._rule_based_extraction(paper, field)
            return value, confidence, note
            
        except Exception as e:
            logger.error(f"LLM extraction failed for field {field.name}: {e}")
            return None, 0.0, f"Extraction error: {str(e)}"
    
    def _rule_based_extraction(
        self,
        paper: Dict[str, Any],
        field: ExtractionField
    ) -> Tuple[Any, float, str]:
        """Fallback rule-based extraction for common fields."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Architecture type
        if field.name == "architecture_type":
            patterns = {
                "CNN": r'\b(cnn|convolutional neural network|convnet)\b',
                "RNN": r'\b(rnn|recurrent neural network|lstm|gru)\b',
                "Transformer": r'\b(transformer|attention|bert|gpt)\b',
                "Hybrid": r'\b(hybrid|combined|ensemble)\b',
                "Other": r'\b(neural network|deep learning)\b'
            }
            for arch, pattern in patterns.items():
                if re.search(pattern, text):
                    return arch, 0.8, f"Found pattern: {pattern}"
            return None, 0.0, "No architecture mentioned"
        
        # Dataset name
        elif field.name == "dataset_name":
            datasets = [
                "TUSZ", "CHB-MIT", "Bonn", "Temple", "DEAP", "SEED",
                "BCI Competition", "PhysioNet", "MNIST", "ImageNet"
            ]
            for dataset in datasets:
                if dataset.lower() in text:
                    return dataset, 0.9, f"Found dataset name: {dataset}"
            
            # Generic extraction
            dataset_match = re.search(r'(?:dataset|database)[\s:]+([A-Z][A-Za-z0-9-]+)', text)
            if dataset_match:
                return dataset_match.group(1), 0.6, "Extracted from pattern"
            
            return None, 0.0, "No dataset identified"
        
        # Reported accuracy
        elif field.name == "reported_accuracy":
            # Look for accuracy/F1/AUC values
            acc_patterns = [
                r'accuracy[:\s]+([0-9]{2,3}(?:\.[0-9]{1,2})?)\s*%',
                r'acc[:\s=]+([0-9]{2,3}(?:\.[0-9]{1,2})?)',
                r'f1[:\s=]+([0-9]?\.[0-9]{2,4})',
                r'auc[:\s=]+([0-9]?\.[0-9]{2,4})'
            ]
            for pattern in acc_patterns:
                match = re.search(pattern, text)
                if match:
                    value = float(match.group(1))
                    # Normalize to 0-1 if > 1
                    if value > 1:
                        value = value / 100
                    return value, 0.8, f"Extracted from pattern: {pattern}"
            
            return None, 0.0, "No performance metric found"
        
        # Code availability
        elif field.name == "code_available":
            if re.search(r'github\.com|gitlab\.com|bitbucket\.org', text):
                return "GitHub link found", 0.95, "Repository URL present"
            elif re.search(r'code (is )?available|publicly available|open.?source', text):
                return "Code available upon request", 0.7, "Availability statement found"
            elif re.search(r'code will be (made )?available|upon acceptance', text):
                return "Code available upon publication", 0.6, "Future availability mentioned"
            else:
                return "Not available", 0.5, "No availability statement"
        
        # Sample size
        elif field.name == "sample_size":
            size_match = re.search(r'(\d{1,5})\s*(subjects?|patients?|participants?|recordings?)', text)
            if size_match:
                return int(size_match.group(1)), 0.8, "Extracted from pattern"
            return None, 0.0, "No sample size found"
        
        # Task type
        elif field.name == "task_type":
            tasks = {
                "Seizure Detection": r'\b(seizure detection|epilepsy|ictal)\b',
                "Sleep Staging": r'\b(sleep stag|sleep classification|polysomnography)\b',
                "BCI": r'\b(brain.?computer interface|bci|motor imagery)\b',
                "ERP Analysis": r'\b(event.?related potential|erp|p300|n400)\b',
                "Cognitive State": r'\b(cognitive|workload|attention|drowsiness)\b'
            }
            for task, pattern in tasks.items():
                if re.search(pattern, text):
                    return task, 0.85, f"Matched pattern: {pattern}"
            return "Other", 0.3, "No specific task identified"
        
        # Default
        return None, 0.0, "Field not supported by rule-based extraction"
    
    def extract_from_paper(self, paper: Dict[str, Any]) -> ExtractedData:
        """Extract all fields from a single paper."""
        result = ExtractedData(
            paper_id=paper.get("id", paper.get("pmid", "unknown")),
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            year=paper.get("year", 0),
            doi=paper.get("doi"),
            pmid=paper.get("pmid")
        )
        
        for field in self.fields:
            try:
                value, confidence, note = self._extract_field_llm(paper, field)
                
                result.extracted_fields[field.name] = value
                result.confidence_scores[field.name] = confidence
                result.extraction_notes[field.name] = note
                
                # Check if required field is missing
                if field.required and (value is None or confidence < self.confidence_threshold):
                    result.extraction_errors.append(
                        f"Required field '{field.name}' extraction failed (confidence: {confidence})"
                    )
                    result.extraction_success = False
                
            except Exception as e:
                logger.error(f"Error extracting field {field.name}: {e}")
                result.extraction_errors.append(f"{field.name}: {str(e)}")
                result.extraction_success = False
        
        return result
    
    def run(self, papers: Optional[List[Dict]] = None, max_papers: int = 500) -> pd.DataFrame:
        """
        Run extraction on papers.
        
        Args:
            papers: List of papers to process. If None, will retrieve using query.
            max_papers: Maximum number of papers to process
        
        Returns:
            DataFrame with extracted data
        """
        if papers is None:
            # TODO: Integrate with retrieval system
            logger.warning("Paper retrieval not yet implemented. Use papers parameter.")
            papers = []
        
        logger.info(f"Starting extraction on {len(papers)} papers...")
        
        self.results = []
        for i, paper in enumerate(papers[:max_papers]):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{min(len(papers), max_papers)} papers")
            
            result = self.extract_from_paper(paper)
            self.results.append(result)
        
        logger.info(f"Extraction complete. {len(self.results)} papers processed")
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert extraction results to DataFrame."""
        rows = []
        for result in self.results:
            row = {
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": ", ".join(result.authors) if result.authors else "",
                "year": result.year,
                "doi": result.doi,
                "pmid": result.pmid,
                "extraction_success": result.extraction_success,
                "extraction_timestamp": result.extraction_timestamp
            }
            
            # Add extracted fields
            for field_name, value in result.extracted_fields.items():
                row[field_name] = value
                row[f"{field_name}_confidence"] = result.confidence_scores.get(field_name, 0.0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export(self, output_path: Union[str, Path], format: str = "csv"):
        """Export results to file."""
        df = self.to_dataframe()
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported results to {output_path}")
    
    def get_low_confidence_extractions(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get papers with low-confidence extractions for manual review."""
        df = self.to_dataframe()
        
        # Find rows with any confidence score below threshold
        confidence_cols = [col for col in df.columns if col.endswith("_confidence")]
        low_conf_mask = (df[confidence_cols] < threshold).any(axis=1)
        
        return df[low_conf_mask]
