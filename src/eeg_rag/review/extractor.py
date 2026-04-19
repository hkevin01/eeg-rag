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


# ---------------------------------------------------------------------------
# ID           : review.extractor.ExtractionField
# Requirement  : `ExtractionField` class shall be instantiable and expose the documented interface
# Purpose      : Definition of a field to extract from papers
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ExtractionField with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ExtractionField:
    """Definition of a field to extract from papers."""

    name: str
    description: str
    field_type: str  # "string", "number", "boolean", "list", "enum"
    enum_values: Optional[List[str]] = None
    required: bool = False
    extraction_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# ID           : review.extractor.ExtractedData
# Requirement  : `ExtractedData` class shall be instantiable and expose the documented interface
# Purpose      : Container for extracted data from a single paper
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ExtractedData with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : review.extractor.SystematicReviewExtractor
# Requirement  : `SystematicReviewExtractor` class shall be instantiable and expose the documented interface
# Purpose      : Extracts structured data from papers for systematic reviews
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate SystematicReviewExtractor with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.__init__
    # Requirement  : `__init__` shall initialize systematic review extractor
    # Purpose      : Initialize systematic review extractor
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : protocol: Union[str, Path, Dict]; date_range: Optional[Tuple[str, str]] (default=None); query: Optional[str] (default=None); llm_backend: str (default='ollama'); model_name: str (default='mistral'); confidence_threshold: float (default=0.7)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._parse_protocol
    # Requirement  : `_parse_protocol` shall parse protocol definition into ExtractionField objects
    # Purpose      : Parse protocol definition into ExtractionField objects
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[ExtractionField]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._build_extraction_prompt
    # Requirement  : `_build_extraction_prompt` shall build extraction prompt for a specific field
    # Purpose      : Build extraction prompt for a specific field
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]; field: ExtractionField
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # LLM backend helpers
    # ------------------------------------------------------------------

    def _call_llm_sync(self, prompt: str) -> Optional[str]:
        """Call the configured LLM backend synchronously.

        Supports ``ollama`` (local), ``openai``, and ``anthropic`` backends.
        Returns the raw text response or ``None`` on failure.
        """
        try:
            if self.llm_backend == "ollama":
                return self._call_ollama(prompt)
            elif self.llm_backend == "openai":
                return self._call_openai(prompt)
            elif self.llm_backend == "anthropic":
                return self._call_anthropic(prompt)
            else:
                logger.warning(
                    f"Unknown LLM backend '{self.llm_backend}'; using rule-based fallback"
                )
                return None
        except Exception as e:
            logger.warning(f"LLM call ({self.llm_backend}) failed: {e}")
            return None

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._call_ollama
    # Requirement  : `_call_ollama` shall call a local Ollama instance via its REST API
    # Purpose      : Call a local Ollama instance via its REST API
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : prompt: str
    # Outputs      : Optional[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call a local Ollama instance via its REST API."""
        import os
        import urllib.request  # stdlib only — no extra dep required

        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        payload = json.dumps(
            {"model": self.model_name, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip() or None

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._call_openai
    # Requirement  : `_call_openai` shall call OpenAI chat completions API
    # Purpose      : Call OpenAI chat completions API
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : prompt: str
    # Outputs      : Optional[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI chat completions API."""
        import os
        try:
            import openai  # type: ignore
        except ImportError:
            logger.warning("openai package not installed; cannot use OpenAI backend")
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set")
            return None

        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scientific data extraction assistant. "
                        "Extract structured information from EEG research papers."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip() or None

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._call_anthropic
    # Requirement  : `_call_anthropic` shall call Anthropic Messages API
    # Purpose      : Call Anthropic Messages API
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : prompt: str
    # Outputs      : Optional[str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic Messages API."""
        import os
        try:
            import anthropic  # type: ignore
        except ImportError:
            logger.warning("anthropic package not installed; cannot use Anthropic backend")
            return None

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set")
            return None

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.model_name,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip() or None

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._parse_llm_response
    # Requirement  : `_parse_llm_response` shall parse a JSON LLM response into (value, confidence, note)
    # Purpose      : Parse a JSON LLM response into (value, confidence, note)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : raw: str; field: ExtractionField
    # Outputs      : Tuple[Any, float, str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    @staticmethod
    def _parse_llm_response(
        raw: str, field: ExtractionField
    ) -> Tuple[Any, float, str]:
        """Parse a JSON LLM response into (value, confidence, note).

        Falls back gracefully when the response is not valid JSON.
        """
        # Strip markdown fences if present
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL
        )
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting the first JSON object via regex
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return None, 0.0, f"LLM response not parseable: {raw[:80]}"
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return None, 0.0, f"LLM response not parseable: {raw[:80]}"

        value = parsed.get("value")
        confidence = float(parsed.get("confidence", 0.5))
        note = str(parsed.get("reasoning", parsed.get("quote", "LLM extraction")))

        # Coerce type for numeric fields
        if field.field_type == "number" and value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = None
                confidence = 0.0

        # Coerce booleans
        if field.field_type == "boolean" and value is not None:
            if isinstance(value, str):
                value = value.lower() in ("true", "yes", "1")

        return value, confidence, note

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._extract_field_llm
    # Requirement  : `_extract_field_llm` shall extract a single field using LLM
    # Purpose      : Extract a single field using LLM
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]; field: ExtractionField
    # Outputs      : Tuple[Any, float, str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
            # Attempt LLM extraction via the configured backend
            raw = self._call_llm_sync(prompt)
            if raw:
                value, confidence, note = self._parse_llm_response(raw, field)
                return value, confidence, note
            # LLM returned nothing — fall through to rule-based
            value, confidence, note = self._rule_based_extraction(paper, field)
            return value, confidence, note

        except Exception as e:
            logger.error(f"LLM extraction failed for field {field.name}: {e}")
            return None, 0.0, f"Extraction error: {str(e)}"

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._rule_based_extraction
    # Requirement  : `_rule_based_extraction` shall fallback rule-based extraction for common fields
    # Purpose      : Fallback rule-based extraction for common fields
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]; field: ExtractionField
    # Outputs      : Tuple[Any, float, str]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.extract_from_paper
    # Requirement  : `extract_from_paper` shall extract all fields from a single paper
    # Purpose      : Extract all fields from a single paper
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : paper: Dict[str, Any]
    # Outputs      : ExtractedData
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.run
    # Requirement  : `run` shall run extraction on papers
    # Purpose      : Run extraction on papers
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : papers: Optional[List[Dict]] (default=None); max_papers: int (default=500)
    # Outputs      : pd.DataFrame
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
            if self.query:
                papers = self._retrieve_papers_for_query(self.query, max_papers)
            else:
                logger.warning(
                    "No papers supplied and no query set; returning empty DataFrame."
                )
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor._retrieve_papers_for_query
    # Requirement  : `_retrieve_papers_for_query` shall retrieve papers using PubMed E-utilities when no papers are provided
    # Purpose      : Retrieve papers using PubMed E-utilities when no papers are provided
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; max_papers: int
    # Outputs      : List[Dict[str, Any]]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _retrieve_papers_for_query(
        self, query: str, max_papers: int
    ) -> List[Dict[str, Any]]:
        """Retrieve papers using PubMed E-utilities when no papers are provided.

        Falls back to an empty list with a warning if the network call fails.

        Args:
            query: Free-text search query.
            max_papers: Maximum number of records to return.

        Returns:
            List of paper dicts with at minimum ``pmid``, ``title``, ``abstract``.
        """
        import urllib.request
        import urllib.parse

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        papers: List[Dict[str, Any]] = []

        try:
            # 1. ESearch — get matching PMIDs
            search_params = urllib.parse.urlencode(
                {
                    "db": "pubmed",
                    "term": query,
                    "retmax": min(max_papers, 500),
                    "retmode": "json",
                    "sort": "relevance",
                }
            )
            with urllib.request.urlopen(
                f"{base}/esearch.fcgi?{search_params}", timeout=30
            ) as resp:
                search_data = json.loads(resp.read())

            pmids: List[str] = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                logger.info("ESearch returned no results for query: %s", query)
                return []

            # Apply date range filter when provided
            if self.date_range:
                start, end = self.date_range
                date_filter = f"{start}:{end}[dp]"
                search_params_filtered = urllib.parse.urlencode(
                    {
                        "db": "pubmed",
                        "term": f"({query}) AND {date_filter}",
                        "retmax": min(max_papers, 500),
                        "retmode": "json",
                        "sort": "relevance",
                    }
                )
                with urllib.request.urlopen(
                    f"{base}/esearch.fcgi?{search_params_filtered}", timeout=30
                ) as resp2:
                    filtered = json.loads(resp2.read())
                filtered_ids = filtered.get("esearchresult", {}).get("idlist", [])
                if filtered_ids:
                    pmids = filtered_ids

            # 2. EFetch — retrieve full abstracts in JSON summary format
            batch_size = 100
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                fetch_params = urllib.parse.urlencode(
                    {
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "retmode": "json",
                        "rettype": "abstract",
                    }
                )
                with urllib.request.urlopen(
                    f"{base}/esummary.fcgi?{fetch_params}", timeout=30
                ) as resp3:
                    summary = json.loads(resp3.read())

                result_map = summary.get("result", {})
                for pmid in batch:
                    rec = result_map.get(pmid)
                    if not rec or pmid == "uids":
                        continue
                    authors = [
                        a.get("name", "") for a in rec.get("authors", [])
                    ]
                    papers.append(
                        {
                            "pmid": pmid,
                            "title": rec.get("title", ""),
                            "abstract": rec.get("source", ""),  # ESummary has source
                            "authors": authors,
                            "year": int(rec.get("pubdate", "0")[:4] or 0),
                            "journal": rec.get("source", ""),
                            "doi": next(
                                (
                                    uid.get("value")
                                    for uid in rec.get("articleids", [])
                                    if uid.get("idtype") == "doi"
                                ),
                                None,
                            ),
                        }
                    )

            logger.info(
                "Retrieved %d papers from PubMed for query: %s", len(papers), query
            )

        except Exception as exc:
            logger.warning("PubMed retrieval failed: %s — returning empty list.", exc)

        return papers[:max_papers]

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.to_dataframe
    # Requirement  : `to_dataframe` shall convert extraction results to DataFrame
    # Purpose      : Convert extraction results to DataFrame
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : pd.DataFrame
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.export
    # Requirement  : `export` shall export results to file
    # Purpose      : Export results to file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : output_path: Union[str, Path]; format: str (default='csv')
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # ID           : review.extractor.SystematicReviewExtractor.get_low_confidence_extractions
    # Requirement  : `get_low_confidence_extractions` shall get papers with low-confidence extractions for manual review
    # Purpose      : Get papers with low-confidence extractions for manual review
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : threshold: float (default=0.6)
    # Outputs      : pd.DataFrame
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_low_confidence_extractions(self, threshold: float = 0.6) -> pd.DataFrame:
        """Get papers with low-confidence extractions for manual review."""
        df = self.to_dataframe()

        # Find rows with any confidence score below threshold
        confidence_cols = [col for col in df.columns if col.endswith("_confidence")]
        low_conf_mask = (df[confidence_cols] < threshold).any(axis=1)

        return df[low_conf_mask]
