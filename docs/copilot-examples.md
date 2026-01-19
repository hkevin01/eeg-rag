# GitHub Copilot Usage Examples for EEG-RAG

This document provides real-world examples of effective Copilot prompts and the resulting code for EEG-RAG development.

## Example 1: Creating a New EEG Data Validator

### Prompt Used:
```
Create an EEG data validator class that:
- Validates 10-20 electrode system names
- Checks frequency band ranges (Delta: 0.5-4Hz, Theta: 4-8Hz, Alpha: 8-13Hz, Beta: 13-30Hz, Gamma: 30-100Hz)
- Validates ERP component latencies (P300: 250-500ms, N400: 300-500ms)
- Includes async validation methods
- Has comprehensive error handling
- Returns structured validation results
- Includes full type hints and Google-style docstrings
- Follows EEG-RAG production standards
```

### Generated Code:
```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class FrequencyBand(Enum):
    """Standard EEG frequency bands."""
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"

@dataclass
class ValidationResult:
    """Result of EEG data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validated_data: Optional[Dict] = None

class EEGDataValidator:
    """Validates EEG data according to standard conventions."""
    
    # 10-20 electrode system standard positions
    VALID_10_20_ELECTRODES = {
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
        "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
        "P3", "P4", "Pz", "O1", "O2", "A1", "A2"
    }
    
    # Frequency band ranges in Hz
    FREQUENCY_BANDS = {
        FrequencyBand.DELTA: (0.5, 4.0),
        FrequencyBand.THETA: (4.0, 8.0),
        FrequencyBand.ALPHA: (8.0, 13.0),
        FrequencyBand.BETA: (13.0, 30.0),
        FrequencyBand.GAMMA: (30.0, 100.0),
    }
    
    # ERP component typical latency ranges in ms
    ERP_LATENCIES = {
        "P300": (250, 500),
        "N400": (300, 500),
        "P600": (500, 700),
        "MMN": (150, 250),
        "N170": (130, 200),
    }
    
    def __init__(self, strict_mode: bool = False):
        """Initialize the EEG data validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors.
        """
        self.strict_mode = strict_mode
    
    async def validate_electrode_names(
        self, 
        electrodes: List[str]
    ) -> ValidationResult:
        """Validate electrode names against 10-20 system.
        
        Args:
            electrodes: List of electrode names to validate.
            
        Returns:
            ValidationResult with electrode validation status.
            
        Raises:
            ValueError: If electrodes list is empty.
        """
        if not electrodes:
            raise ValueError("Electrode list cannot be empty")
        
        errors = []
        warnings = []
        validated_electrodes = []
        
        for electrode in electrodes:
            electrode_clean = electrode.strip()
            
            if electrode_clean in self.VALID_10_20_ELECTRODES:
                validated_electrodes.append(electrode_clean)
            elif electrode_clean.lower() in {e.lower() for e in self.VALID_10_20_ELECTRODES}:
                warnings.append(f"Electrode '{electrode}' has incorrect case")
                # Find correct case
                correct_case = next(e for e in self.VALID_10_20_ELECTRODES 
                                  if e.lower() == electrode_clean.lower())
                validated_electrodes.append(correct_case)
            else:
                errors.append(f"Invalid electrode name: '{electrode}'")
        
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_data={"electrodes": validated_electrodes}
        )
```

## Example 2: Implementing Citation Extraction

### Prompt Used:
```
Implement a PMID extractor for EEG-RAG that:
- Extracts PMIDs from research text using regex pattern PMID[:\s]*(\d{7,8})
- Validates PMID format (7-8 digits)
- Handles multiple citation formats
- Includes async PubMed validation
- Returns structured citation data
- Has comprehensive error handling and logging
- Uses type hints and follows production standards
```

### Generated Code:
```python
import re
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import logging

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """Represents a research citation with PMID."""
    pmid: str
    context: str
    position: int
    is_validated: bool = False
    pubmed_data: Optional[Dict] = None

class PMIDExtractor:
    """Extracts and validates PMIDs from research text."""
    
    # Regex pattern for PMID extraction
    PMID_PATTERN = re.compile(r'PMID[:\s]*(\d{7,8})', re.IGNORECASE)
    PMID_FORMAT_PATTERN = re.compile(r'^\d{7,8}$')
    
    def __init__(self, validate_online: bool = True):
        """Initialize PMID extractor.
        
        Args:
            validate_online: Whether to validate PMIDs against PubMed.
        """
        self.validate_online = validate_online
        self._validated_pmids: Set[str] = set()
    
    def extract_pmids(self, text: str) -> List[Citation]:
        """Extract all PMIDs from text.
        
        Args:
            text: Text to search for PMIDs.
            
        Returns:
            List of Citation objects found in text.
            
        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        citations = []
        
        for match in self.PMID_PATTERN.finditer(text):
            pmid = match.group(1)
            
            # Validate PMID format
            if self._is_valid_pmid_format(pmid):
                # Get context around the PMID
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                citation = Citation(
                    pmid=pmid,
                    context=context,
                    position=match.start()
                )
                citations.append(citation)
            else:
                logger.warning(f"Invalid PMID format: {pmid}")
        
        # Remove duplicates while preserving order
        seen_pmids = set()
        unique_citations = []
        
        for citation in citations:
            if citation.pmid not in seen_pmids:
                seen_pmids.add(citation.pmid)
                unique_citations.append(citation)
        
        return unique_citations
```

## Example 3: Creating an Async Query Router

### Prompt Used:
```
Create an intelligent query router for EEG-RAG that:
- Routes queries to appropriate agents based on content analysis
- Recognizes EEG terminology and clinical contexts
- Uses async patterns for performance
- Includes confidence scoring for routing decisions
- Has comprehensive logging and error handling
- Supports multiple routing strategies
- Returns structured routing results
- Follows EEG-RAG production standards
```

### Generated Code:
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle."""
    DEFINITIONAL = "definitional"
    METHODOLOGICAL = "methodological"
    COMPARATIVE = "comparative"
    STATISTICAL = "statistical"
    CLINICAL = "clinical"
    RESEARCH = "research"

class AgentType(Enum):
    """Available agent types for query routing."""
    LOCAL_DATA = "local_data"
    WEB_SEARCH = "web_search"
    GRAPH_QUERY = "graph_query"
    HYBRID = "hybrid"

@dataclass
class RoutingResult:
    """Result of query routing analysis."""
    recommended_agent: AgentType
    query_type: QueryType
    confidence: float
    reasoning: str
    routing_time_ms: float
    metadata: Dict[str, Any]

class QueryAnalyzer(ABC):
    """Abstract base class for query analysis."""
    
    @abstractmethod
    async def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query and return analysis results."""
        pass

class EEGDomainAnalyzer(QueryAnalyzer):
    """Analyzes queries for EEG domain content."""
    
    # EEG-specific keywords and their weights
    EEG_KEYWORDS = {
        # Frequency bands
        "delta": 0.9, "theta": 0.9, "alpha": 0.9, "beta": 0.9, "gamma": 0.9,
        # Electrodes
        "fp1": 0.8, "fp2": 0.8, "c3": 0.8, "c4": 0.8, "pz": 0.8,
        # ERP components
        "p300": 0.95, "n400": 0.95, "mmn": 0.9, "p600": 0.9,
        # Clinical terms
        "epilepsy": 0.85, "seizure": 0.85, "eeg": 1.0,
        "electroencephalogram": 1.0, "brainwaves": 0.8
    }
    
    async def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query for EEG domain relevance.
        
        Args:
            query: Query text to analyze.
            
        Returns:
            Dictionary with EEG domain analysis results.
        """
        query_lower = query.lower()
        
        eeg_score = 0.0
        found_terms = []
        
        for term, weight in self.EEG_KEYWORDS.items():
            if term in query_lower:
                eeg_score += weight
                found_terms.append(term)
        
        # Normalize score
        max_possible_score = len(self.EEG_KEYWORDS)
        eeg_relevance = min(1.0, eeg_score / max_possible_score)
        
        return {
            "eeg_relevance": eeg_relevance,
            "found_terms": found_terms,
            "is_eeg_query": eeg_relevance > 0.3
        }

class IntelligentQueryRouter:
    """Routes queries to appropriate agents using intelligent analysis."""
    
    def __init__(self, enable_metrics: bool = True):
        """Initialize the query router.
        
        Args:
            enable_metrics: Whether to collect routing metrics.
        """
        self.enable_metrics = enable_metrics
        self.routing_stats: Dict[str, int] = {}
        self.analyzers = {
            "eeg_domain": EEGDomainAnalyzer()
        }
    
    async def route_query(self, query: str) -> RoutingResult:
        """Route a query to the most appropriate agent.
        
        Args:
            query: User query to route.
            
        Returns:
            RoutingResult with routing decision and analysis.
            
        Raises:
            ValueError: If query is empty or invalid.
        """
        start_time = time.time()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Run all analyzers in parallel
            analysis_tasks = [
                analyzer.analyze(query) 
                for analyzer in self.analyzers.values()
            ]
            analysis_results = await asyncio.gather(*analysis_tasks)
            
            # Combine analysis results
            combined_analysis = {}
            for result in analysis_results:
                combined_analysis.update(result)
            
            # Determine query type and routing
            query_type = self._determine_query_type(query, combined_analysis)
            agent_type, confidence, reasoning = self._select_agent(
                query, query_type, combined_analysis
            )
            
            routing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            if self.enable_metrics:
                self._update_stats(agent_type, query_type)
            
            return RoutingResult(
                recommended_agent=agent_type,
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                routing_time_ms=routing_time,
                metadata=combined_analysis
            )
            
        except Exception as e:
            logger.error(f"Error routing query '{query}': {str(e)}")
            # Return safe default routing
            return RoutingResult(
                recommended_agent=AgentType.LOCAL_DATA,
                query_type=QueryType.DEFINITIONAL,
                confidence=0.1,
                reasoning=f"Fallback routing due to error: {str(e)}",
                routing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )
```

## Example 4: Performance Optimization

### Prompt Used:
```
Optimize this EEG document retrieval function for:
- Processing 10K+ documents efficiently
- Sub-100ms latency for retrieval
- Concurrent request handling
- Memory-efficient embedding caching
- Batch processing capabilities
- Comprehensive error handling
- Performance monitoring and logging
```

### Generated Code:
```python
import asyncio
import time
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    """Performance metrics for document retrieval."""
    query_time_ms: float
    num_documents: int
    cache_hits: int
    embedding_time_ms: float
    search_time_ms: float

class OptimizedEEGRetriever:
    """High-performance EEG document retriever."""
    
    def __init__(
        self,
        max_workers: int = 4,
        cache_size: int = 10000,
        batch_size: int = 32
    ):
        """Initialize optimized retriever.
        
        Args:
            max_workers: Maximum thread pool workers.
            cache_size: Maximum number of cached embeddings.
            batch_size: Batch size for processing.
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance monitoring
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_queries = 0
        
        # Initialize embedding cache
        self._init_cache(cache_size)
    
    def _init_cache(self, cache_size: int) -> None:
        """Initialize LRU cache for embeddings."""
        @lru_cache(maxsize=cache_size)
        def _cached_embed(text: str) -> np.ndarray:
            """Cached embedding function."""
            return self._compute_embedding(text)
        
        self._get_cached_embedding = _cached_embed
    
    async def retrieve_documents(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 10
    ) -> Tuple[List[Dict], RetrievalMetrics]:
        """Retrieve most relevant documents efficiently.
        
        Args:
            query: Search query.
            documents: List of documents to search.
            top_k: Number of top results to return.
            
        Returns:
            Tuple of (top documents, performance metrics).
            
        Raises:
            ValueError: If query is empty or documents list is invalid.
        """
        start_time = time.time()
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        try:
            # Phase 1: Get query embedding
            embedding_start = time.time()
            query_embedding = await self._get_embedding_async(query)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Phase 2: Batch process document similarities
            search_start = time.time()
            similarities = await self._batch_compute_similarities(
                query_embedding, documents
            )
            search_time = (time.time() - search_start) * 1000
            
            # Phase 3: Get top-k results
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
            
            top_documents = [
                {**documents[i], 'similarity_score': float(similarities[i])}
                for i in top_indices
            ]
            
            # Update metrics
            total_time = (time.time() - start_time) * 1000
            self._total_queries += 1
            
            metrics = RetrievalMetrics(
                query_time_ms=total_time,
                num_documents=len(documents),
                cache_hits=self._cache_hits,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time
            )
            
            logger.info(
                f"Retrieved {len(top_documents)} documents in {total_time:.2f}ms"
            )
            
            return top_documents, metrics
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            raise
```

## Best Practices from Examples

1. **Always Include Domain Context**: Every prompt references EEG-specific requirements
2. **Specify Quality Standards**: All examples request production-grade code
3. **Request Complete Solutions**: Prompts ask for error handling, logging, and documentation
4. **Include Performance Requirements**: Specific latency and throughput targets are mentioned
5. **Ask for Type Safety**: Type hints and validation are always requested
6. **Specify Testing Needs**: Examples often mention test coverage requirements
7. **Request Monitoring**: Logging and metrics collection are standard requirements

## Common Prompt Patterns

- **"Create a [component] that:"** - Use for new implementations
- **"Optimize this [code] for:"** - Use for performance improvements  
- **"Add [feature] to:"** - Use for extending existing code
- **"Implement [pattern] that:"** - Use for architectural components
- **"Generate tests for [component] that:"** - Use for test creation
- **"Refactor [code] to:"** - Use for code improvements

These examples demonstrate how to get high-quality, production-ready code from GitHub Copilot by being specific about requirements, domain context, and quality standards.