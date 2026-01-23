"""
Comprehensive agent registry with detailed metadata, capabilities, and configuration.

This module provides complete documentation for each agent in the EEG-RAG system,
including:
- Detailed descriptions and long-form documentation
- Capability listings with enable/disable flags
- Input/output schemas for validation
- Configuration options
- Usage examples
- Troubleshooting guides
- Runtime metrics tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class AgentStatus(Enum):
    """Runtime status of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    DISABLED = "disabled"


class AgentCategory(Enum):
    """Functional category of an agent."""
    ORCHESTRATION = "orchestration"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


@dataclass
class AgentCapability:
    """A specific capability of an agent."""
    name: str
    description: str
    enabled: bool = True


@dataclass
class AgentMetrics:
    """Runtime metrics for an agent."""
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    last_invocation: Optional[str] = None
    last_error: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_invocations == 0:
            return 100.0
        return (self.successful_invocations / self.total_invocations) * 100


@dataclass
class AgentConfig:
    """Configuration options for an agent."""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    cache_ttl_seconds: int = 3600
    batch_size: int = 10
    custom_params: dict = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Complete information about an agent."""
    id: str
    name: str
    description: str
    long_description: str
    category: AgentCategory
    icon: str
    color: str
    
    # Technical details
    capabilities: list
    dependencies: list
    input_schema: dict
    output_schema: dict
    
    # Configuration
    config: AgentConfig
    
    # Runtime state
    status: AgentStatus = AgentStatus.IDLE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    
    # Documentation
    usage_examples: list = field(default_factory=list)
    troubleshooting: list = field(default_factory=list)


# Complete Agent Registry
AGENT_REGISTRY: dict[str, AgentInfo] = {
    
    "orchestrator": AgentInfo(
        id="orchestrator",
        name="Orchestrator",
        description="Coordinates all agents and manages query execution pipeline",
        long_description="""
The Orchestrator is the central coordinator of the EEG-RAG system. It receives incoming 
queries, determines the optimal execution strategy, dispatches tasks to specialized agents, 
monitors their progress, handles failures with retry logic, and assembles the final response.

**Key Responsibilities:**
- Query intake and validation
- Execution plan creation based on Query Planner output
- Parallel agent dispatch for independent tasks
- Sequential coordination for dependent tasks
- Timeout and error handling
- Result aggregation and response assembly
- Logging and observability

**Execution Modes:**
1. **Simple Mode**: Direct retrieval for straightforward factual queries
2. **Standard Mode**: Plan → Retrieve → Validate → Synthesize
3. **Deep Mode**: Multi-hop reasoning with iterative refinement

The Orchestrator uses an event-driven architecture with async message passing, 
allowing it to efficiently manage multiple concurrent agent operations while 
maintaining strict ordering guarantees where required.
        """,
        category=AgentCategory.ORCHESTRATION,
        icon="🎯",
        color="#6366F1",  # Indigo
        
        capabilities=[
            AgentCapability(
                name="Parallel Dispatch",
                description="Execute multiple independent agent tasks concurrently"
            ),
            AgentCapability(
                name="Dependency Resolution",
                description="Automatically order tasks based on data dependencies"
            ),
            AgentCapability(
                name="Failure Recovery",
                description="Retry failed tasks with exponential backoff"
            ),
            AgentCapability(
                name="Timeout Management",
                description="Enforce per-agent and global query timeouts"
            ),
            AgentCapability(
                name="Result Caching",
                description="Cache intermediate results to avoid redundant work"
            ),
            AgentCapability(
                name="Observability",
                description="Emit structured logs and metrics for all operations"
            ),
        ],
        
        dependencies=[],  # No dependencies - top-level coordinator
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User's natural language query"},
                "session_id": {"type": "string", "description": "Session identifier for context"},
                "options": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["simple", "standard", "deep"]},
                        "max_sources": {"type": "integer", "minimum": 1, "maximum": 50},
                        "include_trials": {"type": "boolean"},
                        "validate_citations": {"type": "boolean"},
                    }
                }
            },
            "required": ["query"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"$ref": "#/definitions/Citation"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "execution_trace": {"type": "array", "items": {"$ref": "#/definitions/TraceEvent"}},
                "latency_ms": {"type": "number"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=120.0,
            max_retries=2,
            custom_params={
                "max_parallel_agents": 5,
                "enable_caching": True,
                "trace_level": "detailed",
            }
        ),
        
        usage_examples=[
            {
                "title": "Simple Query",
                "query": "What is the normal alpha frequency range?",
                "mode": "simple",
                "description": "Direct retrieval without complex planning"
            },
            {
                "title": "Complex Multi-hop Query",
                "query": "How do sleep spindle abnormalities in MCI patients compare to those seen in early Alzheimer's, and what are the implications for BCI-based interventions?",
                "mode": "deep",
                "description": "Requires multiple retrieval steps and synthesis"
            },
        ],
        
        troubleshooting=[
            {
                "issue": "Query timeout",
                "cause": "Complex query requiring many agent calls",
                "solution": "Increase timeout or simplify query"
            },
            {
                "issue": "Partial results",
                "cause": "Some agents failed but others succeeded",
                "solution": "Check individual agent logs for specific failures"
            },
        ]
    ),
    
    "query_planner": AgentInfo(
        id="query_planner",
        name="Query Planner",
        description="Decomposes complex queries into executable sub-tasks",
        long_description="""
The Query Planner analyzes incoming queries to understand their complexity, identify 
required information types, and decompose them into a structured execution plan. It uses 
a combination of rule-based heuristics and LLM-powered analysis to create optimal plans.

**Planning Process:**
1. **Query Classification**: Determine query type (factual, comparative, procedural, etc.)
2. **Entity Extraction**: Identify key concepts, conditions, and constraints
3. **Decomposition**: Break complex queries into atomic sub-questions
4. **Dependency Analysis**: Determine which sub-questions depend on others
5. **Source Selection**: Identify which agents are needed for each sub-question
6. **Plan Optimization**: Maximize parallelism while respecting dependencies

**Query Types Supported:**
- **Factual**: "What is the sensitivity of EEG for detecting seizures?"
- **Comparative**: "How does focal vs. generalized epilepsy present on EEG?"
- **Temporal**: "How has EEG-based seizure detection evolved since 2010?"
- **Procedural**: "What is the protocol for performing a sleep-deprived EEG?"
- **Diagnostic**: "What EEG findings suggest autoimmune encephalitis?"

The planner outputs a directed acyclic graph (DAG) of tasks that the Orchestrator executes.
        """,
        category=AgentCategory.PLANNING,
        icon="🧩",
        color="#8B5CF6",  # Purple
        
        capabilities=[
            AgentCapability(
                name="Query Classification",
                description="Categorize queries by type and complexity"
            ),
            AgentCapability(
                name="Entity Extraction",
                description="Extract medical entities, conditions, and relationships"
            ),
            AgentCapability(
                name="Query Decomposition",
                description="Split complex queries into atomic sub-questions"
            ),
            AgentCapability(
                name="Dependency Graph",
                description="Build execution DAG with proper ordering"
            ),
            AgentCapability(
                name="Source Routing",
                description="Select optimal retrieval agents for each sub-task"
            ),
            AgentCapability(
                name="Complexity Estimation",
                description="Predict query difficulty and resource requirements"
            ),
        ],
        
        dependencies=["orchestrator"],
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {
                    "type": "object",
                    "properties": {
                        "previous_queries": {"type": "array", "items": {"type": "string"}},
                        "user_expertise": {"type": "string", "enum": ["novice", "intermediate", "expert"]},
                    }
                }
            },
            "required": ["query"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "query_type": {"type": "string"},
                "complexity_score": {"type": "number", "minimum": 0, "maximum": 1},
                "entities": {"type": "array", "items": {"type": "object"}},
                "sub_queries": {"type": "array", "items": {"type": "object"}},
                "execution_plan": {"type": "object"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=10.0,
            max_retries=2,
            custom_params={
                "model": "gpt-4",
                "max_sub_queries": 5,
                "enable_caching": True,
            }
        ),
        
        usage_examples=[
            {
                "title": "Multi-part Query Decomposition",
                "input": "Compare EEG biomarkers for Alzheimer's vs Lewy body dementia and their correlation with cognitive scores",
                "output": {
                    "sub_queries": [
                        "What are EEG biomarkers for Alzheimer's disease?",
                        "What are EEG biomarkers for Lewy body dementia?",
                        "How do these biomarkers correlate with cognitive assessments?",
                    ],
                    "dependencies": [[0, 1], [2]],  # First two parallel, third depends on both
                }
            }
        ],
        
        troubleshooting=[
            {
                "issue": "Over-decomposition",
                "cause": "Simple query split into too many parts",
                "solution": "Adjust complexity threshold in config"
            },
        ]
    ),
    
    "local_data_agent": AgentInfo(
        id="local_data_agent",
        name="Local Data Agent",
        description="Performs semantic search over indexed EEG literature using FAISS",
        long_description="""
The Local Data Agent provides fast, semantic search over the locally indexed corpus of 
EEG research papers, clinical guidelines, and educational materials. It uses FAISS 
(Facebook AI Similarity Search) for efficient approximate nearest neighbor search over 
dense vector embeddings.

**Index Contents:**
- 50,000+ peer-reviewed EEG research papers
- ACNS, ILAE, and AAN clinical guidelines
- EEG atlas reference materials
- Curated case studies and clinical vignettes

**Search Process:**
1. **Query Embedding**: Convert query to dense vector using medical-domain encoder
2. **ANN Search**: Find k-nearest neighbors in FAISS index
3. **Re-ranking**: Apply cross-encoder for precise relevance scoring
4. **Metadata Filtering**: Apply date, journal, and topic filters
5. **Chunk Assembly**: Reconstruct coherent passages from indexed chunks

**Embedding Model:**
Uses PubMedBERT fine-tuned on EEG literature for domain-specific semantic understanding.
Captures nuances like "spike-and-wave" vs "polyspike" that general models miss.

**Index Updates:**
The index is refreshed weekly with new publications from PubMed and monthly 
guideline updates from major neurological societies.
        """,
        category=AgentCategory.RETRIEVAL,
        icon="📚",
        color="#10B981",  # Emerald
        
        capabilities=[
            AgentCapability(
                name="Semantic Search",
                description="Dense vector similarity search with medical embeddings"
            ),
            AgentCapability(
                name="Hybrid Search",
                description="Combine semantic search with BM25 keyword matching"
            ),
            AgentCapability(
                name="Cross-encoder Reranking",
                description="Precise relevance scoring for top candidates"
            ),
            AgentCapability(
                name="Metadata Filtering",
                description="Filter by date, journal, author, MeSH terms"
            ),
            AgentCapability(
                name="Chunk Reassembly",
                description="Reconstruct full passages from indexed chunks"
            ),
            AgentCapability(
                name="Citation Extraction",
                description="Extract and format PMID/DOI citations"
            ),
        ],
        
        dependencies=["query_planner"],
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
                "filters": {
                    "type": "object",
                    "properties": {
                        "year_min": {"type": "integer"},
                        "year_max": {"type": "integer"},
                        "journals": {"type": "array", "items": {"type": "string"}},
                        "mesh_terms": {"type": "array", "items": {"type": "string"}},
                    }
                },
                "rerank": {"type": "boolean", "default": True},
            },
            "required": ["query"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "score": {"type": "number"},
                            "pmid": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array"},
                            "journal": {"type": "string"},
                            "year": {"type": "integer"},
                        }
                    }
                },
                "total_searched": {"type": "integer"},
                "search_latency_ms": {"type": "number"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=15.0,
            max_retries=2,
            custom_params={
                "embedding_model": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
                "index_type": "IVF4096,PQ64",
                "nprobe": 128,
                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "rerank_top_k": 50,
            }
        ),
        
        usage_examples=[
            {
                "title": "Filtered Search",
                "query": "EEG biomarkers for early Alzheimer's detection",
                "filters": {"year_min": 2020, "journals": ["Clinical Neurophysiology", "Neurology"]},
                "top_k": 15
            }
        ],
        
        troubleshooting=[
            {
                "issue": "Low relevance scores",
                "cause": "Query too general or out of domain",
                "solution": "Add specific EEG terminology to query"
            },
            {
                "issue": "Missing recent papers",
                "cause": "Index not yet updated",
                "solution": "Check last index refresh date; trigger manual refresh"
            },
        ]
    ),
    
    "web_search_agent": AgentInfo(
        id="web_search_agent",
        name="Web Search Agent",
        description="Queries PubMed API for real-time access to latest publications",
        long_description="""
The Web Search Agent provides real-time access to PubMed, the premier biomedical 
literature database. It complements the local index by retrieving the latest 
publications that may not yet be indexed locally, and by accessing the full 
breadth of PubMed's 35+ million citations.

**API Capabilities:**
- **E-utilities API**: Official NCBI API for programmatic access
- **Rate Limits**: 10 requests/second with API key, 3/second without
- **Full Metadata**: Abstracts, MeSH terms, author affiliations, grants

**Search Features:**
1. **Boolean Queries**: Complex AND/OR/NOT combinations
2. **Field-specific Search**: Title, abstract, author, journal, MeSH
3. **Date Filtering**: Publication date, entry date, modification date
4. **Citation Linking**: Find papers that cite or are cited by a paper

**Query Translation:**
Natural language queries are translated to optimized PubMed syntax:
- "EEG in Alzheimer's" → `(electroencephalography[MeSH] OR EEG[tiab]) AND (Alzheimer disease[MeSH])`

**Result Processing:**
- Fetches full abstracts for top results
- Extracts structured data (authors, affiliations, MeSH terms)
- Identifies related papers via PubMed's Related Articles feature
        """,
        category=AgentCategory.RETRIEVAL,
        icon="🌐",
        color="#3B82F6",  # Blue
        
        capabilities=[
            AgentCapability(
                name="PubMed Search",
                description="Query NCBI E-utilities for biomedical literature"
            ),
            AgentCapability(
                name="Query Translation",
                description="Convert natural language to PubMed search syntax"
            ),
            AgentCapability(
                name="MeSH Expansion",
                description="Automatically expand queries with MeSH term hierarchy"
            ),
            AgentCapability(
                name="Citation Network",
                description="Find citing and cited papers"
            ),
            AgentCapability(
                name="Clinical Trials",
                description="Search ClinicalTrials.gov for active studies"
            ),
            AgentCapability(
                name="Preprint Search",
                description="Query medRxiv/bioRxiv for preprints",
                enabled=True
            ),
        ],
        
        dependencies=["query_planner"],
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 20, "maximum": 200},
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"},
                    }
                },
                "include_mesh_expansion": {"type": "boolean", "default": True},
                "include_related": {"type": "boolean", "default": False},
            },
            "required": ["query"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pmid": {"type": "string"},
                            "title": {"type": "string"},
                            "abstract": {"type": "string"},
                            "authors": {"type": "array"},
                            "journal": {"type": "string"},
                            "pub_date": {"type": "string"},
                            "mesh_terms": {"type": "array"},
                            "doi": {"type": "string"},
                        }
                    }
                },
                "total_count": {"type": "integer"},
                "query_translation": {"type": "string"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=30.0,
            max_retries=3,
            retry_delay_seconds=2.0,
            custom_params={
                "api_key": "${PUBMED_API_KEY}",
                "email": "eeg-rag@example.com",
                "tool": "EEG-RAG",
                "rate_limit_per_second": 10,
            }
        ),
        
        usage_examples=[
            {
                "title": "Recent Publications Search",
                "query": "machine learning EEG seizure prediction",
                "date_range": {"start": "2023-01-01", "end": "2024-12-31"},
                "max_results": 50
            }
        ],
        
        troubleshooting=[
            {
                "issue": "API rate limit exceeded",
                "cause": "Too many concurrent requests",
                "solution": "Ensure API key is configured; implement request queuing"
            },
            {
                "issue": "No results found",
                "cause": "Query too specific or syntax error",
                "solution": "Check query translation; broaden search terms"
            },
        ]
    ),
    
    "knowledge_graph_agent": AgentInfo(
        id="knowledge_graph_agent",
        name="Knowledge Graph Agent",
        description="Executes Cypher queries against Neo4j for relationship-based retrieval",
        long_description="""
The Knowledge Graph Agent leverages a Neo4j graph database to answer queries that 
require understanding relationships between entities. Unlike vector search which 
finds similar documents, the graph enables precise traversal of connections like 
author collaborations, citation networks, concept hierarchies, and temporal patterns.

**Graph Schema:**

**Nodes:**
- `Paper`: Research publications with PMID, title, abstract, year
- `Author`: Researchers with name, affiliation, ORCID
- `Concept`: Medical concepts (disorders, findings, techniques)
- `Journal`: Publication venues with impact metrics
- `Institution`: Research organizations
- `ClinicalTrial`: Registered clinical studies

**Relationships:**
- `(:Author)-[:AUTHORED]->(:Paper)`
- `(:Paper)-[:CITES]->(:Paper)`
- `(:Paper)-[:DISCUSSES]->(:Concept)`
- `(:Concept)-[:SUBTYPE_OF]->(:Concept)`
- `(:Paper)-[:PUBLISHED_IN]->(:Journal)`

**Query Capabilities:**
1. **Path Finding**: "What connects researcher A to concept B?"
2. **Aggregation**: "Which authors have published most on topic X?"
3. **Pattern Matching**: "Find papers that cite both A and B"
4. **Temporal Analysis**: "How has research on X evolved over time?"

The agent translates natural language queries to Cypher using an LLM with 
schema-aware prompting, then executes against Neo4j and formats results.
        """,
        category=AgentCategory.RETRIEVAL,
        icon="🔗",
        color="#F59E0B",  # Amber
        
        capabilities=[
            AgentCapability(
                name="Natural Language to Cypher",
                description="Translate questions to graph queries"
            ),
            AgentCapability(
                name="Path Finding",
                description="Discover connections between entities"
            ),
            AgentCapability(
                name="Citation Network Analysis",
                description="Traverse paper citation relationships"
            ),
            AgentCapability(
                name="Author Collaboration",
                description="Find co-authorship patterns"
            ),
            AgentCapability(
                name="Concept Hierarchy",
                description="Navigate MeSH/SNOMED term hierarchies"
            ),
            AgentCapability(
                name="Temporal Patterns",
                description="Analyze research trends over time"
            ),
        ],
        
        dependencies=["query_planner"],
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "query_type": {
                    "type": "string",
                    "enum": ["path", "aggregate", "pattern", "temporal"]
                },
                "limit": {"type": "integer", "default": 25},
                "include_evidence": {"type": "boolean", "default": True},
            },
            "required": ["query"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "cypher_query": {"type": "string"},
                "execution_time_ms": {"type": "number"},
                "nodes_traversed": {"type": "integer"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=20.0,
            max_retries=2,
            custom_params={
                "neo4j_uri": "${NEO4J_URI}",
                "database": "eeg_knowledge",
                "max_path_length": 5,
                "enable_query_cache": True,
            }
        ),
        
        usage_examples=[
            {
                "title": "Citation Network Query",
                "query": "Find influential papers that bridge epilepsy and sleep research",
                "description": "Uses PageRank over citation subgraph filtered by topics"
            },
            {
                "title": "Collaboration Discovery",
                "query": "Which researchers collaborate across both EEG-BCI and epilepsy monitoring?",
                "description": "Finds authors with papers in both topic clusters"
            },
        ],
        
        troubleshooting=[
            {
                "issue": "Query timeout",
                "cause": "Unbounded path traversal",
                "solution": "Add path length limits; check for Cartesian products"
            },
            {
                "issue": "Empty results",
                "cause": "Entity not in graph or misspelled",
                "solution": "Verify entity exists; use fuzzy matching"
            },
        ]
    ),
    
    "citation_validator": AgentInfo(
        id="citation_validator",
        name="Citation Validator",
        description="Verifies PMID existence and checks for retractions",
        long_description="""
The Citation Validator ensures the accuracy and reliability of all citations in 
generated responses. It addresses a critical weakness of LLM-based systems: 
hallucinated references that look plausible but don't exist or have been retracted.

**Validation Steps:**

1. **Existence Check**: Verify PMID exists in PubMed database
2. **Metadata Verification**: Confirm title/authors match the claim
3. **Retraction Status**: Check Retraction Watch database and PubMed errata
4. **Expression of Concern**: Flag papers with editorial concerns
5. **Preprint Status**: Identify if citing preprint vs. peer-reviewed version

**Validation Levels:**
- **Quick**: Existence check only (~50ms per citation)
- **Standard**: Existence + retraction status (~200ms)
- **Thorough**: Full metadata verification + related errata (~500ms)

**Handling Invalid Citations:**
- Log warning with details
- Optionally remove from response
- Trigger regeneration with corrected context
- Flag to user with explanation

**Retraction Database:**
Maintains local cache of known retractions updated weekly from Retraction Watch 
(10,000+ retracted papers) and PubMed's retraction notices.
        """,
        category=AgentCategory.VALIDATION,
        icon="✅",
        color="#EF4444",  # Red
        
        capabilities=[
            AgentCapability(
                name="PMID Verification",
                description="Confirm citations exist in PubMed"
            ),
            AgentCapability(
                name="Retraction Detection",
                description="Check Retraction Watch and PubMed for retractions"
            ),
            AgentCapability(
                name="Metadata Validation",
                description="Verify paper details match citation claims"
            ),
            AgentCapability(
                name="Batch Validation",
                description="Efficiently validate multiple citations in parallel"
            ),
            AgentCapability(
                name="Citation Formatting",
                description="Standardize citation format (AMA, APA, Vancouver)"
            ),
            AgentCapability(
                name="DOI Resolution",
                description="Resolve and validate DOI links"
            ),
        ],
        
        dependencies=["context_aggregator"],
        
        input_schema={
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pmid": {"type": "string"},
                            "doi": {"type": "string"},
                            "claimed_title": {"type": "string"},
                        }
                    }
                },
                "validation_level": {
                    "type": "string",
                    "enum": ["quick", "standard", "thorough"],
                    "default": "standard"
                },
            },
            "required": ["citations"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "validations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pmid": {"type": "string"},
                            "is_valid": {"type": "boolean"},
                            "exists": {"type": "boolean"},
                            "is_retracted": {"type": "boolean"},
                            "verified_title": {"type": "string"},
                            "error": {"type": "string"},
                        }
                    }
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "valid": {"type": "integer"},
                        "invalid": {"type": "integer"},
                        "retracted": {"type": "integer"},
                    }
                }
            }
        },
        
        config=AgentConfig(
            timeout_seconds=30.0,
            max_retries=3,
            custom_params={
                "pubmed_api_key": "${PUBMED_API_KEY}",
                "retraction_db_path": "./data/retractions.db",
                "cache_valid_for_days": 7,
                "parallel_validations": 5,
            }
        ),
        
        usage_examples=[
            {
                "title": "Batch Validation",
                "citations": [
                    {"pmid": "32470456"},
                    {"pmid": "99999999"},  # Invalid
                    {"pmid": "12345678", "claimed_title": "Some Title"},
                ],
                "validation_level": "standard"
            }
        ],
        
        troubleshooting=[
            {
                "issue": "Validation timeout",
                "cause": "PubMed API slow or rate limited",
                "solution": "Use API key; implement request queuing"
            },
            {
                "issue": "False retraction flag",
                "cause": "Retraction database out of date",
                "solution": "Trigger retraction database refresh"
            },
        ]
    ),
    
    "context_aggregator": AgentInfo(
        id="context_aggregator",
        name="Context Aggregator",
        description="Merges and deduplicates results from multiple retrieval agents",
        long_description="""
The Context Aggregator receives results from all retrieval agents (Local Data, Web Search, 
Knowledge Graph) and produces a unified, deduplicated, and ranked context for the 
Response Generator. This is critical for avoiding redundancy and ensuring the most 
relevant information is prioritized within the LLM's context window.

**Aggregation Pipeline:**

1. **Collection**: Gather results from all retrieval agents
2. **Normalization**: Standardize formats and extract common fields
3. **Deduplication**: 
   - Exact match by PMID/DOI
   - Fuzzy match by title similarity (>0.9 Jaccard)
   - Semantic similarity for passages (>0.95 cosine)
4. **Relevance Fusion**: Combine scores from different sources
   - Reciprocal Rank Fusion (RRF) for rank-based combination
   - Weighted average for score-based combination
5. **Diversity Injection**: Ensure result variety (topics, years, sources)
6. **Context Window Optimization**: Select passages that fit token budget

**Fusion Strategies:**
- **RRF (Reciprocal Rank Fusion)**: `1 / (k + rank)` for each source, sum across sources
- **CombMNZ**: Multiply normalized scores × number of sources retrieving document
- **Learned Fusion**: Neural model trained on relevance judgments

**Token Budget Management:**
Given an LLM context limit (e.g., 128k tokens), the aggregator:
- Prioritizes highest-scored passages
- Truncates or summarizes lower-ranked content
- Preserves citation metadata even when truncating
        """,
        category=AgentCategory.SYNTHESIS,
        icon="🔄",
        color="#06B6D4",  # Cyan
        
        capabilities=[
            AgentCapability(
                name="Result Deduplication",
                description="Remove duplicate documents across sources"
            ),
            AgentCapability(
                name="Score Fusion",
                description="Combine relevance scores using RRF or learned fusion"
            ),
            AgentCapability(
                name="Diversity Ranking",
                description="Ensure topical and temporal diversity"
            ),
            AgentCapability(
                name="Token Budgeting",
                description="Optimize context for LLM token limits"
            ),
            AgentCapability(
                name="Source Attribution",
                description="Track which agent retrieved each result"
            ),
            AgentCapability(
                name="Passage Chunking",
                description="Split long documents into coherent passages"
            ),
        ],
        
        dependencies=["local_data_agent", "web_search_agent", "knowledge_graph_agent"],
        
        input_schema={
            "type": "object",
            "properties": {
                "retrieval_results": {
                    "type": "object",
                    "properties": {
                        "local": {"type": "array"},
                        "web": {"type": "array"},
                        "graph": {"type": "array"},
                    }
                },
                "max_documents": {"type": "integer", "default": 20},
                "max_tokens": {"type": "integer", "default": 16000},
                "fusion_method": {
                    "type": "string",
                    "enum": ["rrf", "combmnz", "weighted"],
                    "default": "rrf"
                },
            },
            "required": ["retrieval_results"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "context": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "pmid": {"type": "string"},
                            "source": {"type": "string"},
                            "fused_score": {"type": "number"},
                        }
                    }
                },
                "total_tokens": {"type": "integer"},
                "documents_merged": {"type": "integer"},
                "duplicates_removed": {"type": "integer"},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=10.0,
            max_retries=1,
            custom_params={
                "rrf_k": 60,
                "dedup_similarity_threshold": 0.92,
                "diversity_lambda": 0.3,
                "min_passage_length": 100,
            }
        ),
        
        usage_examples=[
            {
                "title": "Multi-source Aggregation",
                "description": "Combine 15 local results, 10 PubMed results, and 5 graph results into top 20"
            }
        ],
        
        troubleshooting=[
            {
                "issue": "Context too long",
                "cause": "Token budget exceeded",
                "solution": "Reduce max_documents or enable passage truncation"
            },
            {
                "issue": "Redundant passages",
                "cause": "Dedup threshold too low",
                "solution": "Increase similarity threshold to 0.95"
            },
        ]
    ),
    
    "response_generator": AgentInfo(
        id="response_generator",
        name="Response Generator",
        description="Synthesizes final answer using LLM with retrieved context",
        long_description="""
The Response Generator is the final stage of the pipeline, using a large language model 
to synthesize a coherent, accurate, and well-cited response from the aggregated context. 
It is specifically prompted for medical/scientific accuracy and citation discipline.

**Generation Pipeline:**

1. **Context Formatting**: Structure retrieved passages for LLM consumption
2. **Prompt Construction**: Build task-specific prompt with:
   - System instructions for medical accuracy
   - Citation requirements and formatting rules
   - User's original query
   - Retrieved context with source metadata
3. **Response Generation**: Call LLM with structured prompt
4. **Post-processing**:
   - Extract and format citations
   - Verify all claims are grounded in context
   - Add confidence indicators
   - Format for readability

**Prompting Strategy:**
- Zero-shot CoT (Chain of Thought) for complex queries
- Few-shot examples for specific formats (differential diagnosis, etc.)
- Structured output with JSON mode for programmatic processing

**Citation Enforcement:**
The prompt explicitly requires:
- Every factual claim must cite a source
- Citations must use [PMID: XXXXXXXX] format
- No claims beyond what the context supports
- Explicit uncertainty when evidence is limited

**Quality Controls:**
- Groundedness check: Verify claims trace to context
- Hallucination detection: Flag unsupported statements
- Confidence scoring: Estimate answer reliability
        """,
        category=AgentCategory.SYNTHESIS,
        icon="📝",
        color="#EC4899",  # Pink
        
        capabilities=[
            AgentCapability(
                name="Context-grounded Generation",
                description="Synthesize answers strictly from provided context"
            ),
            AgentCapability(
                name="Citation Integration",
                description="Inline citations in consistent format"
            ),
            AgentCapability(
                name="Multi-format Output",
                description="Generate prose, lists, tables, or structured data"
            ),
            AgentCapability(
                name="Confidence Scoring",
                description="Estimate reliability of generated answer"
            ),
            AgentCapability(
                name="Uncertainty Flagging",
                description="Explicitly note when evidence is limited"
            ),
            AgentCapability(
                name="Follow-up Suggestions",
                description="Propose related questions for deeper exploration"
            ),
        ],
        
        dependencies=["context_aggregator", "citation_validator"],
        
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "array"},
                "output_format": {
                    "type": "string",
                    "enum": ["prose", "structured", "clinical_note"],
                    "default": "prose"
                },
                "max_length": {"type": "integer", "default": 1000},
                "include_confidence": {"type": "boolean", "default": True},
            },
            "required": ["query", "context"]
        },
        
        output_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pmid": {"type": "string"},
                            "title": {"type": "string"},
                            "relevance": {"type": "string"},
                        }
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            }
        },
        
        config=AgentConfig(
            timeout_seconds=60.0,
            max_retries=2,
            custom_params={
                "model": "gpt-4-turbo",
                "temperature": 0.3,
                "max_output_tokens": 2000,
                "citation_format": "pmid_inline",
            }
        ),
        
        usage_examples=[
            {
                "title": "Clinical Question",
                "query": "What EEG findings suggest non-convulsive status epilepticus?",
                "output_format": "clinical_note"
            }
        ],
        
        troubleshooting=[
            {
                "issue": "Missing citations",
                "cause": "Context didn't contain citable sources",
                "solution": "Check context aggregator output; expand retrieval"
            },
            {
                "issue": "Overly verbose response",
                "cause": "max_length not enforced",
                "solution": "Reduce max_output_tokens; add length instruction to prompt"
            },
        ]
    ),
}


def get_agent_info(agent_id: str) -> Optional[AgentInfo]:
    """Get detailed information about an agent."""
    return AGENT_REGISTRY.get(agent_id)


def get_all_agents() -> list[AgentInfo]:
    """Get all registered agents."""
    return list(AGENT_REGISTRY.values())


def get_agents_by_category(category: AgentCategory) -> list[AgentInfo]:
    """Get agents filtered by category."""
    return [a for a in AGENT_REGISTRY.values() if a.category == category]


def get_agent_dependencies(agent_id: str) -> list[str]:
    """Get list of agent IDs this agent depends on."""
    agent = AGENT_REGISTRY.get(agent_id)
    return agent.dependencies if agent else []


def update_agent_metrics(agent_id: str, latency_ms: float, success: bool, error: Optional[str] = None) -> None:
    """Update runtime metrics for an agent."""
    agent = AGENT_REGISTRY.get(agent_id)
    if agent:
        agent.metrics.total_invocations += 1
        if success:
            agent.metrics.successful_invocations += 1
        else:
            agent.metrics.failed_invocations += 1
            agent.metrics.last_error = error
        
        agent.metrics.total_latency_ms += latency_ms
        agent.metrics.avg_latency_ms = (
            agent.metrics.total_latency_ms / agent.metrics.total_invocations
        )
        agent.metrics.last_invocation = datetime.now().isoformat()


def set_agent_status(agent_id: str, status: AgentStatus) -> None:
    """Set the runtime status of an agent."""
    agent = AGENT_REGISTRY.get(agent_id)
    if agent:
        agent.status = status


__all__ = [
    "AgentStatus",
    "AgentCategory",
    "AgentCapability",
    "AgentMetrics",
    "AgentConfig",
    "AgentInfo",
    "AGENT_REGISTRY",
    "get_agent_info",
    "get_all_agents",
    "get_agents_by_category",
    "get_agent_dependencies",
    "update_agent_metrics",
    "set_agent_status",
]
