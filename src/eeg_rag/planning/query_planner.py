"""
Query Planning Module with CoT and ReAct

This module implements sophisticated query planning strategies including:
- Chain-of-Thought (CoT) reasoning
- ReAct (Reasoning + Acting) patterns
- Query decomposition
- Execution graph generation

Requirements Covered:
- REQ-PLAN-001: Query intent classification
- REQ-PLAN-002: Chain-of-Thought reasoning
- REQ-PLAN-003: ReAct planning logic
- REQ-PLAN-004: Query decomposition for complex queries
- REQ-PLAN-005: Execution graph with dependencies
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# REQ-PLAN-006: Define query intents for routing
# ---------------------------------------------------------------------------
# ID           : planning.query_planner.QueryIntent
# Requirement  : `QueryIntent` class shall be instantiable and expose the documented interface
# Purpose      : Classification of query intent
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
# Verification : Instantiate QueryIntent with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class QueryIntent(Enum):
    """Classification of query intent"""
    FACTUAL = "factual"  # Simple fact retrieval
    COMPARISON = "comparison"  # Compare methods/biomarkers
    ANALYSIS = "analysis"  # Analyze trends/patterns
    PROCEDURE = "procedure"  # How-to questions
    CODE_GEN = "code_generation"  # Generate code
    DATASET = "dataset"  # Find datasets
    REVIEW = "review"  # Literature review
    MULTI_PART = "multi_part"  # Multiple sub-questions
    UNKNOWN = "unknown"  # Cannot classify


# REQ-PLAN-007: Define complexity levels
# ---------------------------------------------------------------------------
# ID           : planning.query_planner.QueryComplexity
# Requirement  : `QueryComplexity` class shall be instantiable and expose the documented interface
# Purpose      : Query complexity assessment
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
# Verification : Instantiate QueryComplexity with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class QueryComplexity(Enum):
    """Query complexity assessment"""
    SIMPLE = "simple"  # Single agent, single source
    MODERATE = "moderate"  # Multiple sources, single agent
    COMPLEX = "complex"  # Multiple agents, sequential
    VERY_COMPLEX = "very_complex"  # Multiple agents, parallel + sequential


# ---------------------------------------------------------------------------
# ID           : planning.query_planner.SubQuery
# Requirement  : `SubQuery` class shall be instantiable and expose the documented interface
# Purpose      : Individual sub-query from decomposition
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
# Verification : Instantiate SubQuery with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class SubQuery:
    """
    Individual sub-query from decomposition
    
    REQ-PLAN-008: Sub-queries maintain context and dependencies
    """
    text: str
    intent: QueryIntent
    priority: int = 0  # Higher = more important
    dependencies: List[int] = field(default_factory=list)  # Indices of required sub-queries
    required_agents: List[str] = field(default_factory=list)
    estimated_complexity: QueryComplexity = QueryComplexity.SIMPLE
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.SubQuery.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "intent": self.intent.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "required_agents": self.required_agents,
            "estimated_complexity": self.estimated_complexity.value
        }


# ---------------------------------------------------------------------------
# ID           : planning.query_planner.CoTStep
# Requirement  : `CoTStep` class shall be instantiable and expose the documented interface
# Purpose      : Single step in Chain-of-Thought reasoning
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
# Verification : Instantiate CoTStep with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class CoTStep:
    """
    Single step in Chain-of-Thought reasoning
    
    REQ-PLAN-009: Capture reasoning process for transparency
    """
    step_number: int
    thought: str
    reasoning: str
    conclusion: str
    confidence: float = 1.0  # 0.0 to 1.0
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.CoTStep.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "conclusion": self.conclusion,
            "confidence": self.confidence
        }


# ---------------------------------------------------------------------------
# ID           : planning.query_planner.ReActAction
# Requirement  : `ReActAction` class shall be instantiable and expose the documented interface
# Purpose      : Action in ReAct (Reasoning + Acting) pattern
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
# Verification : Instantiate ReActAction with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class ReActAction:
    """
    Action in ReAct (Reasoning + Acting) pattern
    
    REQ-PLAN-010: Actions have reasoning and expected outcomes
    """
    action_type: str  # "search_local", "search_web", "query_mcp", etc.
    reasoning: str  # Why this action?
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    agent_name: Optional[str] = None
    parallel_group: int = 0  # Actions in same group run in parallel
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.ReActAction.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_type": self.action_type,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "agent_name": self.agent_name,
            "parallel_group": self.parallel_group
        }


# ---------------------------------------------------------------------------
# ID           : planning.query_planner.QueryPlan
# Requirement  : `QueryPlan` class shall be instantiable and expose the documented interface
# Purpose      : Complete query execution plan
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
# Verification : Instantiate QueryPlan with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class QueryPlan:
    """
    Complete query execution plan
    
    REQ-PLAN-011: Comprehensive plan with all execution details
    """
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    cot_reasoning: List[CoTStep] = field(default_factory=list)
    sub_queries: List[SubQuery] = field(default_factory=list)
    actions: List[ReActAction] = field(default_factory=list)
    required_agents: Set[str] = field(default_factory=set)
    estimated_latency: float = 0.0  # Estimated time in seconds
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extended metadata
    
    # REQ-PLAN-012: Track execution requirements
    requires_local_data: bool = False
    requires_web_search: bool = False
    requires_cloud_kb: bool = False
    requires_mcp_tools: bool = False
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlan.to_dict
    # Requirement  : `to_dict` shall convert to dictionary
    # Purpose      : Convert to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "cot_reasoning": [step.to_dict() for step in self.cot_reasoning],
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "actions": [action.to_dict() for action in self.actions],
            "required_agents": list(self.required_agents),
            "estimated_latency": self.estimated_latency,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "requires_local_data": self.requires_local_data,
            "requires_web_search": self.requires_web_search,
            "requires_cloud_kb": self.requires_cloud_kb,
            "requires_mcp_tools": self.requires_mcp_tools,
            "timestamp": self.timestamp.isoformat()
        }


# ---------------------------------------------------------------------------
# ID           : planning.query_planner.QueryPlanner
# Requirement  : `QueryPlanner` class shall be instantiable and expose the documented interface
# Purpose      : Sophisticated query planner with CoT and ReAct
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
# Verification : Instantiate QueryPlanner with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class QueryPlanner:
    """
    Sophisticated query planner with CoT and ReAct
    
    REQ-PLAN-013: Main planning interface
    """
    
    # REQ-PLAN-014: EEG-specific keywords for intent classification
    EEG_KEYWORDS = {
        "erp_components": ["p300", "n400", "n170", "p100", "n100", "mmn", "ern", "pe"],
        "frequency_bands": ["delta", "theta", "alpha", "beta", "gamma"],
        "clinical": ["epilepsy", "seizure", "sleep", "coma", "encephalopathy"],
        "paradigms": ["oddball", "motor imagery", "resting state", "task"],
        "biomarkers": ["spike-wave", "spindles", "k-complex", "asymmetry"],
        "analysis": ["fft", "wavelet", "ica", "source localization", "connectivity"],
        "datasets": ["dataset", "database", "corpus", "physionet", "openneuro"]
    }
    
    # REQ-PLAN-015: Comparison keywords
    COMPARISON_KEYWORDS = [
        "compare", "versus", "vs", "difference", "better", "worse",
        "which is", "contrast", "comparison"
    ]
    
    # REQ-PLAN-016: Code generation keywords
    CODE_KEYWORDS = [
        "code", "python", "matlab", "script", "function", "implement",
        "how to", "example", "snippet"
    ]
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner.__init__
    # Requirement  : `__init__` shall initialize query planner
    # Purpose      : Initialize query planner
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : logger: Optional[logging.Logger] (default=None)
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
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize query planner
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("eeg_rag.planning.query_planner")
        self.logger.info("Initialized QueryPlanner")
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner.plan
    # Requirement  : `plan` shall create comprehensive query plan
    # Purpose      : Create comprehensive query plan
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : QueryPlan
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
    def plan(self, query: str) -> QueryPlan:
        """
        Create comprehensive query plan
        
        Args:
            query: User query string
            
        Returns:
            QueryPlan with full execution strategy
            
        REQ-PLAN-017: Main planning method
        """
        self.logger.info(f"Planning query: '{query}'")
        
        # Step 1: Classify intent
        intent = self._classify_intent(query)
        self.logger.debug(f"Classified intent: {intent.value}")
        
        # Step 2: Chain-of-Thought reasoning
        cot_steps = self._chain_of_thought_reasoning(query, intent)
        self.logger.debug(f"Generated {len(cot_steps)} CoT steps")
        
        # Step 3: Decompose query if needed
        sub_queries = self._decompose_query(query, intent)
        self.logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")
        
        # Step 4: Assess complexity
        complexity = self._assess_complexity(query, intent, sub_queries)
        self.logger.debug(f"Assessed complexity: {complexity.value}")
        
        # Step 5: ReAct planning - determine actions
        actions = self._plan_actions(query, intent, sub_queries, cot_steps)
        self.logger.debug(f"Planned {len(actions)} actions")
        
        # Step 6: Identify required agents
        required_agents = self._identify_required_agents(actions)
        
        # Step 7: Create execution plan
        plan = QueryPlan(
            original_query=query,
            intent=intent,
            complexity=complexity,
            cot_reasoning=cot_steps,
            sub_queries=sub_queries,
            actions=actions,
            required_agents=required_agents
        )
        
        # Step 8: Set agent requirements
        plan.requires_local_data = self._check_requirement(actions, "local")
        plan.requires_web_search = self._check_requirement(actions, "web")
        plan.requires_cloud_kb = self._check_requirement(actions, "cloud")
        plan.requires_mcp_tools = self._check_requirement(actions, "mcp")
        
        # Step 9: Estimate latency
        plan.estimated_latency = self._estimate_latency(plan)
        
        self.logger.info(
            f"Query plan complete: intent={intent.value}, "
            f"complexity={complexity.value}, "
            f"actions={len(actions)}, "
            f"estimated_latency={plan.estimated_latency:.2f}s"
        )
        
        return plan
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._classify_intent
    # Requirement  : `_classify_intent` shall classify query intent
    # Purpose      : Classify query intent
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : QueryIntent
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
    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent
        
        Args:
            query: Query string
            
        Returns:
            QueryIntent classification
            
        REQ-PLAN-018: Intent classification logic
        """
        query_lower = query.lower()
        
        # Check for comparison
        if any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON
        
        # Check for code generation
        if any(keyword in query_lower for keyword in self.CODE_KEYWORDS):
            return QueryIntent.CODE_GEN
        
        # Check for dataset queries
        if any(keyword in query_lower for keyword in self.EEG_KEYWORDS["datasets"]):
            return QueryIntent.DATASET
        
        # Check for multi-part (contains "and" with multiple questions)
        if " and " in query_lower and "?" in query:
            parts = query.split(" and ")
            if len(parts) > 1 and any("?" in p for p in parts):
                return QueryIntent.MULTI_PART
        
        # Check for procedure (how-to)
        if query_lower.startswith(("how to", "how do", "how can")):
            return QueryIntent.PROCEDURE
        
        # Check for analysis
        if any(keyword in query_lower for keyword in ["analyze", "analysis", "trend", "pattern"]):
            return QueryIntent.ANALYSIS
        
        # Default to factual
        return QueryIntent.FACTUAL
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._chain_of_thought_reasoning
    # Requirement  : `_chain_of_thought_reasoning` shall generate Chain-of-Thought reasoning steps
    # Purpose      : Generate Chain-of-Thought reasoning steps
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; intent: QueryIntent
    # Outputs      : List[CoTStep]
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
    def _chain_of_thought_reasoning(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[CoTStep]:
        """
        Generate Chain-of-Thought reasoning steps
        
        Args:
            query: Query string
            intent: Classified intent
            
        Returns:
            List of CoT reasoning steps
            
        REQ-PLAN-019: CoT implementation
        """
        steps = []
        
        # Step 1: Understand the query
        steps.append(CoTStep(
            step_number=1,
            thought="Understanding the query",
            reasoning=f"Query intent is {intent.value}. Need to identify key concepts.",
            conclusion=f"This is a {intent.value} query requiring specific handling.",
            confidence=0.9
        ))
        
        # Step 2: Identify key concepts (EEG-specific)
        eeg_concepts = []
        for category, keywords in self.EEG_KEYWORDS.items():
            found = [kw for kw in keywords if kw in query.lower()]
            if found:
                eeg_concepts.extend([(category, kw) for kw in found])
        
        if eeg_concepts:
            concepts_str = ", ".join([f"{cat}:{kw}" for cat, kw in eeg_concepts])
            steps.append(CoTStep(
                step_number=2,
                thought="Identifying EEG-specific concepts",
                reasoning=f"Found concepts: {concepts_str}",
                conclusion="Query contains domain-specific terminology that requires specialized knowledge.",
                confidence=1.0
            ))
        
        # Step 3: Determine data sources needed
        steps.append(CoTStep(
            step_number=3,
            thought="Determining required data sources",
            reasoning=self._reason_about_sources(query, intent),
            conclusion="Will route to appropriate agents based on requirements.",
            confidence=0.85
        ))
        
        return steps
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._reason_about_sources
    # Requirement  : `_reason_about_sources` shall helper to reason about required data sources
    # Purpose      : Helper to reason about required data sources
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; intent: QueryIntent
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
    def _reason_about_sources(self, query: str, intent: QueryIntent) -> str:
        """Helper to reason about required data sources"""
        query_lower = query.lower()
        sources = []
        
        if any(kw in query_lower for kw in ["latest", "recent", "new", "2024", "2025"]):
            sources.append("web search (for recent papers)")
        
        if intent in [QueryIntent.FACTUAL, QueryIntent.COMPARISON]:
            sources.append("local data (for established facts)")
        
        if intent == QueryIntent.CODE_GEN:
            sources.append("MCP tools (for code snippets)")
        
        if intent == QueryIntent.DATASET:
            sources.append("web search + cloud KB (for dataset repositories)")
        
        if not sources:
            sources.append("local data (default)")
        
        return "Need: " + ", ".join(sources)
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._decompose_query
    # Requirement  : `_decompose_query` shall decompose complex queries into sub-queries
    # Purpose      : Decompose complex queries into sub-queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; intent: QueryIntent
    # Outputs      : List[SubQuery]
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
    def _decompose_query(self, query: str, intent: QueryIntent) -> List[SubQuery]:
        """
        Decompose complex queries into sub-queries
        
        Args:
            query: Query string
            intent: Classified intent
            
        Returns:
            List of sub-queries
            
        REQ-PLAN-020: Query decomposition logic
        """
        sub_queries = []
        
        if intent == QueryIntent.MULTI_PART:
            # Split on "and" or other conjunctions
            parts = re.split(r'\s+and\s+|\s+also\s+', query, flags=re.IGNORECASE)
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    sub_intent = self._classify_intent(part)
                    sub_queries.append(SubQuery(
                        text=part,
                        intent=sub_intent,
                        priority=len(parts) - i,  # Earlier parts higher priority
                        required_agents=self._suggest_agents_for_intent(sub_intent)
                    ))
        
        elif intent == QueryIntent.COMPARISON:
            # Extract items being compared
            items = self._extract_comparison_items(query)
            for i, item in enumerate(items):
                sub_queries.append(SubQuery(
                    text=f"Information about {item}",
                    intent=QueryIntent.FACTUAL,
                    priority=len(items) - i,
                    required_agents=["local_data", "web_search"]
                ))
        
        # If no sub-queries, use original as single sub-query
        if not sub_queries:
            sub_queries.append(SubQuery(
                text=query,
                intent=intent,
                priority=1,
                required_agents=self._suggest_agents_for_intent(intent)
            ))
        
        return sub_queries
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._extract_comparison_items
    # Requirement  : `_extract_comparison_items` shall extract items being compared from query
    # Purpose      : Extract items being compared from query
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
    # Outputs      : List[str]
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
    def _extract_comparison_items(self, query: str) -> List[str]:
        """Extract items being compared from query"""
        # Simple heuristic - look for patterns like "X vs Y" or "X and Y"
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:vs|versus)\s+(\w+(?:\s+\w+)*)',
            r'(?:compare|comparison of)\s+(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return []
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._suggest_agents_for_intent
    # Requirement  : `_suggest_agents_for_intent` shall suggest agents based on intent
    # Purpose      : Suggest agents based on intent
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : intent: QueryIntent
    # Outputs      : List[str]
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
    def _suggest_agents_for_intent(self, intent: QueryIntent) -> List[str]:
        """Suggest agents based on intent"""
        agent_map = {
            QueryIntent.FACTUAL: ["local_data"],
            QueryIntent.COMPARISON: ["local_data", "cloud_kb"],
            QueryIntent.CODE_GEN: ["mcp_server", "web_search"],
            QueryIntent.DATASET: ["web_search", "cloud_kb"],
            QueryIntent.PROCEDURE: ["mcp_server", "local_data"],
            QueryIntent.ANALYSIS: ["local_data", "cloud_kb"],
            QueryIntent.REVIEW: ["local_data", "web_search", "cloud_kb"]
        }
        return agent_map.get(intent, ["local_data"])
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._assess_complexity
    # Requirement  : `_assess_complexity` shall assess query complexity
    # Purpose      : Assess query complexity
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; intent: QueryIntent; sub_queries: List[SubQuery]
    # Outputs      : QueryComplexity
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
    def _assess_complexity(
        self,
        query: str,
        intent: QueryIntent,
        sub_queries: List[SubQuery]
    ) -> QueryComplexity:
        """
        Assess query complexity
        
        REQ-PLAN-021: Complexity assessment
        """
        # Simple query: single sub-query, factual intent
        if len(sub_queries) == 1 and intent == QueryIntent.FACTUAL:
            return QueryComplexity.SIMPLE
        
        # Moderate: 2-3 sub-queries or comparison
        if len(sub_queries) <= 3 or intent == QueryIntent.COMPARISON:
            return QueryComplexity.MODERATE
        
        # Complex: 4+ sub-queries or requires multiple specialized agents
        required_agent_types = set()
        for sq in sub_queries:
            required_agent_types.update(sq.required_agents)
        
        if len(required_agent_types) >= 3 or len(sub_queries) > 3:
            return QueryComplexity.COMPLEX
        
        # Very complex: many sub-queries with dependencies
        if len(sub_queries) > 5:
            return QueryComplexity.VERY_COMPLEX
        
        return QueryComplexity.MODERATE
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._plan_actions
    # Requirement  : `_plan_actions` shall plan ReAct actions based on reasoning
    # Purpose      : Plan ReAct actions based on reasoning
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; intent: QueryIntent; sub_queries: List[SubQuery]; cot_steps: List[CoTStep]
    # Outputs      : List[ReActAction]
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
    def _plan_actions(
        self,
        query: str,
        intent: QueryIntent,
        sub_queries: List[SubQuery],
        cot_steps: List[CoTStep]
    ) -> List[ReActAction]:
        """
        Plan ReAct actions based on reasoning
        
        REQ-PLAN-022: ReAct action planning
        """
        actions = []
        parallel_group = 0
        
        for i, sub_query in enumerate(sub_queries):
            # Determine if this can run in parallel with previous
            can_parallel = (i > 0 and len(sub_query.dependencies) == 0)
            if not can_parallel:
                parallel_group += 1
            
            # Create actions for each required agent
            for agent_name in sub_query.required_agents:
                action = ReActAction(
                    action_type=f"search_{agent_name}",
                    reasoning=f"Need {agent_name} to answer: '{sub_query.text}'",
                    parameters={"query": sub_query.text, "top_k": 10},
                    expected_outcome=f"Relevant documents from {agent_name}",
                    agent_name=agent_name,
                    parallel_group=parallel_group
                )
                actions.append(action)
        
        return actions
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._identify_required_agents
    # Requirement  : `_identify_required_agents` shall extract unique agent names from actions
    # Purpose      : Extract unique agent names from actions
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : actions: List[ReActAction]
    # Outputs      : Set[str]
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
    def _identify_required_agents(self, actions: List[ReActAction]) -> Set[str]:
        """Extract unique agent names from actions"""
        return {action.agent_name for action in actions if action.agent_name}
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._check_requirement
    # Requirement  : `_check_requirement` shall check if any action requires specific agent type
    # Purpose      : Check if any action requires specific agent type
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : actions: List[ReActAction]; keyword: str
    # Outputs      : bool
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
    def _check_requirement(self, actions: List[ReActAction], keyword: str) -> bool:
        """Check if any action requires specific agent type"""
        return any(keyword in action.action_type.lower() for action in actions)
    
    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._estimate_latency
    # Requirement  : `_estimate_latency` shall estimate query latency based on plan
    # Purpose      : Estimate query latency based on plan
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : plan: QueryPlan
    # Outputs      : float
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
    def _estimate_latency(self, plan: QueryPlan) -> float:
        """
        Estimate query latency based on plan
        
        REQ-PLAN-023: Latency estimation
        """
        base_times = {
            "local": 0.1,  # 100ms for local search
            "web": 1.5,    # 1.5s for web search
            "cloud": 0.8,  # 800ms for cloud KB
            "mcp": 0.5     # 500ms for MCP tools
        }
        
        # Group actions by parallel group
        parallel_groups = {}
        for action in plan.actions:
            group = action.parallel_group
            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append(action)
        
        # Sum max time per parallel group (parallel actions don't add time)
        total_time = 0.0
        for group_actions in parallel_groups.values():
            max_group_time = 0.0
            for action in group_actions:
                for key, time in base_times.items():
                    if key in action.action_type:
                        max_group_time = max(max_group_time, time)
            total_time += max_group_time
        
        # Add generation time (assume 2-3s)
        total_time += 2.5
        
        # Add overhead (10%)
        total_time *= 1.1
        
        return round(total_time, 2)

    # ---------------------------------------------------------------------------
    # ID           : planning.query_planner.QueryPlanner._estimate_execution_time
    # Requirement  : `_estimate_execution_time` shall estimate execution time for a list of actions
    # Purpose      : Estimate execution time for a list of actions
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : actions: List[ReActAction]
    # Outputs      : float
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
    def _estimate_execution_time(self, actions: List[ReActAction]) -> float:
        """
        Estimate execution time for a list of actions
        
        REQ-PLAN-024: Execution time estimation for action lists
        
        Args:
            actions: List of ReActAction to estimate time for
            
        Returns:
            Estimated execution time in seconds
        """
        base_times = {
            "local": 0.1,   # 100ms for local search
            "web": 1.5,     # 1.5s for web search
            "cloud": 0.8,   # 800ms for cloud KB
            "mcp": 0.5,     # 500ms for MCP tools
            "graph": 0.3,   # 300ms for knowledge graph
            "citation": 0.4 # 400ms for citation validation
        }
        
        # Group actions by parallel group
        parallel_groups = {}
        for action in actions:
            group = action.parallel_group
            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append(action)
        
        # Sum max time per parallel group (parallel actions don't add time)
        total_time = 0.0
        for group_actions in parallel_groups.values():
            max_group_time = 0.1  # Minimum time per group
            for action in group_actions:
                for key, time in base_times.items():
                    if key in action.action_type.lower():
                        max_group_time = max(max_group_time, time)
            total_time += max_group_time
        
        # Add overhead (10%)
        total_time *= 1.1
        
        return round(total_time, 2)


# REQ-PLAN-024: Export public interface
__all__ = [
    "QueryIntent",
    "QueryComplexity",
    "SubQuery",
    "CoTStep",
    "ReActAction",
    "QueryPlan",
    "QueryPlanner"
]
