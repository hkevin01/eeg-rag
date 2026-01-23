"""
Enhanced Multi-Agent Orchestrator for EEG Literature Research.

Coordinates LocalData, PubMed, SemanticScholar, and Synthesis agents
with intelligent query planning, parallel execution, and result fusion.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time

from ..agents.base_agent import BaseAgent, AgentQuery, AgentResult, AgentType

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of research queries."""
    EXPLORATORY = "exploratory"
    SPECIFIC = "specific"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    AUTHOR_FOCUSED = "author_focused"
    DATASET_FOCUSED = "dataset_focused"
    CITATION_NETWORK = "citation_network"


class ExecutionStrategy(Enum):
    """How to execute the multi-agent query."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CASCADING = "cascading"
    ADAPTIVE = "adaptive"


@dataclass
class QueryPlan:
    """Plan for executing a research query."""
    query_id: str
    original_query: str
    query_type: QueryType
    strategy: ExecutionStrategy
    agent_tasks: List[Tuple[str, AgentQuery]]
    synthesis_config: Dict[str, Any]
    estimated_time_ms: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueryContext:
    """Context accumulated during query execution."""
    query_id: str
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    entities_found: List[Dict] = field(default_factory=list)
    papers_by_source: Dict[str, List[Dict]] = field(default_factory=dict)
    intermediate_results: List[AgentResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """Final result from orchestrator."""
    query_id: str
    success: bool
    papers: List[Dict[str, Any]]
    synthesis: Optional[Dict[str, Any]]
    sources_used: List[str]
    total_found: int
    execution_time_ms: float
    query_plan: QueryPlan
    errors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """Analyze query to determine type and optimal strategy."""
    
    def analyze(self, query: str) -> Tuple[QueryType, ExecutionStrategy, Dict[str, Any]]:
        """Analyze query and return type, strategy, and extracted parameters."""
        import re
        
        query_lower = query.lower()
        params = {}
        
        # Check for comparative patterns
        if re.search(r'compar|versus|vs\.?|difference between|which is better', query_lower):
            params['is_comparison'] = True
            return QueryType.COMPARATIVE, ExecutionStrategy.PARALLEL, params
        
        # Check for temporal patterns
        if re.search(r'evolution|history|trend|over time|recent|latest|20\d{2}', query_lower):
            year_match = re.search(r'20\d{2}', query)
            if year_match:
                params['focus_year'] = int(year_match.group())
            params['sort_by'] = 'date'
            return QueryType.TEMPORAL, ExecutionStrategy.PARALLEL, params
        
        # Check for citation network
        if re.search(r'citations?|cited by|references|influential|seminal', query_lower):
            params['include_citations'] = True
            return QueryType.CITATION_NETWORK, ExecutionStrategy.CASCADING, params
        
        # Check for author focus
        if re.search(r'papers by|works of|author|published by|researcher', query_lower):
            params['focus_on_authors'] = True
            return QueryType.AUTHOR_FOCUSED, ExecutionStrategy.PARALLEL, params
        
        # Check for dataset focus
        if re.search(r'using .* dataset|on .* data|benchmark|evaluated on', query_lower):
            return QueryType.DATASET_FOCUSED, ExecutionStrategy.PARALLEL, params
        
        # Check for specific method/technique
        if re.search(r'how to|implement|method for|technique|algorithm|architecture', query_lower):
            return QueryType.SPECIFIC, ExecutionStrategy.PARALLEL, params
        
        # Default to exploratory with cascading
        return QueryType.EXPLORATORY, ExecutionStrategy.CASCADING, params


class QueryPlanner:
    """Create execution plans for queries."""
    
    def __init__(self, analyzer: QueryAnalyzer):
        self.analyzer = analyzer
    
    def create_plan(
        self,
        query: str,
        max_results: int = 50,
        sources: Optional[List[str]] = None,
        date_range: Optional[Tuple[int, int]] = None,
        use_local: bool = True,
        use_pubmed: bool = True,
        use_s2: bool = True
    ) -> QueryPlan:
        """Create an execution plan for the query."""
        
        query_id = str(uuid.uuid4())[:8]
        
        # Analyze query
        query_type, strategy, params = self.analyzer.analyze(query)
        
        # Determine which sources to use
        if sources:
            use_local = "local" in sources
            use_pubmed = "pubmed" in sources
            use_s2 = "semantic_scholar" in sources or "s2" in sources
        
        agent_tasks = []
        
        # Create tasks for each agent
        if use_local:
            local_query = AgentQuery(
                text=query,
                context={},
                parameters={
                    "max_results": max_results,
                    "expand_query": True,
                    "rerank": True
                }
            )
            agent_tasks.append(("local", local_query))
        
        if use_pubmed:
            pubmed_query = AgentQuery(
                text=query,
                context={},
                parameters={
                    "max_results": max_results,
                    "use_mesh": True,
                    "date_range": date_range
                }
            )
            agent_tasks.append(("pubmed", pubmed_query))
        
        if use_s2:
            s2_query = AgentQuery(
                text=query,
                context={},
                parameters={
                    "max_results": max_results,
                    "year_range": date_range,
                    "fields_of_study": ["Computer Science", "Medicine", "Biology"]
                }
            )
            agent_tasks.append(("s2", s2_query))
        
        # Synthesis configuration
        synthesis_config = {
            "extract_themes": True,
            "rank_evidence": True,
            "identify_gaps": True,
            "query_type": query_type.value,
            "params": params
        }
        
        # Estimate time
        estimated_time = 500
        if use_pubmed:
            estimated_time += 800
        if use_s2:
            estimated_time += 600
        if use_local:
            estimated_time += 200
        
        return QueryPlan(
            query_id=query_id,
            original_query=query,
            query_type=query_type,
            strategy=strategy,
            agent_tasks=agent_tasks,
            synthesis_config=synthesis_config,
            estimated_time_ms=estimated_time
        )


class Orchestrator:
    """
    Main orchestrator coordinating all agents.
    
    Features:
    - Query analysis and planning
    - Parallel/cascading execution
    - Result fusion and deduplication
    - Progress callbacks
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        local_agent: Optional[BaseAgent] = None,
        pubmed_agent: Optional[BaseAgent] = None,
        s2_agent: Optional[BaseAgent] = None,
        synthesis_agent: Optional[BaseAgent] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # Initialize agents lazily to avoid import issues
        self.local_agent = local_agent
        self.pubmed_agent = pubmed_agent
        self.s2_agent = s2_agent
        self.synthesis_agent = synthesis_agent
        
        # Agent registry
        self.agents = {}
        if local_agent:
            self.agents["local"] = local_agent
        if pubmed_agent:
            self.agents["pubmed"] = pubmed_agent
        if s2_agent:
            self.agents["s2"] = s2_agent
        if synthesis_agent:
            self.agents["synthesis"] = synthesis_agent
        
        # Query planning
        self.analyzer = QueryAnalyzer()
        self.planner = QueryPlanner(self.analyzer)
        
        # Execution tracking
        self._active_queries: Dict[str, QueryContext] = {}
        
        logger.info(f"Orchestrator initialized with agents: {', '.join(self.agents.keys())}")
    
    def _ensure_agents(self):
        """Lazy initialization of agents."""
        if not self.local_agent:
            try:
                from ..agents.local_agent.local_data_agent import LocalDataAgent
                self.local_agent = LocalDataAgent()
                self.agents["local"] = self.local_agent
            except ImportError:
                logger.warning("LocalDataAgent not available")
        
        if not self.pubmed_agent:
            try:
                from ..agents.pubmed_agent.pubmed_agent import PubMedAgent
                self.pubmed_agent = PubMedAgent(
                    api_key=self.config.get("pubmed_api_key"),
                    email=self.config.get("email", "researcher@example.com")
                )
                self.agents["pubmed"] = self.pubmed_agent
            except ImportError:
                logger.warning("PubMedAgent not available")
        
        if not self.s2_agent:
            try:
                from ..agents.semantic_scholar_agent.s2_agent import SemanticScholarAgent
                self.s2_agent = SemanticScholarAgent(
                    api_key=self.config.get("s2_api_key")
                )
                self.agents["s2"] = self.s2_agent
            except ImportError:
                logger.warning("SemanticScholarAgent not available")
        
        if not self.synthesis_agent:
            try:
                from ..agents.synthesis_agent.synthesis_agent import SynthesisAgent
                self.synthesis_agent = SynthesisAgent()
                self.agents["synthesis"] = self.synthesis_agent
            except ImportError:
                logger.warning("SynthesisAgent not available")
    
    async def search(
        self,
        query: str,
        max_results: int = 50,
        sources: Optional[List[str]] = None,
        date_range: Optional[Tuple[int, int]] = None,
        synthesize: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> OrchestratorResult:
        """
        Execute a research query across all agents.
        
        Args:
            query: Research query
            max_results: Maximum results per source
            sources: Specific sources to use (default: all)
            date_range: (start_year, end_year) filter
            synthesize: Whether to synthesize results
            progress_callback: Callback for progress updates (stage, percent)
        
        Returns:
            OrchestratorResult with merged and synthesized results
        """
        start_time = time.monotonic()
        
        # Ensure agents are initialized
        self._ensure_agents()
        
        # Create execution plan
        plan = self.planner.create_plan(
            query=query,
            max_results=max_results,
            sources=sources,
            date_range=date_range
        )
        
        logger.info(f"Query {plan.query_id}: {plan.query_type.value} strategy={plan.strategy.value}")
        
        # Initialize context
        context = QueryContext(
            query_id=plan.query_id,
            original_query=query
        )
        self._active_queries[plan.query_id] = context
        
        try:
            if progress_callback:
                progress_callback("planning", 0.05)
            
            # Execute based on strategy
            if plan.strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(plan, context, progress_callback)
            elif plan.strategy == ExecutionStrategy.CASCADING:
                await self._execute_cascading(plan, context, progress_callback)
            else:
                await self._execute_parallel(plan, context, progress_callback)
            
            # Merge results
            if progress_callback:
                progress_callback("merging", 0.7)
            
            merged_papers = self._merge_papers(context)
            
            # Synthesize if requested
            synthesis = None
            if synthesize and merged_papers and self.synthesis_agent:
                if progress_callback:
                    progress_callback("synthesizing", 0.85)
                
                synthesis = await self._synthesize_results(
                    query, merged_papers, plan.synthesis_config
                )
            
            execution_time_ms = (time.monotonic() - start_time) * 1000
            
            if progress_callback:
                progress_callback("complete", 1.0)
            
            return OrchestratorResult(
                query_id=plan.query_id,
                success=True,
                papers=merged_papers,
                synthesis=synthesis,
                sources_used=list(context.papers_by_source.keys()),
                total_found=len(merged_papers),
                execution_time_ms=execution_time_ms,
                query_plan=plan,
                errors=context.errors,
                metadata={
                    "query_type": plan.query_type.value,
                    "strategy": plan.strategy.value,
                    "timing": context.timing,
                    "papers_per_source": {
                        src: len(papers)
                        for src, papers in context.papers_by_source.items()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            execution_time_ms = (time.monotonic() - start_time) * 1000
            
            return OrchestratorResult(
                query_id=plan.query_id,
                success=False,
                papers=[],
                synthesis=None,
                sources_used=[],
                total_found=0,
                execution_time_ms=execution_time_ms,
                query_plan=plan,
                errors=[str(e)] + context.errors
            )
        
        finally:
            if plan.query_id in self._active_queries:
                del self._active_queries[plan.query_id]
    
    async def _execute_parallel(
        self,
        plan: QueryPlan,
        context: QueryContext,
        progress_callback: Optional[Callable]
    ):
        """Execute all agent tasks in parallel."""
        
        async def run_agent_task(agent_id: str, query: AgentQuery):
            """Run a single agent task with timing."""
            agent = self.agents.get(agent_id)
            if not agent:
                context.errors.append(f"Agent not found: {agent_id}")
                return
            
            start = time.monotonic()
            try:
                result = await agent.execute(query)
                context.timing[agent_id] = (time.monotonic() - start) * 1000
                context.intermediate_results.append(result)
                
                if result.success:
                    papers = self._extract_papers(result, agent_id)
                    context.papers_by_source[agent_id] = papers
                    logger.info(f"Agent {agent_id}: found {len(papers)} papers")
                else:
                    context.errors.append(f"{agent_id}: {result.error}")
                    
            except Exception as e:
                context.errors.append(f"{agent_id}: {str(e)}")
                logger.error(f"Agent {agent_id} failed: {e}")
        
        # Create tasks
        tasks = [
            run_agent_task(agent_id, query)
            for agent_id, query in plan.agent_tasks
        ]
        
        # Track progress
        if progress_callback and tasks:
            total_tasks = len(tasks)
            completed = 0
            
            for coro in asyncio.as_completed(tasks):
                await coro
                completed += 1
                progress = 0.1 + (completed / total_tasks) * 0.55
                progress_callback("searching", progress)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_cascading(
        self,
        plan: QueryPlan,
        context: QueryContext,
        progress_callback: Optional[Callable]
    ):
        """Execute agents in cascading fashion."""
        
        local_tasks = [(aid, t) for aid, t in plan.agent_tasks if aid == "local"]
        external_tasks = [(aid, t) for aid, t in plan.agent_tasks if aid != "local"]
        
        # Execute local first
        if local_tasks and "local" in self.agents:
            if progress_callback:
                progress_callback("local_search", 0.1)
            
            agent_id, query = local_tasks[0]
            agent = self.agents[agent_id]
            
            start = time.monotonic()
            result = await agent.execute(query)
            context.timing[agent_id] = (time.monotonic() - start) * 1000
            
            if result.success:
                papers = self._extract_papers(result, agent_id)
                context.papers_by_source[agent_id] = papers
                
                # Check if sufficient local results
                threshold = query.parameters.get("max_results", 50) * 0.8
                if len(papers) >= threshold:
                    logger.info(f"Sufficient local results ({len(papers)})")
                    if progress_callback:
                        progress_callback("local_sufficient", 0.65)
                    return
        
        # Execute external sources
        if external_tasks:
            if progress_callback:
                progress_callback("external_search", 0.3)
            
            async def run_external(agent_id: str, query: AgentQuery):
                agent = self.agents.get(agent_id)
                if not agent:
                    return
                
                start = time.monotonic()
                try:
                    result = await agent.execute(query)
                    context.timing[agent_id] = (time.monotonic() - start) * 1000
                    
                    if result.success:
                        papers = self._extract_papers(result, agent_id)
                        context.papers_by_source[agent_id] = papers
                    else:
                        context.errors.append(f"{agent_id}: {result.error}")
                except Exception as e:
                    context.errors.append(f"{agent_id}: {str(e)}")
            
            await asyncio.gather(*[
                run_external(agent_id, query)
                for agent_id, query in external_tasks
            ], return_exceptions=True)
        
        if progress_callback:
            progress_callback("search_complete", 0.65)
    
    def _extract_papers(self, result: AgentResult, source: str) -> List[Dict]:
        """Extract papers from agent result."""
        papers = []
        data = result.data
        
        if "papers" in data:
            raw_papers = data["papers"]
        elif "documents" in data:
            raw_papers = data["documents"]
        else:
            return []
        
        for paper in raw_papers:
            normalized = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "journal": paper.get("journal", "") or paper.get("venue", ""),
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "paper_id": paper.get("paper_id") or paper.get("id"),
                "citation_count": paper.get("citation_count", 0),
                "source": source,
                "url": self._get_paper_url(paper),
                "score": paper.get("score", 0) or paper.get("evidence_score", 0)
            }
            
            # Source-specific fields
            if source == "pubmed":
                normalized["mesh_terms"] = paper.get("mesh_terms", [])
                normalized["keywords"] = paper.get("keywords", [])
            elif source == "s2":
                normalized["influential_citations"] = paper.get("influential_citation_count", 0)
                normalized["tldr"] = paper.get("tldr")
                normalized["fields_of_study"] = paper.get("fields_of_study", [])
            elif source == "local":
                normalized["entities"] = paper.get("entities", [])
            
            papers.append(normalized)
        
        return papers
    
    def _get_paper_url(self, paper: Dict) -> Optional[str]:
        """Generate URL for paper."""
        if paper.get("doi"):
            return f"https://doi.org/{paper['doi']}"
        elif paper.get("pmid"):
            return f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
        elif paper.get("paper_id"):
            return f"https://www.semanticscholar.org/paper/{paper['paper_id']}"
        return None
    
    def _merge_papers(self, context: QueryContext) -> List[Dict]:
        """Merge and deduplicate papers from all sources."""
        all_papers = []
        for papers in context.papers_by_source.values():
            all_papers.extend(papers)
        
        # Simple deduplication by title similarity
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            title_lower = paper.get("title", "").lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        # Sort by score
        unique_papers.sort(key=lambda p: p.get("score", 0), reverse=True)
        
        return unique_papers
    
    async def _synthesize_results(
        self,
        query: str,
        papers: List[Dict],
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Synthesize results using synthesis agent."""
        if not self.synthesis_agent:
            return None
        
        synth_query = AgentQuery(
            text=query,
            context={},
            parameters={
                "papers": papers,
                **config
            }
        )
        
        try:
            result = await self.synthesis_agent.execute(synth_query)
            if result.success:
                return result.data.get("synthesis")
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
        
        return None
    
    async def close(self):
        """Close all agent connections."""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'close'):
                try:
                    await agent.close()
                except Exception as e:
                    logger.error(f"Error closing {agent_id}: {e}")


async def quick_search(
    query: str,
    max_results: int = 30,
    config: Optional[Dict] = None
) -> OrchestratorResult:
    """Quick search without maintaining orchestrator state."""
    orchestrator = Orchestrator(config=config)
    try:
        return await orchestrator.search(query, max_results=max_results)
    finally:
        await orchestrator.close()
