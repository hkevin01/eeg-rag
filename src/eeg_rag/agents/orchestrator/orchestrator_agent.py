"""
Orchestrator Agent for Agentic RAG System

This agent coordinates the entire multi-agent workflow:
1. Query understanding and planning
2. Parallel agent execution
3. Result aggregation
4. Adaptive replanning on failures

Requirements Covered:
- REQ-ORCH-001: Query understanding and intent classification
- REQ-ORCH-002: Multi-agent coordination
- REQ-ORCH-003: Parallel execution management
- REQ-ORCH-004: Adaptive replanning
- REQ-ORCH-005: Memory integration
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import base agent framework
from eeg_rag.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    AgentQuery,
    AgentRegistry
)

# Import planning system
from eeg_rag.planning.query_planner import (
    QueryPlanner,
    QueryPlan,
    ReActAction,
    QueryComplexity
)

# Import memory system
from eeg_rag.memory.memory_manager import MemoryManager


@dataclass
class ExecutionNode:
    """
    Node in the execution graph
    
    REQ-ORCH-006: Execution graph node tracking
    """
    action: ReActAction
    agent_name: str
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[AgentResult] = None
    dependencies: List[str] = field(default_factory=list)
    parallel_group: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if node execution is complete"""
        return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]


@dataclass
class ExecutionPlan:
    """
    Complete execution plan with dependency graph
    
    REQ-ORCH-007: Execution plan management
    """
    query_plan: QueryPlan
    nodes: List[ExecutionNode]
    parallel_groups: Dict[int, List[ExecutionNode]]
    
    def __post_init__(self):
        """Build parallel groups from nodes"""
        self.parallel_groups = {}
        for node in self.nodes:
            group_id = node.parallel_group
            if group_id not in self.parallel_groups:
                self.parallel_groups[group_id] = []
            self.parallel_groups[group_id].append(node)
    
    def get_ready_nodes(self) -> List[ExecutionNode]:
        """
        Get nodes that are ready to execute
        
        Returns:
            List of nodes whose dependencies are satisfied
            
        REQ-ORCH-008: Dependency resolution
        """
        ready = []
        for node in self.nodes:
            if node.status != AgentStatus.IDLE:
                continue
            
            # Check if all dependencies are complete
            deps_satisfied = all(
                any(n.agent_name == dep and n.is_complete 
                    for n in self.nodes)
                for dep in node.dependencies
            )
            
            if deps_satisfied:
                ready.append(node)
        
        return ready
    
    def get_next_parallel_group(self) -> Optional[List[ExecutionNode]]:
        """
        Get next group of nodes that can execute in parallel
        
        Returns:
            List of nodes in the same parallel group, or None if none ready
            
        REQ-ORCH-009: Parallel group execution
        """
        ready_nodes = self.get_ready_nodes()
        if not ready_nodes:
            return None
        
        # Group by parallel_group
        min_group = min(node.parallel_group for node in ready_nodes)
        return [n for n in ready_nodes if n.parallel_group == min_group]
    
    def all_complete(self) -> bool:
        """Check if all nodes are complete"""
        return all(node.is_complete for node in self.nodes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        completed = [n for n in self.nodes if n.status == AgentStatus.COMPLETED]
        failed = [n for n in self.nodes if n.status == AgentStatus.FAILED]
        
        return {
            "total_nodes": len(self.nodes),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.nodes) if self.nodes else 0.0,
            "total_time": sum(n.elapsed_time for n in self.nodes),
            "parallel_groups": len(self.parallel_groups)
        }


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the entire multi-agent workflow
    
    REQ-ORCH-010: Main orchestrator implementation
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        agent_registry: AgentRegistry,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize orchestrator
        
        Args:
            memory_manager: Memory management system
            agent_registry: Registry of available agents
            config: Configuration options
            logger: Logger instance
        """
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            name="orchestrator",
            config=config or {},
            logger=logger or logging.getLogger("eeg_rag.orchestrator")
        )
        
        # Core components
        self.memory = memory_manager
        self.registry = agent_registry
        self.planner = QueryPlanner(logger=self.logger)
        
        # Configuration
        self.max_replanning_attempts = config.get("max_replanning_attempts", 2) if config else 2
        self.enable_adaptive_replanning = config.get("enable_adaptive_replanning", True) if config else True
        
        self.logger.info("OrchestratorAgent initialized")
    
    async def execute(self, query: AgentQuery) -> AgentResult:
        """
        Execute the full orchestration workflow
        
        Args:
            query: User query with context
            
        Returns:
            AgentResult with orchestrated results
            
        REQ-ORCH-011: Main execution workflow
        """
        try:
            # Step 1: Add query to memory
            self.memory.add_query(query.text, metadata={"query_id": query.query_id})
            self.logger.info(f"Processing query: {query.text[:50]}...")
            
            # Step 2: Get recent context from memory
            recent_context = self.memory.get_recent_context(n=3)
            self.logger.debug(f"Retrieved {recent_context['context_size']} recent context items")
            
            # Step 3: Plan the query
            query_plan = await self._plan_query(query, recent_context)
            self.logger.info(
                f"Query plan created: {query_plan.complexity.value} complexity, "
                f"{len(query_plan.actions)} actions, "
                f"estimated latency {query_plan.estimated_latency:.2f}s"
            )
            
            # Step 4: Create execution plan
            exec_plan = self._create_execution_plan(query_plan)
            self.logger.info(
                f"Execution plan: {len(exec_plan.nodes)} nodes, "
                f"{len(exec_plan.parallel_groups)} parallel groups"
            )
            
            # Step 5: Execute plan with possible replanning
            results = await self._execute_plan(exec_plan, query)
            
            # Step 6: Collect and aggregate results
            aggregated_data = self._aggregate_results(results, query_plan)
            
            # Step 7: Get execution statistics
            exec_stats = exec_plan.get_statistics()
            
            return AgentResult(
                success=True,
                data=aggregated_data,
                metadata={
                    "query_plan": {
                        "intent": query_plan.intent.value,
                        "complexity": query_plan.complexity.value,
                        "sub_queries": len(query_plan.sub_queries),
                        "actions": len(query_plan.actions),
                        "estimated_latency": query_plan.estimated_latency
                    },
                    "execution_stats": exec_stats,
                    "recent_context": recent_context,
                    "agent_results": len(results)
                },
                agent_type=AgentType.ORCHESTRATOR
            )
            
        except Exception as e:
            self.logger.exception(f"Orchestration failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.ORCHESTRATOR
            )
    
    async def _plan_query(
        self,
        query: AgentQuery,
        recent_context: Dict[str, Any]
    ) -> QueryPlan:
        """
        Plan the query using QueryPlanner
        
        Args:
            query: User query
            recent_context: Recent conversation context
            
        Returns:
            QueryPlan with actions and agents
            
        REQ-ORCH-012: Query planning integration
        """
        # Enhance query with context if available
        enhanced_query = query.text
        if recent_context.get("recent_queries"):
            # Consider previous queries for context
            self.logger.debug("Enhancing query with recent context")
        
        # Plan the query
        plan = self.planner.plan(enhanced_query)
        
        return plan
    
    def _create_execution_plan(self, query_plan: QueryPlan) -> ExecutionPlan:
        """
        Create execution plan from query plan
        
        Args:
            query_plan: Query plan with actions
            
        Returns:
            ExecutionPlan with nodes and dependencies
            
        REQ-ORCH-013: Execution plan creation
        """
        nodes = []
        
        for action in query_plan.actions:
            node = ExecutionNode(
                action=action,
                agent_name=action.agent_name,
                parallel_group=action.parallel_group,
                dependencies=[]  # Could extract from action parameters
            )
            nodes.append(node)
        
        exec_plan = ExecutionPlan(
            query_plan=query_plan,
            nodes=nodes,
            parallel_groups={}
        )
        
        return exec_plan
    
    async def _execute_plan(
        self,
        exec_plan: ExecutionPlan,
        query: AgentQuery
    ) -> List[AgentResult]:
        """
        Execute the plan with parallel execution and adaptive replanning
        
        Args:
            exec_plan: Execution plan
            query: Original query
            
        Returns:
            List of agent results
            
        REQ-ORCH-014: Plan execution with replanning
        """
        all_results = []
        replanning_attempts = 0
        
        while not exec_plan.all_complete():
            # Get next group of nodes to execute in parallel
            parallel_group = exec_plan.get_next_parallel_group()
            
            if not parallel_group:
                # No more nodes ready - check if we need replanning
                if self.enable_adaptive_replanning and replanning_attempts < self.max_replanning_attempts:
                    self.logger.warning("No ready nodes, attempting adaptive replanning")
                    replanning_attempts += 1
                    # TODO: Implement adaptive replanning logic
                    break
                else:
                    self.logger.error("No ready nodes and replanning exhausted")
                    break
            
            # Execute group in parallel
            self.logger.info(f"Executing parallel group with {len(parallel_group)} nodes")
            group_results = await self._execute_parallel_group(parallel_group, query)
            
            all_results.extend(group_results)
        
        return all_results
    
    async def _execute_parallel_group(
        self,
        nodes: List[ExecutionNode],
        query: AgentQuery
    ) -> List[AgentResult]:
        """
        Execute a group of nodes in parallel
        
        Args:
            nodes: Nodes to execute
            query: Original query
            
        Returns:
            List of agent results
            
        REQ-ORCH-015: Parallel group execution
        """
        # Mark nodes as executing
        for node in nodes:
            node.status = AgentStatus.EXECUTING
            node.start_time = datetime.now()
        
        # Create tasks for parallel execution
        tasks = []
        for node in nodes:
            task = self._execute_node(node, query)
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_results = []
        for i, (node, result) in enumerate(zip(nodes, results)):
            node.end_time = datetime.now()
            
            if isinstance(result, Exception):
                self.logger.error(f"Node {node.agent_name} failed: {result}")
                node.status = AgentStatus.FAILED
                node.result = AgentResult(
                    success=False,
                    data={},
                    error=str(result),
                    agent_type=AgentType.ORCHESTRATOR
                )
            else:
                node.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
                node.result = result
                agent_results.append(result)
                
                self.logger.info(
                    f"Node {node.agent_name} completed in {node.elapsed_time:.2f}s "
                    f"(success={result.success})"
                )
        
        return agent_results
    
    async def _execute_node(
        self,
        node: ExecutionNode,
        query: AgentQuery
    ) -> AgentResult:
        """
        Execute a single node (agent)
        
        Args:
            node: Execution node
            query: Original query
            
        Returns:
            AgentResult from the agent
            
        REQ-ORCH-016: Single agent execution
        """
        try:
            # Get agent from registry
            agent = self.registry.get(node.agent_name)
            
            if not agent:
                raise ValueError(f"Agent not found: {node.agent_name}")
            
            # Create agent-specific query from action
            agent_query = AgentQuery(
                text=query.text,
                intent=query.intent,
                context={
                    "action": node.action.action_type,
                    "reasoning": node.action.reasoning,
                    "parameters": node.action.parameters,
                    "expected_outcome": node.action.expected_outcome
                },
                parameters=node.action.parameters,
                query_id=query.query_id,
                timestamp=datetime.now()
            )
            
            # Execute agent
            result = await agent.run(agent_query)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Node execution failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                agent_type=AgentType.ORCHESTRATOR
            )
    
    def _aggregate_results(
        self,
        results: List[AgentResult],
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents
        
        Args:
            results: List of agent results
            query_plan: Original query plan
            
        Returns:
            Aggregated data dictionary
            
        REQ-ORCH-017: Result aggregation
        """
        # Separate successful and failed results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Collect data by agent type
        data_by_type = {}
        for result in successful:
            agent_type = result.agent_type.value
            if agent_type not in data_by_type:
                data_by_type[agent_type] = []
            data_by_type[agent_type].append(result.data)
        
        aggregated = {
            "agent_results": data_by_type,
            "successful_agents": len(successful),
            "failed_agents": len(failed),
            "total_agents": len(results),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "query_complexity": query_plan.complexity.value,
            "query_intent": query_plan.intent.value
        }
        
        # Add timing information
        if successful:
            aggregated["agent_times"] = {
                r.agent_type.value: r.elapsed_time
                for r in successful
            }
            aggregated["total_agent_time"] = sum(r.elapsed_time for r in successful)
        
        return aggregated


class EnhancedOrchestratorAgent(OrchestratorAgent):
    """
    Enhanced orchestrator with advanced coordination capabilities
    
    REQ-ORCH-019: Advanced multi-agent coordination
    REQ-ORCH-020: Circuit breakers and load balancing
    REQ-ORCH-021: Adaptive replanning and optimization
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        memory_manager: MemoryManager,
        planner: QueryPlanner,
        performance_monitor: Optional['PerformanceMonitor'] = None,
        enable_coordination: bool = True,
        **kwargs
    ):
        super().__init__(registry, memory_manager, planner, **kwargs)
        
        # Enhanced coordination
        if enable_coordination:
            self.coordinator = AgentCoordinator()
            self._setup_coordination()
        else:
            self.coordinator = None
        
        # Performance monitoring
        self.performance_monitor = performance_monitor
        
        # Adaptive planning parameters
        self.adaptation_enabled = True
        self.success_threshold = 0.8  # 80% success rate required
        self.performance_history = deque(maxlen=100)
        
        self.logger.info("Enhanced orchestrator initialized with coordination features")
    
    def _setup_coordination(self) -> None:
        """Setup circuit breakers and load balancers for registered agents"""
        if not self.coordinator:
            return
        
        # Setup circuit breakers for each agent type
        agent_types = {agent.agent_type for agent in self.registry.get_all()}
        
        for agent_type in agent_types:
            # Add circuit breaker for each agent
            agents = self.registry.get_by_type(agent_type)
            for agent in agents:
                self.coordinator.add_circuit_breaker(
                    agent.name,
                    failure_threshold=3,
                    timeout_seconds=30.0
                )
            
            # Add load balancer if multiple agents of same type
            if len(agents) > 1:
                self.coordinator.add_load_balancer(
                    agent_type.value,
                    strategy="weighted_round_robin"
                )
                
                # Add agents to load balancer
                load_balancer = self.coordinator.load_balancers[agent_type.value]
                for agent in agents:
                    load_balancer.add_node(agent, weight=1.0)
        
        self.logger.info(f"Coordination setup complete: {len(self.coordinator.circuit_breakers)} circuit breakers, "
                        f"{len(self.coordinator.load_balancers)} load balancers")
    
    async def execute_with_coordination(
        self,
        query: AgentQuery,
        enable_monitoring: bool = True,
        enable_adaptation: bool = True
    ) -> AgentResult:
        """Execute query with enhanced coordination and monitoring"""
        
        execution_start = datetime.now()
        
        try:
            # Performance monitoring context
            if enable_monitoring and self.performance_monitor:
                with monitor_performance(self.performance_monitor, "orchestrator_execution"):
                    result = await self._execute_with_enhancements(
                        query, enable_adaptation
                    )
            else:
                result = await self._execute_with_enhancements(
                    query, enable_adaptation
                )
            
            # Record performance metrics
            execution_time = (datetime.now() - execution_start).total_seconds()
            self.performance_history.append({
                'success': result.success,
                'execution_time': execution_time,
                'timestamp': execution_start
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced orchestrator execution failed: {str(e)}")
            return AgentResult(
                success=False,
                data=None,
                error=str(e),
                agent_type=self.agent_type
            )
    
    async def _execute_with_enhancements(
        self,
        query: AgentQuery,
        enable_adaptation: bool
    ) -> AgentResult:
        """Execute with coordination and adaptive features"""
        
        # Create execution plan
        plan = await self._create_enhanced_execution_plan(query)
        
        # Execute with coordination if available
        if self.coordinator:
            return await self._execute_plan_with_coordination(plan)
        else:
            return await self._execute_plan(plan)
    
    async def _create_enhanced_execution_plan(self, query: AgentQuery) -> ExecutionPlan:
        """Create execution plan with adaptive optimizations"""
        
        # Get base plan from planner
        query_plan = await self.planner.plan_query(query.text, query.context)
        
        # Apply adaptive optimizations based on performance history
        if self.adaptation_enabled and self.performance_history:
            query_plan = self._optimize_plan_based_on_history(query_plan)
        
        # Create execution nodes
        nodes = []
        for i, action in enumerate(query_plan.actions):
            # Select best agent for action (with load balancing if available)
            agent_name = await self._select_optimal_agent(action)
            
            node = ExecutionNode(
                action=action,
                agent_name=agent_name,
                parallel_group=action.parallel_group
            )
            nodes.append(node)
        
        return ExecutionPlan(
            query_plan=query_plan,
            nodes=nodes,
            parallel_groups={}
        )
    
    async def _select_optimal_agent(self, action: ReActAction) -> str:
        """Select optimal agent for action with load balancing"""
        
        # Get agents that can handle this action
        capable_agents = []
        for agent in self.registry.get_all():
            if self._can_handle_action(agent, action):
                capable_agents.append(agent)
        
        if not capable_agents:
            raise RuntimeError(f"No agents capable of handling action: {action.action}")
        
        # Use load balancer if available
        agent_type = capable_agents[0].agent_type.value
        if (self.coordinator and 
            agent_type in self.coordinator.load_balancers and 
            len(capable_agents) > 1):
            
            selected_agent = self.coordinator.load_balancers[agent_type].select_agent()
            return selected_agent.name if selected_agent else capable_agents[0].name
        
        # Default to first capable agent
        return capable_agents[0].name
    
    async def _execute_plan_with_coordination(self, plan: ExecutionPlan) -> AgentResult:
        """Execute plan using coordination features"""
        
        results = []
        
        # Execute in parallel groups
        for group_id in sorted(plan.parallel_groups.keys()):
            group_nodes = plan.parallel_groups[group_id]
            
            # Execute group in parallel
            group_tasks = []
            for node in group_nodes:
                agent = self.registry.get(node.agent_name)
                if agent:
                    task = self._execute_node_with_coordination(node, agent)
                    group_tasks.append(task)
            
            # Wait for group completion
            if group_tasks:
                group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                
                # Process results and handle failures
                for i, result in enumerate(group_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Node execution failed: {str(result)}")
                        # Consider adaptive replanning here
                    else:
                        results.append(result)
        
        # Aggregate results
        return await self._aggregate_coordinated_results(results)
    
    async def _execute_node_with_coordination(
        self,
        node: ExecutionNode,
        agent: BaseAgent
    ) -> AgentResult:
        """Execute individual node with coordination features"""
        
        node.start_time = datetime.now()
        node.status = AgentStatus.EXECUTING
        
        try:
            # Create query from action
            query = AgentQuery(
                text=node.action.observation,
                intent=node.action.action,
                context=node.action.context or {}
            )
            
            # Execute with coordination if available
            if self.coordinator:
                result = await self.coordinator.execute_with_coordination(
                    agent, query,
                    use_circuit_breaker=True,
                    use_retry=True
                )
            else:
                result = await agent.execute_with_monitoring(query)
            
            node.result = result
            node.status = AgentStatus.COMPLETED if result.success else AgentStatus.FAILED
            
            return result
            
        except Exception as e:
            node.status = AgentStatus.FAILED
            self.logger.error(f"Node execution failed: {str(e)}")
            
            return AgentResult(
                success=False,
                data=None,
                error=str(e),
                agent_type=agent.agent_type
            )
        
        finally:
            node.end_time = datetime.now()
    
    def _optimize_plan_based_on_history(self, plan: QueryPlan) -> QueryPlan:
        """Optimize plan based on performance history"""
        
        if not self.performance_history:
            return plan
        
        # Calculate recent success rate
        recent_history = list(self.performance_history)[-20:]  # Last 20 executions
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        
        # If success rate is low, add redundancy
        if success_rate < self.success_threshold:
            self.logger.warning(f"Low success rate ({success_rate:.2%}), adding redundancy to plan")
            # Add parallel execution for critical actions
            for action in plan.actions:
                if action.confidence < 0.8:  # Low confidence actions
                    action.parallel_group = max(a.parallel_group for a in plan.actions) + 1
        
        return plan
    
    async def _aggregate_coordinated_results(self, results: List[AgentResult]) -> AgentResult:
        """Aggregate results with enhanced error handling"""
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            # All failed - return aggregated error
            errors = [r.error for r in failed_results if r.error]
            return AgentResult(
                success=False,
                data=None,
                error=f"All agents failed: {'; '.join(errors)}",
                agent_type=self.agent_type
            )
        
        # Aggregate successful results
        aggregated_data = []
        total_confidence = 0.0
        
        for result in successful_results:
            if result.data:
                aggregated_data.append(result.data)
            total_confidence += result.confidence_score
        
        avg_confidence = total_confidence / len(successful_results) if successful_results else 0.0
        
        return AgentResult(
            success=True,
            data=aggregated_data,
            metadata={
                'successful_agents': len(successful_results),
                'failed_agents': len(failed_results),
                'average_confidence': avg_confidence,
                'coordination_used': self.coordinator is not None
            },
            agent_type=self.agent_type,
            confidence_score=avg_confidence
        )
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination and performance statistics"""
        
        stats = {
            'coordination_enabled': self.coordinator is not None,
            'performance_monitoring': self.performance_monitor is not None,
            'adaptation_enabled': self.adaptation_enabled
        }
        
        if self.coordinator:
            stats['circuit_breakers'] = len(self.coordinator.circuit_breakers)
            stats['load_balancers'] = len(self.coordinator.load_balancers)
        
        if self.performance_history:
            recent = list(self.performance_history)[-10:]
            stats['recent_success_rate'] = sum(1 for h in recent if h['success']) / len(recent)
            stats['average_execution_time'] = sum(h['execution_time'] for h in recent) / len(recent)
        
        return stats


# REQ-ORCH-018: Export public interface
__all__ = [
    "OrchestratorAgent",
    "EnhancedOrchestratorAgent",
    "ExecutionNode",
    "ExecutionPlan"
]
