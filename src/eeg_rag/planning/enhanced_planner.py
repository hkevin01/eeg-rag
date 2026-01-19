"""
Enhanced Query Planning with Advanced Reasoning and Adaptive Optimization

This module extends the base query planner with sophisticated capabilities:
- Advanced Chain-of-Thought reasoning with confidence tracking
- Adaptive planning based on execution history
- Multi-modal query understanding
- Context-aware query decomposition
- Pattern caching for performance optimization
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque, defaultdict
from datetime import datetime

from .query_planner import (
    QueryPlanner, QueryPlan, QueryIntent, QueryComplexity,
    SubQuery, CoTStep, ReActAction
)


class EnhancedQueryPlanner(QueryPlanner):
    """
    Enhanced query planner with advanced reasoning and adaptive optimization
    
    REQ-PLAN-025: Advanced CoT reasoning with confidence tracking
    REQ-PLAN-026: Adaptive planning based on execution history
    REQ-PLAN-027: Multi-modal query understanding
    REQ-PLAN-028: Context-aware query decomposition
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enhanced reasoning capabilities
        self.confidence_threshold = 0.8
        self.max_reasoning_depth = 5
        self.adaptive_optimization = True
        
        # Execution history for adaptive planning
        self.execution_history = deque(maxlen=100)
        self.pattern_cache = {}  # Cache successful planning patterns
        
        # Advanced intent classification patterns
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'what is|what are|define|definition of|explain',
                r'how many|how much|which|when|where|who'
            ],
            QueryIntent.COMPARISON: [
                r'compare|versus|vs|difference between|better than',
                r'advantages|disadvantages|pros and cons'
            ],
            QueryIntent.ANALYSIS: [
                r'analyze|analysis|trend|pattern|relationship',
                r'correlation|impact|effect|influence'
            ],
            QueryIntent.PROCEDURE: [
                r'how to|step by step|procedure|process|method',
                r'instructions|guide|tutorial'
            ],
            QueryIntent.CODE_GEN: [
                r'code|script|program|implement|generate',
                r'function|algorithm|example'
            ],
            QueryIntent.DATASET: [
                r'dataset|data|corpus|collection|database',
                r'download|access|available data'
            ],
            QueryIntent.REVIEW: [
                r'review|survey|overview|state of art',
                r'literature|research|studies|papers'
            ]
        }
        
        self.logger.info("Enhanced query planner initialized with advanced reasoning")
    
    async def plan_query_enhanced(
        self,
        query_text: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Enhanced query planning with adaptive optimization"""
        
        context = context or {}
        user_preferences = user_preferences or {}
        
        try:
            # Enhanced intent classification
            intent = self._classify_intent_enhanced(query_text, context)
            
            # Advanced complexity assessment
            complexity = self._assess_complexity_enhanced(query_text, context)
            
            # Context-aware decomposition
            sub_queries = await self._decompose_query_enhanced(
                query_text, intent, complexity, context
            )
            
            # Advanced CoT reasoning
            cot_steps = await self._enhanced_chain_of_thought(
                query_text, sub_queries, context
            )
            
            # Adaptive action planning
            actions = await self._plan_actions_adaptive(
                sub_queries, cot_steps, user_preferences
            )
            
            # Optimize plan based on history
            if self.adaptive_optimization:
                actions = self._optimize_actions_from_history(actions, query_text)
            
            plan = QueryPlan(
                original_query=query_text,
                intent=intent,
                complexity=complexity,
                sub_queries=sub_queries,
                cot_reasoning=cot_steps,
                actions=actions,
                estimated_latency=self._estimate_execution_time_enhanced(actions),
                metadata={
                    'enhanced_planning': True,
                    'adaptive_optimization': self.adaptive_optimization,
                    'pattern_cache_hits': len([p for p in self.pattern_cache.keys() if p in query_text]),
                    'user_preferences': user_preferences
                }
            )
            
            self.logger.info(
                f"Enhanced planning complete: {intent.value} query, "
                f"{complexity.value} complexity, {len(actions)} actions"
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Enhanced planning failed: {str(e)}")
            # Fallback to standard planning
            return await self.plan_query(query_text, context)
    
    async def plan_query(self, query_text: str, context: Dict[str, Any] = None) -> QueryPlan:
        """Standard plan_query method for compatibility"""
        if context is None:
            context = {}
        
        # Use enhanced planning if available
        if hasattr(self, 'adaptive_optimization') and self.adaptive_optimization:
            return await self.plan_query_enhanced(query_text, context)
        else:
            # Fallback to basic planning logic
            intent = self._classify_intent_enhanced(query_text, context)
            complexity = self._assess_complexity_enhanced(query_text, context)
            
            # Create simple plan
            sub_queries = [SubQuery(
                text=query_text,
                intent=intent,
                priority=QueryPriority.HIGH,
                dependencies=set()
            )]
            
            actions = [ReActAction(
                action_type="search_local",
                reasoning=f"Search for information about: {query_text}",
                parameters={'query': query_text},
                expected_outcome="Relevant information"
            )]
            
            return QueryPlan(
                original_query=query_text,
                intent=intent,
                complexity=complexity,
                sub_queries=sub_queries,
                actions=actions,
                estimated_latency=5.0
            )
    
    def _classify_intent_enhanced(
        self,
        query_text: str,
        context: Dict[str, Any]
    ) -> QueryIntent:
        """Enhanced intent classification with context awareness"""
        
        query_lower = query_text.lower()
        
        # Check for context clues
        context_intent = self._infer_intent_from_context(context)
        if context_intent != QueryIntent.UNKNOWN:
            return context_intent
        
        # Pattern matching with confidence scoring
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            
            # Normalize by number of patterns
            intent_scores[intent] = score / len(patterns) if patterns else 0
        
        # Multi-part query detection
        if ('and' in query_lower or 'also' in query_lower or 
            query_text.count('?') > 1):
            return QueryIntent.MULTI_PART
        
        # Return highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0.3:  # Confidence threshold
                return best_intent[0]
        
        return QueryIntent.UNKNOWN
    
    def _infer_intent_from_context(self, context: Dict[str, Any]) -> QueryIntent:
        """Infer intent from conversation context"""
        
        # Check for previous queries in context
        if 'previous_queries' in context:
            prev_queries = context['previous_queries']
            if prev_queries:
                last_query = prev_queries[-1]
                if 'compare' in last_query.lower():
                    return QueryIntent.COMPARISON
        
        # Check for user session context
        if 'session_context' in context:
            session = context['session_context']
            if session.get('research_mode'):
                return QueryIntent.REVIEW
            if session.get('coding_mode'):
                return QueryIntent.CODE_GEN
        
        return QueryIntent.UNKNOWN
    
    def _assess_complexity_enhanced(
        self,
        query_text: str,
        context: Dict[str, Any]
    ) -> QueryComplexity:
        """Enhanced complexity assessment with context awareness"""
        
        # Base complexity scoring
        complexity_score = 0
        
        # Text-based complexity indicators
        word_count = len(query_text.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Multiple questions
        question_count = query_text.count('?')
        if question_count > 1:
            complexity_score += 1
        
        # Complex keywords
        complex_keywords = [
            'compare', 'analyze', 'relationship', 'correlation',
            'methodology', 'systematic', 'comprehensive', 'detailed'
        ]
        keyword_matches = sum(1 for kw in complex_keywords if kw in query_text.lower())
        complexity_score += keyword_matches
        
        # Context-based complexity
        if context.get('multi_modal'):
            complexity_score += 1
        if context.get('real_time_required'):
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 5:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 3:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    async def _decompose_query_enhanced(
        self,
        query_text: str,
        intent: QueryIntent,
        complexity: QueryComplexity,
        context: Dict[str, Any]
    ) -> List[SubQuery]:
        """Enhanced query decomposition with context awareness"""
        
        # Check pattern cache first
        cache_key = f"{intent.value}_{complexity.value}_{hash(query_text) % 1000}"
        if cache_key in self.pattern_cache:
            pattern = self.pattern_cache[cache_key]
            return self._apply_cached_pattern(query_text, pattern)
        
        sub_queries = []
        
        if complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            # Single sub-query for simple/moderate queries
            sub_queries.append(SubQuery(
                text=query_text,
                intent=intent,
                priority=1,
                required_agents=['local_data', 'web_search'] if complexity == QueryComplexity.MODERATE else ['local_data']
            ))
        
        elif intent == QueryIntent.COMPARISON:
            # Decompose comparison queries
            sub_queries.extend(self._decompose_comparison_query(query_text))
        
        elif intent == QueryIntent.ANALYSIS:
            # Decompose analysis queries
            sub_queries.extend(self._decompose_analysis_query(query_text))
        
        elif intent == QueryIntent.MULTI_PART:
            # Decompose multi-part queries
            sub_queries.extend(self._decompose_multi_part_query(query_text))
        
        else:
            # Complex single query - break into steps
            sub_queries.extend([
                SubQuery(
                    text=f"Background information on: {query_text}",
                    intent=QueryIntent.FACTUAL,
                    priority=1,
                    required_agents=['local_data', 'web_search']
                ),
                SubQuery(
                    text=query_text,
                    intent=intent,
                    priority=2,
                    dependencies=[0],
                    required_agents=['local_data', 'web_search', 'knowledge_graph']
                )
            ])
        
        # Cache successful pattern
        if sub_queries:
            pattern = {
                'structure': [sq.text for sq in sub_queries],
                'agents': [sq.required_agents for sq in sub_queries],
                'dependencies': [sq.dependencies for sq in sub_queries]
            }
            self.pattern_cache[cache_key] = pattern
        
        return sub_queries
    
    def _apply_cached_pattern(self, query_text: str, pattern: Dict[str, Any]) -> List[SubQuery]:
        """Apply cached pattern to new query"""
        sub_queries = []
        
        for i, (structure, agents, dependencies) in enumerate(zip(
            pattern['structure'], pattern['agents'], pattern['dependencies']
        )):
            # Adapt structure to current query
            adapted_text = structure.replace(structure.split(': ')[-1] if ': ' in structure else structure, query_text)
            
            sub_queries.append(SubQuery(
                text=adapted_text,
                intent=QueryIntent.FACTUAL,  # Default for cached patterns
                priority=i + 1,
                dependencies=dependencies,
                required_agents=agents
            ))
        
        return sub_queries
    
    async def _enhanced_chain_of_thought(
        self,
        query_text: str,
        sub_queries: List[SubQuery],
        context: Dict[str, Any]
    ) -> List[CoTStep]:
        """Enhanced Chain-of-Thought reasoning with confidence tracking"""
        
        steps = []
        
        # Step 1: Understanding
        steps.append(CoTStep(
            step_number=1,
            thought="Query Understanding",
            reasoning=f"The user is asking about: {query_text}. This requires {len(sub_queries)} sub-queries to address comprehensively.",
            conclusion=f"I need to gather information from multiple sources to provide a complete answer.",
            confidence=0.9
        ))
        
        # Step 2: Information Sources
        required_agents = set()
        for sq in sub_queries:
            required_agents.update(sq.required_agents)
        
        steps.append(CoTStep(
            step_number=2,
            thought="Information Source Planning",
            reasoning=f"Based on the query complexity, I need to consult: {', '.join(required_agents)}",
            conclusion="Multiple information sources will provide comprehensive coverage",
            confidence=0.85
        ))
        
        # Step 3: Execution Strategy
        parallel_possible = any(not sq.dependencies for sq in sub_queries[1:])
        
        steps.append(CoTStep(
            step_number=3,
            thought="Execution Strategy",
            reasoning=f"{'Parallel' if parallel_possible else 'Sequential'} execution will be most efficient based on dependencies.",
            conclusion=f"Execute {len(sub_queries)} queries {'in parallel groups' if parallel_possible else 'sequentially'}",
            confidence=0.8
        ))
        
        # Step 4: Quality Control
        steps.append(CoTStep(
            step_number=4,
            thought="Quality Assurance",
            reasoning="Results will be cross-validated across sources and checked for consistency.",
            conclusion="Multi-source validation ensures answer quality and reliability",
            confidence=0.75
        ))
        
        return steps
    
    async def _plan_actions_adaptive(
        self,
        sub_queries: List[SubQuery],
        cot_steps: List[CoTStep],
        user_preferences: Dict[str, Any]
    ) -> List[ReActAction]:
        """Adaptive action planning with user preferences"""
        
        actions = []
        parallel_group = 0
        
        for i, sub_query in enumerate(sub_queries):
            # Adjust parallel grouping based on dependencies
            if sub_query.dependencies:
                parallel_group = max(parallel_group, max(sub_query.dependencies) + 1)
            
            # Select agents based on preferences and requirements
            preferred_agents = user_preferences.get('preferred_agents', sub_query.required_agents)
            selected_agents = self._select_agents_for_query(sub_query, preferred_agents)
            
            for agent_type in selected_agents:
                action = ReActAction(
                    action_type=f"search_{agent_type}",
                    reasoning=f"I need to search {agent_type} for: {sub_query.text}",
                    parameters={
                        'query': sub_query.text,
                        'sub_query_index': i,
                        'priority': sub_query.priority,
                        'intent': sub_query.intent.value
                    },
                    expected_outcome=f"Relevant information about {sub_query.text}",
                    parallel_group=parallel_group if not sub_query.dependencies else parallel_group
                )
                actions.append(action)
        
        return actions
    
    def _optimize_actions_from_history(self, actions: List[ReActAction], query_text: str) -> List[ReActAction]:
        """Optimize actions based on execution history"""
        
        if not self.execution_history:
            return actions
        
        # Analyze successful patterns from history
        successful_executions = [h for h in self.execution_history if h.get('success', False)]
        
        if not successful_executions:
            return actions
        
        # Find similar queries in history
        similar_queries = []
        query_words = set(query_text.lower().split())
        
        for execution in successful_executions:
            if 'query_text' in execution:
                hist_words = set(execution['query_text'].lower().split())
                similarity = len(query_words.intersection(hist_words)) / len(query_words.union(hist_words))
                if similarity > 0.3:  # 30% similarity threshold
                    similar_queries.append(execution)
        
        if not similar_queries:
            return actions
        
        # Analyze successful action patterns
        successful_agents = defaultdict(int)
        for query in similar_queries:
            for agent in query.get('successful_agents', []):
                successful_agents[agent] += 1
        
        # Boost confidence for successful agents
        for action in actions:
            agent_type = action.action.replace('search_', '')
            if agent_type in successful_agents:
                boost = min(0.2, successful_agents[agent_type] / len(similar_queries))
                action.confidence = min(1.0, action.confidence + boost)
        
        return actions
    
    def record_execution_result(
        self,
        query_text: str,
        plan: QueryPlan,
        success: bool,
        execution_time: float,
        successful_agents: List[str]
    ) -> None:
        """Record execution result for adaptive learning"""
        
        execution_record = {
            'query_text': query_text,
            'intent': plan.intent.value,
            'complexity': plan.complexity.value,
            'success': success,
            'execution_time': execution_time,
            'successful_agents': successful_agents,
            'timestamp': datetime.now().isoformat(),
            'action_count': len(plan.actions)
        }
        
        self.execution_history.append(execution_record)
        self.logger.debug(f"Recorded execution result: success={success}, agents={len(successful_agents)}")
    
    def _select_agents_for_query(self, sub_query: SubQuery, preferred_agents: List[str]) -> List[str]:
        """Select optimal agents for sub-query"""
        
        # Filter preferred agents by what's available for this query type
        available_agents = set(sub_query.required_agents)
        selected = [agent for agent in preferred_agents if agent in available_agents]
        
        # Ensure at least one agent is selected
        if not selected and sub_query.required_agents:
            selected = [sub_query.required_agents[0]]
        
        return selected or ['local_data']  # Fallback
    
    def _estimate_execution_time_enhanced(self, actions: List[ReActAction]) -> float:
        """Enhanced execution time estimation with historical data"""
        
        if not self.execution_history:
            return self._estimate_execution_time(actions)
        
        # Calculate average execution time per action from history
        total_time = 0
        total_actions = 0
        
        for record in self.execution_history[-20:]:  # Last 20 executions
            if record.get('execution_time') and record.get('action_count'):
                total_time += record['execution_time']
                total_actions += record['action_count']
        
        if total_actions > 0:
            avg_time_per_action = total_time / total_actions
            return len(actions) * avg_time_per_action * 1.1  # 10% buffer
        
        return self._estimate_execution_time(actions)
    
    def _decompose_comparison_query(self, query_text: str) -> List[SubQuery]:
        """Decompose comparison queries"""
        return [
            SubQuery(
                text=f"Information about first item in: {query_text}",
                intent=QueryIntent.FACTUAL,
                priority=1,
                required_agents=['local_data', 'web_search']
            ),
            SubQuery(
                text=f"Information about second item in: {query_text}",
                intent=QueryIntent.FACTUAL,
                priority=1,
                required_agents=['local_data', 'web_search']
            ),
            SubQuery(
                text=query_text,
                intent=QueryIntent.COMPARISON,
                priority=2,
                dependencies=[0, 1],
                required_agents=['knowledge_graph']
            )
        ]
    
    def _decompose_analysis_query(self, query_text: str) -> List[SubQuery]:
        """Decompose analysis queries"""
        return [
            SubQuery(
                text=f"Data collection for: {query_text}",
                intent=QueryIntent.DATASET,
                priority=1,
                required_agents=['local_data', 'web_search']
            ),
            SubQuery(
                text=query_text,
                intent=QueryIntent.ANALYSIS,
                priority=2,
                dependencies=[0],
                required_agents=['knowledge_graph']
            )
        ]
    
    def _decompose_multi_part_query(self, query_text: str) -> List[SubQuery]:
        """Decompose multi-part queries"""
        # Simple split on question marks or 'and' keywords
        parts = re.split(r'\?|and also|also', query_text)
        parts = [part.strip() for part in parts if part.strip()]
        
        sub_queries = []
        for i, part in enumerate(parts):
            sub_queries.append(SubQuery(
                text=part.strip(),
                intent=QueryIntent.FACTUAL,
                priority=1,
                required_agents=['local_data', 'web_search']
            ))
        
        return sub_queries


__all__ = ["EnhancedQueryPlanner"]
