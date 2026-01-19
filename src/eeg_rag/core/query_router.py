#!/usr/bin/env python3
"""
Query Routing System

Routes queries to appropriate agents based on question type.
Reduces latency and API costs by selecting the most suitable agent.
"""

import re
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle"""
    DEFINITIONAL = "definitional"  # What is X?
    RECENT_LITERATURE = "recent_literature"  # Recent findings about X
    COMPARATIVE = "comparative"  # Compare X vs Y
    METHODOLOGICAL = "methodological"  # How to do X?
    CLINICAL = "clinical"  # Clinical applications
    STATISTICAL = "statistical"  # Statistical analysis questions
    UNKNOWN = "unknown"  # Fallback


@dataclass
class RoutingResult:
    """Result of query routing"""
    query_type: QueryType
    confidence: float
    recommended_agent: str
    reasoning: str
    keywords: List[str]
    complexity: str  # 'simple', 'medium', 'complex'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_type': self.query_type.value,
            'confidence': self.confidence,
            'recommended_agent': self.recommended_agent,
            'reasoning': self.reasoning,
            'keywords': self.keywords,
            'complexity': self.complexity
        }


class QueryRouter:
    """Routes queries to appropriate agents based on content analysis"""
    
    def __init__(self):
        # Define patterns for different query types
        self.query_patterns = {
            QueryType.DEFINITIONAL: {
                'patterns': [
                    r'\bwhat\s+is\b',
                    r'\bdefine\b',
                    r'\bdefiniton\b',
                    r'\bmean(ing)?\s+(of|by)\b',
                    r'\bexplain\b',
                    r'\bdescribe\b'
                ],
                'keywords': ['definition', 'meaning', 'concept', 'term']
            },
            QueryType.RECENT_LITERATURE: {
                'patterns': [
                    r'\b(recent|latest|new|current|modern)\s+(research|studies|findings|literature)\b',
                    r'\b(2020|2021|2022|2023|2024)\b',
                    r'\bstate\s+of\s+the\s+art\b',
                    r'\bcutting\s+edge\b',
                    r'\brecent\s+advances\b'
                ],
                'keywords': ['recent', 'latest', 'new', 'current', 'advances', 'breakthrough']
            },
            QueryType.COMPARATIVE: {
                'patterns': [
                    r'\b(compare|comparison|versus|vs\.?)\b',
                    r'\b(difference|differ)\b',
                    r'\b(better|worse|superior|inferior)\b',
                    r'\b(advantage|disadvantage)\b',
                    r'\bwhich\s+(is\s+)?(better|more|less)\b'
                ],
                'keywords': ['compare', 'versus', 'difference', 'better', 'advantage']
            },
            QueryType.METHODOLOGICAL: {
                'patterns': [
                    r'\bhow\s+to\b',
                    r'\bmethod(s|ology)?\b',
                    r'\bprocedure\b',
                    r'\bprotocol\b',
                    r'\btechnique\b',
                    r'\bapproach\b',
                    r'\bstep(s)?\b',
                    r'\bprocess\b'
                ],
                'keywords': ['method', 'procedure', 'protocol', 'technique', 'how', 'steps']
            },
            QueryType.CLINICAL: {
                'patterns': [
                    r'\b(patient|clinical|treatment|therapy|diagnosis)\b',
                    r'\b(symptom|disease|disorder|condition)\b',
                    r'\b(medical|healthcare)\b',
                    r'\b(efficacy|effectiveness)\b',
                    r'\bside\s+effect\b'
                ],
                'keywords': ['patient', 'clinical', 'treatment', 'diagnosis', 'medical']
            },
            QueryType.STATISTICAL: {
                'patterns': [
                    r'\b(statistical|statistics|analysis|analyze)\b',
                    r'\b(correlation|regression|anova|t-test)\b',
                    r'\b(p-value|significance|significant)\b',
                    r'\b(mean|median|standard\s+deviation)\b',
                    r'\b(sample\s+size|power\s+analysis)\b'
                ],
                'keywords': ['statistical', 'analysis', 'correlation', 'significance', 'p-value']
            }
        }
        
        # Agent routing table
        self.agent_routing = {
            QueryType.DEFINITIONAL: "local_agent",  # Fast for basic definitions
            QueryType.RECENT_LITERATURE: "web_agent",  # Need web search for recent papers
            QueryType.COMPARATIVE: "graph_agent",  # Good for relationships
            QueryType.METHODOLOGICAL: "local_agent",  # Protocols often in corpus
            QueryType.CLINICAL: "local_agent",  # Medical knowledge in corpus
            QueryType.STATISTICAL: "local_agent",  # Statistical methods in papers
            QueryType.UNKNOWN: "orchestrator"  # Let orchestrator decide
        }
        
        # EEG-specific keywords that boost confidence
        self.eeg_keywords = {
            'eeg', 'electroencephalography', 'electroencephalogram',
            'brain', 'neural', 'neuroscience', 'cognitive',
            'seizure', 'epilepsy', 'sleep', 'consciousness',
            'erp', 'event-related', 'potential', 'p300', 'n400',
            'alpha', 'beta', 'gamma', 'delta', 'theta', 'mu',
            'bci', 'brain-computer', 'interface',
            'electrode', 'montage', '10-20', 'artifact',
            'frequency', 'amplitude', 'power', 'spectrum',
            'epileptiform', 'spike', 'sharp', 'wave'
        }
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> RoutingResult:
        """Route query to appropriate agent"""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        matched_keywords = []
        
        for query_type, config in self.query_patterns.items():
            score = self._calculate_type_score(query_lower, config)
            type_scores[query_type] = score
            
            # Collect matched keywords
            for keyword in config['keywords']:
                if keyword in query_lower:
                    matched_keywords.append(keyword)
        
        # Find best match
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        # Boost confidence if EEG-related
        eeg_boost = self._calculate_eeg_relevance(query_lower)
        confidence = min(confidence + eeg_boost * 0.2, 1.0)
        
        # Determine complexity
        complexity = self._assess_complexity(query, context)
        
        # Select agent
        recommended_agent = self.agent_routing.get(best_type, "orchestrator")
        
        # Adjust agent based on complexity and context
        if complexity == "complex" and best_type != QueryType.RECENT_LITERATURE:
            recommended_agent = "orchestrator"  # Use orchestrator for complex queries
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, best_type, confidence, complexity)
        
        # Extract key terms
        keywords = self._extract_key_terms(query) + matched_keywords
        keywords = list(set(keywords))  # Remove duplicates
        
        return RoutingResult(
            query_type=best_type,
            confidence=confidence,
            recommended_agent=recommended_agent,
            reasoning=reasoning,
            keywords=keywords,
            complexity=complexity
        )
    
    def _calculate_type_score(self, query: str, config: Dict[str, Any]) -> float:
        """Calculate score for a specific query type"""
        pattern_matches = 0
        total_patterns = len(config['patterns'])
        
        for pattern in config['patterns']:
            if re.search(pattern, query, re.IGNORECASE):
                pattern_matches += 1
        
        # Keyword matches
        keyword_matches = 0
        total_keywords = len(config['keywords'])
        
        for keyword in config['keywords']:
            if keyword in query:
                keyword_matches += 1
        
        # Combine pattern and keyword scores
        pattern_score = pattern_matches / max(total_patterns, 1)
        keyword_score = keyword_matches / max(total_keywords, 1)
        
        return (pattern_score * 0.7) + (keyword_score * 0.3)
    
    def _calculate_eeg_relevance(self, query: str) -> float:
        """Calculate how EEG-relevant the query is"""
        eeg_matches = sum(1 for keyword in self.eeg_keywords if keyword in query)
        return min(eeg_matches / 10.0, 1.0)  # Normalize to 0-1
    
    def _assess_complexity(self, query: str, context: Dict[str, Any] = None) -> str:
        """Assess query complexity"""
        # Simple heuristics for complexity
        word_count = len(query.split())
        
        # Check for complex indicators
        complex_indicators = [
            'multiple', 'several', 'various', 'compare', 'contrast',
            'analyze', 'evaluate', 'assess', 'relationship', 'correlation',
            'interaction', 'mechanism', 'pathway'
        ]
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        # Determine complexity
        if word_count > 20 or complex_count >= 2:
            return "complex"
        elif word_count > 10 or complex_count >= 1:
            return "medium"
        else:
            return "simple"
    
    def _generate_reasoning(self, query: str, query_type: QueryType, 
                           confidence: float, complexity: str) -> str:
        """Generate human-readable reasoning for the routing decision"""
        reasoning_templates = {
            QueryType.DEFINITIONAL: "Query asks for definition or explanation",
            QueryType.RECENT_LITERATURE: "Query seeks recent research or current findings",
            QueryType.COMPARATIVE: "Query involves comparison between concepts",
            QueryType.METHODOLOGICAL: "Query asks about methods or procedures",
            QueryType.CLINICAL: "Query relates to clinical applications or patient care",
            QueryType.STATISTICAL: "Query involves statistical analysis or metrics",
            QueryType.UNKNOWN: "Query type unclear from content analysis"
        }
        
        base_reasoning = reasoning_templates.get(query_type, "Unknown query type")
        
        # Add confidence and complexity info
        confidence_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        
        return f"{base_reasoning} (confidence: {confidence_desc}, complexity: {complexity})"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Simple keyword extraction (could be improved with NLP)
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and keep meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Keep only top terms (by length as a simple heuristic)
        key_terms.sort(key=len, reverse=True)
        
        return key_terms[:10]  # Return top 10 key terms
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and configuration"""
        return {
            'supported_query_types': [qt.value for qt in QueryType],
            'agent_mapping': {qt.value: agent for qt, agent in self.agent_routing.items()},
            'eeg_keywords_count': len(self.eeg_keywords)
        }
    
    def add_custom_pattern(self, query_type: QueryType, pattern: str, keywords: List[str] = None):
        """Add custom pattern for query type detection"""
        if query_type in self.query_patterns:
            self.query_patterns[query_type]['patterns'].append(pattern)
            if keywords:
                self.query_patterns[query_type]['keywords'].extend(keywords)
        else:
            logger.warning(f"Unknown query type: {query_type}")


# Convenience function for quick routing
def route_query(query: str, context: Dict[str, Any] = None) -> RoutingResult:
    """Quick function to route a query"""
    router = QueryRouter()
    return router.route_query(query, context)
