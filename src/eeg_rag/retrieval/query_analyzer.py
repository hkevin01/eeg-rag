"""
Query Analyzer for Adaptive Reranking

Analyzes queries to determine:
1. Complexity level (simple/medium/complex)
2. Whether reranking would be beneficial
3. Query characteristics (length, technical terms, ambiguity)

Use Cases:
- Enable reranking only for complex queries (save latency on simple ones)
- Adjust retrieval parameters based on query type
- Route queries to appropriate processing pipelines
"""

import re
import logging
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"       # Short, specific queries
    MEDIUM = "medium"       # Moderate complexity
    COMPLEX = "complex"     # Long, multi-concept queries


@dataclass
class QueryAnalysis:
    """Analysis results for a query"""
    query: str
    complexity: QueryComplexity
    should_rerank: bool
    confidence: float
    
    # Query characteristics
    word_count: int
    technical_terms: List[str]
    has_boolean_operators: bool
    has_negation: bool
    is_question: bool
    ambiguity_score: float
    
    # Reasoning
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "complexity": self.complexity.value,
            "should_rerank": self.should_rerank,
            "confidence": self.confidence,
            "characteristics": {
                "word_count": self.word_count,
                "technical_terms": self.technical_terms,
                "has_boolean_operators": self.has_boolean_operators,
                "has_negation": self.has_negation,
                "is_question": self.is_question,
                "ambiguity_score": self.ambiguity_score
            },
            "reasoning": self.reasoning
        }


class QueryAnalyzer:
    """
    Analyze queries to determine optimal retrieval strategy
    
    Example:
        >>> analyzer = QueryAnalyzer()
        >>> analysis = analyzer.analyze("deep learning for seizure detection")
        >>> if analysis.should_rerank:
        ...     # Use reranking
        ...     results = hybrid_retriever.search(query, use_reranking=True)
        ... else:
        ...     # Skip reranking for simple query
        ...     results = hybrid_retriever.search(query, use_reranking=False)
    """
    
    # EEG-specific technical terms
    EEG_TECHNICAL_TERMS = {
        # Frequency bands
        'delta', 'theta', 'alpha', 'beta', 'gamma',
        # Methods
        'fft', 'wavelet', 'spectrogram', 'coherence', 'psd', 'eeg',
        # Clinical
        'epilepsy', 'seizure', 'ictal', 'interictal', 'preictal',
        'sleep', 'rem', 'nrem', 'arousal',
        # Components
        'p300', 'n400', 'p600', 'mmn', 'erp', 'ern',
        # Techniques
        'ica', 'pca', 'csp', 'bci', 'ssvep', 'auditory',
        # Machine learning
        'cnn', 'rnn', 'lstm', 'transformer', 'neural', 'network',
        'classification', 'detection', 'prediction'
    }
    
    # Boolean operators
    BOOLEAN_OPERATORS = {'and', 'or', 'not', 'but'}
    
    # Negation words
    NEGATION_WORDS = {
        'no', 'not', 'never', 'without', 'exclude', 'except',
        'none', 'neither', 'nor'
    }
    
    # Question words
    QUESTION_WORDS = {
        'what', 'when', 'where', 'who', 'why', 'how',
        'which', 'whose', 'whom'
    }
    
    def __init__(
        self,
        rerank_threshold_words: int = 5,
        rerank_threshold_technical: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize query analyzer
        
        Args:
            rerank_threshold_words: Min word count to consider reranking
            rerank_threshold_technical: Min technical terms to consider reranking
            logger: Logger instance
        """
        self.rerank_threshold_words = rerank_threshold_words
        self.rerank_threshold_technical = rerank_threshold_technical
        self.logger = logger or logging.getLogger("eeg_rag.query_analyzer")
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and determine retrieval strategy
        
        Args:
            query: Search query
            
        Returns:
            QueryAnalysis with recommendations
        """
        # Normalize query
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Extract characteristics
        word_count = len(words)
        technical_terms = self._extract_technical_terms(words)
        has_boolean = self._has_boolean_operators(words)
        has_negation = self._has_negation(words)
        is_question = self._is_question(query_lower)
        ambiguity_score = self._compute_ambiguity(query_lower, words)
        
        # Determine complexity
        complexity = self._determine_complexity(
            word_count, technical_terms, has_boolean,
            has_negation, is_question, ambiguity_score
        )
        
        # Decide if reranking should be used
        should_rerank, confidence, reasoning = self._should_rerank(
            complexity, word_count, len(technical_terms),
            has_boolean, ambiguity_score
        )
        
        return QueryAnalysis(
            query=query,
            complexity=complexity,
            should_rerank=should_rerank,
            confidence=confidence,
            word_count=word_count,
            technical_terms=technical_terms,
            has_boolean_operators=has_boolean,
            has_negation=has_negation,
            is_question=is_question,
            ambiguity_score=ambiguity_score,
            reasoning=reasoning
        )
    
    def _extract_technical_terms(self, words: List[str]) -> List[str]:
        """Extract technical terms from query"""
        return [w for w in words if w in self.EEG_TECHNICAL_TERMS]
    
    def _has_boolean_operators(self, words: List[str]) -> bool:
        """Check if query contains boolean operators"""
        return any(w in self.BOOLEAN_OPERATORS for w in words)
    
    def _has_negation(self, words: List[str]) -> bool:
        """Check if query contains negation"""
        return any(w in self.NEGATION_WORDS for w in words)
    
    def _is_question(self, query: str) -> bool:
        """Check if query is a question"""
        query_words = query.split()
        if not query_words:
            return False
        
        # Check for question mark
        if '?' in query:
            return True
        
        # Check for question words
        return query_words[0] in self.QUESTION_WORDS
    
    def _compute_ambiguity(self, query: str, words: List[str]) -> float:
        """
        Compute query ambiguity score (0-1)
        
        Higher score = more ambiguous (benefits from reranking)
        """
        score = 0.0
        
        # Short queries are often ambiguous
        if len(words) <= 2:
            score += 0.3
        
        # Generic terms indicate ambiguity
        generic_terms = {'method', 'analysis', 'study', 'research', 'approach'}
        if any(term in words for term in generic_terms):
            score += 0.2
        
        # Multiple concepts increase ambiguity
        if len(words) >= 6:
            score += 0.2
        
        # Vague modifiers
        vague_modifiers = {'best', 'good', 'effective', 'novel', 'new'}
        if any(mod in words for mod in vague_modifiers):
            score += 0.2
        
        # Lack of technical terms increases ambiguity
        technical_count = len(self._extract_technical_terms(words))
        if technical_count == 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_complexity(
        self,
        word_count: int,
        technical_terms: List[str],
        has_boolean: bool,
        has_negation: bool,
        is_question: bool,
        ambiguity_score: float
    ) -> QueryComplexity:
        """Determine query complexity level"""
        
        # Complex criteria
        if (word_count >= 8 or
            len(technical_terms) >= 3 or
            has_boolean or
            has_negation or
            ambiguity_score >= 0.6):
            return QueryComplexity.COMPLEX
        
        # Simple criteria
        if (word_count <= 3 and
            len(technical_terms) <= 1 and
            not is_question and
            ambiguity_score < 0.3):
            return QueryComplexity.SIMPLE
        
        # Default to medium
        return QueryComplexity.MEDIUM
    
    def _should_rerank(
        self,
        complexity: QueryComplexity,
        word_count: int,
        technical_count: int,
        has_boolean: bool,
        ambiguity_score: float
    ) -> tuple[bool, float, str]:
        """
        Decide if reranking should be used
        
        Returns:
            (should_rerank, confidence, reasoning)
        """
        reasons = []
        confidence = 0.5
        
        # Complex queries benefit from reranking
        if complexity == QueryComplexity.COMPLEX:
            reasons.append("Complex query structure")
            confidence += 0.3
        
        # Long queries need precision
        if word_count >= self.rerank_threshold_words:
            reasons.append(f"Long query ({word_count} words)")
            confidence += 0.1
        
        # Technical queries need accuracy
        if technical_count >= self.rerank_threshold_technical:
            reasons.append(f"Multiple technical terms ({technical_count})")
            confidence += 0.2
        
        # Boolean operators indicate complex intent
        if has_boolean:
            reasons.append("Contains boolean operators")
            confidence += 0.15
        
        # Ambiguous queries benefit from reranking
        if ambiguity_score >= 0.5:
            reasons.append(f"High ambiguity (score: {ambiguity_score:.2f})")
            confidence += 0.15
        
        # Simple queries don't need reranking
        if complexity == QueryComplexity.SIMPLE:
            reasons.append("Simple query - reranking not needed")
            confidence = max(0.2, confidence - 0.4)
        
        # Decision
        should_rerank = confidence >= 0.6
        confidence = min(confidence, 1.0)
        
        reasoning = "; ".join(reasons) if reasons else "Standard query"
        
        return should_rerank, confidence, reasoning
    
    def batch_analyze(self, queries: List[str]) -> List[QueryAnalysis]:
        """Analyze multiple queries"""
        return [self.analyze(q) for q in queries]
    
    def get_statistics(self, analyses: List[QueryAnalysis]) -> Dict[str, Any]:
        """Get statistics from multiple analyses"""
        if not analyses:
            return {}
        
        total = len(analyses)
        
        return {
            "total_queries": total,
            "complexity_distribution": {
                "simple": sum(1 for a in analyses if a.complexity == QueryComplexity.SIMPLE),
                "medium": sum(1 for a in analyses if a.complexity == QueryComplexity.MEDIUM),
                "complex": sum(1 for a in analyses if a.complexity == QueryComplexity.COMPLEX)
            },
            "rerank_recommendations": {
                "yes": sum(1 for a in analyses if a.should_rerank),
                "no": sum(1 for a in analyses if not a.should_rerank),
                "percentage": sum(1 for a in analyses if a.should_rerank) / total * 100
            },
            "average_confidence": sum(a.confidence for a in analyses) / total,
            "average_word_count": sum(a.word_count for a in analyses) / total,
            "average_ambiguity": sum(a.ambiguity_score for a in analyses) / total
        }


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)
    
    analyzer = QueryAnalyzer()
    
    # Test queries
    test_queries = [
        "EEG",  # Simple
        "seizure detection",  # Simple
        "convolutional neural networks for epileptic seizure detection",  # Complex
        "What are the best methods for sleep staging?",  # Complex (question)
        "alpha beta gamma oscillations",  # Medium
        "BCI not using P300",  # Complex (negation)
    ]
    
    print("Query Analysis Results:\n" + "="*80)
    
    analyses = []
    for query in test_queries:
        analysis = analyzer.analyze(query)
        analyses.append(analysis)
        
        print(f"\nQuery: '{query}'")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Should Rerank: {'✓' if analysis.should_rerank else '✗'} (confidence: {analysis.confidence:.2f})")
        print(f"  Word Count: {analysis.word_count}, Technical Terms: {len(analysis.technical_terms)}")
        print(f"  Ambiguity: {analysis.ambiguity_score:.2f}")
        print(f"  Reasoning: {analysis.reasoning}")
    
    print(f"\n{'='*80}")
    print("Statistics:")
    stats = analyzer.get_statistics(analyses)
    print(f"  Reranking recommended: {stats['rerank_recommendations']['percentage']:.1f}% of queries")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    print(f"  Average ambiguity: {stats['average_ambiguity']:.2f}")
