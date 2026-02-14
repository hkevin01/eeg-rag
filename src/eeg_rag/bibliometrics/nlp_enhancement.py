"""
EEG Bibliometrics NLP Enhancement Module

Provides KeyBERT keyword extraction, spaCy lemmatization, 
topic modeling, and semantic query expansion for EEG research.

Requirements:
- REQ-NLP-001: KeyBERT keyword extraction from abstracts
- REQ-NLP-002: spaCy lemmatization for text matching
- REQ-NLP-003: Topic modeling for document categorization  
- REQ-NLP-004: Semantic query expansion using extracted keywords
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ExtractedKeywords:
    """
    Container for extracted keywords with metadata.
    
    Attributes:
        keywords: List of (keyword, score) tuples
        document_id: Source document identifier
        method: Extraction method used
        ngram_range: N-gram range used for extraction
    """
    keywords: List[Tuple[str, float]]
    document_id: Optional[str] = None
    method: str = "keybert"
    ngram_range: Tuple[int, int] = (1, 2)
    
    def get_top_n(self, n: int = 10) -> List[str]:
        """Get top N keywords by score."""
        return [kw for kw, _ in sorted(self.keywords, key=lambda x: x[1], reverse=True)[:n]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "keywords": self.keywords,
            "document_id": self.document_id,
            "method": self.method,
            "ngram_range": self.ngram_range,
        }


@dataclass
class TopicCluster:
    """
    Represents a topic cluster from topic modeling.
    
    Attributes:
        topic_id: Unique topic identifier
        keywords: Representative keywords for the topic
        document_ids: Documents belonging to this topic
        coherence_score: Topic coherence score
    """
    topic_id: int
    keywords: List[str]
    document_ids: List[str] = field(default_factory=list)
    coherence_score: float = 0.0
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic_id": self.topic_id,
            "keywords": self.keywords,
            "document_ids": self.document_ids,
            "coherence_score": self.coherence_score,
            "label": self.label,
        }


class EEGNLPEnhancer:
    """
    NLP enhancement engine for EEG bibliometric data.
    
    Provides keyword extraction, lemmatization, topic modeling,
    and query expansion capabilities specifically tuned for EEG research.
    
    Example:
        >>> nlp = EEGNLPEnhancer()
        >>> keywords = nlp.extract_keywords(articles)
        >>> expanded = nlp.expand_query("seizure detection EEG")
    """
    
    # EEG-specific stopwords to filter out
    EEG_STOPWORDS = {
        "eeg", "electroencephalography", "electroencephalogram",
        "study", "research", "analysis", "method", "methods",
        "result", "results", "conclusion", "conclusions",
        "patient", "patients", "subject", "subjects",
        "data", "using", "used", "use", "based",
        "showed", "shows", "show", "found", "find",
        "significant", "significantly", "compared", "between",
        "however", "therefore", "moreover", "furthermore",
        "aim", "objective", "purpose", "background",
    }
    
    # EEG domain-specific terms to boost
    EEG_DOMAIN_TERMS = {
        "frequency_bands": ["delta", "theta", "alpha", "beta", "gamma", "mu"],
        "signal_types": ["erp", "evoked", "spontaneous", "oscillation", "spike", "wave"],
        "clinical": ["epilepsy", "seizure", "ictal", "interictal", "postictal", "absence"],
        "cognitive": ["p300", "n400", "p600", "mmn", "n170", "erp", "cognitive"],
        "sleep": ["rem", "nrem", "slow-wave", "spindle", "k-complex", "polysomnography"],
        "bci": ["motor imagery", "ssvep", "p300 speller", "brain-computer"],
        "preprocessing": ["artifact", "ica", "filtering", "bandpass", "notch", "rejection"],
        "electrodes": ["10-20", "10-10", "channel", "electrode", "montage", "reference"],
    }
    
    def __init__(
        self,
        use_keybert: bool = True,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """
        Initialize NLP enhancer.
        
        Args:
            use_keybert: Whether to use KeyBERT for extraction
            use_spacy: Whether to use spaCy for lemmatization
            spacy_model: spaCy model name to use
        """
        self.use_keybert = use_keybert
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        
        self._keybert = None
        self._nlp = None
        
        # Initialize on first use
        self._keybert_available = self._check_keybert()
        self._spacy_available = self._check_spacy()
    
    def _check_keybert(self) -> bool:
        """Check if KeyBERT is available."""
        if not self.use_keybert:
            return False
        try:
            from keybert import KeyBERT
            return True
        except ImportError:
            logger.warning("KeyBERT not available. Install with: pip install keybert")
            return False
    
    def _check_spacy(self) -> bool:
        """Check if spaCy is available."""
        if not self.use_spacy:
            return False
        try:
            import spacy
            return True
        except ImportError:
            logger.warning("spaCy not available. Install with: pip install spacy")
            return False
    
    def _get_keybert(self):
        """Get or initialize KeyBERT model."""
        if self._keybert is None and self._keybert_available:
            from keybert import KeyBERT
            self._keybert = KeyBERT()
            logger.info("Initialized KeyBERT model")
        return self._keybert
    
    def _get_spacy_nlp(self):
        """Get or initialize spaCy NLP pipeline."""
        if self._nlp is None and self._spacy_available:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model)
                logger.info(f"Loaded spaCy model: {self.spacy_model}")
            except OSError:
                logger.warning(f"spaCy model {self.spacy_model} not found. Downloading...")
                from spacy.cli import download
                download(self.spacy_model)
                self._nlp = spacy.load(self.spacy_model)
        return self._nlp
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text using spaCy.
        
        REQ-NLP-002: SpaCy lemmatization for text matching.
        
        Args:
            text: Input text to lemmatize
            
        Returns:
            Lemmatized text string
        """
        if not self._spacy_available:
            return text.lower()
        
        nlp = self._get_spacy_nlp()
        if nlp is None:
            return text.lower()
        
        doc = nlp(text)
        lemmatized = " ".join([
            token.lemma_ for token in doc 
            if not token.is_punct and not token.is_space
        ])
        return lemmatized
    
    def extract_keywords_from_text(
        self,
        text: str,
        top_n: int = 10,
        ngram_range: Tuple[int, int] = (1, 2),
        diversity: float = 0.5,
        use_eeg_boost: bool = True,
    ) -> ExtractedKeywords:
        """
        Extract keywords from a single text using KeyBERT.
        
        REQ-NLP-001: KeyBERT keyword extraction.
        
        Args:
            text: Input text (abstract or full text)
            top_n: Number of keywords to extract
            ngram_range: Range of n-grams to extract
            diversity: MMR diversity parameter (0-1)
            use_eeg_boost: Whether to boost EEG domain terms
            
        Returns:
            ExtractedKeywords object
        """
        if not text or not text.strip():
            return ExtractedKeywords(keywords=[], ngram_range=ngram_range)
        
        if not self._keybert_available:
            # Fallback: simple term frequency
            return self._fallback_keyword_extraction(text, top_n, ngram_range)
        
        # Preprocess: lemmatize if spaCy available
        processed_text = self.lemmatize_text(text) if self._spacy_available else text.lower()
        
        # Extract with KeyBERT
        kw_model = self._get_keybert()
        try:
            keywords = kw_model.extract_keywords(
                processed_text,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=top_n * 2,  # Extract more for filtering
                use_mmr=True,
                diversity=diversity,
            )
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
            return self._fallback_keyword_extraction(text, top_n, ngram_range)
        
        # Filter and rerank
        filtered_keywords = []
        for kw, score in keywords:
            # Skip EEG stopwords
            kw_lower = kw.lower()
            if kw_lower in self.EEG_STOPWORDS:
                continue
            
            # Boost EEG domain terms
            boost = 1.0
            if use_eeg_boost:
                for domain, terms in self.EEG_DOMAIN_TERMS.items():
                    if any(term in kw_lower for term in terms):
                        boost = 1.3
                        break
            
            filtered_keywords.append((kw, score * boost))
        
        # Sort by boosted score and take top_n
        filtered_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return ExtractedKeywords(
            keywords=filtered_keywords[:top_n],
            method="keybert",
            ngram_range=ngram_range,
        )
    
    def _fallback_keyword_extraction(
        self,
        text: str,
        top_n: int,
        ngram_range: Tuple[int, int],
    ) -> ExtractedKeywords:
        """Fallback keyword extraction using term frequency."""
        # Simple tokenization
        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        
        # Remove stopwords
        english_stopwords = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'was', 'were', 'been',
            'has', 'had', 'are', 'is', 'may', 'more', 'such',
        }
        
        filtered = [w for w in words 
                   if w not in english_stopwords 
                   and w not in self.EEG_STOPWORDS]
        
        # Count frequencies
        freq = Counter(filtered)
        
        # Also extract bigrams if requested
        if ngram_range[1] >= 2:
            for i in range(len(filtered) - 1):
                bigram = f"{filtered[i]} {filtered[i+1]}"
                freq[bigram] += 0.5  # Weight bigrams less
        
        # Convert to (keyword, score) format
        total = sum(freq.values()) or 1
        keywords = [(kw, count / total) for kw, count in freq.most_common(top_n)]
        
        return ExtractedKeywords(
            keywords=keywords,
            method="frequency",
            ngram_range=ngram_range,
        )
    
    def extract_keywords_from_articles(
        self,
        articles: List[Any],
        top_n: int = 10,
        aggregate: bool = True,
    ) -> Union[Dict[str, ExtractedKeywords], ExtractedKeywords]:
        """
        Extract keywords from multiple articles.
        
        REQ-NLP-001: Batch keyword extraction.
        
        Args:
            articles: List of EEGArticle objects or dicts
            top_n: Keywords per article (or total if aggregate)
            aggregate: Whether to aggregate across all articles
            
        Returns:
            Dict of doc_id -> keywords if not aggregating,
            otherwise single aggregated ExtractedKeywords
        """
        all_keywords: Dict[str, List[Tuple[str, float]]] = {}
        
        for article in articles:
            # Extract article info
            if hasattr(article, 'openalex_id'):
                doc_id = article.openalex_id
                abstract = getattr(article, 'abstract', '') or ''
                title = getattr(article, 'title', '') or ''
            elif isinstance(article, dict):
                doc_id = article.get('openalex_id', article.get('id', str(hash(str(article)))))
                abstract = article.get('abstract', '') or ''
                title = article.get('title', '') or ''
            else:
                continue
            
            # Combine title and abstract
            text = f"{title}. {abstract}"
            
            if text.strip():
                result = self.extract_keywords_from_text(text, top_n=top_n)
                result.document_id = doc_id
                all_keywords[doc_id] = result
        
        if not aggregate:
            return all_keywords
        
        # Aggregate keywords across all documents
        keyword_scores = Counter()
        for doc_id, extracted in all_keywords.items():
            for kw, score in extracted.keywords:
                keyword_scores[kw] += score
        
        # Normalize and return top N
        aggregated = [
            (kw, score / len(all_keywords))
            for kw, score in keyword_scores.most_common(top_n)
        ]
        
        return ExtractedKeywords(
            keywords=aggregated,
            method="aggregated_keybert" if self._keybert_available else "aggregated_frequency",
        )
    
    def get_keyword_trends(
        self,
        articles: List[Any],
        interval: str = "year",
        top_n: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get keyword trends over time.
        
        REQ-NLP-004: Keyword trend analysis.
        
        Args:
            articles: List of articles with abstracts
            interval: Time interval ("month", "quarter", "year")
            top_n: Top keywords to track per interval
            
        Returns:
            Dict mapping time_period -> {keyword: score}
        """
        period_keywords: Dict[str, Counter] = defaultdict(Counter)
        
        for article in articles:
            # Get date
            if hasattr(article, 'publication_date'):
                date_str = article.publication_date
                abstract = getattr(article, 'abstract', '') or ''
                title = getattr(article, 'title', '') or ''
            elif isinstance(article, dict):
                date_str = article.get('publication_date')
                abstract = article.get('abstract', '') or ''
                title = article.get('title', '') or ''
            else:
                continue
            
            if not date_str:
                continue
            
            # Parse date
            try:
                pub_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            
            # Time key
            if interval == "month":
                key = pub_date.strftime("%Y-%m")
            elif interval == "quarter":
                quarter = (pub_date.month - 1) // 3 + 1
                key = f"{pub_date.year}-Q{quarter}"
            else:
                key = str(pub_date.year)
            
            # Extract keywords
            text = f"{title}. {abstract}"
            if text.strip():
                extracted = self.extract_keywords_from_text(text, top_n=top_n)
                for kw, score in extracted.keywords:
                    period_keywords[key][kw] += score
        
        # Convert to normalized dict
        result = {}
        for period, kw_counts in sorted(period_keywords.items()):
            total = sum(kw_counts.values()) or 1
            result[period] = {
                kw: score / total
                for kw, score in kw_counts.most_common(top_n)
            }
        
        return result
    
    def expand_query(
        self,
        query: str,
        method: str = "synonyms",
        max_expansions: int = 5,
    ) -> List[str]:
        """
        Expand a query with related terms.
        
        REQ-NLP-004: Semantic query expansion.
        
        Args:
            query: Original search query
            method: Expansion method ("synonyms", "eeg_domain", "both")
            max_expansions: Maximum number of expansions to add
            
        Returns:
            List of expanded query terms
        """
        expanded = [query]
        query_lower = query.lower()
        
        # EEG domain expansions
        eeg_expansions = {
            "eeg": ["electroencephalography", "electroencephalogram", "brain waves"],
            "seizure": ["epileptic", "ictal", "convulsion", "epilepsy"],
            "epilepsy": ["seizure", "ictal", "interictal", "epileptic"],
            "sleep": ["polysomnography", "rem", "nrem", "sleep staging"],
            "bci": ["brain-computer interface", "brain machine interface", "motor imagery"],
            "erp": ["event-related potential", "evoked potential", "p300", "n400"],
            "p300": ["erp", "event-related potential", "oddball"],
            "alpha": ["alpha rhythm", "alpha band", "8-13 hz", "alpha wave"],
            "beta": ["beta rhythm", "beta band", "13-30 hz", "beta wave"],
            "theta": ["theta rhythm", "theta band", "4-8 hz", "theta wave"],
            "delta": ["delta rhythm", "delta band", "0.5-4 hz", "slow wave"],
            "gamma": ["gamma rhythm", "gamma band", "30-100 hz", "high frequency"],
            "artifact": ["noise", "artifact removal", "ica", "artifact rejection"],
            "motor imagery": ["mi-bci", "imagined movement", "motor task"],
            "attention": ["attentional", "vigilance", "sustained attention"],
            "memory": ["working memory", "episodic memory", "memory encoding"],
            "preprocessing": ["filtering", "artifact removal", "signal processing"],
        }
        
        if method in ("eeg_domain", "both"):
            for term, synonyms in eeg_expansions.items():
                if term in query_lower:
                    expanded.extend(synonyms[:max_expansions])
        
        # Simple synonym expansion using domain knowledge
        if method in ("synonyms", "both"):
            # Check domain term categories
            for domain, terms in self.EEG_DOMAIN_TERMS.items():
                for term in terms:
                    if term in query_lower:
                        # Add other terms from same domain
                        related = [t for t in terms if t != term and t not in query_lower]
                        expanded.extend(related[:2])
        
        # Deduplicate while preserving order
        seen = set()
        unique_expanded = []
        for term in expanded:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_expanded.append(term)
        
        return unique_expanded[:max_expansions + 1]  # Include original
    
    def categorize_by_topic(
        self,
        articles: List[Any],
        categories: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Categorize articles by topic using keyword matching.
        
        REQ-NLP-003: Topic-based document categorization.
        
        Args:
            articles: List of articles to categorize
            categories: Dict of category_name -> keywords. 
                       If None, uses EEG domain categories.
            
        Returns:
            Dict of category_name -> list of document IDs
        """
        if categories is None:
            categories = {
                "epilepsy": ["epilepsy", "seizure", "ictal", "interictal", "convulsion"],
                "sleep": ["sleep", "rem", "nrem", "polysomnography", "sleep staging", "insomnia"],
                "bci": ["bci", "brain-computer", "motor imagery", "ssvep", "p300 speller"],
                "cognitive": ["cognitive", "attention", "memory", "erp", "p300", "n400", "executive"],
                "clinical": ["clinical", "diagnosis", "neurological", "patient", "hospital"],
                "signal_processing": ["artifact", "filtering", "ica", "preprocessing", "wavelet"],
                "connectivity": ["connectivity", "coherence", "phase", "synchronization", "network"],
                "deep_learning": ["deep learning", "neural network", "cnn", "lstm", "transformer"],
            }
        
        categorized: Dict[str, List[str]] = {cat: [] for cat in categories}
        
        for article in articles:
            # Get article info
            if hasattr(article, 'openalex_id'):
                doc_id = article.openalex_id
                abstract = (getattr(article, 'abstract', '') or '').lower()
                title = (getattr(article, 'title', '') or '').lower()
            elif isinstance(article, dict):
                doc_id = article.get('openalex_id', article.get('id', ''))
                abstract = (article.get('abstract', '') or '').lower()
                title = (article.get('title', '') or '').lower()
            else:
                continue
            
            text = f"{title} {abstract}"
            
            # Match against categories
            for category, keywords in categories.items():
                if any(kw.lower() in text for kw in keywords):
                    categorized[category].append(doc_id)
        
        return categorized
    
    def compute_text_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute similarity between two texts.
        
        Uses keyword overlap as a simple similarity metric.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Extract keywords from both
        kw1 = self.extract_keywords_from_text(text1, top_n=20)
        kw2 = self.extract_keywords_from_text(text2, top_n=20)
        
        # Get keyword sets
        set1 = {kw.lower() for kw, _ in kw1.keywords}
        set2 = {kw.lower() for kw, _ in kw2.keywords}
        
        if not set1 or not set2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


def extract_eeg_keywords(
    articles: List[Any],
    top_n: int = 20,
) -> ExtractedKeywords:
    """
    Convenience function to extract keywords from EEG articles.
    
    Args:
        articles: List of EEGArticle objects or dicts
        top_n: Number of top keywords
        
    Returns:
        Aggregated ExtractedKeywords
    """
    nlp = EEGNLPEnhancer()
    return nlp.extract_keywords_from_articles(articles, top_n=top_n, aggregate=True)


def expand_eeg_query(query: str) -> List[str]:
    """
    Convenience function to expand an EEG query.
    
    Args:
        query: Original query string
        
    Returns:
        List of expanded query terms
    """
    nlp = EEGNLPEnhancer(use_keybert=False, use_spacy=False)
    return nlp.expand_query(query, method="both")
