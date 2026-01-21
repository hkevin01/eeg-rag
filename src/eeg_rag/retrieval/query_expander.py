"""
Query Expansion for EEG Domain.

Expands queries with domain-specific synonyms and related terms to improve
retrieval coverage for EEG research papers.
"""

import logging
from typing import List, Set, Dict
import re

logger = logging.getLogger(__name__)


class EEGQueryExpander:
    """
    Query expansion with EEG domain knowledge.
    
    Expands queries by adding synonyms, abbreviations, and related terms
    specific to EEG research and neuroscience.
    
    Example:
        >>> expander = EEGQueryExpander()
        >>> expanded = expander.expand("CNN for seizure detection")
        >>> print(expanded)
        "CNN convolutional neural network for seizure epileptic detection"
    """
    
    def __init__(self):
        """Initialize with EEG domain synonym dictionary."""
        # Build synonym dictionary
        self.synonyms = self._build_synonym_dict()
        logger.info(f"Initialized EEGQueryExpander with {len(self.synonyms)} terms")
    
    def _build_synonym_dict(self) -> Dict[str, Set[str]]:
        """
        Build dictionary of EEG-specific synonyms.
        
        Returns:
            Dict mapping terms to their synonyms
        """
        # Define bidirectional synonym groups
        synonym_groups = [
            # Neural network architectures
            {"cnn", "convolutional neural network", "convnet"},
            {"rnn", "recurrent neural network"},
            {"lstm", "long short-term memory"},
            {"gru", "gated recurrent unit"},
            {"transformer", "attention model"},
            {"autoencoder", "ae"},
            
            # EEG analysis tasks
            {"seizure", "epileptic", "epilepsy", "ictal"},
            {"seizure detection", "epilepsy detection", "ictal detection"},
            {"seizure prediction", "epilepsy prediction", "seizure forecasting"},
            {"sleep staging", "sleep classification", "sleep scoring"},
            {"motor imagery", "mi", "movement imagination"},
            {"bci", "brain-computer interface", "brain machine interface", "bmi"},
            {"emotion recognition", "affective computing", "emotion classification"},
            {"cognitive load", "mental workload", "cognitive state"},
            
            # EEG frequency bands
            {"delta", "delta wave", "delta band", "0.5-4 hz"},
            {"theta", "theta wave", "theta band", "4-8 hz"},
            {"alpha", "alpha wave", "alpha band", "8-13 hz"},
            {"beta", "beta wave", "beta band", "13-30 hz"},
            {"gamma", "gamma wave", "gamma band", "30-100 hz"},
            
            # Signal processing methods
            {"wavelet", "wavelet transform", "wt"},
            {"fft", "fast fourier transform", "fourier"},
            {"stft", "short-time fourier transform"},
            {"psd", "power spectral density", "spectral power"},
            {"ica", "independent component analysis"},
            {"csp", "common spatial pattern"},
            
            # Feature extraction
            {"feature extraction", "feature engineering", "feature selection"},
            {"time domain", "temporal features"},
            {"frequency domain", "spectral features"},
            {"time-frequency", "timefrequency", "tf"},
            
            # Classification methods
            {"classification", "categorization", "recognition"},
            {"svm", "support vector machine"},
            {"random forest", "rf"},
            {"knn", "k-nearest neighbors", "k nearest neighbor"},
            
            # Deep learning concepts
            {"deep learning", "dl", "neural network"},
            {"machine learning", "ml"},
            {"transfer learning", "pretrained model"},
            {"data augmentation", "augmentation"},
            
            # EEG recording
            {"eeg", "electroencephalography", "electroencephalogram"},
            {"ecog", "electrocorticography"},
            {"meg", "magnetoencephalography"},
            {"electrode", "channel", "sensor"},
            
            # Datasets and paradigms
            {"dataset", "database", "corpus"},
            {"subject", "participant", "patient"},
            {"trial", "epoch", "segment"},
            {"artifact", "noise", "interference"},
            {"preprocessing", "pre-processing", "data cleaning"},
            
            # Performance metrics
            {"accuracy", "acc"},
            {"precision", "positive predictive value"},
            {"recall", "sensitivity", "true positive rate"},
            {"f1 score", "f1-score", "f-measure"},
            {"auc", "area under curve", "auroc"},
        ]
        
        # Build bidirectional mapping
        synonym_dict = {}
        for group in synonym_groups:
            group_list = list(group)
            for term in group_list:
                # Add all other terms in group as synonyms
                synonyms = set(group_list) - {term}
                synonym_dict[term.lower()] = synonyms
        
        return synonym_dict
    
    def expand(
        self,
        query: str,
        max_expansions: int = 3,
        add_original: bool = True
    ) -> str:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query text
            max_expansions: Maximum number of synonyms to add per term
            add_original: Keep original terms in expanded query
            
        Returns:
            Expanded query string
            
        Example:
            >>> expander.expand("CNN for seizure detection")
            "CNN convolutional neural network seizure epileptic detection"
        """
        # Tokenize query (simple whitespace split)
        tokens = query.lower().split()
        
        # Track expanded terms to avoid duplicates
        expanded_terms = set()
        if add_original:
            expanded_terms.update(tokens)
        
        # Expand each token
        for token in tokens:
            # Clean token (remove punctuation)
            clean_token = re.sub(r'[^\w\s-]', '', token)
            
            if clean_token in self.synonyms:
                # Get synonyms for this term
                synonyms = list(self.synonyms[clean_token])[:max_expansions]
                expanded_terms.update(synonyms)
        
        # Also check for multi-word phrases (bigrams)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in self.synonyms:
                synonyms = list(self.synonyms[bigram])[:max_expansions]
                expanded_terms.update(synonyms)
        
        # Build expanded query
        expanded_query = " ".join(expanded_terms)
        
        if expanded_query != query.lower():
            logger.debug(f"Expanded query: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        Get synonyms for a specific term.
        
        Args:
            term: Term to look up
            
        Returns:
            List of synonyms
        """
        return list(self.synonyms.get(term.lower(), []))
    
    def has_synonyms(self, term: str) -> bool:
        """Check if term has synonyms."""
        return term.lower() in self.synonyms


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)
    
    expander = EEGQueryExpander()
    
    # Test queries
    test_queries = [
        "CNN for seizure detection",
        "RNN for sleep staging",
        "BCI using motor imagery",
        "deep learning EEG classification",
        "wavelet features for epilepsy",
        "alpha band power in cognitive load"
    ]
    
    print("\n" + "="*80)
    print("🔍 EEG QUERY EXPANSION DEMO")
    print("="*80)
    
    for query in test_queries:
        expanded = expander.expand(query, max_expansions=2)
        print(f"\nOriginal:  {query}")
        print(f"Expanded:  {expanded}")
    
    print("\n" + "="*80)
    print("📚 SYNONYM LOOKUP")
    print("="*80)
    
    # Show some specific synonyms
    terms = ["cnn", "seizure", "eeg", "bci", "alpha"]
    for term in terms:
        syns = expander.get_synonyms(term)
        print(f"\n'{term}' -> {syns}")
    
    print("\n✅ Demo complete!\n")
