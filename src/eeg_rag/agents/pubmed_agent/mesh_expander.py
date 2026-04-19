"""
MeSH Term Expander for PubMed Queries

Expands queries with Medical Subject Headings (MeSH) terms for better recall.

Requirements Covered:
- REQ-PUBMED-001: MeSH term expansion for EEG domain
- REQ-PUBMED-002: Query enhancement with controlled vocabulary
"""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : agents.pubmed_agent.mesh_expander.MeSHExpander
# Requirement  : `MeSHExpander` class shall be instantiable and expose the documented interface
# Purpose      : Expand queries with MeSH (Medical Subject Headings) terms
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
# Verification : Instantiate MeSHExpander with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class MeSHExpander:
    """Expand queries with MeSH (Medical Subject Headings) terms."""
    
    # Common EEG-related MeSH mappings
    MESH_MAPPINGS: Dict[str, List[str]] = {
        # Brain signals and recording
        "eeg": ["Electroencephalography", "Brain Waves", "Evoked Potentials"],
        "electroencephalography": ["Electroencephalography", "Brain Waves"],
        "brain waves": ["Brain Waves", "Alpha Rhythm", "Beta Rhythm", "Theta Rhythm", "Delta Rhythm"],
        "erp": ["Evoked Potentials", "Event-Related Potentials, P300", "Contingent Negative Variation"],
        "event related potentials": ["Evoked Potentials", "Event-Related Potentials, P300"],
        
        # Clinical conditions
        "seizure": ["Seizures", "Epilepsy", "Electroencephalography"],
        "epilepsy": ["Epilepsy", "Seizures", "Anticonvulsants", "Electroencephalography"],
        "epileptic": ["Epilepsy", "Seizures", "Status Epilepticus"],
        "absence seizure": ["Epilepsy, Absence", "Seizures"],
        "status epilepticus": ["Status Epilepticus", "Epilepsy"],
        
        # Sleep
        "sleep": ["Sleep", "Sleep Stages", "Polysomnography", "Sleep Wake Disorders"],
        "sleep staging": ["Sleep Stages", "Polysomnography", "Electroencephalography"],
        "insomnia": ["Sleep Initiation and Maintenance Disorders", "Sleep Wake Disorders"],
        "rem sleep": ["Sleep, REM", "Sleep Stages"],
        "nrem sleep": ["Sleep Stages", "Sleep"],
        "polysomnography": ["Polysomnography", "Sleep"],
        
        # Brain-computer interfaces
        "bci": ["Brain-Computer Interfaces", "Neurofeedback", "Human-Computer Interaction"],
        "brain computer interface": ["Brain-Computer Interfaces", "Neurofeedback"],
        "neurofeedback": ["Neurofeedback", "Biofeedback, Psychology"],
        
        # Motor and movement
        "motor imagery": ["Imagination", "Motor Cortex", "Brain-Computer Interfaces"],
        "movement": ["Movement", "Motor Activity", "Motor Cortex"],
        "motor cortex": ["Motor Cortex", "Movement"],
        
        # Cognitive and emotion
        "emotion": ["Emotions", "Affect", "Facial Expression", "Mood Disorders"],
        "attention": ["Attention", "Cognition", "Mental Processes", "Attention Deficit Disorder with Hyperactivity"],
        "memory": ["Memory", "Memory, Short-Term", "Memory, Long-Term", "Cognition"],
        "cognition": ["Cognition", "Mental Processes", "Cognitive Dysfunction"],
        "cognitive load": ["Workload", "Cognition", "Mental Processes"],
        "mental workload": ["Workload", "Cognition", "Task Performance and Analysis"],
        
        # Neurodegenerative diseases
        "alzheimer": ["Alzheimer Disease", "Dementia", "Cognitive Dysfunction"],
        "alzheimers": ["Alzheimer Disease", "Dementia"],
        "parkinson": ["Parkinson Disease", "Movement Disorders"],
        "dementia": ["Dementia", "Alzheimer Disease", "Cognitive Dysfunction"],
        "mild cognitive impairment": ["Cognitive Dysfunction", "Dementia"],
        
        # Mental health
        "depression": ["Depression", "Depressive Disorder", "Mental Disorders"],
        "anxiety": ["Anxiety", "Anxiety Disorders", "Mental Disorders"],
        "schizophrenia": ["Schizophrenia", "Psychotic Disorders"],
        "adhd": ["Attention Deficit Disorder with Hyperactivity", "Mental Disorders"],
        
        # Machine learning and AI
        "deep learning": ["Deep Learning", "Neural Networks, Computer", "Machine Learning"],
        "machine learning": ["Machine Learning", "Artificial Intelligence", "Pattern Recognition, Automated"],
        "cnn": ["Neural Networks, Computer", "Deep Learning"],
        "convolutional neural network": ["Neural Networks, Computer", "Deep Learning"],
        "lstm": ["Neural Networks, Computer", "Deep Learning"],
        "neural network": ["Neural Networks, Computer", "Artificial Intelligence"],
        "classification": ["Classification", "Pattern Recognition, Automated", "Machine Learning"],
        "artificial intelligence": ["Artificial Intelligence", "Machine Learning"],
        
        # Signal processing
        "artifact": ["Artifacts", "Signal Processing, Computer-Assisted"],
        "preprocessing": ["Signal Processing, Computer-Assisted"],
        "signal processing": ["Signal Processing, Computer-Assisted"],
        "filtering": ["Signal Processing, Computer-Assisted"],
        "wavelet": ["Wavelet Analysis", "Signal Processing, Computer-Assisted"],
        "fourier": ["Fourier Analysis", "Signal Processing, Computer-Assisted"],
        
        # Frequency bands
        "alpha": ["Alpha Rhythm", "Brain Waves"],
        "beta": ["Beta Rhythm", "Brain Waves"],
        "gamma": ["Gamma Rhythm", "Brain Waves"],
        "theta": ["Theta Rhythm", "Brain Waves"],
        "delta": ["Delta Rhythm", "Brain Waves"],
        
        # ERP components
        "p300": ["Event-Related Potentials, P300", "Evoked Potentials"],
        "n400": ["Evoked Potentials", "Language"],
        "mmn": ["Evoked Potentials", "Evoked Potentials, Auditory"],
        "mismatch negativity": ["Evoked Potentials", "Evoked Potentials, Auditory"],
        
        # Electrode systems
        "10-20 system": ["Electroencephalography", "Electrodes"],
        "electrode": ["Electrodes", "Electroencephalography"],
        "scalp": ["Scalp", "Electroencephalography"],
        
        # Datasets and benchmarks
        "dataset": ["Databases, Factual", "Benchmarking"],
        "benchmark": ["Benchmarking", "Reference Standards"],
    }
    
    # MeSH subheadings relevant for EEG research
    RELEVANT_SUBHEADINGS = [
        "diagnosis",
        "classification", 
        "methods",
        "analysis",
        "physiology",
        "pathophysiology",
        "therapy",
        "drug therapy",
        "instrumentation"
    ]
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.mesh_expander.MeSHExpander.__init__
    # Requirement  : `__init__` shall initialize MeSH expander
    # Purpose      : Initialize MeSH expander
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : custom_mappings: Dict[str, List[str]] (default=None)
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
    def __init__(self, custom_mappings: Dict[str, List[str]] = None):
        """
        Initialize MeSH expander.
        
        Args:
            custom_mappings: Additional custom MeSH mappings to add
        """
        self.mappings = self.MESH_MAPPINGS.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)
        logger.info(f"MeSH expander initialized with {len(self.mappings)} term mappings")
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.mesh_expander.MeSHExpander.expand_query
    # Requirement  : `expand_query` shall convert natural language query to PubMed query with MeSH terms
    # Purpose      : Convert natural language query to PubMed query with MeSH terms
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str
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
    def expand_query(self, query: str) -> str:
        """
        Convert natural language query to PubMed query with MeSH terms.
        
        Args:
            query: Natural language search query
            
        Returns:
            PubMed query string with MeSH term expansion
        """
        query_lower = query.lower()
        mesh_terms: Set[str] = set()
        
        # Find matching MeSH terms
        for keyword, terms in self.mappings.items():
            if keyword in query_lower:
                mesh_terms.update(terms)
        
        if not mesh_terms:
            # Return basic query if no MeSH mappings found
            return query
        
        # Build PubMed query
        # Original text search in title/abstract
        text_query = f'({query}[Title/Abstract])'
        
        # MeSH term search (limit to top 5 most relevant)
        mesh_list = list(mesh_terms)[:5]
        mesh_parts = [f'"{term}"[MeSH Terms]' for term in mesh_list]
        mesh_query = f'({" OR ".join(mesh_parts)})'
        
        # Combine: papers matching text OR MeSH terms
        combined = f'{text_query} OR {mesh_query}'
        
        logger.debug(f"Expanded query: {query[:50]}... -> {len(mesh_terms)} MeSH terms")
        return combined
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.mesh_expander.MeSHExpander.get_mesh_suggestions
    # Requirement  : `get_mesh_suggestions` shall get suggested MeSH terms for a query without building full query
    # Purpose      : Get suggested MeSH terms for a query without building full query
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
    def get_mesh_suggestions(self, query: str) -> List[str]:
        """
        Get suggested MeSH terms for a query without building full query.
        
        Args:
            query: Search query
            
        Returns:
            List of suggested MeSH terms
        """
        query_lower = query.lower()
        suggestions: Set[str] = set()
        
        for keyword, terms in self.mappings.items():
            if keyword in query_lower:
                suggestions.update(terms)
        
        return sorted(list(suggestions))
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.mesh_expander.MeSHExpander.get_mesh_for_term
    # Requirement  : `get_mesh_for_term` shall get MeSH terms for a specific keyword
    # Purpose      : Get MeSH terms for a specific keyword
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : term: str
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
    def get_mesh_for_term(self, term: str) -> List[str]:
        """
        Get MeSH terms for a specific keyword.
        
        Args:
            term: Keyword to look up
            
        Returns:
            List of matching MeSH terms
        """
        return self.mappings.get(term.lower(), [])
    
    # ---------------------------------------------------------------------------
    # ID           : agents.pubmed_agent.mesh_expander.MeSHExpander.add_mapping
    # Requirement  : `add_mapping` shall add a custom keyword-to-MeSH mapping
    # Purpose      : Add a custom keyword-to-MeSH mapping
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : keyword: str; mesh_terms: List[str]
    # Outputs      : None
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
    def add_mapping(self, keyword: str, mesh_terms: List[str]) -> None:
        """
        Add a custom keyword-to-MeSH mapping.
        
        Args:
            keyword: Keyword to map
            mesh_terms: List of MeSH terms
        """
        self.mappings[keyword.lower()] = mesh_terms
        logger.debug(f"Added MeSH mapping: {keyword} -> {mesh_terms}")
