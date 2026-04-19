"""
Continuous Learning and Feedback Loop.

Collects user feedback and generates training data for model improvement.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID           : feedback.learning.Feedback
# Requirement  : `Feedback` class shall be instantiable and expose the documented interface
# Purpose      : User feedback for a query response
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
# Verification : Instantiate Feedback with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class Feedback:
    """User feedback for a query response."""
    query_id: str
    rating: int  # 1-5 stars
    clicked_pmids: List[str] = field(default_factory=list)
    ignored_pmids: List[str] = field(default_factory=list)
    user_corrections: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.Feedback.is_positive
    # Requirement  : `is_positive` shall check if feedback is positive
    # Purpose      : Check if feedback is positive
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        return self.rating >= 4 or len(self.clicked_pmids) > 0


# ---------------------------------------------------------------------------
# ID           : feedback.learning.TrainingPair
# Requirement  : `TrainingPair` class shall be instantiable and expose the documented interface
# Purpose      : Query-document pair for retriever fine-tuning
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
# Verification : Instantiate TrainingPair with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class TrainingPair:
    """Query-document pair for retriever fine-tuning."""
    query: str
    positive_doc: str
    negative_docs: List[str] = field(default_factory=list)
    score: float = 1.0


# ---------------------------------------------------------------------------
# ID           : feedback.learning.TrainingDataset
# Requirement  : `TrainingDataset` class shall be instantiable and expose the documented interface
# Purpose      : Dataset for fine-tuning retrieval models
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
# Verification : Instantiate TrainingDataset with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class TrainingDataset:
    """Dataset for fine-tuning retrieval models."""
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.TrainingDataset.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pairs: List[TrainingPair]
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
    def __init__(self, pairs: List[TrainingPair]):
        self.pairs = pairs
        logger.info(f"TrainingDataset created with {len(pairs)} pairs")
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.TrainingDataset.export_jsonl
    # Requirement  : `export_jsonl` shall export as JSONL for model training
    # Purpose      : Export as JSONL for model training
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : filepath: str
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
    def export_jsonl(self, filepath: str):
        """Export as JSONL for model training."""
        with open(filepath, 'w') as f:
            for pair in self.pairs:
                f.write(json.dumps({
                    "query": pair.query,
                    "positive": pair.positive_doc,
                    "negatives": pair.negative_docs,
                    "score": pair.score
                }) + '\n')
        logger.info(f"Exported {len(self.pairs)} pairs to {filepath}")
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.TrainingDataset.split
    # Requirement  : `split` shall split into train/val sets
    # Purpose      : Split into train/val sets
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : train_ratio: float (default=0.8)
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
    def split(self, train_ratio: float = 0.8):
        """Split into train/val sets."""
        split_idx = int(len(self.pairs) * train_ratio)
        return (
            TrainingDataset(self.pairs[:split_idx]),
            TrainingDataset(self.pairs[split_idx:])
        )


# ---------------------------------------------------------------------------
# ID           : feedback.learning.FeedbackStore
# Requirement  : `FeedbackStore` class shall be instantiable and expose the documented interface
# Purpose      : Storage backend for feedback data
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
# Verification : Instantiate FeedbackStore with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FeedbackStore:
    """Storage backend for feedback data."""
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackStore.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : filepath: str (default='data/feedback.jsonl')
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
    def __init__(self, filepath: str = "data/feedback.jsonl"):
        self.filepath = filepath
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackStore.save
    # Requirement  : `save` shall append feedback to storage
    # Purpose      : Append feedback to storage
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : feedback: Dict[str, Any]
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
    def save(self, feedback: Dict[str, Any]):
        """Append feedback to storage."""
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackStore.load_all
    # Requirement  : `load_all` shall load all feedback records
    # Purpose      : Load all feedback records
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[Dict[str, Any]]
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
    def load_all(self) -> List[Dict[str, Any]]:
        """Load all feedback records."""
        if not Path(self.filepath).exists():
            return []
        
        records = []
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackStore.get_positive_feedback
    # Requirement  : `get_positive_feedback` shall get records with positive feedback
    # Purpose      : Get records with positive feedback
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[Dict[str, Any]]
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
    def get_positive_feedback(self) -> List[Dict[str, Any]]:
        """Get records with positive feedback."""
        all_records = self.load_all()
        return [r for r in all_records if r.get("rating", 0) >= 4]


# ---------------------------------------------------------------------------
# ID           : feedback.learning.FeedbackCollector
# Requirement  : `FeedbackCollector` class shall be instantiable and expose the documented interface
# Purpose      : Collect and process user feedback for continuous learning
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
# Verification : Instantiate FeedbackCollector with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FeedbackCollector:
    """
    Collect and process user feedback for continuous learning.
    
    Features:
    - Record explicit feedback (ratings)
    - Track implicit feedback (clicks)
    - Generate training data for fine-tuning
    - Active learning for NER improvement
    
    Example:
        collector = FeedbackCollector()
        
        # Record feedback
        feedback = Feedback(
            query_id="abc123",
            rating=5,
            clicked_pmids=["12345678", "87654321"]
        )
        collector.record_feedback("search_query", feedback)
        
        # Generate training data
        dataset = collector.generate_training_data()
        dataset.export_jsonl("training_data.jsonl")
    """
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackCollector.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : store: Optional[FeedbackStore] (default=None)
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
    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()
        logger.info("FeedbackCollector initialized")
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackCollector.record_feedback
    # Requirement  : `record_feedback` shall store feedback for model improvement
    # Purpose      : Store feedback for model improvement
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; feedback: Feedback
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
    def record_feedback(self, query: str, feedback: Feedback):
        """
        Store feedback for model improvement.
        
        Args:
            query: Original search query
            feedback: User feedback object
        """
        record = {
            "query_id": feedback.query_id,
            "query": query,
            "rating": feedback.rating,
            "clicked_pmids": feedback.clicked_pmids,
            "ignored_pmids": feedback.ignored_pmids,
            "user_corrections": feedback.user_corrections,
            "timestamp": feedback.timestamp,
            "is_positive": feedback.is_positive()
        }
        
        self.store.save(record)
        logger.info(f"Recorded feedback for query_id={feedback.query_id}, rating={feedback.rating}")
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackCollector.generate_training_data
    # Requirement  : `generate_training_data` shall convert feedback into training pairs for retriever fine-tuning
    # Purpose      : Convert feedback into training pairs for retriever fine-tuning
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : min_rating: int (default=4)
    # Outputs      : TrainingDataset
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
    def generate_training_data(self, min_rating: int = 4) -> TrainingDataset:
        """
        Convert feedback into training pairs for retriever fine-tuning.
        
        Args:
            min_rating: Minimum rating to consider positive
        
        Returns:
            Training dataset with query-document pairs
        """
        logger.info("Generating training data from feedback...")
        
        records = self.store.load_all()
        pairs = []
        
        for record in records:
            if record.get("rating", 0) < min_rating:
                continue
            
            query = record["query"]
            clicked_pmids = record.get("clicked_pmids", [])
            ignored_pmids = record.get("ignored_pmids", [])
            
            # Create training pairs
            for pmid in clicked_pmids:
                pair = TrainingPair(
                    query=query,
                    positive_doc=pmid,  # In production, fetch full document
                    negative_docs=ignored_pmids[:5],  # Sample negatives
                    score=record["rating"] / 5.0
                )
                pairs.append(pair)
        
        dataset = TrainingDataset(pairs)
        logger.info(f"Generated {len(pairs)} training pairs")
        return dataset
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackCollector.get_active_learning_candidates
    # Requirement  : `get_active_learning_candidates` shall identify queries that need human annotation for active learning
    # Purpose      : Identify queries that need human annotation for active learning
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : top_k: int (default=100)
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
    def get_active_learning_candidates(self, top_k: int = 100) -> List[str]:
        """
        Identify queries that need human annotation for active learning.
        
        Returns queries with:
        - Low confidence predictions
        - High disagreement between models
        - No user feedback yet
        
        Args:
            top_k: Number of candidates to return
        
        Returns:
            List of query IDs needing annotation
        """
        # In production: analyze model confidence scores, user interactions
        # Return queries where model is uncertain
        
        records = self.store.load_all()
        
        # Find queries with no feedback
        query_feedback_count = {}
        for record in records:
            qid = record["query_id"]
            query_feedback_count[qid] = query_feedback_count.get(qid, 0) + 1
        
        # Prioritize queries with no or low feedback
        candidates = sorted(
            query_feedback_count.items(),
            key=lambda x: x[1]
        )[:top_k]
        
        return [qid for qid, count in candidates]
    
    # ---------------------------------------------------------------------------
    # ID           : feedback.learning.FeedbackCollector.get_statistics
    # Requirement  : `get_statistics` shall get feedback statistics for monitoring
    # Purpose      : Get feedback statistics for monitoring
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
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics for monitoring."""
        records = self.store.load_all()
        
        if not records:
            return {
                "total_feedback": 0,
                "avg_rating": 0,
                "positive_ratio": 0,
                "click_through_rate": 0
            }
        
        ratings = [r["rating"] for r in records if "rating" in r]
        positive = len([r for r in records if r.get("is_positive")])
        clicks = sum(len(r.get("clicked_pmids", [])) for r in records)
        total_shown = sum(len(r.get("clicked_pmids", [])) + len(r.get("ignored_pmids", [])) for r in records)
        
        return {
            "total_feedback": len(records),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "positive_ratio": positive / len(records) if records else 0,
            "click_through_rate": clicks / total_shown if total_shown > 0 else 0,
            "total_clicks": clicks
        }
