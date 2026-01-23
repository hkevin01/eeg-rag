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


@dataclass
class Feedback:
    """User feedback for a query response."""
    query_id: str
    rating: int  # 1-5 stars
    clicked_pmids: List[str] = field(default_factory=list)
    ignored_pmids: List[str] = field(default_factory=list)
    user_corrections: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        return self.rating >= 4 or len(self.clicked_pmids) > 0


@dataclass
class TrainingPair:
    """Query-document pair for retriever fine-tuning."""
    query: str
    positive_doc: str
    negative_docs: List[str] = field(default_factory=list)
    score: float = 1.0


class TrainingDataset:
    """Dataset for fine-tuning retrieval models."""
    
    def __init__(self, pairs: List[TrainingPair]):
        self.pairs = pairs
        logger.info(f"TrainingDataset created with {len(pairs)} pairs")
    
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
    
    def split(self, train_ratio: float = 0.8):
        """Split into train/val sets."""
        split_idx = int(len(self.pairs) * train_ratio)
        return (
            TrainingDataset(self.pairs[:split_idx]),
            TrainingDataset(self.pairs[split_idx:])
        )


class FeedbackStore:
    """Storage backend for feedback data."""
    
    def __init__(self, filepath: str = "data/feedback.jsonl"):
        self.filepath = filepath
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, feedback: Dict[str, Any]):
        """Append feedback to storage."""
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
    
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
    
    def get_positive_feedback(self) -> List[Dict[str, Any]]:
        """Get records with positive feedback."""
        all_records = self.load_all()
        return [r for r in all_records if r.get("rating", 0) >= 4]


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
    
    def __init__(self, store: Optional[FeedbackStore] = None):
        self.store = store or FeedbackStore()
        logger.info("FeedbackCollector initialized")
    
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
