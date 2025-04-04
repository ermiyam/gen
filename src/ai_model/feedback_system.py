from typing import Dict, Any, List
import json
import os
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class Feedback:
    query: str
    response: str
    feedback_type: FeedbackType
    rating: float  # 1-5
    comments: str
    timestamp: str
    metadata: Dict[str, Any]

class FeedbackSystem:
    def __init__(
        self,
        feedback_path: str = "data/feedback",
        model_path: str = "data/model_metrics"
    ):
        self.feedback_path = feedback_path
        self.model_path = model_path
        
        # Create directories if they don't exist
        os.makedirs(feedback_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize feedback storage
        self.feedback_file = os.path.join(feedback_path, "feedback.json")
        self.metrics_file = os.path.join(model_path, "metrics.json")
        
        # Load existing feedback and metrics
        self.feedback_history = self._load_feedback()
        self.model_metrics = self._load_metrics()
    
    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load existing feedback from file."""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing model metrics."""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "feedback_by_type": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "performance_history": []
        }
    
    def add_feedback(
        self,
        query: str,
        response: str,
        feedback_type: FeedbackType,
        rating: float,
        comments: str,
        metadata: Dict[str, Any] = None
    ):
        """Add new feedback and update metrics."""
        feedback = Feedback(
            query=query,
            response=response,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Add to feedback history
        self.feedback_history.append(feedback.__dict__)
        
        # Update metrics
        self._update_metrics(feedback)
        
        # Save updates
        self._save_updates()
    
    def _update_metrics(self, feedback: Feedback):
        """Update model metrics with new feedback."""
        # Update basic metrics
        self.model_metrics["total_feedback"] += 1
        self.model_metrics["feedback_by_type"][feedback.feedback_type.value] += 1
        
        # Update average rating
        current_total = self.model_metrics["average_rating"] * (self.model_metrics["total_feedback"] - 1)
        self.model_metrics["average_rating"] = (current_total + feedback.rating) / self.model_metrics["total_feedback"]
        
        # Add to performance history
        self.model_metrics["performance_history"].append({
            "timestamp": feedback.timestamp,
            "rating": feedback.rating,
            "feedback_type": feedback.feedback_type.value
        })
    
    def _save_updates(self):
        """Save feedback and metrics to disk."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of model performance."""
        return {
            "total_feedback": self.model_metrics["total_feedback"],
            "average_rating": round(self.model_metrics["average_rating"], 2),
            "feedback_distribution": self.model_metrics["feedback_by_type"],
            "recent_performance": self.model_metrics["performance_history"][-10:] if self.model_metrics["performance_history"] else []
        }
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get feedback data suitable for model training."""
        return [
            {
                "query": f["query"],
                "response": f["response"],
                "rating": f["rating"],
                "feedback_type": f["feedback_type"]
            }
            for f in self.feedback_history
            if f["rating"] >= 4  # Only use positive examples for training
        ] 