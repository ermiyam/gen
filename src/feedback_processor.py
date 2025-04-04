import json
from pathlib import Path
from typing import List, Dict
import datetime

class FeedbackProcessor:
    def __init__(self, feedback_file: str = "data/feedback/gen_feedback.jsonl"):
        """Initialize the feedback processor."""
        self.feedback_file = Path(feedback_file)
        self.training_file = Path("data/training/nexgencreators_training_v1.jsonl")
        
    def load_feedback(self) -> List[Dict]:
        """Load all feedback entries from the JSONL file."""
        if not self.feedback_file.exists():
            return []
            
        feedback = []
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                feedback.append(json.loads(line))
        return feedback
    
    def filter_high_quality_examples(self, feedback: List[Dict], min_rating: int = 4) -> List[Dict]:
        """Filter feedback to get high-quality examples."""
        return [
            entry for entry in feedback
            if entry["rating"] >= min_rating and entry["used"]
        ]
    
    def convert_to_training_format(self, feedback_entry: Dict) -> Dict:
        """Convert a feedback entry to training format."""
        return {
            "instruction": feedback_entry["prompt"],
            "input": "",  # Can be enhanced later with metadata
            "output": feedback_entry["response"]
        }
    
    def prepare_training_data(self, min_rating: int = 4) -> List[Dict]:
        """Prepare high-quality feedback for training."""
        # Load all feedback
        feedback = self.load_feedback()
        
        # Filter high-quality examples
        high_quality = self.filter_high_quality_examples(feedback, min_rating)
        
        # Convert to training format
        training_data = [self.convert_to_training_format(entry) for entry in high_quality]
        
        return training_data
    
    def merge_with_existing_training(self, new_examples: List[Dict]) -> List[Dict]:
        """Merge new examples with existing training data."""
        if not self.training_file.exists():
            return new_examples
            
        # Load existing training data
        existing_data = []
        with open(self.training_file, "r", encoding="utf-8") as f:
            for line in f:
                existing_data.append(json.loads(line))
        
        # Combine and deduplicate
        combined = existing_data + new_examples
        unique = {json.dumps(entry, sort_keys=True): entry for entry in combined}
        return list(unique.values())
    
    def save_training_data(self, training_data: List[Dict]):
        """Save training data to file."""
        # Create backup of existing file if it exists
        if self.training_file.exists():
            backup_path = self.training_file.with_suffix(f".jsonl.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.training_file.rename(backup_path)
        
        # Save new training data
        with open(self.training_file, "w", encoding="utf-8") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")
    
    def process_feedback(self, min_rating: int = 4):
        """Process feedback and update training data."""
        # Prepare new training examples
        new_examples = self.prepare_training_data(min_rating)
        
        # Merge with existing data
        combined_data = self.merge_with_existing_training(new_examples)
        
        # Save updated training data
        self.save_training_data(combined_data)
        
        print(f"Processed {len(new_examples)} new examples")
        print(f"Total training examples: {len(combined_data)}")

def main():
    """Process feedback and prepare for retraining."""
    processor = FeedbackProcessor()
    processor.process_feedback()

if __name__ == "__main__":
    main() 