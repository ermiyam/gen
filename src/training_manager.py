import json
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple

class TrainingManager:
    def __init__(self):
        self.priority_file = "src/priority_topics.json"
        self.knowledge_base = "data/knowledge"
        self.feedback_dir = "data/feedback"
        self.load_priorities()
        
    def load_priorities(self):
        """Load and validate priority configuration."""
        try:
            with open(self.priority_file, 'r') as f:
                self.priorities = json.load(f)
        except FileNotFoundError:
            logging.error(f"Priority file not found: {self.priority_file}")
            raise
            
    def get_next_training_focus(self) -> Tuple[str, Dict]:
        """Determine the next topic to focus on based on priorities and feedback."""
        topics = self.priorities['topics']
        
        # Sort topics by priority and training count
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: (
                -x[1]['priority'],
                x[1]['training_count']
            )
        )
        
        # Get the highest priority, least trained topic
        next_topic, details = sorted_topics[0]
        return next_topic, details
        
    def update_training_stats(self, topic: str):
        """Update training statistics for a topic."""
        if topic in self.priorities['topics']:
            self.priorities['topics'][topic]['last_trained'] = datetime.now().isoformat()
            self.priorities['topics'][topic]['training_count'] += 1
            
            # Save updated priorities
            with open(self.priority_file, 'w') as f:
                json.dump(self.priorities, f, indent=4)
                
    def format_training_example(self, 
                              user_input: str,
                              mak_response: str,
                              tone: str,
                              strategy: str,
                              platform: str,
                              category: str,
                              source: str,
                              rating: int) -> str:
        """Format a training example in the new format."""
        return f"""### USER:
{user_input}

### MAK:
{mak_response}

### TONE:
{tone}

### STRATEGY:
{strategy}

### PLATFORM:
{platform}

### CATEGORY:
{category}

### SOURCE:
{source}

### RATING:
{rating}
---
"""
        
    def collect_training_data(self) -> List[str]:
        """Collect and format training data from various sources."""
        training_data = []
        
        # Process high priority knowledge
        high_priority_dir = os.path.join(self.knowledge_base, "high")
        if os.path.exists(high_priority_dir):
            for file in os.listdir(high_priority_dir):
                if file.endswith(".txt"):
                    with open(os.path.join(high_priority_dir, file), 'r') as f:
                        content = f.read()
                        training_data.extend(content.split("---\n"))
                        
        # Process feedback
        if os.path.exists(self.feedback_dir):
            for file in os.listdir(self.feedback_dir):
                if file.endswith(".json"):
                    with open(os.path.join(self.feedback_dir, file), 'r') as f:
                        feedback = json.load(f)
                        training_data.append(
                            self.format_training_example(
                                feedback['user_input'],
                                feedback['mak_response'],
                                feedback['tone'],
                                feedback['strategy'],
                                feedback['platform'],
                                feedback['category'],
                                feedback['source'],
                                feedback['rating']
                            )
                        )
                        
        return training_data
        
    def save_training_data(self, data: List[str], output_file: str = "data/train.txt"):
        """Save formatted training data to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("\n".join(data))
            
    def get_training_plan(self) -> Dict:
        """Generate a training plan based on current priorities."""
        next_topic, details = self.get_next_training_focus()
        
        return {
            "next_topic": next_topic,
            "priority": details['priority'],
            "category": details['category'],
            "description": details['description'],
            "last_trained": details['last_trained'],
            "training_count": details['training_count']
        } 