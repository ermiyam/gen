import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class DataType(Enum):
    INSTRUCTION = "instruction"
    INPUT = "input"
    OUTPUT = "output"

@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any] = None

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: str) -> List[TrainingExample]:
        """Load training examples from a JSONL file."""
        self.logger.info(f"Loading training data from {file_path}")
        
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        example = TrainingExample(
                            instruction=data.get("instruction", ""),
                            input=data.get("input", ""),
                            output=data.get("output", ""),
                            metadata={
                                "source": file_path,
                                "timestamp": data.get("timestamp", ""),
                                "type": self._determine_example_type(data)
                            }
                        )
                        examples.append(example)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing JSON line: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing example: {e}")
                        continue
            
            self.logger.info(f"Successfully loaded {len(examples)} training examples")
            return examples
        
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def _determine_example_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of training example based on its content."""
        instruction = data.get("instruction", "").lower()
        
        if any(word in instruction for word in ["tiktok", "reel", "video"]):
            return "video_content"
        elif any(word in instruction for word in ["caption", "post", "social"]):
            return "social_media"
        elif any(word in instruction for word in ["strategy", "plan", "campaign"]):
            return "marketing_strategy"
        elif any(word in instruction for word in ["script", "write", "content"]):
            return "content_creation"
        else:
            return "general"
    
    def save_examples(self, examples: List[TrainingExample], file_path: str):
        """Save training examples to a JSONL file."""
        self.logger.info(f"Saving {len(examples)} examples to {file_path}")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    data = {
                        "instruction": example.instruction,
                        "input": example.input,
                        "output": example.output,
                        "metadata": example.metadata
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Successfully saved examples to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving examples to {file_path}: {e}")
    
    def merge_examples(self, examples1: List[TrainingExample], examples2: List[TrainingExample]) -> List[TrainingExample]:
        """Merge two lists of training examples, removing duplicates."""
        merged = examples1.copy()
        seen = {(e.instruction, e.input) for e in examples1}
        
        for example in examples2:
            if (example.instruction, example.input) not in seen:
                merged.append(example)
                seen.add((example.instruction, example.input))
        
        return merged
    
    def filter_examples(
        self,
        examples: List[TrainingExample],
        min_length: int = 0,
        max_length: int = None,
        example_type: str = None
    ) -> List[TrainingExample]:
        """Filter training examples based on various criteria."""
        filtered = examples.copy()
        
        # Filter by length
        if min_length > 0:
            filtered = [
                e for e in filtered
                if len(e.instruction) >= min_length and
                len(e.input) >= min_length and
                len(e.output) >= min_length
            ]
        
        if max_length:
            filtered = [
                e for e in filtered
                if len(e.instruction) <= max_length and
                len(e.input) <= max_length and
                len(e.output) <= max_length
            ]
        
        # Filter by type
        if example_type:
            filtered = [
                e for e in filtered
                if e.metadata.get("type") == example_type
            ]
        
        return filtered
    
    def get_example_stats(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Get statistics about the training examples."""
        stats = {
            "total_examples": len(examples),
            "type_distribution": {},
            "avg_instruction_length": 0,
            "avg_input_length": 0,
            "avg_output_length": 0
        }
        
        # Calculate type distribution
        for example in examples:
            example_type = example.metadata.get("type", "unknown")
            stats["type_distribution"][example_type] = stats["type_distribution"].get(example_type, 0) + 1
        
        # Calculate average lengths
        if examples:
            stats["avg_instruction_length"] = sum(len(e.instruction) for e in examples) / len(examples)
            stats["avg_input_length"] = sum(len(e.input) for e in examples) / len(examples)
            stats["avg_output_length"] = sum(len(e.output) for e in examples) / len(examples)
        
        return stats 