import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from rich.console import Console
from rich.progress import Progress

class ResponseType(Enum):
    SCRIPT = "script"
    CAPTION = "caption"
    HOOK = "hook"
    AUDIT = "audit"
    STRATEGY = "strategy"
    OTHER = "other"

@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str
    type: ResponseType
    rating: int
    timestamp: str

class RetrainingDataPreparator:
    """Prepares best-rated responses for model retraining."""
    
    def __init__(
        self,
        log_dir: str = "data/response_logs",
        output_dir: str = "data/training",
        min_rating: int = 4,
        min_responses: int = 50
    ):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.min_rating = min_rating
        self.min_responses = min_responses
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize console
        self.console = Console()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"retraining_prep_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_response_logs(self) -> List[Dict[str, Any]]:
        """Load all response logs from JSONL files."""
        responses = []
        
        # Find all JSONL files in log directory
        log_files = list(self.log_dir.glob("*.jsonl"))
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading response logs...", total=len(log_files))
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                response = json.loads(line)
                                responses.append(response)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Invalid JSON in {log_file}")
                                continue
                except Exception as e:
                    self.logger.error(f"Error reading {log_file}: {str(e)}")
                
                progress.advance(task)
        
        return responses
    
    def _filter_best_responses(
        self,
        responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter responses based on rating and usage."""
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(responses)
        
        # Filter responses
        best_responses = df[
            (df['rating'] >= self.min_rating) &
            (df.get('used', True) == True)
        ].to_dict('records')
        
        return best_responses
    
    def _convert_to_training_format(
        self,
        responses: List[Dict[str, Any]]
    ) -> List[TrainingExample]:
        """Convert responses to training format."""
        training_examples = []
        
        for response in responses:
            # Determine response type
            response_type = ResponseType.OTHER
            if "type" in response:
                try:
                    response_type = ResponseType(response["type"])
                except ValueError:
                    pass
            
            # Create training example
            example = TrainingExample(
                instruction=response["prompt"],
                input="",  # We don't use input for now
                output=response["response"],
                type=response_type,
                rating=response["rating"],
                timestamp=response["timestamp"]
            )
            
            training_examples.append(example)
        
        return training_examples
    
    def _save_training_data(
        self,
        examples: List[TrainingExample],
        version: str = None
    ):
        """Save training examples to JSONL file."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = self.output_dir / f"nexgencreators_training_v{version}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                # Convert to training format
                training_data = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output
                }
                
                # Save to file
                f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(examples)} training examples to {output_file}")
    
    def prepare_data(self, version: str = None) -> bool:
        """Prepare training data from best-rated responses."""
        try:
            # Load response logs
            self.console.print("[cyan]Loading response logs...")
            responses = self._load_response_logs()
            
            if not responses:
                self.console.print("[red]No response logs found!")
                return False
            
            # Filter best responses
            self.console.print("[cyan]Filtering best responses...")
            best_responses = self._filter_best_responses(responses)
            
            if len(best_responses) < self.min_responses:
                self.console.print(
                    f"[yellow]Not enough best responses ({len(best_responses)}/{self.min_responses})"
                )
                return False
            
            # Convert to training format
            self.console.print("[cyan]Converting to training format...")
            training_examples = self._convert_to_training_format(best_responses)
            
            # Save training data
            self.console.print("[cyan]Saving training data...")
            self._save_training_data(training_examples, version)
            
            # Print summary
            self.console.print("\n[green]Training data preparation complete!")
            self.console.print(f"Total responses: {len(responses)}")
            self.console.print(f"Best responses: {len(best_responses)}")
            self.console.print(f"Training examples: {len(training_examples)}")
            
            # Print type distribution
            type_counts = {}
            for example in training_examples:
                type_counts[example.type.value] = type_counts.get(example.type.value, 0) + 1
            
            self.console.print("\n[cyan]Response type distribution:")
            for type_name, count in type_counts.items():
                self.console.print(f"{type_name}: {count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}")
            return False

def main():
    """Main function to prepare retraining data."""
    preparator = RetrainingDataPreparator()
    preparator.prepare_data()

if __name__ == "__main__":
    main() 