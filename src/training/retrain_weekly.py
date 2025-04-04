import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import subprocess
from rich.console import Console
from rich.progress import Progress

class WeeklyRetrainer:
    """Automates weekly retraining of Gen using best-rated responses."""
    
    def __init__(
        self,
        model_path: str = "models/mistral-gen",
        log_dir: str = "data/response_logs",
        min_responses: int = 50,
        min_stars: int = 4
    ):
        self.model_path = Path(model_path)
        self.log_dir = Path(log_dir)
        self.min_responses = min_responses
        self.min_stars = min_stars
        
        # Setup logging
        self._setup_logging()
        
        # Initialize console
        self.console = Console()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"retraining_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_best_responses(self) -> Optional[str]:
        """Get best responses from logs and prepare training data."""
        try:
            # Run data preparation script
            self.console.print("[cyan]Preparing training data...")
            result = subprocess.run(
                ["python", "-m", "src.training.prepare_retraining_data"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Error preparing data: {result.stderr}")
                return None
            
            # Get latest training file
            training_dir = Path("data/training")
            training_files = list(training_dir.glob("nexgencreators_training_v*.jsonl"))
            
            if not training_files:
                self.logger.error("No training files found")
                return None
            
            # Return latest file
            return str(max(training_files, key=lambda x: x.stat().st_mtime))
            
        except Exception as e:
            self.logger.error(f"Error getting best responses: {str(e)}")
            return None
    
    def _backup_model(self):
        """Create backup of current model."""
        try:
            # Create backup directory
            backup_dir = self.model_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"gen_backup_{timestamp}"
            
            self.console.print("[cyan]Creating model backup...")
            subprocess.run(
                ["cp", "-r", str(self.model_path), str(backup_path)],
                check=True
            )
            
            self.logger.info(f"Created backup at {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up model: {str(e)}")
            return False
    
    def retrain(self) -> bool:
        """Run the retraining process."""
        try:
            # Get best responses
            training_file = self._get_best_responses()
            if not training_file:
                self.console.print("[red]Failed to prepare training data")
                return False
            
            # Backup current model
            if not self._backup_model():
                self.console.print("[red]Failed to backup model")
                return False
            
            # Run training script
            self.console.print("[cyan]Starting model retraining...")
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.training.train_mistral",
                    "--data_file",
                    training_file,
                    "--output_dir",
                    str(self.model_path),
                    "--model_name",
                    "mistralai/Mistral-7B-Instruct-v0.2"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Training failed: {result.stderr}")
                self.console.print("[red]Training failed!")
                return False
            
            self.console.print("[green]Retraining completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}")
            return False

def main():
    """Main function to run weekly retraining."""
    retrainer = WeeklyRetrainer()
    retrainer.retrain()

if __name__ == "__main__":
    main() 