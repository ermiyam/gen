import os
import time
import logging
from datetime import datetime
import json
from typing import Dict, Any
import schedule
from dataclasses import dataclass
import wandb
from pathlib import Path

from ..ai_model.unified_system import UnifiedSystem, ResponseConfig
from .trainer import TrainingConfig

@dataclass
class LearningConfig:
    check_interval: int = 3600  # Check for new data every hour
    min_feedback_count: int = 100  # Minimum feedback entries to trigger training
    max_training_duration: int = 7200  # Maximum training duration in seconds
    save_interval: int = 3600  # Save model state every hour
    wandb_project: str = "nexgencreators"
    wandb_entity: str = "nexgencreators"
    output_dir: str = "models"
    state_dir: str = "state"

class ContinuousLearningManager:
    def __init__(self, config: LearningConfig):
        self.config = config
        self.unified_system = UnifiedSystem(
            model_name="mistralai/Mistral-7B-v0.1",
            config=ResponseConfig()
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("continuous_learning.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=f"continuous_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Create necessary directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.state_dir, exist_ok=True)
    
    def _should_train(self) -> bool:
        """Check if there's enough new data to trigger training."""
        feedback_count = len(self.unified_system.feedback_system.feedback_history)
        return feedback_count >= self.config.min_feedback_count
    
    def _get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "total_feedback": len(self.unified_system.feedback_system.feedback_history),
            "average_rating": self.unified_system.feedback_system.get_performance_summary()["average_rating"],
            "feedback_distribution": self.unified_system.feedback_system.get_performance_summary()["feedback_distribution"],
            "context_documents": len(self.unified_system.rag_system.knowledge_base["documents"])
        }
    
    def _save_state(self):
        """Save the current state of the system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = os.path.join(self.config.state_dir, f"state_{timestamp}")
        
        # Save unified system state
        self.unified_system.save_state(state_path)
        
        # Save training metrics
        metrics = self._get_training_metrics()
        with open(os.path.join(state_path, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved system state to {state_path}")
    
    def _load_latest_state(self):
        """Load the latest saved state."""
        state_dirs = sorted(Path(self.config.state_dir).glob("state_*"))
        if not state_dirs:
            return
        
        latest_state = str(state_dirs[-1])
        self.unified_system.load_state(latest_state)
        self.logger.info(f"Loaded system state from {latest_state}")
    
    def _run_training_iteration(self):
        """Run a single training iteration."""
        self.logger.info("Starting training iteration")
        
        # Get current metrics
        metrics = self._get_training_metrics()
        wandb.log(metrics)
        
        # Run continuous learning
        start_time = time.time()
        self.unified_system.run_continuous_learning()
        
        # Check if training exceeded maximum duration
        duration = time.time() - start_time
        if duration > self.config.max_training_duration:
            self.logger.warning(f"Training exceeded maximum duration: {duration:.2f}s")
        
        # Save state after training
        self._save_state()
        
        self.logger.info("Completed training iteration")
    
    def _check_and_train(self):
        """Check for new data and run training if needed."""
        self.logger.info("Checking for new training data")
        
        if self._should_train():
            self.logger.info("Starting training due to sufficient new data")
            self._run_training_iteration()
        else:
            self.logger.info("Not enough new data for training")
    
    def run(self):
        """Run the continuous learning loop."""
        self.logger.info("Starting continuous learning manager")
        
        # Load latest state if available
        self._load_latest_state()
        
        # Schedule tasks
        schedule.every(self.config.check_interval).seconds.do(self._check_and_train)
        schedule.every(self.config.save_interval).seconds.do(self._save_state)
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt, shutting down")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    # Create learning configuration
    config = LearningConfig()
    
    # Create and run manager
    manager = ContinuousLearningManager(config)
    manager.run()

if __name__ == "__main__":
    main() 