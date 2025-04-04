"""
Mak Auto-Learning Engine:
- Scrapes new content every 30 seconds
- Trains on new data every 10 seconds
- Dynamically prioritizes and filters better quality training data
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/learning_loop.log'),
        logging.StreamHandler()
    ]
)

class LearningLoop:
    def __init__(self):
        self.scraper_path = "src/scraper.py"
        self.learner_path = "src/learn.py"
        self.data_path = "data/train.txt"
        self.metrics_path = "logs/learning_metrics.json"
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize metrics
        self.metrics = self.load_metrics()
        
    def load_metrics(self) -> Dict:
        """Load learning metrics from file."""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")
        return {
            "total_training_examples": 0,
            "last_scrape_time": None,
            "last_train_time": None,
            "platform_performance": {},
            "quality_scores": []
        }
    
    def save_metrics(self):
        """Save current metrics to file."""
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
    
    def run_scraper(self):
        """Run the content scraper."""
        try:
            logging.info("Starting content scrape...")
            result = subprocess.run(
                ["python", self.scraper_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info("Scrape completed successfully")
                self.metrics["last_scrape_time"] = datetime.now().isoformat()
                self.save_metrics()
            else:
                logging.error(f"Scrape failed: {result.stderr}")
                
        except Exception as e:
            logging.error(f"Error running scraper: {e}")
    
    def run_trainer(self):
        """Run the model trainer."""
        try:
            logging.info("Starting training cycle...")
            result = subprocess.run(
                ["python", self.learner_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info("Training completed successfully")
                self.metrics["last_train_time"] = datetime.now().isoformat()
                
                # Update metrics from training output
                if "eval_loss" in result.stdout:
                    self.metrics["last_eval_loss"] = float(
                        result.stdout.split("eval_loss:")[1].split()[0]
                    )
                
                self.save_metrics()
            else:
                logging.error(f"Training failed: {result.stderr}")
                
        except Exception as e:
            logging.error(f"Error running trainer: {e}")
    
    def check_data_quality(self) -> bool:
        """Check if new data meets quality thresholds."""
        try:
            if not os.path.exists(self.data_path):
                return False
                
            # Get file size and modification time
            file_stats = os.stat(self.data_path)
            size_mb = file_stats.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            # Check if file has grown significantly
            if size_mb < 1:  # Less than 1MB
                return False
                
            # Check if file was modified recently
            if (datetime.now() - mod_time).total_seconds() > 3600:  # 1 hour
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking data quality: {e}")
            return False
    
    def run(self):
        """Main learning loop."""
        logging.info("🚀 Starting Mak's learning loop")
        
        last_scrape = 0
        last_train = 0
        scrape_interval = 30  # seconds
        train_interval = 10   # seconds
        
        while True:
            current_time = time.time()
            
            # Scrape new content every 30 seconds
            if current_time - last_scrape >= scrape_interval:
                self.run_scraper()
                last_scrape = current_time
            
            # Train on new data every 10 seconds if quality check passes
            if current_time - last_train >= train_interval:
                if self.check_data_quality():
                    self.run_trainer()
                else:
                    logging.info("Skipping training - insufficient new data")
                last_train = current_time
            
            # Sleep for a short time to prevent CPU overload
            time.sleep(1)

if __name__ == "__main__":
    loop = LearningLoop()
    loop.run() 