"""
Continuous Learning Script for Mak: Runs scraper and training process
"""

import os
import logging
import time
import schedule
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_learn.log'),
        logging.StreamHandler()
    ]
)

def run_scraper():
    """Run the content scraper."""
    try:
        logging.info("Starting content scraper...")
        result = subprocess.run(["python", "src/scraper.py"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Content scraper completed successfully")
        else:
            logging.error(f"Content scraper failed: {result.stderr}")
    except Exception as e:
        logging.error(f"Error running scraper: {e}")

def run_training():
    """Run the model training process."""
    try:
        logging.info("Starting model training...")
        result = subprocess.run(["python", "src/learn.py"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Model training completed successfully")
        else:
            logging.error(f"Model training failed: {result.stderr}")
    except Exception as e:
        logging.error(f"Error running training: {e}")

def main():
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Schedule tasks
    schedule.every(6).hours.do(run_scraper)  # Run scraper every 6 hours
    schedule.every(12).hours.do(run_training)  # Run training every 12 hours
    
    # Run immediately on startup
    logging.info("Starting continuous learning process...")
    run_scraper()
    run_training()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 