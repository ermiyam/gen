"""
Process Manager for Mak: Manages the continuous learning process
"""

import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/process_manager.log'),
        logging.StreamHandler()
    ]
)

class ProcessManager:
    def __init__(self):
        self.pid_file = Path("logs/learning_pid.txt")
        self.process = None
        
    def start(self):
        """Start the continuous learning process."""
        try:
            # Create necessary directories
            os.makedirs("data", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            # Start the process
            self.process = subprocess.Popen(
                [sys.executable, "src/continuous_learn.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Save PID
            self.pid_file.write_text(str(self.process.pid))
            
            logging.info(f"Started continuous learning process with PID: {self.process.pid}")
            logging.info("Check logs/continuous_learn.log for progress")
            
        except Exception as e:
            logging.error(f"Error starting process: {e}")
            
    def stop(self):
        """Stop the continuous learning process."""
        try:
            if self.pid_file.exists():
                pid = int(self.pid_file.read_text())
                os.kill(pid, signal.SIGTERM)
                self.pid_file.unlink()
                logging.info(f"Stopped process with PID: {pid}")
            else:
                logging.warning("No process found. PID file does not exist.")
                
        except Exception as e:
            logging.error(f"Error stopping process: {e}")
            
    def status(self):
        """Check the status of the continuous learning process."""
        try:
            if self.pid_file.exists():
                pid = int(self.pid_file.read_text())
                try:
                    process = psutil.Process(pid)
                    logging.info(f"Process is running (PID: {pid})")
                    logging.info(f"Started at: {process.create_time()}")
                    logging.info(f"CPU Usage: {process.cpu_percent()}%")
                    logging.info(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    
                    # Check log file
                    log_file = Path("logs/continuous_learn.log")
                    if log_file.exists():
                        logging.info("\nLast 5 log entries:")
                        for line in log_file.read_text().splitlines()[-5:]:
                            logging.info(line)
                            
                except psutil.NoSuchProcess:
                    logging.warning(f"Process with PID {pid} is not running")
                    self.pid_file.unlink()
                    
            else:
                logging.warning("No process found. PID file does not exist.")
                
        except Exception as e:
            logging.error(f"Error checking status: {e}")

def main():
    manager = ProcessManager()
    
    if len(sys.argv) < 2:
        print("Usage: python process_manager.py [start|stop|status]")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    
    if command == "start":
        manager.start()
    elif command == "stop":
        manager.stop()
    elif command == "status":
        manager.status()
    else:
        print("Invalid command. Use: start, stop, or status")
        sys.exit(1)

if __name__ == "__main__":
    main() 