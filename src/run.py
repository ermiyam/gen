import os
import sys
import logging
import subprocess
from pathlib import Path
import time
import signal
import atexit
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler()
    ]
)

class MakLauncher:
    def __init__(self):
        self.processes = []
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories."""
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("models/latest").mkdir(parents=True, exist_ok=True)
        Path("data/knowledge").mkdir(parents=True, exist_ok=True)
        
        # Create knowledge subdirectories if they don't exist
        for subdir in ["books", "youtube", "reels", "copywriting", "frameworks"]:
            Path(f"data/knowledge/{subdir}").mkdir(parents=True, exist_ok=True)
        
    def cleanup(self):
        """Cleanup function to ensure all processes are terminated."""
        logging.info("Cleaning up processes...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
                    
    def run_training(self):
        """Run the training process."""
        try:
            logging.info("Starting training process...")
            
            # Check if we have enough knowledge data
            knowledge_files = list(Path("data/knowledge").rglob("*.txt"))
            if not knowledge_files:
                logging.warning("No knowledge files found. Please add training data to data/knowledge/")
                return False
                
            train_process = subprocess.Popen(
                [sys.executable, "src/learn.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(train_process)
            
            # Monitor training progress
            while True:
                output = train_process.stdout.readline()
                if output == '' and train_process.poll() is not None:
                    break
                if output:
                    logging.info(output.strip())
                    
            # Check training result
            if train_process.returncode != 0:
                logging.error("Training failed")
                return False
                
            logging.info("Training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            return False
            
    def run_chat_interface(self):
        """Run the chat interface."""
        try:
            logging.info("Starting chat interface...")
            chat_process = subprocess.Popen(
                [sys.executable, "src/chat_interface.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(chat_process)
            
            # Monitor the process
            while True:
                output = chat_process.stdout.readline()
                if output == '' and chat_process.poll() is not None:
                    break
                if output:
                    logging.info(output.strip())
                    
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Error running chat interface: {str(e)}")
            return False
            
    def backup_knowledge(self):
        """Create a backup of the knowledge base."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"backups/knowledge_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy knowledge files
            knowledge_dir = Path("data/knowledge")
            for file in knowledge_dir.rglob("*.txt"):
                relative_path = file.relative_to(knowledge_dir)
                target_path = backup_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(file.read_text())
                
            logging.info(f"Knowledge base backed up to {backup_dir}")
            
        except Exception as e:
            logging.error(f"Error backing up knowledge: {str(e)}")
            
    def run(self):
        """Main function to run the entire system."""
        try:
            # Register cleanup handler
            atexit.register(self.cleanup)
            
            # Backup knowledge base
            self.backup_knowledge()
            
            # Run training
            if not self.run_training():
                logging.error("Failed to complete training")
                return
                
            # Run chat interface
            self.run_chat_interface()
            
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
        except Exception as e:
            logging.error(f"Error in main process: {str(e)}")
        finally:
            self.cleanup()

def main():
    """Entry point for the application."""
    launcher = MakLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 