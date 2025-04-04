"""
Smart Learning System Setup for Mak
"""

import os
from datetime import datetime
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/setup.log'),
        logging.StreamHandler()
    ]
)

def setup_folders():
    """Create necessary folders for the smart learning system."""
    folders = [
        "data/sources",       # Raw scraped content
        "data/priority",      # High-priority curated learning data
        "data/goals",         # Goals Mak should master
        "logs/learning",      # Continuous training logs
        "scripts/modules",    # Modular learn functions
        "models/checkpoints", # Model checkpoints
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Created folder: {folder}")
    
    return folders

def create_initial_files():
    """Create initial files with example content."""
    files_to_create = {
        "data/priority/priority.txt": """# 🔥 Priority Learning Content

## High-Value Instagram Hooks
- "Wait until you see what happened next..."
- "This one trick changed everything..."
- "You won't believe what I found..."

## Top-Performing CTAs
- "Drop a ❤️ if you agree!"
- "Tag someone who needs to see this!"
- "Save this for later!"

## Best Replies
- "Thanks for the love! 🙌"
- "Glad you found this helpful! 😊"
- "Stay tuned for more tips! 🔥"
""",
        
        "data/goals/goals.md": f"""# 🎯 Mak's Learning Goals ({datetime.now().date()})

## Core Objectives
- Master Instagram hooks and engagement strategies
- Write high-converting CTAs
- Learn persuasive copywriting techniques
- Understand viral content patterns
- Develop emotional triggers
- Create compelling storytelling

## Platform-Specific Goals
- YouTube: Hook writing, video descriptions
- Instagram: Caption writing, engagement
- TikTok: Trend adaptation, viral hooks
""",
        
        "data/sources/README.txt": """📂 Raw Content Sources

This directory stores raw scraped content from:
- YouTube videos and descriptions
- Instagram posts and captions
- TikTok videos and comments

Files are organized by:
- Platform
- Date
- Content type
""",
        
        "logs/learning/README.txt": """📊 Learning Logs

Mak stores continuous learning logs here:
- Training progress
- Content analysis
- Performance metrics
- Error logs
""",
        
        "scripts/modules/README.txt": """🛠️ Custom Modules

Optional functions for:
- Custom scraping
- Content preprocessing
- Data analysis
- Model evaluation
"""
    }
    
    for file_path, content in files_to_create.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Created file: {file_path}")

def main():
    """Main setup function."""
    try:
        # Create folders
        folders = setup_folders()
        
        # Create initial files
        create_initial_files()
        
        # Create status report
        setup_status = pd.DataFrame([
            {"System": "Smart Folder", "Path": folder, "Status": "✅ Ready"} 
            for folder in folders
        ])
        
        # Display status
        print("\nSmart Learning System Setup Status:")
        print(setup_status.to_string(index=False))
        
        logging.info("Smart learning system setup completed successfully")
        
    except Exception as e:
        logging.error(f"Error during setup: {e}")
        raise

if __name__ == "__main__":
    main() 