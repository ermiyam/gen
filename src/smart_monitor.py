"""
Smart Content Monitoring System for Mak: Continuously monitors and processes high-value marketing content
"""

import os
import json
import time
import logging
import requests
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from priority_feeder import PriorityFeeder
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/smart_monitoring.log'),
        logging.StreamHandler()
    ]
)

class SmartMonitor:
    def __init__(self):
        self.config_path = Path("config/sources.json")
        self.training_data_dir = Path("training_data")
        self.smart_feed_dir = Path("smart_feed_sources")
        self.feeder = PriorityFeeder()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.marketing_patterns = {
            "scarcity": [
                r"limited time",
                r"only \d+ spots left",
                r"exclusive offer",
                r"while supplies last",
                r"last chance"
            ],
            "psychology": [
                r"buyer psychology",
                r"emotional triggers",
                r"decision making",
                r"behavioral economics",
                r"cognitive biases"
            ],
            "frameworks": [
                r"PAS framework",
                r"AIDA model",
                r"storytelling framework",
                r"marketing funnel",
                r"conversion framework"
            ],
            "engagement": [
                r"community building",
                r"audience retention",
                r"content loops",
                r"engagement strategies",
                r"brand loyalty"
            ]
        }
        
    def load_config(self) -> Dict:
        """Load monitoring configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {"sources": []}
            
    def extract_content(self, url: str) -> Optional[str]:
        """Extract content from a given URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = ""
            for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3']):
                content += paragraph.get_text() + "\n"
                
            return content
        except Exception as e:
            logging.error(f"Error extracting content from {url}: {e}")
            return None
            
    def find_marketing_ideas(self, content: str) -> List[str]:
        """Find marketing-specific ideas in content."""
        ideas = []
        for category, patterns in self.marketing_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content.lower())
                for match in matches:
                    # Get the full sentence containing the match
                    start = max(0, content.rfind(".", 0, match.start()) + 1)
                    end = content.find(".", match.end())
                    if end == -1:
                        end = len(content)
                    sentence = content[start:end].strip()
                    if sentence not in ideas:
                        ideas.append(sentence)
        return ideas
            
    def format_training_data(self, content: str, source: str, ideas: List[str]) -> str:
        """Format content into training data format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ideas_str = "\n".join([f"- {idea}" for idea in ideas])
        return f"""
### Input: {content}
### Marketing Ideas:
{ideas_str}
### Response: [High-value content from {source} at {timestamp}]
"""
        
    def save_training_data(self, data: str):
        """Save formatted training data to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = self.training_data_dir / f"training_{timestamp}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
                
            logging.info(f"Saved training data to {file_path}")
        except Exception as e:
            logging.error(f"Error saving training data: {e}")
            
    def process_source(self, url: str):
        """Process a single source URL."""
        try:
            content = self.extract_content(url)
            if content:
                # Find marketing ideas
                ideas = self.find_marketing_ideas(content)
                
                # Feed content to priority system
                self.feeder.process_source_file(Path(url))
                
                # Format and save training data
                training_data = self.format_training_data(content, url, ideas)
                self.save_training_data(training_data)
                
                logging.info(f"Successfully processed {url} and found {len(ideas)} marketing ideas")
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            
    def run(self, interval: int = 30):
        """Run the monitoring system continuously."""
        logging.info("Starting Smart Content Monitoring System...")
        
        while True:
            try:
                config = self.load_config()
                for source in config.get("sources", []):
                    self.process_source(source)
                    
                logging.info(f"Completed monitoring cycle. Next cycle in {interval} seconds...")
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring cycle: {e}")
                time.sleep(interval)

def main():
    monitor = SmartMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 