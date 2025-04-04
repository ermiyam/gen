"""
Priority Content Feeder for Mak: Automatically identifies and saves high-value content
"""

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/priority_feeder.log'),
        logging.StreamHandler()
    ]
)

class PriorityFeeder:
    def __init__(self):
        self.priority_file = Path("data/priority/priority.txt")
        self.sources_dir = Path("data/sources")
        self.patterns = {
            "hooks": [
                r"wait until you see",
                r"you won't believe",
                r"this changed everything",
                r"here's what happened",
                r"before you scroll",
                r"stop scrolling",
                r"this is important",
                r"you need to see this",
                r"game changer",
                r"life changing",
                r"serious about growing",
                r"makes people buy emotionally"
            ],
            "ctas": [
                r"save this",
                r"tag someone",
                r"drop a \w+",
                r"comment \w+",
                r"share this",
                r"follow for more",
                r"like if you",
                r"double tap if",
                r"swipe up",
                r"link in bio",
                r"comment 'READY'",
                r"building your first"
            ],
            "replies": [
                r"thanks for the \w+",
                r"glad you \w+",
                r"stay tuned",
                r"more coming soon",
                r"appreciate the \w+",
                r"means a lot",
                r"love the \w+",
                r"great question",
                r"exactly right",
                r"spot on",
                r"more AI tools",
                r"coming your way"
            ],
            "value_prompts": [
                r"what makes people buy",
                r"emotional triggers",
                r"conversion secrets",
                r"growth hacks",
                r"brand building",
                r"million-dollar funnel",
                r"AI tools and hacks"
            ]
        }
        
    def find_high_value_content(self, text: str) -> Dict[str, List[str]]:
        """Find high-value content patterns in text."""
        found = {
            "hooks": [],
            "ctas": [],
            "replies": [],
            "value_prompts": []
        }
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    # Get the full sentence containing the match
                    start = max(0, text.rfind(".", 0, match.start()) + 1)
                    end = text.find(".", match.end())
                    if end == -1:
                        end = len(text)
                    sentence = text[start:end].strip()
                    if sentence not in found[category]:
                        found[category].append(sentence)
                        
        return found
    
    def process_source_file(self, file_path: Path) -> Optional[Dict[str, List[str]]]:
        """Process a single source file for high-value content."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            return self.find_high_value_content(content)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None
            
    def update_priority_file(self, new_content: Dict[str, List[str]]):
        """Update the priority file with new high-value content."""
        try:
            # Read existing content
            existing_content = {}
            if self.priority_file.exists():
                with open(self.priority_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Parse existing sections
                for category in ["hooks", "ctas", "replies", "value_prompts"]:
                    section_pattern = f"## {category.title()}\n(.*?)(?=##|\Z)"
                    match = re.search(section_pattern, content, re.DOTALL)
                    if match:
                        items = [line.strip("- ") for line in match.group(1).split("\n") if line.strip()]
                        existing_content[category] = items
                    else:
                        existing_content[category] = []
                        
            # Merge new content
            for category, items in new_content.items():
                if category not in existing_content:
                    existing_content[category] = []
                existing_content[category].extend(items)
                # Remove duplicates while preserving order
                existing_content[category] = list(dict.fromkeys(existing_content[category]))
                
            # Write updated content
            with open(self.priority_file, "w", encoding="utf-8") as f:
                f.write("# 🔥 Priority Learning Content\n\n")
                for category, items in existing_content.items():
                    f.write(f"## {category.title()}\n")
                    for item in items:
                        f.write(f"- {item}\n")
                    f.write("\n")
                    
            logging.info(f"Updated priority file with {sum(len(items) for items in new_content.values())} new items")
            
        except Exception as e:
            logging.error(f"Error updating priority file: {e}")
            
    def run(self):
        """Main function to process all source files and update priority content."""
        try:
            # Process all source files
            total_found = {
                "hooks": [],
                "ctas": [],
                "replies": [],
                "value_prompts": []
            }
            
            for file_path in self.sources_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in [".txt", ".json", ".md"]:
                    found = self.process_source_file(file_path)
                    if found:
                        for category, items in found.items():
                            total_found[category].extend(items)
                            
            # Update priority file
            self.update_priority_file(total_found)
            
            # Log summary
            for category, items in total_found.items():
                logging.info(f"Found {len(items)} new {category}")
                
        except Exception as e:
            logging.error(f"Error in priority feeder: {e}")

def main():
    feeder = PriorityFeeder()
    feeder.run()

if __name__ == "__main__":
    main() 