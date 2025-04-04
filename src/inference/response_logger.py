import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ResponseRating(Enum):
    GOOD = "✅"
    BAD = "❌"
    STAR_1 = "⭐"
    STAR_2 = "⭐⭐"
    STAR_3 = "⭐⭐⭐"
    STAR_4 = "⭐⭐⭐⭐"
    STAR_5 = "⭐⭐⭐⭐⭐"

@dataclass
class ResponseLog:
    timestamp: str
    prompt: str
    response: str
    rating: Optional[str] = None
    stars: Optional[int] = None
    feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ResponseLogger:
    """Logger for tracking and rating Gen's responses."""
    
    def __init__(self, log_dir: str = "data/response_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize log file
        self.log_file = self.log_dir / f"gen_responses_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Load existing logs
        self.logs: list[ResponseLog] = []
        self._load_existing_logs()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"response_logger_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_existing_logs(self):
        """Load existing logs from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        self.logs.append(ResponseLog(**data))
                self.logger.info(f"Loaded {len(self.logs)} existing logs")
            except Exception as e:
                self.logger.error(f"Error loading logs: {str(e)}")
    
    def log_response(
        self,
        prompt: str,
        response: str,
        rating: Optional[ResponseRating] = None,
        stars: Optional[int] = None,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ResponseLog:
        """Log a response with optional rating and feedback."""
        log_entry = ResponseLog(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            rating=rating.value if rating else None,
            stars=stars,
            feedback=feedback,
            metadata=metadata
        )
        
        # Add to memory
        self.logs.append(log_entry)
        
        # Save to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(log_entry), ensure_ascii=False) + '\n')
            self.logger.info("Response logged successfully")
        except Exception as e:
            self.logger.error(f"Error saving log: {str(e)}")
        
        return log_entry
    
    def get_best_responses(self, min_stars: int = 4) -> list[ResponseLog]:
        """Get responses rated above minimum stars."""
        return [log for log in self.logs if log.stars and log.stars >= min_stars]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged responses."""
        total = len(self.logs)
        rated = len([log for log in self.logs if log.rating])
        avg_stars = sum(log.stars or 0 for log in self.logs) / total if total > 0 else 0
        
        return {
            "total_responses": total,
            "rated_responses": rated,
            "average_rating": round(avg_stars, 2),
            "best_responses": len(self.get_best_responses())
        }
    
    def export_for_training(self, output_file: Optional[str] = None) -> str:
        """Export best responses for training."""
        if output_file is None:
            output_file = self.log_dir / f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        best_responses = self.get_best_responses()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for log in best_responses:
                    training_example = {
                        "instruction": log.prompt,
                        "input": "",
                        "output": log.response,
                        "metadata": {
                            "rating": log.rating,
                            "stars": log.stars,
                            "feedback": log.feedback,
                            **(log.metadata or {})
                        }
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Exported {len(best_responses)} responses for training")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Error exporting training data: {str(e)}")
            raise 