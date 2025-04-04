import json
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

class ValidationError(Enum):
    MISSING_FIELD = "missing_field"
    INVALID_TYPE = "invalid_type"
    INVALID_METRICS = "invalid_metrics"
    INVALID_FORMAT = "invalid_format"
    INCONSISTENT_DATA = "inconsistent_data"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    stats: Dict[str, Any]

class ExampleValidator:
    def __init__(self):
        self.required_fields = ["instruction", "input", "output", "metadata"]
        self.metadata_fields = ["type", "platform", "industry", "goal", "audience"]
        self.metric_ranges = {
            "engagement_rate": (0, 1),
            "conversion_rate": (0, 1),
            "roi": (0, float('inf')),
            "cpa": (0, float('inf')),
            "ltv": (0, float('inf')),
            "reach": (0, float('inf')),
            "impressions": (0, float('inf')),
            "clicks": (0, float('inf')),
            "ctr": (0, 1),
            "bounce_rate": (0, 1),
            "time_on_page": (0, float('inf')),
            "social_shares": (0, float('inf')),
            "comments": (0, float('inf')),
            "likes": (0, float('inf')),
            "followers_gained": (0, float('inf')),
            "email_open_rate": (0, 1),
            "email_click_rate": (0, 1),
            "email_unsubscribe_rate": (0, 1),
            "influencer_engagement": (0, 1),
            "brand_sentiment": (0, 1),
            "content_quality_score": (0, 1)
        }
    
    def validate_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in example:
                errors.append({
                    "error": ValidationError.MISSING_FIELD.value,
                    "field": field,
                    "message": f"Missing required field: {field}"
                })
        
        # Validate metadata
        if "metadata" in example:
            metadata = example["metadata"]
            
            # Check metadata fields
            if "type" not in metadata:
                errors.append({
                    "error": ValidationError.MISSING_FIELD.value,
                    "field": "metadata.type",
                    "message": "Missing example type in metadata"
                })
            
            # Validate metrics if present
            if "metrics" in metadata:
                for metric, value in metadata["metrics"].items():
                    if metric in self.metric_ranges:
                        min_val, max_val = self.metric_ranges[metric]
                        if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                            errors.append({
                                "error": ValidationError.INVALID_METRICS.value,
                                "field": f"metadata.metrics.{metric}",
                                "message": f"Invalid value for {metric}: {value}"
                            })
        
        # Validate content length
        if "instruction" in example and len(example["instruction"]) < 10:
            errors.append({
                "error": ValidationError.INVALID_FORMAT.value,
                "field": "instruction",
                "message": "Instruction too short"
            })
        
        if "output" in example and len(example["output"]) < 50:
            errors.append({
                "error": ValidationError.INVALID_FORMAT.value,
                "field": "output",
                "message": "Output too short"
            })
        
        return errors
    
    def validate_examples_file(self, file_path: Path) -> ValidationResult:
        errors = []
        warnings = []
        stats = {
            "total_examples": 0,
            "valid_examples": 0,
            "invalid_examples": 0,
            "example_types": {},
            "industries": {},
            "platforms": {}
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        example = json.loads(line)
                        stats["total_examples"] += 1
                        
                        # Validate example
                        example_errors = self.validate_example(example)
                        if example_errors:
                            errors.extend([{**error, "line": line_num} for error in example_errors])
                            stats["invalid_examples"] += 1
                        else:
                            stats["valid_examples"] += 1
                        
                        # Update statistics
                        if "metadata" in example:
                            metadata = example["metadata"]
                            if "type" in metadata:
                                stats["example_types"][metadata["type"]] = stats["example_types"].get(metadata["type"], 0) + 1
                            if "industry" in metadata:
                                stats["industries"][metadata["industry"]] = stats["industries"].get(metadata["industry"], 0) + 1
                            if "platform" in metadata:
                                stats["platforms"][metadata["platform"]] = stats["platforms"].get(metadata["platform"], 0) + 1
                    
                    except json.JSONDecodeError:
                        errors.append({
                            "error": ValidationError.INVALID_FORMAT.value,
                            "line": line_num,
                            "message": "Invalid JSON format"
                        })
                        stats["invalid_examples"] += 1
        
        except Exception as e:
            errors.append({
                "error": ValidationError.INVALID_FORMAT.value,
                "message": f"Error reading file: {str(e)}"
            })
        
        # Add warnings for potential issues
        if stats["total_examples"] > 0:
            valid_percentage = (stats["valid_examples"] / stats["total_examples"]) * 100
            if valid_percentage < 90:
                warnings.append({
                    "message": f"Low validation success rate: {valid_percentage:.1f}%"
                })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

def main():
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"validation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Initialize validator
    validator = ExampleValidator()
    
    # Find the most recent examples file
    data_dir = Path("data")
    example_files = list(data_dir.glob("specialized_examples_*.jsonl"))
    if not example_files:
        logger.error("No example files found in data directory")
        return
    
    latest_file = max(example_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Validating file: {latest_file}")
    
    # Validate examples
    result = validator.validate_examples_file(latest_file)
    
    # Log results
    logger.info(f"Validation complete. Found {len(result.errors)} errors and {len(result.warnings)} warnings")
    logger.info(f"Total examples: {result.stats['total_examples']}")
    logger.info(f"Valid examples: {result.stats['valid_examples']}")
    logger.info(f"Invalid examples: {result.stats['invalid_examples']}")
    
    if result.errors:
        logger.error("Validation errors:")
        for error in result.errors:
            logger.error(f"- Line {error.get('line', 'N/A')}: {error['message']}")
    
    if result.warnings:
        logger.warning("Validation warnings:")
        for warning in result.warnings:
            logger.warning(f"- {warning['message']}")
    
    logger.info("Example type distribution:")
    for type_name, count in result.stats["example_types"].items():
        logger.info(f"- {type_name}: {count}")

if __name__ == "__main__":
    main() 