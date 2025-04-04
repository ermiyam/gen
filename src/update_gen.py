import argparse
import logging
from pathlib import Path
from datetime import datetime
from feedback_processor import FeedbackProcessor
from retrain import GenRetrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_gen(
    feedback_file: str = "data/feedback/gen_feedback.jsonl",
    training_file: str = "data/training/nexgencreators_training_v1.jsonl",
    model_dir: str = "models/gen_v1",
    base_model: str = "gpt2",
    min_rating: int = 4
):
    """Update Gen with new feedback and retrain."""
    # Step 1: Process feedback
    logger.info("Processing feedback data...")
    processor = FeedbackProcessor(feedback_file)
    processor.process_feedback(min_rating)
    
    # Step 2: Retrain model
    logger.info("Retraining model...")
    retrainer = GenRetrainer(
        base_model=base_model,
        output_dir=model_dir,
        training_file=training_file
    )
    retrainer.train()
    
    logger.info("Gen update completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Update Gen with new feedback and retrain")
    parser.add_argument("--feedback-file", type=str, default="data/feedback/gen_feedback.jsonl", help="Path to feedback data")
    parser.add_argument("--training-file", type=str, default="data/training/nexgencreators_training_v1.jsonl", help="Path to training data")
    parser.add_argument("--model-dir", type=str, default="models/gen_v1", help="Output directory for the trained model")
    parser.add_argument("--base-model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--min-rating", type=int, default=4, help="Minimum rating for feedback to be included in training")
    
    args = parser.parse_args()
    
    update_gen(
        feedback_file=args.feedback_file,
        training_file=args.training_file,
        model_dir=args.model_dir,
        base_model=args.base_model,
        min_rating=args.min_rating
    )

if __name__ == "__main__":
    main() 