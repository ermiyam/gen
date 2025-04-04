import os
import logging
from datetime import datetime
import wandb
from pathlib import Path
import json

from .trainer import MarketingAITrainer, TrainingConfig, TrainingPhase
from .data_loader import DataLoader
from .generate_examples import MarketingExampleGenerator
from .validate_examples import ExampleValidator

def setup_logging():
    """Setup logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_specialized_examples(logger):
    """Generate and save specialized marketing examples."""
    logger.info("Generating specialized marketing examples")
    
    # Initialize generator and validator
    generator = MarketingExampleGenerator()
    validator = ExampleValidator()
    
    # Generate examples
    all_examples = []
    all_examples.extend(generator.generate_video_content_examples())
    all_examples.extend(generator.generate_social_media_examples())
    all_examples.extend(generator.generate_marketing_strategy_examples())
    all_examples.extend(generator.generate_content_creation_examples())
    all_examples.extend(generator.generate_influencer_marketing_examples())
    all_examples.extend(generator.generate_email_campaign_examples())
    
    # Save examples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("data") / f"specialized_examples_{timestamp}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Generated {len(all_examples)} specialized marketing examples")
    logger.info(f"Saved to {output_file}")
    
    # Validate examples
    validation_result = validator.validate_examples_file(output_file)
    
    if not validation_result.is_valid:
        logger.warning("Validation found issues in generated examples:")
        for error in validation_result.errors:
            logger.warning(f"- {error['message']}")
        for warning in validation_result.warnings:
            logger.warning(f"- {warning['message']}")
    else:
        logger.info("All examples validated successfully")
    
    logger.info("Example type distribution:")
    for type_name, count in validation_result.stats["example_types"].items():
        logger.info(f"- {type_name}: {count}")
    
    return output_file

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting initial training process")
    
    # Create training configuration
    config = TrainingConfig(
        model_name="mistralai/Mistral-7B-v0.1",
        output_dir="models",
        data_dir="data",
        num_train_epochs=5,  # Increased epochs for better learning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        max_seq_length=2048,
        load_in_8bit=True,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        min_example_length=10,
        max_example_length=2048
    )
    
    # Initialize data loader
    data_loader = DataLoader(data_dir=config.data_dir)
    
    try:
        # Generate and validate specialized examples
        specialized_file = generate_specialized_examples(logger)
        
        # Load and analyze training data
        training_file = "c:/Users/ermiy/Downloads/nexgencreators_dataset_20250401-011745.jsonl"
        examples = data_loader.load_jsonl(training_file)
        
        # Load specialized examples
        specialized_examples = data_loader.load_jsonl(str(specialized_file))
        
        # Merge examples
        all_examples = data_loader.merge_examples(examples, specialized_examples)
        
        # Get and log dataset statistics
        stats = data_loader.get_example_stats(all_examples)
        logger.info(f"Combined dataset statistics: {stats}")
        
        # Initialize trainer
        trainer = MarketingAITrainer(
            config=config,
            feedback_system=None,  # We'll use these later
            rag_system=None,
            tools_system=None
        )
        
        # Run instruction tuning phase
        logger.info("Starting instruction tuning phase")
        trainer.train(
            training_file=training_file,
            phase=TrainingPhase.INSTRUCTION_TUNING
        )
        
        # Save the final model
        trainer.save_model(TrainingPhase.INSTRUCTION_TUNING)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 