import os
import logging
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from datasets import load_dataset
import wandb
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
from rouge_score import rouge_scorer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from src.config.api_keys import api_keys

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    model_path: str = "models/mistral-gen"
    data_file: str = "data/nexgencreators_dataset.jsonl"
    num_samples: int = 100
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    wandb_project: str = "mistral-gen-evaluation"
    wandb_entity: str = "nexgencreators"
    wandb_name: str = f"eval-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"model_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def format_instruction(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Format instruction for evaluation."""
    text = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST]"
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores for generated text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
        'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
        'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
    }
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for metric in scores:
            scores[metric]['precision'] += result[metric].precision
            scores[metric]['recall'] += result[metric].recall
            scores[metric]['fmeasure'] += result[metric].fmeasure
    
    # Average scores
    num_samples = len(predictions)
    for metric in scores:
        for key in scores[metric]:
            scores[metric][key] /= num_samples
    
    return scores

def create_evaluation_dashboard(metrics: Dict[str, float]):
    """Create a rich dashboard for evaluation results."""
    console = Console()
    
    # Create metrics table
    metrics_table = Table(title="Evaluation Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                metrics_table.add_row(f"{key}_{subkey}", f"{subvalue:.4f}")
        else:
            metrics_table.add_row(key, f"{value:.4f}")
    
    # Display table
    console.print(metrics_table)

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model evaluation")
    
    # Create evaluation configuration
    config = EvaluationConfig()
    
    try:
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
            config=config.__dict__
        )
        
        # Set Mistral AI API key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = api_keys.mistral_key
        
        # Load tokenizer and model
        logger.info(f"Loading model from {config.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load evaluation dataset
        logger.info(f"Loading evaluation dataset from {config.data_file}")
        dataset = load_dataset("json", data_files=config.data_file)
        
        # Sample evaluation data
        eval_data = dataset["train"].shuffle(seed=42).select(range(config.num_samples))
        
        # Format evaluation data
        logger.info("Formatting evaluation data")
        tokenized_data = eval_data.map(
            lambda x: format_instruction(x, tokenizer),
            batched=False,
            remove_columns=eval_data.column_names
        )
        
        # Setup generation config
        generation_config = GenerationConfig(
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Generate predictions
        logger.info("Generating predictions")
        predictions = []
        references = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=Console()
        ) as progress:
            eval_task = progress.add_task("Evaluating...", total=len(tokenized_data))
            
            for item in tokenized_data:
                # Generate prediction
                inputs = torch.tensor([item["input_ids"]]).to(model.device)
                outputs = model.generate(
                    inputs,
                    generation_config=generation_config
                )
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part
                prediction = prediction.split("[/INST]")[-1].strip()
                
                predictions.append(prediction)
                references.append(item["output"])
                
                progress.update(eval_task, advance=1)
        
        # Compute metrics
        logger.info("Computing evaluation metrics")
        rouge_scores = compute_rouge_scores(predictions, references)
        
        # Log metrics to wandb
        wandb.log(rouge_scores)
        
        # Create and display dashboard
        create_evaluation_dashboard(rouge_scores)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 