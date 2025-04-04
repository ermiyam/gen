import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenRetrainer:
    def __init__(
        self,
        base_model: str = "gpt2",
        output_dir: str = "models/gen_v1",
        training_file: str = "data/training/nexgencreators_training_v1.jsonl"
    ):
        """Initialize the retrainer."""
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.training_file = Path(training_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_data(self) -> List[Dict]:
        """Load training data from JSONL file."""
        if not self.training_file.exists():
            raise FileNotFoundError(f"Training file not found: {self.training_file}")
            
        data = []
        with open(self.training_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def create_prompt(self, example: Dict) -> str:
        """Create a prompt from an example."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            return f"[INSTRUCTION] {instruction}\n[INPUT] {input_text}\n[RESPONSE] {output}\n[END]"
        else:
            return f"[INSTRUCTION] {instruction}\n[RESPONSE] {output}\n[END]"
    
    def preprocess_function(self, examples, tokenizer):
        """Preprocess examples for training."""
        prompts = [self.create_prompt(ex) for ex in examples]
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = tokenized.input_ids.clone()
        labels[tokenized.attention_mask == 0] = -100  # Mask padding tokens
        
        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }
    
    def train(self):
        """Run the training process."""
        logger.info(f"Loading base model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load and prepare training data
        logger.info("Loading training data...")
        train_data = self.load_training_data()
        train_dataset = Dataset.from_dict({"text": [self.create_prompt(ex) for ex in train_data]})
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            save_total_limit=2,
            fp16=True,
            max_grad_norm=0.5,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda examples: self.preprocess_function(examples, tokenizer)
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Retrain Gen on feedback data")
    parser.add_argument("--base-model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="models/gen_v1", help="Output directory for the trained model")
    parser.add_argument("--training-file", type=str, default="data/training/nexgencreators_training_v1.jsonl", help="Path to training data")
    
    args = parser.parse_args()
    
    retrainer = GenRetrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        training_file=args.training_file
    )
    
    retrainer.train()

if __name__ == "__main__":
    main() 