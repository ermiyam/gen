import argparse
import json
import logging
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(data_file: str) -> List[Dict]:
    """Load training data from a JSONL file."""
    logger.info("Loading dataset...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Only filter out examples with missing fields
    filtered_data = []
    for example in data:
        if all(key in example for key in ["instruction", "output"]):
            filtered_data.append(example)
    
    logger.info(f"Loaded {len(filtered_data)} examples after filtering")
    return filtered_data

def create_prompt(example: Dict) -> str:
    """Create a prompt from an example."""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    return f"[INSTRUCTION] {instruction}\n[RESPONSE] {output}\n[END]"

def preprocess_function(examples, tokenizer):
    """Preprocess the examples for training."""
    prompts = [create_prompt(ex) for ex in examples]
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,  # Increased max length
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    logger.info("Loading model and tokenizer from gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",  # Using larger GPT-2 model
        torch_dtype=torch.float32,
        device_map=None  # Force CPU
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    logger.info("Setting up LoRA configuration")
    config = LoraConfig(
        r=32,  # Increased rank for better adaptation
        lora_alpha=64,  # Increased alpha for stronger adaptation
        target_modules=["c_attn", "c_proj", "wte", "wpe"],  # Added more target modules
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    # Load and preprocess data
    train_data = load_training_data(args.data_file)
    
    # Training arguments optimized for small dataset
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=100,  # More epochs for better learning
        per_device_train_batch_size=1,  # Smaller batch size
        gradient_accumulation_steps=8,  # Increased gradient accumulation
        warmup_steps=20,  # More warmup steps
        learning_rate=2e-4,  # Slightly higher learning rate
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        no_cuda=True,  # Force CPU training
        max_grad_norm=0.5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        remove_unused_columns=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=lambda examples: preprocess_function(examples, tokenizer)
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed")
    trainer.save_model()

if __name__ == "__main__":
    main() 