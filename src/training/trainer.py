import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import os
from typing import Dict, Any

class MarketingAITrainer:
    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "trained_model",
        training_data_path: str = "src/training/data/training_examples.json"
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.training_data_path = training_data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        
    def load_training_data(self) -> Dataset:
        """Load and format training data from JSON file."""
        with open(self.training_data_path, 'r') as f:
            data = json.load(f)
        
        # Format data for training
        formatted_data = []
        for example in data["examples"]:
            formatted_data.append({
                "text": f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
            })
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the examples for training."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    def train(self, num_train_epochs: int = 3):
        """Train the model using LoRA fine-tuning."""
        # Load and prepare data
        dataset = self.load_training_data()
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training complete! Model saved to {self.output_dir}")

if __name__ == "__main__":
    trainer = MarketingAITrainer()
    trainer.train() 