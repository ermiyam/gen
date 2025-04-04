from typing import Dict, Any, List, Optional
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
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import json
import os
from datetime import datetime
import logging
import wandb
from dataclasses import dataclass
from enum import Enum
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.config.api_keys import api_keys

from .data_loader import DataLoader, TrainingExample

class TrainingPhase(Enum):
    PRETRAINING = "pretraining"
    INSTRUCTION_TUNING = "instruction_tuning"
    RLHF = "rlhf"
    CONTINUOUS_LEARNING = "continuous_learning"

@dataclass
class TrainingConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    output_dir: str = "models"
    data_dir: str = "data"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    max_seq_length: int = 2048
    load_in_8bit: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    wandb_project: str = "nexgencreators"
    wandb_entity: str = "nexgencreators"
    wandb_name: str = None
    min_example_length: int = 10
    max_example_length: int = 2048

class MarketingAITrainer:
    def __init__(
        self,
        config: TrainingConfig,
        feedback_system: Optional[FeedbackSystem] = None,
        rag_system: Optional[RAGSystem] = None,
        tools_system: Optional[ToolsSystem] = None
    ):
        self.config = config
        self.feedback_system = feedback_system
        self.rag_system = rag_system
        self.tools_system = tools_system
        
        # Initialize data loader
        self.data_loader = DataLoader(data_dir=self.config.data_dir)
        
        # Initialize wandb
        if self.config.wandb_name is None:
            self.config.wandb_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_name,
            config=self.config.__dict__
        )
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set Mistral AI API key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = api_keys.mistral_key
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        
        # Initialize training components
        self.training_args = self._create_training_args()
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
    def _initialize_model(self):
        """Initialize the model and tokenizer with proper configuration."""
        self.logger.info(f"Loading model {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 8-bit quantization if specified
        model_kwargs = {}
        if self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # Prepare model for k-bit training if needed
        if self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA if specified
        if self.config.use_lora:
            self._apply_lora()
        
        return self.model, self.tokenizer
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        if self.config.lora_target_modules is None:
            self.config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _prepare_dataset(self, examples: List[TrainingExample], phase: TrainingPhase) -> Dataset:
        """Prepare dataset for training."""
        # Filter examples based on length
        filtered_examples = self.data_loader.filter_examples(
            examples,
            min_length=self.config.min_example_length,
            max_length=self.config.max_example_length
        )
        
        # Format data based on training phase
        if phase == TrainingPhase.INSTRUCTION_TUNING:
            formatted_data = [
                {
                    "text": f"Instruction: {example.instruction}\nInput: {example.input}\nOutput: {example.output}"
                }
                for example in filtered_examples
            ]
        elif phase == TrainingPhase.RLHF:
            # Use feedback data for RLHF
            formatted_data = [
                {
                    "text": f"Query: {item['query']}\nResponse: {item['response']}\nRating: {item['rating']}"
                }
                for item in self.feedback_system.get_training_data()
            ]
        else:
            formatted_data = [{"text": example.instruction} for example in filtered_examples]
        
        # Tokenize data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length
            )
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Log dataset statistics
        stats = self.data_loader.get_example_stats(filtered_examples)
        self.logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        wandb.log(stats)
        
        return tokenized_dataset
    
    def _create_training_args(self):
        """Create training arguments based on the configuration."""
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, self.config.wandb_name),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to="wandb"
        )
        return training_args
    
    def train(
        self,
        training_file: str,
        eval_file: Optional[str] = None,
        phase: TrainingPhase = TrainingPhase.INSTRUCTION_TUNING
    ):
        """Train the model with the specified phase and data."""
        self.logger.info(f"Starting {phase.value} training phase")
        
        # Load training data
        training_examples = self.data_loader.load_jsonl(training_file)
        eval_examples = self.data_loader.load_jsonl(eval_file) if eval_file else None
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(training_examples, phase)
        eval_dataset = self._prepare_dataset(eval_examples, phase) if eval_examples else None
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(os.path.join(self.config.output_dir, self.config.wandb_name))
        
        self.logger.info(f"Completed {phase.value} training phase")
    
    def continuous_learning(self):
        """Implement continuous learning loop using feedback and performance data."""
        self.logger.info("Starting continuous learning loop")
        
        # Get high-quality training data from feedback system
        feedback_data = self.feedback_system.get_training_data()
        
        # Convert feedback data to training examples
        feedback_examples = [
            TrainingExample(
                instruction=item["query"],
                input="",
                output=item["response"],
                metadata={
                    "source": "feedback",
                    "rating": item["rating"],
                    "type": "feedback"
                }
            )
            for item in feedback_data
        ]
        
        # Get relevant context from RAG system
        context_examples = []
        for item in feedback_data:
            context = self.rag_system.get_relevant_context(item["query"])
            if context:
                context_examples.append(
                    TrainingExample(
                        instruction=item["query"],
                        input=context,
                        output=item["response"],
                        metadata={
                            "source": "rag",
                            "type": "context"
                        }
                    )
                )
        
        # Combine all examples
        combined_examples = self.data_loader.merge_examples(
            feedback_examples,
            context_examples
        )
        
        # Save combined examples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.config.data_dir, f"continuous_learning_{timestamp}.jsonl")
        self.data_loader.save_examples(combined_examples, save_path)
        
        # Train with combined data
        self.train(
            training_file=save_path,
            phase=TrainingPhase.CONTINUOUS_LEARNING
        )
        
        self.logger.info("Completed continuous learning loop")
    
    def save_model(self, phase: TrainingPhase):
        """Save the model and tokenizer."""
        save_path = os.path.join(self.config.output_dir, phase.value)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Saved model to {save_path}")
    
    def load_model(self, phase: TrainingPhase):
        """Load a saved model and tokenizer."""
        load_path = os.path.join(self.config.output_dir, phase.value)
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.logger.info(f"Loaded model from {load_path}")

if __name__ == "__main__":
    config = TrainingConfig()
    feedback_system = None  # Assuming a default implementation
    rag_system = None  # Assuming a default implementation
    tools_system = None  # Assuming a default implementation
    trainer = MarketingAITrainer(config, feedback_system, rag_system, tools_system)
    
    # Train with the provided dataset
    trainer.train(
        training_file="c:/Users/ermiy/Downloads/nexgencreators_dataset_20250401-011745.jsonl",
        phase=TrainingPhase.INSTRUCTION_TUNING
    ) 