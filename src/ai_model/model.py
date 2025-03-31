from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union, Any
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from torch.cuda.amp import autocast, GradScaler
import wandb
import os

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_model.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the AI model"""
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"
    max_length: int = 2048
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    use_wandb: bool = True
    cache_dir: str = ".model_cache"
    checkpoint_dir: str = "checkpoints"

class CustomDataset(Dataset):
    """Custom dataset for training"""
    def __init__(self, data: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_text = item['output']

        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class ModelMonitor:
    """Monitor model performance and resource usage"""
    def __init__(self):
        self.metrics = {
            'training_loss': [],
            'validation_loss': [],
            'gpu_memory_used': [],
            'inference_times': []
        }
        self._lock = threading.Lock()

    def update_metric(self, metric_name: str, value: float):
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)

    def get_metrics(self) -> Dict[str, List[float]]:
        with self._lock:
            return self.metrics.copy()

    def log_gpu_stats(self):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            self.update_metric('gpu_memory_used', memory_allocated)

class CustomAI:
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the AI model with advanced configuration"""
        self.config = config or ModelConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = ModelMonitor()
        
        logger.info(f"Initializing CustomAI with device: {self.device}")
        
        # Create cache and checkpoint directories
        Path(self.config.cache_dir).mkdir(exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)

        # Initialize model components
        self._initialize_model()
        self._setup_optimization()
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            wandb.init(project="custom-ai", config=vars(self.config))

    def _initialize_model(self):
        """Initialize the model and tokenizer with error handling"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Initialize tokenizer with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Initialize model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir=self.config.cache_dir,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            ).to(self.device)

            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _setup_optimization(self):
        """Setup optimization components"""
        self.scaler = GradScaler() if self.config.mixed_precision else None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

    def train_on_data(self, training_data: List[Dict[str, str]], validation_split: float = 0.1):
        """
        Train the model on custom data with advanced features
        """
        logger.info("Starting advanced training process")
        try:
            # Split data into train and validation
            np.random.shuffle(training_data)
            split_idx = int(len(training_data) * (1 - validation_split))
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]

            # Create datasets
            train_dataset = CustomDataset(train_data, self.tokenizer, self.config.max_length)
            val_dataset = CustomDataset(val_data, self.tokenizer, self.config.max_length)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # Training loop with mixed precision
            self.model.train()
            for epoch in range(3):  # Number of epochs
                epoch_loss = 0
                for batch_idx, batch in enumerate(train_loader):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Forward pass with mixed precision
                    with autocast(enabled=self.config.mixed_precision):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        loss = outputs.loss / self.config.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    if self.config.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Update weights with gradient accumulation
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        if self.config.mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    
                    # Log metrics
                    self.monitor.update_metric('training_loss', loss.item())
                    self.monitor.log_gpu_stats()
                    
                    if self.config.use_wandb:
                        wandb.log({
                            'train_loss': loss.item(),
                            'epoch': epoch,
                            'batch': batch_idx
                        })

                # Validation phase
                self.model.eval()
                val_loss = self._validate(val_loader)
                self.model.train()

                logger.info(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

                # Save checkpoint
                self._save_checkpoint(epoch, val_loss)

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _validate(self, val_loader: DataLoader) -> float:
        """Perform validation"""
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                val_loss += outputs.loss.item()

        return val_loss / len(val_loader)

    def generate_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Union[str, List[str]]:
        """
        Generate a response with advanced parameters and error handling
        """
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)

            # Generate with advanced parameters
            with torch.no_grad(), autocast(enabled=self.config.mixed_precision):
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length or self.config.max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.2,
                    length_penalty=1.0
                )

            # Decode outputs
            responses = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            # Log inference time
            inference_time = time.time() - start_time
            self.monitor.update_metric('inference_times', inference_time)
            
            return responses[0] if num_return_sequences == 1 else responses

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save a model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        return {
            'model_name': self.config.model_name,
            'device': str(self.device),
            'metrics': self.monitor.get_metrics(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        } 