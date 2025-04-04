"""
Continuous Learning System for Mak: Processes marketing insights and trains the model indefinitely
"""

import os
import time
import random
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from datasets import Dataset
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_trainer.log'),
        logging.StreamHandler()
    ]
)

class AutoTrainer:
    def __init__(self):
        self.model_dir = Path("models")
        self.training_data_dir = Path("training_data")
        self.checkpoint_dir = Path("models/checkpoints")
        self.data_dir = Path("data")
        self.config_dir = Path("configs")
        self.model = None
        self.tokenizer = None
        self.training_args = None
        self.optimizer = None
        self.scaler = GradScaler()
        self.best_loss = float('inf')
        self.training_topics = []
        self.current_topic_index = 0
        self.current_topic = None
        self.topic_start_time = None
        self.learning_history = []
        self.last_common_crawl_update = None
        self.last_live_update = None
        self.model_version = "v1"  # Fixed model version
        self.trained_topics_log = {}  # Track trained topics
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/training", exist_ok=True)
        
        # Add file handler for logging
        file_handler = logging.FileHandler('logs/auto_trainer.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Load configurations and trained topics log
        self.load_configurations()
        self.load_trained_topics_log()
        
    def load_trained_topics_log(self):
        """Load the log of already trained topics"""
        try:
            trained_log_path = Path("logs/trained_topics_log.json")
            if trained_log_path.exists():
                with open(trained_log_path, "r", encoding="utf-8") as file:
                    self.trained_topics_log = json.load(file)
                self.logger.info(f"Loaded {len(self.trained_topics_log)} trained topics from log")
            else:
                self.trained_topics_log = {}
                self.logger.info("Created new trained topics log")
        except Exception as e:
            self.logger.error(f"Error loading trained topics log: {str(e)}")
            self.trained_topics_log = {}

    def save_trained_topics_log(self):
        """Save the log of trained topics"""
        try:
            trained_log_path = Path("logs/trained_topics_log.json")
            with open(trained_log_path, "w", encoding="utf-8") as file:
                json.dump(self.trained_topics_log, file, indent=4)
            self.logger.info("Saved trained topics log")
        except Exception as e:
            self.logger.error(f"Error saving trained topics log: {str(e)}")

    def is_topic_trained(self, topic: str) -> bool:
        """Check if a topic has already been trained"""
        return topic in self.trained_topics_log

    def mark_topic_as_trained(self, topic: str, success: bool = True):
        """Mark a topic as trained, only if training was successful"""
        if success:
            self.trained_topics_log[topic] = datetime.now().isoformat()
            self.save_trained_topics_log()
            self.logger.info(f"Marked topic as trained: {topic}")
        else:
            self.logger.warning(f"Topic {topic} failed training and was not marked as trained")

    def load_configurations(self):
        """Load and validate configurations"""
        try:
            # Load Common Crawl config
            common_crawl_path = self.config_dir / "common_crawl_config.json"
            if common_crawl_path.exists():
                with open(common_crawl_path, 'r') as f:
                    self.common_crawl_config = json.load(f)
            else:
                self.common_crawl_config = {
                    "enabled": True,
                    "source": "common_crawl",
                    "filters": {
                        "min_words": 100,
                        "include_topics": [
                            "sales", "psychology", "branding", "persuasion", "copywriting",
                            "productivity", "startups", "ecommerce", "customer behavior",
                            "digital marketing", "ads", "social media growth", "neuromarketing",
                            "scaling", "entrepreneurship", "influencer marketing", "lead generation"
                        ]
                    },
                    "cleaning": {
                        "remove_html": True,
                        "remove_duplicate_lines": True,
                        "deduplicate_documents": True
                    },
                    "output_path": "data/common_crawl_chunks",
                    "auto_feed_to_training": True,
                    "auto_discover_topics": True,
                    "update_frequency_secs": 1800
                }
                with open(common_crawl_path, 'w') as f:
                    json.dump(self.common_crawl_config, f, indent=4)

            # Load Live Update config
            live_update_path = self.config_dir / "live_update_config.json"
            if live_update_path.exists():
                with open(live_update_path, 'r') as f:
                    self.live_update_config = json.load(f)
            else:
                self.live_update_config = {
                    "allow_hot_reload": True,
                    "watch_paths": [
                        "data/training_topics.json",
                        "src/auto_trainer.py",
                        "src/conversation.py"
                    ],
                    "auto_discover_topics_enabled": True,
                    "discovery_sources": [
                        "news_sites", "blog_feeds", "youtube_transcripts", "quora", "stackexchange"
                    ],
                    "topic_addition_log": f"logs/topic_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
                with open(live_update_path, 'w') as f:
                    json.dump(self.live_update_config, f, indent=4)

            self.logger.info("Loaded configurations successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
            raise

    def check_for_updates(self):
        """Check for updates from Common Crawl and other sources"""
        try:
            current_time = datetime.now()
            
            # Check Common Crawl updates
            if self.common_crawl_config["enabled"]:
                if (self.last_common_crawl_update is None or 
                    (current_time - self.last_common_crawl_update).seconds >= 
                    self.common_crawl_config["update_frequency_secs"]):
                    self.update_from_common_crawl()
                    self.last_common_crawl_update = current_time
            
            # Check live updates
            if self.live_update_config["allow_hot_reload"]:
                if (self.last_live_update is None or 
                    (current_time - self.last_live_update).seconds >= 300):  # Check every 5 minutes
                    self.check_live_updates()
                    self.last_live_update = current_time
                    
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")

    def update_from_common_crawl(self):
        """Update training data from Common Crawl"""
        try:
            self.logger.info("Updating from Common Crawl...")
            
            # Simulated scraped content (replace with actual Common Crawl scraping)
            scraped_content = """
            AI marketing automation, customer engagement strategies, email funnel optimization,
            neuro-marketing hacks, emotional branding, cross-channel analytics, UGC techniques,
            high-converting ad copy formulas, scarcity psychology, influencer activation framework,
            retail store heatmaps, e-commerce checkout optimizations, retention growth loops,
            viral social hook formulas, omni-channel messaging, upsell/downsell timing tactics,
            creator monetization models, trend hijacking methods, chatbot personalization flows
            """
            
            # Generate and save new topics
            new_topics = self.generate_training_topics(scraped_content)
            
            if new_topics:
                self.logger.info(f"Added {len(new_topics)} new topics from Common Crawl")
                # Reload training topics to include new ones
                self.load_training_topics()
            else:
                self.logger.info("No new topics found in Common Crawl update")
                
        except Exception as e:
            self.logger.error(f"Error updating from Common Crawl: {str(e)}")

    def check_live_updates(self):
        """Check for live updates in watched paths"""
        try:
            for path in self.live_update_config["watch_paths"]:
                if os.path.exists(path):
                    # Check for file modifications
                    # Implement file watching logic here
                    pass
        except Exception as e:
            self.logger.error(f"Error checking live updates: {e}")

    def load_model(self):
        """Load the model and tokenizer with memory optimizations"""
        try:
            # Clear GPU memory before loading
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            model_path = self.get_latest_model()
            
            if model_path is None or not any(model_path.glob("pytorch_model.bin")) and not any(model_path.glob("model.safetensors")):
                self.logger.info("No existing model found. Downloading Mistral 7B...")
                # Download the model with aggressive memory optimizations
                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-v0.1",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    offload_folder="offload",
                    max_memory={0: "4GB", "cpu": "16GB"},  # Reduced GPU memory limit
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    ),
                    low_cpu_mem_usage=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-v0.1",
                    trust_remote_code=True
                )
                
                # Set padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    self.logger.info("Set pad_token to eos_token")
                
                # Save the model and tokenizer
                model_path = self.model_dir / self.model_version
                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                self.logger.info(f"Model saved to {model_path}")
            else:
                self.logger.info(f"Loading existing model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    offload_folder="offload",
                    max_memory={0: "4GB", "cpu": "16GB"},  # Reduced GPU memory limit
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    ),
                    low_cpu_mem_usage=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True
                )
                
                # Set padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    self.logger.info("Set pad_token to eos_token")
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Initialize optimizer with memory-efficient settings
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=5e-6,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def log_memory_usage(self, prefix: str = ""):
        """Log current GPU and CPU memory usage"""
        try:
            # Log CPU memory usage
            import psutil
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / 1024**3  # Convert to GB
            self.logger.info(f"{prefix}CPU Memory - Used: {cpu_memory:.2f}GB")
            
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_memory_free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                gpu_memory_free = gpu_memory_free / 1024**3  # Convert to GB
                
                self.logger.info(f"{prefix}GPU Memory - Allocated: {gpu_memory:.2f}GB, Reserved: {gpu_memory_reserved:.2f}GB, Free: {gpu_memory_free:.2f}GB")
                
                # If memory usage is high, clear cache
                if gpu_memory_free < 1.0:  # Less than 1GB free
                    self.logger.warning("Low GPU memory detected. Clearing cache...")
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
        except Exception as e:
            self.logger.error(f"Error logging memory usage: {str(e)}")

    def clear_gpu_memory(self):
        """Clear GPU memory and collect IPC"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                self.logger.info("GPU memory cleared")
        except Exception as e:
            self.logger.error(f"Error clearing GPU memory: {str(e)}")

    def get_latest_model(self) -> Optional[Path]:
        """Get the path to the latest model checkpoint."""
        try:
            model_path = self.model_dir / self.model_version
            if model_path.exists():
                return model_path
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest model: {e}")
            return None
            
    def load_training_topics(self):
        """Load training topics from JSON files."""
        try:
            # Find the latest training topics file
            topic_files = list(self.data_dir.glob("training_topics_*.json"))
            if not topic_files:
                self.logger.warning("No training topics files found")
                return
                
            latest_file = max(topic_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading training topics from {latest_file}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.training_topics = data
                self.logger.info(f"Loaded {len(self.training_topics)} training topics")
                
        except Exception as e:
            self.logger.error(f"Error loading training topics: {e}")
            
    def select_new_topic(self):
        """Select a new topic that hasn't been trained yet"""
        try:
            if not self.training_topics:
                self.load_training_topics()
            
            # Find an untrained topic
            untrained_topics = [topic for topic in self.training_topics 
                              if not self.is_topic_trained(topic["prompt"])]
            
            if not untrained_topics:
                self.logger.warning("No untrained topics available")
                return None
            
            # Select a random untrained topic
            self.current_topic = random.choice(untrained_topics)
            self.topic_start_time = datetime.now()
            self.logger.info(f"Selected new topic: {self.current_topic['prompt']}")
            
            return self.current_topic
            
        except Exception as e:
            self.logger.error(f"Error selecting new topic: {str(e)}")
            return None
            
    def generate_training_data(self) -> List[Dict]:
        """Generate training data for the current topic."""
        try:
            if not self.training_topics or self.current_topic is None:
                return []
                
            topic_data = self.training_topics[self.current_topic_index]
            num_examples = random.randint(5, 15)
            examples = []
            
            for _ in range(num_examples):
                # Generate variations of the prompt and response
                prompt_variations = [
                    f"Teach me about {self.current_topic}",
                    f"Explain {self.current_topic} in detail",
                    f"What are the key insights about {self.current_topic}?",
                    f"How can I master {self.current_topic}?",
                    f"Break down {self.current_topic} for me"
                ]
                
                response_variations = [
                    f"Here's a comprehensive analysis of {self.current_topic}...",
                    f"Let me break down the key aspects of {self.current_topic}...",
                    f"Here's what you need to know about {self.current_topic}...",
                    f"Let me explain the fundamentals of {self.current_topic}...",
                    f"Here's a detailed guide to {self.current_topic}..."
                ]
                
                input_text = random.choice(prompt_variations)
                response_text = random.choice(response_variations)
                
                examples.append({
                    "input": input_text,
                    "response": response_text,
                    "length": len(input_text) + len(response_text),
                    "topic": self.current_topic,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            return examples
            
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")
            return []
            
    def prepare_training_data(self, topic):
        """Prepare training data for a specific topic"""
        try:
            # Tokenize inputs and responses with padding
            inputs = self.tokenizer(
                topic["prompt"],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            responses = self.tokenizer(
                topic["response"],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Create dataset with proper padding
            dataset = Dataset.from_dict({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": responses["input_ids"]
            })
            
            # Set format for PyTorch
            dataset.set_format(type="torch")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def train(self, dataset):
        """Train the model on the dataset with memory optimizations"""
        try:
            if self.optimizer is None:
                self.logger.error("Optimizer not initialized")
                return None
            
            # Clear GPU memory before training
            self.clear_gpu_memory()
            
            # Log initial memory usage
            self.log_memory_usage("Before training - ")
            
            # Initialize accelerator with memory-efficient settings
            accelerator = Accelerator(
                gradient_accumulation_steps=4,  # Increased for memory efficiency
                mixed_precision="fp16",  # Use mixed precision
                log_with="tensorboard",
                project_dir="logs/training"
            )
            
            # Prepare data loader with minimal batch size
            data_loader = DataLoader(
                dataset,
                batch_size=1,  # Minimal batch size
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            
            # Prepare model and optimizer
            model, optimizer, data_loader = accelerator.prepare(
                self.model,
                self.optimizer,
                data_loader
            )
            
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()
            
            # Training loop
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in data_loader:
                try:
                    # Log memory usage before each batch
                    self.log_memory_usage(f"Batch {num_batches} - ")
                    
                    # Move batch to the correct device
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    
                    # Forward pass with gradient checkpointing
                    with accelerator.autocast():
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            use_cache=False  # Disable cache for memory efficiency
                        )
                        loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accelerator.gradient_accumulation_steps
                    
                    # Backward pass
                    accelerator.backward(loss)
                    
                    # Clip gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 0.5)
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item() * accelerator.gradient_accumulation_steps
                    num_batches += 1
                    
                    # Clear cache periodically
                    if num_batches % 2 == 0:  # Clear more frequently
                        self.clear_gpu_memory()
                    
                    # Log progress
                    if num_batches % 2 == 0:
                        avg_loss = total_loss / num_batches
                        self.logger.info(f"Step {num_batches}, Loss: {avg_loss:.4f}")
                        
                except Exception as e:
                    self.logger.error(f"Error in training batch: {str(e)}")
                    if "CUDA out of memory" in str(e):
                        self.logger.error("GPU out of memory. Skipping this topic.")
                        return None
                    continue
            
            # Calculate average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Save model if it's the best so far
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                unwrapped_model = accelerator.unwrap_model(model)
                model_path = self.model_dir / self.model_version
                unwrapped_model.save_pretrained(str(model_path))
                self.logger.info(f"Updated model {self.model_version} with loss: {avg_loss:.4f}")
            
            # Clear cache after training
            self.clear_gpu_memory()
            
            # Log final memory usage
            self.log_memory_usage("After training - ")
            
            return avg_loss
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            if "CUDA out of memory" in str(e):
                self.logger.error("GPU out of memory. Skipping this topic.")
                return None
            raise
            
    def run(self, topic_interval: int = 300, training_interval: int = 60):
        """Run the continuous learning process."""
        self.logger.info("Starting Continuous Learning System...")
        
        # Load training topics
        self.load_training_topics()
        
        # Initialize first topic
        self.select_new_topic()
        
        while True:
            try:
                # Check for updates
                self.check_for_updates()
                
                # Load model if not loaded
                if self.model is None:
                    self.load_model()
                
                # Select new topic periodically
                if (datetime.now() - self.topic_start_time).seconds >= topic_interval:
                    topic = self.select_new_topic()
                    if topic is None:
                        self.logger.warning("No new topics available. Waiting for updates...")
                        time.sleep(training_interval)
                        continue
                
                # Clear GPU memory before training
                self.clear_gpu_memory()
                
                # Prepare and train on new data
                dataset = self.prepare_training_data(self.current_topic)
                if dataset:
                    loss = self.train(dataset)
                    # Only mark topic as trained if training was successful
                    self.mark_topic_as_trained(self.current_topic["prompt"], success=loss is not None)
                    
                self.logger.info(f"Completed training cycle. Next cycle in {training_interval} seconds...")
                time.sleep(training_interval)
                
            except Exception as e:
                self.logger.error(f"Error in learning cycle: {e}")
                time.sleep(training_interval)

    def process_scraped_content(self, content: str) -> List[str]:
        """Process scraped content into potential training topics"""
        try:
            # Split and clean potential new topics
            possible_topics = list(set([x.strip().capitalize() for x in content.split(",")]))
            
            # Filter out topics already trained
            new_topics = [topic for topic in possible_topics 
                          if not self.is_topic_trained(topic)]
            
            if new_topics:
                self.logger.info(f"Found {len(new_topics)} new potential topics from scraped content")
                return new_topics
            else:
                self.logger.info("No new topics found in scraped content")
                return []
                
        except Exception as e:
            self.logger.error(f"Error processing scraped content: {str(e)}")
            return []

    def generate_training_topics(self, scraped_content: str) -> List[Dict]:
        """Generate training topics from scraped content"""
        try:
            new_topics = self.process_scraped_content(scraped_content)
            if not new_topics:
                return []
            
            # Generate training data format for each new topic
            training_topics = []
            for topic in new_topics:
                training_topics.append({
                    "prompt": f"What should I learn about {topic}?",
                    "response": f"Let me teach you about {topic}. This is a comprehensive guide...",
                    "source": "scraped_content",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save new topics to training data
            self.save_training_topics(training_topics)
            
            return training_topics
            
        except Exception as e:
            self.logger.error(f"Error generating training topics: {str(e)}")
            return []

    def save_training_topics(self, topics: List[Dict]):
        """Save new training topics to file"""
        try:
            # Load existing topics
            if self.training_topics:
                existing_topics = self.training_topics
            else:
                existing_topics = []
            
            # Add new topics
            existing_topics.extend(topics)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topics_file = self.data_dir / f"training_topics_{timestamp}.json"
            
            with open(topics_file, "w", encoding="utf-8") as f:
                json.dump(existing_topics, f, indent=4)
            
            self.logger.info(f"Saved {len(topics)} new training topics to {topics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving training topics: {str(e)}")

def main():
    trainer = AutoTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 