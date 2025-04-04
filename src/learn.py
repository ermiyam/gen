import os
import logging
import time
import smtplib
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
from training_manager import TrainingManager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

# Email configuration
EMAIL_CONFIG = {
    'sender': 'mak@aimarketing.com',
    'recipient': 'ermiya@aimarketing.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'smtp_username': 'mak@aimarketing.com',
    'smtp_password': 'your_app_specific_password'  # Use app-specific password for Gmail
}

def send_email_notification(subject: str, message: str) -> bool:
    """Send email notification about training completion."""
    if not SEND_EMAIL:
        return False
        
    try:
        server = smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port'])
        server.starttls()
        server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['sender_password'])
        
        email_message = f"Subject: {subject}\n\n{message}"
        server.sendmail(
            EMAIL_SETTINGS['sender_email'],
            EMAIL_SETTINGS['recipient_email'],
            email_message
        )
        server.quit()
        logging.info("Email notification sent successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")
        return False

def send_training_notification(status, metrics=None, error=None):
    """Send email notification about training status"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['sender']
    msg['To'] = EMAIL_CONFIG['recipient']
    
    if status == 'start':
        subject = '🚀 MAK Training Started'
        body = f"""
        MAK training has started at {datetime.now()}
        
        Training Configuration:
        - Model: Mistral-7B-v0.1
        - Training Data: {len(load_training_data())} examples
        - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
        """
    elif status == 'complete':
        subject = '✅ MAK Training Completed'
        body = f"""
        MAK training completed successfully at {datetime.now()}
        
        Training Metrics:
        - Loss: {metrics.get('loss', 'N/A')}
        - Accuracy: {metrics.get('accuracy', 'N/A')}
        - Training Time: {metrics.get('training_time', 'N/A')}
        """
    else:  # error
        subject = '❌ MAK Training Failed'
        body = f"""
        MAK training failed at {datetime.now()}
        
        Error Details:
        {str(error)}
        """
    
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['smtp_username'], EMAIL_CONFIG['smtp_password'])
        server.send_message(msg)
        server.quit()
        logging.info(f"Email notification sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email notification: {str(e)}")

def load_model_and_tokenizer():
    """Load model and tokenizer with quantization"""
    model_id = "mistralai/Mistral-7B-v0.1"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def create_training_dataset(tokenizer, training_manager: TrainingManager):
    """Create a training dataset from collected data."""
    try:
        # Get training plan
        training_plan = training_manager.get_training_plan()
        logging.info(f"Training focus: {training_plan['next_topic']} (Priority: {training_plan['priority']})")
        
        # Collect and format training data
        training_data = training_manager.collect_training_data()
        training_manager.save_training_data(training_data)
        
        # Convert to dataset format
        dataset = Dataset.from_list([{"text": item} for item in training_data])
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset, training_plan
        
    except Exception as e:
        logging.error(f"Error creating dataset: {str(e)}")
        raise

def generate_sample_responses(model, tokenizer, examples):
    """Generate sample responses to test the model."""
    logging.info("\n📊 Model Response Samples:")
    for ex in examples:
        inputs = tokenizer(ex, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=50)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info(f"\n[USER] {ex}\n[MAK] {response}")

def get_next_model_version():
    """Get the next version number for model saving"""
    base_dir = "models"
    version = 1
    while os.path.exists(f"{base_dir}/v{version}"):
        version += 1
    return f"{base_dir}/v{version}"

def collect_prioritized_data():
    """Collect training data based on priority topics."""
    priority_file = "src/priority_topics.json"
    if not os.path.exists(priority_file):
        print("No priority file found. Training from feedback only.")
        return open("data/train.txt", "r").read()

    with open(priority_file, "r") as f:
        priorities = json.load(f)

    # Sort topics by priority
    topics = priorities['topics']
    sorted_topics = sorted(
        topics.items(),
        key=lambda x: (
            -x[1]['priority'],
            x[1]['training_count']
        )
    )

    combined_data = ""
    for topic, details in sorted_topics:
        folder = f"data/knowledge/{details['category']}/{topic}"
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".txt"):
                    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                        combined_data += f.read() + "\n\n---\n\n"

    return combined_data

def load_training_data():
    """Load and prepare training data"""
    try:
        with open('data/train.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into examples
        examples = content.split('---')
        training_data = []
        
        for example in examples:
            if not example.strip():
                continue
                
            # Extract input and response
            parts = example.split('### Response:')
            if len(parts) != 2:
                continue
                
            input_part = parts[0]
            response_part = parts[1]
            
            # Extract input
            input_text = input_part.split('### Input:')[1].strip()
            
            # Create training example
            training_example = {
                'text': f"### Input: {input_text}\n### Response: {response_part.strip()}"
            }
            training_data.append(training_example)
        
        return training_data
    except Exception as e:
        logging.error(f"Error loading training data: {str(e)}")
        return []

def create_training_dataset(tokenizer):
    """Create training dataset from loaded data"""
    training_data = load_training_data()
    
    if not training_data:
        logging.error("No training data found")
        return None
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(training_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train():
    """Main training function"""
    try:
        start_time = datetime.now()
        logging.info("Starting training process...")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        logging.info("Model and tokenizer loaded successfully")
        
        # Create training dataset
        train_dataset = create_training_dataset(tokenizer)
        if train_dataset is None:
            raise ValueError("Failed to create training dataset")
        
        logging.info(f"Training dataset created with {len(train_dataset)} examples")
        
        # Get next model version
        model_version = get_next_model_version()
        logging.info(f"Training to version: {model_version}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_version,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='logs',
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard"  # Enable TensorBoard logging
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Using same dataset for eval
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model(model_version)
        tokenizer.save_pretrained(model_version)
        
        # Calculate training time
        training_time = datetime.now() - start_time
        
        # Get training metrics
        train_metrics = train_result.metrics
        
        # Prepare metrics
        metrics = {
            'train_loss': train_metrics.get('train_loss', 'N/A'),
            'eval_loss': train_metrics.get('eval_loss', 'N/A'),
            'training_time': str(training_time),
            'model_version': model_version,
            'epochs': training_args.num_train_epochs,
            'samples_per_second': train_metrics.get('train_samples_per_second', 'N/A')
        }
        
        # Save training metrics
        metrics_file = f'logs/training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info("Training completed successfully")
        logging.info(f"Training metrics saved to: {metrics_file}")
        logging.info(f"Model saved to: {model_version}")
        
        return True
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Start training
    success = train()
    if success:
        logging.info("Training completed successfully!")
    else:
        logging.error("Training failed!") 