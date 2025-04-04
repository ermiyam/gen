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
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from typing import List, Dict, Any
from src.config.api_keys import api_keys

class GenChat:
    """Interactive chat interface for Gen model."""
    
    def __init__(
        self,
        model_path: str = "models/mistral-gen",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ):
        self.console = Console()
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.chat_history: List[Dict[str, str]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Load model and tokenizer
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"gen_chat_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.generation_config = GenerationConfig(
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the prompt for the model."""
        if input_text:
            return f"<s>[INST] {instruction}\n{input_text} [/INST]"
        return f"<s>[INST] {instruction} [/INST]"
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part
            response = response.split("[/INST]")[-1].strip()
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _create_chat_table(self) -> Table:
        """Create a rich table for chat history."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Role", style="cyan")
        table.add_column("Message", style="green")
        
        for message in self.chat_history:
            table.add_row(message["role"], message["content"])
        
        return table
    
    def chat(self):
        """Start an interactive chat session."""
        self.console.print(Panel.fit(
            "[bold green]Welcome to Gen Chat![/bold green]\n"
            "Ask me anything about marketing, content creation, or strategy.\n"
            "Type 'exit' to end the conversation.",
            title="Gen Chat Interface"
        ))
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.console.print("\n[bold green]Goodbye! Thanks for chatting![/bold green]")
                break
            
            # Add user message to history
            self.chat_history.append({
                "role": "User",
                "content": user_input
            })
            
            # Generate response
            prompt = self._format_prompt(user_input)
            response = self._generate_response(prompt)
            
            # Add assistant response to history
            self.chat_history.append({
                "role": "Gen",
                "content": response
            })
            
            # Display response with markdown formatting
            self.console.print("\n[bold green]Gen[/bold green]")
            self.console.print(Markdown(response))
            
            # Display chat history
            self.console.print("\n[bold]Chat History:[/bold]")
            self.console.print(self._create_chat_table())

def main():
    try:
        # Initialize chat interface
        chat = GenChat()
        
        # Start chat session
        chat.chat()
        
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise

if __name__ == "__main__":
    main() 