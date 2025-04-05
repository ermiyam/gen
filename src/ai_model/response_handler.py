import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import json
from pathlib import Path
import re
from dataclasses import dataclass
import numpy as np
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model initialization and inference"""
    model_name: str
    checkpoint_dir: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    load_in_8bit: bool = False
    use_cache: bool = True
    max_length: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    attention_threshold: float = 0.2
    
    def __post_init__(self):
        """Validate configuration and adjust settings based on available packages"""
        if self.load_in_8bit:
            try:
                import accelerate
                import bitsandbytes
                logger.info("8-bit quantization packages available")
            except ImportError:
                logger.warning("8-bit quantization packages not available. Disabling 8-bit mode.")
                self.load_in_8bit = False
        
        # Adjust device map based on CUDA availability
        if self.device_map == "auto" and not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU.")
            self.device_map = None
            self.torch_dtype = torch.float32

class SpecializedModel:
    """Base class for specialized models"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize()
    
    def initialize(self):
        """Initialize the specialized model"""
        raise NotImplementedError
    
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the specialized model"""
        raise NotImplementedError

class TransformerComponents:
    """Core transformer architecture components"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.position_embeddings = None
        self.attention_weights = {}
        
    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product attention"""
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        self.attention_weights['last_layer'] = attention_weights.detach()
        return torch.matmul(attention_weights, value)
    
    def add_positional_encoding(self, x: torch.Tensor, max_len: int = 512) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x.size(-1), 2) * -(torch.log(torch.tensor(10000.0)) / x.size(-1)))
        pos_encoding = torch.zeros(max_len, x.size(-1))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.position_embeddings = pos_encoding.detach()
        return x + pos_encoding[:x.size(0)].to(x.device)

class ModelManager:
    """Manages model initialization and inference"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.transformer = TransformerComponents(config)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model with proper error handling and checkpoint management"""
        try:
            logger.info(f"Initializing model {self.config.model_name}")
            
            # Set up model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Initialize tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = {
                "config": model_config,
                "trust_remote_code": True,
                "use_cache": self.config.use_cache
            }
            
            # Add quantization settings if available
            if self.config.load_in_8bit:
                model_kwargs.update({
                    "load_in_8bit": True,
                    "device_map": self.config.device_map
                })
            else:
                model_kwargs.update({
                    "torch_dtype": self.config.torch_dtype,
                    "device_map": self.config.device_map
                })
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Try fallback to CPU with default settings
            try:
                logger.info("Attempting fallback to CPU with default settings")
                self.config.load_in_8bit = False
                self.config.device_map = None
                self.config.torch_dtype = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    use_cache=True
                )
                logger.info("Model initialized successfully with fallback settings")
            except Exception as fallback_error:
                logger.error(f"Fallback initialization failed: {fallback_error}")
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with attention visualization"""
        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Add positional encoding
            if 'input_ids' in inputs:
                embedded = self.model.get_input_embeddings()(inputs['input_ids'])
                embedded = self.transformer.add_positional_encoding(embedded)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Get attention patterns
            attention_patterns = {
                'weights': self.transformer.attention_weights,
                'position_embeddings': self.transformer.position_embeddings
            }
            
            return {
                "response": response,
                "token_count": len(outputs[0]),
                "attention_patterns": attention_patterns,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response": "",
                "error": str(e),
                "success": False
            }

class ResponseHandler:
    def __init__(self):
        self.base_model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        self.model_config = ModelConfig(
            model_name=self.base_model_name,
            checkpoint_dir="checkpoints",
            load_in_8bit=True,
            attention_threshold=0.2,
            temperature=0.7
        )
        self.model_manager = ModelManager(self.model_config)
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 5
        self.specialized_models: Dict[str, SpecializedModel] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            'response_times': [],
            'attention_scores': [],
            'token_counts': [],
            'success_rate': []
        }
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all required components"""
        try:
            logger.info("Initializing response handler components")
            self._setup_logging()
            self._load_templates()
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "response_handler.log")
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    def _load_templates(self):
        """Load response templates"""
        self.templates = {
            'system': "I am Gen, an AI assistant focused on providing detailed and accurate responses.",
            'error': "I apologize, but I encountered an issue. Could you please rephrase your question?",
            'clarification': "To better assist you, could you provide more details about your request?"
        }
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the model manager"""
        try:
            # Add conversation history context
            context = self._prepare_context(prompt)
            
            # Generate response
            start_time = time.time()
            result = self.model_manager.generate(context, **kwargs)
            generation_time = time.time() - start_time
            
            if result["success"]:
                # Update metrics
                self._update_metrics(result, generation_time)
                
                # Update conversation history
                self._update_history(prompt, result["response"])
                
                # Analyze attention patterns
                self._analyze_attention(result.get("attention_patterns", {}))
                
                return result["response"]
            else:
                logger.error(f"Generation failed: {result.get('error', 'Unknown error')}")
                return self.templates['error']
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self.templates['error']
    
    def _prepare_context(self, prompt: str) -> str:
        """Prepare context from conversation history"""
        history = self.conversation_history[-self.max_history:] if self.conversation_history else []
        context = "\n".join([
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in history
        ])
        return f"{self.templates['system']}\n\n{context}\nUser: {prompt}\nAssistant:"
    
    def _update_metrics(self, result: Dict[str, Any], generation_time: float):
        """Update performance metrics"""
        self.performance_metrics['token_counts'].append(result['token_count'])
        self.performance_metrics['response_times'].append(generation_time)
        self.performance_metrics['success_rate'].append(1.0)
        
        if 'attention_patterns' in result:
            attention_scores = np.mean([
                w.mean().item() 
                for w in result['attention_patterns'].get('weights', {}).values()
            ])
            self.performance_metrics['attention_scores'].append(attention_scores)
    
    def _update_history(self, prompt: str, response: str):
        """Update conversation history"""
        self.conversation_history.append({
            'user': prompt,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def _analyze_attention(self, attention_patterns: Dict[str, Any]):
        """Analyze attention patterns for insights"""
        if not attention_patterns:
            return
            
        try:
            weights = attention_patterns.get('weights', {})
            embeddings = attention_patterns.get('position_embeddings')
            
            if weights and embeddings is not None:
                # Analyze attention focus
                attention_focus = np.argmax(weights.get('last_layer', torch.zeros(1)).mean(dim=0).cpu().numpy())
                logger.info(f"Attention focus at position: {attention_focus}")
                
                # Analyze positional influence
                pos_influence = torch.norm(embeddings, dim=1).mean().item()
                logger.info(f"Average positional influence: {pos_influence:.4f}")
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_token_count': np.mean(self.performance_metrics['token_counts']) if self.performance_metrics['token_counts'] else 0,
            'avg_response_time': np.mean(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0,
            'avg_attention_score': np.mean(self.performance_metrics['attention_scores']) if self.performance_metrics['attention_scores'] else 0,
            'success_rate': np.mean(self.performance_metrics['success_rate']) if self.performance_metrics['success_rate'] else 0,
            'total_requests': len(self.performance_metrics['token_counts'])
        }

def main():
    """Main entry point for direct chat interface"""
    try:
        import sys
        import code
        
        print("Initializing AI system...")
        handler = ResponseHandler()
        print("AI system initialized. Ready for chat.")
        print("\nType 'exit', 'quit', or 'bye' to end the chat.")
        print("Type your message and press Enter to chat.")
        
        # Create a simple chat loop
        while True:
            try:
                # Get user input using Python's input
                sys.stdout.write("\nYou: ")
                sys.stdout.flush()
                user_input = sys.stdin.readline().strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for using Gen. Goodbye!")
                    break
                
                # Generate and display response
                response = handler.generate_response(user_input)
                print(f"\nGen: {response}")
                
                # Show performance stats every 5 interactions
                if len(handler.performance_metrics['token_counts']) % 5 == 0:
                    stats = handler.get_performance_stats()
                    print("\nPerformance Stats:")
                    for key, value in stats.items():
                        print(f"{key}: {value:.2f}")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using Gen. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                logger.error(f"Error in chat loop: {e}")
                continue
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logger.error(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    main() 