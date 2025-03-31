from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

class ModelManager:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Initializing ModelManager with {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Enable model evaluation mode
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self,
                         prompt: str,
                         context: Optional[List[Dict[str, Any]]] = None,
                         max_length: int = 500,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         num_return_sequences: int = 1) -> str:
        """
        Generate a response using the DeepSeek model.
        
        Args:
            prompt: The input prompt
            context: Optional list of context dictionaries
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate
            
        Returns:
            Generated response text
        """
        try:
            # Prepare context
            context_text = ""
            if context:
                context_text = "\n".join([
                    f"Previous {m['category']}: {m['text']}" 
                    for m in context[-3:]  # Use last 3 relevant contexts
                ])
                context_text += "\n\n"
            
            # Combine context and prompt
            full_prompt = f"{context_text}Input: {prompt}\n\nResponse:"
            
            # Tokenize
            inputs = self.tokenizer(full_prompt, 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  max_length=1024)
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "requires_grad": any(p.requires_grad for p in self.model.parameters())
        }
    
    def __del__(self):
        """Cleanup when the model manager is destroyed."""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}") 