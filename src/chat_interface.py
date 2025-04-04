import gradio as gr
import json
import datetime
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

class GenChatInterface:
    def __init__(self, model_path: str = "models/gen_v1"):
        """Initialize the chat interface with the trained model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.feedback_file = Path("data/feedback/gen_feedback.jsonl")
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Configure quantization for better memory usage
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer with proper configuration
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True
            )
            
            # Ensure tokenizer has a pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Ensure model is in evaluation mode
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def format_prompt(self, prompt: str) -> str:
        """Format the prompt for the model with proper instruction format."""
        # Add system message and format instruction
        formatted_prompt = f"""<s>[INST] You are Gen, an expert AI marketing assistant. Your responses should be creative, engaging, and aligned with modern marketing best practices.

{prompt} [/INST]"""
        return formatted_prompt
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the fine-tuned model."""
        try:
            # Format the prompt
            formatted_prompt = self.format_prompt(prompt)
            
            # Prepare input with proper padding and truncation
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                add_special_tokens=True
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response with optimized parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response with proper cleanup
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract only the response part after [/INST]
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
            
            # Remove any remaining special tokens and clean up
            response = response.replace("<s>", "").replace("</s>", "").strip()
            
            # Ensure we have a response
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def save_feedback(self, prompt: str, response: str, rating: int, tag: str, used: bool):
        """Save feedback to the JSONL file."""
        try:
            feedback = {
                "prompt": prompt,
                "response": response,
                "rating": rating,
                "tag": tag,
                "used": used,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(feedback) + "\n")
                
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
            
    def create_interface(self):
        """Create the Gradio interface."""
        def chat_with_feedback(prompt: str, rating: int, tag: str, used: bool):
            # Generate response
            response = self.generate_response(prompt)
            
            # Save feedback
            self.save_feedback(prompt, response, rating, tag, used)
            
            return response
        
        # Example prompts for testing
        example_prompts = [
            "Write a TikTok ad for a protein powder targeting Gen Z",
            "Create an Instagram caption for a fitness transformation post",
            "Write a hook for a social media video about productivity tips",
            "Create a marketing strategy for a new eco-friendly product",
            "Write a LinkedIn post announcing a new product feature"
        ]
        
        # Create the interface
        with gr.Blocks(title="Gen - AI Marketing Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🧠 Gen - Your AI Marketing Assistant")
            gr.Markdown("""
            ### 💡 How to Use
            1. Choose an example prompt or write your own
            2. Click "Ask Gen" to generate a response
            3. Rate the response (1-5 stars)
            4. Tag the content type
            5. Mark if you used the response
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Your Marketing Question",
                        placeholder="Write a TikTok script for a protein powder...",
                        lines=3
                    )
                    gr.Examples(
                        examples=example_prompts,
                        inputs=prompt,
                        outputs=None,
                        fn=None,
                        cache_examples=False
                    )
                    submit_btn = gr.Button("Ask Gen", variant="primary", size="large")
                
                with gr.Column(scale=3):
                    response = gr.Textbox(
                        label="Gen's Response",
                        lines=5,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    rating = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Rate Gen's Response (1-5)"
                    )
                    tag = gr.Dropdown(
                        choices=["caption", "script", "hook", "strategy", "ad", "other"],
                        value="other",
                        label="Content Type"
                    )
                    used = gr.Checkbox(
                        label="I Used This Response",
                        value=False
                    )
                
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 Tips
                    - Be specific in your prompts
                    - Rate responses to help Gen learn
                    - Tag content types for better organization
                    - Mark used responses for future training
                    """)
            
            submit_btn.click(
                fn=chat_with_feedback,
                inputs=[prompt, rating, tag, used],
                outputs=response
            )
        
        return interface

def main():
    """Launch the chat interface."""
    try:
        chat = GenChatInterface()
        interface = chat.create_interface()
        interface.launch(share=False)  # Disable share to avoid connection issues
    except Exception as e:
        print(f"Error launching interface: {str(e)}")
        raise

if __name__ == "__main__":
    main() 