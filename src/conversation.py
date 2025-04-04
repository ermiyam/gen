import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/conversation.log'),
        logging.StreamHandler()
    ]
)

def get_latest_model_path(base_path="models"):
    """Get the path to the latest model version."""
    try:
        versions = [d for d in os.listdir(base_path) if d.startswith("v")]
        if not versions:
            raise FileNotFoundError("No model versions found in models directory")
        versions.sort(key=lambda x: int(x[1:]))  # Sort by version number
        latest_version = versions[-1]
        model_path = os.path.join(base_path, latest_version)
        logging.info(f"Using model version: {latest_version}")
        return model_path
    except Exception as e:
        logging.error(f"Error finding latest model: {e}")
        raise

def load_model_and_tokenizer():
    """Load the model and tokenizer with error handling."""
    try:
        model_path = get_latest_model_path()
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logging.info("Loading model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        ).to(device)
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def chat_with_mak():
    """Main chat loop with Mak."""
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("\n🧠 Mak is loaded and ready to chat! Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Mak: Catch you later 🚀")
                break

            try:
                prompt = f"### Input:\n{user_input}\n### Response:\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "### Response:" in response:
                    print("\nMak:", response.split("### Response:")[-1].strip())
                else:
                    print("\nMak:", response.strip())
                    
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                print("\nMak: Sorry, I encountered an error. Let me try that again...")

    except Exception as e:
        logging.error(f"Fatal error in chat loop: {e}")
        print("\n❌ Error: Failed to initialize Mak. Please check the logs for details.")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    chat_with_mak() 