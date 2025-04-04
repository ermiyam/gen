import argparse
import logging
import sys
from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path: str, base_model_name: str = "distilgpt2") -> Tuple[PeftModel, AutoTokenizer]:
    """Load the model and tokenizer."""
    try:
        logger.info(f"Loading base model {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=None,  # Force CPU
            torch_dtype=torch.float32,
            local_files_only=False
        )
        logger.info("Base model loaded successfully")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        logger.info(f"Loading adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()  # Set to evaluation mode
        logger.info("Adapter loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    instruction: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_return_sequences: int = 1
) -> str:
    """Generate a response using the model."""
    try:
        logger.info("Preparing prompt...")
        prompt = f"### Instruction: {instruction}\n### Response:"
        logger.info(f"Generated prompt: {prompt}")
        
        logger.info("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        logger.info(f"Input shape: {inputs.input_ids.shape}")
        
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.2
            )
        logger.info(f"Output shape: {outputs.shape}")
        
        logger.info("Decoding response...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        if "### End" in response:
            response = response.split("### End")[0].strip()
        
        logger.info("Response generated successfully")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
        parser.add_argument("--instruction", type=str, required=True, help="Instruction for content generation")
        args = parser.parse_args()

        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model(args.model_path)
        
        logger.info("Generating response...")
        print("\nGenerated Response:")
        print("-" * 50)
        response = generate_response(model, tokenizer, args.instruction)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 