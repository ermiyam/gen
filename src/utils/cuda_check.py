import torch
import sys
import subprocess
from transformers import AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_cuda_setup():
    """Run comprehensive CUDA and GPU diagnostics."""
    
    # Check PyTorch CUDA availability
    logger.info("Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        logger.error("CUDA is not available! Please check your PyTorch installation and NVIDIA drivers.")
        return False
    
    # Get CUDA version
    logger.info("Checking CUDA version...")
    cuda_version = torch.version.cuda
    logger.info(f"CUDA Version: {cuda_version}")
    
    # Get GPU info
    logger.info("Checking GPU information...")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU Name: {gpu_name}")
    
    # Check GPU memory
    logger.info("Checking GPU memory...")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024**3)
    logger.info(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    
    # Try to allocate a small tensor on GPU
    logger.info("Testing GPU tensor allocation...")
    try:
        test_tensor = torch.zeros(1000, 1000).cuda()
        del test_tensor
        logger.info("Successfully allocated test tensor on GPU")
    except Exception as e:
        logger.error(f"Failed to allocate tensor on GPU: {str(e)}")
        return False
    
    # Try to load a small model
    logger.info("Testing model loading...")
    try:
        model = AutoModel.from_pretrained("distilgpt2")
        model.to('cuda')
        logger.info("Successfully loaded test model on GPU")
        del model
    except Exception as e:
        logger.error(f"Failed to load model on GPU: {str(e)}")
        return False
    
    # Check NVIDIA driver version
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        logger.info("NVIDIA Driver Info:")
        logger.info(nvidia_smi)
    except Exception as e:
        logger.error(f"Failed to get NVIDIA driver info: {str(e)}")
        return False
    
    logger.info("All CUDA checks passed successfully!")
    return True

if __name__ == "__main__":
    success = check_cuda_setup()
    sys.exit(0 if success else 1) 