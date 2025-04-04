import logging
from pathlib import Path
from src.config.api_keys import api_keys

def setup_logging():
    """Setup logging for the test."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "api_key_test.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_mistral_api_key():
    """Test the Mistral AI API key validation."""
    logger = setup_logging()
    logger.info("Starting Mistral AI API key validation test")
    
    # Test the API key
    is_valid = api_keys.test_mistral_key()
    
    if is_valid:
        logger.info("✅ API key validation successful!")
        logger.info("Key is valid and has proper permissions")
    else:
        logger.error("❌ API key validation failed!")
        logger.error("Please check the logs for detailed error information")
    
    return is_valid

if __name__ == "__main__":
    test_mistral_api_key() 