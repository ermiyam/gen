import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import requests
import logging
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class APIKeys:
    """Secure API key management."""
    
    def __init__(self):
        self._mistral_key: Optional[str] = None
        self.logger = logging.getLogger(__name__)
    
    @property
    def mistral_key(self) -> Optional[str]:
        """Get Mistral AI API key."""
        if self._mistral_key is None:
            self._mistral_key = os.getenv('MISTRAL_API_KEY')
        return self._mistral_key
    
    @mistral_key.setter
    def mistral_key(self, value: str):
        """Set Mistral AI API key."""
        self._mistral_key = value
        # Update .env file
        self._update_env_file('MISTRAL_API_KEY', value)
        # Test the key after setting
        self.test_mistral_key()
    
    def _update_env_file(self, key: str, value: str):
        """Update or create .env file with new key-value pair."""
        env_path = Path('.env')
        if not env_path.exists():
            env_path.touch()
        
        # Read existing content
        content = env_path.read_text()
        
        # Check if key exists
        if f"{key}=" in content:
            # Update existing key
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}")
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
        else:
            # Add new key
            content = f"{content}\n{key}={value}" if content else f"{key}={value}"
        
        # Write back to file
        env_path.write_text(content)
        
        # Update environment variable
        os.environ[key] = value
    
    def test_mistral_key(self) -> bool:
        """Test if the Mistral AI API key is working correctly.
        
        Returns:
            bool: True if the key is valid and working, False otherwise.
        """
        if not self.mistral_key:
            self.logger.error("Mistral AI API key is not set")
            return False
        
        try:
            # Test endpoint for model access
            headers = {
                "Authorization": f"Bearer {self.mistral_key}",
                "Content-Type": "application/json"
            }
            
            # Try to access the Mistral model list
            response = requests.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Mistral AI API key is valid and working")
                return True
            else:
                self.logger.error(f"Mistral AI API key validation failed with status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error testing Mistral AI API key: {str(e)}")
            return False

# Create global instance
api_keys = APIKeys()

# Set the Mistral AI API key
api_keys.mistral_key = "bkREqw7alt1vSYFgsrQ8wTbj2q9C1wlx" 