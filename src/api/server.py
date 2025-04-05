from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import warnings
from functools import wraps
import threading
from typing import Dict, Any, Optional
import json
from datetime import datetime
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.ai_model.response_handler import ResponseHandler, ModelConfig
from src.ai_model.code_gen_model import CodeGenModel

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    handlers=[RotatingFileHandler(log_dir / "server.log", maxBytes=10000000, backupCount=5)],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

class AIServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize rate limiter with Redis storage
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["100 per day", "10 per minute"],
            storage_uri="memory://"  # Change to redis://localhost:6379/0 if Redis is available
        )
        
        # Initialize cache with filesystem storage
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        self.cache = Cache(
            self.app,
            config={
                'CACHE_TYPE': 'filesystem',
                'CACHE_DIR': str(cache_dir),
                'CACHE_DEFAULT_TIMEOUT': 300
            }
        )
        
        # Initialize model loading semaphore
        self.model_loading_lock = threading.Lock()
        
        # Initialize AI components
        self._initialize_ai()
        
        # Register routes
        self._register_routes()
        
        logger.info("AI server initialized successfully")
    
    def _initialize_ai(self):
        """Initialize AI components with specialized models"""
        try:
            with self.model_loading_lock:
                # Initialize base response handler
                self.ai_handler = ResponseHandler()
                
                # Initialize specialized models with proper error handling
                code_gen_config = ModelConfig(
                    model_name="deepseek-ai/deepseek-coder-6.7b-base",
                    attention_threshold=0.2,
                    temperature=0.7
                )
                
                # Ensure models are loaded with proper timeout and retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.ai_handler.specialized_models['code_generation'] = CodeGenModel(code_gen_config)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"Retry {attempt + 1}/{max_retries} loading model: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                logger.info("AI handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            raise
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.route('/api/chat', methods=['POST'])
        @self.limiter.limit("10 per minute")
        @self.cache.memoize(timeout=60)
        def chat():
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return jsonify({'error': 'Invalid request data'}), 400
                
                start_time = time.time()
                
                # Ensure model is loaded before processing
                with self.model_loading_lock:
                    response = self.ai_handler.generate_response(data['message'])
                
                processing_time = time.time() - start_time
                
                return jsonify({
                    'response': response,
                    'processing_time': processing_time,
                    'stats': self.ai_handler.get_performance_stats()
                })
            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def stats():
            try:
                return jsonify(self.ai_handler.get_performance_stats())
            except Exception as e:
                logger.error(f"Error in stats endpoint: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):  # Set debug=False for production
        """Run the AI server"""
        self.app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=False  # Disable reloader to prevent duplicate model loading
        )

def main():
    """Main entry point"""
    try:
        server = AIServer()
        server.run()
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise

if __name__ == "__main__":
    main() 