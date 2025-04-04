from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from datetime import datetime
import json
import os
from dataclasses import dataclass
from enum import Enum

from .feedback_system import FeedbackSystem, FeedbackType
from .rag_system import RAGSystem
from .tools_system import ToolsSystem
from ..training.trainer import MarketingAITrainer, TrainingConfig, TrainingPhase

class ResponseType(Enum):
    MARKETING_STRATEGY = "marketing_strategy"
    SOCIAL_MEDIA = "social_media"
    CONTENT_CREATION = "content_creation"
    ANALYTICS = "analytics"
    GENERAL = "general"

@dataclass
class ResponseConfig:
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True
    use_tools: bool = True
    use_rag: bool = True
    use_feedback: bool = True

class UnifiedSystem:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        config: ResponseConfig = ResponseConfig()
    ):
        self.config = config
        self.model_name = model_name
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize training system
        self._initialize_training()
    
    def _initialize_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components")
        
        # Initialize feedback system
        self.feedback_system = FeedbackSystem()
        
        # Initialize RAG system
        self.rag_system = RAGSystem()
        
        # Initialize tools system
        self.tools_system = ToolsSystem()
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _initialize_training(self):
        """Initialize the training system."""
        self.logger.info("Initializing training system")
        
        training_config = TrainingConfig(
            model_name=self.model_name,
            output_dir="models"
        )
        
        self.trainer = MarketingAITrainer(
            config=training_config,
            feedback_system=self.feedback_system,
            rag_system=self.rag_system,
            tools_system=self.tools_system
        )
    
    def _determine_response_type(self, query: str) -> ResponseType:
        """Determine the type of response needed."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["strategy", "plan", "campaign"]):
            return ResponseType.MARKETING_STRATEGY
        elif any(word in query_lower for word in ["social", "post", "tweet", "instagram"]):
            return ResponseType.SOCIAL_MEDIA
        elif any(word in query_lower for word in ["content", "write", "create"]):
            return ResponseType.CONTENT_CREATION
        elif any(word in query_lower for word in ["analytics", "data", "metrics"]):
            return ResponseType.ANALYTICS
        else:
            return ResponseType.GENERAL
    
    def _get_relevant_context(self, query: str, response_type: ResponseType) -> str:
        """Get relevant context from RAG system."""
        if not self.config.use_rag:
            return ""
        
        context = self.rag_system.get_relevant_context(query)
        if not context:
            return ""
        
        return f"Context: {context}\n"
    
    def _get_tool_outputs(self, query: str, response_type: ResponseType) -> Dict[str, Any]:
        """Get relevant tool outputs."""
        if not self.config.use_tools:
            return {}
        
        tool_outputs = {}
        
        if response_type == ResponseType.SOCIAL_MEDIA:
            # Get TikTok trends
            trends = self.tools_system.execute_tool("tiktok_trends", region="US")
            if trends and "error" not in trends:
                tool_outputs["tiktok_trends"] = trends
        
        elif response_type == ResponseType.ANALYTICS:
            # Get Facebook Ads data
            ads_data = self.tools_system.execute_tool("facebook_ads", account_id="your_account_id")
            if ads_data and "error" not in ads_data:
                tool_outputs["facebook_ads"] = ads_data
        
        return tool_outputs
    
    def _format_prompt(
        self,
        query: str,
        context: str,
        tool_outputs: Dict[str, Any],
        response_type: ResponseType
    ) -> str:
        """Format the prompt for the model."""
        prompt_parts = [
            "You are an expert marketing AI assistant. Provide a detailed, helpful response.",
            f"Response Type: {response_type.value}",
            f"Query: {query}"
        ]
        
        if context:
            prompt_parts.append(context)
        
        if tool_outputs:
            prompt_parts.append("Relevant Data:")
            for tool, data in tool_outputs.items():
                prompt_parts.append(f"{tool}: {json.dumps(data)}")
        
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def generate_response(
        self,
        query: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response using all system components."""
        self.logger.info(f"Generating response for query: {query}")
        
        # Determine response type
        response_type = self._determine_response_type(query)
        
        # Get relevant context
        context = self._get_relevant_context(query, response_type)
        
        # Get tool outputs
        tool_outputs = self._get_tool_outputs(query, response_type)
        
        # Format prompt
        prompt = self._format_prompt(query, context, tool_outputs, response_type)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Response:")[-1].strip()
        
        # Record feedback if enabled
        if self.config.use_feedback and user_id:
            self.feedback_system.add_feedback(
                query=query,
                response=response,
                feedback_type=FeedbackType.NEUTRAL,  # Default to neutral
                rating=3.0,  # Default rating
                comments="",
                metadata=metadata or {}
            )
        
        return {
            "response": response,
            "response_type": response_type.value,
            "context_used": bool(context),
            "tools_used": list(tool_outputs.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def update_feedback(
        self,
        query: str,
        response: str,
        feedback_type: FeedbackType,
        rating: float,
        comments: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update feedback for a response."""
        self.feedback_system.add_feedback(
            query=query,
            response=response,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            metadata=metadata
        )
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the knowledge base."""
        self.rag_system.add_document(text, metadata)
    
    def update_tool_config(self, tool_name: str, credentials: Dict[str, str]):
        """Update tool credentials."""
        self.tools_system.update_config(tool_name, credentials)
    
    def run_continuous_learning(self):
        """Run the continuous learning loop."""
        self.trainer.continuous_learning()
    
    def save_state(self, path: str):
        """Save the current state of all components."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(os.path.join(path, "model"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save configurations
        configs = {
            "response_config": self.config.__dict__,
            "model_name": self.model_name
        }
        with open(os.path.join(path, "configs.json"), 'w') as f:
            json.dump(configs, f, indent=2)
        
        self.logger.info(f"Saved system state to {path}")
    
    def load_state(self, path: str):
        """Load a saved state of all components."""
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(os.path.join(path, "model"))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        
        # Load configurations
        with open(os.path.join(path, "configs.json"), 'r') as f:
            configs = json.load(f)
            self.model_name = configs["model_name"]
            self.config = ResponseConfig(**configs["response_config"])
        
        self.logger.info(f"Loaded system state from {path}") 