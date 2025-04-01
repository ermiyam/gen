from typing import Dict, Any, List, Optional
import requests
import json
import os
from datetime import datetime
import logging
from abc import ABC, abstractmethod

class Tool(ABC):
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class TikTokTrendsTool(Tool):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tiktok.com/v2/research/trends"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Get trending topics from TikTok."""
        try:
            response = requests.get(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                params=kwargs
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching TikTok trends: {str(e)}")
            return {"error": str(e)}

class FacebookAdsTool(Tool):
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Get Facebook Ads data."""
        try:
            response = requests.get(
                f"{self.base_url}/act_{kwargs.get('account_id')}/insights",
                params={
                    "access_token": self.access_token,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching Facebook Ads data: {str(e)}")
            return {"error": str(e)}

class InstagramPostTool(Tool):
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.instagram.com/v18.0"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Post content to Instagram."""
        try:
            response = requests.post(
                f"{self.base_url}/me/media",
                params={"access_token": self.access_token},
                json=kwargs
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error posting to Instagram: {str(e)}")
            return {"error": str(e)}

class ToolsSystem:
    def __init__(self, config_path: str = "config/tools_config.json"):
        self.config_path = config_path
        self.tools: Dict[str, Tool] = {}
        self._load_config()
        self._initialize_tools()
    
    def _load_config(self):
        """Load tool configurations from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "tiktok": {"api_key": ""},
                "facebook": {"access_token": ""},
                "instagram": {"access_token": ""}
            }
            self._save_config()
    
    def _save_config(self):
        """Save tool configurations to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _initialize_tools(self):
        """Initialize available tools."""
        if self.config["tiktok"]["api_key"]:
            self.tools["tiktok_trends"] = TikTokTrendsTool(self.config["tiktok"]["api_key"])
        
        if self.config["facebook"]["access_token"]:
            self.tools["facebook_ads"] = FacebookAdsTool(self.config["facebook"]["access_token"])
        
        if self.config["instagram"]["access_token"]:
            self.tools["instagram_post"] = InstagramPostTool(self.config["instagram"]["access_token"])
    
    def update_config(self, tool_name: str, credentials: Dict[str, str]):
        """Update tool credentials."""
        if tool_name in self.config:
            self.config[tool_name].update(credentials)
            self._save_config()
            self._initialize_tools()
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool."""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self.tools.keys())
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get status of all tools."""
        return {
            "tiktok_trends": "tiktok_trends" in self.tools,
            "facebook_ads": "facebook_ads" in self.tools,
            "instagram_post": "instagram_post" in self.tools
        } 