"""
Expert LLM service for handling expert model interactions
"""
import logging
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from typing import Optional, Dict, Any
from openai import OpenAI
from lpm_kernel.configs.config import Config

logger = logging.getLogger(__name__)

class ExpertLLMService:
    """Service for managing expert LLM client"""
    
    def __init__(self):
        self._client = None
        self.user_llm_config_service = UserLLMConfigService()
        # self.user_llm_config = self.user_llm_config_service.get_available_llm()
        # self._config = None
        # Load configuration during initialization
        # self.config  # Trigger configuration loading
        # self.configure()  # Initialize using configuration from environment variables
        
    # @property
    # def config(self) -> Config:
    #     """Get the configuration"""
    #     if self._config is None:
    #         self._config = Config.from_env()
    #     return self._config
        
    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client for expert LLM"""
        if self._client is None:
            self.user_llm_config = self.user_llm_config_service.get_available_llm()
            self._client = OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
        return self._client

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model specific parameters for expert LLM
        
        Returns:
            Dict containing model specific parameters
        """
        return {
            "model": self.user_llm_config.chat_model_name,
            "response_format": {"type": "text"},
            "seed": 42,  # Optional: Fixed random seed to get consistent responses
            "tools": None,  # Optional: If function calling or similar features are needed
            "tool_choice": None,  # Optional: If function calling or similar features are needed
        }
    
    # def configure(self,
    #              api_key: Optional[str] = None,
    #              base_url: Optional[str] = None,
    #              model_name: Optional[str] = None,
    #              **model_params):
    #     """
    #     Configure the expert LLM client and its parameters
    #
    #     Args:
    #         api_key: Optional API key. If None, uses LLM_API_KEY from config
    #         base_url: Optional base URL. If None, uses LLM_BASE_URL from config
    #         model_name: Optional model name to use
    #         **model_params: Additional model specific parameters
    #     """
    #     # Reset existing client
    #     self._client = None
    #
    #     # Update config if provided
    #     # if api_key:
    #     #     self.config.set("LLM_API_KEY", api_key)
    #     # if base_url:
    #     #     self.config.set("LLM_BASE_URL", base_url)
    #     # if model_name:
    #     #     self.config.set("DATA_GEN_MODEL", model_name)
    #
    #     # Update any additional model parameters
    #     # for key, value in model_params.items():
    #     #     self.config.set(f"EXPERT_MODEL_{key.upper()}", value)
    #     #
    #     # logger.info("Expert LLM service configured with new settings")
    #     # logger.info(f"Using model: {self.config.get('DATA_GEN_MODEL')}")
    #     # logger.info(f"Using base URL: {self.config.get('LLM_BASE_URL')}")
    #     return self.client


# Global instance
expert_llm_service = ExpertLLMService()
