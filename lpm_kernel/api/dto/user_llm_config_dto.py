from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class BaseUserLLMConfigDTO(BaseModel):
    """Base User LLM Configuration DTO with separate chat and embedding settings"""
    provider_type: str = 'openai'
    key: Optional[str] = None
    
    # Chat configuration
    chat_endpoint: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_model_name: Optional[str] = None
    
    # Embedding configuration
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model_name: Optional[str] = None
    
    # Thinking configuration
    thinking_model_name: Optional[str] = None
    thinking_endpoint: Optional[str] = None
    thinking_api_key: Optional[str] = None

    # Judge configuration (automatic reinforcement judge)
    judge_model_name: Optional[str] = None
    judge_endpoint: Optional[str] = None
    judge_api_key: Optional[str] = None
    
    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        return result


class CreateUserLLMConfigDTO(BaseUserLLMConfigDTO):
    """Create User LLM Configuration DTO"""
    pass


class UpdateUserLLMConfigDTO(BaseModel):
    """Update User LLM Configuration DTO"""
    provider_type: Optional[str] = None
    key: Optional[str] = None
    
    # Chat configuration
    chat_endpoint: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_model_name: Optional[str] = None
    
    # Embedding configuration
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model_name: Optional[str] = None
    
    # Thinking configuration
    thinking_model_name: Optional[str] = None
    thinking_endpoint: Optional[str] = None
    thinking_api_key: Optional[str] = None

    # Judge configuration (automatic reinforcement judge)
    judge_model_name: Optional[str] = None
    judge_endpoint: Optional[str] = None
    judge_api_key: Optional[str] = None
    
    def dict(self, *args, **kwargs):
        result = super().dict(*args, **kwargs)
        return result


class UserLLMConfigDTO(BaseUserLLMConfigDTO):
    """User LLM Configuration DTO"""
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_model(cls, model):
        """Create DTO from model"""
        if not model:
            return None
        return cls(
            id=model.id,
            provider_type=model.provider_type,
            key=model.key,
            chat_endpoint=model.chat_endpoint,
            chat_api_key=model.chat_api_key,
            chat_model_name=model.chat_model_name,
            embedding_endpoint=model.embedding_endpoint,
            embedding_api_key=model.embedding_api_key,
            embedding_model_name=model.embedding_model_name,
            thinking_model_name=model.thinking_model_name,
            thinking_endpoint=model.thinking_endpoint,
            thinking_api_key=model.thinking_api_key,
            judge_model_name=getattr(model, "judge_model_name", None),
            judge_endpoint=getattr(model, "judge_endpoint", None),
            judge_api_key=getattr(model, "judge_api_key", None),
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class UserLLMConfigListDTO(BaseModel):
    """User LLM Configuration List DTO"""
    items: List[UserLLMConfigDTO]
