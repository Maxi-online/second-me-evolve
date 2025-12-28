from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from lpm_kernel.common.repository.database_session import Base


class UserLLMConfig(Base):
    """User-defined LLM configuration model with separate chat and embedding settings"""
    __tablename__ = 'user_llm_configs'

    id = Column(Integer, primary_key=True)
    provider_type = Column(String(50), nullable=False, default='openai', comment='Provider type (e.g., openai)')
    key = Column(String(200), nullable=True, comment='Common API key for provider-specific configurations')
    
    # Chat configuration
    chat_endpoint = Column(String(200), nullable=True, comment='Chat API endpoint')
    chat_api_key = Column(String(200), nullable=True, comment='Chat API key')
    chat_model_name = Column(String(200), nullable=True, comment='Chat model name')
    
    # Embedding configuration
    embedding_endpoint = Column(String(200), nullable=True, comment='Embedding API endpoint')
    embedding_api_key = Column(String(200), nullable=True, comment='Embedding API key')
    embedding_model_name = Column(String(200), nullable=True, comment='Embedding model name')
    
    # Thinking configuration
    thinking_model_name = Column(String(200), nullable=True, comment='Thinking model name')
    thinking_endpoint = Column(String(200), nullable=True, comment='Thinking API endpoint')
    thinking_api_key = Column(String(200), nullable=True, comment='Thinking API key')

    # Judge configuration (automatic reinforcement judge)
    judge_model_name = Column(String(200), nullable=True, comment='Judge model name')
    judge_endpoint = Column(String(200), nullable=True, comment='Judge API endpoint')
    judge_api_key = Column(String(200), nullable=True, comment='Judge API key')
    
    created_at = Column(DateTime, default=datetime.utcnow, comment='Creation time')
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='Update time')

    def __repr__(self):
        return f'<UserLLMConfig {self.id}>'

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'provider_type': self.provider_type,
            'key': self.key,
            'chat_endpoint': self.chat_endpoint,
            'chat_api_key': self.chat_api_key,
            'chat_model_name': self.chat_model_name,
            'embedding_endpoint': self.embedding_endpoint,
            'embedding_api_key': self.embedding_api_key,
            'embedding_model_name': self.embedding_model_name,
            'thinking_model_name': self.thinking_model_name,
            'thinking_endpoint': self.thinking_endpoint,
            'thinking_api_key': self.thinking_api_key,
            'judge_model_name': self.judge_model_name,
            'judge_endpoint': self.judge_endpoint,
            'judge_api_key': self.judge_api_key,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            provider_type=data.get('provider_type', 'openai'),
            key=data.get('key'),
            chat_endpoint=data.get('chat_endpoint'),
            chat_api_key=data.get('chat_api_key'),
            chat_model_name=data.get('chat_model_name'),
            embedding_endpoint=data.get('embedding_endpoint'),
            embedding_api_key=data.get('embedding_api_key'),
            embedding_model_name=data.get('embedding_model_name'),
            thinking_model_name=data.get('thinking_model_name'),
            thinking_endpoint=data.get('thinking_endpoint'),
            thinking_api_key=data.get('thinking_api_key'),
            judge_model_name=data.get('judge_model_name'),
            judge_endpoint=data.get('judge_endpoint'),
            judge_api_key=data.get('judge_api_key'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
