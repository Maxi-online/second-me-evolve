"""
System prompt builder and related strategies
"""
from typing import Optional, Any
import logging

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.role_service import role_service
from lpm_kernel.api.domains.kernel2.services.knowledge_service import (
    default_retriever,
    default_l1_retriever,
)
from lpm_kernel.L2.training_prompt import CONTEXT_PROMPT, MEMORY_PROMPT, JUDGE_PROMPT


logger = logging.getLogger(__name__)


class SystemPromptStrategy:
    """Base class for system prompt building strategies"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build system prompt"""
        raise NotImplementedError()


class BasePromptStrategy(SystemPromptStrategy):
    """Most basic system prompt building strategy"""
    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Return the basic system prompt"""
        # Try to find a system message in the messages
        if request.messages:
            for message in request.messages:
                if message.get('role') == 'system':
                    return message.get('content', '')
        
        # Default empty prompt if no system message found
        return ""

class ContextEnhancedStrategy(SystemPromptStrategy):
    """Context-enhanced system prompt building strategy"""
    def build_prompt(self, request: ChatRequest) -> str:
        """Build context-enhanced system prompt"""
        base_prompt = CONTEXT_PROMPT
        return base_prompt

class ContextCriticStrategy(SystemPromptStrategy):
    """Context-critic system prompt building strategy"""
    def build_prompt(self, request: ChatRequest) -> str:
        """Build context-critic system prompt"""
        base_prompt = JUDGE_PROMPT
        return base_prompt

class RoleBasedStrategy(SystemPromptStrategy):
    """Role-based system prompt building strategy"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build system prompt based on role"""
        # Get role_id from metadata if available
        role_id = None
        if hasattr(request, 'metadata') and request.metadata:
            role_id = request.metadata.get('role_id')
        
        if role_id:
            role = role_service.get_role_by_uuid(role_id)
            if role:
                prompt = role.system_prompt
                logger.info(f"RoleBasedStrategy (from role): {prompt}")
                return prompt
                
        prompt = self.base_strategy.build_prompt(request, context)
        # logger.info(f"RoleBasedStrategy (from base): {prompt}")
        return prompt


class KnowledgeEnhancedStrategy(SystemPromptStrategy):
    """Knowledge-enhanced system prompt building strategy"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def get_user_message(self, request: ChatRequest) -> str:
        """
        Get the last user message from messages field.
        """
        if request.messages:
            # Find the last message with role='user'
            for message in reversed(request.messages):
                if message.get('role') == 'user':
                    return message.get('content', '')
        
        return ''

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        """Build knowledge-enhanced system prompt"""
        base_prompt = self.base_strategy.build_prompt(request, context)
        
        logger.info(f"KnowledgeEnhancedStrategy request: {request}")
        logger.info(f"KnowledgeEnhancedStrategy (from base): {base_prompt}")
        
        # Add knowledge retrieval results if enabled
        knowledge_sections = []
        user_message = self.get_user_message(request)
        
        # Get configuration from metadata if available
        enable_l0_retrieval = False
        enable_l1_retrieval = False
        role_id = None
        
        if hasattr(request, 'metadata') and request.metadata:
            enable_l0_retrieval = request.metadata.get('enable_l0_retrieval', False)
            enable_l1_retrieval = request.metadata.get('enable_l1_retrieval', False)
            role_id = request.metadata.get('role_id')
        
        # if role exists, role config has priority
        if role_id:
            role = role_service.get_role_by_uuid(role_id)
            if role:
                if role.enable_l0_retrieval:
                    l0_knowledge = default_retriever.retrieve(user_message)
                    if l0_knowledge:
                        knowledge_sections.append(f"Role knowledge:\n{l0_knowledge}")
                if role.enable_l1_retrieval:
                    l1_knowledge = default_l1_retriever.retrieve(user_message)
                    if l1_knowledge:
                        knowledge_sections.append(f"Reference shades:\n{l1_knowledge}")
        else:
            # Retrieve L0 knowledge if enabled
            if enable_l0_retrieval:
                l0_knowledge = default_retriever.retrieve(user_message)
                if l0_knowledge:
                    knowledge_sections.append(f"Reference knowledge:\n{l0_knowledge}")
            
            # Retrieve L1 knowledge if enabled
            if enable_l1_retrieval:
                l1_knowledge = default_l1_retriever.retrieve(user_message)
                if l1_knowledge:
                    knowledge_sections.append(f"Reference shades:\n{l1_knowledge}")
            
        if knowledge_sections:
            if len(base_prompt) == 0:
                prompt = "\n\n".join(knowledge_sections)
            else:
                prompt = base_prompt + "\n\n" + "\n\n".join(knowledge_sections)
            logger.info(f"KnowledgeEnhancedStrategy (with knowledge): {prompt}")
            return prompt
            
        # logger.info(f"KnowledgeEnhancedStrategy (no knowledge found): {base_prompt}")
        return base_prompt


class SystemPromptBuilder:
    """System prompt builder"""
    def __init__(self):
        self.strategy: Optional[SystemPromptStrategy] = None

    def set_strategy(self, strategy: SystemPromptStrategy):
        self.strategy = strategy

    def build_prompt(self, request: ChatRequest, context: Optional[Any] = None) -> str:
        if not self.strategy:
            raise ValueError("No strategy set for SystemPromptBuilder")
        prompt = self.strategy.build_prompt(request, context)
        # logger.info(f"Final system prompt: {prompt}")
        return prompt
