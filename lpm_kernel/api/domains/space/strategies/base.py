"""
Base strategy for space discussion
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..space_dto import SpaceDTO, SpaceMessageDTO
from ..context.context_manager import SpaceContextManager
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.prompt_builder import SystemPromptStrategy


class SpaceBaseStrategy(SystemPromptStrategy, ABC):
    """Base strategy class for Space discussion"""
    
    def __init__(self, base_strategy: Optional[SystemPromptStrategy] = None):
        """
        Initialize strategy
        
        Args:
            base_strategy: Base strategy
        """
        self.base_strategy = base_strategy
        
    def build_prompt(self, request: ChatRequest, context: Optional[SpaceContextManager] = None) -> str:
        """
        Build prompt
        
        Args:
            request: Chat request
            context: Space context manager
            
        Returns:
            Built prompt
            
        Raises:
            ValueError: If context is None
        """
        
        if not context:
            # If there's a base strategy, use it to build the prompt
            if self.base_strategy:
                return self.base_strategy.build_prompt(request, context)
            # Otherwise return default prompt
            return "You are an AI assistant, please help with the user's questions."
            
        return self._build_space_prompt(request, context.space_dto, context)
    
    def _format_message_for_context(self, message_dto: SpaceMessageDTO) -> str:
        """
        Format message for context
        
        Args:
            message_dto: SpaceMessageDTO instance
            
        Returns:
            Formatted message text
        """
        role = "Host" if message_dto.role == "host" else "Participant"
        return f"{role} {message_dto.sender_endpoint.split('/')[-2]}: {message_dto.content}"
        
    def _build_context_from_messages(self, messages_dto: List[SpaceMessageDTO]) -> str:
        """
        Build context from message list, organized by rounds
        
        Args:
            messages_dto: List of messages
            
        Returns:
            Built context text organized by rounds
        """
        # form messages by rounds
        messages_by_round = {}
        for message_dto in messages_dto:
            round_num = message_dto.round
            if round_num not in messages_by_round:
                messages_by_round[round_num] = []
            messages_by_round[round_num].append(message_dto)
        
        # build context by rounds
        context_parts = []
        for round_num in sorted(messages_by_round.keys()):
            # add round title
            if round_num == 0:
                context_parts.append("--- Opening ---")
            else:
                context_parts.append(f"\n--- Round {round_num} ---")
            
            # add all info in this round
            for message_dto in messages_by_round[round_num]:
                context_parts.append(self._format_message_for_context(message_dto))
        
        return "\n".join(context_parts)
    
    def _get_space_info(self, space_dto: SpaceDTO) -> str:
        """
        Get Space information
        
        Args:
            space_dto: Space DTO instance
            
        Returns:
            Space information text
        """
        return f"""Discussion Topic: {space_dto.title}
Discussion Objective: {space_dto.objective}"""

    @abstractmethod
    def _build_space_prompt(self, request: ChatRequest, space_dto: SpaceDTO, context_manager: SpaceContextManager) -> str:
        """
        Concrete prompt building logic to be implemented by subclasses
        
        Args:
            request: Chat request
            space_dto: Space DTO instance
            context_manager: Space context manager
            
        Returns:
            Built prompt
        """
        pass
