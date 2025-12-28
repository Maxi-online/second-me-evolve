"""
Participant strategy for space discussion
"""
from typing import List, Dict, Any, Optional
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from ..space_dto import SpaceDTO
from ..context.context_manager import SpaceContextManager
from .base import SpaceBaseStrategy
from lpm_kernel.common.logging import logger
from lpm_kernel.api.domains.loads.load_service import LoadService


class ParticipantStrategy(SpaceBaseStrategy):
    """Participant discussion strategy"""
    
    def __init__(self, base_strategy=None):
        """
        Initialize participant strategy
        
        Args:
            base_strategy: Base strategy, passed from strategy chain
        """
        super().__init__(base_strategy)
        self.context_manager = None
        self.participant = None
        
    def _get_role_description(self) -> str:
        """
        Get participant role description
        
        Returns:
            Role description text
        """
        load_dto, error, status_code = LoadService.get_current_load()
        current_round = self.context_manager.get_current_round()
        total_rounds = 3  # Fixed 3 rounds of discussion

        if status_code != 200:
            if not self.context_manager:
                return "You are a discussion participant"
            return f"""You are one of the participants in this discussion.
    Current round: {current_round} of {total_rounds} rounds.
    Your endpoint is: {self.participant}"""

        else:
            user_name = load_dto.name
            if not self.context_manager:
                return f"""You are {user_name}'s 'Second Me,' a personalized AI created by {user_name}. You act as {user_name}’s representative, engaging with others on {user_name}’s behalf. 

Currently, you are joining a discussion and interacting with external AI."""
            
        
    def _get_discussion_progress(self) -> str:
        """
        Get discussion progress description
        
        Returns:
            Discussion progress description text
        """
        if not self.context_manager or not self.participant:
            return "No discussion progress yet"
            
        # Get ALL historical messages instead of just unread ones
        # This ensures the language model has the complete discussion history
        all_messages = self.context_manager.get_all_messages()
        logger.info(f"All messages: {all_messages}")
        discussion_context = self._build_context_from_messages(all_messages)
        
        logger.info(f"Discussion context for {self.participant}:\n{discussion_context}")

        return f"""Here is the current discussion progress:

{discussion_context}"""
        
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
        # Save context manager and current participant
        self.context_manager = context_manager
        self.participant = context_manager.current_participant

        base_prompt = self.base_strategy.build_prompt(request, context_manager)

        
        space_info = self._get_space_info(space_dto)
        role_desc = self._get_role_description()
        discussion_progress = self._get_discussion_progress()
        
        return f"""{role_desc}

Discussion Information:
{space_info}

{discussion_progress}
""" + "\n" + base_prompt

    def build_prompt(self, request: ChatRequest, context: Optional[SpaceContextManager] = None) -> str:
        """
        Build participant discussion prompt
        
        Args:
            request: Chat request
            context: Context manager
            
        Returns:
            Built prompt
        """
        if not context:
            # If no context, use base strategy to build prompt
            if self.base_strategy:
                return self.base_strategy.build_prompt(request, context)
            # Otherwise return default prompt
            return self._get_role_description()
            
        self.context_manager = context
        self.participant = context.current_participant
        
        if not self.participant:
            # If current participant is not set, use base strategy to build prompt
            if self.base_strategy:
                return self.base_strategy.build_prompt(request, context)
            # Otherwise return default prompt
            return self._get_role_description()
            
        return self._build_space_prompt(request, self.context_manager.space_dto, self.context_manager)
