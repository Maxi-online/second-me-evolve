"""
Host strategies for space discussion
"""
from typing import List, Dict, Any, Optional
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from ..space_dto import SpaceDTO
from ..context.context_manager import SpaceContextManager
from .base import SpaceBaseStrategy
from lpm_kernel.api.domains.loads.load_service import LoadService


class HostOpeningStrategy(SpaceBaseStrategy):
    """Host opening strategy"""
    
    def __init__(self, base_strategy: Optional[SpaceBaseStrategy] = None):
        """
        Initialize host opening strategy
        
        Args:
            base_strategy: Base strategy, passed from strategy chain
        """
        super().__init__(base_strategy)
    
    def _build_space_prompt(self, request: ChatRequest, space_dto: SpaceDTO, context_manager: SpaceContextManager) -> str:
        """
        Build prompt for host opening
        
        Args:
            request: Chat request
            space_dto: Space DTO instance
            context_manager: Space context manager
            
        Returns:
            Built prompt
        """
        participants = space_dto.participants

        base_prompt = self.base_strategy.build_prompt(request, context_manager)

        load_dto, error, status_code = LoadService.get_current_load()
        if status_code != 200:
        
            return f"""You are the host of this discussion. Please organize an opening statement and present your first perspective based on the following information:

    {self._get_space_info(space_dto)}

    Participant List:
    {chr(10).join([f"- {p}" for p in participants])}

    Please structure your response as follows:

    1. Opening Statement:
    - Welcome participants
    - Introduce discussion topic and objectives
    - Explain discussion rules (each person speaks in turn, 3 rounds of discussion)

    2. Your First Perspective:
    - Analyze based on the topic
    - Present your initial thoughts
    - Guide the discussion direction appropriately

    Please ensure your response is:
    - Clear and concise
    - Guiding and directive
    - Able to stimulate participants' thinking and desire to discuss""" + "\n\n" + base_prompt
        else:
            user_name = load_dto.name
            return f"""You are {user_name}'s 'Second Me,' a personalized AI created by {user_name}. You act as {user_name}’s representative, engaging with others on {user_name}’s behalf. 

Currently, you are joining a discussion and interacting with external AI. 

{self._get_space_info(space_dto)}

Participant List:
    {chr(10).join([f"- {p.split('/')[-2]}" for p in participants])}

Please follow this speaking order:

1. Begin with an opening statement welcoming all participants to the discussion.
2. Explain the discussion rules: each participant speaks in turn, and the discussion will end after 3 rounds.
3. From {user_name}'s perspective, present your views on the discussion topic and tasks that align with {user_name}'s viewpoints.

Remember that you are representing {user_name} in this conversation. All your statements should be based on {user_name}'s relevant experiences and background. Your response should be clean and clearly articulated.
""" + "\n\n" + base_prompt


class HostSummaryStrategy(SpaceBaseStrategy):
    """Host summary strategy"""
    
    def __init__(self, base_strategy: Optional[SpaceBaseStrategy] = None):
        """
        Initialize host summary strategy
        
        Args:
            base_strategy: Base strategy, passed from strategy chain
        """
        super().__init__(base_strategy)
    
    def _build_space_prompt(self, request: ChatRequest, space_dto: SpaceDTO, context_manager: SpaceContextManager) -> str:
        """
        Build prompt for host summary
        
        Args:
            request: Chat request
            space_dto: Space DTO instance
            context_manager: Space context manager
            
        Returns:
            Built prompt
        """
        messages = context_manager.get_all_messages()
        discussion_context = self._build_context_from_messages(messages)
        
        return f"""You are the host of this discussion. Please summarize the discussion based on the following information:

{self._get_space_info(space_dto)}

Discussion Record:
{discussion_context}

Please structure your summary as follows:

1. Key Discussion Points:
   - List main perspectives
   - Highlight important insights

2. Consensus and Differences:
   - Summarize agreements reached
   - Point out existing differences

3. Conclusion and Recommendations:
   - Provide recommendations based on discussion
   - Suggest possible next steps

Please ensure your summary is:
- Objective and fair
- Focused on key points
- Constructive"""
