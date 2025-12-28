from typing import List, Dict, Any, Type, Optional

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    SystemPromptBuilder,
    SystemPromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)

class MessageBuilder:
    """Base class for building chat messages"""
    
    def build_messages(self, context: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Build messages for chat completion"""
        raise NotImplementedError()


class MultiTurnMessageBuilder(MessageBuilder):
    """Message builder for multi-turn chat"""
    
    def __init__(self, chat_request: ChatRequest, strategy_chain: List[Type[SystemPromptStrategy]] = None):
        """
        Initialize the builder with a chat request and optional strategy chain.
        
        Args:
            chat_request: The chat request to build messages for
            strategy_chain: List of strategy classes in the order they should be applied.
                          Default is [RoleBasedStrategy, KnowledgeEnhancedStrategy]
        """
        self.chat_request = chat_request
        self.strategy_chain = strategy_chain or [RoleBasedStrategy, KnowledgeEnhancedStrategy]
        
    def build_messages(self, context: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Build messages for multi-turn chat"""

        # Since we now use standard OpenAI format, directly return the messages
        # without any transformation
        # if self.chat_request.messages:
        #     # get messages' system_prompt, history and tmp message
        #     system_messages = []
        #     history = []
        #     current_message = None
            
        #     for msg in self.chat_request.messages:
        #         role = msg.get("role", "")
        #         content = msg.get("content", "")
                
        #         if role == "system":
        #             system_messages.append(content)
        #         elif role == "user" or role == "assistant":
        #             # if current message has been set, add to history
        #             if current_message is not None and role == "user":
        #                 history.append({"role": "user", "content": current_message})
        #                 current_message = content
        #             elif current_message is not None and role == "assistant":
        #                 history.append({"role": "assistant", "content": content})
        #             else:
        #                 # the first non-system message is the current user message
        #                 if role == "user" and current_message is None:
        #                     current_message = content
        #                 else:
        #                     # else add to chat history
        #                     history.append({"role": role, "content": content})
            
        #     # update chat_request related fields
        #     if system_messages:
        #         self.chat_request.system_prompt = "\n".join(system_messages)
            
        #     if history:
        #         self.chat_request.history = [
        #             ChatMessage(role=msg["role"], content=msg["content"]) 
        #             for msg in history
        #         ]
            
        #     if current_message:
        #         self.chat_request.message = current_message

        # IMPORTANT:
        # - Do NOT mutate the original list in-place (it breaks history and training logs).
        # - System prompt must be the first message (not appended at the end).
        original_messages = list(self.chat_request.messages or [])
        non_system_messages = [m for m in original_messages if m.get("role") != "system"]

        # 1) Build system prompt
        builder = SystemPromptBuilder()
        
        # Build strategy chain from bottom up
        current_strategy = None
        # iter from the most basic to the most advanced
        for strategy_class in self.strategy_chain:
            if current_strategy is None:
                # BasePromptStrategy
                current_strategy = strategy_class()
            else:
                # use tmp strategy to create new strategy
                current_strategy = strategy_class(base_strategy=current_strategy)
        
        if current_strategy is None:
            raise ValueError("No strategy provided")
            
        builder.set_strategy(current_strategy)
        system_prompt = builder.build_prompt(self.chat_request, context)
        system_message = {"role": "system", "content": system_prompt or ""}

        # 2) Return prepared messages with system first
        return [system_message, *non_system_messages]
