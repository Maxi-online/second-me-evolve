"""
Discussion service implementation
"""
from calendar import c
import logging
from typing import Optional, Any
from datetime import datetime
import uuid

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.space.space_dto import SpaceDTO, SpaceMessageDTO
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.prompt_builder import BasePromptStrategy, KnowledgeEnhancedStrategy
from lpm_kernel.configs.config import Config
from urllib.parse import urlparse, urlunparse
from openai import OpenAI


from ..context.context_manager import SpaceContextManager
from ..context.factory import SpaceContextManagerFactory
from ..strategies import (
    HostOpeningStrategy,
    HostSummaryStrategy,
    ParticipantStrategy,
)

logger = logging.getLogger(__name__)


class DiscussionService:
    """Discussion service, handles discussion flow in Space"""
    
    def __init__(self):
        """
        Initialize discussion service
        
        Args:
            context_manager_factory: Context manager factory
        """
        self._context_manager_factory = SpaceContextManagerFactory()
        self.max_rounds = 3  # Fixed 3 rounds of discussion
        
    def start_discussion(self, space_dto: SpaceDTO) -> dict:
        """
        Start Space discussion
        
        Args:
            space_dto: Space DTO object
            
        Returns:
            Dictionary containing discussion results, including all messages and summary
        """
            
        # Create context manager
        context_manager = self._context_manager_factory.create_context_manager(space_dto)
        
        # Process discussion flow
        success = self._run_discussion(space_dto, context_manager)
        if not success:
            return {"success": False}
            
        # Get all messages and summary
        messages = context_manager.get_all_messages()
        summary = None
        if messages and len(messages) > 0:
            # The last message is usually the summary
            summary = messages[-1].content if messages[-1].role == "host" else None
            
        return {
            "success": True,
            "messages": [msg.to_dict() for msg in messages],
            "summary": summary
        }
        
    def _run_discussion(self, space_dto: SpaceDTO, context_manager: SpaceContextManager) -> bool:
        """
        Run discussion flow
        
        Args:
            space_dto: Space DTO object
            context_manager: Context manager
            
        Returns:
            Whether the discussion completed successfully
        """
        try:
            # 1. Host opening
            opening_message = self._process_host_opening(context_manager)
            if not opening_message:
                logger.error("Discussion failed at opening")
                return False
                
            # 2. Multiple rounds of discussion
            for round_num in range(self.max_rounds):
                context_manager.advance_round()
                logger.info(f"Starting discussion round {round_num + 1}" + "="*20)
                
                # Each participant speaks in turn
                for participant in space_dto.participants:
                    message = self._process_participant_discussion(
                        context_manager, participant
                    )
                    if not message:
                        logger.warning(f"Participant {participant} failed to respond in round {round_num + 1}")
                        continue
                        
            # 3. Host summary
            summary_message = self._process_host_summary(context_manager)
            if not summary_message:
                logger.error("Discussion failed at summary")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Discussion failed: {str(e)}")
            return False
            
    def _get_client_for_endpoint(self, endpoint: str) -> Optional[Any]:
        """
        Get corresponding client based on endpoint
        
        Args:
            endpoint: Endpoint URL
            
        Returns:
            Client for the corresponding endpoint, returns None for local endpoint (uses default client)
        """
        
        config = Config.from_env()
        
        # Get local service URL
        local_url = config.get("LOCAL_LLM_SERVICE_URL", "")
        
        # If the endpoint matches the local URL, return None (use default client)
        if local_url and endpoint.startswith(local_url):
            return None
        
        # Get API prefix configuration
        api_prefix = config.get("LLM_API_PREFIX")  # Default value /api, does not include /chat, as it will be added automatically
        
        # Convert URL
        parsed_url = urlparse(endpoint)
        
        # Extract instance_id from path if it matches the pattern /{name}/{instance_id}
        path_parts = parsed_url.path.strip('/').split('/')
        instance_id = None
        
        # Check if the path has the expected format
        if len(path_parts) == 2:
            # The second part is the instance_id
            instance_id = path_parts[1]
            logger.debug(f"Extracted instance_id: {instance_id} from path: {parsed_url.path}")
        
        # Construct a new path, adding the API prefix
        if instance_id:
            # If we have an instance_id, use it in the new path
            new_path = f"{api_prefix}/{instance_id}"
        else:
            # Otherwise, just add the API prefix to the original path
            new_path = endpoint
        
        # Reassemble the URL
        api_endpoint = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            new_path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))
        
        # Create a new client connection to the specified endpoint
        try:
            client = OpenAI(
                base_url=api_endpoint,
                api_key="sk-no-key-required"
            )
            return client
        except Exception as e:
            logger.error(f"Failed to create client for endpoint {endpoint}: {str(e)}")
            return None
            
    def _process_host_opening(self, context_manager: SpaceContextManager) -> Optional[SpaceMessageDTO]:
        """
        Process host opening
        
        Args:
            context_manager: Context manager
            
        Returns:
            Created opening message, returns None if failed
        """
        try:
            # logger.info("Starting host opening process")
            request = ChatRequest(
                messages=[{"role": "user", "content": "Please start hosting the discussion"}],
                metadata={"enable_l0_retrieval": True}
            )
            
            # Get the host endpoint's corresponding client
            host_endpoint = context_manager.space_dto.host
            # logger.info(f"Using host endpoint: {host_endpoint}")
            client = self._get_client_for_endpoint(host_endpoint)
            
            # Use chat_service to process the request, passing in HostOpeningStrategy as the strategy chain
            # logger.info("Sending chat request with HostOpeningStrategy")
            stream_response = chat_service.chat(request,
                                         strategy_chain=[BasePromptStrategy, KnowledgeEnhancedStrategy, HostOpeningStrategy],
                                         context=context_manager,
                                         stream=True,
                                         client=client)
            
            # logger.info("Collecting stream response")
            response = chat_service.collect_stream_response(stream_response)

            logger.info(f"Final response object: {response}")
            
            if not response:
                logger.error("Host opening failed: no response returned")
                return None
                
            if "choices" not in response or not response["choices"]:
                logger.error("Host opening failed: no choices in response")
                return None
                
            choice = response["choices"][0]
            if "delta" not in choice:
                logger.error("Host opening failed: no delta in first choice")
                return None
                
            content = choice["delta"]["content"]
            logger.info(f"Got response content length: {len(content) if content else 0}")
            
            if not content:
                logger.error("Host opening failed: no content in message")
                return None
                
            # Create and save the opening message
            logger.info("Creating opening message")
            result = context_manager.create_message(
                sender_endpoint=context_manager.space_dto.host,
                content=content,
                message_type="opening",
                round=0
            )
            logger.info("Opening message created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Host opening failed: {str(e)}", exc_info=True)
            return None

    def _process_participant_discussion(
        self, context_manager, participant: str
    ) -> Optional[SpaceMessageDTO]:
        """
        Process participant discussion
        
        Args:
            context_manager: Context manager
            participant: Participant endpoint
            
        Returns:
            Created discussion message, returns None if failed
        """
        try:
            # Set the current participant
            context_manager.current_participant = participant
            
            # Create a chat request
            request = ChatRequest(
                messages=[{"role": "user", "content": "Please share your thoughts"}],
                metadata={"enable_l0_retrieval": True}
            )
            
            # Get the participant endpoint's corresponding client
            client = self._get_client_for_endpoint(participant)
            
            # Log the context for this conversation round
            # visible_messages = context_manager.get_context_for_participant(participant)
            # logger.info(f"Context for participant {participant} in round {context_manager.get_current_round()}:")
            # for msg in visible_messages:
            #     logger.info(f"  Message from {msg.sender_endpoint}: {msg.content}..." if len(msg.content) > 100 else f"  Message from {msg.sender_endpoint}: {msg.content}")

            # Use chat_service to process the request, passing in ParticipantStrategy as the strategy chain
            response = chat_service.collect_stream_response(chat_service.chat(
                request, 
                strategy_chain=[BasePromptStrategy,KnowledgeEnhancedStrategy, ParticipantStrategy], 
                context=context_manager,
                client=client,
                stream=True
            ))
            
            if not response or "choices" not in response or not response["choices"]:
                logger.error(f"Participant discussion failed for {participant}: empty response")
                return None
                
            choice = response["choices"][0]
            if "delta" not in choice:
                logger.error(f"Participant discussion failed for {participant}: no delta in first choice")
                return None
                
            content = choice.get("delta", {}).get("content")
            
            if not content:
                logger.error(f"Participant discussion failed for {participant}: no content in message")
                return None
            
            logger.info(f"Participant {participant} discussion content: {content}")
            # Create and save the discussion message
            return context_manager.create_message(
                sender_endpoint=participant,
                content=content,
                message_type="discussion",
                round=context_manager.get_current_round()
            )
            
        except Exception as e:
            logger.error(f"Participant discussion failed for {participant}: {str(e)}", exc_info=True)
            return None
            
    def _process_host_summary(self, context_manager) -> Optional[SpaceMessageDTO]:
        """
        Process host summary
        
        Args:
            context_manager: Context manager
            
        Returns:
            Created summary message, returns None if failed
        """
        try:
            request = ChatRequest(
                messages=[{"role": "user", "content": "Please summarize this discussion"}]
            )
            
            # Get the host endpoint's corresponding client
            host_endpoint = context_manager.space_dto.host
            client = self._get_client_for_endpoint(host_endpoint)
            
            # Use chat_service to process the request, passing in HostSummaryStrategy as the strategy chain
            response = chat_service.collect_stream_response(chat_service.chat(
                request, 
                strategy_chain=[HostSummaryStrategy], 
                context=context_manager,
                client=client,
                stream=True
            ))
            
            if not response or "choices" not in response or not response["choices"]:
                logger.error("Host summary failed: empty response")
                return None
                
            choice = response["choices"][0]
            if "delta" not in choice:
                logger.error("Host summary failed: no delta in first choice")
                return None
                
            content = choice.get("delta", {}).get("content")
            
            if not content:
                logger.error("Host summary failed: no content in message")
                return None
                
            # Create and save the summary message
            summary_message = context_manager.create_message(
                sender_endpoint=context_manager.space_dto.host,
                content=content,
                message_type="summary",
                round=0
            )
            
            # Update the Space's conclusion
            context_manager.space_dto.conclusion = content
            
            return summary_message
            
        except Exception as e:
            logger.error(f"Host summary failed: {str(e)}", exc_info=True)
            return None
