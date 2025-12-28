"""
Chat service for handling different types of chat interactions
"""
import logging
from typing import Optional, List, Dict, Any, Union, Iterator, Type
import uuid
from typing import Tuple
from datetime import datetime
import json

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.services.local_llm_service import local_llm_service
from lpm_kernel.api.domains.kernel2.services.message_builder import MultiTurnMessageBuilder
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    SystemPromptStrategy,
    BasePromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service for handling different types of chat interactions"""
    
    def __init__(self):
        """Initialize chat service"""
        # Base strategy chain, must contain at least one base strategy
        self.default_strategy_chain = [BasePromptStrategy, RoleBasedStrategy]
    
    def _get_strategy_chain(
        self,
        request: ChatRequest,
        strategy_chain: Optional[List[Type[SystemPromptStrategy]]] = None,
    ) -> List[Type[SystemPromptStrategy]]:
        """
        Get the strategy chain to use for message building
        
        Args:
            request: Chat request containing message and other parameters
            strategy_chain: Optional list of strategy classes to use
            
        Returns:
            List of strategy classes to use
            
        Raises:
            ValueError: If strategy_chain is empty or None and no default chain is available
        """
        # If custom strategy chain is provided, validate and return
        if strategy_chain is not None:
            if not strategy_chain:
                raise ValueError("Strategy chain cannot be empty")
            if not any(issubclass(s, BasePromptStrategy) for s in strategy_chain):
                raise ValueError("Strategy chain must contain at least one base strategy")
            return strategy_chain
            
        # Use default strategy chain
        result_chain = self.default_strategy_chain.copy()
        
        # Add knowledge enhancement strategy based on request parameters
        if request.enable_l0_retrieval or request.enable_l1_retrieval:
            result_chain.append(KnowledgeEnhancedStrategy)
            
        return result_chain
    
    def _build_messages(
        self,
        request: ChatRequest,
        strategy_chain: Optional[List[Type[SystemPromptStrategy]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages using the specified strategy chain
        
        Args:
            request: Chat request containing message and other parameters
            strategy_chain: Optional list of strategy classes to use. If None, uses default chain
            
        Returns:
            List of message dictionaries
        """
        # Get and validate strategy chain
        final_strategy_chain = self._get_strategy_chain(request, strategy_chain)
        
        # Build messages
        message_builder = MultiTurnMessageBuilder(request, strategy_chain=final_strategy_chain)
        messages = message_builder.build_messages()
        
        # Log debug information
        logger.info("Using strategy chain: %s", [s.__name__ for s in final_strategy_chain])
        logger.info("Final messages for LLM:")
        for msg in messages:
            logger.info(f"Role: {msg['role']}, Content: {msg['content']}")
            
        return messages
    
    def _process_chat_response(self, chunk, full_response: Optional[Any], full_content: str) -> Tuple[Any, str, Optional[str]]:
        """
        Process custom chat_response format data

        Args:
            chunk: Response data chunk
            full_response: Current complete response object
            full_content: Current accumulated content

        Returns:
            Tuple[Any, str, Optional[str]]: (Updated response object, Updated content, Finish reason)
        """
        finish_reason = None
        logger.info(f"Processing custom format response: {chunk}")
        
        # Get content
        content = ""
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            is_done = chunk.get("done", False)
        else:
            content = chunk.content if hasattr(chunk, 'content') else ""
            is_done = chunk.done if hasattr(chunk, 'done') else False
            
        if content:
            full_content += content
            logger.info(f"Added content from custom format, current length: {len(full_content)}")
            
        # Initialize response object (if needed)
        if not full_response:
            full_response = {
                "id": str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": ""
                        },
                        "finish_reason": None
                    }
                ]
            }
            
        # Check if completed
        if is_done:
            finish_reason = 'stop'
            logger.info("Got finish_reason from custom format: stop")
            
        return full_response, full_content, finish_reason

    def _process_openai_response(self, chunk, full_response: Optional[Any], full_content: str) -> Tuple[Any, str, Optional[str]]:
        """
        Process OpenAI format response data

        Args:
            chunk: Response data chunk
            full_response: Current complete response object
            full_content: Current accumulated content

        Returns:
            Tuple[Any, str, Optional[str]]: (Updated response object, Updated content, Finish reason)
        """
        finish_reason = None
        
        if not hasattr(chunk, 'choices'):
            logger.warning(f"Chunk has no choices attribute: {chunk}")
            return full_response, full_content, finish_reason
            
        choices = getattr(chunk, 'choices', None)
        if not choices:
            logger.warning("Chunk has empty choices")
            return full_response, full_content, finish_reason
            
        # Save basic information of the first response
        if full_response is None:
            full_response = {
                "id": getattr(chunk, 'id', str(uuid.uuid4())),
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": getattr(chunk, 'system_fingerprint', None),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": ""
                        },
                        "finish_reason": None
                    }
                ]
            }
        
        # Collect content and finish reason
        choice = choices[0]
        if hasattr(choice, 'delta'):
            delta = choice.delta
            if hasattr(delta, 'content') and delta.content is not None:
                full_content += delta.content
                # logger.info(f"Added content from OpenAI format, current length: {len(full_content)}")
            if choice.finish_reason:
                finish_reason = choice.finish_reason
                logger.info(f"Got finish_reason: {finish_reason}")
                
        return full_response, full_content, finish_reason

    def collect_stream_response(self, response_iterator: Iterator[Dict[str, Any]]):
        """
        Collect streaming response into a complete response

        Args:
            response_iterator: Streaming response iterator

        Returns:
            Complete response dictionary
        """
        logger.info("Starting to collect stream response")
        full_response = None
        full_content = ""
        finish_reason = None
        chunk_count = 0
        
        try:
            for chunk in response_iterator:
                if chunk is None:
                    logger.warning("Received None chunk, skipping")
                    continue
                    
                chunk_count += 1
                # logger.info(f"Processing chunk #{chunk_count}: {chunk}")
                
                # Check if it's a custom format response
                is_chat_response = (
                    (hasattr(chunk, 'type') and chunk.type == 'chat_response') or
                    (isinstance(chunk, dict) and chunk.get("type") == "chat_response")
                )
                
                if is_chat_response:
                    full_response, full_content, chunk_finish_reason = self._process_chat_response(
                        chunk, full_response, full_content
                    )
                else:
                    full_response, full_content, chunk_finish_reason = self._process_openai_response(
                        chunk, full_response, full_content
                    )
                    
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
        
            # logger.info(f"Finished processing all chunks. Total chunks: {chunk_count}")
            # logger.info(f"Final content length: {len(full_content)}")
            # logger.info(f"Final finish_reason: {finish_reason}")
            
            if not full_response:
                logger.error("No valid response collected")
                return None
                
            if not full_content:
                logger.error("No content collected")
                return None
                
            # Update response with complete content
            full_response["choices"][0]["delta"]["content"] = full_content
            if finish_reason:
                full_response["choices"][0]["finish_reason"] = finish_reason
                
            # logger.info(f"Final response object: {full_response}")
            # logger.info(f"Final response content: {full_content}")
            return full_response
            
        except Exception as e:
            logger.error(f"Error collecting stream response: {str(e)}", exc_info=True)
            return None

    def chat(
            self,
            request: ChatRequest,
            strategy_chain: Optional[List[Type[SystemPromptStrategy]]] = None,
            stream: bool = True,
            json_response: bool = False,
            client: Optional[Any] = None,
            model_params: Optional[Dict[str, Any]] = None,
            context: Optional[Any] = None,
            experience_id: Optional[str] = None,
            experience_source: str = "kernel2",
        ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Main chat method supporting both streaming and non-streaming responses
        
        Args:
            request: Chat request containing message and other parameters
            strategy_chain: Optional list of strategy classes to use
            stream: Whether to return a streaming response
            json_response: Whether to request JSON formatted response from LLM
            client: Optional OpenAI client to use. If None, uses local_llm_service.client
            model_params: Optional model specific parameters to override defaults
            context: Optional context to pass to strategies
            
        Returns:
            Either an iterator for streaming responses or a single response dictionary
        """
        logger.info(f"Chat request: {request}")
        # Build messages (system prompt will be first)
        message_builder = MultiTurnMessageBuilder(request, strategy_chain=strategy_chain)
        messages = message_builder.build_messages(context)
        
        # Log debug information
        # logger.info("Using strategy chain: %s", [s.__name__ for s in strategy_chain] if strategy_chain else "default")
        logger.info("Final messages for LLM:")
        for msg in messages:
            logger.info(f"Role: {msg['role']}, Content: {msg['content']}")

        # Use provided client or default local_llm_service.client
        current_client = client if client is not None else local_llm_service.client
        
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        
        # Prepare API call parameters
        # Resolve seed (avoid fixed seed unless caller explicitly requests it)
        seed = None
        if hasattr(request, "metadata") and request.metadata:
            seed = request.metadata.get("seed")
        if seed is None:
            seed = 42

        api_params = {
            "messages": messages,
            "temperature": request.temperature,
            "response_format": {"type": "text"},
            "seed": seed,
            "tools": None,  # Optional: If function calling or similar features are needed
            "tool_choice": None,  # Optional: If function calling or similar features are needed
            "max_tokens": request.max_tokens,
            "stream": stream,
            "model": request.model or "models/lpm",
            "metadata": request.metadata
        }
        
        # Add JSON format requirement (if needed)
        if json_response:
            api_params["response_format"] = {"type": "json_object"}
            
        # Update custom model parameters (if provided)
        if model_params:
            api_params.update(model_params)

        logger.info(f"Current client base URL: {current_client.base_url}")
        # logger.info(f"Using model parameters: {api_params}")
        
        # Save "experience" record early (prompt + params), then update with completion after generation.
        if experience_id:
            try:
                from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store
                now = datetime.now()
                clickhouse_store.insert_experience({
                    "id": experience_id,
                    "source": experience_source,
                    "model": api_params.get("model"),
                    "prompt_messages": json.dumps(messages, ensure_ascii=False),
                    "temperature": int(float(request.temperature) * 1000),
                    "max_tokens": int(request.max_tokens) if request.max_tokens is not None else None,
                    "seed": int(seed) if seed is not None else None,
                    "completion": None,
                    "finish_reason": None,
                    "meta_data": json.dumps(request.metadata or {}, ensure_ascii=False),
                    "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception as e:
                logger.warning(f"Failed to persist chat_experience {experience_id}: {e}", exc_info=True)

        # Call LLM API
        try:
            response = current_client.chat.completions.create(**api_params)

            # If streaming, wrap iterator to accumulate completion and persist on finish.
            if stream and experience_id:
                def _wrapped_iter():
                    full_content = ""
                    finish_reason = None
                    try:
                        for chunk in response:
                            # Accumulate content for OpenAI-like chunks
                            try:
                                if hasattr(chunk, "choices") and chunk.choices:
                                    choice = chunk.choices[0]
                                    if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content:
                                        full_content += choice.delta.content
                                    if getattr(choice, "finish_reason", None):
                                        finish_reason = choice.finish_reason
                                # Custom dict format support (rare in this path)
                                if isinstance(chunk, dict):
                                    content = chunk.get("content")
                                    if content:
                                        full_content += content
                                    if chunk.get("done"):
                                        finish_reason = finish_reason or "stop"
                            except Exception:
                                pass
                            yield chunk
                    finally:
                        try:
                            from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store
                            now = datetime.now()
                            # ClickHouse "update": insert a new version (ReplacingMergeTree)
                            clickhouse_store.insert_experience({
                                "id": experience_id,
                                "source": experience_source,
                                "model": api_params.get("model"),
                                "prompt_messages": json.dumps(messages, ensure_ascii=False),
                                "temperature": int(float(request.temperature) * 1000),
                                "max_tokens": int(request.max_tokens) if request.max_tokens is not None else None,
                                "seed": int(seed) if seed is not None else None,
                                "completion": full_content,
                                "finish_reason": finish_reason,
                                "meta_data": json.dumps(request.metadata or {}, ensure_ascii=False),
                                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                            })
                        except Exception as e:
                            logger.warning(f"Failed to update chat_experience completion {experience_id}: {e}", exc_info=True)

                return _wrapped_iter()

            if not stream:
                logger.info(f"Response: {response.json() if hasattr(response, 'json') else response}")
                if experience_id:
                    try:
                        from lpm_kernel.api.domains.kernel2.services.clickhouse_store import clickhouse_store
                        now = datetime.now()
                        content = None
                        finish_reason = None
                        try:
                            if hasattr(response, "choices") and response.choices:
                                content = response.choices[0].message.content
                                finish_reason = getattr(response.choices[0], "finish_reason", None)
                        except Exception:
                            pass
                        clickhouse_store.insert_experience({
                            "id": experience_id,
                            "source": experience_source,
                            "model": api_params.get("model"),
                            "prompt_messages": json.dumps(messages, ensure_ascii=False),
                            "temperature": int(float(request.temperature) * 1000),
                            "max_tokens": int(request.max_tokens) if request.max_tokens is not None else None,
                            "seed": int(seed) if seed is not None else None,
                            "completion": content,
                            "finish_reason": finish_reason,
                            "meta_data": json.dumps(request.metadata or {}, ensure_ascii=False),
                            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to update chat_experience completion (non-stream) {experience_id}: {e}", exc_info=True)
            return response
            
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}", exc_info=True)
            raise


# Global chat service instance
chat_service = ChatService()
