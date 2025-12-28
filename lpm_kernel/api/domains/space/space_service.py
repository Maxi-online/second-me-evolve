"""
Space service layer, handles all Space-related business logic
"""
from typing import List, Optional
import uuid
from datetime import datetime
import threading
import os
import requests

from lpm_kernel.configs.config import Config
from lpm_kernel.api.domains.space.space_dto import SpaceDTO, SpaceMessageDTO, CreateSpaceDTO
from lpm_kernel.models.space import Space as DBSpace
from lpm_kernel.common.logging import logger
from .space_repository import SpaceRepository
from .services.discussion_service import DiscussionService
from lpm_kernel.api.domains.loads.load_service import LoadService

class SpaceService:
    def __init__(self):
        self._repository = SpaceRepository()
        self._discussion_service = DiscussionService()

    def create_space(self, title: str, objective: str, host: str, participants: List[str]) -> SpaceDTO:
        """
        Create a new Space
        
        Args:
            title: Space topic
            objective: Space objective
            host: Host endpoint
            participants: List of other participant endpoints
            
        Returns:
            SpaceDTO: Created Space DTO object
        """
        # Ensure the participants list includes the host
        all_participants = list(set([host] + participants))
        
        # Create Space
        space_dto = SpaceDTO(
            id=str(uuid.uuid4()),
            title=title,
            objective=objective,
            host=host,
            participants=all_participants,
            create_time=datetime.utcnow().isoformat(),
            status=1,  # Initial status
            messages=[],
            conclusion=None
        )
        
        # Save to database
        return self._repository.create(space_dto)
        
    def get_space(self, space_id: str) -> Optional[SpaceDTO]:
        """
        Get Space information
        
        Args:
            space_id: Space unique identifier
            
        Returns:
            Optional[SpaceDTO]: Space DTO object, returns None if not found
        """
        # Get Space
        space_dto = self._repository.get(space_id)
        if not space_dto:
            return None
            
        # Ensure Space includes messages
        if not space_dto.messages:
            # Get all messages for the Space
            messages = self._repository.get_messages(space_id)
            space_dto.messages = messages
            
        return space_dto
        
    def get_all_spaces(self, host: Optional[str] = None) -> List[SpaceDTO]:
        """
        Get all Spaces list
        
        Args:
            host: Optional host endpoint filter
            
        Returns:
            List[SpaceDTO]: List of Space DTO objects
        """
        return self._repository.list_spaces(host)

    def start_discussion(self, space_id: str) -> bool:
        """
        Start Space discussion

        Args:
            space_id: Space ID

        Returns:
            Whether the discussion was successfully started
        """
        # Get Space
        space_dto = self._repository.get(space_id)
        if not space_dto:
            raise ValueError(f"Space not found: {space_id}")

        # Check current status, don't allow restart if already in discussion
        if space_dto.status == SpaceDTO.STATUS_DISCUSSING:
            return False

        # Set status to discussing
        space_dto.status = SpaceDTO.STATUS_DISCUSSING
        self._repository.save(space_dto)

        # Start async discussion thread
        thread = threading.Thread(target=self._run_discussion_async, args=(space_id,))
        thread.start()

        return True

    def _run_discussion_async(self, space_id: str) -> None:
        """
        Run discussion process asynchronously

        Args:
            space_id: Space ID
        """
        try:
            # Get Space
            space_dto = self._repository.get(space_id)
            if not space_dto:
                return

            # Start discussion
            result = self._discussion_service.start_discussion(space_dto)
            
            # Update Space status and messages
            space_dto = self._repository.get(space_id)  # Refresh to ensure latest data
            if not space_dto:
                return

            if result.get("success", False):
                messages_dict = result.get("messages", [])
                summary = result.get("summary", None)
                
                # Convert dictionaries to SpaceMessageDTO objects
                messages = []
                for msg_dict in messages_dict:
                    msg = SpaceMessageDTO(
                        id=msg_dict.get("id"),
                        space_id=msg_dict.get("space_id"),
                        sender_endpoint=msg_dict.get("sender_endpoint"),
                        content=msg_dict.get("content"),
                        message_type=msg_dict.get("message_type"),
                        round=msg_dict.get("round"),
                        create_time=datetime.fromisoformat(msg_dict.get("create_time")),
                        role=msg_dict.get("role")
                    )
                    messages.append(msg)
                
                # Update Space messages and conclusion
                if messages:
                    space_dto.messages = messages
                if summary:
                    space_dto.conclusion = summary
                    
                # Set status to discussion finished
                space_dto.status = SpaceDTO.STATUS_FINISHED
            else:
                # If discussion failed, set status to interrupted
                space_dto.status = SpaceDTO.STATUS_INTERRUPTED
                
            # Save updated Space
            self._repository.save(space_dto)

        except Exception as e:
            # When exception occurs, set status to interrupted
            try:
                space_dto = self._repository.get(space_id)
                if space_dto:
                    space_dto.status = SpaceDTO.STATUS_INTERRUPTED
                    self._repository.save(space_dto)
            except:
                pass  # If updating status also fails, ignore

    def get_discussion_status(self, space_id: str) -> dict:
        """
        Get discussion status

        Args:
            space_id: Space ID

        Returns:
            Status information dictionary
        """
        space_dto = self._repository.get(space_id)
        if not space_dto:
            raise ValueError(f"Space not found: {space_id}")

        messages = self._repository.get_messages(space_id)

        return {
            "current_round": max([msg.round for msg in messages]) if messages else 0,
            "total_rounds": 3,  # Fixed 3 rounds of discussion
            "message_count": len(messages),
            "is_completed": space_dto.status == SpaceDTO.STATUS_FINISHED,
            "last_message_time": messages[-1].create_time if messages else None
        }

    def add_message(self, space_id: str, content: str, sender_endpoint: str,
                    message_type: str, round: int = 0) -> Optional[SpaceMessageDTO]:
        """
        Add discussion message

        Args:
            space_id: Space ID
            content: Message content
            sender_endpoint: Sender endpoint
            message_type: Message type (opening/discussion/summary)
            round: Discussion round, default is 0

        Returns:
            Added message, returns None if Space not found
        """
        message = SpaceMessageDTO(
            id=str(uuid.uuid4()),
            space_id=space_id,
            sender_endpoint=sender_endpoint,
            content=content,
            message_type=message_type,
            round=round
        )

        return self._repository.add_message(message)

    def share_space(self, space_id: str) -> str:
        """
        Share a Space to remote service and get a share ID
        
        Args:
            space_id: Space ID
            
        Returns:
            str: Share ID generated by remote service
            
        Raises:
            ValueError: When Space is not found or not in finished status
        """
        # 1. Validate Space status
        space_dto = self._repository.get(space_id)
        if not space_dto:
            raise ValueError(f"Space not found: {space_id}")
            
        if space_dto.status != SpaceDTO.STATUS_FINISHED:
            raise ValueError("Only spaces with finished status can be shared")
            
        # 2. Get remote service URL
        config = Config.from_env()
        registry_url = config.get('REGISTRY_SHARE_SPACE_SERVICE_URL')
        if not registry_url:
            raise ValueError("REGISTRY_SHARE_SPACE_SERVICE_URL not configured in environment")
        
        load_dict, error, status_code = LoadService.get_current_load()
        if error:
            raise ValueError(f"Failed to get current load: {error}")
        if not load_dict:
            raise ValueError("Current load not found")
        if not load_dict.instance_id:
            raise ValueError("Current load has no instance_id")

        # 3. Prepare request data
        request_data = {
            "space_share_id": space_dto.space_share_id,
            "instance_id": load_dict.instance_id,
            "space_data": space_dto.model_dump(mode="json")  # use mode="json" to transfer datetime to ISO format str
        }
        
        logger.info(f"Request data: {request_data}")

        
        # 4. Send request to remote service
        try:
            response = requests.post(
                registry_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
           
            
            # 5. Process response
            if response.status_code != 200:
                logger.error(f"Failed to register space: {response.text}")
                raise ValueError("Failed to register space to remote service")
                
            # Parse response data
            response_data = response.json()

            logger.info(f"response data: {response_data}")

            # Check if response follows APIResponse format
            if "code" not in response_data:
                logger.error(f"Invalid response format: {response_data}")
                raise ValueError("Remote service returned invalid response format")
                
            # Check if request was successful (code=0 means success)
            if response_data.get("code") != 0:
                error_msg = response_data.get("message", "Unknown error")
                logger.error(f"Remote service error: {error_msg}")
                raise ValueError(f"Remote service error: {error_msg}")
                
            # Get share ID from data field
            data = response_data.get("data", {})
            if not data or not isinstance(data, dict):
                logger.error("No data in response")
                raise ValueError("Remote service did not return data")
                
            space_share_id = data.get("space_share_id")
            if not space_share_id:
                logger.error("No space_share_id in response data")
                raise ValueError("Remote service did not return a space_share_id")
                
            # 6. Update local data
            space_dto.space_share_id = space_share_id
            self._repository.save(space_dto)
            
            return space_share_id
            
        except requests.RequestException as e:
            logger.error(f"Connection error: {str(e)}")
            raise ValueError(f"Error connecting to remote registry")

    def delete_space(self, space_id: str) -> bool:
        """
        delete specified Space
        
        Args:
            space_id: Space ID
            
        Returns:
            bool: if success
        """
        # check if Space exists
        space_dto = self._repository.get(space_id)
        if not space_dto:
            return False
            
        # delete Space
        return self._repository.delete(space_id)


# Create singleton instance
space_service = SpaceService()
