"""
Definition of Space-related DTOs (Data Transfer Objects)
"""
from typing import List, Optional, ClassVar
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re
from typing import Optional

class CreateSpaceDTO(BaseModel):
    """Request DTO for creating a Space"""
    title: str = Field(..., description="Topic of the Space")
    objective: str = Field(..., description="Objective of the Space")
    host: str = Field(..., description="Host's endpoint")
    participants: List[str] = Field(default_factory=list, description="List of other participants' endpoints")

    @validator('host')
    def validate_host(cls, v):
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid host endpoint format: {v}, must be a valid http(s) address")
        return v

    @validator('participants')
    def validate_participants(cls, v):
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        invalid_endpoints = []
        for endpoint in v:
            if not re.match(pattern, endpoint):
                invalid_endpoints.append(endpoint)
        if invalid_endpoints:
            raise ValueError(f"The following participant endpoints are invalid: {', '.join(invalid_endpoints)}, must be valid http(s) addresses")
        return v

class SpaceMessageDTO(BaseModel):
    """DTO for Space messages"""
    id: str = Field(..., description="Message ID")
    space_id: str = Field(..., description="ID of the Space this message belongs to")
    sender_endpoint: str = Field(..., description="Sender's endpoint")
    content: str = Field(..., description="Message content")
    message_type: str = Field(..., description="Message type, such as opening, discussion, summary, etc.")
    round: int = Field(0, description="Discussion round, 0 indicates a non-discussion message (such as opening or summary)")
    create_time: datetime = Field(..., description="Message creation time")
    role: str = Field("participant", description="Role of the message sender, such as host or participant")

    @classmethod
    def from_db(cls, db_message) -> "SpaceMessageDTO":
        """Create DTO from database model"""
        return cls(
            id=db_message.id,
            space_id=db_message.space_id,
            sender_endpoint=db_message.sender_endpoint,
            content=db_message.content,
            message_type=getattr(db_message, "message_type", "unknown"),
            round=getattr(db_message, "round", 0),
            create_time=db_message.create_time,
            role=getattr(db_message, "role", "participant")
        )
        
    def to_dict(self) -> dict:
        """Convert DTO to dictionary"""
        return {
            "id": self.id,
            "space_id": self.space_id,
            "sender_endpoint": self.sender_endpoint,
            "content": self.content,
            "message_type": self.message_type,
            "round": self.round,
            "create_time": self.create_time.isoformat(),
            "role": self.role
        }

class SpaceDTO(BaseModel):
    """Response DTO for Space"""
    # Status constant definitions
    STATUS_INITIALIZED: ClassVar[int] = 1  # Initialized
    STATUS_DISCUSSING: ClassVar[int] = 2   # In discussion
    STATUS_INTERRUPTED: ClassVar[int] = 3  # Discussion interrupted
    STATUS_FINISHED: ClassVar[int] = 4     # Discussion finished

    id: str
    title: str
    objective: str
    participants: List[str]
    host: str
    create_time: datetime
    messages: List[SpaceMessageDTO] = Field(default_factory=list, description="List of messages in the Space")
    conclusion: Optional[str] = Field(default=None, description="Final conclusion of the discussion")
    status: int = Field(STATUS_INITIALIZED, description="Discussion status: 1-Initialized, 2-In discussion, 3-Discussion interrupted, 4-Discussion finished")
    space_share_id: Optional[str] = Field(default=None, description="Share ID for the Space, used for sharing with others")

    def get_all_participants(self) -> List[str]:
        """
        Get the list of endpoints for all participants (including the host)
        
        Returns:
            List[str]: List of endpoints for all participants
        """
        # Ensure the host is in the list and not duplicated
        all_participants = set(self.participants)
        all_participants.add(self.host)
        return list(all_participants)
        
    def add_message(self, message_dto: SpaceMessageDTO) -> None:
        """
        Add a message to the Space
        
        Args:
            message_dto: The message DTO to be added
        """
        self.messages.append(message_dto)

    @classmethod
    def from_db(cls, db_space) -> "SpaceDTO":
        """Create DTO from database model"""
        # Get message list
        messages = []
        if hasattr(db_space, "messages") and db_space.messages:
            messages = [SpaceMessageDTO.from_db(msg) for msg in db_space.messages]
            
        return cls(
            id=db_space.id,
            space_share_id=db_space.space_share_id,
            title=db_space.title,
            objective=db_space.objective,
            participants=db_space.participants,
            host=db_space.host,
            create_time=db_space.create_time,
            messages=messages,
            conclusion=db_space.conclusion,
            status=getattr(db_space, "status", 1)
        )
    
    def to_dict(self) -> dict:
        """Convert DTO to dictionary"""
        return {
            "id": self.id,
            "space_share_id": self.space_share_id,
            "title": self.title,
            "objective": self.objective,
            "participants": self.participants,
            "host": self.host,
            "create_time": self.create_time.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "conclusion": self.conclusion,
            "status": self.status
        }
