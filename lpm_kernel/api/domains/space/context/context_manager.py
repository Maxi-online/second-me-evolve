"""
Space context manager, responsible for managing context information during Space discussions
"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from ..space_dto import SpaceDTO, SpaceMessageDTO
from ..space_repository import space_repository

class SpaceContextManager:
    """Space Context Manager"""

    def __init__(self, space: SpaceDTO):
        """
        Initialize context manager

        Args:
            space: Space DTO object
        """
        self.space_dto = space
        # Record current message position for each participant
        self.participant_positions: Dict[str, int] = {
            participant: 0 for participant in space.get_all_participants()
        }
        # Current discussion round
        self.current_round: int = 0
        # Currently active participant
        self.current_participant: Optional[str] = None

    def save_message(self, message_dto: SpaceMessageDTO) -> None:
        """
        Save new message

        Args:
            message_dto: SpaceMessage DTO instance
        """
        self.space_dto.add_message(message_dto)

        space_repository.add_message(self.space_dto.id, message_dto)

    def get_context_for_participant(self, participants: str) -> List[SpaceMessageDTO]:
        """
        Get context message list for participant

        Args:
            participants: Participant's endpoint

        Returns:
            List of messages visible to this participant
        """
        if participants not in self.participant_positions:
            raise ValueError(f"Participant not in discussion: {participants}")

        # Get participant's last read position
        last_position = self.participant_positions[participants]
        # Update read position to latest
        self.participant_positions[participants] = len(self.space_dto.messages)

        # Return unread messages
        return self.space_dto.messages[last_position:]

    def create_message(
        self,
        sender_endpoint: str,
        content: str,
        message_type: str,
        round: Optional[int] = None
    ) -> SpaceMessageDTO:
        """
        Create new message

        Args:
            sender_endpoint: Sender endpoint
            content: Message content
            message_type: Message type
            round: Optional round number, uses current round if not specified

        Returns:
            Created SpaceMessage DTO instance
        """
        if round is None:
            round = self.current_round

        # Check if content is empty (None, empty string, whitespace) or starts with "error"
        if not content or (isinstance(content, str) and (content.strip() == "" or content.lower().strip().startswith("error"))):
            content = "I am currently not accessible."
            
        # Determine message sender's role
        role = "host" if sender_endpoint == self.space_dto.host else "participant"

        message_dto = SpaceMessageDTO(
            id=str(uuid.uuid4()),  # Generate unique ID
            space_id=self.space_dto.id,
            sender_endpoint=sender_endpoint,
            content=content,
            message_type=message_type,
            round=round,
            create_time=datetime.now(),
            role=role
        )
        
        self.save_message(message_dto)
        return message_dto

    def advance_round(self) -> None:
        """Advance to next discussion round"""
        self.current_round += 1

    def get_current_round(self) -> int:
        """Get current round"""
        return self.current_round

    def get_messages_in_round(self, round: int) -> List[SpaceMessageDTO]:
        """
        Get all messages for specified round

        Args:
            round: Round number

        Returns:
            All messages in this round
        """
        return self.space_dto.get_messages_by_round(round)

    def get_all_messages(self) -> List[SpaceMessageDTO]:
        """
        Get all messages

        Returns:
            List of all messages
        """
        return self.space_dto.messages

    def get_participant_last_message(self, participants: str) -> Optional[SpaceMessageDTO]:
        """
        Get participant's last message

        Args:
            participants: Participant endpoint

        Returns:
            Last message, or None if no messages exist
        """
        for message in reversed(self.space_dto.messages):
            if message.sender_endpoint == participants:
                return message
        return None

    def get_opening_message(self) -> str:
        """Get opening message"""
        return f"""Welcome to the discussion on "{self.space_dto.title}"!

Discussion objective: {self.space_dto.objective}

Let's begin!"""

    def get_round_prompt(self, round: int) -> str:
        """
        Get prompt for specified round

        Args:
            round: Round number

        Returns:
            str: Prompt content
        """
        # Get messages from previous rounds
        previous_messages = [msg for msg in self.space_dto.messages if msg.round < round]

        # Build prompt
        prompt = f"""This is round {round} of the discussion on "{self.space_dto.title}".

Discussion objective: {self.space_dto.objective}

"""
        if previous_messages:
            prompt += "\nPrevious discussion content:\n"
            for msg in previous_messages:
                prompt += f"{msg.sender_endpoint}: {msg.content}\n"

        prompt += "\nPlease share your thoughts:"
        return prompt

    def get_summary_prompt(self) -> str:
        """Get summary prompt"""
        # Get all discussion messages
        discussion_messages = [msg for msg in self.space_dto.messages if msg.message_type == "discussion"]

        prompt = f"""Please summarize the discussion on "{self.space_dto.title}".

Discussion objective: {self.space_dto.objective}

Discussion content:
"""
        for msg in discussion_messages:
            prompt += f"{msg.sender_endpoint} (Round {msg.round}): {msg.content}\n"

        prompt += "\nPlease summarize the main points and conclusions:"
        return prompt
