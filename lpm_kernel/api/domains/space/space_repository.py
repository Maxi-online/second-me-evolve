"""
Space repository implementation
"""
from typing import List, Optional
from datetime import datetime

from lpm_kernel.common.repository.base_repository import BaseRepository
from sqlalchemy import select, update
import sqlalchemy.orm
from lpm_kernel.models.space import Space, SpaceMessage
from .space_dto import SpaceDTO, SpaceMessageDTO

class SpaceRepository(BaseRepository[Space]):
    """Space repository"""
    
    def __init__(self):
        super().__init__(Space)
    
    def create(self, space_dto: SpaceDTO) -> SpaceDTO:
        """
        Create a Space
        
        Args:
            space_dto: Space DTO object
            
        Returns:
            SpaceDTO: Created Space DTO object
        """
        with self._db.session() as session:
            # Create database model instance
            db_space = self.model(
                id=space_dto.id,
                space_share_id=space_dto.space_share_id,
                title=space_dto.title,
                objective=space_dto.objective,
                host=space_dto.host,
                participants=space_dto.participants,
                create_time=datetime.now(),
                conclusion=space_dto.conclusion,
                status=1  # Initial status
            )
            
            # Save to database
            session.add(db_space)
            session.commit()
            
            # Return DTO
            return SpaceDTO.from_db(db_space)
    
    def get(self, space_id: str) -> Optional[SpaceDTO]:
        """
        Get Space
        
        Args:
            space_id: Space ID
            
        Returns:
            Optional[SpaceDTO]: If found, returns Space DTO object, otherwise returns None
        """
        with self._db.session() as session:
            # Query database and load associated messages
            stmt = select(self.model).where(self.model.id == space_id).options(
                sqlalchemy.orm.selectinload(self.model.messages)
            )
            db_space = session.execute(stmt).scalar_one_or_none()
            
            if not db_space:
                return None
            
            # Convert to DTO
            return SpaceDTO.from_db(db_space)
    
    def list_spaces(self, host: Optional[str] = None) -> List[SpaceDTO]:
        """
        List Spaces
        
        Args:
            host: Optional host endpoint filter
            
        Returns:
            List[SpaceDTO]: List of Space DTO objects
        """
        with self._db.session() as session:
            # Build query
            stmt = select(self.model)
            if host:
                stmt = stmt.where(self.model.host == host)
                
            # Execute query
            db_spaces = session.execute(stmt).scalars().all()
            
            # Convert to DTO list
            return [SpaceDTO.from_db(db_space) for db_space in db_spaces]
    
    def update_status(self, space_id: str, status: int) -> bool:
        """
        Update the status of the Space
        
        Args:
            space_id: Space ID
            status: New status value
            
        Returns:
            bool: Whether the update was successful
        """
        with self._db.session() as session:
            # Build update statement
            stmt = (
                update(self.model)
                .where(self.model.id == space_id)
                .values(status=status)
            )
            
            # Execute update
            result = session.execute(stmt)
            session.commit()
            
            # Check if any records were updated
            return result.rowcount > 0
    
    def update_conclusion(self, space_id: str, conclusion: str) -> Optional[SpaceDTO]:
        """
        Update the conclusion of the Space
        
        Args:
            space_id: Space ID
            conclusion: Conclusion content
            
        Returns:
            Optional[SpaceDTO]: If update is successful, returns updated Space DTO object, otherwise returns None
        """
        with self._db.session() as session:
            # Query database
            stmt = select(self.model).where(self.model.id == space_id)
            db_space = session.execute(stmt).scalar_one_or_none()
            
            if not db_space:
                return None
            
            # Update conclusion
            db_space.conclusion = conclusion
            session.commit()
            
            # Return updated DTO
            return SpaceDTO.from_db(db_space)
    
    def add_message(self, space_id: str, message_dto: SpaceMessageDTO) -> Optional[SpaceMessageDTO]:
        """
        Add a message to the Space
        
        Args:
            space_id: Space ID
            message_dto: SpaceMessage DTO object
            
        Returns:
            Optional[SpaceMessageDTO]: If successful, returns the created SpaceMessage DTO object, otherwise returns None
        """
        with self._db.session() as session:
            # query Space
            stmt = select(self.model).where(self.model.id == space_id)
            db_space = session.execute(stmt).scalar_one_or_none()
            
            if not db_space:
                return None
                
            # create message
            db_message = SpaceMessage(
                id=message_dto.id,
                space_id=space_id,
                sender_endpoint=message_dto.sender_endpoint,
                content=message_dto.content,
                message_type=message_dto.message_type,
                round=message_dto.round,
                role=message_dto.role,
                create_time=datetime.now()
            )
            
            # save to database
            session.add(db_message)
            session.commit()
            
            # Return DTO
            return SpaceMessageDTO.from_db(db_message)
    
    def get_messages(self, space_id: str) -> List[SpaceMessageDTO]:
        """
        Get all messages of the Space
        
        Args:
            space_id: Space ID
            
        Returns:
            List[SpaceMessageDTO]: List of SpaceMessage DTO objects
        """
        with self._db.session() as session:
            # Build query
            stmt = select(SpaceMessage).where(SpaceMessage.space_id == space_id).order_by(SpaceMessage.create_time)
            
            # execute query
            db_messages = session.execute(stmt).scalars().all()
            
            # convert to DTO list
            return [SpaceMessageDTO.from_db(db_message) for db_message in db_messages]
    
    def save(self, space_dto: SpaceDTO) -> SpaceDTO:
        """
        Save Space (update existing records)
        
        Args:
            space_dto: Space DTO object
            
        Returns:
            SpaceDTO: Updated Space DTO object
        """
        with self._db.session() as session:
            # Query database and load associated messages
            stmt = select(self.model).where(self.model.id == space_dto.id).options(
                sqlalchemy.orm.selectinload(self.model.messages)
            )
            db_space = session.execute(stmt).scalar_one_or_none()
            
            if not db_space:
                raise ValueError(f"Space not found: {space_dto.id}")
            
            # update fields
            db_space.title = space_dto.title
            db_space.objective = space_dto.objective
            db_space.host = space_dto.host
            db_space.participants = space_dto.participants
            db_space.conclusion = space_dto.conclusion
            db_space.status = getattr(space_dto, "status", 1)
            db_space.space_share_id = space_dto.space_share_id
            
            # Process messages
            # if space_dto.messages:
            #     # Clear existing messages
            #     db_space.messages = []
                
            #     # Add new messages
            #     for message_dto in space_dto.messages:
            #         db_message = SpaceMessage(
            #             id=message_dto.id,
            #             space_id=space_dto.id,
            #             sender_endpoint=message_dto.sender_endpoint,
            #             content=message_dto.content,
            #             message_type=message_dto.message_type,
            #             round=message_dto.round,
            #             role=message_dto.role,
            #             create_time=message_dto.create_time,
            #         )
            #         db_space.messages.append(db_message)
            
            # Save to database
            session.commit()
            
            # Return updated DTO
            return SpaceDTO.from_db(db_space)
            
    def delete(self, space_id: str) -> bool:
        """
        Delete the specified Space
        
        Args:
            space_id: Space ID
            
        Returns:
            bool: Whether the deletion was successful
        """
        with self._db.session() as session:
            # Query database
            stmt = select(self.model).where(self.model.id == space_id)
            db_space = session.execute(stmt).scalar_one_or_none()
            
            if not db_space:
                return False
            
            # Delete Space
            session.delete(db_space)
            session.commit()
            
            # Return True if deletion was successful
            return True


# Global repository instance
space_repository = SpaceRepository()
