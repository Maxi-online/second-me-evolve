from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy import select, and_
from lpm_kernel.common.repository.base_repository import BaseRepository
from lpm_kernel.api.models.user_llm_config import UserLLMConfig
from lpm_kernel.api.dto.user_llm_config_dto import UserLLMConfigDTO, UpdateUserLLMConfigDTO


class UserLLMConfigRepository(BaseRepository[UserLLMConfig]):
    def __init__(self):
        super().__init__(UserLLMConfig)

    def get_default_config(self) -> Optional[UserLLMConfigDTO]:
        """Get default configuration (ID=1)"""
        return self._get_by_id(1)


    def _get_by_id(self, id: int) -> Optional[UserLLMConfigDTO]:
        """Get configuration by ID"""
        with self._db.session() as session:
            result = session.get(UserLLMConfig, id)
            return UserLLMConfigDTO.from_model(result) if result else None
            
    def count(self) -> int:
        """Count total number of configurations"""
        with self._db.session() as session:
            return session.query(UserLLMConfig).count()

    def create(self, dto: UserLLMConfigDTO) -> UserLLMConfigDTO:
        """Create a new LLM configuration
        
        Args:
            dto: UserLLMConfigDTO object
            
        Returns:
            Created configuration
        """
        with self._db.session() as session:
            # Convert DTO to dictionary, filtering out None values
            create_dict = {k: v for k, v in dto.dict().items() if v is not None and k != 'id'}
            
            # Set timestamps
            now = datetime.now()
            create_dict['created_at'] = now
            create_dict['updated_at'] = now
            
            # Create entity
            entity = UserLLMConfig(**create_dict)
            entity.id = 1  # Force ID to be 1
            
            session.add(entity)
            session.commit()
            return UserLLMConfigDTO.from_model(entity)
    
    def update(self, id: int, dto: Union[UserLLMConfigDTO, UpdateUserLLMConfigDTO]) -> UserLLMConfigDTO:
        """Update LLM configuration or create if not exists
        
        Args:
            id: Configuration ID (should be 1)
            dto: UserLLMConfigDTO or UpdateUserLLMConfigDTO object
            
        Returns:
            Updated or created configuration
        """
        with self._db.session() as session:
            entity = session.get(UserLLMConfig, id)
            
            if not entity:
                # If entity doesn't exist, create a new one
                session.commit()  # Close current transaction
                return self.create(UserLLMConfigDTO(**dto.dict()))
            
            # Convert DTO to dictionary, filtering out None values
            update_dict = {k: v for k, v in dto.dict().items() if v is not None}
            
            # Update entity attributes
            for key, value in update_dict.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            # Update timestamp
            entity.updated_at = datetime.now()
            
            session.commit()
            return UserLLMConfigDTO.from_model(entity)

    def delete(self, id: int) -> Optional[UserLLMConfigDTO]:
        """Delete specified ID LLM configuration
        
        Args:
            id: ID of configuration to delete
        
        Returns:
        return True if delete success
        """
        with self._db.session() as session:
            entity = session.get(UserLLMConfig, id)
            if not entity:
                return None
                
            session.delete(entity)
            session.commit()
            return UserLLMConfigDTO.from_model(entity)



