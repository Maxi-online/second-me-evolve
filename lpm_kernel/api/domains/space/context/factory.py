"""
Space Context Manager Factory
"""
from ..space_dto import SpaceDTO
from .context_manager import SpaceContextManager

class SpaceContextManagerFactory:
    """Space Context Manager Factory"""
    
    def create_context_manager(self, space_dto: SpaceDTO) -> SpaceContextManager:
        """
        Create context manager
        
        Args:
            space_dto: Space DTO object
            
        Returns:
            SpaceContextManager: Created context manager
        """
        return SpaceContextManager(space_dto)
