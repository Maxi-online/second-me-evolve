from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from lpm_kernel.models.load import Load

@dataclass
class LoadDTO:
    """Data Transfer Object for Load"""
    
    # Basic Information
    id: str
    name: str
    description: Optional[str] = None
    email: str = ""
    avatar_data: Optional[str] = None
    
    # Instance Information
    instance_id: Optional[str] = None
    instance_password: Optional[str] = None  # Only included when needed
    status: str = "active"
    
    # Connection Status
    is_connected: bool = False
    last_heartbeat: Optional[datetime] = None
    last_ws_check: Optional[datetime] = None
    
    # Time Information
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_model(cls, model: Load, with_password: bool = False):
        """Create DTO from database model
        
        Args:
            model: Load database model instance
            with_password (bool): Whether to include password information
            
        Returns:
            LoadDTO: Data transfer object instance
        """
        data = {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "email": model.email,
            "avatar_data": model.avatar_data,
            "instance_id": model.instance_id,
            "status": model.status,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }
        
        if with_password:
            data["instance_password"] = model.instance_password
            
        return cls(**data)
    
    def to_dict(self, with_password: bool = False) -> dict:
        """Convert to dictionary format
        
        Args:
            with_password (bool): Whether to include password information
            
        Returns:
            dict: Dictionary representation
        """
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "email": self.email,
            "avatar_data": self.avatar_data,
            "instance_id": self.instance_id,
            "status": self.status,
            "is_connected": self.is_connected,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_ws_check": self.last_ws_check.isoformat() if self.last_ws_check else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if with_password:
            data["instance_password"] = self.instance_password
            
        return data
