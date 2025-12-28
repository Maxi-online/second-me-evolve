"""
Space related request and response models
"""
from typing import List
from pydantic import BaseModel, Field, validator
import re
from datetime import datetime

class CreateSpaceRequest(BaseModel):
    """Create Space request model"""
    title: str = Field(..., description="Space theme")
    objective: str = Field(..., description="Space objective")
    host: str = Field(..., description="Host endpoint")
    participants: List[str] = Field(default_factory=list, description="Other participant endpoints")

    @validator('host', 'participants')
    def validate_endpoint(cls, v):
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        if isinstance(v, str):
            if not re.match(pattern, v):
                raise ValueError(f"Invalid endpoint format: {v}")
            return v
        elif isinstance(v, list):
            for endpoint in v:
                if not re.match(pattern, endpoint):
                    raise ValueError(f"Invalid endpoint format: {endpoint}")
            return v
        return v

class SpaceResponse(BaseModel):
    """Space response model"""
    id: str = Field(..., description="Space ID")
    title: str = Field(..., description="Space theme")
    objective: str = Field(..., description="Space objective")
    host: str = Field(..., description="Host endpoint")
    participants: List[str] = Field(..., description="Other participant endpoints")
    create_time: datetime = Field(..., description="Creation time")
    status: int = Field(..., description="Space status: 1- discussion in progress, 2- discussion ended")
