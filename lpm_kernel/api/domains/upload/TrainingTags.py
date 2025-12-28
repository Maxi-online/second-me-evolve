from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TrainingTags(BaseModel):
    model_name: str
    is_cot: bool = False
    document_count: int = Field(ge=0, default=0)
    
    class Config:
        extra = "allow"  # Allows additional fields for extensibility
        validate_assignment = True
