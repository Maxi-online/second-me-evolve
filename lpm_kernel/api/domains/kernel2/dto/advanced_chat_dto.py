"""
DTOs for advanced chat mode
"""

from typing import Optional, List, Any

from pydantic import BaseModel, Field


class AdvancedChatRequest(BaseModel):
    """Request model for advanced chat mode"""
    requirement: str = Field(..., description="User's rough requirement")
    max_iterations: int = Field(default=3, description="Maximum number of refinement iterations")
    temperature: float = Field(default=0.7, description="Temperature for model generation")
    enable_l0_retrieval: bool = Field(default=True, description="Whether to enable L0 knowledge retrieval")
    enable_l1_retrieval: bool = Field(default=True, description="Whether to enable L1 knowledge retrieval")


class ValidationResult(BaseModel):
    """Model for solution validation result"""
    is_valid: bool = Field(..., description="Whether the solution meets requirements")
    feedback: Optional[str] = Field(None, description="Feedback for improvement if invalid")


class AdvancedChatResponse(BaseModel):
    """Response model for advanced chat mode"""
    enhanced_requirement: str = Field(..., description="Enhanced requirement with context")
    solution: str = Field(..., description="Generated solution")
    validation_history: List[ValidationResult] = Field(default=[], description="History of validation results")
    final_format: Optional[str] = Field(None, description="Final formatted solution if valid")
    final_response: Optional[Any] = Field(None, description="Final response, can be streaming or complete response")

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True  # permit any type
