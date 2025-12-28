"""
DTOs for reinforcement feedback / preferences
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    experience_id: str = Field(..., description="chat_experiences.id / response id returned by chat endpoint")
    rating: int = Field(..., description="1=like, -1=dislike, 0=neutral")
    comment: Optional[str] = Field(default=None, description="Optional free-form comment")


class PreferenceRequest(BaseModel):
    prompt_messages: List[Dict[str, str]] = Field(..., description="OpenAI messages array used as prompt")
    chosen: str = Field(..., description="Preferred completion")
    rejected: str = Field(..., description="Non-preferred completion")
    model: Optional[str] = Field(default=None, description="Model identifier")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra metadata")


class ReinforceFromFeedbackRequest(BaseModel):
    base_model_path: str = Field(..., description="HF base model path (e.g. resources/model/output/merged_model)")
    output_data_path: str = Field(default="resources/L2/data/dpo/dpo_from_feedback.json", description="Where to export DPO json")
    min_pairs: int = Field(default=20, description="Minimum number of preference pairs required to start training")
    num_train_epochs: int = Field(default=2)
    learning_rate: float = Field(default=5e-6)
    batch_size: int = Field(default=2)
    beta: float = Field(default=0.1)
    lora_r: int = Field(default=32)
    lora_alpha: int = Field(default=64)
    lora_dropout: float = Field(default=0.1)

