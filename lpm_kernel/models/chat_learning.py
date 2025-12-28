from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, ForeignKey
from sqlalchemy.orm import Mapped

from lpm_kernel.common.repository.database_session import Base


class ChatExperience(Base):
    """
    Stores a single model completion ("experience") for later reinforcement learning.

    Notes:
    - We store the prompt as OpenAI-compatible `messages` (list[{"role","content"}]).
    - We store the final assistant completion as text.
    - This table is intentionally generic and can be used by multiple endpoints (/api/kernel2/chat, /api/talk/chat, etc.).
    """

    __tablename__ = "chat_experiences"

    id: Mapped[str] = Column(String(64), primary_key=True)  # UUID string
    source: Mapped[str] = Column(String(50), nullable=False, default="kernel2")
    model: Mapped[Optional[str]] = Column(String(200), nullable=True)

    # Prompt used for generation (OpenAI messages array)
    prompt_messages: Mapped[List[Dict[str, str]]] = Column(JSON, nullable=False)

    # Generation params snapshot
    temperature: Mapped[Optional[float]] = Column(Integer, nullable=True)  # stored as int*1000 in routes for sqlite safety
    max_tokens: Mapped[Optional[int]] = Column(Integer, nullable=True)
    seed: Mapped[Optional[int]] = Column(Integer, nullable=True)

    # Completion output
    completion: Mapped[Optional[str]] = Column(Text, nullable=True)
    finish_reason: Mapped[Optional[str]] = Column(String(50), nullable=True)

    # Extra metadata (role_id, retrieval flags, etc.)
    meta: Mapped[Dict[str, Any]] = Column("meta_data", JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = Column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = Column(DateTime, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "model": self.model,
            "prompt_messages": self.prompt_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "completion": self.completion,
            "finish_reason": self.finish_reason,
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ChatFeedback(Base):
    """
    Scalar feedback for a ChatExperience (e.g., like/dislike).
    """

    __tablename__ = "chat_feedback"

    id: Mapped[str] = Column(String(64), primary_key=True)  # UUID string
    experience_id: Mapped[str] = Column(String(64), ForeignKey("chat_experiences.id"), nullable=False)

    # rating: +1 (like), -1 (dislike), 0 (neutral)
    rating: Mapped[int] = Column(Integer, nullable=False, default=0)
    comment: Mapped[Optional[str]] = Column(Text, nullable=True)

    created_at: Mapped[datetime] = Column(DateTime, nullable=False)


class ChatPreference(Base):
    """
    Pairwise preference for DPO-style reinforcement:
    prompt + chosen completion + rejected completion.
    """

    __tablename__ = "chat_preferences"

    id: Mapped[str] = Column(String(64), primary_key=True)  # UUID string
    source: Mapped[str] = Column(String(50), nullable=False, default="kernel2")
    model: Mapped[Optional[str]] = Column(String(200), nullable=True)

    prompt_messages: Mapped[List[Dict[str, str]]] = Column(JSON, nullable=False)
    chosen: Mapped[str] = Column(Text, nullable=False)
    rejected: Mapped[str] = Column(Text, nullable=False)

    meta: Mapped[Dict[str, Any]] = Column("meta_data", JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = Column(DateTime, nullable=False)

