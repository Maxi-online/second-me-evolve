from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, relationship

from lpm_kernel.common.repository.database_session import Base

class Space(Base):
    """Space model, representing a multi-Upload chat room with a specific topic"""
    __tablename__ = 'spaces'
    
    id: Mapped[str] = Column(String, primary_key=True)  # Space unique identifier
    space_share_id: Mapped[Optional[str]] = Column(String, nullable=True)  # Space share ID
    title: Mapped[str] = Column(String, nullable=False)  # discussion topic
    objective: Mapped[str] = Column(String, nullable=False)  # discussion objective
    participants: Mapped[List[str]] = Column(JSON, nullable=False)  # participants' endpoint list
    host: Mapped[str] = Column(String, nullable=False)  # host's endpoint
    create_time: Mapped[datetime] = Column(DateTime, nullable=False)  # discussion start time
    status: Mapped[int] = Column(Integer, default=1)  # discussion status: 1-discussion, 2-discussion ended
    conclusion: Mapped[Optional[str]] = Column(String, nullable=True)  # discussion conclusion
    
    # relationship with SpaceMessage
    messages = relationship("SpaceMessage", back_populates="space", cascade="all, delete-orphan")

class SpaceMessage(Base):
    """Space message model, database definition"""
    __tablename__ = 'space_messages'
    
    id: Mapped[str] = Column(String, primary_key=True)  # message unique identifier
    space_id: Mapped[str] = Column(String, ForeignKey('spaces.id'), nullable=False)  # associated Space ID
    sender_endpoint: Mapped[str] = Column(String, nullable=False)  # sender's endpoint
    content: Mapped[str] = Column(String, nullable=False)  # message content
    message_type: Mapped[str] = Column(String, nullable=False)  # message type
    round: Mapped[int] = Column(Integer, default=0)  # message round
    create_time: Mapped[datetime] = Column(DateTime, nullable=False)  # message creation time
    role: Mapped[str] = Column(String, default="participant")  # message sender's role
    
    # relationship with Space
    space = relationship("Space", back_populates="messages")
