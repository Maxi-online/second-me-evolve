from datetime import datetime
from sqlalchemy import String, Integer, Enum, Text, DateTime, JSON
from sqlalchemy.orm import mapped_column, Mapped
from typing import Optional, Dict
from lpm_kernel.common.repository.base_repository import Base
from .process_status import ProcessStatus
from .document_dto import DocumentDTO


class Document(Base):
    __tablename__ = "document"

    id: Mapped[Optional[int]] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    extract_status: Mapped[ProcessStatus] = mapped_column(
        Enum(ProcessStatus), default=ProcessStatus.INITIALIZED
    )
    embedding_status: Mapped[ProcessStatus] = mapped_column(
        Enum(ProcessStatus), default=ProcessStatus.INITIALIZED
    )
    analyze_status: Mapped[ProcessStatus] = mapped_column(
        Enum(ProcessStatus), default=ProcessStatus.INITIALIZED
    )
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    document_size: Mapped[int] = mapped_column(Integer, default=0)
    insight: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    summary: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict:
        """Convert Document to dictionary for internal use"""
        return {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "extract_status": self.extract_status.value
            if self.extract_status
            else None,
            "embedding_status": self.embedding_status.value
            if self.embedding_status
            else None,
            "analyze_status": self.analyze_status.value
            if self.analyze_status
            else None,
            "mime_type": self.mime_type,
            "raw_content": self.raw_content,
            "user_description": self.user_description,
            "create_time": self.create_time,
            "url": self.url,
            "document_size": self.document_size,
            "insight": self.insight,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create Document from dictionary for internal use"""
        if "extract_status" in data and isinstance(data["extract_status"], str):
            data["extract_status"] = ProcessStatus(data["extract_status"])
        if "embedding_status" in data and isinstance(data["embedding_status"], str):
            data["embedding_status"] = ProcessStatus(data["embedding_status"])
        if "analyze_status" in data and isinstance(data["analyze_status"], str):
            data["analyze_status"] = ProcessStatus(data["analyze_status"])

        return cls(**data)

    def to_dto(self) -> DocumentDTO:
        """Convert to DTO for external use"""
        return DocumentDTO(
            id=self.id,
            name=self.name,
            title=self.title,
            extract_status=self.extract_status,
            embedding_status=self.embedding_status,
            analyze_status=self.analyze_status,
            mime_type=self.mime_type,
            raw_content=self.raw_content,
            user_description=self.user_description,
            create_time=self.create_time,
            url=self.url,
            document_size=self.document_size,
            insight=self.insight,
            summary=self.summary,
        )

    @classmethod
    def from_dto(cls, dto: DocumentDTO) -> "Document":
        """Create from DTO for external use"""
        return cls(
            id=dto.id,
            name=dto.name,
            title=dto.title,
            extract_status=dto.extract_status,
            embedding_status=dto.embedding_status,
            analyze_status=dto.analyze_status,
            mime_type=dto.mime_type,
            raw_content=dto.raw_content,
            user_description=dto.user_description,
            create_time=dto.create_time,
            url=dto.url,
            document_size=dto.document_size,
            insight=dto.insight,
            summary=dto.summary,
        )
