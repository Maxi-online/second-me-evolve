from datetime import datetime
from typing import Optional, Dict
from .process_status import ProcessStatus
from pydantic import BaseModel, Field


class DocumentDTO(BaseModel):
    id: Optional[int] = None
    name: str = Field(default="")
    title: Optional[str] = None
    extract_status: ProcessStatus = Field(default=ProcessStatus.INITIALIZED)
    embedding_status: ProcessStatus = Field(default=ProcessStatus.INITIALIZED)
    analyze_status: ProcessStatus = Field(default=ProcessStatus.INITIALIZED)
    mime_type: Optional[str] = None
    raw_content: Optional[str] = None
    user_description: Optional[str] = None
    create_time: Optional[datetime] = None
    url: Optional[str] = None
    document_size: int = Field(default=0)
    insight: Optional[Dict] = None
    summary: Optional[Dict] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            ProcessStatus: lambda v: v.value if v else None,
        }

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        if d.get("extract_status"):
            d["extract_status"] = d["extract_status"].value
        if d.get("embedding_status"):
            d["embedding_status"] = d["embedding_status"].value
        if d.get("analyze_status"):
            d["analyze_status"] = d["analyze_status"].value
        return d

    @classmethod
    def from_dict(cls, data: Dict):
        if not data:
            return None

        if "extract_status" in data:
            data["extract_status"] = ProcessStatus(data["extract_status"])
        if "embedding_status" in data:
            data["embedding_status"] = ProcessStatus(data["embedding_status"])
        if "analyze_status" in data:
            data["analyze_status"] = ProcessStatus(data["analyze_status"])
        if "create_time" in data and isinstance(data["create_time"], str):
            data["create_time"] = datetime.fromisoformat(data["create_time"])

        return cls(**data)


class CreateDocumentRequest(BaseModel):
    name: str = Field(
        ..., description="Document name", max_length=255, example="example.pdf"
    )
    title: Optional[str] = Field(
        None, description="Document title", max_length=255, example="Example Document"
    )
    mime_type: Optional[str] = Field(
        None, description="MIME type", max_length=100, example="application/pdf"
    )
    user_description: Optional[str] = Field(
        None,
        description="User provided description",
        example="This is an example document",
    )
    url: Optional[str] = Field(
        None,
        description="Document URL or file path",
        example="https://example.com/doc.pdf or /path/to/file",
    )
    document_size: int = Field(
        0, description="Document size in bytes", ge=0, example=1024
    )
    raw_content: Optional[str] = Field(
        None, description="Extracted raw content from the document"
    )
    extract_status: ProcessStatus = Field(
        ProcessStatus.INITIALIZED, description="Extraction status"
    )
    embedding_status: ProcessStatus = Field(
        ProcessStatus.INITIALIZED, description="Embedding status"
    )
    analyze_status: ProcessStatus = Field(
        ProcessStatus.INITIALIZED, description="Analysis status"
    )

    def to_dto(self) -> "DocumentDTO":
        return DocumentDTO(
            name=self.name,
            title=self.title,
            mime_type=self.mime_type,
            user_description=self.user_description,
            url=self.url,
            document_size=self.document_size,
            raw_content=self.raw_content,
            extract_status=self.extract_status,
            embedding_status=self.embedding_status,
            analyze_status=self.analyze_status,
        )
