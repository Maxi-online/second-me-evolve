from datetime import datetime
from sqlalchemy import Column, String, BigInteger, JSON, DateTime, Enum
from sqlalchemy.sql import func
from lpm_kernel.common.repository.database_session import Base
import os


class Memory(Base):
    """Memory model class"""

    __tablename__ = "memories"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    size = Column(BigInteger, nullable=False)
    type = Column(String(50), nullable=False)
    path = Column(String(1024), nullable=False)
    meta_data = Column(JSON)
    document_id = Column(String(36), nullable=True)  # associated document ID
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    status = Column(Enum("active", "deleted"), nullable=False, default="active")

    def __init__(self, name, size, path, metadata=None):
        import uuid

        self.id = str(uuid.uuid4())
        self.name = name
        self.size = size
        self.path = path
        self.meta_data = metadata or {}
        # get type from file extension, if no extension, set to 'unknown'
        _, ext = os.path.splitext(path)
        self.type = ext[1:].lower() if ext else "unknown"
        # set default time
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self):
        """Convert to dictionary, including document_id"""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "meta_data": self.meta_data,
        }
        if self.document_id:
            result["document_id"] = self.document_id
        return result
