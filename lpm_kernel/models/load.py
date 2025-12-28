from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum
from sqlalchemy.sql import func

from lpm_kernel.common.repository.database_session import Base


class Load(Base):
    """Load model class, used to manage personal Load data"""

    __tablename__ = "loads"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String, nullable=True)
    email = Column(String(255), nullable=False, default="")
    avatar_data = Column(String, nullable=True)  # store base64 encoded avatar data
    instance_id = Column(String(255), nullable=True)  # store upload instance ID
    instance_password = Column(String(255), nullable=True)  # store upload instance password
    status = Column(
        Enum("active", "inactive", "deleted", name="load_status"),
        nullable=False,
        default="active",
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __init__(self, name, description=None, email="", instance_id=None, instance_password=None):
        """Initialize Load instance

        Args:
            name (str): Load name
            description (str, optional): Load description
            email (str, optional): Load email
            instance_id (str, optional): Upload instance ID
            instance_password (str, optional): Upload instance password
        """
        import uuid

        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.email = email
        self.instance_id = instance_id
        self.instance_password = instance_password

    def to_dict(self):
        """Convert model to dictionary format

        Returns:
            dict: Load dictionary representation, including all fields
        """
        return {
            # basic information
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "email": self.email,
            "avatar_data": self.avatar_data,
            "instance_id": self.instance_id,
            "instance_password": self.instance_password,
            "status": self.status,

            # time information
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
