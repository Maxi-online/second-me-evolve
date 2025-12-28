from sqlalchemy import Column, Integer, Text, DateTime, func
from lpm_kernel.common.repository.database_session import Base


class StatusBiography(Base):
    """Status biography table"""

    __tablename__ = "status_biography"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    content_third_view = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    summary_third_view = Column(Text, nullable=False)
    create_time = Column(DateTime, nullable=False, server_default=func.now())
    update_time = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
