from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from lpm_kernel.common.repository.database_session import Base
from datetime import datetime
from typing import List, Dict, Optional
from lpm_kernel.L1.bio import Bio, Cluster, ShadeInfo


class L1ClusterModel(Base):
    """L1 layer clustering result model"""

    __tablename__ = "l1_clusters"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    center_embedding = Column(JSON)  # Cluster center vector
    memory_ids = Column(JSON)  # List of document IDs belonging to this cluster
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    shades = relationship("L1ShadeModel", back_populates="cluster")

    def to_domain(self) -> Cluster:
        """Convert to domain object"""
        return Cluster(
            id=self.id,
            name=self.name,
            clusterCenter=self.center_embedding,
            memoryList=[{"memoryId": mid} for mid in self.memory_ids],
        )

    @classmethod
    def from_domain(cls, cluster: Dict) -> "L1ClusterModel":
        """Create model from domain object"""
        return cls(
            name=cluster.get("name", ""),
            center_embedding=cluster.get("clusterCenter", []),
            memory_ids=[m["memoryId"] for m in cluster.get("memoryList", [])],
        )


class L1ShadeModel(Base):
    """L1 layer feature information model"""

    __tablename__ = "l1_shades"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("l1_clusters.id"))
    name = Column(String(255))
    aspect = Column(String(50))
    icon = Column(String(50))
    desc_third_view = Column(Text)
    content_third_view = Column(Text)
    desc_second_view = Column(Text)
    content_second_view = Column(Text)
    confidence_level = Column(String(20))
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cluster = relationship("L1ClusterModel", back_populates="shades")

    def to_domain(self) -> ShadeInfo:
        """Convert to domain object"""
        return ShadeInfo(
            id=self.id,
            name=self.name,
            aspect=self.aspect,
            icon=self.icon,
            descThirdView=self.desc_third_view,
            contentThirdView=self.content_third_view,
            descSecondView=self.desc_second_view,
            contentSecondView=self.content_second_view,
            confidenceLevel=self.confidence_level,
        )

    @classmethod
    def from_domain(cls, shade: ShadeInfo) -> "L1ShadeModel":
        """Create model from domain object"""
        return cls(
            name=shade.name,
            aspect=shade.aspect,
            icon=shade.icon,
            desc_third_view=shade.desc_third_view,
            content_third_view=shade.content_third_view,
            desc_second_view=shade.desc_second_view,
            content_second_view=shade.content_second_view,
            confidence_level=shade.confidence_level.value
            if shade.confidence_level
            else None,
        )


class L1BiographyModel(Base):
    """L1 layer biography model"""

    __tablename__ = "l1_biographies"

    id = Column(Integer, primary_key=True)
    content_third_view = Column(Text)
    content_second_view = Column(Text)
    summary_third_view = Column(Text)
    summary_second_view = Column(Text)
    attributes = Column(JSON)  # Store attribute list
    create_time = Column(DateTime, default=datetime.utcnow)
    update_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_domain(self) -> Bio:
        """Convert to domain object"""
        return Bio(
            contentThirdView=self.content_third_view,
            content=self.content_second_view,
            summaryThirdView=self.summary_third_view,
            summary=self.summary_second_view,
            attributeList=self.attributes,
            shadesList=[],  # shades are retrieved through relationship query
        )

    @classmethod
    def from_domain(cls, bio: Bio) -> "L1BiographyModel":
        """Create model from domain object"""
        return cls(
            content_third_view=bio.content_third_view,
            content_second_view=bio.content_second_view,
            summary_third_view=bio.summary_third_view,
            summary_second_view=bio.summary_second_view,
            attributes=bio.attribute_list,
        )
