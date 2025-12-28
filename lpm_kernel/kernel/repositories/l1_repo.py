from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime

from ..models.l1_model import L1ClusterModel, L1ShadeModel, L1BiographyModel
from lpm_kernel.L1.bio import Cluster, ShadeInfo, Bio
import logging

logger = logging.getLogger(__name__)


class L1Repository:
    """L1 data access repository"""

    def __init__(self, session: Session):
        self._session = session

    # Cluster related operations
    def save_cluster(self, cluster: Dict) -> L1ClusterModel:
        """Save clustering results"""
        model = L1ClusterModel.from_domain(cluster)
        self._session.add(model)
        self._session.commit()
        return model

    def get_cluster_by_id(self, cluster_id: int) -> Optional[L1ClusterModel]:
        """Get specified cluster"""
        return self._session.query(L1ClusterModel).get(cluster_id)

    def get_all_clusters(self) -> List[L1ClusterModel]:
        """Get all clusters"""
        return self._session.query(L1ClusterModel).all()

    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete cluster and its associated features"""
        cluster = self.get_cluster_by_id(cluster_id)
        if cluster:
            self._session.delete(cluster)
            self._session.commit()
            return True
        return False

    # Shade related operations
    def save_shade(self, shade: ShadeInfo, cluster_id: int) -> L1ShadeModel:
        """Save feature information"""
        model = L1ShadeModel.from_domain(shade)
        model.cluster_id = cluster_id
        self._session.add(model)
        self._session.commit()
        return model

    def get_shade_by_id(self, shade_id: int) -> Optional[L1ShadeModel]:
        """Get specified feature"""
        return self._session.query(L1ShadeModel).get(shade_id)

    def get_shades_by_cluster(self, cluster_id: int) -> List[L1ShadeModel]:
        """Get all features of a cluster"""
        return (
            self._session.query(L1ShadeModel)
            .filter(L1ShadeModel.cluster_id == cluster_id)
            .all()
        )

    def delete_shade(self, shade_id: int) -> bool:
        """Delete feature"""
        shade = self.get_shade_by_id(shade_id)
        if shade:
            self._session.delete(shade)
            self._session.commit()
            return True
        return False

    # Biography related operations
    def save_biography(self, bio: Bio) -> L1BiographyModel:
        """Save biography"""
        model = L1BiographyModel.from_domain(bio)
        self._session.add(model)
        self._session.commit()
        return model

    def get_latest_biography(self) -> Optional[L1BiographyModel]:
        """Get the latest biography"""
        return (
            self._session.query(L1BiographyModel)
            .order_by(desc(L1BiographyModel.create_time))
            .first()
        )

    def get_biography_history(self, limit: int = 10) -> List[L1BiographyModel]:
        """Get biography history"""
        return (
            self._session.query(L1BiographyModel)
            .order_by(desc(L1BiographyModel.create_time))
            .limit(limit)
            .all()
        )
