"""
Role related services
"""
import logging
from typing import List, Optional
from datetime import datetime

from lpm_kernel.api.domains.loads.load_service import LoadService
from sqlalchemy.exc import IntegrityError
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.api.domains.kernel2.dto.role_dto import (
    Role,
    RoleDTO,
    CreateRoleRequest,
    UpdateRoleRequest,
    ShareRoleRequest,
    generate_role_uuid,
)
from lpm_kernel.api.domains.upload.client import RegistryClient
from lpm_kernel.api.domains.loads.dto import LoadDTO
logger = logging.getLogger(__name__)


class RoleService:
    """Role Service"""

    @staticmethod
    def create_role(request: CreateRoleRequest) -> Optional[RoleDTO]:
        """
        Create new Role

        Args:
            request: Create Role's request object

        Returns:
            Optional[RoleDTO]: if success return RoleDTO, else return None
        """
        try:
            with DatabaseSession.session() as session:
                role = Role(
                    uuid=generate_role_uuid(),  # gen uuid
                    name=request.name,
                    description=request.description,
                    system_prompt=request.system_prompt,
                    icon=request.icon,
                    enable_l0_retrieval=request.enable_l0_retrieval,
                    enable_l1_retrieval=request.enable_l1_retrieval,
                )
                session.add(role)
                session.commit()
                return RoleDTO.from_model(role)
        except IntegrityError:
            logger.error(f"Failed to create Role: name '{request.name}' already exists")
            return None
        except Exception as e:
            logger.error(f"Error creating Role: {str(e)}")
            return None

    # @staticmethod
    # def get_role(role_id: int) -> Optional[RoleDTO]:
    #     """
    #     Get specific Role

    #     Args:
    #         role_id: Role ID

    #     Returns:
    #         Optional[RoleDTO]: if found return RoleDTO, else return None
    #     """
    #     try:
    #         with DatabaseSession.session() as session:
    #             role = session.query(Role).filter(Role.id == role_id).first()
    #             return RoleDTO.from_model(role) if role else None
    #     except Exception as e:
    #         logger.error(f"Error getting Role: {str(e)}")
    #         return None

    @staticmethod
    def get_role_by_uuid(uuid: str) -> Optional[RoleDTO]:
        """
        get role by UUID

        Args:
            uuid: Role UUID

        Returns:
            Optional[RoleDTO]: if found, return RoleDTO, else return None
        """
        try:
            with DatabaseSession.session() as session:
                role = session.query(Role).filter(Role.uuid == uuid).first()
                return RoleDTO.from_model(role) if role else None
        except Exception as e:
            logger.error(f"Error getting Role by UUID: {str(e)}")
            return None

    @staticmethod
    def get_all_roles() -> List[RoleDTO]:
        """
        Get all Roles ordered by creation time in descending order

        Returns:
            List[RoleDTO]: Role List sorted by id desc
        """
        try:
            with DatabaseSession.session() as session:
                query = session.query(Role).order_by(Role.id.desc())
                roles = query.all()
                return [RoleDTO.from_model(role) for role in roles]
        except Exception as e:
            logger.error(f"Error getting all Roles: {str(e)}")
            return []

    @staticmethod
    def update_role(role_id: int, request: UpdateRoleRequest) -> Optional[RoleDTO]:
        """
        Update Role

        Args:
            role_id: Role ID
            request: Update Role's request object

        Returns:
            Optional[RoleDTO]: if update success return RoleDTO, else return None
        """
        try:
            with DatabaseSession.session() as session:
                role = session.query(Role).filter(Role.id == role_id).first()
                if not role:
                    return None

                # only update not None fields
                if request.name is not None:
                    role.name = request.name
                if request.description is not None:
                    role.description = request.description
                if request.system_prompt is not None:
                    role.system_prompt = request.system_prompt
                if request.icon is not None:
                    role.icon = request.icon
                if request.is_active is not None:
                    role.is_active = request.is_active

                role.update_time = datetime.now()
                session.commit()
                return RoleDTO.from_model(role)
        except IntegrityError:
            logger.error(f"Failed to update Role: name '{request.name}' already exists")
            return None
        except Exception as e:
            logger.error(f"Error updating Role: {str(e)}")
            return None

    @staticmethod
    def update_role_by_uuid(uuid: str, request: UpdateRoleRequest) -> Optional[RoleDTO]:
        """
        update role by UUID

        Args:
            uuid: Role UUID
            request: update Role's request object

        Returns:
            Optional[RoleDTO]: if success return RoleDTO, else return None
        """
        try:
            with DatabaseSession.session() as session:
                role = session.query(Role).filter(Role.uuid == uuid).first()
                if not role:
                    return None

                # only update not-None fields
                if request.name is not None:
                    role.name = request.name
                if request.description is not None:
                    role.description = request.description
                if request.system_prompt is not None:
                    role.system_prompt = request.system_prompt
                if request.icon is not None:
                    role.icon = request.icon
                if request.is_active is not None:
                    role.is_active = request.is_active
                if request.enable_l0_retrieval is not None:
                    role.enable_l0_retrieval = request.enable_l0_retrieval
                if request.enable_l1_retrieval is not None:
                    role.enable_l1_retrieval = request.enable_l1_retrieval

                role.update_time = datetime.now()
                session.commit()
                return RoleDTO.from_model(role)
        except IntegrityError:
            logger.error(f"Failed to update Role: name '{request.name}' already exists")
            return None
        except Exception as e:
            logger.error(f"Error updating Role: {str(e)}")
            return None

    @staticmethod
    def delete_role(role_id: int) -> bool:
        """
        Delete Role

        Args:
            role_id: Role ID

        Returns:
            bool: if success
        """
        try:
            with DatabaseSession.session() as session:
                role = session.query(Role).filter(Role.id == role_id).first()
                if not role:
                    return False
                session.delete(role)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting Role: {str(e)}")
            return False

    @staticmethod
    def delete_role_by_uuid(uuid: str) -> bool:
        """
        Delete Role by UUID

        Args:
            uuid: Role UUID

        Returns:
            bool: if delete successfully
        """
        try:
            with DatabaseSession.session() as session:
                role = session.query(Role).filter(Role.uuid == uuid).first()
                if not role:
                    return False
                session.delete(role)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting Role: {str(e)}")
            return False
    
    @staticmethod
    def share_role(request: ShareRoleRequest) -> Optional[RoleDTO]:
        """
        Share role

        Args:
            request: Share role request object

        Returns:
            Optional[RoleDTO]: if success return RoleDTO, else return None
        """
        try:
            current_load, error, status_code = LoadService.get_current_load()
            if error:
                logger.error(f"Failed to get current load: {error}")
                return None
            instance_id = current_load.instance_id
            if not instance_id:
                logger.error("Instance ID not found in current load")
                return None
            
            role_dto = RoleService.get_role_by_uuid(request.role_id)
            if not role_dto:
                logger.error(f"Role not found with uuid: {request.role_id}")
                return None
                
            registry_client = RegistryClient()
            
            logger.info(f"Sharing role {role_dto.name} (ID: {role_dto.uuid}) to registry center . instance_id: {instance_id}")
            
            result = registry_client.create_role(
                role_id=role_dto.uuid,
                name=role_dto.name,
                description=role_dto.description,
                system_prompt=role_dto.system_prompt,
                icon=role_dto.icon,
                instance_id=instance_id,
                is_active=role_dto.is_active,
                enable_l0_retrieval=role_dto.enable_l0_retrieval,
                enable_l1_retrieval=role_dto.enable_l1_retrieval
            )
                
            if result:
                logger.info(f"Role {role_dto.name} (ID: {role_dto.uuid}) shared successfully")
                return role_dto
            else:
                logger.error(f"Failed to share role: {role_dto.name} (ID: {role_dto.uuid})")
                return None
                
        except Exception as e:
            logger.error(f"Error sharing Role: {str(e)}", exc_info=True)
            return None


# create global RoleService instance
role_service = RoleService()
