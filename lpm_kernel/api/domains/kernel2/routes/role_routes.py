"""
Role related API route
"""
import logging

from flask import Blueprint, request, jsonify

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.kernel2.dto.role_dto import CreateRoleRequest, UpdateRoleRequest
from lpm_kernel.api.domains.kernel2.dto.role_dto import ShareRoleRequest
from lpm_kernel.api.domains.kernel2.services.role_service import role_service

logger = logging.getLogger(__name__)

role_bp = Blueprint("role", __name__, url_prefix="/api/kernel2/roles")


@role_bp.route("", methods=["POST"])
def create_role():
    """create new Role"""
    try:
        data = request.get_json()
        create_request = CreateRoleRequest.from_dict(data)
        role = role_service.create_role(create_request)

        if role:
            return jsonify(APIResponse.success(role.to_dict()))
        else:
            return jsonify(APIResponse.error("create role failed, maybe the name existed")), 400

    except Exception as e:
        logger.error(f"Error creating Role: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when creating role: {str(e)}")), 500


@role_bp.route("", methods=["GET"])
def get_all_roles():
    """Get all Role"""
    try:
        roles = role_service.get_all_roles()
        return jsonify(APIResponse.success([role.to_dict() for role in roles]))

    except Exception as e:
        logger.error(f"Error getting Role list: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when getting Role list: {str(e)}")), 500


@role_bp.route("/<string:uuid>", methods=["GET"])
def get_role(uuid: str):
    """Get specified Role"""
    try:
        role = role_service.get_role_by_uuid(uuid)
        if role:
            return jsonify(APIResponse.success(role.to_dict()))
        else:
            return jsonify(APIResponse.error("Role not existed")), 404

    except Exception as e:
        logger.error(f"Error getting Role: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when getting Role: {str(e)}")), 500


@role_bp.route("/<string:uuid>", methods=["PUT"])
def update_role(uuid: str):
    """Update Role"""
    try:
        data = request.get_json()
        update_request = UpdateRoleRequest.from_dict(data)
        role = role_service.update_role_by_uuid(uuid, update_request)

        if role:
            return jsonify(APIResponse.success(role.to_dict()))
        else:
            return jsonify(
                APIResponse.error(
                    "Error occurred when updating Role，Role does not exist or the role name is already in use")
            ), 400

    except Exception as e:
        logger.error(f"Error updating Role: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when updating Role: {str(e)}")), 500


@role_bp.route("/<string:uuid>", methods=["DELETE"])
def delete_role(uuid: str):
    """Delete Role"""
    try:
        success = role_service.delete_role_by_uuid(uuid)
        if success:
            return jsonify(APIResponse.success("Role deletion successed"))
        else:
            return jsonify(APIResponse.error("Role not existed")), 404

    except Exception as e:
        logger.error(f"Error deleting Role: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when deleting Role: {str(e)}")), 500


@role_bp.route("/share", methods=["POST"])
def share_role():
    """
    Share role to registry center
    
    This API shares local role information to the remote registry center, making it accessible to other instances.
    
    Request body:
    {
        "role_id": "string"  // Required, UUID of the role to be shared
    }
    
    Response:
    Success:
    {
        "code": 0,
        "message": "success",
        "data": {
            "id": 1,
            "uuid": "role_abc123",
            "name": "Role Name",
            "description": "Role Description",
            "system_prompt": "System Prompt",
            "icon": "Icon URL",
            "is_active": true,
            "enable_l0_retrieval": true,
            "enable_l1_retrieval": true,
            "create_time": "2025-03-17T15:24:42",
            "update_time": "2025-03-17T15:24:42"
        }
    }
    
    Failure:
    {
        "code": -1,
        "message": "share role failed",
        "data": null
    }
    
    Possible errors:
    - 400: Share role failed, possibly due to non-existent role or remote service error
    - 500: Internal server error
    """
    try:
        data = request.get_json()
        share_request = ShareRoleRequest.from_dict(data)
        role = role_service.share_role(share_request)

        if role:
            return jsonify(APIResponse.success(role.to_dict()))
        else:
            return jsonify(APIResponse.error("share role failed")), 400

    except Exception as e:
        logger.error(f"Error sharing Role: {str(e)}")
        return jsonify(APIResponse.error(f"Error occurred when sharing role: {str(e)}")), 500
