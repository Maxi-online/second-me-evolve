import os
from flask import Blueprint, request, current_app
from werkzeug.utils import secure_filename

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.services.local_llm_service import local_llm_service
from lpm_kernel.common.logging import logger
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.models.load import Load, Base
from .load_service import LoadService
from lpm_kernel.api.domains.upload.client import RegistryClient


registry_client = RegistryClient()


loads_bp = Blueprint("loads", __name__)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@loads_bp.route("/api/loads", methods=["POST"])
def create_load():
    """Create a new load record"""
    try:
        data = request.get_json()

        # Validate required fields
        if not data.get("name"):
            return APIResponse.error(
                code=400, message="Name is required"
            )

        # Use LoadService to create load
        new_load, error, status_code = LoadService.create_load(
            name=data["name"],
            description=data.get("description"),
            email=data.get("email", "")
        )

        if error:
            return APIResponse.error(code=status_code, message=error)

        # Return success response with the load dictionary
        return APIResponse.success(data=new_load, message="Created successfully")

    except Exception as e:
        logger.error("fail !", exc_info=True)
        return APIResponse.error(code=500, message="Internal server error")


@loads_bp.route("/api/loads/current", methods=["GET"])
def get_current_load():
    """Get current load record"""
    try:
        # Use LoadService to get current load
        current_load, error, status_code = LoadService.get_current_load()

        if error:
            return APIResponse.error(code=status_code, message=error)

        instance_id = current_load.instance_id
        if not instance_id:
            current_load.status = "unregistered"
            return APIResponse.success(data=current_load, message="Retrieved successfully")
    
        # Check if instance exists
        detail = registry_client.get_upload_detail(instance_id)


        if detail:
            current_load.status = "online" if detail.get("is_connected") else "offline"
        else:
            current_load.status = "unregistered"


        return APIResponse.success(data=current_load, message="Retrieved successfully")
    except Exception as e:
        logger.error("fail !", exc_info=True)
        return APIResponse.error(code=500, message="Internal server error")


@loads_bp.route("/api/loads/current", methods=["PUT"])
def update_current_load():
    """Update current load information"""
    try:
        data = request.get_json()

        # Use LoadService to update current load
        updated_load_dict, error, status_code = LoadService.update_current_load(data)

        if error:
            return APIResponse.error(code=status_code, message=error)

        return APIResponse.success(data=updated_load_dict, message="Updated successfully")
    except Exception as e:
        logger.error("fail !", exc_info=True)
        return APIResponse.error(code=500, message="Internal server error")


@loads_bp.route("/api/loads/<load_name>", methods=["DELETE"])
def delete_load(load_name):
    """Delete specified load record and related data
    
    Args:
        load_name (str): Name of the load to delete

    Returns:
        APIResponse: Delete operation response
    """
    logger.info(f"Starting to delete load: {load_name}")

    try:
        # First, check and stop the llama-server if it's running
        try:
            # Check if server is running
            status = local_llm_service.get_server_status()
            if status.is_running:
                logger.info("llama-server is running, attempting to stop it")
                stop_status = local_llm_service.stop_server()
                if stop_status.is_running and stop_status.process_info:
                    pid = stop_status.process_info.pid
                    logger.warning(f"llama-server process still running: {pid}")
            else:
                logger.info("llama-server is not running")
        except Exception as e:
            logger.error(f"Failed to check/stop llama-server: {str(e)}")
            # Continue with deletion even if stopping server fails

        # Use LoadService to delete load
        error, status_code = LoadService.delete_load(load_name)

        if error:
            return APIResponse.error(code=status_code, message=error)

        logger.info(f"Successfully deleted load and related data: {load_name}")
        return APIResponse.success(message="Successfully deleted all data")

    except Exception as e:
        logger.error(f"An unknown error occurred during deletion: {str(e)}", exc_info=True)
        return APIResponse.error(code=500, message=f"Deletion failed: {str(e)}")


@loads_bp.route("/api/loads/<load_name>/avatar", methods=["POST"])
def upload_avatar(load_name):
    """Upload load's avatar, store it as base64 string in the database"""
    try:
        # Get JSON request body
        data = request.get_json()
        if not data or 'avatar_data' not in data:
            return APIResponse.error(
                code=400,
                message="Request body missing avatar_data field"
            )

        # Get base64 format avatar data
        base64_string = data['avatar_data']

        # Validate base64 format
        if not base64_string.startswith('data:image/'):
            return APIResponse.error(
                code=400,
                message="Invalid base64 format, should start with 'data:image/'"
            )

        # Use LoadService to update avatar
        updated_load, error, status_code = LoadService.update_avatar(load_name, base64_string)

        if error:
            return APIResponse.error(code=status_code, message=error)

        return APIResponse.success(
            data={"avatar_data": base64_string},
            message="Avatar uploaded successfully"
        )

    except Exception as e:
        logger.error(f"An error occurred while uploading avatar: {str(e)}", exc_info=True)
        return APIResponse.error(code=500, message="Internal server error")


@loads_bp.route("/api/loads/<load_name>/avatar", methods=["GET"])
def get_avatar(load_name):
    """Get load's avatar data"""
    try:
        # Use LoadService to get avatar
        avatar_data, error, status_code = LoadService.get_avatar(load_name)

        if error:
            return APIResponse.error(code=status_code, message=error)

        return APIResponse.success(
            data={"avatar_data": avatar_data},
            message="Avatar retrieved successfully"
        )

    except Exception as e:
        logger.error(f"An error occurred while getting avatar: {str(e)}", exc_info=True)
        return APIResponse.error(code=500, message="Internal server error")
