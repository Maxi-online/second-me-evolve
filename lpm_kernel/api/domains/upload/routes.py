import string
import secrets
import logging
import asyncio
import time
from flask import Blueprint, request, jsonify
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.models.load import Load
from lpm_kernel.api.domains.loads.load_service import LoadService
from .client import RegistryClient
import threading
from lpm_kernel.api.domains.loads.dto import LoadDTO
from lpm_kernel.api.domains.trainprocess.training_params_manager import TrainingParamsManager
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.api.domains.upload.TrainingTags import TrainingTags

upload_bp = Blueprint("upload", __name__)
registry_client = RegistryClient()

logger = logging.getLogger(__name__)

@upload_bp.route("/api/upload/register", methods=["POST"])
def register_upload():
    """Register upload instance"""
    try:
        current_load, error, status_code = LoadService.get_current_load()
        
        upload_name = current_load.name
        instance_id = current_load.instance_id
        email = current_load.email
        description = current_load.description
        params = TrainingParamsManager.get_latest_training_params()
        model_name = params.get("model_name")
        is_cot = params.get("is_cot")
        document_count = len(document_service.list_documents())
        tags = TrainingTags(
            model_name=model_name,
            is_cot=is_cot,
            document_count=document_count
        )
        
        result = registry_client.register_upload(
            upload_name, instance_id, description, email, tags
        )

        instance_id_new = result.get("instance_id")
        if not instance_id_new:
            return jsonify(APIResponse.error(
                code=400, message="Failed to register upload instance"
            ))
        
        instance_password = result.get("instance_password")
        LoadService.update_instance_credentials(instance_id_new, instance_password)

        return jsonify(APIResponse.success(
            data=result
        ))
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            code=500, message=f"An error occurred: {str(e)}"
        ))

@upload_bp.route("/api/upload/connect", methods=["POST"])
async def connect_upload():
    """
    Establish WebSocket connection for the specified Upload instance
    
    URL parameters:
        instance_id: Instance ID
        upload_name: Upload name
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "ws_url": str  # WebSocket connection URL
        }
    }
    """
    
    try:
        logger.info("Starting WebSocket connection process...")
        current_load, error, status_code = LoadService.get_current_load(with_password=True)
        if error:
            return jsonify(APIResponse.error(
                code=status_code, message=error
            ))
            
        instance_id = current_load.instance_id
        instance_password = current_load.instance_password

        
        
        # Use thread to establish WebSocket connection asynchronously
        def connect_ws():
            asyncio.run(registry_client.connect_websocket(instance_id, instance_password))
            
        
        thread = threading.Thread(target=connect_ws)
        thread.daemon = True  # Set as daemon thread, so it will end automatically when main program exits
        thread.start()
        
        result = {
            "instance_id": instance_id,
            "upload_name": current_load.name
        }
        
        return jsonify(APIResponse.success(
            data=result,
            message="WebSocket connection task started"
        ))
        
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {str(e)}")
        return jsonify(APIResponse.error(
            message=f"Failed to establish WebSocket connection: {str(e)}",
            code=500
        ))
    finally:
        logger.info("WebSocket connection process completed.")

@upload_bp.route("/api/upload/status", methods=["GET"])
def get_upload_status():
    """
    Get the status of the specified Upload instance
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "upload_name": str,
            "instance_id": str,
            "status": str,
            "description": str,
            "email": str,
            "is_connected": bool,
            "last_ws_check": str,
            "connection_alive": bool
        }
    }
    """
    try:

        current_load, error, status_code = LoadService.get_current_load()
        if error:
            return jsonify(APIResponse.error(
                code=status_code, message=error
            ))
        
        instance_id = current_load.instance_id

        # Check if instance exists
        detail = registry_client.get_upload_detail(instance_id)
        
        logger.info(f"Upload status: {detail}")
        
        # Get basic information from local
        upload_data = {
            "upload_name": current_load.name,
            "instance_id": instance_id,
            "description": current_load.description,
            "email": current_load.email
        }
        
        # Process remote data, provide default values if null
        if detail:
            # Merge remote data
            upload_data.update({
                "status": "online" if detail.get("is_connected") else "offline",
                "last_heartbeat": detail.get("last_heartbeat"),
                "is_connected": detail.get("is_connected", False),
                "last_ws_check": detail.get("last_ws_check")
            })
        else:
            # Provide default values
            upload_data.update({
                "status": "unregistered",
                "last_heartbeat": None,
                "is_connected": False,
                "last_ws_check": None
            })
        
        return jsonify(APIResponse.success(
            data=upload_data,
            message="Successfully retrieved Upload instance status"
        ))
            
    except Exception as e:
        logger.error(f"Failed to get Upload instance status: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get status: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["DELETE"])
def unregister_upload():
    """
    API for unregistering Upload instance
    
    URL parameters:
        instance_id: Instance ID
        upload_name: Upload name
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "instance_id": str,
            "upload_name": str
        }
    }
    """
    try:
        current_load, error, status_code = LoadService.get_current_load()
        instance_id = current_load.instance_id
        registry_client.unregister_upload(instance_id)
        
        return jsonify(APIResponse.success(
            data={
                "instance_id": instance_id,
                "upload_name": current_load.name
            },
            message="Upload instance unregistered successfully"
        ))
            
    except Exception as e:
        logger.error(f"Failed to unregister Upload: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Unregistration failed: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["GET"])
def list_uploads():
    """
    List registered Upload instances with pagination and status filter
    
    Query Parameters:
        page_no (int): Page number, starting from 1
        page_size (int): Number of items per page
        status (List[str], optional): List of status to filter by
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "instance_id": {
                "upload_name": str,
                "description": str,
                "email": str,
                "status": str
            }
        }
    }
    """
    try:
        page_no = request.args.get("page_no", 1, type=int)
        page_size = request.args.get("page_size", 10, type=int)
        status = request.args.getlist("status")
        
        result = registry_client.list_uploads(
            page_no=page_no,
            page_size=page_size,
            status=status if status else None
        )
        
        return jsonify(APIResponse.success(
            data=result,
            message="Successfully retrieved Upload list"
        ))
        
    except Exception as e:
        logger.error(f"Failed to get Upload list: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get list: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload/count", methods=["GET"])
def count_uploads():
    """
    Get the number of registered Upload instances
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "count": int
        }
    }
    """
    try:
        result = registry_client.count_uploads()
        
        return jsonify(APIResponse.success(
            data=result,
            message="Successfully retrieved Upload count"
        ))
        
    except Exception as e:
        logger.error(f"Failed to get Upload count: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Failed to get count: {str(e)}",
            code=500
        ))

@upload_bp.route("/api/upload", methods=["PUT"])
def update_upload():
    """
    API for updating Upload instance information
    
    URL parameters:
        instance_id: Instance ID
    
    Request body:
    {
        "upload_name": str (optional),
        "description": str (optional),
        "email": str (optional)
    }
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "instance_id": str,
            "upload_name": str,
            "description": str,
            "email": str,
            "status": str
        }
    }
    """
    try:
        current_load, error, status_code = LoadService.get_current_load()
        instance_id = current_load.instance_id
        
        data = request.get_json()
        if not data:
            return jsonify(APIResponse.error(
                message="Request body cannot be empty",
                code=400
            ))
        
        upload_name = data.get("upload_name")
        description = data.get("description")
        email = data.get("email")
        
        # At least one update field is required
        if upload_name is None and description is None and email is None:
            return jsonify(APIResponse.error(
                message="At least one update field is required: upload_name, description, or email",
                code=400
            ))
        
        result = registry_client.update_upload(
            instance_id=instance_id,
            upload_name=upload_name,
            description=description,
            email=email
        )
        return jsonify(APIResponse.success(
            data=result,
            message="Upload instance updated successfully"
        ))
            
        
    except Exception as e:
        logger.error(f"Failed to update Upload: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(
            message=f"Update failed: {str(e)}",
            code=500
        ))
