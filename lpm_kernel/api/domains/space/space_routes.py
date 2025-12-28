"""
Space API Routes

This module provides endpoints for managing Space, including:
- Creating a new Space with participants
- Retrieving Space information
"""

from flask import Blueprint, request
from pydantic import ValidationError

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.common.logging import logger
from .space_dto import CreateSpaceDTO
from .space_service import space_service

# Create Blueprint
space_bp = Blueprint('space', __name__, url_prefix="/api/space")


@space_bp.route('/create', methods=['POST'])
def create_space() -> dict:
    """
    Create a new Space and automatically start a discussion
    
    Request Body:
        {
            "title": str,  # Space theme
            "objective": str,  # Space objective
            "host": str,  # Host endpoint
            "participants": List[str]  # List of participant endpoints
        }
    """
    try:
        # Manually validate request data
        body = CreateSpaceDTO(**request.get_json())

        # Use service to create Space
        space_dto = space_service.create_space(
            title=body.title,
            objective=body.objective,
            host=body.host,
            participants=body.participants
        )

        return APIResponse.success(space_dto.model_dump())

    except ValidationError as e:
        # Handle pydantic validation errors
        errors = e.errors()
        # Extract all error messages
        error_messages = []
        for error in errors:
            field = '.'.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            error_messages.append(f"{field}: {msg}")
        return APIResponse.error(message='; '.join(error_messages), code=400)
    except ValueError as e:
        logger.error(' ValueError. ', exc_info=True)
        # Handle other validation errors
        return APIResponse.error(message=str(e), code=400)
    except Exception as e:
        logger.error(' error. ', exc_info=True)
        # Handle other errors
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/<space_id>', methods=['GET'])
def get_space(space_id: str) -> dict:
    """
    Get the information of a specified Space
    
    Args:
        space_id: Space ID
    """
    try:
        space_dto = space_service.get_space(space_id)
        if not space_dto:
            return APIResponse.error(message="Space not found", code=404)

        return APIResponse.success(space_dto.model_dump())

    except Exception as e:
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/all', methods=['GET'])
def get_all_spaces() -> dict:
    """
    Get all Space lists, optional filter by host endpoint   
    
    Query Parameters:
        host: Optional, host endpoint
    """
    try:
        # Get host parameter from query string
        host = request.args.get('host', None)

        # Get Space list
        spaces = space_service.get_all_spaces(host)

        # Convert to dict list
        spaces_data = [space.model_dump() for space in spaces]

        return APIResponse.success(spaces_data)

    except Exception as e:
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/<space_id>', methods=['DELETE'])
def delete_space(space_id: str) -> dict:
    """
    Delete the specified Space
    
    Args:
        space_id: Space ID
    """
    try:
        # Delete Space
        result = space_service.delete_space(space_id)
        
        if not result:
            return APIResponse.error(message="Space not found or could not be deleted", code=404)
            
        return APIResponse.success({"message": "Space deleted successfully"})
        
    except Exception as e:
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/<space_id>/start', methods=['POST'])
async def start_discussion(space_id: str) -> dict:
    """
    Start a discussion for the specified Space
    
    Args:
        space_id: Space ID
    """
    try:
        # Start discussion
        success = space_service.start_discussion(space_id)

        if not success:
            return APIResponse.error(message="Discussion start failed", code=500)

        # Get latest Space information
        space_dto = space_service.get_space(space_id)
        if not space_dto:
            return APIResponse.error(message="Space not found", code=404)

        return APIResponse.success({
            "message": "Discussion started",
            "space": space_dto.model_dump()
        })

    except ValueError as e:
        return APIResponse.error(message=str(e), code=404)
    except Exception as e:
        logger.error(' error. ', exc_info=True)
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/<space_id>/status', methods=['GET'])
def get_discussion_status(space_id: str) -> dict:
    """
    Get the status of the discussion for the specified Space
    
    Args:
        space_id: Space ID
    """
    try:
        # Get discussion status
        status = space_service.get_discussion_status(space_id)
        return APIResponse.success(status)

    except ValueError as e:
        return APIResponse.error(message=str(e), code=404)
    except Exception as e:
        return APIResponse.error(message=str(e), code=500)


@space_bp.route('/<space_id>/share', methods=['POST'])
def share_space(space_id: str) -> dict:
    """
    Share a space by registering it to remote and generating a share ID
    
    Args:
        space_id: Space ID
    
    Returns:
        A response containing the generated share ID
        
    Note:
        Only spaces with finished status can be shared
    """
    try:
        # Share the space
        space_share_id = space_service.share_space(space_id)
        
        # Get the updated space information
        space_dto = space_service.get_space(space_id)
        
        return APIResponse.success({
            "space_share_id": space_share_id,
            "space": space_dto.model_dump()
        })
        
    except ValueError as e:
        logger.warning(f'Share space error: {str(e)}')
        
        # Process specific error messages
        error_message = str(e)
        if "Only spaces with finished status can be shared" in error_message:
            return APIResponse.error(message=error_message, code=400)
        elif "Space not found" in error_message:
            return APIResponse.error(message=error_message, code=404)
        elif "REGISTRY_SHARE_SPACE_SERVICE_URL not configured" in error_message:
            return APIResponse.error(message="Service configuration error, unable to share", code=500)
        elif "Failed to register space to remote" in error_message or "Error connecting to remote registry" in error_message:
            return APIResponse.error(message="Remote registration service is temporarily unavailable, please try again later", code=503)
        else:
            return APIResponse.error(message=error_message, code=400)
    except Exception as e:
        logger.error(f'Share space unexpected error: {str(e)}', exc_info=True)
        return APIResponse.error(message=f"Error occurred while sharing: {str(e)}", code=500)
