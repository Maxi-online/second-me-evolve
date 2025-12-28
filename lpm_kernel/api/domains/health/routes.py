from flask import Blueprint, jsonify, send_from_directory
from ...common.responses import APIResponse
from ...common.errors import APIError, ErrorCodes
import logging
from ....configs.config import Config
import os

logger = logging.getLogger(__name__)
health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        config = Config.from_env()
        app_name = config.app_name or "Service"  # add default value to prevent None

        return jsonify(
            APIResponse.success(data={"status": "ok"}, message=f"{app_name} is healthy")
        )
    except APIError as e:
        logger.error(f"API error in health check: {str(e)}")
        return jsonify(APIResponse.error(message=e.message, code=e.code, data=e.data))
    except Exception as e:
        logger.error(f"Unexpected error in health check: {str(e)}")
        return jsonify(
            APIResponse.error(
                message="Internal server error", code=ErrorCodes.INTERNAL_ERROR
            )
        )


@health_bp.route("/favicon.ico")
def favicon():
    """Provide website icon"""
    return send_from_directory(
        os.path.join(health_bp.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )
