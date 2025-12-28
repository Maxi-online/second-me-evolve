from flask import Blueprint, request, jsonify
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.file_data.memory_service import StorageService
from lpm_kernel.configs.config import Config
from lpm_kernel.common.logging import logger
from lpm_kernel.file_data.document_service import DocumentService

memories_bp = Blueprint("memories", __name__)
storage_service = StorageService(Config.from_env())

# Allowed file formats
ALLOWED_EXTENSIONS = {"txt", "pdf", "md"}


def allowed_file(filename):
    """Check if the file has an allowed format"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@memories_bp.route("/api/memories/file", methods=["POST"])
def upload_file():
    """
    File upload API
    """
    try:
        logger.info("Starting to process file upload request")

        # Check if there is a file
        if "file" not in request.files:
            logger.warning("No file in request")
            return APIResponse.error(message="No file was uploaded", code=400)

        file = request.files["file"]
        if file.filename == "":
            logger.warning("Filename is empty")
            return APIResponse.error(message="No file was selected", code=400)

        logger.info(f"Received file: {file.filename}")

        # Validate file format
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file format: {file.filename}")
            return APIResponse.error(
                message=f'Unsupported file format. Only the following formats are allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                code=400,
            )

        # Get metadata
        metadata = request.form.to_dict() if request.form else None
        logger.info(f"File metadata: {metadata}")

        try:
            # Save file and process document
            memory, document = storage_service.save_file(file, metadata)
        except ValueError as e:
            # Handle case where file already exists
            logger.warning(f"File upload failed: {str(e)}")
            return APIResponse.error(
                message=str(e),
                code=409,  # Use 409 Conflict status code to indicate resource conflict
            )

        # Prepare response data
        result = memory.to_dict()
        if document:
            result["document"] = {
                "id": document.id,
                "name": document.name,
                "title": document.title,
                "mime_type": document.mime_type,
                "document_size": document.document_size,
                "extract_status": document.extract_status.value
                if document.extract_status
                else None,
                "embedding_status": document.embedding_status.value
                if document.embedding_status
                else None,
                "create_time": document.create_time.isoformat()
                if document.create_time
                else None,
            }

        logger.info("File upload processing completed")
        return APIResponse.success(data=result, message="Upload successful")

    except ValueError as e:
        logger.error(f"Request parameter error: {str(e)}")
        return APIResponse.error(message=str(e), code=400)
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        return APIResponse.error(message=f"Internal server error: {str(e)}", code=500)


@memories_bp.route("/api/memories/file/<filename>", methods=["DELETE"])
def delete_file(filename):
    """
    File deletion API
    """
    try:
        logger.info(f"Starting to process file deletion request: {filename}")
        
        # Call DocumentService to delete file
        document_service = DocumentService()
        result = document_service.delete_file_by_name(filename)
        
        if result:
            logger.info(f"File deleted successfully: {filename}")
            return APIResponse.success(message=f"File '{filename}' has been successfully deleted")
        else:
            logger.warning(f"File does not exist or deletion failed: {filename}")
            return APIResponse.error(message=f"File '{filename}' does not exist or cannot be deleted", code=404)
            
    except Exception as e:
        logger.error(f"Error occurred while deleting file: {str(e)}", exc_info=True)
        return APIResponse.error(message=f"Internal server error: {str(e)}", code=500)
