from typing import Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    @staticmethod
    def success(data: Any = None, message: str = "success") -> dict:
        return {"code": 0, "message": message, "data": data}

    @staticmethod
    def error(message: str, code: int = 1, data: Any = None) -> dict:
        return {"code": code, "message": message, "data": data}


class ResponseHandler:
    """HTTP Response Handler Utility Class"""
    
    @staticmethod
    def handle_response(response, success_log=None, error_prefix="Operation"):
        """
        Handle standard HTTP response
        
        Args:
            response: requests.Response object
            success_log: Success log message (optional)
            error_prefix: Error message prefix
            
        Returns:
            dict: Response data
            
        Raises:
            Exception: When response is not successful
        """
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 0:
                if success_log:
                    logger.info(success_log)
                return data.get("data")
            else:
                error_text = data.get("message", "Unknown error")
                logger.error(f"{error_prefix} failed: {error_text}")
                raise Exception(f"{error_prefix} failed: {error_text}")
        else:
            error_text = response.text
            logger.error(f"{error_prefix} failed: {error_text}")
            raise Exception(f"{error_prefix} failed: {error_text}")
