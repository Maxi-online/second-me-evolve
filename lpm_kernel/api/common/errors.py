from typing import Any


class ErrorCodes:
    SUCCESS = 0
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_ERROR = 500
    CONFIGURATION_ERROR = 501
    PATH_NOT_FOUND = 10404


class APIError(Exception):
    def __init__(
        self, message: str, code: int = ErrorCodes.INTERNAL_ERROR, data: Any = None
    ):
        self.message = message
        self.code = code
        self.data = data
        super().__init__(self.message)
