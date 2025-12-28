from enum import Enum


class ProcessStatus(Enum):
    """Process status enum"""

    INITIALIZED = "INITIALIZED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
