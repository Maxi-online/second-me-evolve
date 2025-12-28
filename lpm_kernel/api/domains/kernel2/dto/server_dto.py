"""
ServerStatus related data transfer objects
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ProcessInfo:
    """Process Information"""

    pid: int  # process ID
    cpu_percent: float  # CPU usage percentage
    memory_percent: float  # memory usage percentage
    create_time: float
    cmdline: List[str]


@dataclass
class ServerStatus:
    """server status"""

    is_running: bool  # if service is running
    process_info: Optional[ProcessInfo] = None  # process info

    @classmethod
    def not_running(cls) -> "ServerStatus":
        """create a ServerStatus object representing a not running server"""
        return cls(is_running=False)

    @classmethod
    def running(cls, process_info: ProcessInfo) -> "ServerStatus":
        """create a ServerStatus object representing a running server"""
        return cls(is_running=True, process_info=process_info)
