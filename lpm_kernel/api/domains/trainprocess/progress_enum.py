from typing import List, Dict, Optional, Union, Any
import json
from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"