from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkDTO:
    id: int
    document_id: int
    content: str
    embedding: Optional[List[float]] = None
    has_embedding: bool = False
    tags: Optional[List[str]] = None
    topic: Optional[str] = None
    length: Optional[int] = None
