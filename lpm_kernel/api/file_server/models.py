from typing import Optional, List
from pydantic import BaseModel


class FileItem(BaseModel):
    name: str
    type: str  # "file" or "directory"
    size: Optional[int]
    path: str
    url: Optional[str]


class DirectoryListing(BaseModel):
    current_path: str
    items: List[FileItem]
