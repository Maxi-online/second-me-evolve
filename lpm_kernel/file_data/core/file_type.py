from enum import Enum
from typing import Dict


class FileType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    EXCEL = "xlsx"
    TEXT = "text"
    MARKDOWN = "md"

    @classmethod
    def get_mime_mapping(cls) -> Dict[str, "FileType"]:
        return {
            ".pdf": cls.PDF,
            ".jpg": cls.IMAGE,
            ".jpeg": cls.IMAGE,
            ".png": cls.IMAGE,
            ".docx": cls.DOCX,
            ".xlsx": cls.EXCEL,
            ".txt": cls.TEXT,
            ".md": cls.MARKDOWN,
        }
