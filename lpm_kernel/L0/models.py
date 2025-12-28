from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass
class FileInfo:
    """File-related information"""

    data_type: str
    filename: str
    content: str
    file_content: Optional[Any]


class DocumentType(str, Enum):
    DOCUMENT = "DOCUMENT"
    TEXT = "TEXT"

    @classmethod
    def from_mime_type(cls, mime_type: str) -> "DocumentType":
        """Converts a MIME type to DocumentType.
        
        Args:
            mime_type: String representing the MIME type.
            
        Returns:
            The appropriate DocumentType based on the given MIME type.
        """
        if mime_type == "text":
            return cls.TEXT
        elif mime_type == "pdf":
            return cls.DOCUMENT
        elif mime_type == "md":
            return cls.DOCUMENT
        else:
            return cls.DOCUMENT


@dataclass
class BioInfo:
    """User biographical information"""

    global_bio: str
    status_bio: str
    about_me: str


@dataclass
class InsighterInput:
    """Raw input parameters for the Insighter"""

    file_info: FileInfo
    bio_info: BioInfo

    @classmethod
    def from_dict(cls, inputs: dict) -> "InsighterInput":
        """Creates an InsighterInput instance from a dictionary.
        
        Args:
            inputs: Dictionary containing the input parameters.
            
        Returns:
            An InsighterInput object populated with values from the dictionary.
        """
        return cls(
            file_info=FileInfo(
                data_type=inputs.get("dataType", "DOCUMENT"),
                filename=inputs.get("filename", ""),
                content=inputs.get("content", "").strip(),
                file_content=inputs.get("fileContent", ""),
            ),
            bio_info=BioInfo(
                global_bio=inputs.get("globalBio", ""),
                status_bio=inputs.get("statusBio", ""),
                about_me=inputs.get("aboutMe", ""),
            ),
        )


@dataclass
class SummarizerInput:
    """Raw input parameters for the Summarizer"""

    file_info: FileInfo
    insight: str

    @classmethod
    def from_dict(cls, inputs: dict) -> "SummarizerInput":
        """Creates a SummarizerInput instance from a dictionary.
        
        Args:
            inputs: Dictionary containing the input parameters.
            
        Returns:
            A SummarizerInput object populated with values from the dictionary.
        """
        return cls(
            file_info=FileInfo(
                data_type=inputs.get("dataType", "DOCUMENT"),
                filename=inputs.get("filename", ""),
                content=inputs.get("content", "").strip(),
                file_content=inputs.get("fileContent", ""),
            ),
            insight=inputs.get("insight", ""),
        )
