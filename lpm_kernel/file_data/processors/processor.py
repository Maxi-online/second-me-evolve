from pathlib import Path
from typing import Optional, Set, List
from ..core.file_type import FileType
from ..core.exceptions import UnsupportedFileType
from ..document import Document, ProcessStatus
from ..core.exceptions import FileProcessingError
import logging

logger = logging.getLogger(__name__)


class BaseFileProcessor:
    """Base processor"""

    # processor supported file types
    SUPPORTED_TYPES: Set[FileType] = set()

    @classmethod
    def process(
        cls, file_path: str, expected_type: Optional[FileType] = None
    ) -> Document:
        """
        Main entry point for processing files
        :param file_path: file path
        :param expected_type: expected file type
        :return: Document object
        """
        path = Path(file_path)
        file_type = cls._detect_type(path, expected_type)

        if file_type not in cls.SUPPORTED_TYPES:
            raise UnsupportedFileType(f"{cls.__name__} doesn't support {file_type}")

        return cls._process_file(path, cls._create_document(path, file_type))

    @classmethod
    def _detect_type(
        cls, path: Path, expected_type: Optional[FileType] = None
    ) -> FileType:
        """Detect file type"""
        if expected_type:
            return expected_type

        suffix = path.suffix.lower()
        mime_mapping = FileType.get_mime_mapping()

        if suffix not in mime_mapping:
            raise UnsupportedFileType(f"Unsupported file type: {suffix}")

        return mime_mapping[suffix]

    @classmethod
    def _create_document(cls, path: Path, file_type: FileType) -> Document:
        """Create base document object"""
        # try:
        #     file_size = path.stat().st_size if path.exists() else 0
        # except (OSError, IOError) as e:
        #     raise FileProcessingError(f"Cannot access file {path}: {str(e)}")

        """Create base document object"""
        return Document(
            name=path.name, mime_type=file_type.value, document_size=path.stat().st_size
        )

    @classmethod
    def _process_file(cls, file_path: Path, doc: Document) -> Document:
        """Specific processing logic implemented by subclasses"""
        raise NotImplementedError

    @classmethod
    def process_directory(
        cls,
        directory_path: str,
        expected_type: Optional[FileType] = None,
        recursive: bool = False,
    ) -> List[Document]:
        """
        Process all files in the directory
        :param directory_path: directory path
        :param expected_type: expected file type
        :param recursive: whether to process subdirectories
        :return: list of Document objects
        """
        path = Path(directory_path)
        logger.info(f"Processing directory: {path}")

        if not path.is_dir():
            logger.error(f"{directory_path} is not a directory")
            raise FileProcessingError(f"{directory_path} is not a directory")

        documents = []
        pattern = "**/*" if recursive else "*"

        # list all files
        files = list(path.glob(pattern))
        logger.info(f"Found files: {files}")

        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    logger.info(f"Processing file: {file_path}")
                    logger.info(f"File suffix: {file_path.suffix}")
                    logger.info(f"Supported types: {cls.SUPPORTED_TYPES}")

                    doc = cls.process(str(file_path), expected_type)
                    doc.status = ProcessStatus.SUCCESS
                    documents.append(doc)
                    logger.info(f"Successfully processed file: {file_path}")
                except UnsupportedFileType as e:
                    logger.warning(f"Unsupported file type: {file_path} - {str(e)}")
                    # create a document object representing failed processing
                    doc = Document(
                        name=file_path.name,
                        mime_type="unknown",
                        document_size=file_path.stat().st_size,
                        status=ProcessStatus.FAILED,
                        error_message=f"Unsupported file type: {str(e)}",
                    )
        return documents
