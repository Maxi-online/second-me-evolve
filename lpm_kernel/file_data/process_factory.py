from typing import Type, Dict, Optional, List
from .processors.processor import BaseFileProcessor
from .core.file_type import FileType
from pathlib import Path
from .document import Document
import logging

logger = logging.getLogger(__name__)


class ProcessorFactory:
    _processors: Dict[FileType, Type[BaseFileProcessor]] = {}
    _initialized = False

    @classmethod
    def register(cls, processor_class: Type[BaseFileProcessor]):
        """Register processor"""
        for file_type in processor_class.SUPPORTED_TYPES:
            cls._processors[file_type] = processor_class
            print(
                f"Registered processor {processor_class.__name__} for type {file_type}"
            )

    @classmethod
    def get_processor(cls, file_type: FileType) -> Type[BaseFileProcessor]:
        """Get processor before ensuring initialization"""
        if not cls._initialized:
            cls.init()
        print(f"Current registered processors: {cls._processors}")
        if file_type not in cls._processors:
            raise ValueError(f"No processor found for {file_type}")
        return cls._processors[file_type]

    @classmethod
    def init(cls):
        """Explicit initialization"""
        if not cls._initialized:
            from .core.discovery import auto_discover_processors

            auto_discover_processors()
            cls._initialized = True

    @classmethod
    def auto_detect_and_process(cls, file_path: str) -> Document:
        """
        Automatically detect file type and process
        :param file_path: file path
        :return: Document object
        """
        logger.info("Available processors: %s", ProcessorFactory._processors)
        path = Path(file_path)
        # use BaseFileProcessor's type detection method
        file_type = BaseFileProcessor._detect_type(path, None)
        # get corresponding processor and process
        processor = cls.get_processor(file_type)
        return processor.process(file_path)

    @classmethod
    def process_directory(
        cls,
        directory_path: str,
        file_type: Optional[FileType] = None,
        recursive: bool = False,
    ) -> List[Document]:
        """
        Process all files in the specified directory
        :param directory_path: directory path
        :param file_type: specified file type (optional)
        :param recursive: whether to process subdirectories
        :return: list of processed Document objects
        """
        if not cls._initialized:
            cls.init()

        documents = []
        # path = Path(directory_path)

        if file_type:
            # if specified file type, only use corresponding processor
            processor = cls.get_processor(file_type)
            documents.extend(
                processor.process_directory(directory_path, file_type, recursive)
            )
        else:
            # if no specified file type, process all supported file types
            for file_type, processor in cls._processors.items():
                documents.extend(
                    processor.process_directory(directory_path, file_type, recursive)
                )

        return documents
