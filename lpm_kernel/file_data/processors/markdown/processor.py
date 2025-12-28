# lpm_kernel/file_data/processors/markdown/processor.py
from pathlib import Path

from lpm_kernel.file_data.processors.processor import BaseFileProcessor
from ...core.file_type import FileType
from ...core.decorators import processor_register
from ...core.exceptions import FileProcessingError
from ...document import Document, ProcessStatus


@processor_register
class MarkdownProcessor(BaseFileProcessor):
    SUPPORTED_TYPES = {FileType.MARKDOWN}

    @classmethod
    def _process_file(cls, file_path: Path, doc: Document) -> Document:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                doc.raw_content = text
                doc.extract_status = ProcessStatus.SUCCESS

        except Exception as e:
            doc.extract_status = ProcessStatus.FAILED
            raise FileProcessingError(f"Failed to process markdown file: {str(e)}")

        return doc
