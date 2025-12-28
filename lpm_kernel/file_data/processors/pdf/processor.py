from pathlib import Path
import fitz  # PyMuPDF

# from ...core.processor import BaseFileProcessor
from ...core.file_type import FileType
from ...core.decorators import processor_register
from ...core.exceptions import FileProcessingError
from ...document import Document, ProcessStatus
from lpm_kernel.file_data.processors.processor import BaseFileProcessor


@processor_register
class PDFProcessor(BaseFileProcessor):
    SUPPORTED_TYPES = {FileType.PDF}

    @classmethod
    def _process_file(cls, file_path: Path, doc: Document) -> Document:
        try:
            with fitz.open(file_path) as pdf:
                text = ""
                for page in pdf:
                    text += page.get_text()

                doc.raw_content = text
                doc.extract_status = ProcessStatus.SUCCESS

        except Exception as e:
            doc.extract_status = ProcessStatus.FAILED
            raise FileProcessingError(f"Failed to process PDF: {str(e)}")

        return doc
