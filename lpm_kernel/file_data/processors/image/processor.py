from pathlib import Path
from PIL import Image
import pytesseract
from lpm_kernel.file_data.processors.processor import BaseFileProcessor
from ...core.file_type import FileType
from ...core.exceptions import FileProcessingError
from ...document import Document, ProcessStatus
from ...core.decorators import processor_register


@processor_register
class ImageProcessor(BaseFileProcessor):
    SUPPORTED_TYPES = {FileType.IMAGE}

    @classmethod
    def _process_file(cls, file_path: Path, doc: Document) -> Document:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

            doc.raw_content = text
            doc.extract_status = ProcessStatus.SUCCESS

        except Exception as e:
            doc.extract_status = ProcessStatus.FAILED
            raise FileProcessingError(f"Failed to process image: {str(e)}")

        return doc
