from pathlib import Path

# Changed to relative import
from lpm_kernel.file_data.processors.processor import BaseFileProcessor
from ...core.file_type import FileType
from ...core.decorators import processor_register
from ...core.exceptions import FileProcessingError
from ...document import Document, ProcessStatus

print("Loading Text processor...")  # Add debug statement


@processor_register
class TEXTProcessor(BaseFileProcessor):
    SUPPORTED_TYPES = {FileType.TEXT}
    # Define supported encoding list
    SUPPORTED_ENCODINGS = [
        'utf-8',        # Unicode encoding, most common
        'utf-8-sig',    # UTF-8 with BOM
        'utf-16',       # Unicode 16-bit encoding
        'gbk',          # Chinese encoding
        'gb2312',       # Subset of Chinese encoding
        'gb18030',      # Superset of Chinese encoding
        'big5',         # Traditional Chinese encoding
        'iso-8859-1',   # Western European encoding
        'ascii',        # ASCII encoding
        'cp936',        # Microsoft Chinese encoding
        'shift-jis',    # Japanese encoding
        'euc-jp',       # Japanese encoding
        'euc-kr',       # Korean encoding
    ]

    @classmethod
    def _process_file(cls, file_path: Path, doc: Document) -> Document:
        last_exception = None
        
        # Try different encoding formats
        for encoding in cls.SUPPORTED_ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    text = file.read()
                    doc.raw_content = text
                    doc.extract_status = ProcessStatus.SUCCESS
                    return doc
            except UnicodeDecodeError as e:
                last_exception = e
                continue
            except Exception as e:
                doc.extract_status = ProcessStatus.FAILED
                raise FileProcessingError(f"Failed to process text file: {str(e)}")

        # If all encodings failed
        doc.extract_status = ProcessStatus.FAILED
        raise FileProcessingError(f"Failed to process text file with all supported encodings: {str(last_exception)}")
