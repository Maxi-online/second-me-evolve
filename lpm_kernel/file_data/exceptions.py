class FileProcessingError(Exception):
    """Base class for file processing errors"""

    pass


class UnsupportedFileType(FileProcessingError):
    """Unsupported file type"""

    pass


class FileReadError(FileProcessingError):
    """File reading error"""

    pass


class FileWriteError(FileProcessingError):
    """File writing error"""

    pass


class ProcessingError(FileProcessingError):
    """Error during processing"""

    pass
