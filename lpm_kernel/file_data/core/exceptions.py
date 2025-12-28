class FileProcessError(Exception):
    """File processing base exception"""

    pass


class UnsupportedFileType(FileProcessError):
    """Unsupported file type"""

    pass


class FileProcessingError(FileProcessError):
    """File processing error"""

    pass
