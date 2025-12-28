from typing import Type
from ..processors.processor import BaseFileProcessor
import logging

logger = logging.getLogger(__name__)


def processor_register(
    processor_class: Type[BaseFileProcessor],
) -> Type[BaseFileProcessor]:
    """processor register decorator"""
    from ..process_factory import ProcessorFactory  # import directly from file_data directory

    logger.info("Registering processor: %s", processor_class)
    ProcessorFactory.register(processor_class)
    return processor_class
