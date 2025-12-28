# lpm_kernel/common/repository/vector_store_factory.py

from typing import Optional
from .vector_repository import ChromaRepository, BaseVectorRepository
from lpm_kernel.configs.config import Config


class VectorStoreFactory:
    _instance: Optional[BaseVectorRepository] = None

    @classmethod
    def get_instance(cls) -> BaseVectorRepository:
        if cls._instance is None:
            config = Config.from_env()
            cls._instance = ChromaRepository(
                collection_name=config.CHROMA_COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            )
        return cls._instance
