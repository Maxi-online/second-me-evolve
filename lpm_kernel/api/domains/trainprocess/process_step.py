from enum import Enum
from typing import List


class ProcessStep(Enum):
    """Training process steps"""

    LIST_DOCUMENTS = "list_documents"
    GENERATE_DOCUMENT_EMBEDDINGS = "generate_document_embeddings"
    CHUNK_DOCUMENT = "process_chunks"
    CHUNK_EMBEDDING = "chunk_embedding"
    EXTRACT_DIMENSIONAL_TOPICS = "extract_dimensional_topics"
    GENERATE_BIOGRAPHY = "generate_biography"
    MODEL_DOWNLOAD = "model_download"
    MAP_ENTITY_NETWORK = "map_your_entity_network"
    DECODE_PREFERENCE_PATTERNS = "decode_preference_patterns"
    REINFORCE_IDENTITY = "reinforce_identity"
    AUGMENT_CONTENT_RETENTION = "augment_content_retention"
    TRAIN = "train"
    MERGE_WEIGHTS = "merge_weights"
    CONVERT_MODEL = "convert_model"

    @classmethod
    def get_ordered_steps(cls) -> List["ProcessStep"]:
        """Get ordered steps"""
        return [
            cls.MODEL_DOWNLOAD,
            cls.LIST_DOCUMENTS,
            cls.GENERATE_DOCUMENT_EMBEDDINGS,
            cls.CHUNK_DOCUMENT,
            cls.CHUNK_EMBEDDING,
            cls.EXTRACT_DIMENSIONAL_TOPICS,
            cls.GENERATE_BIOGRAPHY,
            cls.MAP_ENTITY_NETWORK,
            cls.DECODE_PREFERENCE_PATTERNS,
            cls.REINFORCE_IDENTITY,
            cls.AUGMENT_CONTENT_RETENTION,
            cls.TRAIN,
            cls.MERGE_WEIGHTS,
            cls.CONVERT_MODEL,
        ]
        
    def get_method_name(self) -> str:
        """Get the corresponding method name for this step"""
        return self.value
