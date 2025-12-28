"""
service about knowledge retrieve
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from lpm_kernel.file_data.embedding_service import EmbeddingService, ChunkDTO
from lpm_kernel.kernel.l1.l1_manager import get_latest_global_bio

logger = logging.getLogger(__name__)


class L0KnowledgeRetriever:
    """L0 knowledge retriever"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.7,
        max_chunks: int = 3,
    ):
        """
        init L0 knowledge retriever

        Args:
            embedding_service: Embedding service instance
            similarity_threshold: only return contents whose similarity bigger than this value
            max_chunks: the maximum number of return chunks
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks

    def retrieve(self, query: str) -> str:
        """
        retrieve L0 knowledge

        Args:
            query: query content

        Returns:
            str: structured knowledge content, or empty string if no relevant knowledge found
        """
        try:
            # search related chunks
            similar_chunks: List[
                Tuple[ChunkDTO, float]
            ] = self.embedding_service.search_similar_chunks(
                query=query, limit=self.max_chunks
            )

            # filter out low similarity chunks
            if not similar_chunks:
                return ""

            knowledge_parts = []
            for chunk, similarity in similar_chunks:
                if similarity >= self.similarity_threshold:
                    knowledge_parts.append(chunk.content)

            if not knowledge_parts:
                return ""

            # merge multiple knowledge parts into one
            return "\n\n".join(knowledge_parts)

        except Exception as e:
            logger.error(f"L0 knowledge retrieval failed: {str(e)}")
            return ""


class L1KnowledgeRetriever:
    """L1 knowledge retriever"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.7,
        max_shades: int = 3,
    ):
        """
        init L1 knowledge retriever

        Args:
            embedding_service: Embedding service instance
            similarity_threshold: only return contents whose similarity bigger than this value
            max_shades: the maximum number of return shades
        """
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.max_shades = max_shades

    def retrieve(self, query: str) -> str:
        """
        search related L1 shades

        Args:
            query: query content

        Returns:
            str: structured knowledge content, or empty string if no relevant knowledge found
        """
        try:
            # get global bio shades
            global_bio = get_latest_global_bio()
            if not global_bio or not global_bio.shades:
                logger.info("Global Bio not found or Shades is empty")
                return ""

            # get query embedding
            query_embedding = self.embedding_service.get_embedding(query)
            if not query_embedding:
                logger.error("Failed to get embedding for query text")
                return ""

            # get all shades' embeddings
            shade_embeddings = []
            for shade in global_bio.shades:
                shade_text = (
                    f"{shade.get('title', '')} - {shade.get('description', '')}"
                )
                embedding = self.embedding_service.get_embedding(shade_text)
                if embedding:
                    shade_embeddings.append((shade, embedding))

            if not shade_embeddings:
                logger.info("No available Shades embeddings found")
                return ""

            # calculate similarity and sort
            similar_shades = []
            for shade, embedding in shade_embeddings:
                similarity = self.embedding_service.calculate_similarity(
                    query_embedding, embedding
                )
                if similarity >= self.similarity_threshold:
                    similar_shades.append((shade, similarity))

            # sort according to similarity and limit the number of returned shades
            similar_shades.sort(key=lambda x: x[1], reverse=True)
            similar_shades = similar_shades[: self.max_shades]

            if not similar_shades:
                return ""

            # structured output
            shade_parts = []
            for shade, similarity in similar_shades:
                shade_text = f"Shade: {shade.get('title', '')}\n"
                shade_text += f"Description: {shade.get('description', '')}\n"
                shade_text += f"Similarity: {similarity:.2f}"
                shade_parts.append(shade_text)

            return "\n\n".join(shade_parts)

        except Exception as e:
            logger.error(f"L1 knowledge retrieval failed: {str(e)}")
            return ""


# create overall knowledge retriever instance
default_retriever = L0KnowledgeRetriever(
    embedding_service=EmbeddingService(), similarity_threshold=0.7, max_chunks=3
)

default_l1_retriever = L1KnowledgeRetriever(
    embedding_service=EmbeddingService(), similarity_threshold=0.7, max_shades=3
)
