from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
import os
from .dto.chunk_dto import ChunkDTO
from lpm_kernel.common.llm import LLMClient
from lpm_kernel.file_data.document_dto import DocumentDTO
from typing import List, Dict, Optional
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class EmbeddingService:
    def __init__(self):
        from lpm_kernel.file_data.chroma_utils import detect_embedding_model_dimension
        from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
        
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.llm_client = LLMClient()
        
        # Get embedding model dimension from user config
        try:
            user_llm_config_service = UserLLMConfigService()
            user_llm_config = user_llm_config_service.get_available_llm()
            
            if user_llm_config and user_llm_config.embedding_model_name:
                # Detect dimension based on model name
                self.dimension = detect_embedding_model_dimension(user_llm_config.embedding_model_name)
                logger.info(f"Detected embedding dimension: {self.dimension} for model: {user_llm_config.embedding_model_name}")
            else:
                # Default to OpenAI dimension if no config found
                self.dimension = 1536
                logger.info(f"No embedding model configured, using default dimension: {self.dimension}")
        except Exception as e:
            # Default to OpenAI dimension if error occurs
            self.dimension = 1536
            logger.error(f"Error detecting embedding dimension, using default: {self.dimension}. Error: {str(e)}", exc_info=True)

        # Check for dimension mismatches in all collections first
        collections_to_init = ["documents", "document_chunks"]
        dimension_mismatch_detected = False
        
        # First pass: check all collections for dimension mismatches
        for collection_name in collections_to_init:
            try:
                collection = self.client.get_collection(name=collection_name)
                if collection.metadata.get("dimension") != self.dimension:
                    logger.warning(f"Dimension mismatch in '{collection_name}' collection: {collection.metadata.get('dimension')} vs {self.dimension}")
                    dimension_mismatch_detected = True
            except ValueError:
                # Collection doesn't exist yet, will be created later
                pass
        
        # Handle dimension mismatch if detected in any collection
        if dimension_mismatch_detected:
            self._handle_dimension_mismatch()
        
        # Second pass: create or get collections with the correct dimension
        try:
            self.document_collection = self.client.get_collection(name="documents")
            # Verify dimension after possible reinitialization
            doc_dimension = self.document_collection.metadata.get("dimension")
            if doc_dimension != self.dimension:
                logger.error(f"Collection 'documents' still has incorrect dimension after reinitialization: {doc_dimension} vs {self.dimension}")
                # Try to reinitialize again if dimension is still incorrect
                raise RuntimeError(f"Failed to set correct dimension for 'documents' collection: {doc_dimension} vs {self.dimension}")
        except ValueError:
            # Collection doesn't exist, create it with the correct dimension
            try:
                self.document_collection = self.client.create_collection(
                    name="documents", metadata={"hnsw:space": "cosine", "dimension": self.dimension}
                )
                logger.info(f"Created 'documents' collection with dimension {self.dimension}")
            except Exception as e:
                logger.error(f"Failed to create 'documents' collection: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to create 'documents' collection: {str(e)}")

        try:
            self.chunk_collection = self.client.get_collection(name="document_chunks")
            # Verify dimension after possible reinitialization
            chunk_dimension = self.chunk_collection.metadata.get("dimension")
            if chunk_dimension != self.dimension:
                logger.error(f"Collection 'document_chunks' still has incorrect dimension after reinitialization: {chunk_dimension} vs {self.dimension}")
                # Try to reinitialize again if dimension is still incorrect
                raise RuntimeError(f"Failed to set correct dimension for 'document_chunks' collection: {chunk_dimension} vs {self.dimension}")
        except ValueError:
            # Collection doesn't exist, create it with the correct dimension
            try:
                self.chunk_collection = self.client.create_collection(
                    name="document_chunks", metadata={"hnsw:space": "cosine", "dimension": self.dimension}
                )
                logger.info(f"Created 'document_chunks' collection with dimension {self.dimension}")
            except Exception as e:
                logger.error(f"Failed to create 'document_chunks' collection: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to create 'document_chunks' collection: {str(e)}")

    def generate_document_embedding(self, document: DocumentDTO) -> List[float]:
        """Process document level embedding and store in ChromaDB"""
        try:
            if not document.raw_content:
                logger.warning(
                    f"Document {document.id} has no content to process embedding"
                )
                return None

            # get embedding
            logger.info(f"Generating embedding for document {document.id}")
            embeddings = self.llm_client.get_embedding([document.raw_content])

            if embeddings is None or len(embeddings) == 0:
                logger.error(f"Failed to get embedding for document {document.id}")
                return None

            embedding = embeddings[0]
            logger.info(f"Successfully got embedding for document {document.id}")

            # store to ChromaDB
            try:
                logger.info(f"Storing embedding for document {document.id} in ChromaDB")
                self.document_collection.add(
                    documents=[document.raw_content],
                    ids=[str(document.id)],
                    embeddings=[embedding.tolist()],
                    metadatas=[
                        {
                            "title": document.title or document.name,
                            "mime_type": document.mime_type,
                            "create_time": document.create_time.isoformat()
                            if document.create_time
                            else None,
                            "document_size": document.document_size,
                            "url": document.url,
                        }
                    ],
                )
                logger.info(f"Successfully stored embedding for document {document.id}")

                # verify embedding storage
                result = self.document_collection.get(
                    ids=[str(document.id)], include=["embeddings"]
                )
                if not result or not result["embeddings"]:
                    logger.error(
                        f"Failed to verify embedding storage for document {document.id}"
                    )
                    return None
                logger.info(f"Verified embedding storage for document {document.id}")

                return embedding

            except Exception as e:
                logger.error(f"Error storing document embedding in ChromaDB: {str(e)}", exc_info=True)
                return None

        except Exception as e:
            logger.error(f"Error processing document embedding: {str(e)}", exc_info=True)
            raise

    def generate_chunk_embeddings(self, chunks: List[ChunkDTO]) -> List[ChunkDTO]:
        """Process chunk level embeddings"""
        """
        Store in ChromaDB, the structure is as follows:
        documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
        """
        try:
            unprocessed_chunks = [c for c in chunks if not c.has_embedding]
            if not unprocessed_chunks:
                logger.info("No unprocessed chunks found")
                return chunks

            logger.info(f"Processing embeddings for {len(unprocessed_chunks)} chunks")

            contents = [c.content for c in unprocessed_chunks]
            logger.info("Getting embeddings from LLM service... {}".format(contents))
            embeddings = self.llm_client.get_embedding(contents)

            if embeddings is None or len(embeddings) == 0:
                logger.error("Failed to get embeddings from LLM service")
                return chunks

            logger.info(f"Successfully got embeddings with shape: {embeddings.shape}")

            try:
                logger.info("Adding embeddings to ChromaDB...")
                self.chunk_collection.add(
                    documents=[c.content for c in unprocessed_chunks],
                    ids=[str(c.id) for c in unprocessed_chunks],
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {
                            "document_id": str(c.document_id),
                            "topic": c.topic or "",
                            "tags": ",".join(c.tags) if c.tags else "",
                        }
                        for c in unprocessed_chunks
                    ],
                )
                logger.info("Successfully added embeddings to ChromaDB")

                # verify embeddings storage
                for chunk in unprocessed_chunks:
                    result = self.chunk_collection.get(
                        ids=[str(chunk.id)], include=["embeddings"]
                    )
                    if result and result["embeddings"]:
                        chunk.has_embedding = True
                        logger.info(f"Verified embedding for chunk {chunk.id}")
                    else:
                        logger.warning(
                            f"Failed to verify embedding for chunk {chunk.id}"
                        )
                        chunk.has_embedding = False

            except Exception as e:
                logger.error(f"Error storing embeddings in ChromaDB: {str(e)}", exc_info=True)
                for chunk in unprocessed_chunks:
                    chunk.has_embedding = False
                raise

            return chunks

        except Exception as e:
            logger.error(f"Error processing chunk embeddings: {str(e)}", exc_info=True)
            raise

    def get_chunk_embedding_by_chunk_id(self, chunk_id: int) -> Optional[List[float]]:
        """Get the corresponding embedding vector by chunk_id

        Args:
            chunk_id (int): chunk ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when chunk_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(chunk_id, int) or chunk_id < 0:
                raise ValueError("Invalid chunk_id")

            # query from ChromaDB
            result = self.chunk_collection.get(
                ids=[str(chunk_id)], include=["embeddings"]
            )

            if not result or not result["embeddings"]:
                logger.warning(f"No embedding found for chunk {chunk_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(f"Error getting embedding for chunk {chunk_id}: {str(e)}")
            raise

    def get_document_embedding_by_document_id(
        self, document_id: int
    ) -> Optional[List[float]]:
        """Get the corresponding embedding vector by document_id

        Args:
            document_id (int): document ID

        Returns:
            List[float]: embedding vector, return None if not found

        Raises:
            ValueError: when document_id is invalid
            Exception: other errors
        """
        try:
            if not isinstance(document_id, int) or document_id < 0:
                raise ValueError("Invalid document_id")

            # query from ChromaDB
            result = self.document_collection.get(
                ids=[str(document_id)], include=["embeddings"]
            )

            if not result or not result["embeddings"]:
                logger.warning(f"No embedding found for document {document_id}")
                return None

            return result["embeddings"][0]

        except Exception as e:
            logger.error(
                f"Error getting embedding for document {document_id}: {str(e)}"
            )
            raise

    def _handle_dimension_mismatch(self):
        """
        Handle dimension mismatch between current embedding model and ChromaDB collections
        This method will reinitialize ChromaDB collections with the new dimension
        """
        from lpm_kernel.file_data.chroma_utils import reinitialize_chroma_collections
        
        logger.warning(f"Detected dimension mismatch in ChromaDB collections. Reinitializing with dimension {self.dimension}")
        # Log the operation for better debugging
        logger.info(f"Calling reinitialize_chroma_collections with dimension {self.dimension}")
        
        try:
            success = reinitialize_chroma_collections(self.dimension)
            
            if success:
                logger.info(f"Successfully reinitialized ChromaDB collections with dimension {self.dimension}")
                # Refresh collection references
                try:
                    self.document_collection = self.client.get_collection(name="documents")
                    self.chunk_collection = self.client.get_collection(name="document_chunks")
                    
                    # Double-check dimensions after refresh
                    doc_dimension = self.document_collection.metadata.get("dimension")
                    chunk_dimension = self.chunk_collection.metadata.get("dimension")
                    
                    if doc_dimension != self.dimension or chunk_dimension != self.dimension:
                        logger.error(f"Dimension mismatch after refresh: documents={doc_dimension}, chunks={chunk_dimension}, expected={self.dimension}")
                        raise RuntimeError(f"Failed to handle dimension mismatch: collections have incorrect dimensions after reinitialization")
                        
                except Exception as e:
                    logger.error(f"Error refreshing collection references: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Failed to refresh ChromaDB collections after reinitialization: {str(e)}")
            else:
                logger.error("Failed to reinitialize ChromaDB collections")
                raise RuntimeError("Failed to handle dimension mismatch in ChromaDB collections")
        except Exception as e:
            logger.error(f"Error during dimension mismatch handling: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to handle dimension mismatch in ChromaDB collections: {str(e)}")
    
    def search_similar_chunks(
        self, query: str, limit: int = 5
    ) -> List[Tuple[ChunkDTO, float]]:
        """Search similar chunks, return list of ChunkDTO objects and their similarity scores

        Args:
            query (str): query text
            limit (int, optional): return result limit. Defaults to 5.

        Returns:
            List[Tuple[ChunkDTO, float]]: return list of (ChunkDTO, similarity score), sorted by similarity score in descending order

        Raises:
            ValueError: when query parameters are invalid
            Exception: other errors
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query string cannot be empty")

            if limit < 1:
                raise ValueError("Limit must be positive")

            # calculate query text embedding
            query_embedding = self.llm_client.get_embedding([query])
            if query_embedding is None or len(query_embedding) == 0:
                raise Exception("Failed to generate embedding for query")

            # query ChromaDB
            results = self.chunk_collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["ids"]:
                return []

            # convert results to ChunkDTO objects
            similar_chunks = []
            for i in range(len(results["ids"])):
                chunk_id = results["ids"][0][i]  # ChromaDB returns nested lists
                document_id = results["metadatas"][0][i]["document_id"]
                content = results["documents"][0][i]
                topic = results["metadatas"][0][i].get("topic", "")
                tags = (
                    results["metadatas"][0][i].get("tags", "").split(",")
                    if results["metadatas"][0][i].get("tags")
                    else []
                )

                # calculate similarity score (ChromaDB returns distances, need to convert to similarity)
                similarity_score = (
                    1 - results["distances"][0][i]
                )  # assume using Euclidean distance or cosine distance

                chunk = ChunkDTO(
                    id=int(chunk_id),
                    document_id=int(document_id),
                    content=content,
                    topic=topic,
                    tags=tags,
                    has_embedding=True,
                )

                similar_chunks.append((chunk, similarity_score))

            # sort by similarity score in descending order
            similar_chunks.sort(key=lambda x: x[1], reverse=True)

            return similar_chunks

        except ValueError as ve:
            logger.error(f"Invalid input parameters: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise