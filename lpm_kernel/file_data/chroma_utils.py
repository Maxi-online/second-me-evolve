from typing import Optional, Dict, Any, List, Tuple
import os
import chromadb
import logging
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


def get_embedding_dimension(embedding: List[float]) -> int:
    """
    Get the dimension of an embedding vector
    
    Args:
        embedding: The embedding vector
        
    Returns:
        The dimension of the embedding vector
    """
    return len(embedding)


def detect_embedding_model_dimension(model_name: str) -> Optional[int]:
    """
    Detect the dimension of an embedding model based on its name
    This is a fallback method when we can't get a sample embedding
    
    Args:
        model_name: The name of the embedding model
        
    Returns:
        The dimension of the embedding model, or None if unknown
    """
    # Common embedding model dimensions
    model_dimensions = {
        # OpenAI models
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        
        # Ollama models
        "snowflake-arctic-embed": 768,
        "snowflake-arctic-embed:110m": 768,
        "nomic-embed-text": 768,
        "nomic-embed-text:v1.5": 768,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large:v1": 1024,
    }
    
    # Try to find exact match
    if model_name in model_dimensions:
        return model_dimensions[model_name]
    
    # Try to find partial match
    for model, dimension in model_dimensions.items():
        if model in model_name:
            return dimension
    
    # Default to OpenAI dimension if unknown
    logger.warning(f"Unknown embedding model: {model_name}, defaulting to 1536 dimensions")
    return 1536


def reinitialize_chroma_collections(dimension: int = 1536) -> bool:
    """
    Reinitialize ChromaDB collections with a new dimension
    
    Args:
        dimension: The new dimension for the collections
        
    Returns:
        True if successful, False otherwise
    """
    try:
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        
        # Delete and recreate document collection
        try:
            # Check if collection exists before attempting to delete
            try:
                client.get_collection(name="documents")
                client.delete_collection(name="documents")
                logger.info("Deleted 'documents' collection")
            except ValueError:
                logger.info("'documents' collection does not exist, will create new")
        except Exception as e:
            logger.error(f"Error deleting 'documents' collection: {str(e)}", exc_info=True)
            return False
        
        # Create document collection with new dimension
        try:
            client.create_collection(
                name="documents",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": dimension
                }
            )
            logger.info(f"Created 'documents' collection with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error creating 'documents' collection: {str(e)}", exc_info=True)
            return False
        
        # Delete and recreate chunk collection
        try:
            # Check if collection exists before attempting to delete
            try:
                client.get_collection(name="document_chunks")
                client.delete_collection(name="document_chunks")
                logger.info("Deleted 'document_chunks' collection")
            except ValueError:
                logger.info("'document_chunks' collection does not exist, will create new")
        except Exception as e:
            logger.error(f"Error deleting 'document_chunks' collection: {str(e)}", exc_info=True)
            return False
        
        # Create chunk collection with new dimension
        try:
            client.create_collection(
                name="document_chunks",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": dimension
                }
            )
            logger.info(f"Created 'document_chunks' collection with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error creating 'document_chunks' collection: {str(e)}", exc_info=True)
            return False
        
        # Verify collections were created with correct dimension
        try:
            doc_collection = client.get_collection(name="documents")
            chunk_collection = client.get_collection(name="document_chunks")
            
            doc_dimension = doc_collection.metadata.get("dimension")
            if doc_dimension != dimension:
                logger.error(f"Verification failed: 'documents' collection has incorrect dimension: {doc_dimension} vs {dimension}")
                return False
                
            chunk_dimension = chunk_collection.metadata.get("dimension")
            if chunk_dimension != dimension:
                logger.error(f"Verification failed: 'document_chunks' collection has incorrect dimension: {chunk_dimension} vs {dimension}")
                return False
                
            logger.info(f"Verification successful: Both collections have correct dimension: {dimension}")
        except Exception as e:
            logger.error(f"Error verifying collections: {str(e)}", exc_info=True)
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error reinitializing ChromaDB collections: {str(e)}", exc_info=True)
        return False