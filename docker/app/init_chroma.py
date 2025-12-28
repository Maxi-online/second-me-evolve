import chromadb
import os
import sys

# Add project root to path to import from lpm_kernel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.file_data.chroma_utils import detect_embedding_model_dimension, reinitialize_chroma_collections

def init_chroma_db():
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # ensure the directory is correct
    os.makedirs(chroma_path, exist_ok=True)

    # Get embedding model dimension from user config
    try:
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        
        if user_llm_config and user_llm_config.embedding_model_name:
            # Detect dimension based on model name
            dimension = detect_embedding_model_dimension(user_llm_config.embedding_model_name)
            print(f"Detected embedding dimension: {dimension} for model: {user_llm_config.embedding_model_name}")
        else:
            # Default to OpenAI dimension if no config found
            dimension = 1536
            print(f"No embedding model configured, using default dimension: {dimension}")
    except Exception as e:
        # Default to OpenAI dimension if error occurs
        dimension = 1536
        print(f"Error detecting embedding dimension, using default: {dimension}. Error: {e}")

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collections_to_init = ["documents", "document_chunks"]
        dimension_mismatch_detected = False
        
        # Check all collections for dimension mismatches first
        for collection_name in collections_to_init:
            try:
                collection = client.get_collection(name=collection_name)
                print(f"Collection '{collection_name}' already exists")
                
                # Check if existing collection has the correct dimension
                if collection.metadata.get("dimension") != dimension:
                    print(f"Warning: Existing '{collection_name}' collection has dimension {collection.metadata.get('dimension')}, but current model requires {dimension}")
                    dimension_mismatch_detected = True
            except ValueError:
                # Collection doesn't exist yet, will be created later
                pass
        
        # Handle dimension mismatch if detected in any collection
        if dimension_mismatch_detected:
            print("Automatically reinitializing ChromaDB collections with the new dimension...")
            if reinitialize_chroma_collections(dimension):
                print("Successfully reinitialized ChromaDB collections with the new dimension")
            else:
                print("Failed to reinitialize ChromaDB collections, you may need to manually delete the data/chroma_db directory")
        
        # Create or get collections with the correct dimension
        for collection_name in collections_to_init:
            try:
                collection = client.get_collection(name=collection_name)
                # Verify dimension after possible reinitialization
                if collection.metadata.get("dimension") != dimension:
                    print(f"Error: Collection '{collection_name}' still has incorrect dimension after reinitialization: {collection.metadata.get('dimension')} vs {dimension}")
            except ValueError:
                # Create collection if it doesn't exist
                collection = client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "dimension": dimension
                    }
                )
                print(f"Successfully created collection '{collection_name}' with dimension {dimension}")

        
        print(f"ChromaDB initialized at {chroma_path}")
    except Exception as e:
        print(f"An error occurred while initializing ChromaDB: {e}")
        # no exception for following process
        # ChromaRepository will create collection if needed

if __name__ == "__main__":
    init_chroma_db()
