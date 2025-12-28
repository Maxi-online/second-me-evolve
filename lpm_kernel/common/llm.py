from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from typing import List, Union
import requests
import numpy as np
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()
import lpm_kernel.common.strategy.classification as classification
from sentence_transformers import SentenceTransformer
import json

class EmbeddingError(Exception):
    """Custom exception class for embedding-related errors"""
    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.original_error = original_error

class LLMClient:
    """LLM client utility class"""

    def __init__(self):
        self.config = Config.from_env()
        self.user_llm_config_service = UserLLMConfigService()
        self.embedding_max_text_length = int(self.config.get('EMBEDDING_MAX_TEXT_LENGTH', 8000))
        # self.user_llm_config = self.user_llm_config_service.get_available_llm()

        # self.chat_api_key = self.user_llm_config.chat_api_key
        # self.chat_base_url = self.user_llm_config.chat_endpoint
        # self.chat_model = self.user_llm_config.chat_model_name
        # self.embedding_api_key = self.user_llm_config.embedding_api_key
        # self.embedding_base_url = self.user_llm_config.embedding_endpoint
        # self.embedding_model = self.user_llm_config.embedding_model_name


    def get_embedding(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Calculate text embedding

        Args:
            texts (str or list): Input text or list of texts

        Returns:
            numpy.ndarray: Embedding vector of the text
        """
        # Ensure texts is in list format
        if isinstance(texts, str):
            texts = [texts]

        # Split long texts into chunks using configured max length
        chunked_texts = []
        text_chunk_counts = []  # Keep track of how many chunks each text was split into
        
        for text in texts:
            if len(text) > self.embedding_max_text_length:
                # Split into chunks of embedding_max_text_length
                chunks = [text[i:i + self.embedding_max_text_length] 
                         for i in range(0, len(text), self.embedding_max_text_length)]
                chunked_texts.extend(chunks)
                text_chunk_counts.append(len(chunks))
            else:
                chunked_texts.append(text)
                text_chunk_counts.append(1)

        user_llm_config = self.user_llm_config_service.get_available_llm()
        if not user_llm_config:
            raise EmbeddingError("No LLM configuration found")
        
        try:
            # Send request to embedding endpoint
            embeddings_array = classification.strategy_classification(user_llm_config, chunked_texts)

            # If we split any texts, we need to merge their embeddings back
            if sum(text_chunk_counts) > len(texts):
                final_embeddings = []
                start_idx = 0
                for chunk_count in text_chunk_counts:
                    if chunk_count > 1:
                        # Average embeddings for split text
                        chunk_embeddings = embeddings_array[start_idx:start_idx + chunk_count]
                        avg_embedding = np.mean(chunk_embeddings, axis=0)
                        final_embeddings.append(avg_embedding)
                    else:
                        final_embeddings.append(embeddings_array[start_idx])
                    start_idx += chunk_count
                return np.array(final_embeddings)
            
            return embeddings_array

        except requests.exceptions.RequestException as e:
            # Handle request errors
            error_msg = f"Request error getting embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            error_msg = f"Invalid JSON response from embedding API: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)
        except (KeyError, IndexError, ValueError) as e:
            # Handle response structure errors
            error_msg = f"Invalid response structure from embedding API: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, e)
        except Exception as e:
            # Fallback for any other errors
            error_msg = f"Unexpected error getting embeddings: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingError(error_msg, e)

    @property
    def chat_credentials(self):
        """Get LLM authentication information"""
        return {"api_key": self.chat_api_key, "base_url": self.chat_base_url}
