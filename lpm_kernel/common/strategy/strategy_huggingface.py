from sentence_transformers import SentenceTransformer
from lpm_kernel.api.dto.user_llm_config_dto import (
    UserLLMConfigDTO,
)
from typing import Optional
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def huggingface_strategy(user_llm_config: Optional[UserLLMConfigDTO], chunked_texts):
    parts = user_llm_config.embedding_endpoint.strip("/").split("/")
    if len(parts) >= 2:
        model_name = "/".join(parts[-2:])
    else:
        raise ValueError("Endpoint error")

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunked_texts)
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")