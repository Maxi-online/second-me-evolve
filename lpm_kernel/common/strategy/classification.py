from lpm_kernel.api.dto.user_llm_config_dto import (
    UserLLMConfigDTO,
)
from typing import Optional
import lpm_kernel.common.strategy.strategy_openai as openai
import lpm_kernel.common.strategy.strategy_huggingface as huggingface

def strategy_classification(user_llm_config: Optional[UserLLMConfigDTO], chunked_texts):
    if "sentence-transformers" in user_llm_config.embedding_endpoint:
        # Using Hugging Face strategy to generate embedding vectors
        return huggingface.huggingface_strategy(user_llm_config, chunked_texts)
    else:
        # Using openai strategy to generate embedding vectors
        return openai.openai_strategy(user_llm_config, chunked_texts)
