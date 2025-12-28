"""
Advanced prompt strategies for multi-phase chat processing
"""
import logging
from typing import Optional

from lpm_kernel.api.domains.kernel2.dto.chat_dto import ChatRequest
from lpm_kernel.api.domains.kernel2.services.prompt_builder import SystemPromptStrategy
from lpm_kernel.api.domains.kernel2.services.knowledge_service import (
    default_retriever,
    default_l1_retriever,
)

logger = logging.getLogger(__name__)

class RequirementEnhancementStrategy(SystemPromptStrategy):
    """Strategy for enhancing requirements with context"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest) -> str:
        prompt = """
        You are a requirement analyst. Your task is to enhance and complete the given rough requirement.
        Consider the following:
        1. Clarify any ambiguous points
        2. Add necessary technical details
        3. Ensure the requirement is specific and actionable
        4. Incorporate the provided context and knowledge
        """
        
        # Add knowledge retrieval results if enabled
        knowledge_sections = []
        
        if request.enable_l0_retrieval:
            l0_knowledge = default_retriever.retrieve(request.message)
            if l0_knowledge:
                knowledge_sections.append(f"Reference knowledge:\n{l0_knowledge}")
                
        if request.enable_l1_retrieval:
            l1_knowledge = default_l1_retriever.retrieve(request.message)
            if l1_knowledge:
                knowledge_sections.append(f"Reference shades:\n{l1_knowledge}")
                
        if knowledge_sections:
            prompt += "\n\nKnowledge context:\n" + "\n\n".join(knowledge_sections)
            
        base_prompt = self.base_strategy.build_prompt(request)
        if base_prompt:
            prompt = f"{base_prompt}\n\n{prompt}"
            
        logger.info(f"RequirementEnhancementStrategy prompt: {prompt}")
        return prompt


class ExpertSolutionStrategy(SystemPromptStrategy):
    """Strategy for generating expert solutions"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest) -> str:
        prompt = """
        You are an expert system designed to generate solutions based on specific requirements.
        Generate a detailed solution that meets all aspects of the requirement.
        Be specific and include implementation details where necessary.
        """
        
        base_prompt = self.base_strategy.build_prompt(request)
        if base_prompt:
            prompt = f"{base_prompt}\n\n{prompt}"
            
        logger.info(f"ExpertSolutionStrategy prompt: {prompt}")
        return prompt


class SolutionValidatorStrategy(SystemPromptStrategy):
    """Strategy for validating solutions"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest) -> str:
        prompt = """
        You are a solution validator. Your task is to validate if the given solution meets all requirements.
        You must return a JSON response in the following format:
        {
            "is_valid": boolean,
            "feedback": string  // Reason and improvement suggestions if invalid
        }
        """
        
        base_prompt = self.base_strategy.build_prompt(request)
        if base_prompt:
            prompt = f"{base_prompt}\n\n{prompt}"
            
        logger.info(f"SolutionValidatorStrategy prompt: {prompt}")
        return prompt


class SolutionFormatterStrategy(SystemPromptStrategy):
    """Strategy for formatting solutions"""
    def __init__(self, base_strategy: SystemPromptStrategy):
        self.base_strategy = base_strategy

    def build_prompt(self, request: ChatRequest) -> str:
        prompt = """
        You are a solution formatter. Your task is to format the given solution to be clear and well-structured.
        Improve readability while maintaining all technical details.
        """
        
        base_prompt = self.base_strategy.build_prompt(request)
        if base_prompt:
            prompt = f"{base_prompt}\n\n{prompt}"
            
        logger.info(f"SolutionFormatterStrategy prompt: {prompt}")
        return prompt
