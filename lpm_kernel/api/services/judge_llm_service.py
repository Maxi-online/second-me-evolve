"""
Judge LLM service (automatic reinforcement judge).

Uses gpt-5-nano to:
- choose best answer among candidates (A/B)
- infer user sentiment/satisfaction from follow-up message
"""

from __future__ import annotations

import json
import logging
import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
import httpx

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService

logger = logging.getLogger(__name__)


class JudgeLLMService:
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None
        self._model_name: str = "gpt-5-nano"
        self._user_llm_config_service = UserLLMConfigService()

    def _ensure_client(self) -> Tuple[AsyncOpenAI, str]:
        cfg = self._user_llm_config_service.get_available_llm()
        if cfg is None:
            cfg = None

        # Prefer judge_* fields; fallback to thinking_* (already used for “serious” reasoning models)
        base_url = (getattr(cfg, "judge_endpoint", None) or getattr(cfg, "thinking_endpoint", None)) if cfg else None
        api_key = (getattr(cfg, "judge_api_key", None) or getattr(cfg, "thinking_api_key", None)) if cfg else None
        model = (getattr(cfg, "judge_model_name", None) or "gpt-5-nano") if cfg else "gpt-5-nano"

        # Env fallback (preferred for secrets)
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not base_url or not api_key:
            raise ValueError("Judge LLM is not configured: set OPENAI_API_KEY and OPENAI_BASE_URL (or judge_* in DB)")

        # Enforce gpt-5-nano as requested
        if model != "gpt-5-nano":
            model = "gpt-5-nano"

        # Proxy (HTTP only)
        proxy_url = (
            os.getenv("OPENAI_HTTP_PROXY")
            or os.getenv("HTTPS_PROXY")
            or os.getenv("HTTP_PROXY")
        )
        http_client = None
        if proxy_url:
            # Enforce HTTP proxy only (no socks)
            if proxy_url.startswith("socks"):
                raise ValueError("SOCKS proxy запрещён в этом проекте: используйте HTTP proxy.")
            http_client = httpx.AsyncClient(proxies=proxy_url, timeout=60.0)

        if self._client is None:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
            self._model_name = model
        return self._client, self._model_name

    async def choose_best_of_two_async(
        self,
        prompt_messages: List[Dict[str, str]],
        answer_a: str,
        answer_b: str,
    ) -> Dict[str, Any]:
        """
        Returns JSON:
        {
          "winner": "A"|"B",
          "score_a": -1..1,
          "score_b": -1..1,
          "user_mood_prediction": "positive"|"neutral"|"negative",
          "brief_reason": "..."
        }
        """
        client, model = self._ensure_client()

        system = (
            "You are an expert judge for conversational assistants.\n"
            "Goal: pick the answer that will maximize user satisfaction and emotional comfort.\n"
            "Judge based on clarity, helpfulness, correctness, politeness, and alignment with user's intent.\n"
            "You MUST output a strict JSON object with keys: winner, score_a, score_b, user_mood_prediction, brief_reason.\n"
            "Scores are floats in range [-1, 1]. winner must be exactly 'A' or 'B'.\n"
        )

        user = {
            "prompt_messages": prompt_messages,
            "answer_a": answer_a,
            "answer_b": answer_b,
        }

        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )

        content = res.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except Exception:
            logger.warning(f"Judge returned non-JSON: {content}")
            return {"winner": "A", "score_a": 0.0, "score_b": 0.0, "user_mood_prediction": "neutral", "brief_reason": "fallback"}

    async def infer_user_mood_reward_async(
        self,
        user_message: str,
        prev_assistant_message: str,
    ) -> Dict[str, Any]:
        """
        Returns JSON:
        {
          "mood": "positive"|"neutral"|"negative",
          "satisfaction": "satisfied"|"unsatisfied"|"unclear",
          "reward": -1..1,
          "brief_reason": "..."
        }
        """
        client, model = self._ensure_client()

        system = (
            "You are an expert judge.\n"
            "Infer user's mood and satisfaction given the user's last message and the previous assistant message.\n"
            "Output strict JSON object with keys: mood, satisfaction, reward, brief_reason.\n"
            "reward is float in [-1, 1].\n"
        )

        payload = {
            "prev_assistant_message": prev_assistant_message,
            "user_message": user_message,
        }

        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        content = res.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except Exception:
            logger.warning(f"Judge returned non-JSON for mood: {content}")
            return {"mood": "neutral", "satisfaction": "unclear", "reward": 0.0, "brief_reason": "fallback"}

    # Sync wrappers (Flask routes are sync)
    def choose_best_of_two(self, prompt_messages: List[Dict[str, str]], answer_a: str, answer_b: str) -> Dict[str, Any]:
        return asyncio.run(self.choose_best_of_two_async(prompt_messages, answer_a, answer_b))

    def infer_user_mood_reward(self, user_message: str, prev_assistant_message: str) -> Dict[str, Any]:
        return asyncio.run(self.infer_user_mood_reward_async(user_message, prev_assistant_message))


judge_llm_service = JudgeLLMService()

