"""LLM 接続モジュール。"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse
from app.llm.ollama_client import OllamaClient

__all__ = ["ChatMessage", "LLMClient", "LLMError", "LLMResponse", "get_llm_client"]


@lru_cache
def get_llm_client() -> LLMClient:
    """通常の推論に使うクライアント。

    フェーズ3でクラウド、フェーズ7で MLX を足すときは、ここで
    用途に応じた LLMClient を返すようにする。
    """
    settings = get_settings()
    return OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
        timeout=settings.llm_timeout_seconds,
    )
