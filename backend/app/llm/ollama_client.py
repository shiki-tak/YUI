"""Ollama によるローカル推論（通常の返答生成）。"""

from __future__ import annotations

import re
import time
from typing import Any

from ollama import AsyncClient, ResponseError

from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse

# qwen3 など思考を出すモデルが本文に混ぜてくる場合に取り除く。
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str | None]:
    blocks = _THINK_BLOCK.findall(text)
    if not blocks:
        return text.strip(), None
    body = _THINK_BLOCK.sub("", text).strip()
    thinking = "\n".join(b[len("<think>") : -len("</think>")].strip() for b in blocks)
    return body, thinking or None


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        *,
        host: str,
        model: str,
        temperature: float = 0.8,
        num_ctx: int = 8192,
        timeout: float = 120.0,
        think: bool | None = False,
    ) -> None:
        self.model = model
        self._think = think
        self._client = AsyncClient(host=host, timeout=timeout)
        self._default_options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
        }
        self._digest_cache: dict[str, str] = {}

    async def _model_digest(self) -> str | None:
        """モデルの版。並行改善（学習）で採用モデルを追えるように記録する。"""
        if self.model in self._digest_cache:
            return self._digest_cache[self.model]
        try:
            listing = await self._client.list()
        except Exception:
            return None
        wanted = {self.model, f"{self.model}:latest"}
        for entry in listing.models:
            name = getattr(entry, "model", None)
            digest = getattr(entry, "digest", None)
            if name in wanted and digest:
                self._digest_cache[self.model] = digest
                return digest
        return None

    async def chat(
        self, messages: list[ChatMessage], *, options: dict[str, Any] | None = None
    ) -> LLMResponse:
        merged = {**self._default_options, **(options or {})}
        payload = [{"role": m.role, "content": m.content} for m in messages]

        started = time.perf_counter()
        try:
            # think はオプションではなく最上位の引数。None はモデルの既定に任せる
            # 意味で、省略した場合と同じ扱いになる。
            raw = await self._client.chat(
                model=self.model, messages=payload, options=merged, think=self._think
            )
        except ResponseError as exc:
            raise LLMError(f"Ollama がエラーを返しました（model={self.model}）: {exc}") from exc
        except Exception as exc:
            raise LLMError(
                f"Ollama に接続できませんでした（model={self.model}）: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        content = raw.message.content or ""
        body, inline_thinking = split_thinking(content)
        thinking = getattr(raw.message, "thinking", None) or inline_thinking
        if not body:
            raise LLMError("Ollama から空の返答が返りました。")

        return LLMResponse(
            text=body,
            provider=self.provider,
            model=self.model,
            model_digest=await self._model_digest(),
            # 実行記録に残す設定。think も再現に必要なので含める。
            options={**merged, "think": self._think},
            latency_ms=latency_ms,
            prompt_tokens=raw.prompt_eval_count,
            completion_tokens=raw.eval_count,
            thinking=thinking,
        )

    async def health(self) -> dict[str, Any]:
        try:
            listing = await self._client.list()
        except Exception as exc:
            return {"ok": False, "provider": self.provider, "error": str(exc)}
        available = [m.model for m in listing.models if m.model]
        base = self.model.split(":")[0]
        return {
            "ok": any(name.split(":")[0] == base for name in available),
            "provider": self.provider,
            "model": self.model,
            "available_models": available,
        }
