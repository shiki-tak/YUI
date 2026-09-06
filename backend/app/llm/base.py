"""LLM 接続の共通インターフェース。

設計書「4. Pythonバックエンドの責任分担」の LLM接続モジュール。
エージェントはこの窓口だけを使い、Ollama・Claude・将来の MLX 推論を
呼び分けても呼び出し側を変えずに済むようにする。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    model_digest: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # 思考出力を分けて返すモデル向け。返答本文には含めない。
    thinking: str | None = None


class LLMError(RuntimeError):
    """LLM 呼び出しの失敗。呼び出し側はこれを捕まえて縮退する。"""


class LLMClient(ABC):
    provider: str

    @abstractmethod
    async def chat(
        self, messages: list[ChatMessage], *, options: dict[str, Any] | None = None
    ) -> LLMResponse: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """接続確認。モデルが使える状態かを返す。"""

    async def aclose(self) -> None:
        return None
