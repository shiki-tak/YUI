"""テスト用の土台。

LLM は共通インターフェース（LLMClient）越しに差し替える。生成内容に依存せず、
人格・記憶・訂正・実行記録の振る舞いを確認するため。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent import ConversationAgent, get_agent
from app.config import get_settings
from app.db import get_session
from app.llm import get_llm_client
from app.llm.base import ChatMessage, LLMClient, LLMResponse
from app.main import app
from app.models import Base
from app.persona import BASE_PERSONA


class FakeLLM(LLMClient):
    """呼び出し内容を記録し、決めた文字列を返すクライアント。"""

    provider = "fake"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []
        self.scripted: list[str] = []
        self.default = "はい、覚えていますよ。"

    def push(self, text: str) -> None:
        self.scripted.append(text)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def chat(
        self, messages: list[ChatMessage], *, options: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        text = self.scripted.pop(0) if self.scripted else self.default
        return LLMResponse(
            text=text,
            provider=self.provider,
            model="fake-model",
            model_digest="sha256:test",
            options=options or {},
            latency_ms=1,
            prompt_tokens=10,
            completion_tokens=5,
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.provider}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest_asyncio.fixture
async def client(tmp_path, fake_llm: FakeLLM) -> AsyncIterator[AsyncClient]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    agent = ConversationAgent(llm=fake_llm, persona=BASE_PERSONA, settings=get_settings())
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()
    await engine.dispose()
