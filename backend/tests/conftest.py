"""テスト用の土台。

LLM は共通インターフェース（LLMClient）越しに差し替える。生成内容に依存せず、
人格・記憶・訂正・実行記録の振る舞いを確認するため。
"""

from __future__ import annotations

import asyncio
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
from app.voice import get_speech_client
from app.voice.base import SpeechClient, SpeechResult


class FakeLLM(LLMClient):
    """呼び出し内容を記録し、決めた文字列を返すクライアント。"""

    provider = "fake"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []
        self.scripted: list[str] = []
        self.default = "はい、覚えていますよ。"
        # 生成中の状態を再現するための門。gate を待たせると応答待ちになる。
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None

    def push(self, text: str) -> None:
        self.scripted.append(text)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def chat(
        self, messages: list[ChatMessage], *, options: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
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


class FakeSpeech(SpeechClient):
    """音声合成の差し替え。VOICEVOX を起動していなくても検証できるようにする。"""

    provider = "fake-voice"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.ok = True

    async def synthesize(self, text: str, *, speaker_id: int | None = None) -> SpeechResult:
        self.calls.append((text, speaker_id))
        return SpeechResult(
            audio=b"RIFF\x00\x00\x00\x00WAVE",
            media_type="audio/wav",
            provider=self.provider,
            speaker_id=speaker_id if speaker_id is not None else 3,
            text=text,
            engine_version="test",
            query_ms=1,
            synthesis_ms=2,
            audio_ms=500,
        )

    async def health(self) -> dict[str, Any]:
        if not self.ok:
            return {"ok": False, "provider": self.provider, "error": "接続できません"}
        return {"ok": True, "provider": self.provider, "engine_version": "test", "speaker": 3}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_speech() -> FakeSpeech:
    return FakeSpeech()


@pytest_asyncio.fixture
async def session_factory(tmp_path) -> AsyncIterator[async_sessionmaker]:
    """テスト用の一時DB。通常利用の会話・記憶は変更しない。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker, fake_llm: FakeLLM, fake_speech: FakeSpeech
) -> AsyncIterator[AsyncClient]:
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
    app.dependency_overrides[get_speech_client] = lambda: fake_speech

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()
