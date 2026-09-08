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

from app.agent import ConversationAgent, get_agent, reflection_job
from app.agent.goal_reflection import _INSTRUCTION as GOAL_INSTRUCTION
from app.agent.reflection import _PICKUP_INSTRUCTION as PICKUP_INSTRUCTION
from app.agent.state_reflection import _INSTRUCTION as STATE_INSTRUCTION
from app.config import get_settings
from app.db import get_session, get_session_factory
from app.llm import get_llm_client
from app.llm.base import ChatMessage, LLMClient, LLMResponse
from app.main import app
from app.models import Base
from app.persona import load_persona
from app.voice import get_speech_client
from app.voice.base import SpeechClient, SpeechError, SpeechResult


class FakeLLM(LLMClient):
    """呼び出し内容を記録し、決めた文字列を返すクライアント。"""

    provider = "fake"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []
        self.scripted: list[str] = []
        self.default = "はい、覚えていますよ。"
        # 関心・関係性の抽出は別の呼び出しなので、返す内容も別に持つ。
        # 既定は「更新なし」。記憶の抽出を確かめるテストが、状態の抽出まで
        # 用意しなくても済むようにする。
        self.state_scripted: list[str] = []
        self.state_default = "[]"
        # 目標の抽出も別の呼び出し（振り返りの4回目）。既定は「目標なし」。
        # 記憶や関心を確かめるテストが、目標まで用意しなくても済むようにする。
        self.goal_scripted: list[str] = []
        self.goal_default = "[]"
        # 記憶の抽出は2段階（拾う → 選ぶ）。1段階目は既定で1件拾ったことに
        # して、テストは「選ぶ」側の出力だけを書けばよいようにする。
        self.pickup_scripted: list[str] = []
        self.pickup_default = '[{"content": "会話に出てきた内容", "source_message_id": null}]' 
        # 生成中の状態を再現するための門。gate を待たせると応答待ちになる。
        self.entered = asyncio.Event()
        self.gate: asyncio.Event | None = None
        # 特定の呼び出しだけを止めたいとき、その指示文を入れる。振り返りは
        # 3回呼ぶため、何回目で止めるかを選べないと、止めたい経路を測れない。
        self.gate_on: str | None = None
        # 実際に門で止まったことを知らせる。entered は呼び出しごとに立つため、
        # 何回目で止まったかを待ち分けられない。
        self.held = asyncio.Event()

    def push(self, text: str) -> None:
        self.scripted.append(text)

    def push_goal(self, text: str) -> None:
        """目標の抽出が返す内容。"""
        self.goal_scripted.append(text)

    def push_state(self, text: str) -> None:
        """関心・関係性の抽出が返す内容。"""
        self.state_scripted.append(text)

    def push_pickup(self, text: str) -> None:
        """記憶の抽出の1段階目（拾う）が返す内容。"""
        self.pickup_scripted.append(text)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0].content

    async def chat(
        self, messages: list[ChatMessage], *, options: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        self.entered.set()
        system = messages[0].content if messages else ""
        if self.gate is not None and (self.gate_on is None or self.gate_on == system):
            self.held.set()
            await self.gate.wait()
        if messages and messages[0].content == GOAL_INSTRUCTION:
            text = self.goal_scripted.pop(0) if self.goal_scripted else self.goal_default
        elif messages and messages[0].content == STATE_INSTRUCTION:
            text = self.state_scripted.pop(0) if self.state_scripted else self.state_default
        elif messages and messages[0].content == PICKUP_INSTRUCTION:
            text = self.pickup_scripted.pop(0) if self.pickup_scripted else self.pickup_default
        else:
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
        # 合成に失敗する状況を作る。会話が続けられることを確かめるため。
        self.fail = False

    async def synthesize(self, text: str, *, speaker_id: int | None = None) -> SpeechResult:
        self.calls.append((text, speaker_id))
        if self.fail:
            raise SpeechError("音声合成に失敗しました（テスト）。")
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

    agent = ConversationAgent(llm=fake_llm, persona=load_persona(), settings=get_settings())
    app.dependency_overrides[get_session] = override_session
    # 振り返りのジョブはリクエストより長く生きるので、独自に接続を作る。
    # 一時DBへ向けないと、テストが通常利用のDBを書き換える。
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_speech_client] = lambda: fake_speech

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()


async def reflection_progress(client: AsyncClient, conversation_id: int) -> dict:
    """振り返りの進み具合。"""
    response = await client.get(f"/api/conversations/{conversation_id}/reflection")
    return response.json()


async def end_and_wait(client: AsyncClient, conversation_id: int):
    """会話を終了し、振り返りのジョブが終わるまで待つ。

    終了は待たずに返るようになった（フェーズ4 PR5）。テストは結果を見たいので、
    ここで待ち合わせる。**同期に戻しているのではなく、ジョブの完了を待つだけ**で、
    通る経路は実際の運用と同じである。
    """
    response = await client.post(f"/api/conversations/{conversation_id}/end")
    await reflection_job.wait(conversation_id)
    return response


async def end_and_candidates(client: AsyncClient, conversation_id: int) -> list[dict]:
    """終了して振り返り、**成功したことを確かめてから**候補を返す。

    候補一覧を読むだけだと、失敗して0件だった場合と、残すものが無くて0件
    だった場合を区別できない（第1回レビューの指摘）。
    """
    response = await end_and_wait(client, conversation_id)
    assert response.status_code == 202, response.text
    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "completed", progress
    return (await client.get(f"/api/conversations/{conversation_id}/candidates")).json()


async def end_and_expect_failure(client: AsyncClient, conversation_id: int) -> dict:
    """終了して振り返り、ジョブが失敗したことを確かめて理由を返す。

    終了そのものは受け付ける（202）。抽出の失敗はジョブの中で起きるので、
    HTTP の応答ではなく進行状態で見る（フェーズ4 PR5）。
    """
    response = await end_and_wait(client, conversation_id)
    assert response.status_code == 202, response.text
    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed", progress
    return progress
