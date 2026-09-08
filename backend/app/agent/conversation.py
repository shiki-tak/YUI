"""会話進行。設計書「6. 会話・自発的行動の流れ」のうち、現在実装している範囲。

1. 発言を受け取り、会話履歴に保存する。
2. 相手・話題に合う長期記憶を検索する。
3. 人格・記憶・直近の会話を LLM へ渡す。
4. 返答を保存し、実行記録（モデル・設定・参照した記憶・応答時間）を残す。

検索・クラウド分析・出力検査はフェーズ5Aで 2 と 3 の間に入る。行動選択
（回答・確認質問・話題提案・調査・待機）はフェーズ4で 2 の後に入る。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent import prompt as prompt_builder
from app.agent.character_state import active_states
from app.agent.memory_store import RetrievedMemory, search_memories
from app.config import Settings
from app.llm.base import LLMClient
from app.models import (
    Conversation,
    DeliveryState,
    Message,
    RunRecord,
    SourceKind,
    Speaker,
    SpeakerKind,
    utcnow,
)
from app.persona import Persona


@dataclass
class ReplyResult:
    user_message: Message
    reply_message: Message
    run: RunRecord
    memories: list[RetrievedMemory]


class ConversationAgent:
    def __init__(self, *, llm: LLMClient, persona: Persona, settings: Settings) -> None:
        self._llm = llm
        self._persona = persona
        self._settings = settings

    async def _recent_history(
        self, session: AsyncSession, conversation_id: int, *, exclude_id: int
    ) -> list[Message]:
        stmt = (
            select(Message)
            # 誰の発言かをプロンプトに残すため、話者を一緒に読む。遅延読み込みは
            # 非同期セッションでは使えない。
            .options(selectinload(Message.speaker))
            .where(Message.conversation_id == conversation_id, Message.id != exclude_id)
            .order_by(Message.id.desc())
            .limit(self._settings.recent_message_limit)
        )
        return list(reversed(list((await session.execute(stmt)).scalars())))

    async def respond(
        self,
        session: AsyncSession,
        *,
        conversation: Conversation,
        speaker: Speaker,
        text: str,
        source: str = SourceKind.LOCAL_TEXT.value,
        persona: Persona | None = None,
        now: datetime | None = None,
    ) -> ReplyResult:
        # 人格は会話ごとに差し替えられる。3A で同じ入力を別の版へ通し、
        # 着眼点や口調の違いを比べるため。指定が無ければ設定の版を使う。
        persona = persona or self._persona
        # 現在時刻は差し替えられる。評価で期限の到来をまたぐために使う
        # （実際に待つ代わりに、渡す時刻を進める）。記憶検索の減衰と、
        # プロンプトへ書く現在時刻を同じ値で揃える。ばらばらに utcnow() を
        # 呼ぶと、進めた時刻と実時計が混ざる。
        reference_time = now or utcnow()
        user_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.USER.value,
            speaker_id=speaker.id,
            source=source,
            content=text,
            delivery_state=DeliveryState.COMPLETED.value,
        )
        session.add(user_message)
        await session.flush()

        history = await self._recent_history(
            session, conversation.id, exclude_id=user_message.id
        )
        # 記憶検索の時間は生成と分けて残す。どちらが待ち時間の大半かを
        # 見分けられるようにするため。
        retrieval_started = time.perf_counter()
        memories = await search_memories(
            session,
            query=text,
            speaker_id=speaker.id,
            mode=conversation.mode,
            limit=self._settings.memory_retrieval_limit,
            now=reference_time,
        )
        # 可変状態（関心・関係性）も、記憶と同じ区間で読む。固定人格とは
        # 分けて渡し、どれを使ったかを実行記録に残す。
        states = await active_states(session, speaker_id=speaker.id, mode=conversation.mode)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        messages, system_prompt = prompt_builder.build_messages(
            persona=persona,
            memories=memories,
            speaker=speaker,
            history=history,
            user_text=text,
            states=states,
            now=reference_time,
        )

        # 生成に入る前にトランザクションを閉じる。SQLite は書き込みロックを
        # 1 つしか持てないため、応答を待つ間ロックを保持すると、他の会話や
        # 記憶の訂正が「database is locked」で失敗する。
        # 相手の発言はこの時点で確定させ、生成に失敗しても履歴には残す。
        await session.commit()

        # LLMError はここでは握らず、API 層で 503 として返す。
        response = await self._llm.chat(messages)

        recorded_at = utcnow()
        # 読み上げる構成では、生成しただけの状態から始める。実際に鳴ったかは
        # 再生側の通知で進める。読み上げない構成では、画面に出た時点で届く。
        spoken = self._settings.speech_enabled
        reply_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.CHARACTER.value,
            speaker_id=None,
            source=source,
            content=response.text,
            delivery_state=(
                DeliveryState.GENERATED.value if spoken else DeliveryState.COMPLETED.value
            ),
            delivery_started_at=None if spoken else recorded_at,
            delivery_finished_at=None if spoken else recorded_at,
        )
        session.add(reply_message)
        await session.flush()

        run = RunRecord(
            message_id=reply_message.id,
            provider=response.provider,
            model=response.model,
            model_digest=response.model_digest,
            persona_version=persona.version,
            options=response.options,
            referenced_memory_ids=[m.memory.id for m in memories],
            referenced_state_ids=[state.id for state in states] or None,
            system_prompt=system_prompt,
            retrieval_ms=retrieval_ms,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        session.add(run)
        await session.flush()

        return ReplyResult(
            user_message=user_message,
            reply_message=reply_message,
            run=run,
            memories=memories,
        )
