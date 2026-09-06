"""会話進行。設計書「5. 1回の会話の流れ」のフェーズ1相当。

1. 発言を受け取り、会話履歴に保存する。
2. 相手・話題に合う長期記憶を検索する。
3. 人格・記憶・直近の会話を LLM へ渡す。
4. 返答を保存し、実行記録（モデル・設定・参照した記憶・応答時間）を残す。

検索・クラウド・出力検査はフェーズ3以降で 2 と 3 の間に入る。
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import prompt as prompt_builder
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
    ) -> ReplyResult:
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
        memories = await search_memories(
            session,
            query=text,
            speaker_id=speaker.id,
            mode=conversation.mode,
            limit=self._settings.memory_retrieval_limit,
        )

        messages, system_prompt = prompt_builder.build_messages(
            persona=self._persona,
            memories=memories,
            speaker=speaker,
            history=history,
            user_text=text,
        )

        # LLMError はここでは握らず、API 層で 503 として返す。
        response = await self._llm.chat(messages)

        now = utcnow()
        reply_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.CHARACTER.value,
            speaker_id=None,
            source=source,
            content=response.text,
            # 文字会話では生成した時点で相手に届く。フェーズ2で音声再生の
            # 開始・完了・中断に応じて状態を進める。
            delivery_state=DeliveryState.COMPLETED.value,
            delivery_started_at=now,
            delivery_finished_at=now,
        )
        session.add(reply_message)
        await session.flush()

        run = RunRecord(
            message_id=reply_message.id,
            provider=response.provider,
            model=response.model,
            model_digest=response.model_digest,
            options=response.options,
            referenced_memory_ids=[m.memory.id for m in memories],
            system_prompt=system_prompt,
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
