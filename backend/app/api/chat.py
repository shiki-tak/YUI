"""会話 API。文字入力を受け取り、返答を返す。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import ConversationAgent, get_agent
from app.agent.memory_store import get_or_create_speaker
from app.db import get_session
from app.llm.base import LLMError
from app.models import Conversation, ConversationMode
from app.schemas import ChatRequest, ChatResponse, MemoryOut, RetrievedMemoryOut

router = APIRouter(tags=["chat"])


async def _resolve_conversation(
    session: AsyncSession, conversation_id: int | None
) -> Conversation:
    if conversation_id is not None:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会話が見つかりません。")
        if conversation.ended_at is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "この会話は終了しています。新しい会話を始めてください。"
            )
        return conversation
    conversation = Conversation(mode=ConversationMode.LOCAL.value)
    session.add(conversation)
    await session.flush()
    return conversation


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_session),
    agent: ConversationAgent = Depends(get_agent),
) -> ChatResponse:
    conversation = await _resolve_conversation(session, payload.conversation_id)
    speaker = await get_or_create_speaker(
        session,
        source=payload.speaker.source,
        external_id=payload.speaker.external_id,
        display_name=payload.speaker.display_name,
    )

    try:
        result = await agent.respond(
            session,
            conversation=conversation,
            speaker=speaker,
            text=payload.text,
        )
    except LLMError as exc:
        # 相手の発言は生成前にコミット済みなので、履歴に残ったまま失敗を伝える。
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if conversation.title is None:
        conversation.title = payload.text[:40]

    return ChatResponse(
        conversation_id=conversation.id,
        user_message=result.user_message,
        reply=result.reply_message,
        run=result.run,
        used_memories=[
            RetrievedMemoryOut(
                memory=MemoryOut.model_validate(item.memory),
                score=item.score,
                reason=item.reason,
            )
            for item in result.memories
        ],
    )
