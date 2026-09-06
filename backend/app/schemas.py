"""API の入出力定義（Pydantic）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import Certainty, MemoryKind, Visibility


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SpeakerOut(ORMModel):
    id: int
    source: str
    external_id: str
    display_name: str


class MessageOut(ORMModel):
    id: int
    conversation_id: int
    speaker_kind: str
    speaker_id: int | None
    source: str
    content: str
    delivery_state: str
    created_at: datetime


class MemoryOut(ORMModel):
    id: int
    kind: str
    content: str
    subject_speaker_id: int | None
    certainty: str
    visibility: str
    status: str
    keywords: str
    occurred_at: datetime | None
    source_message_id: int | None
    source_conversation_id: int | None
    superseded_by_id: int | None
    created_at: datetime
    updated_at: datetime


class RetrievedMemoryOut(BaseModel):
    """返答に渡した記憶と、選ばれた理由。"""

    memory: MemoryOut
    score: float
    reason: str


class RunRecordOut(ORMModel):
    id: int
    message_id: int
    provider: str
    model: str
    model_digest: str | None
    options: dict[str, Any] | None
    referenced_memory_ids: list[int] | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime


class RunRecordDetail(RunRecordOut):
    """開発画面で根拠を確認するための詳細。"""

    system_prompt: str | None


class ConversationOut(ORMModel):
    id: int
    mode: str
    title: str | None
    started_at: datetime
    ended_at: datetime | None


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class SpeakerRef(BaseModel):
    """相手の指定。表示名ではなく、入力元と識別子で同定する。"""

    source: str = "local"
    external_id: str = "developer"
    display_name: str = "開発者"


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None
    speaker: SpeakerRef = SpeakerRef()


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    reply: MessageOut
    run: RunRecordOut
    used_memories: list[RetrievedMemoryOut]


class MemoryCreate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=2000)
    subject_speaker_id: int | None = None
    certainty: Certainty = Certainty.FACT
    visibility: Visibility = Visibility.PRIVATE
    keywords: str = ""
    occurred_at: datetime | None = None
    source_message_id: int | None = None
    source_conversation_id: int | None = None
    reason: str | None = None


class MemoryUpdate(BaseModel):
    """記憶の訂正。指定した項目だけを更新し、更新前を履歴に残す。"""

    content: str | None = Field(default=None, min_length=1, max_length=2000)
    kind: MemoryKind | None = None
    certainty: Certainty | None = None
    visibility: Visibility | None = None
    keywords: str | None = None
    occurred_at: datetime | None = None
    reason: str | None = None


class MemoryRevisionOut(ORMModel):
    id: int
    memory_id: int
    action: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    reason: str | None
    created_at: datetime


class MemoryCandidateOut(ORMModel):
    id: int
    conversation_id: int
    kind: str
    content: str
    subject_speaker_id: int | None
    certainty: str
    visibility: str
    keywords: str
    source_message_id: int | None
    status: str
    accepted_memory_id: int | None
    created_at: datetime


class CandidateDecision(BaseModel):
    """候補の採用・却下。採用時は内容を直してから保存できる。"""

    decision: Literal["accept", "reject"]
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    kind: MemoryKind | None = None
    certainty: Certainty | None = None
    visibility: Visibility | None = None
    keywords: str | None = None
    reason: str | None = None


class IdealResponseCreate(BaseModel):
    ideal_text: str = Field(min_length=1, max_length=4000)
    note: str | None = None


class IdealResponseOut(ORMModel):
    id: int
    message_id: int
    ideal_text: str
    note: str | None
    created_at: datetime


class MemorySearchResult(BaseModel):
    query: str
    results: list[RetrievedMemoryOut]
