"""API の入出力定義（Pydantic）。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.models import Certainty, MemoryKind, Visibility


def _as_utc(value: datetime) -> datetime:
    """timezone を持たない日時に UTC を補う。

    保存は UTC（models.utcnow）で行っているが、SQLite は timezone を落として
    返す。補わずに返すと、受け取った側が自分の地域の時刻として解釈し、
    表示が実際の時刻とずれる。
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


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
    delivery_started_at: UtcDatetime | None
    delivery_finished_at: UtcDatetime | None
    created_at: UtcDatetime


class MemoryOut(ORMModel):
    id: int
    kind: str
    content: str
    subject_speaker_id: int | None
    visible_to_speaker_id: int | None
    certainty: str
    visibility: str
    status: str
    keywords: str
    occurred_at: UtcDatetime | None
    source_message_id: int | None
    source_conversation_id: int | None
    superseded_by_id: int | None
    created_at: UtcDatetime
    updated_at: UtcDatetime


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
    persona_version: str | None
    options: dict[str, Any] | None
    referenced_memory_ids: list[int] | None
    retrieval_ms: int | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: UtcDatetime


class RunRecordDetail(RunRecordOut):
    """開発画面で根拠を確認するための詳細。"""

    system_prompt: str | None


class SpeechRunOut(ORMModel):
    """音声合成の実行記録。待ち時間の内訳を見るために使う。"""

    id: int
    message_id: int
    provider: str
    speaker_id: int
    engine_version: str | None
    query_ms: int | None
    synthesis_ms: int | None
    audio_ms: int | None
    byte_size: int | None
    created_at: UtcDatetime


class ConversationOut(ORMModel):
    id: int
    mode: str
    title: str | None
    started_at: UtcDatetime
    ended_at: UtcDatetime | None
    reflection_started_at: UtcDatetime | None
    reflection_completed_at: UtcDatetime | None


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
    # 固定人格の版。省略すると設定の版を使う。同じ会話へ別の版で答えさせて
    # 比べられるようにする（設計書フェーズ3の3A）。用意されている版だけを許す。
    persona_version: str | None = Field(default=None, max_length=64)


class PersonaOut(BaseModel):
    """固定人格の内容。開発画面で、どの版で話しているかを確認する。"""

    name: str
    version: str
    traits: list[str]
    speech: list[str]
    rules: list[str]
    identity: list[str]
    curiosity: list[str]
    boundaries: list[str]
    prompt: str
    available_versions: list[str]


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    reply: MessageOut
    run: RunRecordOut
    used_memories: list[RetrievedMemoryOut]


class DeliveryUpdate(BaseModel):
    """再生の通知。generated は生成時の状態なので受け付けない。"""

    state: Literal["playing", "completed", "aborted"]


class MemoryCreate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=2000)
    subject_speaker_id: int | None = None
    # 非公開の記憶を、この相手との会話に限定する。None は限定しない。
    visible_to_speaker_id: int | None = None
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
    created_at: UtcDatetime


class MemoryCandidateOut(ORMModel):
    id: int
    conversation_id: int
    kind: str
    content: str
    subject_speaker_id: int | None
    visible_to_speaker_id: int | None
    certainty: str
    visibility: str
    keywords: str
    source_message_id: int | None
    status: str
    accepted_memory_id: int | None
    created_at: UtcDatetime


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
    created_at: UtcDatetime


class MemorySearchResult(BaseModel):
    query: str
    results: list[RetrievedMemoryOut]
