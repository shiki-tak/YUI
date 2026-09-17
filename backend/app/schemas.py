"""API の入出力定義（Pydantic）。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from app.models import Certainty, MemoryKind, Provenance, Visibility


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
    # 中断（aborted）のときだけ入る近似値。completed では常に NULL
    # （「全文届いた」を意味する。ISSUE-051）。
    delivered_char_count: int | None
    created_at: UtcDatetime


class MemoryOut(ORMModel):
    id: int
    kind: str
    content: str
    # どうやって知ったか。伝聞を本人の発言と区別する。
    provenance: str
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
    referenced_state_ids: list[int] | None
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
    # 比べられるようにする（人格の版管理。v0.1）。用意されている版だけを許す。
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


class ConversationStateOut(ORMModel):
    """v0.2 の会話状態（今の用件・未回答の質問・延期・終了・食い違い）。"""

    id: int
    conversation_id: int
    kind: str
    content: str
    speaker_id: int | None
    target_speaker_id: int | None
    source_message_id: int
    ref_kind: str | None
    ref_id: int | None
    status: str
    withdraw_reason: str | None
    asked_message_id: int | None
    responded_message_id: int | None
    judged_message_id: int | None
    followup_needed: bool
    resolved_message_id: int | None
    detected_by: str
    decided_by: str | None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ConversationStateDecision(BaseModel):
    """開発者による解決・取消。`decided_by = operator` になる。

    `detected_by`（規則／解釈のどちらが作ったか）は変えない。誤検出の是正と、
    解釈を有効にしていない構成での手動解決に使う（計画 §6）。
    """

    decision: Literal["resolved", "withdrawn"]
    withdraw_reason: Literal["reopened", "cancelled", "misdetected", "superseded"] | None = None

    @model_validator(mode="after")
    def _require_reason_for_withdrawal(self) -> ConversationStateDecision:
        if self.decision == "withdrawn" and self.withdraw_reason is None:
            raise ValueError("withdrawn には withdraw_reason が要ります。")
        if self.decision == "resolved" and self.withdraw_reason is not None:
            raise ValueError("resolved に withdraw_reason は指定できません。")
        return self


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    reply: MessageOut
    run: RunRecordOut
    used_memories: list[RetrievedMemoryOut]
    # 開いている会話状態（v0.2）。既存のクライアントを壊さないよう既定は空。
    # **画面が使う種類だけ**（`conversation_state.DISPLAYED_STATE_KINDS`）。
    # `presented`・`confirmed` は含めない——全件を毎ターン返すと、会話が
    # 長くなるほど応答が肥大する（レビューで実測）。全件が要る場合は
    # `GET /conversations/{id}/states` を使う。
    conversation_states: list[ConversationStateOut] = Field(default_factory=list)


class DeliveryUpdate(BaseModel):
    """再生の通知。generated は生成時の状態なので受け付けない。"""

    state: Literal["playing", "completed", "aborted"]
    # state="aborted" のときだけ使う。画面が再生位置（currentTime / duration）
    # から測った比率（ISSUE-051）。他の state では無視する。
    progress: float | None = Field(default=None, ge=0.0, le=1.0)


class MemoryCreate(BaseModel):
    """手動で追加する記憶。

    参照範囲は必ず指定させる。既定で「誰との会話でも参照してよい」にすると、
    ある相手との会話から作られた記憶が、別の相手へ黙って渡る（ISSUE-010）。
    限定しないこと自体は選べるが、選んだ結果としてそうなるようにする。
    """

    kind: MemoryKind
    content: str = Field(min_length=1, max_length=2000)
    subject_speaker_id: int | None = None
    provenance: Provenance = Provenance.UNKNOWN
    # この相手との会話に限定する。
    visible_to_speaker_id: int | None = None
    # 相手を限定しない。visible_to_speaker_id と同時には指定できない。
    visible_to_all: bool = False
    certainty: Certainty = Certainty.FACT
    visibility: Visibility = Visibility.PRIVATE
    keywords: str = ""
    occurred_at: datetime | None = None
    source_message_id: int | None = None
    source_conversation_id: int | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _require_explicit_scope(self) -> MemoryCreate:
        if self.visible_to_all and self.visible_to_speaker_id is not None:
            raise ValueError("visible_to_speaker_id と visible_to_all は同時に指定できません。")
        if not self.visible_to_all and self.visible_to_speaker_id is None:
            raise ValueError(
                "参照範囲を指定してください。"
                "特定の相手との会話に限る場合は visible_to_speaker_id、"
                "限定しない場合は visible_to_all=true。"
            )
        return self


class MemoryUpdate(BaseModel):
    """記憶の訂正。指定した項目だけを更新し、更新前を履歴に残す。"""

    # 参照範囲の振り分け。限定しない状態へ戻すには visible_to_all を使う。
    visible_to_speaker_id: int | None = None
    visible_to_all: bool | None = None
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
    provenance: str
    subject_speaker_id: int | None
    visible_to_speaker_id: int | None
    certainty: str
    visibility: str
    keywords: str
    occurred_at: UtcDatetime | None
    source_message_id: int | None
    # 内容が近い既存の記憶。二重に覚えないための手がかり。
    similar_memory_ids: list[int] | None
    status: str
    accepted_memory_id: int | None
    created_at: UtcDatetime


class CandidateDecision(BaseModel):
    """候補の採用・却下。採用時は内容を直してから保存できる。

    誰についての記憶か（subject_speaker_id）と、どうやって知ったか
    （provenance）も直せる。モデルの分類は確実ではないため、採用の時点で
    人が直せる経路を残す（v0.1）。
    """

    decision: Literal["accept", "reject"]
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    kind: MemoryKind | None = None
    # 誰についての記憶か。相手に紐づけない場合は subject_to_none=true。
    subject_speaker_id: int | None = None
    subject_to_none: bool = False
    provenance: Provenance | None = None
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


class CharacterStateOut(ORMModel):
    """変化する状態：YUI の関心と、相手との関係。"""

    id: int
    kind: str
    subject_speaker_id: int | None
    topic: str | None
    content: str
    basis_memory_ids: list[int] | None
    # 自動で並べた暫定の根拠かどうか。完全に特定した根拠と区別する。
    basis_is_provisional: bool
    needs_review: bool
    review_reason: str | None
    status: str
    visibility: str
    visible_to_speaker_id: int | None
    superseded_by_id: int | None
    source_conversation_id: int | None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class CharacterStateCreate(BaseModel):
    """状態の追加。既定は候補（pending）で、採用するまで会話に使わない。"""

    kind: Literal["interest", "relationship"]
    content: str = Field(min_length=1, max_length=2000)
    topic: str | None = Field(default=None, max_length=120)
    subject_speaker_id: int | None = None
    # 根拠にした記憶。ここに挙げた記憶が訂正・削除されると、再評価の印が付く。
    basis_memory_ids: list[int] = Field(default_factory=list)
    visibility: Visibility = Visibility.PRIVATE
    # 参照範囲。記憶と同じく、既定で全員に渡さない（ISSUE-010 と同じ考え方）。
    visible_to_speaker_id: int | None = None
    visible_to_all: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _check_scope(self) -> CharacterStateCreate:
        if self.visible_to_all and self.visible_to_speaker_id is not None:
            raise ValueError("visible_to_speaker_id と visible_to_all は同時に指定できません。")
        if not self.visible_to_all and self.visible_to_speaker_id is None:
            raise ValueError(
                "参照範囲を指定してください。"
                "特定の相手との会話に限る場合は visible_to_speaker_id、"
                "限定しない場合は visible_to_all=true。"
            )
        return self

    @model_validator(mode="after")
    def _check_target(self) -> CharacterStateCreate:
        if self.kind == "relationship" and self.subject_speaker_id is None:
            raise ValueError(
                "関係性には subject_speaker_id が要ります。誰との関係かを決めてください。"
            )
        if self.kind == "interest" and self.subject_speaker_id is not None:
            raise ValueError(
                "関心は YUI 自身のものです。subject_speaker_id は指定しないでください。"
            )
        return self


class CharacterStateDecision(BaseModel):
    """候補の採用・却下。採用時に内容を直せる。"""

    decision: Literal["accept", "reject"]
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    reason: str | None = None


class CharacterStateUpdate(BaseModel):
    """採用済みの状態を直す。再評価の印を下ろすときにも使う。"""

    content: str | None = Field(default=None, min_length=1, max_length=2000)
    status: Literal["active", "withdrawn"] | None = None
    # 確認したので印を下ろす。内容を直したかどうかとは別に指定する。
    reviewed: bool = False
    reason: str | None = None


class CharacterStateRevisionOut(ORMModel):
    id: int
    state_id: int
    action: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    reason: str | None
    created_at: UtcDatetime


class MemorySearchResult(BaseModel):
    query: str
    results: list[RetrievedMemoryOut]
