"""SQLAlchemy モデル。

設計書「6. 保存するデータ」に対応する。特に次の区別を構造として持たせている。

- 会話履歴（messages）と長期記憶（memories）を分ける。
- 事実（fact）と推測（inference）を certainty で区別する。
- 記憶には根拠の会話（source_message_id）を残し、訂正可能にする。
- 生成しただけの文章と、実際に相手へ届いた文章を delivery_state で区別する。
- 人物は表示名ではなく、入力元＋その識別子（source, external_id）で同定する。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


# --- 語彙 -------------------------------------------------------------------


class SourceKind(StrEnum):
    """入力元。フェーズ2以降で local_voice・youtube が増える。"""

    LOCAL_TEXT = "local_text"
    LOCAL_VOICE = "local_voice"
    YOUTUBE = "youtube"


class ConversationMode(StrEnum):
    """会話モード。stream では公開可能な記憶だけを参照する（フェーズ4）。"""

    LOCAL = "local"
    STREAM = "stream"


class SpeakerKind(StrEnum):
    USER = "user"
    CHARACTER = "character"


class DeliveryState(StrEnum):
    """発話の状態。フェーズ1の文字会話では generated → completed のみ。

    フェーズ2で音声再生の playing・aborted を使い、生成しただけの文章と
    実際に話し終えた内容を混同しないようにする。
    """

    GENERATED = "generated"
    PLAYING = "playing"
    COMPLETED = "completed"
    ABORTED = "aborted"


class MemoryKind(StrEnum):
    EXPERIENCE = "experience"  # 出来事・経験
    ABOUT_PERSON = "about_person"  # 相手について知ったこと
    PROMISE = "promise"  # 約束
    IMPRESSION = "impression"  # キャラクターの受け止め方
    FACT = "fact"  # 外部情報として確認した事実


class Certainty(StrEnum):
    FACT = "fact"
    INFERENCE = "inference"


class Visibility(StrEnum):
    PRIVATE = "private"  # ローカル会話限定
    PUBLIC = "public"  # 配信で参照してよい


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CORRECTED = "corrected"  # 訂正され、後続の記憶に置き換わった
    DELETED = "deleted"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# --- テーブル ---------------------------------------------------------------


class Speaker(Base):
    """会話相手。表示名は変わりうるので、同定は (source, external_id) で行う。"""

    __tablename__ = "speakers"
    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_speaker_identity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    """1 まとまりの会話。終了時に長期記憶の候補を抽出する単位。"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(16), default=ConversationMode.LOCAL, nullable=False)
    title: Mapped[str | None] = mapped_column(String(200))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        order_by="Message.id",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """会話履歴。長期記憶とは別物で、そのまま記憶に昇格させない。"""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation", "conversation_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    speaker_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # キャラクターの発言では NULL。
    speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    source: Mapped[str] = mapped_column(String(32), default=SourceKind.LOCAL_TEXT, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    delivery_state: Mapped[str] = mapped_column(
        String(16), default=DeliveryState.COMPLETED, nullable=False
    )
    delivery_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    speaker: Mapped[Speaker | None] = relationship()


class Memory(Base):
    """長期記憶。根拠の会話を残し、訂正・取り消しができる。"""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_status_subject", "status", "subject_speaker_id"),
        Index("ix_memories_visible_to", "visible_to_speaker_id"),
        Index("ix_memories_occurred_at", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 「誰について」の記憶か。キャラクター自身の経験なら NULL。
    subject_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    # 「誰との会話で参照してよいか」。非公開記憶をその相手との会話に限定する。
    # NULL は相手を限定しない記憶。「誰について」とは別の軸として持つ。
    visible_to_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    certainty: Mapped[str] = mapped_column(String(16), default=Certainty.FACT, nullable=False)
    visibility: Mapped[str] = mapped_column(String(16), default=Visibility.PRIVATE, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=MemoryStatus.ACTIVE, nullable=False)
    # 記憶検索のキーワード。フェーズ1は語の一致で検索し、意味検索は後から足す。
    keywords: Mapped[str] = mapped_column(Text, default="", nullable=False)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 根拠の会話。どの発言からこの記憶ができたかを追える。
    source_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    source_conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    # 訂正で置き換わった場合の後継。
    superseded_by_id: Mapped[int | None] = mapped_column(ForeignKey("memories.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    subject: Mapped[Speaker | None] = relationship(foreign_keys=[subject_speaker_id])
    source_message: Mapped[Message | None] = relationship()


class MemoryRevision(Base):
    """記憶の変更履歴。元の状態へ戻せるように更新前後を残す。"""

    __tablename__ = "memory_revisions"
    __table_args__ = (Index("ix_memory_revisions_memory", "memory_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    memory_id: Mapped[int] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryCandidate(Base):
    """会話終了時に抽出した長期記憶の候補。開発者が採用・保留・却下する。"""

    __tablename__ = "memory_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    subject_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    visible_to_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    certainty: Mapped[str] = mapped_column(String(16), default=Certainty.INFERENCE, nullable=False)
    visibility: Mapped[str] = mapped_column(String(16), default=Visibility.PRIVATE, nullable=False)
    keywords: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    status: Mapped[str] = mapped_column(String(16), default=CandidateStatus.PENDING, nullable=False)
    accepted_memory_id: Mapped[int | None] = mapped_column(ForeignKey("memories.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunRecord(Base):
    """実行記録。モデルの版・生成設定・参照した記憶・応答時間を残す。"""

    __tablename__ = "run_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(128))
    options: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    # 生成時に実際に渡した記憶の id。返答の根拠を後から追える。
    referenced_memory_ids: Mapped[list[int] | None] = mapped_column(JSON)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdealResponse(Base):
    """理想の返答。フェーズ7の教師データと、フェーズ5の比較評価に使う。"""

    __tablename__ = "ideal_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    ideal_text: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
