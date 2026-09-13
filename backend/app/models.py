"""SQLAlchemy モデル。

設計書「5. 記憶・経験・目標のデータ設計」に対応する。特に次の区別を構造として
持たせている。

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
    Boolean,
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
    """入力元。フェーズ6A（YouTube接続）で youtube が増える。

    ローカルの入力は現在は文字だけ。マイク入力はフェーズ10で追加する。
    その場合も文字起こしを同じ入力受付へ送るため、この列は文字入力と同じ扱いになる。
    """

    LOCAL_TEXT = "local_text"
    YOUTUBE = "youtube"


class ConversationMode(StrEnum):
    """会話モード。stream では公開可能な記憶だけを参照する（フェーズ6A）。"""

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


class Provenance(StrEnum):
    """その内容をどうやって知ったか（設計書フェーズ3の3C）。

    「AさんがBさんの好みを話した」を、Bさん本人の発言と区別する。事実か
    推測かを表す certainty とは別の軸で、こちらは情報の入手経路を表す。

    観察（画像）や調査（検索）はフェーズ5で扱う。ここでは会話から知る2つと、
    判断できない場合だけを持つ。分類できないものを、どちらかへ寄せない。
    """

    # 本人が、自分のこととして話した。
    FIRSTHAND = "firsthand"
    # 別の人について話した（伝聞）。話した人は source_message_id の発言者。
    HEARSAY = "hearsay"
    # 判断できない。移行前の記憶もこれになる。
    UNKNOWN = "unknown"


class Visibility(StrEnum):
    PRIVATE = "private"  # ローカル会話限定
    PUBLIC = "public"  # 配信で参照してよい


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CORRECTED = "corrected"  # 訂正され、後続の記憶に置き換わった
    DELETED = "deleted"


class StateKind(StrEnum):
    """可変状態の種類（設計書 4.1「固定人格と変化する状態」）。

    固定人格（口調・価値観・自己設定）は personas/<版>.toml にあり、ここでは
    扱わない。ここに置くのは、経験によって変わっていくものだけ。

    目標（次に感想を聞く、一緒に調べる）はフェーズ4の担当なので、まだ持たない。
    ただし「候補 → 開発者が採用 → 根拠を残す」という流れは同じ形にしてあり、
    フェーズ4では kind を1つ増やし、実行条件・期限・完了条件の列を足せば足りる
    ようにしている。
    """

    # YUI 自身の関心・好み。相手の好みとは別に持つ。
    INTEREST = "interest"
    # 相手との関係。共有した経験、距離感、相手への理解。
    RELATIONSHIP = "relationship"


class StateStatus(StrEnum):
    """可変状態の扱い。候補から採用までを同じ表で追う。"""

    # 振り返りが出した更新候補。開発者が確認するまで会話には使わない。
    PENDING = "pending"
    # 採用済み。会話で参照する。
    ACTIVE = "active"
    # 採用しなかった。
    REJECTED = "rejected"
    # 新しい版に置き換えられた。
    SUPERSEDED = "superseded"
    # 取り消した。
    WITHDRAWN = "withdrawn"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ConversationStateKind(StrEnum):
    """v0.2 の会話状態（設計書「6. 会話・自発的行動の流れ」、計画 docs/plan/v0.2.md 3節）。

    request / confirmed / correction / discrepancy は解釈（LLM）でしか作らない。
    それ以外は規則だけで作る（v0.2 PR1 の範囲）。
    """

    REQUEST = "request"
    QUESTION_TO_YUI = "question_to_yui"
    QUESTION_TO_PARTNER = "question_to_partner"
    PRESENTED = "presented"
    CONFIRMED = "confirmed"
    DEFERRAL = "deferral"
    CLOSING = "closing"
    DISCREPANCY = "discrepancy"
    CORRECTION = "correction"


class ConversationStateStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"
    # 会話が終わって閉じただけで、解決したという意味ではない（計画 5節）。
    EXPIRED = "expired"


class ConversationStateWithdrawReason(StrEnum):
    # 相手が延期・終了した話題を明示的に再開した。
    REOPENED = "reopened"
    # 相手が取り下げた（質問等）。
    CANCELLED = "cancelled"
    # 誤検出。
    MISDETECTED = "misdetected"
    # 新しい行に置き換えられた（訂正の再訂正、用件の変化）。
    SUPERSEDED = "superseded"


class ConversationStateRefKind(StrEnum):
    MEMORY = "memory"
    MESSAGE = "message"
    STATE = "state"


class DetectionSource(StrEnum):
    """作成の経路。更新しても変えない（計画 3節）。"""

    RULE = "rule"
    LLM = "llm"


class DecisionSource(StrEnum):
    """解決・取消の経路。作成の経路とは別に持つ（計画 3節）。"""

    RULE = "rule"
    LLM = "llm"
    OPERATOR = "operator"


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
    # 振り返りの「処理中」と「完了」を分ける。中断やプロセス停止で
    # 開始だけが残った場合に、やり直せるようにするため。
    reflection_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reflection_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
    # どうやって知ったか。伝聞を本人の発言として扱わないために持つ。
    provenance: Mapped[str] = mapped_column(
        String(16), default=Provenance.UNKNOWN, server_default=Provenance.UNKNOWN, nullable=False
    )
    visibility: Mapped[str] = mapped_column(String(16), default=Visibility.PRIVATE, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default=MemoryStatus.ACTIVE, nullable=False)
    # 記憶検索のキーワード。いまは語の一致で検索する。意味検索は、検索漏れが
    # 具体的に確認された段階で足す（ISSUE-008）。
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
    provenance: Mapped[str] = mapped_column(
        String(16), default=Provenance.UNKNOWN, server_default=Provenance.UNKNOWN, nullable=False
    )
    visibility: Mapped[str] = mapped_column(String(16), default=Visibility.PRIVATE, nullable=False)
    keywords: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # いつの出来事か（ISSUE-004）。会話に日付の手がかりが無ければ NULL。
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    # 内容が近い既存の記憶（ISSUE-018）。二重に覚えないための手がかりとして
    # 採用の判断時に見せる。自動では捨てない。
    similar_memory_ids: Mapped[list[int] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default=CandidateStatus.PENDING, nullable=False)
    accepted_memory_id: Mapped[int | None] = mapped_column(ForeignKey("memories.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CharacterState(Base):
    """変化する状態：YUI の関心と、相手との関係（設計書 4.1・5、ISSUE-015）。

    固定人格と分ける。人格は版として管理し、日常の振り返りで上書きしない。
    こちらは経験を根拠に更新していく。

    単一の好感度の数値にしない。何を根拠にそう思っているかを本文と根拠で残し、
    後から訂正できるようにする。
    """

    __tablename__ = "character_states"
    __table_args__ = (Index("ix_character_states_kind_status", "kind", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # 関係性のとき：誰との関係か。関心のときは NULL。
    subject_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    # 関心のとき：どの分野・話題についてか。関係性のときは NULL。
    topic: Mapped[str | None] = mapped_column(String(120))
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 根拠になった記憶。訂正・削除されたら再評価が要る（ISSUE-016）。
    basis_memory_ids: Mapped[list[int] | None] = mapped_column(JSON)
    # 上の根拠が「暫定」かどうか。採用のときに、その会話から採用済みの記憶を
    # 自動で並べたものは暫定になる。後から採用された記憶が抜けているため、
    # 完全に特定した根拠として扱わない（フェーズ3再々レビューの指摘1）。
    basis_is_provisional: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    # 根拠が変わった。開発者が確認するまで印を残す。自動では消さない。
    needs_review: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    review_reason: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(String(16), default=StateStatus.PENDING, nullable=False)
    # 参照範囲。記憶と同じ考え方で、非公開のものを配信で使わない。
    visibility: Mapped[str] = mapped_column(String(16), default=Visibility.PRIVATE, nullable=False)
    visible_to_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))

    superseded_by_id: Mapped[int | None] = mapped_column(ForeignKey("character_states.id"))
    source_conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    subject: Mapped[Speaker | None] = relationship(foreign_keys=[subject_speaker_id])


class CharacterStateRevision(Base):
    """状態の変更履歴。何を根拠に、誰が採用したかを残して戻せるようにする。"""

    __tablename__ = "character_state_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_id: Mapped[int] = mapped_column(
        ForeignKey("character_states.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ConversationState(Base):
    """v0.2 の会話状態（設計書「6. 会話・自発的行動の流れ」、計画 docs/plan/v0.2.md 3節）。

    発言があったことと解決したことを分けて持つ。`responded_message_id` は
    「一度応答があった」だけを示し、`status` を `resolved` にはしない。解決・
    取消は解釈（LLM）か開発者の操作でだけ起き、規則は作成と応答の記録しか
    しない。`target_speaker_id` と一致しない相手の発言では、どの遷移も
    起こさない（v0.2 の検証は1対1に限る）。

    変更履歴の表は作らない。遷移は1段で、根拠・確認・応答・解消の発言 ID を
    この行が持つため、後から辿れる。
    """

    __tablename__ = "conversation_states"
    __table_args__ = (
        Index("ix_conversation_states_conv_status", "conversation_id", "status"),
        Index(
            "ix_conversation_states_conv_ref",
            "conversation_id",
            "kind",
            "ref_kind",
            "ref_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 誰の発言に由来するか。YUI の返答由来なら NULL。
    speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    # 誰に適用するか。別の相手の発言ではこの状態を変えない。
    target_speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"))
    source_message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    # 対象（食い違い・訂正・確認の相手）。discrepancy/correction は記憶か発言、
    # confirmed は対象の presented。
    ref_kind: Mapped[str | None] = mapped_column(String(16))
    ref_id: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(16), default=ConversationStateStatus.OPEN, nullable=False
    )
    withdraw_reason: Mapped[str | None] = mapped_column(String(16))
    # 実際にその食い違いを確認した YUI の返答（discrepancy だけが使う）。
    asked_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    # 最初に応答した発言。入っていても open でありうる（未判定・聞き返し・不明確）。
    responded_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    # 最後に判定した応答候補。これより新しい応答候補が来たら再判定の対象になる
    # （v0.2 PR3）。
    judged_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    # 解釈が「応答したが答えていない」と判定した（v0.2 PR3）。
    followup_needed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 解決・取消・失効を決めた発言。
    resolved_message_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"))
    # 作成の経路。更新しても変えない。
    detected_by: Mapped[str] = mapped_column(String(16), nullable=False)
    # 解決・取消の経路。作成の経路とは別に持つ。
    decided_by: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


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
    # 生成に使った固定人格の版（personas/<版>.toml）。版を変えた前後を
    # 同じ会話例で比べられるようにする（設計書フェーズ3の3A）。
    # system_prompt も引き続き残す。版の定義を後から書き換えても、
    # そのとき実際に渡した文面は変わらないため。
    persona_version: Mapped[str | None] = mapped_column(String(64))
    options: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    # 生成時に実際に渡した記憶の id。返答の根拠を後から追える。
    referenced_memory_ids: Mapped[list[int] | None] = mapped_column(JSON)
    # 返答に渡した可変状態（関心・関係性）。完了条件「参照した記憶・人格版・
    # 可変状態を追跡できる」のために、記憶と分けて残す。
    referenced_state_ids: Mapped[list[int] | None] = mapped_column(JSON)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    # 記憶検索にかかった時間。生成の時間と分けて、どこが遅いかを見る。
    retrieval_ms: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SpeechRun(Base):
    """音声合成の実行記録。

    設計書フェーズ2の実装内容6「送信から音声の再生開始までの時間を測り、
    待ち時間の大きい部分を確認する」に使う。合成用データの作成と音声生成を
    分けて残し、音声そのものの長さとも区別する。

    1 つの発言を聞き直すたびに 1 行増える。同じ文章の合成にかかる時間が
    毎回どうなるかを見られるようにするため。
    """

    __tablename__ = "speech_runs"
    __table_args__ = (Index("ix_speech_runs_message", "message_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    speaker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    engine_version: Mapped[str | None] = mapped_column(String(64))
    # 合成用データの作成にかかった時間。
    query_ms: Mapped[int | None] = mapped_column(Integer)
    # 音声そのものの生成にかかった時間。
    synthesis_ms: Mapped[int | None] = mapped_column(Integer)
    # 生成された音声の長さ。合成の速さと区別する。
    audio_ms: Mapped[int | None] = mapped_column(Integer)
    byte_size: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdealResponse(Base):
    """理想の返答。並行改善（学習）の教師データと、フェーズ3の比較評価に使う。"""

    __tablename__ = "ideal_responses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    ideal_text: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
