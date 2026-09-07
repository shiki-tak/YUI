"""長期記憶の保存・検索・訂正。

いまの検索は「相手・時刻・キーワード」で行う。設計書は、意味検索を検索漏れが
具体的に確認された段階で追加するとしており、ベクトルDBの導入を前提にしない。
まず一致の理由が説明できる方式を保つ（ISSUE-008）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Certainty,
    ConversationMode,
    Memory,
    MemoryKind,
    MemoryRevision,
    MemoryStatus,
    Speaker,
    Visibility,
    utcnow,
)

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]+")
_CJK_RUN = re.compile(r"[ぁ-んァ-ヶー一-龠々]+")
_HIRAGANA_ONLY = re.compile(r"^[ぁ-ん]+$")
# 弱い語（ひらがなだけの2文字）1つだけの一致では拾わない。
_MIN_MATCH_WEIGHT = 0.5
# 助詞・助動詞など、単独では検索の手がかりにならない語。
_STOP_TOKENS = {
    "です", "ます", "した", "して", "ない",
    "ある", "いる", "こと", "もの", "これ", "それ",
}

# 記憶の種別。検索語にも使えるよう、日本語の呼び名を検索対象に含める。
KIND_LABEL = {
    MemoryKind.EXPERIENCE.value: "経験",
    MemoryKind.ABOUT_PERSON.value: "相手について知ったこと",
    MemoryKind.PROMISE.value: "約束",
    MemoryKind.IMPRESSION.value: "受け止め方",
    MemoryKind.FACT.value: "確認した事実",
}


def tokenize(text: str) -> set[str]:
    """検索用の語を取り出す。

    形態素解析は入れず、英数字語と日本語の2文字ぶんの並びを手がかりにする。
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: set[str] = set(_ASCII_WORD.findall(normalized))
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            tokens.add(run)
            continue
        for i in range(len(run) - 1):
            tokens.add(run[i : i + 2])
    return tokens - _STOP_TOKENS


def token_weight(token: str) -> float:
    """語の手がかりとしての強さ。

    ひらがなだけの2文字は「をし」のように助詞の切れ端になりやすく、
    偶然一致しやすい。漢字・カタカナ・英数字を含む語を優先する。
    """
    return 0.25 if _HIRAGANA_ONLY.match(token) else 1.0


def memory_tokens(memory: Memory) -> set[str]:
    """記憶の検索対象。本文とキーワードに加え、種別の呼び名も含める。"""
    label = KIND_LABEL.get(memory.kind, "")
    return tokenize(f"{memory.content} {memory.keywords} {label}")


@dataclass
class RetrievedMemory:
    """検索で選ばれた記憶と、選ばれた理由。開発者が確認できるようにする。"""

    memory: Memory
    score: float
    reason: str


async def get_or_create_speaker(
    session: AsyncSession, *, source: str, external_id: str, display_name: str
) -> Speaker:
    """入力元とその識別子で相手を同定する。表示名だけでは同定しない。"""
    stmt = select(Speaker).where(Speaker.source == source, Speaker.external_id == external_id)
    speaker = (await session.execute(stmt)).scalar_one_or_none()
    if speaker is None:
        speaker = Speaker(source=source, external_id=external_id, display_name=display_name)
        session.add(speaker)
        await session.flush()
        return speaker
    if display_name and speaker.display_name != display_name:
        speaker.display_name = display_name
    return speaker


def _access_condition(mode: str, speaker_id: int | None):
    """その会話でこの記憶を参照してよいかの条件。

    「誰についての記憶か」（subject_speaker_id）とは別の軸として、
    「誰との会話で参照してよいか」（visible_to_speaker_id）で絞る。
    非公開記憶が、由来と関係のない相手との会話へ渡らないようにする。
    """
    if mode == ConversationMode.STREAM:
        # 配信では公開可能な記憶だけを使う。
        return Memory.visibility == Visibility.PUBLIC.value

    # ローカルでは、公開可能な記憶と、相手を限定しない記憶と、
    # この相手との会話で参照してよい記憶を使う。
    condition = (Memory.visibility == Visibility.PUBLIC.value) | (
        Memory.visible_to_speaker_id.is_(None)
    )
    if speaker_id is not None:
        condition = condition | (Memory.visible_to_speaker_id == speaker_id)
    return condition


async def search_memories(
    session: AsyncSession,
    *,
    query: str,
    speaker_id: int | None,
    mode: str = ConversationMode.LOCAL,
    limit: int = 8,
    now: datetime | None = None,
) -> list[RetrievedMemory]:
    """会話に渡す記憶を選ぶ。

    - 相手：その相手についての記憶と、相手に紐づかない自分の経験を対象にする。
    - キーワード：語の重なりで点数をつける。
    - 時刻：新しい記憶をわずかに優先する。
    - 約束は、語が一致しなくても直近のものを落とさない。
    """
    now = now or utcnow()
    stmt = select(Memory).where(
        Memory.status == MemoryStatus.ACTIVE.value,
        _access_condition(mode, speaker_id),
    )
    if speaker_id is not None:
        stmt = stmt.where(
            (Memory.subject_speaker_id == speaker_id) | (Memory.subject_speaker_id.is_(None))
        )
    else:
        stmt = stmt.where(Memory.subject_speaker_id.is_(None))

    memories = list((await session.execute(stmt)).scalars())
    if not memories:
        return []

    query_tokens = tokenize(query)
    scored: dict[int, RetrievedMemory] = {}

    for memory in memories:
        tokens = memory_tokens(memory)
        if not tokens:
            continue
        overlap = query_tokens & tokens
        matched_weight = sum(token_weight(t) for t in overlap)
        if matched_weight < _MIN_MATCH_WEIGHT:
            continue
        # 重なりの強さを、記憶の長さで割って正規化する。
        total_weight = sum(token_weight(t) for t in tokens)
        keyword_score = matched_weight / (total_weight**0.5)
        age_days = max((now - _aware(memory.created_at)).total_seconds() / 86400.0, 0.0)
        recency_score = 1.0 / (1.0 + age_days / 30.0)
        score = keyword_score + 0.2 * recency_score
        shown = sorted(overlap, key=lambda t: (-token_weight(t), t))[:5]
        scored[memory.id] = RetrievedMemory(
            memory=memory,
            score=round(score, 4),
            reason=f"キーワード一致: {'、'.join(shown)}",
        )

    ranked = sorted(scored.values(), key=lambda r: r.score, reverse=True)[:limit]
    selected = {r.memory.id for r in ranked}

    # 約束は語が一致しなくても直近ぶんを残す。忘れたことに気づけないため。
    recent_promises = sorted(
        (
            m
            for m in memories
            if m.kind == MemoryKind.PROMISE.value
            and m.id not in selected
            and _aware(m.created_at) > now - timedelta(days=90)
        ),
        key=lambda m: m.created_at,
        reverse=True,
    )[:2]
    ranked += [
        RetrievedMemory(memory=m, score=0.0, reason="直近の約束として常に参照")
        for m in recent_promises
    ]
    return ranked


def _aware(value: datetime) -> datetime:
    """SQLite から素の datetime が返る場合に備えて UTC を補う。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=utcnow().tzinfo)
    return value


def snapshot(memory: Memory) -> dict:
    """変更履歴に残す記憶の内容。"""
    return {
        "kind": memory.kind,
        "content": memory.content,
        "subject_speaker_id": memory.subject_speaker_id,
        "visible_to_speaker_id": memory.visible_to_speaker_id,
        "certainty": memory.certainty,
        "visibility": memory.visibility,
        "status": memory.status,
        "keywords": memory.keywords,
        "occurred_at": memory.occurred_at.isoformat() if memory.occurred_at else None,
        "superseded_by_id": memory.superseded_by_id,
    }


def record_revision(
    session: AsyncSession,
    memory: Memory,
    *,
    action: str,
    before: dict | None,
    reason: str | None = None,
) -> MemoryRevision:
    revision = MemoryRevision(
        memory_id=memory.id,
        action=action,
        before=before,
        after=snapshot(memory),
        reason=reason,
    )
    session.add(revision)
    return revision


async def create_memory(
    session: AsyncSession,
    *,
    kind: str,
    content: str,
    subject_speaker_id: int | None = None,
    visible_to_speaker_id: int | None = None,
    certainty: str = Certainty.FACT.value,
    visibility: str = Visibility.PRIVATE.value,
    keywords: str = "",
    occurred_at: datetime | None = None,
    source_message_id: int | None = None,
    source_conversation_id: int | None = None,
    reason: str | None = None,
) -> Memory:
    memory = Memory(
        kind=kind,
        content=content,
        subject_speaker_id=subject_speaker_id,
        visible_to_speaker_id=visible_to_speaker_id,
        certainty=certainty,
        visibility=visibility,
        keywords=keywords,
        occurred_at=occurred_at,
        source_message_id=source_message_id,
        source_conversation_id=source_conversation_id,
    )
    session.add(memory)
    await session.flush()
    record_revision(session, memory, action="created", before=None, reason=reason)
    return memory
