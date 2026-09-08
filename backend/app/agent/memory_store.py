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
    Provenance,
    Speaker,
    Visibility,
    utcnow,
)

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]+")
# ひらがな・カタカナ・漢字を、それぞれ別の連なりとして取り出す。まとめて
# 1つの連なりにすると、2文字ずつの切り出しが語の切れ目をまたぐ。
# 「この前の山の話」が「の山」「山の」になり、記憶側の「山」と一致しない
# （ISSUE-019）。文字種の変わり目は、日本語では語の切れ目に近い。
_TOKEN_RUN = re.compile(r"[ぁ-ん]+|[ァ-ヶー]+|[一-龠々]+")
_HIRAGANA_ONLY = re.compile(r"^[ぁ-ん]+$")
# 弱い一致1つだけでは拾わない。
_MIN_MATCH_WEIGHT = 0.5
# 長い語の中から切り出した1文字の強さ。「山田」の「山」で「高尾山」の記憶を
# 引き寄せないよう、語そのものより弱く扱う。
_PARTIAL_WEIGHT = 0.3
# 助詞・助動詞など、単独では検索の手がかりにならない語。
# 1文字のものは、どの会話にも出てきて記憶を絞れないもの（「話」「人」など）。
# ここは評価用会話で見つかった取りこぼし・拾いすぎを見ながら足す。
# 語の重要度を頻度から決める方式は、FTS5 か意味検索を入れる段階で扱う
# （ISSUE-008）。
_STOP_TOKENS = {
    "です", "ます", "した", "して", "ない",
    "ある", "いる", "こと", "もの", "これ", "それ",
    "話", "人", "事", "時", "今", "何", "方", "中",
}

# 記憶の種別。検索語にも使えるよう、日本語の呼び名を検索対象に含める。
KIND_LABEL = {
    MemoryKind.EXPERIENCE.value: "経験",
    MemoryKind.ABOUT_PERSON.value: "相手について知ったこと",
    MemoryKind.PROMISE.value: "約束",
    MemoryKind.IMPRESSION.value: "受け止め方",
    MemoryKind.FACT.value: "確認した事実",
}


def tokenize(text: str) -> dict[str, float]:
    """検索用の語と、その手がかりとしての強さを返す。

    形態素解析は入れず、文字種の変わり目で区切って、2文字ぶんの並びを
    手がかりにする。同じ語を別の経路で拾った場合は、強いほうを採る。

    強さを語ごとに持つのは、同じ「山」でも、語として単独で出てきたのか、
    「山田」から切り出したのかで確かさが違うため（ISSUE-019）。
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: dict[str, float] = {}

    def add(token: str, weight: float) -> None:
        if token in _STOP_TOKENS:
            return
        if tokens.get(token, 0.0) < weight:
            tokens[token] = weight

    for word in _ASCII_WORD.findall(normalized):
        add(word, 1.0)

    for run in _TOKEN_RUN.findall(normalized):
        # ひらがなだけの並びは「をし」のように助詞の切れ端になりやすく、
        # 偶然一致しやすい。漢字・カタカナ・英数字を含む語を優先する。
        weight = 0.25 if _HIRAGANA_ONLY.match(run) else 1.0
        if len(run) == 1:
            add(run, weight)
            continue
        for i in range(len(run) - 1):
            add(run[i : i + 2], weight)
        if weight == 1.0:
            for char in run:
                add(char, _PARTIAL_WEIGHT)
    return tokens


def memory_tokens(memory: Memory) -> dict[str, float]:
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
        # 片方が語の一部でしかないなら、その一致は弱い。弱いほうに合わせる。
        matched = {
            token: min(query_tokens[token], weight)
            for token, weight in tokens.items()
            if token in query_tokens
        }
        matched_weight = sum(matched.values())
        if matched_weight < _MIN_MATCH_WEIGHT:
            continue
        # 重なりの強さを、記憶の長さで割って正規化する。
        total_weight = sum(tokens.values())
        keyword_score = matched_weight / (total_weight**0.5)
        age_days = max((now - _aware(memory.created_at)).total_seconds() / 86400.0, 0.0)
        recency_score = 1.0 / (1.0 + age_days / 30.0)
        score = keyword_score + 0.2 * recency_score
        shown = sorted(matched, key=lambda t: (-matched[t], t))[:5]
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


# 「既存の記憶と近い」と見なす境目。実際の記憶と、実モデルが出した候補で
# 測って決めた。
#
#   出す   1.00 同じ文 ／ 0.97 同じ出来事の再言及 ／ 0.81 訂正（別の好み）
#          0.72 言い換え ／ 0.66 2回目に話したときの言い方（実測）
#   出さない 0.52 同じ話題だが別の事実 ／ 0.47・0.44 別の話
#
# 「コーヒーが好き」と「紅茶が好き」（0.81）も出す。これは重複ではなく訂正だが、
# 開発者に見せる価値がある。自動で捨てず、判断してもらうための印である。
_SIMILAR_THRESHOLD = 0.6


def similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """語の重なりから見た近さ。0〜1。

    短いほうを基準にする（min）。片方が言い換えで短くても、同じことを
    言っていれば高くなるようにするため。
    """
    shared = sum(min(left[token], right[token]) for token in left.keys() & right.keys())
    total = min(sum(left.values()), sum(right.values()))
    return shared / total if total else 0.0


async def find_similar_memories(
    session: AsyncSession,
    *,
    content: str,
    keywords: str = "",
    speaker_id: int | None,
    mode: str = ConversationMode.LOCAL,
    limit: int = 5,
) -> list[RetrievedMemory]:
    """内容が近い既存の記憶を探す（ISSUE-018）。

    同じ出来事を二重に覚えないための手がかり。自動では捨てず、開発者が
    採用を判断するときに見せる。参照範囲は会話のときと同じ条件で絞る。
    """
    stmt = select(Memory).where(
        Memory.status == MemoryStatus.ACTIVE.value,
        _access_condition(mode, speaker_id),
    )
    candidate_tokens = tokenize(f"{content} {keywords}")
    if not candidate_tokens:
        return []

    found: list[RetrievedMemory] = []
    for memory in (await session.execute(stmt)).scalars():
        # 近さの計算には種別の呼び名を混ぜない。検索では手がかりになるが、
        # ここでは「相手について知ったこと」のような共通の語が重なって、
        # 内容の近さを薄める。
        score = similarity(candidate_tokens, tokenize(f"{memory.content} {memory.keywords}"))
        if score < _SIMILAR_THRESHOLD:
            continue
        found.append(
            RetrievedMemory(
                memory=memory,
                score=round(score, 4),
                reason=f"内容が近い（{score:.0%}）",
            )
        )
    return sorted(found, key=lambda r: r.score, reverse=True)[:limit]


def snapshot(memory: Memory) -> dict:
    """変更履歴に残す記憶の内容。"""
    return {
        "kind": memory.kind,
        "content": memory.content,
        "provenance": memory.provenance,
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
    provenance: str = Provenance.UNKNOWN.value,
    visibility: str = Visibility.PRIVATE.value,
    keywords: str = "",
    occurred_at: datetime | None = None,
    source_message_id: int | None = None,
    source_conversation_id: int | None = None,
    reason: str | None = None,
    created_at: datetime | None = None,
) -> Memory:
    memory = Memory(
        kind=kind,
        content=content,
        subject_speaker_id=subject_speaker_id,
        visible_to_speaker_id=visible_to_speaker_id,
        certainty=certainty,
        provenance=provenance,
        visibility=visibility,
        keywords=keywords,
        occurred_at=occurred_at,
        source_message_id=source_message_id,
        source_conversation_id=source_conversation_id,
        # 既定は保存した時刻。差し替えられるのは、評価で時間を進めたときに
        # 検索の減衰を進めた側の時刻で計算するため。
        **({"created_at": created_at} if created_at is not None else {}),
    )
    session.add(memory)
    await session.flush()
    record_revision(session, memory, action="created", before=None, reason=reason)
    return memory
