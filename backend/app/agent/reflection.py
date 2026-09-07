"""会話終了後の振り返り。長期記憶の候補を作る。

会話履歴をそのまま記憶に昇格させない。残す価値のある内容だけを候補として
出し、採用は開発者が画面で判断する（フェーズ4で自動化の範囲を広げる）。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.base import ChatMessage, LLMClient
from app.models import (
    CandidateStatus,
    Certainty,
    Conversation,
    MemoryCandidate,
    MemoryKind,
    Message,
    SpeakerKind,
    Visibility,
)

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class ReflectionParseError(RuntimeError):
    """振り返りの出力を候補として読み取れなかった。

    「残す価値のある内容が無かった（空の配列）」とは区別する。区別しないと、
    抽出の失敗が「候補なしの成功」として会話を終了させてしまう。

    配列の要素が1つでも読み取れない場合も失敗として扱う。落とした候補は
    「記憶になり損ねた経験」であり、静かに捨てると失われたことに気づけない。
    振り返りの再実行は短時間で済むため、部分的に採らず全体をやり直す。
    """


class _ItemError(ValueError):
    """候補1件を読み取れなかった。理由を添えて上位へ伝える。"""

_INSTRUCTION = """あなたは会話ログから、後の会話で役に立つ記憶の候補を抜き出す担当です。

次の会話を読み、長期的に覚えておく価値のあることだけを JSON 配列で出力してください。

各要素の形式:
{
  "kind": "experience" | "about_person" | "promise" | "impression",
  "content": "一文で書いた覚えておく内容",
  "certainty": "fact" | "inference",
  "keywords": "検索用の語を空白区切りで3〜6個",
  "about_partner": true | false,
  "source_message_id": 根拠になった発言の番号（会話の [#番号] から選ぶ）
}

規則:
- kind の意味は experience=出来事、about_person=相手について知ったこと、promise=約束、
  impression=キャラクター側の受け止め方。
- promise は「次に話す」「次にする」と決めた内容。「次は◯◯の話をしよう」
  「今度◯◯しよう」のような話題の予告も含む。会話に出てきたら、他に何を出すかに
  関わらず必ず1件残す。
- 相手が話した出来事や事情は、一度しか出てこなくても残す。後の会話で「その話」
  として触れられる内容かどうかで判断する。
- 会話の中で相手が実際に言ったことだけを根拠にする。書かれていないことを補わない。
- キャラクター自身の経験や過去を作らない。会話に出てきていない体験を、
  どの種別でもキャラクターのものとして書かない。impression はキャラクター側の
  受け止め方だけに使う。
- 明言された内容は "fact"、読み取っただけの推測は "inference" とする。
- あいさつ、天気の話のようなその場限りのやり取り、既に一般常識であることは出さない。
  ただし、上の promise と、相手が話した出来事はこれに当たらない。
- source_message_id には、その内容の根拠になった発言の番号を1つだけ選ぶ。
  会話に出てくる [#番号] のいずれかで、推測して番号を作らない。
- 該当が無ければ [] とだけ出力する。
- JSON 配列だけを出力し、説明文やコードブロックは付けない。
"""


class CandidatePayload(BaseModel):
    kind: str
    content: str = Field(min_length=1, max_length=500)

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> object:
        """最小長を見る前に空白を落とす（ISSUE-001）。

        検証を通してから strip() すると、空白・改行・タブだけの本文が
        「本文が空の候補」として保存され、抽出に成功したことになる。
        """
        return value.strip() if isinstance(value, str) else value

    certainty: str = Certainty.INFERENCE.value
    keywords: str = ""
    about_partner: bool = False
    source_message_id: int | None = None


def _excerpt(text: str, limit: int = 200) -> str:
    condensed = " ".join(text.split())
    return condensed[:limit] + ("…" if len(condensed) > limit else "")


def _parse_item(item: object) -> CandidatePayload:
    """候補1件を読み取る。読み取れない場合は理由を添えて失敗させる。"""
    if not isinstance(item, dict):
        raise _ItemError("要素がオブジェクトではありません")
    try:
        payload = CandidatePayload.model_validate(item)
    except ValidationError as exc:
        reasons = "、".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise _ItemError(f"項目を読み取れません（{reasons}）") from exc
    if payload.kind not in {kind.value for kind in MemoryKind}:
        raise _ItemError(f"種別が不正です: {payload.kind}")
    # 項目が無い場合は既定の「推測」を使う。書かれていて読めない値は、
    # 事実か推測かを決められないため失敗にする。
    if payload.certainty not in {certainty.value for certainty in Certainty}:
        raise _ItemError(f"確かさが不正です: {payload.certainty}")
    return payload


def _parse_candidates(text: str) -> list[CandidatePayload]:
    match = _JSON_ARRAY.search(text)
    if not match:
        raise ReflectionParseError(
            f"振り返りの出力にJSON配列が見つかりませんでした: {_excerpt(text)}"
        )
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ReflectionParseError(
            f"振り返りの出力をJSONとして読み取れませんでした: {_excerpt(text)}"
        ) from exc
    if not isinstance(raw, list):
        raise ReflectionParseError(
            f"振り返りの出力が配列ではありませんでした: {_excerpt(text)}"
        )

    results: list[CandidatePayload] = []
    problems: list[str] = []
    for index, item in enumerate(raw):
        try:
            results.append(_parse_item(item))
        except _ItemError as exc:
            dumped = json.dumps(item, ensure_ascii=False, default=str)
            problems.append(f"{index + 1}件目: {exc} / 出力: {_excerpt(dumped, 120)}")

    if problems:
        raise ReflectionParseError(
            f"振り返りの出力に読み取れない候補が {len(problems)} 件ありました。"
            + "".join(f"\n- {problem}" for problem in problems)
        )
    return results


def format_transcript(messages: list[Message], partner_name: str, character_name: str) -> str:
    """発言IDを付けて並べる。候補ごとに根拠の発言を指せるようにするため。"""
    lines = []
    for message in messages:
        who = (
            character_name
            if message.speaker_kind == SpeakerKind.CHARACTER.value
            else partner_name
        )
        lines.append(f"[#{message.id}] {who}: {message.content}")
    return "\n".join(lines)


async def extract_candidates(
    session: AsyncSession,
    *,
    llm: LLMClient,
    conversation: Conversation,
    partner_speaker_id: int | None,
    partner_name: str,
    character_name: str,
) -> list[MemoryCandidate]:
    """会話から記憶の候補を抽出し、pending として保存する。"""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.id)
    )
    messages = list((await session.execute(stmt)).scalars())
    if not messages:
        return []

    transcript = format_transcript(messages, partner_name, character_name)
    response = await llm.chat(
        [
            ChatMessage(role="system", content=_INSTRUCTION),
            ChatMessage(role="user", content=f"会話:\n{transcript}"),
        ],
        # 抽出は毎回同じ結果になってほしいので、返答生成より温度を下げる。
        options={"temperature": 0.2},
    )

    # 存在しない発言を根拠にしないよう、この会話の発言だけを許す。
    valid_message_ids = {m.id for m in messages}

    candidates: list[MemoryCandidate] = []
    for payload in _parse_candidates(response.text):
        source_message_id = payload.source_message_id
        if source_message_id not in valid_message_ids:
            # 番号を作られた場合は根拠未確認として残す。直近の発言へ寄せると、
            # 無関係な発言を確かな根拠として保存してしまうため。
            source_message_id = None
        candidate = MemoryCandidate(
            conversation_id=conversation.id,
            kind=payload.kind,
            content=payload.content.strip(),
            subject_speaker_id=partner_speaker_id if payload.about_partner else None,
            # 非公開の記憶は、この会話の相手との会話でだけ参照する。
            visible_to_speaker_id=partner_speaker_id,
            certainty=payload.certainty,
            visibility=Visibility.PRIVATE.value,
            keywords=payload.keywords.strip(),
            source_message_id=source_message_id,
            status=CandidateStatus.PENDING.value,
        )
        session.add(candidate)
        candidates.append(candidate)
    await session.flush()
    return candidates
