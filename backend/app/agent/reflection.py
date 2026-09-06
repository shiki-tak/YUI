"""会話終了後の振り返り。長期記憶の候補を作る。

会話履歴をそのまま記憶に昇格させない。残す価値のある内容だけを候補として
出し、採用は開発者が画面で判断する（フェーズ5で自動化の範囲を広げる）。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError
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

_INSTRUCTION = """あなたは会話ログから、後の会話で役に立つ記憶の候補を抜き出す担当です。

次の会話を読み、長期的に覚えておく価値のあることだけを JSON 配列で出力してください。

各要素の形式:
{
  "kind": "experience" | "about_person" | "promise" | "impression",
  "content": "一文で書いた覚えておく内容",
  "certainty": "fact" | "inference",
  "keywords": "検索用の語を空白区切りで3〜6個",
  "about_partner": true | false
}

規則:
- kind の意味は experience=出来事、about_person=相手について知ったこと、promise=約束、
  impression=キャラクター側の受け止め方。
- 会話の中で相手が実際に言ったことだけを根拠にする。書かれていないことを補わない。
- 明言された内容は "fact"、読み取っただけの推測は "inference" とする。
- 次に話すと決めたことは必ず promise として残す。
- あいさつ、その場限りのやり取り、既に一般常識であることは出さない。
- 該当が無ければ [] とだけ出力する。
- JSON 配列だけを出力し、説明文やコードブロックは付けない。
"""


class CandidatePayload(BaseModel):
    kind: str
    content: str = Field(min_length=1, max_length=500)
    certainty: str = Certainty.INFERENCE.value
    keywords: str = ""
    about_partner: bool = False


def _parse_candidates(text: str) -> list[CandidatePayload]:
    match = _JSON_ARRAY.search(text)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    valid_kinds = {k.value for k in MemoryKind}
    valid_certainty = {c.value for c in Certainty}
    results: list[CandidatePayload] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            payload = CandidatePayload.model_validate(item)
        except ValidationError:
            continue
        if payload.kind not in valid_kinds:
            continue
        if payload.certainty not in valid_certainty:
            payload.certainty = Certainty.INFERENCE.value
        results.append(payload)
    return results


def format_transcript(messages: list[Message], partner_name: str, character_name: str) -> str:
    lines = []
    for message in messages:
        who = (
            character_name
            if message.speaker_kind == SpeakerKind.CHARACTER.value
            else partner_name
        )
        lines.append(f"{who}: {message.content}")
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

    last_user_message_id = next(
        (m.id for m in reversed(messages) if m.speaker_kind == SpeakerKind.USER.value),
        None,
    )

    candidates: list[MemoryCandidate] = []
    for payload in _parse_candidates(response.text):
        candidate = MemoryCandidate(
            conversation_id=conversation.id,
            kind=payload.kind,
            content=payload.content.strip(),
            subject_speaker_id=partner_speaker_id if payload.about_partner else None,
            certainty=payload.certainty,
            visibility=Visibility.PRIVATE.value,
            keywords=payload.keywords.strip(),
            source_message_id=last_user_message_id,
            status=CandidateStatus.PENDING.value,
        )
        session.add(candidate)
        candidates.append(candidate)
    await session.flush()
    return candidates
