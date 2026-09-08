"""会話から、関心・関係性の更新候補を作る（ISSUE-015 / 設計書フェーズ3の3B）。

記憶の抽出とは**別の呼び出し**にしている。同じ指示文へ項目を足すと、記憶の
抽出そのものが落ちることを実測したため（ISSUE-017：規則を足して 16/21、
組み替えて 13/21、元へ戻して 21/21）。役割ごとに分ければ、片方の調整が
もう片方を壊さない。

ここで作るのは候補だけで、採用は開発者が判断する。会話1回で人格が動かない
ようにするための境目である。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.llm.base import ChatMessage, LLMClient
from app.models import CharacterState, StateKind

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class StateReflectionError(RuntimeError):
    """状態の更新候補を読み取れなかった。

    記憶の振り返りと同じ扱いにする。「候補が無かった」と「読み取れなかった」を
    区別しないと、失敗が「更新なしの成功」として通ってしまう。
    """


_INSTRUCTION = """あなたは会話ログから、キャラクターの「いまの状態」の更新候補を
抜き出す担当です。記憶の抽出とは別の作業で、ここでは出来事そのものではなく、
その経験からキャラクター側に何が生まれたかだけを見ます。

出力は JSON 配列です。各要素の形式:
{
  "kind": "interest" | "relationship",
  "topic": "関心の対象（interest のときだけ。短い語）",
  "content": "一文で書いた内容",
  "reason": "会話のどこからそう考えたか"
}

出すもの:
- interest：キャラクターが会話を通して**知りたくなったこと、面白いと感じたこと**。
  例：相手が朝の光を撮る話をして、キャラクターがその瞬間に興味を示した
  →「人が撮った写真から、その人が何を見ていたかを知りたい」。
- relationship：その相手との関係で分かったこと。何の話でよく続くか、
  どんな距離感か。例：「開発者とは写真の話が続きやすい」。

守ること:
- interest はキャラクター自身のもの。**相手の好みは出さない**（それは記憶の担当）。
- 相手が違う意見を述べただけでは、キャラクターの好みを反転させない。
- すでに「いまの状態」にあるものと同じ内容は出さない。
- あいさつだけの会話など、キャラクター側に何も生まれていなければ [] とだけ出力する。
- JSON 配列だけを出力し、説明文やコードブロックは付けない。
"""


class StatePayload(BaseModel):
    kind: str
    content: str = Field(min_length=1, max_length=500)
    topic: str | None = Field(default=None, max_length=120)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("content", "topic", "reason", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        # 空白だけの本文を、最小長の検証より前に落とす（ISSUE-001 と同じ）。
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


def _parse(text: str) -> list[StatePayload]:
    match = _JSON_ARRAY.search(text)
    if not match:
        raise StateReflectionError(f"状態の更新候補にJSON配列が見つかりませんでした: {text[:200]}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise StateReflectionError(
            f"状態の更新候補をJSONとして読み取れませんでした: {text[:200]}"
        ) from exc
    if not isinstance(raw, list):
        raise StateReflectionError("状態の更新候補が配列ではありませんでした。")

    results: list[StatePayload] = []
    problems: list[str] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            problems.append(f"{index}件目: 要素がオブジェクトではありません")
            continue
        try:
            payload = StatePayload.model_validate(item)
        except ValidationError as exc:
            problems.append(f"{index}件目: 項目を読み取れません（{exc.error_count()}件）")
            continue
        if payload.kind not in {kind.value for kind in StateKind}:
            problems.append(f"{index}件目: 種類が不正です: {payload.kind}")
            continue
        results.append(payload)

    if problems:
        raise StateReflectionError(
            f"状態の更新候補に読み取れないものが {len(problems)} 件ありました。"
            + "".join(f"\n- {problem}" for problem in problems)
        )
    return results


def format_current_states(states: list[CharacterState]) -> str:
    """いまの状態。同じ内容を重ねて出さないために渡す。"""
    if not states:
        return "いまの状態: まだありません。"
    lines = ["いまの状態:"]
    for state in states:
        topic = f"／{state.topic}" if state.topic else ""
        lines.append(f"- [{state.kind}{topic}] {state.content}")
    return "\n".join(lines)


async def propose_state_candidates(
    *,
    llm: LLMClient,
    transcript: str,
    partner_speaker_id: int | None,
    current_states: list[CharacterState],
) -> list[StatePayload]:
    """会話から関心・関係性の更新候補を作り、pending として保存する。

    相手を決められない会話（複数の相手がいる、相手の発言が無い）では作らない。
    非公開の会話から作った状態にも参照範囲を引き継ぐ必要があり、誰に限るかを
    決められないまま作ると、別の相手へ渡る（設計書 2.1、ISSUE-010）。
    """
    if partner_speaker_id is None:
        return []

    response = await llm.chat(
        [
            ChatMessage(role="system", content=_INSTRUCTION),
            ChatMessage(
                role="user",
                content=f"{format_current_states(current_states)}\n\n会話:\n{transcript}",
            ),
        ],
        # 記憶の抽出と同じく、毎回同じ結果になってほしいので温度を下げる。
        options={"temperature": 0.2},
    )

    return _parse(response.text)
