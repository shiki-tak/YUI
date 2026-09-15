"""会話状態の解釈（LLM）。v0.2 PR3（計画 docs/plan/v0.2.md §4 手順3・§3）。

規則（`conversation_state.py`）が作れるのは「質問・依頼・延期・終了の合図が
あった」ことだけで、**意味の解釈**——今の用件は何か、質問に答えたか、
食い違いを確認したか、明示的な訂正か——は要らない。ここでは会話の直近と、
判定対象になる質問・確認候補・記憶を渡し、決めたスキーマの JSON で結果を
受け取る。**ID の検証・状態への適用は `conversation_state.py`（DB を持つ側）
の役目で、ここでは構造だけを見る。**

失敗（例外・timeout・解析失敗・スキーマ違反）は呼び出し側が規則の結果だけで
進められるよう、`InterpretationError` を投げるだけにする。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.llm.base import ChatMessage, LLMClient

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_VALID_WITHDRAW_REASONS = {"reopened", "cancelled", "misdetected", "superseded"}
_VALID_REF_KINDS = {"memory", "message"}


class InterpretationError(RuntimeError):
    """解釈の結果を読み取れなかった。呼び出し側は規則の結果だけで進む。

    `kind` は失敗を数えるための安定した鍵（ISSUE-047）。メッセージには
    モデルの生出力が入るため、そのまま集計の鍵にすると、同じ種類の失敗が
    出力の差だけで別々に数えられ、改行や `|` がレポートの表を壊す。
    """

    def __init__(self, message: str, *, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind


_INSTRUCTION = """あなたは会話ログから、今の会話状態を解釈する担当です。
返答そのものは作りません。渡された「開いている会話状態」「判定対象の質問と
応答候補」「確認待ちの食い違い」「参照できる記憶」をもとに、次のスキーマの
JSON オブジェクトだけを出力してください。説明文やコードブロックは付けません。

{
  "request": "相手が今話したいこと（一文。無ければ null）",
  "answered_state_ids": [判定対象の質問のうち、応答候補が実際に答えているものの id],
  "unanswered_state_ids": [判定対象の質問のうち、応答候補が答えていないものの id],
  "asked_discrepancy_ids": [確認待ちの食い違いのうち、直前の返答が実際に確認したものの id],
  "resolved_discrepancy_ids": [開いている食い違いのうち、今の発言で解消したものの id],
  "withdrawals": [{"state_id": 状態の id, "reason": "reopened|cancelled|misdetected|superseded"}],
  "deferral_topic": "相手が今の発言で新しく後回しにした話題（無ければ null）",
  "is_closing": 相手が今の発言で会話を終えようとしているか（true/false）,
  "confirms_presented_id": "相手が具体的に言い換えて受け止めた presented の id（無ければ null）",
  "correction": {"ref_kind": "memory|message", "ref_id": 対象の id,
                 "content": "訂正後の内容"} または null,
  "discrepancy": {"ref_kind": "memory|message", "ref_id": 対象の id,
                  "note": "食い違いの内容（一文）"} または null
}

守ること:
- id は渡された候補に載っているものだけを使う。無ければ空配列・null にする。
  `ref_kind="message"` の `ref_id` は「直近の会話」の行頭に付いている
  `[id]`（発言そのものの id）か、「開いている会話状態」の「（対象: message N）」
  の N を使う。「参照できる記憶」の `[id]` とは別。状態自体の `[id]` を
  ref_id に書かない。
- reopened は延期・終了の状態にだけ、cancelled は質問にだけ、superseded は
  用件・訂正にだけ使う。
- 強度や自信度は答えない。分からなければ null・空配列のままにする。
- 相づち（「うん」「はい」等の短い相づちだけの発言）からは confirms_presented_id
  を出さない。
- is_closing は今の発言だけで判断する。過去に終了しかけたことは無視する。
- correction と discrepancy を同じ対象に同時に出さない。「直近の会話」の末尾に
  **言い直しの語は含まれていない**と書かれていたら discrepancy にする（相手は
  訂正すると言っていない）。語が含まれている場合は、その語が本当に言い直しを
  指しているかを見て決める。食い違いが無ければどちらも null。
- 開いている correction と**同じ事実**を訂正し直す発言（「日曜でもなくて月曜」等）
  では、新しい correction を出すと同時に、置き換える古い correction の id を
  withdrawals に {"state_id": その id, "reason": "superseded"} として出す。
  新しい correction が無い superseded は受け付けない。
- 応答候補が「分からない」「その情報は無い」と明示的に答えている場合は
  answered_state_ids に含める。答え直す材料が無いので、unanswered にしても
  次の返答で答えられない。
"""


class WithdrawalPayload(BaseModel):
    state_id: int
    reason: str

    @field_validator("reason")
    @classmethod
    def _check_reason(cls, value: str) -> str:
        if value not in _VALID_WITHDRAW_REASONS:
            raise ValueError(f"reason が不正です: {value}")
        return value


class RefPayload(BaseModel):
    ref_kind: str
    ref_id: int

    @field_validator("ref_kind")
    @classmethod
    def _check_ref_kind(cls, value: str) -> str:
        if value not in _VALID_REF_KINDS:
            raise ValueError(f"ref_kind が不正です: {value}")
        return value


class CorrectionPayload(RefPayload):
    content: str = Field(min_length=1, max_length=200)


class DiscrepancyPayload(RefPayload):
    note: str = Field(min_length=1, max_length=200)


class InterpretationResult(BaseModel):
    request: str | None = Field(default=None, max_length=200)
    answered_state_ids: list[int] = Field(default_factory=list)
    unanswered_state_ids: list[int] = Field(default_factory=list)
    asked_discrepancy_ids: list[int] = Field(default_factory=list)
    resolved_discrepancy_ids: list[int] = Field(default_factory=list)
    withdrawals: list[WithdrawalPayload] = Field(default_factory=list)
    deferral_topic: str | None = Field(default=None, max_length=200)
    is_closing: bool = False
    confirms_presented_id: int | None = None
    correction: CorrectionPayload | None = None
    discrepancy: DiscrepancyPayload | None = None

    @field_validator("request", "deferral_topic", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


def parse_interpretation(text: str) -> InterpretationResult:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise InterpretationError(
            f"解釈の結果にJSONオブジェクトが見つかりませんでした: {text[:200]}",
            kind="json_not_found",
        )
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InterpretationError(
            f"解釈の結果をJSONとして読み取れませんでした: {text[:200]}",
            kind="json_decode",
        ) from exc
    if not isinstance(raw, dict):
        raise InterpretationError(
            "解釈の結果がオブジェクトではありませんでした。", kind="not_an_object"
        )
    try:
        return InterpretationResult.model_validate(raw)
    except ValidationError as exc:
        raise InterpretationError(
            f"解釈の結果を読み取れませんでした（{exc.error_count()}件）: {text[:200]}",
            kind="schema",
        ) from exc


async def interpret_conversation(llm: LLMClient, *, context: str) -> InterpretationResult:
    """解釈を1回呼ぶ。timeout は呼び出し側（`conversation.py`）が
    `asyncio.wait_for` で囲む——LLM クライアント自体のタイムアウトより短い
    値を使うため（計画 §9）。
    """
    response = await llm.chat(
        [
            ChatMessage(role="system", content=_INSTRUCTION),
            ChatMessage(role="user", content=context),
        ],
        options={"temperature": 0.2},
    )
    return parse_interpretation(response.text)
