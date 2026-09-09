"""会話から、目標（次に何を話したいか）の候補を作る（設計書 5・6 / フェーズ4）。

記憶の抽出とも、関心・関係性の抽出とも**別の呼び出し**にしている。同じ指示文へ
項目を足すと、元からあった抽出が落ちることを実測したため（ISSUE-017：規則を
足して 16/21、組み替えて 13/21、元へ戻して 21/21）。役割ごとに分ければ、片方の
調整がもう片方を壊さない。振り返りはこれで4回目の呼び出しになる。

ここで作るのは候補だけで、採用は開発者が判断する。会話1回で、勝手に「次に
話しかける約束」ができないようにするための境目である。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, time, timedelta

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import LOCAL_TZ
from app.llm.base import ChatMessage, LLMClient
from app.models import Goal, GoalTrigger

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class GoalReflectionError(RuntimeError):
    """目標の候補を読み取れなかった。

    記憶・状態の振り返りと同じ扱いにする。「候補が無かった」と「読み取れ
    なかった」を区別しないと、失敗が「目標なしの成功」として通ってしまう。
    """


_INSTRUCTION = """あなたは会話ログから、キャラクターが「次に何を話したいか」の
候補を抜き出す担当です。記憶や関心の抽出とは別の作業で、ここでは**後で相手に
聞きたいこと・話したいこと**だけを見ます。

出力は JSON 配列です。各要素の形式:
{
  "content": "一文で書いた目的。誰に何を聞くか・話すかが分かるように書く",
  "event_date": "YYYY-MM-DD。その予定が行われる日。次に話すときに聞いて
                 よいなら空文字",
  "reason": "会話のどこからそう考えたか"
}

出すもの:
- 相手がこれからの予定を話したこと。その予定が終わったあとに、どうだったかを
  聞く。例：「今週の土曜に映画を見に行く」→「土曜に見た映画の感想を聞く」。
  event_date には**予定そのものの日**を書く。聞けるのがその翌日からである
  ことは、こちらで数える。**「明日」「今週の土曜」のような言い方は、その
  言い方が出てきた発言の日付から数える。**
- 続きを話すと言ったこと。例：「次はカメラの設定の話をしよう」。
- 相手が気にかけていたことで、次に会ったときに触れたいこと。
  例：「面接が来週ある」→「面接がどうだったかを聞く」。
- 相手が調べたいと言っていたこと。

守ること:
- content は、予定より後に聞くと分かる書き方にする。「見た映画の感想を聞く」の
  ように、済んだこととして書く。
- 1つの目標に目的は1つ。「感想を聞いて、次の予定も聞く」は2つに分ける。
- すでにある目標と同じ内容は出さない。
- 相手が話したことを根拠にする。会話に無い予定を作らない。
- 次に話したいことが無ければ [] とだけ出力する。
- JSON 配列だけを出力し、説明文やコードブロックは付けない。
"""


class GoalPayload(BaseModel):
    """抽出した目標の候補。保存する形にはここで直さない。"""

    content: str = Field(min_length=1, max_length=500)
    event_date: str | None = Field(default=None, max_length=40)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("content", "event_date", "reason", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        # 空白だけの本文を、最小長の検証より前に落とす（ISSUE-001 と同じ）。
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    def schedule(self) -> tuple[str, datetime | None]:
        """実行条件と、その基準日時。

        空欄なら「次の会話」。予定の日が書いてあれば「その**翌日**以降」。
        翌日にする足し算は _parse_event_date が行う（モデルはやらない）。

        **書いてあるのに読めない日付は、ここへは来ない**（読み取りの段階で
        失敗として扱う）。黙って「次の会話」へ落とすと、予定より前に聞いて
        よい目標になり、**制限が緩む方向へずれる**。記憶の occurred_at は
        欠けても意味が変わらないが、実行条件は欠けると意味が変わる
        （第1回レビューの指摘3）。
        """
        if not self.event_date:
            return GoalTrigger.NEXT_CONVERSATION.value, None
        parsed = _parse_event_date(self.event_date)
        if parsed is None:  # pragma: no cover - _parse で弾いている
            raise GoalReflectionError(f"予定の日を読み取れません: {self.event_date}")
        return GoalTrigger.AFTER_DATE.value, parsed


def _parse_event_date(value: str | None) -> datetime | None:
    """予定の日を読み取り、**その翌日**を実行の基準日時にする。

    **翌日にするのはここ（コード）の仕事にする。** 以前はモデルに「予定の翌日
    を書く」と指示していたが、実測では曜日から日付を出すところまでは当たるのに
    +1 日だけを落とし、`relative-date-is-counted-from-the-utterance` が 3 回とも
    予定の日そのものを返した（ISSUE-028）。日付の足し算は必ず同じ答えになるので、
    モデルにやらせる理由が無い。

    日付だけを扱い、地域時刻のその日の始まりを UTC に直す。SQLite は timezone を
    落とすため、他の日時と同じく UTC で保存しないとずれる（ISSUE-003）。

    過去の日付は弾かない。すでに過ぎた予定について「まだ聞けていない」ことは
    あり、その場合は次の会話で実行してよい状態になるのが正しい。
    """
    if not value:
        return None
    # 解析だけでなく、**翌日にする足し算と UTC への変換まで**ここで通す。
    # date.max（9999-12-31）に1日足すと OverflowError になり、抽出の失敗
    # （GoalReflectionError）ではなく素の例外が測定の外まで飛ぶ
    # （PR10 レビューの指摘8）。読み取れない日付として扱う。
    try:
        parsed = date.fromisoformat(value.strip())
        asked_from = parsed + timedelta(days=1)
        return datetime.combine(asked_from, time.min, tzinfo=LOCAL_TZ).astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _parse(text: str) -> list[GoalPayload]:
    match = _JSON_ARRAY.search(text)
    if not match:
        raise GoalReflectionError(f"目標の候補にJSON配列が見つかりませんでした: {text[:200]}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GoalReflectionError(
            f"目標の候補をJSONとして読み取れませんでした: {text[:200]}"
        ) from exc
    if not isinstance(raw, list):
        raise GoalReflectionError("目標の候補が配列ではありませんでした。")

    results: list[GoalPayload] = []
    problems: list[str] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            problems.append(f"{index}件目: 要素がオブジェクトではありません")
            continue
        try:
            payload = GoalPayload.model_validate(item)
        except ValidationError as exc:
            problems.append(f"{index}件目: 項目を読み取れません（{exc.error_count()}件）")
            continue
        # 書いてあるのに読めない日付は、失敗として扱う。空欄（次の会話）と
        # 区別する。読み取れない実行条件を黙って落とすと、予定より前に聞いて
        # よい目標になる（第1回レビューの指摘3）。
        if payload.event_date and _parse_event_date(payload.event_date) is None:
            problems.append(f"{index}件目: 予定の日を読み取れません: {payload.event_date}")
            continue
        results.append(payload)

    if problems:
        raise GoalReflectionError(
            f"目標の候補に読み取れないものが {len(problems)} 件ありました。"
            + "".join(f"\n- {problem}" for problem in problems)
        )
    return results


def format_current_goals(goals: list[Goal]) -> str:
    """いまある目標。同じ内容を重ねて出さないために渡す。"""
    if not goals:
        return "いまの目標: まだありません。"
    lines = ["いまの目標:"]
    for goal in goals:
        lines.append(f"- {goal.content}")
    return "\n".join(lines)


async def propose_goal_candidates(
    *,
    llm: LLMClient,
    transcript: str,
    partner_speaker_id: int | None,
    current_goals: list[Goal],
    today: date,
) -> list[GoalPayload]:
    """会話から目標の候補を作る。**保存はしない。**

    相手を決められない会話（複数の相手がいる、相手の発言が無い）では作らない。
    目標は「誰に聞くか」を持つもので、相手が決まらないまま作ると、別の相手との
    会話で持ち出される（設計書 3C、関心・関係性と同じ扱い）。
    """
    if partner_speaker_id is None:
        return []

    response = await llm.chat(
        [
            ChatMessage(role="system", content=_INSTRUCTION),
            ChatMessage(
                role="user",
                content=(
                    f"{format_current_goals(current_goals)}\n\n"
                    # 相対的な日付は、発言の日付から数える。会話ログの各行に
                    # 日付が入っている。振り返る日を基準にすると、同じ会話を
                    # 別の日に振り返っただけで予定の日付が変わる（記憶の抽出と
                    # 基準が食い違う。第1回レビューの指摘2）。
                    f"振り返りを行っている日: {today.isoformat()}"
                    "（相対的な日付は、その言い方が出てきた発言の日付から数える。"
                    "この日は、予定がもう過ぎたかどうかの判断にだけ使う）\n\n"
                    f"会話:\n{transcript}"
                ),
            ),
        ],
        # 記憶・状態の抽出と同じく、毎回同じ結果になってほしいので温度を下げる。
        options={"temperature": 0.2},
    )

    return _parse(response.text)
