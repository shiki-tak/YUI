"""回答・確認質問・話題提案・調査・待機から行動を選ぶ（設計書 6 の 3）。

設計書は「常時LLMを呼び続けず、頻度上限・待機・抑制を実装します」としている。
そのため、**実行できる目標が無いときはモデルを呼ばない**。受け身の会話は
「回答」、自分から始める場面は「待機」で決まるので、判断する余地が無い。

待機も行動として記録する。合格は「よく喋ること」ではなく、適切な待機も測る
対象である（設計書「常に話しかけることを自律性の達成条件にはしません」）。

調査は、フェーズ5A が未実装なので実行しない。選ばれた場合は目標を保留し、
「調べた」と発言させない（設計書 6）。架空の活動日誌も作らない。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.llm.base import ChatMessage, LLMClient
from app.models import Action, Goal, Message, SpeakerKind

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# 返答のときに選べる行動。話題提案は、自分から始める場面のもの。
_REPLY_ACTIONS = {Action.ANSWER.value, Action.ASK.value, Action.RESEARCH.value, Action.WAIT.value}
# 自分から会話を始めるときに選べる行動。ここに「回答」は無い。
_OPEN_ACTIONS = {Action.ASK.value, Action.SUGGEST.value, Action.RESEARCH.value, Action.WAIT.value}

_REPLY_INSTRUCTION = """あなたは、会話を続けるキャラクターの「次の一手」を選ぶ担当です。
相手の発言に答えるのは前提として、**持っている目標に、いま触れてよいか**を判断します。

出力は JSON オブジェクト1つ。形式:
{"action": "answer" | "ask" | "research" | "wait", "goal": 目標の番号 or null,
 "reason": "そう判断した理由"}

選び方:
- ask：いま聞いてよい。相手がその話題に触れている、余裕がありそう、自然に
  つながる。goal に聞く目標の番号を書く。
- answer：目標に触れず、相手の発言に答えるだけにする。話の流れが別のところに
  ある場合。
- wait：相手が疲れている、忙しい、話したくなさそう。今日は聞かない。
- research：自分では答えられず、調べないと分からない目標のとき。

守ること:
- 聞くのは1回に1つだけ。複数の目標をまとめて聞かない。
- 相手の様子を優先する。目標があることは、いま聞いてよい理由にならない。
- JSON だけを出力し、説明文やコードブロックは付けない。
"""

_OPEN_INSTRUCTION = """あなたは、キャラクターが自分から話しかけるかどうかを選ぶ担当です。
前回までの会話と、持っている目標を見て決めます。

出力は JSON オブジェクト1つ。形式:
{"action": "ask" | "suggest" | "research" | "wait", "goal": 目標の番号 or null,
 "reason": "そう判断した理由"}

選び方:
- ask：目標について聞く。goal に聞く目標の番号を書く。
- suggest：目標に沿った話題を出す。goal に元になった目標の番号を書く。
- wait：いま話しかけない。聞くことがない、前回の話の流れから間が悪い場合。
- research：調べないと話せない目標のとき。

守ること:
- 話しかけるのは1つの用件だけ。目標を並べない。
- 話しかけないことも正しい選択である。**目標があることは、話しかけてよい理由に
  ならない。**
- JSON だけを出力し、説明文やコードブロックは付けない。
"""


@dataclass
class ActionChoice:
    """選んだ行動と、その対象、そして何がそれを決めたか。

    判断に使ったモデルと生成設定を持ち帰る。**発言を伴わない判断は
    run_records が作られない**ので、ここで持ち帰らないと、どのモデル・設定が
    待機を選んだのかを後から追えない（第2回レビューの指摘2）。モデルを
    呼ばずに決めた判断は None のままにして、呼んだ判断と区別する。
    """

    action: str
    goal_ids: list[int] = field(default_factory=list)
    reason: str | None = None
    # 出力を読み取れずに待機へ倒したか。正常な待機の判断と区別する。
    is_fallback: bool = False
    provider: str | None = None
    model: str | None = None
    model_digest: str | None = None
    options: dict | None = None

    @property
    def executes_a_goal(self) -> bool:
        """その目標に、いま触れるか。

        調査は選ばれても実行しない（フェーズ5A 未実装）。待機と同じく、目標を
        持ち出さないまま返答する。
        """
        return self.action in {Action.ASK.value, Action.SUGGEST.value}


def format_goals(goals: list[Goal]) -> str:
    lines = []
    for index, goal in enumerate(goals, start=1):
        lines.append(f"{index}. {goal.content}")
    return "\n".join(lines)


def format_recent(history: list[Message], character_name: str, limit: int = 6) -> str:
    """直近のやりとり。相手の様子を読み取るために渡す。"""
    lines = []
    for message in history[-limit:]:
        if message.speaker_kind == SpeakerKind.CHARACTER.value:
            name = character_name
        else:
            name = message.speaker.display_name if message.speaker else "相手"
        lines.append(f"{name}: {message.content}")
    return "\n".join(lines) or "（まだ発言はありません）"


def _parse(text: str, *, allowed: set[str], goals: list[Goal]) -> ActionChoice:
    """選んだ行動を読み取る。読み取れなければ、**触れない側へ倒す。**

    判断できなかったときに聞きに行くと、相手の様子を見ずに質問することになる。
    ここは失敗しても会話を止めない場所なので、例外にはしない。
    """
    unreadable = ActionChoice(
        action=Action.WAIT.value, reason="行動を読み取れませんでした。", is_fallback=True
    )
    match = _JSON_OBJECT.search(text)
    if not match:
        return unreadable
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return unreadable
    if not isinstance(raw, dict):
        return unreadable

    action = raw.get("action")
    # 文字列でない値を集合と比べると、配列・辞書のときに例外になる。ここは
    # 会話を止めない場所なので、型から確かめる（第1回レビューの指摘4）。
    if not isinstance(action, str) or action not in allowed:
        return ActionChoice(
            action=Action.WAIT.value,
            reason=f"選べない行動でした: {action}",
            is_fallback=True,
        )

    reason = raw.get("reason")
    reason = reason.strip() if isinstance(reason, str) else None

    goal_ids: list[int] = []
    index = raw.get("goal")
    # bool は int の一種なので、true が 1 番として通る。除いてから見る。
    if isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= len(goals):
        goal_ids = [goals[index - 1].id]
    elif action in {Action.ASK.value, Action.SUGGEST.value}:
        # 目標を指せていないのに聞く・提案するのは、目標に基づく行動ではない。
        # 根拠を追えないまま話しかけることになるので、触れない側へ倒す。
        return ActionChoice(
            action=Action.WAIT.value,
            reason="どの目標か決められませんでした。",
            is_fallback=True,
        )

    return ActionChoice(action=action, goal_ids=goal_ids, reason=reason)


async def select_action(
    *,
    llm: LLMClient,
    goals: list[Goal],
    history: list[Message],
    character_name: str,
    user_text: str | None,
) -> ActionChoice:
    """次の行動を選ぶ。

    user_text があれば返答の場面、無ければ自分から始める場面として扱う。

    **実行できる目標が無ければモデルを呼ばない。** 返答は「回答」、自分から
    始める場面は「待機」で決まっていて、判断の余地が無い。設計書の
    「常時LLMを呼び続けず」に対応する。
    """
    if not goals:
        return ActionChoice(
            action=Action.ANSWER.value if user_text is not None else Action.WAIT.value,
            reason="実行できる目標がありません。",
        )

    if user_text is not None:
        instruction = _REPLY_INSTRUCTION
        allowed = _REPLY_ACTIONS
        situation = (
            f"直近のやりとり:\n{format_recent(history, character_name)}\n\n"
            f"相手のいまの発言: {user_text}"
        )
    else:
        instruction = _OPEN_INSTRUCTION
        allowed = _OPEN_ACTIONS
        situation = f"前回までのやりとり:\n{format_recent(history, character_name)}"

    response = await llm.chat(
        [
            ChatMessage(role="system", content=instruction),
            ChatMessage(
                role="user",
                content=f"持っている目標:\n{format_goals(goals)}\n\n{situation}",
            ),
        ],
        # 判断は毎回同じであってほしい。抽出と同じく温度を下げる。
        options={"temperature": 0.2},
    )
    choice = _parse(response.text, allowed=allowed, goals=goals)
    choice.provider = response.provider
    choice.model = response.model
    choice.model_digest = response.model_digest
    choice.options = response.options
    return choice
