"""記憶から派生したものへの波及（ISSUE-016）。

記憶を訂正・削除・復元すると、その記憶を根拠に作ったもの（関心・関係性、
目標）は古い前提のまま残る。ここは**派生したものの一覧を1か所に集める**
場所で、呼ぶ側が種類を数え上げなくてよいようにする。

フェーズ3では、この波及の漏れをレビューで3回続けて指摘された。漏れた形は
どれも「経路を1つずつ塞いだ結果、後から足した経路がどれかを呼んでいない」
だった。フェーズ4で目標が増え、この先も派生するものは増える。呼ぶ側に
列挙させると、また同じ漏れ方をする。

自動では直さない。印が付いている間は会話・行動選択へ渡さず、訂正が派生先
にも及ぶのかは開発者が判断する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import character_state, goal
from app.models import CharacterState, Goal


@dataclass
class DerivedMarks:
    """印を付けたもの。呼ぶ側が、何に及んだかを画面や記録へ出せるようにする。"""

    states: list[CharacterState] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)


async def mark_derived_for_review(
    session: AsyncSession,
    *,
    memory_id: int,
    reason: str,
    source_conversation_id: int | None = None,
) -> DerivedMarks:
    """その記憶を根拠にした派生物すべてへ、再評価の印を付ける。

    記憶を訂正・削除・復元する経路は、ここだけを呼ぶ。個々の判定規則
    （記憶ID・会話単位・暫定の根拠・撤回中の扱い）は、それぞれの
    mark_for_review にある。
    """
    return DerivedMarks(
        states=await character_state.mark_for_review(
            session,
            memory_id=memory_id,
            reason=reason,
            source_conversation_id=source_conversation_id,
        ),
        goals=await goal.mark_for_review(
            session,
            memory_id=memory_id,
            reason=reason,
            source_conversation_id=source_conversation_id,
        ),
    )
