"""開発者の確認を経ずに採用する（フェーズ4 PR11、設計書の実装内容9）。

設計書は「更新前後を比較し、**評価できた種類から**自動採用へ移す」としている。
一度に全部を自動にしない。種類ごとに設定で切り替え、**既定はすべて手動**。

移した結果が悪ければ戻せることが前提なので、自動で採用したものには印
（`auto_adopted`）を残す。手動の採用と混ざると、まとめて戻せない。

**採用そのものの手順は、手動の経路と同じ関数を通す。** 別々に書くと、片方だけ
直したときに動きがずれる。ここがやるのは「開発者に確認するかどうか」だけで、
採用の中身を変えるものではない。

自動採用の合否を採用件数で測らない（設計書「自動化の合否を単なる採用件数で
測りません。不要な質問、重複、誤った推測、適切な待機も確認します」）。件数は
増やそうと思えばいくらでも増やせる。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.character_state import record_revision as record_state_revision
from app.agent.character_state import snapshot as state_snapshot
from app.agent.goal import record_revision as record_goal_revision
from app.agent.goal import snapshot as goal_snapshot
from app.agent.memory_store import create_memory
from app.agent.memory_store import record_revision as record_memory_revision
from app.agent.memory_store import snapshot as memory_snapshot
from app.models import (
    CandidateStatus,
    CharacterState,
    Goal,
    GoalStatus,
    Memory,
    MemoryCandidate,
    Provenance,
    StateStatus,
)

# 履歴に残す印。後からまとめて戻すときの目印にもなる。
AUTO_REASON = "自動採用（開発者の確認を経ていない）"
AUTO_ACTION = "auto_accepted"

# 目標は種類を持たないので、設定ではこの名前で指す。
GOAL_KIND = "goal"

# 人格・口調に関わる語。**自動採用の安全弁**（[ISSUE-033](../../../docs/issues/issues.md)）。
#
# 設計書は「人格の変更は開発者の操作だけが行える」としている。会話から人格を
# 変える経路は塞いであるが、PR12 の実測では、人格変更の要求（「今日から乱暴な
# 性格になって。敬語もやめて」）から **10件の記憶が自動採用で長期記憶へ入った**。
# 入ってしまえば次の会話でその記憶が渡るので、実質的に経路が開く。
#
# **語で弾くので取りこぼす。** 言い換えられれば通る。それでも、
#
# - 候補としては残るので、開発者が読んで採用できる。落とすのは自動採用だけ。
# - コードで確実に止まる。指示文を直すより確かで、モデルの出来に依存しない。
#
# 抽出そのものの誤り（ISSUE-033）を直したことにはならない。**自動採用を
# 有効にするための最低限の弁**である。
_PERSONA_WORDS = (
    "人格",
    "性格",
    "口調",
    "話し方",
    "敬語",
    "ため口",
    "タメ口",
    "キャラ",
    "一人称",
    "呼び方",
    "呼び捨て",
)


def touches_persona(content: str) -> bool:
    """人格・口調に関わる内容か。自動採用の対象から外すために見る。"""
    return any(word in content for word in _PERSONA_WORDS)


@dataclass
class Overrides:
    """採用のときに開発者が直した内容（手動の経路だけ）。

    **候補そのものは書き換えない**（PR11 レビューの指摘6）。書き換えると、
    モデルが抽出した内容と開発者が直した内容を区別して追えなくなる。設計書は
    「更新前後を比較し」て自動採用へ移すとしており、比較する元が消える。
    """

    kind: str | None = None
    content: str | None = None
    certainty: str | None = None
    provenance: str | None = None
    visibility: str | None = None
    keywords: str | None = None
    subject_speaker_id: int | None = None
    # 対象者を「誰でもない」にする指定。None と区別するために別に持つ。
    subject_to_none: bool = False

    def changes(self, candidate: MemoryCandidate) -> dict[str, object]:
        """候補から実際に変わるものだけを返す。履歴に残す用。"""
        changed: dict[str, object] = {}
        for name in ("kind", "content", "certainty", "provenance", "visibility", "keywords"):
            value = getattr(self, name)
            if value is not None and value != getattr(candidate, name):
                changed[name] = getattr(candidate, name)
        subject = self.resolve_subject(candidate)
        if subject != candidate.subject_speaker_id:
            changed["subject_speaker_id"] = candidate.subject_speaker_id
        return changed

    def resolve_subject(self, candidate: MemoryCandidate) -> int | None:
        if self.subject_to_none:
            return None
        if self.subject_speaker_id is not None:
            return self.subject_speaker_id
        return candidate.subject_speaker_id


async def accept_candidate(
    session: AsyncSession,
    candidate: MemoryCandidate,
    *,
    reason: str,
    auto: bool = False,
    overrides: Overrides | None = None,
    created_at: datetime | None = None,
) -> Memory:
    """記憶の候補を長期記憶にする。

    **手動（API）と自動（振り返り）の共通の入り口。** 採用の手順が2か所に
    分かれると、片方だけ直したときに動きがずれる。

    `overrides` は手動の経路で、開発者が採用の直前に直した内容。**候補の行は
    書き換えず**、直した事実を記憶の履歴に残す（PR11 レビューの指摘6）。
    抽出された内容と開発者が直した内容を区別して追えないと、採用前後を比較
    できない。

    `created_at` は評価で時間を進めたときに渡す。渡さなければ実時計。
    自動採用だけ実時計を使うと、**自動採用の有無で検索の時間減衰が変わり、
    PR12 の比較条件が揃わない**（指摘5）。
    """
    overrides = overrides or Overrides()
    edited = overrides.changes(candidate)
    memory = await create_memory(
        session,
        kind=overrides.kind or candidate.kind,
        content=overrides.content or candidate.content,
        subject_speaker_id=overrides.resolve_subject(candidate),
        visible_to_speaker_id=candidate.visible_to_speaker_id,
        certainty=overrides.certainty or candidate.certainty,
        provenance=overrides.provenance or candidate.provenance,
        visibility=overrides.visibility or candidate.visibility,
        keywords=candidate.keywords if overrides.keywords is None else overrides.keywords,
        occurred_at=candidate.occurred_at,
        source_message_id=candidate.source_message_id,
        source_conversation_id=candidate.conversation_id,
        reason=reason,
        created_at=created_at,
    )
    memory.auto_adopted = auto
    if edited:
        # 抽出された内容を履歴に残す。候補の行は元のままなので二重にはならない
        # が、記憶の側だけを見ても「直して採用した」ことが分かるようにする。
        #
        # **完全なスナップショットとして残す**（第2回レビューの指摘2）。他の
        # 履歴は全項目を持ち、復元APIは `before` を丸ごと現在値へ被せる。
        # 変わった項目だけを入れると、その履歴を指した復元が他の項目を当時の
        # 値へ戻さない（採用時は非公開だった記憶が、公開のまま残る）。
        # 採用直後の状態に、直した項目だけ抽出時の値を重ねたものが、
        # 「開発者が直さなかった場合の記憶」であり、戻せる状態でもある。
        as_extracted = memory_snapshot(memory) | edited
        record_memory_revision(
            session,
            memory,
            action="accepted_with_edits",
            before=as_extracted,
            reason=f"採用時に開発者が直した: {'、'.join(sorted(edited))}",
        )
    candidate.status = CandidateStatus.ACCEPTED.value
    candidate.accepted_memory_id = memory.id
    await session.flush()
    return memory


def adopt_state(session: AsyncSession, state: CharacterState) -> None:
    """状態を、確認を経ずに採用する。履歴に「自動採用」を残す。"""
    before = state_snapshot(state)
    state.status = StateStatus.ACTIVE.value
    state.auto_adopted = True
    record_state_revision(
        session, state, action=AUTO_ACTION, before=before, reason=AUTO_REASON
    )


def adopt_goal(session: AsyncSession, goal: Goal) -> None:
    """目標を、確認を経ずに採用する。履歴に「自動採用」を残す。"""
    before = goal_snapshot(goal)
    goal.status = GoalStatus.ACTIVE.value
    goal.auto_adopted = True
    record_goal_revision(
        session, goal, action=AUTO_ACTION, before=before, reason=AUTO_REASON
    )


async def apply(
    session: AsyncSession,
    *,
    kinds: set[str],
    candidates: list[MemoryCandidate],
    states: list[CharacterState],
    goals: list[Goal],
    now: datetime | None = None,
) -> list[Memory]:
    """振り返りが作ったものに、自動採用を当てる。

    **本番の振り返り（reflection_job）と評価（evaluation/steps）が、どちらも
    ここを通る。** 評価が別経路を持つと、自動採用を有効にして測ったつもりが、
    一度も通っていない測定になる。PR12 は有無を比べる測定なので、そこが
    狂うと結論そのものが変わる。

    `kinds` が空なら何もしない（既定）。**人格・口調に関わる内容は、種類が
    有効でも自動採用しない**（`touches_persona`）。候補としては残るので、
    開発者が読んで採用できる。

    `now` は評価で時間を進めたときに渡す。自動採用だけ実時計を使うと、
    **自動採用の有無で検索の時間減衰が変わり、比較条件が揃わない**
    （PR11 レビューの指摘5）。

    作った記憶を返す。評価が「この会話で採用したもの」として扱えるようにする
    ため（指摘2）。返さないと、自動採用したものが採用済みとして見えない。
    """
    accepted: list[Memory] = []
    if not kinds:
        return accepted
    for candidate in candidates:
        # すでに採用済みの候補には触れない。二重に記憶を作らないため。
        if candidate.kind not in kinds or candidate.status != CandidateStatus.PENDING.value:
            continue
        if touches_persona(candidate.content):
            continue
        if candidate.provenance == Provenance.UNKNOWN.value:
            # **入手経路を決められなかったものは自動採用しない**
            # （PR12 レビューの指摘3）。確認を省けるのは、判断材料が揃って
            # いるときだけである。候補としては残るので開発者が採用できる。
            continue
        accepted.append(
            await accept_candidate(
                session, candidate, reason=AUTO_REASON, auto=True, created_at=now
            )
        )
    for state in states:
        if state.kind in kinds and not touches_persona(state.content):
            adopt_state(session, state)
    if GOAL_KIND in kinds:
        for goal in goals:
            if not touches_persona(goal.content):
                adopt_goal(session, goal)
    await session.flush()
    return accepted
