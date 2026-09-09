"""評価用会話の実行（ISSUE-006）。

シナリオごとに使い捨てのDBを作り、記憶を入れてから会話を流す。通常利用の
会話・記憶は変更しない。

合否を1つの数字にまとめない。同じシナリオを複数回流し、機械で判定できる
観点は「通った回数／試行回数」で出す。生成の揺れと本当の失敗を区別するため
（設計書「複数回の実モデル評価で通常の生成の揺れと区別する」）。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.character_state import create_state
from app.agent.conversation import ConversationAgent
from app.agent.delivery import apply_delivery_state
from app.agent.goal import create_goal
from app.agent.goal_reflection import GoalReflectionError
from app.agent.memory_store import create_memory, get_or_create_speaker
from app.agent.reflection import ReflectionParseError
from app.agent.state_reflection import StateReflectionError
from app.config import Settings, to_local
from app.evaluation.scenario import Scenario, StepSpec
from app.evaluation.steps import (
    ReflectOutcome,
    accept_goal,
    correct_memory,
    delete_memory,
    run_reflection,
)
from app.llm.base import LLMClient, LLMError
from app.models import (
    Base,
    CharacterState,
    Conversation,
    ConversationMode,
    DeliveryState,
    Goal,
    GoalStatus,
    Memory,
    Message,
    Speaker,
    SpeakerKind,
    StateStatus,
    utcnow,
)
from app.persona import Persona


@dataclass
class Check:
    """機械で判定した1項目。"""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class TurnResult:
    text: str
    reply: str
    checks: list[Check] = field(default_factory=list)
    # 渡された記憶。期待と実際の差を読めるように、鍵と本文の両方を残す。
    referenced: list[str] = field(default_factory=list)
    # 渡された可変状態の本文。訂正が状態へ届いたかを、返答の言い回しでは
    # なくプロンプトの中身で見る。
    referenced_states: list[str] = field(default_factory=list)
    # 渡された目標。シナリオの鍵で読む。質問の文面は毎回変わるため、
    # 返答の言い回しではなく、記録に残った参照で判定する。
    referenced_goals: list[str] = field(default_factory=list)
    human_check: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(check.ok for check in self.checks)


@dataclass
class ReflectionResult:
    candidates: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    human_check: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(check.ok for check in self.checks)


@dataclass
class Attempt:
    """シナリオ1回ぶんの実行結果。"""

    turns: list[TurnResult] = field(default_factory=list)
    # 振り返りは複数回あり得る。1つに上書きすると、途中の不合格と根拠が消える
    # （第6回レビューの指摘3）。
    reflections: list[ReflectionResult] = field(default_factory=list)
    # 採用・訂正など、会話以外に行った操作。何をした後の返答かを読めるようにする。
    actions: list[str] = field(default_factory=list)
    # 操作そのものの成否。対象が見つからないまま「訂正後の返答」を測ると、
    # 訂正していない試行が成功に数えられる（第6回レビューの指摘2）。
    action_checks: list[Check] = field(default_factory=list)
    # 実際に応答したモデルとその版。レポートの見出しに残し、後から
    # モデルを変えた比較に使う（設計書「モデル・人格・記憶状態の版」）。
    model: str | None = None
    model_digest: str | None = None

    @property
    def ok(self) -> bool:
        if not all(turn.ok for turn in self.turns):
            return False
        if not all(check.ok for check in self.action_checks):
            return False
        return all(reflection.ok for reflection in self.reflections)


@dataclass
class ScenarioResult:
    scenario: Scenario
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.ok)

    @property
    def human_only(self) -> bool:
        """機械で判定する項目が宣言されていないシナリオ。

        実行できたチェックの数では決めない。モデルの呼び出しが全部失敗した
        シナリオを、人手専用として集計から落とさないため。
        """
        return not self.scenario.has_machine_checks

    @property
    def failed_to_run(self) -> int:
        """モデルの呼び出しなどで実行できなかった試行。"""
        return sum(
            1
            for attempt in self.attempts
            if any(turn.error for turn in attempt.turns)
            or any(reflection.error for reflection in attempt.reflections)
        )

    @property
    def total(self) -> int:
        return len(self.attempts)


def _contains_any(text: str, words: list[str]) -> tuple[bool, list[str]]:
    hit = [word for word in words if word in text]
    return bool(hit), hit


# 返答に日本語以外が混ざっていないかを見る（ISSUE-029）。
#
# **宣言しなくても、すべての返答で見る。** 人が読んで初めて分かる誤りは、
# 気づかれないまま数字だけが良く見える。実測では「印象に残ったのは什么呢？」
# のように部分的に混ざる形と、返答全体が中国語になる形の両方が出た。
#
# **これは言語判定ではなく、検出器である。** 通っても「日本語だと確かめた」
# ことにはならない。この線引きは PR10 レビューの指摘3で明確にした。3つの
# 見方を重ねるが、どれも網羅ではない。
#
# 1. 仮名が1つも無く、かつ中国語の句読点を使っている（_looks_chinese）
#    全文が中国語になる形は、字の一覧に載っていない字だけでも書ける
#    （「你好，我很高兴和你聊天。」は下の一覧に1字も当たらない）。
#    **一覧の取りこぼしを塞ぐのはこちら。**
#
#    **仮名が無いことだけでは落とさない。**「東京都千代田区」のように、住所や
#    名称を漢字だけで答えるのは正しい日本語である（PR10 第2回レビューの指摘1）。
#    全角カンマ「，」を併せて見る。日本語は読点に「、」を使う。
# 2. 日本語で使わない文字体系（ハングル・キリル）
# 3. 簡体字にしか無い字・語（_SIMPLIFIED_ONLY / _SIMPLIFIED_WORDS）
#    部分的に混ざる形は、仮名が十分あるので 1 では拾えない。ここで見る。
#
# **日本語で使う字を1つでも混ぜると、正しい返答を落とす。** 最初の版に
# 「学」「没」「点」「観」「見」「経」を入れて、「文学部に通っているのですが」
# のような普通の日本語が5つの筋書きで落ちた。さらにレビューで「么」（麻雀の
# 么九牌）と「儿」（部首の「ひとあし」）も日本語の用例があると指摘され、
# 外した。**JIS に無いことは、日本語で使われない証明にはならない。** ここに
# 残した字も、固有名詞や引用での使用まで否定できてはいない。
_SIMPLIFIED_ONLY = (
    "这们说请东开发对为过还关"  # 這們説請東開発対為過還関
    "红买卖书车马鸟鱼见觉习长风飞乐师应给"  # 紅買売書車馬鳥魚見覚習長風飛楽師応給
    "样种员级规视华丽历济认识语谁产业务头爱军团电话时间问题实现门图边"
)
# 「么」は単独では落とせない（么九牌）。語として見れば取り違えない。
# 「什么呢」の検出はこちらが引き継ぐ。
_SIMPLIFIED_WORDS = ("什么", "怎么", "这么", "那么", "为什么")
_OTHER_SCRIPTS = (
    ("ハングル", range(0xAC00, 0xD7A4)),
    ("キリル文字", range(0x0400, 0x0500)),
)
_KANA = range(0x3040, 0x3100)
_CJK = range(0x4E00, 0xA000)
# 中国語で使う句読点。日本語の読点は「、」で、「，」は普通の会話では出ない。
_CHINESE_PUNCTUATION = ("，",)


def _looks_chinese(reply: str) -> bool:
    """仮名が1つも無く、中国語の句読点を使っている。

    一覧に載っていない字だけで書かれた中国語を拾うために要る。

    **仮名が無いことだけでは落とさない**（PR10 第2回レビューの指摘1）。
    「東京都千代田区」のように住所・名称を漢字だけで答えるのは正しい日本語で、
    仮名の有無だけで見ると偽の不合格になる。実測ログでは、仮名ゼロの返答は
    4件あって全部が中国語で、**4件とも「，」を含んでいた**。
    """
    if any(ord(char) in _KANA for char in reply):
        return False
    if sum(1 for char in reply if ord(char) in _CJK) < 6:
        return False
    return any(mark in reply for mark in _CHINESE_PUNCTUATION)


def _check_language(reply: str) -> Check | None:
    """返答に日本語以外が混ざっていないか。**宣言せずに、すべての返答で見る。**

    **通っても「日本語である」とは言えない。** 検出しなかっただけである
    （PR10 レビューの指摘3）。名前と detail をその意味に合わせてある。

    **拾えないと分かっている形**（第2回レビューで明示を求められた範囲）:

    - 全文が英語など、漢字も仮名も使わない言語の返答
    - 仮名を含む中国語の混在文のうち、一覧の字も語も使っていないもの

    どちらも人が読む欄（human_check）で見る。ここを増やすと、正しい日本語を
    落とす側の危険が上がる。**判定が落とすのは、落ちるべきものだけにする。**
    """
    if not reply:
        return None
    found = sorted({char for char in reply if char in _SIMPLIFIED_ONLY})
    found += [word for word in _SIMPLIFIED_WORDS if word in reply]
    for name, span in _OTHER_SCRIPTS:
        if any(ord(char) in span for char in reply):
            found.append(name)
    if _looks_chinese(reply):
        found.append("仮名が無く中国語の句読点")
    return Check(
        name="日本語以外の混入",
        ok=not found,
        detail=("検出なし" if not found else f"日本語以外が混ざった: {'、'.join(found)}"),
    )


def _check_turn(
    spec: StepSpec,
    reply: str,
    referenced: list[str],
    referenced_states: list[str],
    referenced_goals: list[str],
    previous_replies: list[str],
    action: str | None = None,
    executed_goals: list[str] | None = None,
    is_fallback: bool = False,
    done_goals: list[str] | None = None,
) -> list[Check]:
    checks: list[Check] = []
    executed_goals = executed_goals or []
    done_goals = done_goals or []

    language = _check_language(reply)
    if language is not None:
        checks.append(language)

    if spec.expect_action:
        # 読み取りに失敗して待機へ倒れた場合を、相手を見て待った判断と区別する。
        # どちらも action は wait なので、印で分ける（PR7 第2回レビューの補足）。
        matched = action == spec.expect_action and not is_fallback
        detail = "期待どおり"
        if action != spec.expect_action:
            detail = f"{spec.expect_action} ではなく {action} を選んだ"
        elif is_fallback:
            detail = f"{action} だが、出力を読み取れずに倒れた結果だった"
        checks.append(Check(name="選んだ行動", ok=matched, detail=detail))

    if spec.expect_memories:
        missing = [key for key in spec.expect_memories if key not in referenced]
        checks.append(
            Check(
                name="渡した記憶",
                ok=not missing,
                detail=(
                    "期待どおり" if not missing else f"渡らなかった記憶: {'、'.join(missing)}"
                ),
            )
        )

    if spec.expect_not_memories:
        leaked = [key for key in spec.expect_not_memories if key in referenced]
        checks.append(
            Check(
                name="渡していない記憶",
                ok=not leaked,
                detail=("期待どおり" if not leaked else f"渡ってしまった: {'、'.join(leaked)}"),
            )
        )

    if spec.expect_states:
        joined = "\n".join(referenced_states)
        missing = [word for word in spec.expect_states if word not in joined]
        checks.append(
            Check(
                name="渡した状態",
                ok=not missing,
                detail=(
                    "期待どおり" if not missing else f"渡らなかった状態: {'、'.join(missing)}"
                ),
            )
        )

    if spec.expect_not_states:
        joined = "\n".join(referenced_states)
        leaked = [word for word in spec.expect_not_states if word in joined]
        checks.append(
            Check(
                name="渡していない状態",
                ok=not leaked,
                detail=("期待どおり" if not leaked else f"渡ってしまった: {'、'.join(leaked)}"),
            )
        )

    if spec.expect_executed_goals or spec.expect_not_executed_goals:
        # 実行済みは、渡ったこととは別に見る。生成しただけの質問を実行済みに
        # しないことを測るために要る（ISSUE-011）。
        missing = [key for key in spec.expect_executed_goals if key not in executed_goals]
        if spec.expect_executed_goals:
            checks.append(
                Check(
                    name="実行済みの目標",
                    ok=not missing,
                    detail=(
                        "期待どおり"
                        if not missing
                        else f"実行済みになっていない: {'、'.join(missing)}"
                    ),
                )
            )
        early = [key for key in spec.expect_not_executed_goals if key in executed_goals]
        if spec.expect_not_executed_goals:
            checks.append(
                Check(
                    name="実行済みでない目標",
                    ok=not early,
                    detail=(
                        "期待どおり"
                        if not early
                        else f"実行済みになってしまった: {'、'.join(early)}"
                    ),
                )
            )

    if spec.expect_done_goals:
        # 達成は、実行済み（届いた）とは別に見る。設計書の小実験は「返答を
        # 受けて完了とし、同じ質問を繰り返さない」までを求める。
        missing = [key for key in spec.expect_done_goals if key not in done_goals]
        checks.append(
            Check(
                name="達成した目標",
                ok=not missing,
                detail=(
                    "期待どおり" if not missing else f"達成になっていない: {'、'.join(missing)}"
                ),
            )
        )

    if spec.expect_goals:
        missing = [key for key in spec.expect_goals if key not in referenced_goals]
        checks.append(
            Check(
                name="渡した目標",
                ok=not missing,
                detail=(
                    "期待どおり" if not missing else f"渡らなかった目標: {'、'.join(missing)}"
                ),
            )
        )

    if spec.expect_not_goals:
        leaked = [key for key in spec.expect_not_goals if key in referenced_goals]
        checks.append(
            Check(
                name="渡していない目標",
                ok=not leaked,
                detail=("期待どおり" if not leaked else f"渡ってしまった: {'、'.join(leaked)}"),
            )
        )

    if spec.expect_any:
        ok, hit = _contains_any(reply, spec.expect_any)
        checks.append(
            Check(
                name="含まれてほしい語",
                ok=ok,
                detail=(
                    f"一致: {'、'.join(hit)}"
                    if ok
                    else f"どれも含まれない: {'、'.join(spec.expect_any)}"
                ),
            )
        )

    if spec.expect_none:
        hit_any, hit = _contains_any(reply, spec.expect_none)
        checks.append(
            Check(
                name="含まれてはいけない語",
                ok=not hit_any,
                detail="含まれていない" if not hit_any else f"含まれた: {'、'.join(hit)}",
            )
        )

    if spec.expect_not_repeating:
        normalized = "".join(reply.split())
        repeated = [
            previous for previous in previous_replies if "".join(previous.split()) == normalized
        ]
        checks.append(
            Check(
                name="繰り返していない",
                ok=not repeated,
                detail="直前までと違う返答" if not repeated else "前と同じ返答をそのまま返した",
            )
        )

    return checks


async def _seed(
    session, scenario: Scenario, *, started: datetime
) -> tuple[dict[str, Speaker], dict[int, str], dict[int, str]]:
    """相手・記憶・目標を用意する。記憶と目標のIDと鍵の対応も返す。"""
    speakers: dict[str, Speaker] = {}
    for spec in scenario.speakers:
        speakers[spec.key] = await get_or_create_speaker(
            session,
            source=spec.source,
            external_id=spec.external_id,
            display_name=spec.display_name,
        )

    for spec in scenario.states:
        await create_state(
            session,
            kind=spec.kind,
            content=spec.content,
            topic=spec.topic,
            subject_speaker_id=speakers[spec.subject].id if spec.subject else None,
            # 会話で使われる状態として置く。採用の流れ自体は pytest で確かめる。
            status=StateStatus.ACTIVE.value,
            reason="評価用会話の準備",
        )

    memory_keys: dict[int, str] = {}
    for spec in scenario.memories:
        memory: Memory = await create_memory(
            session,
            kind=spec.kind,
            content=spec.content,
            subject_speaker_id=speakers[spec.subject].id if spec.subject else None,
            visible_to_speaker_id=speakers[spec.visible_to].id if spec.visible_to else None,
            certainty=spec.certainty,
            visibility=spec.visibility,
            keywords=spec.keywords,
            reason="評価用会話の準備",
        )
        memory_keys[memory.id] = spec.key

    # 記憶の鍵 → 実際のID。目標の根拠を結び付けるのに使う。
    memory_ids = {key: memory_id for memory_id, key in memory_keys.items()}

    goal_keys: dict[int, str] = {}
    for spec in scenario.goals:
        goal: Goal = await create_goal(
            session,
            content=spec.content,
            subject_speaker_id=speakers[spec.subject].id if spec.subject else None,
            # 根拠を結び付けないと、その記憶を訂正しても目標へ波及しない。
            basis_memory_ids=[memory_ids[key] for key in spec.basis] or None,
            trigger=spec.trigger,
            # 期限は会話を始めた時点から数える。シナリオを何日に流しても
            # 同じ結果になるようにするため。
            due_at=(
                started + timedelta(days=spec.due_in_days)
                if spec.due_in_days is not None
                else None
            ),
            # 既定は候補。採用の手順（accept_goal）を通す。
            status=GoalStatus.ACTIVE.value if spec.accepted else GoalStatus.PENDING.value,
            reason="評価用会話の準備",
        )
        goal_keys[goal.id] = spec.key
    return speakers, memory_keys, goal_keys


async def run_attempt(
    scenario: Scenario, *, llm: LLMClient, persona: Persona, settings: Settings
) -> Attempt:
    """シナリオを1回流す。使い捨てのDBを使う。

    手順（say / reflect / new_conversation / restart / correct_memory /
    delete_memory）を順に実行する。restart では接続を作り直し、記憶が保存から
    読み直されることを確かめられるようにする。
    """
    attempt = Attempt()
    # 会話に渡す現在時刻の基準。advance_time で進める分をここへ足す。実時計を
    # 直接使うと、期限の到来をまたぐシナリオが「流した日」で結果を変える。
    started = utcnow()
    clock = timedelta()
    with tempfile.TemporaryDirectory(prefix="yui-eval-") as directory:
        url = f"sqlite+aiosqlite:///{directory}/eval.db"
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            agent = ConversationAgent(llm=llm, persona=persona, settings=settings)

            session = factory()
            speakers, memory_keys, goal_keys = await _seed(session, scenario, started=started)
            conversation = await _new_conversation(session)
            replies: list[str] = []
            # 発言とその時刻。相対的な日付の基準になる（ISSUE-028 の判定）。
            #
            # **最初の発言に固定しない**（PR10 レビューの指摘6）。会話を分けたり
            # 時刻を進めたりすると、別の日の発言から数えて偽の合否になる。
            # 既定は直前の発言で、`expect_goal_due_from` で名指しもできる。
            said_at_log: list[tuple[str, datetime]] = []

            try:
                for step in scenario.effective_steps:
                    if step.kind == "restart":
                        # 接続と会話の担当を作り直し、記憶が保存から読み直される
                        # ことを確かめる。**プロセスの再起動そのものではない**。
                        # 起動時の処理（lifespan、人格の読み込み）は通らないので、
                        # 実プロセス再起動を確認済みとしては扱わない。
                        await session.close()
                        await engine.dispose()
                        engine = create_async_engine(url)
                        factory = async_sessionmaker(engine, expire_on_commit=False)
                        agent = ConversationAgent(
                            llm=llm, persona=persona, settings=settings
                        )
                        session = factory()
                        speakers = await _reload_speakers(session, scenario)
                        # 記憶IDは再起動で変わらないので、採用したものに付けた
                        # 名前もそのまま使える。事前に入れた記憶だけ読み直す。
                        memory_keys = {
                            **memory_keys,
                            **(await _reload_memory_keys(session, scenario)),
                        }
                        goal_keys = {
                            **goal_keys,
                            **(await _reload_goal_keys(session, scenario)),
                        }
                        conversation = await _new_conversation(session)
                        replies = []
                        continue

                    if step.kind == "new_conversation":
                        conversation = await _new_conversation(session)
                        replies = []
                        continue

                    if step.kind in {"correct_memory", "delete_memory"}:
                        assert step.match is not None
                        if step.kind == "correct_memory":
                            assert step.content is not None
                            changed = await correct_memory(
                                session, match=step.match, content=step.content
                            )
                        else:
                            changed = await delete_memory(session, match=step.match)
                        marked = (
                            [goal_keys.get(g.id, f"#{g.id}") for g in changed.marked_goals]
                            if changed
                            else []
                        )
                        detail = (
                            f"記憶 #{changed.id}「{changed.content}」"
                            f"／印が付いた目標: {'、'.join(marked) or 'なし'}"
                            if changed
                            else f"「{step.match}」に当たる記憶が無かった"
                        )
                        attempt.actions.append(f"{step.kind}: {step.match} → {detail}")
                        # 操作できたことを合否に含める。できていない試行を
                        # 「訂正後の返答」の成功に数えない。
                        attempt.action_checks.append(
                            Check(
                                name=f"{step.kind} の実行",
                                ok=changed is not None,
                                detail=detail,
                            )
                        )
                        if step.expect_marked_goals:
                            missing = [
                                key for key in step.expect_marked_goals if key not in marked
                            ]
                            attempt.action_checks.append(
                                Check(
                                    name="再評価の印が付いた目標",
                                    ok=not missing,
                                    detail=(
                                        "期待どおり"
                                        if not missing
                                        else f"印が付かなかった: {'、'.join(missing)}"
                                    ),
                                )
                            )
                        continue

                    if step.kind == "deliver":
                        # 直前の YUI の発言が相手へ届いたことにする。**これを
                        # 書かない限り、発言は生成しただけの状態のまま**で、
                        # 目標も実行済みにならない（ISSUE-011）。
                        last = await _last_character_message(session, conversation.id)
                        if last is None:
                            detail = "届ける発言がありませんでした"
                            ok = False
                        else:
                            await apply_delivery_state(
                                session,
                                last,
                                DeliveryState(step.delivery or "completed"),
                                now=started + clock,
                            )
                            detail = (
                                f"発言 #{last.id} を {step.delivery or 'completed'} にした"
                            )
                            ok = True
                        attempt.actions.append(f"deliver: {detail}")
                        attempt.action_checks.append(
                            Check(name="再生の通知", ok=ok, detail=detail)
                        )
                        continue

                    if step.kind == "accept_goal":
                        assert step.match is not None
                        accepted = await accept_goal(session, match=step.match)
                        for goal in accepted:
                            goal_keys.setdefault(
                                goal.id, step.goal_key or f"採用:{goal.content[:12]}"
                            )
                        detail = (
                            "、".join(f"#{g.id}「{g.content}」" for g in accepted)
                            if accepted
                            else f"「{step.match}」に当たる目標が無かった"
                        )
                        attempt.actions.append(f"accept_goal: {step.match} → {detail}")
                        # 採用できたことを合否に含める。採用されていない試行を
                        # 「採用後の会話」の成功に数えない（correct_memory と同じ）。
                        attempt.action_checks.append(
                            Check(name="accept_goal の実行", ok=bool(accepted), detail=detail)
                        )
                        continue

                    if step.kind == "advance_time":
                        assert step.days is not None
                        clock += timedelta(days=step.days)
                        attempt.actions.append(
                            f"advance_time: {step.days} 日進めた"
                            f"（{to_local(started + clock).strftime('%Y-%m-%d %H:%M')}）"
                        )
                        continue

                    if step.kind == "start_conversation":
                        # YUI の側から会話を始める。話しかけないこともある。
                        # **待機も正しい結果**なので、始まらなかったこと自体は
                        # 失敗にしない。何を選んだかは記録して読めるようにする。
                        conversation = await _new_conversation(session)
                        replies = []
                        speaker = speakers[step.speaker or scenario.speakers[0].key]
                        try:
                            opened = await agent.open(
                                session,
                                conversation=conversation,
                                speaker=speaker,
                                now=started + clock,
                            )
                            await session.commit()
                        except LLMError as exc:
                            attempt.turns.append(
                                TurnResult(
                                    text="（YUI から話しかける）",
                                    reply="",
                                    human_check=step.human_check,
                                    error=str(exc),
                                )
                            )
                            return attempt

                        if attempt.model is None and opened.run is not None:
                            attempt.model = opened.run.model
                            attempt.model_digest = opened.run.model_digest

                        reply = (
                            opened.reply_message.content
                            if opened.reply_message is not None
                            else ""
                        )
                        # 自発発話でも記憶・状態を渡している。空として判定すると、
                        # 渡ってはいけない記憶が渡っても落ちない
                        # （第1回レビューの指摘6）。返答と同じ経路で読む。
                        run = opened.run
                        referenced = [
                            memory_keys.get(memory_id, f"#{memory_id}")
                            for memory_id in (
                                (run.referenced_memory_ids or []) if run is not None else []
                            )
                        ]
                        referenced_states = (
                            await _state_contents(session, run.referenced_state_ids)
                            if run is not None
                            else []
                        )
                        referenced_goals = [
                            goal_keys.get(goal_id, f"#{goal_id}")
                            for goal_id in (
                                (run.referenced_goal_ids or []) if run is not None else []
                            )
                        ]
                        attempt.actions.append(
                            f"start_conversation → {opened.choice.action}"
                            f"（{opened.choice.reason or '理由なし'}）"
                        )
                        attempt.turns.append(
                            TurnResult(
                                text=f"（YUI から話しかける：{opened.choice.action}）",
                                reply=reply,
                                checks=_check_turn(
                                    step,
                                    reply,
                                    referenced,
                                    referenced_states,
                                    referenced_goals,
                                    replies,
                                    opened.choice.action,
                                    await _executed_goal_keys(session, goal_keys),
                                    opened.choice.is_fallback,
                                    await _done_goal_keys(session, goal_keys),
                                ),
                                referenced=referenced,
                                referenced_states=referenced_states,
                                referenced_goals=referenced_goals,
                                human_check=step.human_check,
                            )
                        )
                        if reply:
                            replies.append(reply)
                        continue

                    if step.kind == "reflect":
                        try:
                            outcome = await run_reflection(
                                session,
                                llm=llm,
                                conversation=conversation,
                                character_name=persona.name,
                                step=step,
                                now=started + clock,
                            )
                        except (
                            LLMError,
                            ReflectionParseError,
                            StateReflectionError,
                            GoalReflectionError,
                        ) as exc:
                            # 1回の失敗で評価全体を止めない。失敗として記録し、
                            # この試行だけを終える（第6回レビューの指摘1）。
                            await session.rollback()
                            attempt.reflections.append(ReflectionResult(error=str(exc)))
                            return attempt
                        # 採用した記憶も、シナリオの鍵で読めるようにしておく。
                        # accept_key を書けば、後の say の expect_memories から
                        # 「振り返りで作った記憶が渡ったか」を指せる。
                        for memory in outcome.accepted:
                            memory_keys.setdefault(
                                memory.id, step.accept_key or f"採用:{memory.content[:12]}"
                            )
                        attempt.reflections.append(
                            _check_reflection(
                                step, outcome, said_at=_base_said_at(step, said_at_log)
                            )
                        )
                        continue

                    assert step.text is not None and step.speaker is not None
                    try:
                        result = await agent.respond(
                            session,
                            conversation=conversation,
                            speaker=speakers[step.speaker],
                            text=step.text,
                            now=started + clock,
                        )
                        await session.commit()
                    except LLMError as exc:
                        attempt.turns.append(
                            TurnResult(
                                text=step.text,
                                reply="",
                                human_check=step.human_check,
                                error=str(exc),
                            )
                        )
                        return attempt

                    if attempt.model is None:
                        attempt.model = result.run.model
                        attempt.model_digest = result.run.model_digest
                    said_at_log.append((step.text or "", result.user_message.created_at))

                    reply = result.reply_message.content
                    referenced = [
                        memory_keys.get(item.memory.id, f"#{item.memory.id}")
                        for item in result.memories
                    ]
                    referenced_states = await _state_contents(
                        session, result.run.referenced_state_ids
                    )
                    referenced_goals = [
                        goal_keys.get(goal_id, f"#{goal_id}")
                        for goal_id in (result.run.referenced_goal_ids or [])
                    ]
                    attempt.turns.append(
                        TurnResult(
                            text=step.text,
                            reply=reply,
                            checks=_check_turn(
                                step,
                                reply,
                                referenced,
                                referenced_states,
                                referenced_goals,
                                replies,
                                result.run.selected_action,
                                await _executed_goal_keys(session, goal_keys),
                                result.choice.is_fallback if result.choice else False,
                                await _done_goal_keys(session, goal_keys),
                            ),
                            referenced=referenced,
                            referenced_states=referenced_states,
                            referenced_goals=referenced_goals,
                            human_check=step.human_check,
                        )
                    )
                    replies.append(reply)
            finally:
                await session.close()
        finally:
            await engine.dispose()
    return attempt


async def _last_character_message(
    session: AsyncSession, conversation_id: int
) -> Message | None:
    """その会話で最後に YUI が話した発言。再生の通知を出す先。"""
    stmt = (
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.speaker_kind == SpeakerKind.CHARACTER.value,
        )
        .order_by(Message.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()


async def _done_goal_keys(session: AsyncSession, goal_keys: dict[int, str]) -> list[str]:
    """達成（done）になっている目標。**実行済みとは別に読む。**

    質問が届いたこと（last_executed_at）と、相手が答えたこと（completed_at）を
    混ぜない。設計書の小実験は完了までを求める（PR10 レビューの指摘1）。
    """
    stmt = select(Goal).where(
        Goal.status == GoalStatus.DONE.value, Goal.completed_at.is_not(None)
    )
    return [
        goal_keys.get(goal.id, f"#{goal.id}")
        for goal in (await session.execute(stmt)).scalars()
    ]


async def _executed_goal_keys(
    session: AsyncSession, goal_keys: dict[int, str]
) -> list[str]:
    """実行済みになっている目標。記録から読む。"""
    stmt = select(Goal).where(Goal.last_executed_at.is_not(None))
    return [
        goal_keys.get(goal.id, f"#{goal.id}")
        for goal in (await session.execute(stmt)).scalars()
    ]


async def _state_contents(session: AsyncSession, ids: list[int] | None) -> list[str]:
    """会話へ実際に渡った状態の本文。run_records の記録から引き直す。"""
    if not ids:
        return []
    stmt = select(CharacterState).where(CharacterState.id.in_(ids)).order_by(CharacterState.id)
    return [state.content for state in (await session.execute(stmt)).scalars()]


async def _new_conversation(session: AsyncSession) -> Conversation:
    conversation = Conversation(mode=ConversationMode.LOCAL.value)
    session.add(conversation)
    await session.flush()
    await session.commit()
    return conversation


async def _reload_speakers(session: AsyncSession, scenario: Scenario) -> dict[str, Speaker]:
    """再起動後に、相手を読み直す。"""
    speakers: dict[str, Speaker] = {}
    for spec in scenario.speakers:
        speakers[spec.key] = await get_or_create_speaker(
            session,
            source=spec.source,
            external_id=spec.external_id,
            display_name=spec.display_name,
        )
    await session.commit()
    return speakers


async def _reload_memory_keys(session: AsyncSession, scenario: Scenario) -> dict[int, str]:
    """再起動後に、記憶IDとシナリオの鍵の対応を作り直す。"""
    keys: dict[int, str] = {}
    for spec in scenario.memories:
        stmt = select(Memory).where(Memory.content == spec.content)
        memory = (await session.execute(stmt)).scalars().first()
        if memory is not None:
            keys[memory.id] = spec.key
    return keys


async def _reload_goal_keys(session: AsyncSession, scenario: Scenario) -> dict[int, str]:
    """再起動後に、目標IDとシナリオの鍵の対応を作り直す。"""
    keys: dict[int, str] = {}
    for spec in scenario.goals:
        stmt = select(Goal).where(Goal.content == spec.content)
        goal = (await session.execute(stmt)).scalars().first()
        if goal is not None:
            keys[goal.id] = spec.key
    return keys


def _base_said_at(
    step: StepSpec, said_at_log: list[tuple[str, datetime]]
) -> datetime | None:
    """相対的な日付を数える起点の発言時刻。

    `expect_goal_due_from` があれば、その語を含む**最後の**発言。無ければ
    直前の発言。**シナリオ全体の最初の発言には戻さない**（PR10 レビューの
    指摘6）。会話を分けたり時刻を進めたりすると、別の日から数えてしまう。
    """
    if not said_at_log:
        return None
    if step.expect_goal_due_from:
        for text, said_at in reversed(said_at_log):
            if step.expect_goal_due_from in text:
                return said_at
        return None
    return said_at_log[-1][1]


def _check_reflection(
    step: StepSpec, outcome: ReflectOutcome, said_at: datetime | None = None
) -> ReflectionResult:
    """振り返りの結果を判定する。"""
    if outcome.error:
        return ReflectionResult(error=outcome.error)

    candidates = outcome.candidates
    result = ReflectionResult(
        # 人が判定するのに要る属性まで出す。種別と本文だけでは、伝聞かどうかや
        # 誰についての記憶かを読み取れない（全体レビューの指摘6）。
        candidates=[
            f"[{c.kind}／{c.provenance}／対象 {c.subject_speaker_id}／根拠 "
            f"#{c.source_message_id}] {c.content}"
            for c in candidates
        ],
        human_check=step.human_check,
    )
    if outcome.accepted:
        result.candidates.append(
            f"（採用した記憶 {len(outcome.accepted)} 件）"
        )
    for state in outcome.states:
        mark = "採用" if state in outcome.accepted_states else "候補"
        result.candidates.append(f"[状態／{state.kind}／{mark}] {state.content}")
    for goal in outcome.goals:
        mark = "採用" if goal in outcome.accepted_goals else "候補"
        # 基準日時も出す。本文だけでは、いつから聞ける目標なのかを読み手が
        # 確かめられない（第1回レビュー）。表示は地域時刻に直す。
        due = f"／{to_local(goal.due_at).strftime('%Y-%m-%d %H:%M')}" if goal.due_at else ""
        result.candidates.append(f"[目標／{goal.trigger}{due}／{mark}] {goal.content}")

    if step.expect_goal_any:
        joined = "\n".join(goal.content for goal in outcome.goals)
        ok, hit = _contains_any(joined, step.expect_goal_any)
        result.checks.append(
            Check(
                name="目標の候補",
                ok=ok,
                detail=(
                    f"一致: {'、'.join(hit)}"
                    if ok
                    else f"どれも含まれない: {'、'.join(step.expect_goal_any)}"
                ),
            )
        )

    if step.expect_goal_count_max is not None:
        count = len(outcome.goals)
        result.checks.append(
            Check(
                name="目標の候補の数",
                ok=count <= step.expect_goal_count_max,
                detail=(
                    f"{count} 件（上限 {step.expect_goal_count_max}）"
                    if count <= step.expect_goal_count_max
                    else f"{count} 件で上限 {step.expect_goal_count_max} を超えた"
                ),
            )
        )

    if step.expect_goal_due_days is not None:
        # 発言の日から数えた日数で見る。**本文の語ではなく、日付そのもの。**
        #
        # **判定する目標を1つに決めてから日付を見る**（PR10 レビューの指摘5）。
        # 以前はどれか1つの目標の日付が合えば通していたので、対象の目標の
        # 日付が5日ずれていても、無関係な目標がたまたま合っていれば通った。
        # 本文の一致（expect_goal_any）と日付の一致が別の目標で満たされる。
        expected = (
            to_local(said_at).date() + timedelta(days=step.expect_goal_due_days)
            if said_at is not None
            else None
        )
        targets = outcome.goals
        if step.expect_goal_any:
            targets = [
                goal
                for goal in outcome.goals
                if any(word in goal.content for word in step.expect_goal_any)
            ]
        actual = [
            (goal.content, to_local(goal.due_at).date() if goal.due_at else None)
            for goal in targets
        ]
        # **合格に使った候補を表示する**（PR10 第2回レビューの指摘2）。
        # 先頭を出すと、日付が誤っている候補を「正しかった」と読ませる。
        matched = next(
            (content for content, due in actual if expected is not None and due == expected),
            None,
        )
        result.checks.append(
            Check(
                name="目標の基準日",
                ok=matched is not None,
                detail=(
                    f"期待どおり（{expected}）: {matched}"
                    if matched is not None
                    else (
                        f"{expected} ではなく "
                        + "、".join(f"{due}（{content}）" for content, due in actual)
                        if actual
                        else f"{expected} を判定する目標がありません"
                    )
                ),
            )
        )

    if step.expect_state_any:
        joined = "\n".join(state.content for state in outcome.states)
        ok, hit = _contains_any(joined, step.expect_state_any)
        result.checks.append(
            Check(
                name="関心・関係性の候補",
                ok=ok,
                detail=(
                    f"一致: {'、'.join(hit)}"
                    if ok
                    else f"どれも含まれない: {'、'.join(step.expect_state_any)}"
                ),
            )
        )

    if step.expect_empty:
        result.checks.append(
            Check(
                name="候補が出ないこと",
                ok=not candidates,
                detail=(
                    "候補なし"
                    if not candidates
                    else f"{len(candidates)} 件出た: {'、'.join(c.content for c in candidates)}"
                ),
            )
        )
    if step.expect_occurred_at:
        dated = [c for c in candidates if c.occurred_at is not None]
        result.checks.append(
            Check(
                name="出来事の日付",
                ok=bool(dated),
                detail=(
                    "、".join(to_local(c.occurred_at).strftime("%Y-%m-%d") for c in dated)
                    if dated
                    else "会話に日付の手がかりがあるのに、入らなかった"
                ),
            )
        )
    if step.expect_similar_marked:
        marked = [c for c in candidates if c.similar_memory_ids]
        result.checks.append(
            Check(
                name="近い記憶の印",
                ok=bool(marked),
                detail=(
                    f"{len(marked)} 件に付いた"
                    if marked
                    else "既存の記憶と近いのに、印が付かなかった"
                ),
            )
        )
    if step.expect_kinds:
        kinds = {c.kind for c in candidates}
        missing = [
            entry for entry in step.expect_kinds if not (set(entry.split("|")) & kinds)
        ]
        result.checks.append(
            Check(
                name="出てほしい候補の種別",
                ok=not missing,
                detail="期待どおり" if not missing else f"出なかった種別: {'、'.join(missing)}",
            )
        )
    if step.expect_candidate_any:
        joined = "\n".join(c.content for c in candidates)
        ok, hit = _contains_any(joined, step.expect_candidate_any)
        result.checks.append(
            Check(
                name="候補に含まれてほしい語",
                ok=ok,
                detail=(
                    f"一致: {'、'.join(hit)}"
                    if ok
                    else f"どれも含まれない: {'、'.join(step.expect_candidate_any)}"
                ),
            )
        )
    return result


async def run_scenarios(
    scenarios: list[Scenario],
    *,
    llm: LLMClient,
    persona: Persona,
    settings: Settings,
    repeat: int = 1,
    on_progress=None,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        result = ScenarioResult(scenario=scenario)
        for attempt_index in range(repeat):
            if on_progress is not None:
                on_progress(scenario, attempt_index + 1, repeat)
            result.attempts.append(
                await run_attempt(scenario, llm=llm, persona=persona, settings=settings)
            )
        results.append(result)
    return results
