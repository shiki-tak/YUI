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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.character_state import create_state
from app.agent.conversation import ConversationAgent
from app.agent.conversation_state import detect_question
from app.agent.memory_store import create_memory, get_or_create_speaker
from app.agent.reflection import ReflectionParseError
from app.agent.state_reflection import StateReflectionError
from app.config import Settings, to_local
from app.evaluation.scenario import ConversationStateExpectation, Scenario, StepSpec
from app.evaluation.steps import ReflectOutcome, correct_memory, delete_memory, run_reflection
from app.llm.base import LLMClient, LLMError
from app.models import (
    Base,
    CharacterState,
    Conversation,
    ConversationMode,
    ConversationState,
    Memory,
    Speaker,
    StateStatus,
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
# ことにはならない。3つの
# 見方を重ねるが、どれも網羅ではない。
#
# 1. 仮名が1つも無く、かつ中国語の句読点を使っている（_looks_chinese）
#    全文が中国語になる形は、字の一覧に載っていない字だけでも書ける
#    （「你好，我很高兴和你聊天。」は下の一覧に1字も当たらない）。
#    **一覧の取りこぼしを塞ぐのはこちら。**
#
#    **仮名が無いことだけでは落とさない。**「東京都千代田区」のように、住所や
#    名称を漢字だけで答えるのは正しい日本語である。
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

    **仮名が無いことだけでは落とさない**。
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

    **通っても「日本語である」とは言えない。** 検出しなかっただけである。
    名前と detail をその意味に合わせてある。

    **拾えないと分かっている形**:

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


def _matches_conversation_state(
    state: ConversationState, expectation: ConversationStateExpectation
) -> bool:
    if expectation.status is not None and state.status != expectation.status:
        return False
    if expectation.contains and not all(
        word in (state.content or "") for word in expectation.contains
    ):
        return False
    if (
        expectation.followup_needed is not None
        and state.followup_needed != expectation.followup_needed
    ):
        return False
    if expectation.asked is not None and (state.asked_message_id is not None) != expectation.asked:
        return False
    if (
        expectation.responded is not None
        and (state.responded_message_id is not None) != expectation.responded
    ):
        return False
    return not (
        expectation.withdraw_reason is not None
        and state.withdraw_reason != expectation.withdraw_reason
    )


def _check_turn(
    spec: StepSpec,
    reply: str,
    referenced: list[str],
    referenced_states: list[str],
    previous_replies: list[str],
    conversation_states: list[ConversationState],
    run_checks: dict[str, object],
) -> list[Check]:
    checks: list[Check] = []

    # 宣言しなくても、すべての返答で見る（ISSUE-029）。
    language = _check_language(reply)
    if language is not None:
        checks.append(language)

    if spec.expect_memories:
        missing = [key for key in spec.expect_memories if key not in referenced]
        checks.append(
            Check(
                name="渡した記憶",
                ok=not missing,
                detail=("期待どおり" if not missing else f"渡らなかった記憶: {'、'.join(missing)}"),
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
                detail=("期待どおり" if not missing else f"渡らなかった状態: {'、'.join(missing)}"),
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

    if spec.expect_no_new_question:
        # まだ答えていない相手の質問（question_to_yui）が残っているターンは、
        # 答えと新しい質問を機械判定で分けられないため対象外にする
        # （設計 §4 手順6・計画 §7）。ターン完了後に会話状態を読み直すと、
        # その返答自身が `responded_message_id` を埋めてしまい、常に
        # 「未応答は無い」に見える（レビューで実測）。実装
        # （`conversation.py`）が生成前に判定した結果をそのまま使う。
        if run_checks.get("closing_unresolved_due_to_open_question"):
            checks.append(
                Check(
                    name="新しい質問が無い",
                    ok=True,
                    detail="相手の質問が残っているため対象外（人手判定と併用）",
                )
            )
        else:
            ok = not detect_question(reply)
            checks.append(
                Check(
                    name="新しい質問が無い",
                    ok=ok,
                    detail="新しい質問は無い" if ok else "返答に新しい質問が残っている",
                )
            )

    for kind, expectations in spec.expect_conversation_states.items():
        rows = [state for state in conversation_states if state.kind == kind]
        if not expectations:
            checks.append(
                Check(
                    name=f"会話状態：{kind}",
                    ok=not rows,
                    detail="行が無い" if not rows else f"行が {len(rows)} 件ある",
                )
            )
            continue
        missing = [
            index
            for index, expectation in enumerate(expectations)
            if not any(_matches_conversation_state(row, expectation) for row in rows)
        ]
        checks.append(
            Check(
                name=f"会話状態：{kind}",
                ok=not missing,
                detail=(
                    "期待どおり"
                    if not missing
                    else f"条件 {'、'.join(str(i + 1) for i in missing)} に一致する行が無い"
                ),
            )
        )

    return checks


async def _seed(session, scenario: Scenario) -> tuple[dict[str, Speaker], dict[int, str]]:
    """相手と記憶を用意する。記憶IDと鍵の対応も返す。"""
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
    return speakers, memory_keys


async def run_attempt(
    scenario: Scenario, *, llm: LLMClient, persona: Persona, settings: Settings
) -> Attempt:
    """シナリオを1回流す。使い捨てのDBを使う。

    手順（say / reflect / new_conversation / restart / correct_memory /
    delete_memory）を順に実行する。restart では接続を作り直し、記憶が保存から
    読み直されることを確かめられるようにする。
    """
    attempt = Attempt()
    with tempfile.TemporaryDirectory(prefix="yui-eval-") as directory:
        url = f"sqlite+aiosqlite:///{directory}/eval.db"
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            agent = ConversationAgent(llm=llm, persona=persona, settings=settings)

            session = factory()
            speakers, memory_keys = await _seed(session, scenario)
            conversation = await _new_conversation(session)
            replies: list[str] = []

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
                        agent = ConversationAgent(llm=llm, persona=persona, settings=settings)
                        session = factory()
                        speakers = await _reload_speakers(session, scenario)
                        # 記憶IDは再起動で変わらないので、採用したものに付けた
                        # 名前もそのまま使える。事前に入れた記憶だけ読み直す。
                        memory_keys = {
                            **memory_keys,
                            **(await _reload_memory_keys(session, scenario)),
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
                        detail = (
                            f"記憶 #{changed.id}「{changed.content}」"
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
                        continue

                    if step.kind == "reflect":
                        try:
                            outcome = await run_reflection(
                                session,
                                llm=llm,
                                conversation=conversation,
                                character_name=persona.name,
                                step=step,
                            )
                        except (
                            LLMError,
                            ReflectionParseError,
                            StateReflectionError,
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
                        attempt.reflections.append(_check_reflection(step, outcome))
                        continue

                    assert step.text is not None and step.speaker is not None
                    try:
                        result = await agent.respond(
                            session,
                            conversation=conversation,
                            speaker=speakers[step.speaker],
                            text=step.text,
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

                    reply = result.reply_message.content
                    referenced = [
                        memory_keys.get(item.memory.id, f"#{item.memory.id}")
                        for item in result.memories
                    ]
                    referenced_states = await _state_contents(
                        session, result.run.referenced_state_ids
                    )
                    # v0.2：この相手宛の会話状態（全ステータス。expect_conversation_states
                    # は resolved・withdrawn も見られるようにするため、open だけに絞らない）。
                    # `target` を指定していれば、発言した本人（speaker）ではなく
                    # その相手に適用された状態を見る（計画 §7 シナリオ20：
                    # 「A への質問が B の発言では変わらないか」は B の say の
                    # 後に A 宛の状態を見る形にしかならない。レビュー指摘）。
                    state_target = step.target or step.speaker
                    conversation_states = await _conversation_states_for_speaker(
                        session,
                        conversation_id=conversation.id,
                        target_speaker_id=speakers[state_target].id,
                    )
                    attempt.turns.append(
                        TurnResult(
                            text=step.text,
                            reply=reply,
                            checks=_check_turn(
                                step,
                                reply,
                                referenced,
                                referenced_states,
                                replies,
                                conversation_states,
                                result.run.options.get("checks", {}),
                            ),
                            referenced=referenced,
                            referenced_states=referenced_states,
                            human_check=step.human_check,
                        )
                    )
                    replies.append(reply)
            finally:
                await session.close()
        finally:
            await engine.dispose()
    return attempt


async def _state_contents(session: AsyncSession, ids: list[int] | None) -> list[str]:
    """会話へ実際に渡った状態の本文。run_records の記録から引き直す。"""
    if not ids:
        return []
    stmt = select(CharacterState).where(CharacterState.id.in_(ids)).order_by(CharacterState.id)
    return [state.content for state in (await session.execute(stmt)).scalars()]


async def _conversation_states_for_speaker(
    session: AsyncSession, *, conversation_id: int, target_speaker_id: int
) -> list[ConversationState]:
    """その相手宛の会話状態（v0.2）。全ステータスを返す——`expect_conversation_states`
    は `resolved`／`withdrawn` になった後の状態も見られるようにするため、
    `status == open` では絞らない。
    """
    stmt = (
        select(ConversationState)
        .where(
            ConversationState.conversation_id == conversation_id,
            ConversationState.target_speaker_id == target_speaker_id,
        )
        .order_by(ConversationState.id)
    )
    return list((await session.execute(stmt)).scalars())


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


def _check_reflection(step: StepSpec, outcome: ReflectOutcome) -> ReflectionResult:
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
        result.candidates.append(f"（採用した記憶 {len(outcome.accepted)} 件）")
    for state in outcome.states:
        mark = "採用" if state in outcome.accepted_states else "候補"
        result.candidates.append(f"[状態／{state.kind}／{mark}] {state.content}")

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
    if step.expect_provenance:
        # **入手経路そのものを見る**（ISSUE-023）。本文に語が含まれるかだけでは、
        # 伝聞と本人の発言の取り違えが落ちない。取り違えると、次の会話で本人の
        # 発言が「人づてに聞いた」として渡る。
        for word, expected in step.expect_provenance.items():
            hit = [c for c in candidates if word in c.content]
            actual = sorted({c.provenance for c in hit})
            ok = bool(hit) and actual == [expected]
            result.checks.append(
                Check(
                    name=f"入手経路（{word}）",
                    ok=ok,
                    detail=(
                        f"期待どおり: {expected}"
                        if ok
                        else (
                            f"{expected} ではなく {'、'.join(actual)}"
                            if hit
                            else f"「{word}」を含む候補がありません"
                        )
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
        missing = [entry for entry in step.expect_kinds if not (set(entry.split("|")) & kinds)]
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
