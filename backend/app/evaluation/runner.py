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

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent.character_state import create_state
from app.agent.conversation import ConversationAgent
from app.agent.memory_store import create_memory, get_or_create_speaker
from app.agent.reflection import ReflectionParseError, extract_candidates
from app.config import Settings, to_local
from app.evaluation.scenario import Scenario, TurnSpec
from app.llm.base import LLMClient, LLMError
from app.models import Base, Conversation, ConversationMode, Memory, Speaker, StateStatus
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
    reflection: ReflectionResult | None = None
    # 実際に応答したモデルとその版。レポートの見出しに残し、後から
    # モデルを変えた比較に使う（設計書「モデル・人格・記憶状態の版」）。
    model: str | None = None
    model_digest: str | None = None

    @property
    def ok(self) -> bool:
        if not all(turn.ok for turn in self.turns):
            return False
        return self.reflection is None or self.reflection.ok


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
            or (attempt.reflection is not None and attempt.reflection.error)
        )

    @property
    def total(self) -> int:
        return len(self.attempts)


def _contains_any(text: str, words: list[str]) -> tuple[bool, list[str]]:
    hit = [word for word in words if word in text]
    return bool(hit), hit


def _check_turn(
    spec: TurnSpec,
    reply: str,
    referenced: list[str],
    previous_replies: list[str],
) -> list[Check]:
    checks: list[Check] = []

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
    """シナリオを1回流す。使い捨てのDBを使う。"""
    attempt = Attempt()
    with tempfile.TemporaryDirectory(prefix="yui-eval-") as directory:
        engine = create_async_engine(f"sqlite+aiosqlite:///{directory}/eval.db")
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            factory = async_sessionmaker(engine, expire_on_commit=False)
            agent = ConversationAgent(llm=llm, persona=persona, settings=settings)

            async with factory() as session:
                speakers, memory_keys = await _seed(session, scenario)
                conversation = Conversation(mode=ConversationMode.LOCAL.value)
                session.add(conversation)
                await session.flush()
                await session.commit()

                replies: list[str] = []
                for spec in scenario.turns:
                    assert spec.speaker is not None  # 読み込み時に既定を入れている
                    try:
                        result = await agent.respond(
                            session,
                            conversation=conversation,
                            speaker=speakers[spec.speaker],
                            text=spec.text,
                        )
                        await session.commit()
                    except LLMError as exc:
                        attempt.turns.append(
                            TurnResult(
                                text=spec.text,
                                reply="",
                                human_check=spec.human_check,
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
                    attempt.turns.append(
                        TurnResult(
                            text=spec.text,
                            reply=reply,
                            checks=_check_turn(spec, reply, referenced, replies),
                            referenced=referenced,
                            human_check=spec.human_check,
                        )
                    )
                    replies.append(reply)

                if scenario.reflection is not None and scenario.reflection.run:
                    attempt.reflection = await _run_reflection(
                        session,
                        scenario=scenario,
                        llm=llm,
                        persona=persona,
                        conversation=conversation,
                        speakers=speakers,
                    )
        finally:
            await engine.dispose()
    return attempt


async def _run_reflection(
    session,
    *,
    scenario: Scenario,
    llm: LLMClient,
    persona: Persona,
    conversation: Conversation,
    speakers: dict[str, Speaker],
) -> ReflectionResult:
    spec = scenario.reflection
    assert spec is not None
    try:
        candidates = await extract_candidates(
            session,
            llm=llm,
            conversation=conversation,
            character_name=persona.name,
        )
        await session.commit()
    except (LLMError, ReflectionParseError) as exc:
        return ReflectionResult(error=str(exc))

    result = ReflectionResult(
        # 人が判定するのに要る属性まで出す。種別と本文だけでは、伝聞かどうかや
        # 誰についての記憶かを読み取れない（全体レビューの指摘6）。
        candidates=[
            f"[{c.kind}／{c.provenance}／対象 {c.subject_speaker_id}／根拠 "
            f"#{c.source_message_id}] {c.content}"
            for c in candidates
        ],
        human_check=spec.human_check,
    )
    if spec.expect_empty:
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
    if spec.expect_occurred_at:
        dated = [c for c in candidates if c.occurred_at is not None]
        result.checks.append(
            Check(
                name="出来事の日付",
                ok=bool(dated),
                detail=(
                    "、".join(
                        to_local(c.occurred_at).strftime("%Y-%m-%d") for c in dated
                    )
                    if dated
                    else "会話に日付の手がかりがあるのに、入らなかった"
                ),
            )
        )
    if spec.expect_similar_marked:
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
    if spec.expect_kinds:
        kinds = {c.kind for c in candidates}
        missing = [
            entry for entry in spec.expect_kinds if not (set(entry.split("|")) & kinds)
        ]
        result.checks.append(
            Check(
                name="出てほしい候補の種別",
                ok=not missing,
                detail="期待どおり" if not missing else f"出なかった種別: {'、'.join(missing)}",
            )
        )
    if spec.expect_any:
        joined = "\n".join(c.content for c in candidates)
        ok, hit = _contains_any(joined, spec.expect_any)
        result.checks.append(
            Check(
                name="候補に含まれてほしい語",
                ok=ok,
                detail=(
                    f"一致: {'、'.join(hit)}"
                    if ok
                    else f"どれも含まれない: {'、'.join(spec.expect_any)}"
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
