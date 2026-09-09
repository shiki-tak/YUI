"""会話後の振り返りを、応答を待たせないジョブとして走らせる（フェーズ4 PR5）。

設計書：「重い振り返りは会話後のジョブへ分離します。初期は単一プロセスの
管理可能なジョブで十分です。」そのとおり、**単一プロセス内の asyncio タスク**
で実装する。ジョブ基盤（arq など）は入れない。プロセスをまたぐ場合の制約は
ISSUE-025 に記録してある。

進行はプロセス内の変数ではなく DB（conversations.reflection_step）に置く。
別プロセスへ移すときに、進行の見え方を作り直さずに済むため。

開始権（reflection_started_at + conversation_locks）の仕組みは変えない。
API 側で開始権を取ってからジョブを積み、ジョブは自分の接続で処理する。
ここはフェーズ3のレビュー指摘1で直した箇所で、人格の読み込みが失敗すると
開始権を握ったまま詰まる不具合があった。失敗したら必ず開始権を解放する。
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import selectinload

from app.agent import auto_adopt
from app.agent.character_state import active_states, create_state
from app.agent.goal import create_goal, list_goals
from app.agent.goal_reflection import GoalReflectionError, propose_goal_candidates
from app.agent.reflection import (
    ReflectionParseError,
    extract_candidates,
    format_transcript,
)
from app.agent.state_reflection import StateReflectionError, propose_state_candidates
from app.config import LOCAL_TZ, get_settings
from app.llm.base import LLMClient, LLMError
from app.models import (
    Conversation,
    GoalStatus,
    Message,
    ReflectionStep,
    Speaker,
    SpeakerKind,
    StateKind,
    StateStatus,
    utcnow,
)

logger = logging.getLogger(__name__)

# 走っているジョブ。会話IDで引ける。停止のときに待ち合わせるために持つ。
_running: dict[int, asyncio.Task] = {}


def is_running(conversation_id: int) -> bool:
    task = _running.get(conversation_id)
    return task is not None and not task.done()


async def _conversation_messages(session, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        # 誰の発言かを会話ログに書くため、話者を一緒に読む。遅延読み込みに
        # 任せると、セッションを閉じた後に触れて落ちる。
        .options(selectinload(Message.speaker))
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars())


async def _sole_partner_id(session, conversation_id: int) -> int | None:
    """その会話にひとりだけいる相手。複数いれば None。

    「最後に話した人」を相手として扱うと、複数の相手がいる会話で別人の情報を
    その人のものとして保存する（フェーズ3全体レビューの指摘1）。
    """
    stmt = (
        select(Speaker.id)
        .join(Message, Message.speaker_id == Speaker.id)
        .where(
            Message.conversation_id == conversation_id,
            Message.speaker_kind == SpeakerKind.USER.value,
        )
        .distinct()
        .limit(2)
    )
    found = list((await session.execute(stmt)).scalars())
    return found[0] if len(found) == 1 else None


async def _set_step(factory: async_sessionmaker, conversation_id: int, step: str) -> None:
    """進行を記録する。**短いトランザクションで書いてすぐ閉じる。**

    SQLite は書き込みロックを1つしか持てない。モデルの応答を待つ間ロックを
    保持すると、別の会話や記憶の訂正が「database is locked」で失敗する。
    """
    async with factory() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(reflection_step=step)
        )
        await session.commit()


async def _release(factory: async_sessionmaker, conversation_id: int, error: str) -> None:
    """失敗を記録し、開始権を解放する。

    やり直せる状態に戻すのが目的。理由を残すのは、開始だけが残った状態と、
    失敗して終わった状態を画面で区別するため。
    """
    async with factory() as session:
        await session.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                # 完了したものを失敗で上書きしない。保存を commit した直後に
                # 停止されると、終わっているのに失敗として残る
                # （第1回レビューの指摘1）。
                Conversation.reflection_completed_at.is_(None),
            )
            .values(reflection_started_at=None, reflection_step=None, reflection_error=error)
        )
        await session.commit()


async def run_reflection(
    factory: async_sessionmaker,
    *,
    conversation_id: int,
    llm: LLMClient,
    character_name: str,
) -> None:
    """振り返りを1回分実行する。開始権はすでに取られている前提。

    **抽出だけでなく、進行の記録と保存も含めて失敗と中断を受け止める。**
    ジョブには呼び出し元がいないので、ここで握らないと会話が「処理中」の表示で
    止まったままになり、開始権も残る。保存の直前で落ちる経路を、第1回レビューの
    指摘1で突かれた。
    """
    try:
        await _run(
            factory,
            conversation_id=conversation_id,
            llm=llm,
            character_name=character_name,
        )
    except asyncio.CancelledError:
        # 停止のときは開始権を解放してから通す。握ったままだと、次に起動しても
        # reflection_stale_seconds を過ぎるまでやり直せない。
        await _release(factory, conversation_id, "処理が中断されました。")
        raise
    except (
        LLMError,
        ReflectionParseError,
        StateReflectionError,
        GoalReflectionError,
    ) as exc:
        # 抽出できなかった会話を処理中のまま残すと、やり直せない。特に出力の
        # 解析失敗は「候補なしの成功」と区別する必要がある。
        logger.warning("会話 #%s の振り返りに失敗しました: %s", conversation_id, exc)
        await _release(factory, conversation_id, str(exc))
    except Exception as exc:
        logger.exception("会話 #%s の振り返りが想定外の失敗をしました。", conversation_id)
        await _release(factory, conversation_id, f"振り返りに失敗しました: {exc}")


async def _run(
    factory: async_sessionmaker,
    *,
    conversation_id: int,
    llm: LLMClient,
    character_name: str,
) -> None:
    """抽出から保存まで。失敗と中断の扱いは呼び出し側に集める。

    保存は、すべての抽出が成功してからまとめて行う。途中まで保存すると、
    再試行で二重に保存される（フェーズ3全体レビューの指摘4・5）。
    """
    async with factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise ReflectionParseError("会話が見つかりません。")
        partner_id = await _sole_partner_id(session, conversation_id)
        messages = await _conversation_messages(session, conversation_id)
        current_states = await active_states(
            session, speaker_id=partner_id, mode=conversation.mode
        )
        # いまある目標。同じ内容を重ねて出さないために渡す。終わったものは
        # 除く（達成した目標を、もう一度作らせないため）。
        current_goals = [
            goal
            for goal in await list_goals(session, subject_speaker_id=partner_id)
            if goal.status in {GoalStatus.PENDING.value, GoalStatus.ACTIVE.value}
        ]
        transcript = format_transcript(messages, character_name)

    async def step(name: str) -> None:
        await _set_step(factory, conversation_id, name)

    # モデルを呼んでいる間は、書き込みのトランザクションを開かない。
    async with factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        candidates = await extract_candidates(
            session,
            llm=llm,
            conversation=conversation,
            character_name=character_name,
            on_step=step,
        )

    await step(ReflectionStep.STATES.value)
    # 関心・関係性は別の呼び出しで作る。同じ指示文へ項目を足すと記憶の
    # 抽出が落ちる（ISSUE-017 で実測）。
    state_payloads = await propose_state_candidates(
        llm=llm,
        transcript=transcript,
        partner_speaker_id=partner_id,
        current_states=current_states,
    )

    await step(ReflectionStep.GOALS.value)
    # 目標も別の呼び出しで作る。記憶・関心と混ぜない（ISSUE-017 の教訓）。
    goal_payloads = await propose_goal_candidates(
        llm=llm,
        transcript=transcript,
        partner_speaker_id=partner_id,
        current_goals=current_goals,
        today=utcnow().astimezone(LOCAL_TZ).date(),
    )

    await step(ReflectionStep.SAVING.value)
    await _save(
        factory, conversation_id, candidates, state_payloads, goal_payloads, partner_id
    )
    logger.info(
        "会話 #%s の振り返りが完了しました（候補 %s 件）", conversation_id, len(candidates)
    )


async def _save(
    factory: async_sessionmaker,
    conversation_id: int,
    candidates: list,
    state_payloads: list,
    goal_payloads: list,
    partner_id: int | None,
) -> None:
    """抽出がすべて成功したものを、1つのトランザクションで書く。

    設定で自動採用を有効にした種類は、**候補で止めずにその場で採用する**
    （フェーズ4 PR11）。既定は空なので、何も指定しなければ従来どおり全部が
    候補のまま開発者の確認を待つ。

    採用まで同じトランザクションで確定する。候補だけ書いて採用が落ちると、
    「有効にしたのに候補のまま」という中途半端な状態が残る。
    """
    auto_kinds = get_settings().auto_adopt_kinds
    async with factory() as session:
        session.add_all(candidates)
        # 候補に id を振ってから採用する。採用は accepted_memory_id を書く。
        await session.flush()
        states = []
        goals = []
        for payload in state_payloads:
            is_relationship = payload.kind == StateKind.RELATIONSHIP.value
            state = await create_state(
                session,
                kind=payload.kind,
                content=payload.content,
                topic=None if is_relationship else payload.topic,
                subject_speaker_id=partner_id if is_relationship else None,
                # 非公開の会話から作った状態は、その相手との会話に限る。
                visible_to_speaker_id=partner_id,
                source_conversation_id=conversation_id,
                status=StateStatus.PENDING.value,
                reason=payload.reason or "会話の振り返りから",
            )
            states.append(state)
        for goal_payload in goal_payloads:
            trigger, due_at = goal_payload.schedule()
            goal = await create_goal(
                session,
                content=goal_payload.content,
                subject_speaker_id=partner_id,
                trigger=trigger,
                due_at=due_at,
                # 非公開の会話から作った目標は、その相手との会話に限る。
                visible_to_speaker_id=partner_id,
                source_conversation_id=conversation_id,
                status=GoalStatus.PENDING.value,
                reason=goal_payload.reason or "会話の振り返りから",
            )
            goals.append(goal)
        # 自動採用は、評価と同じ関数を通す（フェーズ4 PR11）。
        await auto_adopt.apply(
            session, kinds=auto_kinds, candidates=candidates, states=states, goals=goals
        )
        completed = utcnow()
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                reflection_completed_at=completed,
                ended_at=completed,
                reflection_step=None,
                reflection_error=None,
            )
        )
        await session.commit()


def schedule(
    factory: async_sessionmaker,
    *,
    conversation_id: int,
    llm: LLMClient,
    character_name: str,
) -> asyncio.Task:
    """振り返りのジョブを積む。開始権を取った後に呼ぶ。"""

    async def runner() -> None:
        try:
            await run_reflection(
                factory,
                conversation_id=conversation_id,
                llm=llm,
                character_name=character_name,
            )
        finally:
            _running.pop(conversation_id, None)

    task = asyncio.create_task(runner(), name=f"reflection-{conversation_id}")
    _running[conversation_id] = task
    return task


async def wait(conversation_id: int) -> None:
    """その会話の振り返りが終わるまで待つ。

    停止処理とテストで使う。ジョブは失敗しても例外を投げずに終わる（失敗は
    conversations.reflection_error に残す）ので、待つ側は結果を DB から読む。
    """
    task = _running.get(conversation_id)
    if task is not None:
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))


async def shutdown(factory: async_sessionmaker | None = None) -> None:
    """走っているジョブを止める。停止のときに呼ぶ。

    途中で終わったジョブは開始権を解放してから終わる（run_reflection の
    CancelledError の扱い）。握ったまま落ちると、次の起動で回収するまで
    やり直せない。

    factory を渡すと、止めたあとに残っている開始権をまとめて回収する。
    取り消しの最中に DB を書けなかった場合の取りこぼしを拾うため。停止の
    直後に落ちても、起動時の回収（recover_orphaned）が同じ処理をする。
    """
    tasks = [task for task in _running.values() if not task.done()]
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("振り返りのジョブが例外で終わりました。")
    _running.clear()
    if factory is not None:
        await recover_orphaned(factory)


async def recover_orphaned(factory: async_sessionmaker) -> int:
    """開始だけが残った振り返りを、やり直せる状態に戻す。起動時に呼ぶ。

    プロセスが落ちると、走っていたジョブは消えるが開始権は DB に残る。放って
    おくと reflection_stale_seconds（既定180秒）を過ぎるまで再試行できない。

    **単一プロセスで動かす前提の処理である。** 複数プロセスで動かすと、起動した
    側が別プロセスの実行中のジョブを解放してしまう（ISSUE-025）。
    """
    async with factory() as session:
        result = await session.execute(
            update(Conversation)
            .where(
                Conversation.reflection_started_at.is_not(None),
                Conversation.reflection_completed_at.is_(None),
            )
            .values(
                reflection_started_at=None,
                reflection_step=None,
                reflection_error="処理の途中で停止しました。もう一度実行してください。",
            )
        )
        await session.commit()
    count = result.rowcount or 0
    if count:
        logger.info("開始だけが残っていた振り返りを %s 件戻しました。", count)
    return count
