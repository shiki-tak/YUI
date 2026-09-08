"""実行済みの判定を、再生の状態に結び付ける（フェーズ4 PR8 / ISSUE-011）。

**生成しただけでは実行済みにしない。** 質問文を作っても音声が鳴らなければ、
相手は一度も聞いていない。実行済みにすると、繰り返し防止の裏返しで「相手は
聞いていないのに、二度と聞かれない」ことになる。

中断は途中まで届いているので、単純に除外しない。実行済みにしたうえで、
どこまで届いたかを履歴に残す。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.goal import create_goal
from app.models import Goal, GoalRevision, GoalStatus
from tests.conftest import FakeLLM


async def _setup(client: AsyncClient, session_factory: async_sessionmaker) -> tuple[int, int]:
    """相手と、採用済みの目標を用意して会話IDを返す。"""
    first = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = first.json()["user_message"]["speaker_id"]
    conversation_id = first.json()["conversation_id"]
    async with session_factory() as session:
        goal = await create_goal(
            session,
            content="土曜に見た映画の感想を聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()
        return conversation_id, goal.id


async def _ask(client: AsyncClient, fake_llm: FakeLLM, conversation_id: int) -> int:
    """YUI から質問させ、その発言のIDを返す。"""
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "期限を過ぎた"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    opened = await client.post(f"/api/conversations/{conversation_id}/open")
    return opened.json()["message"]["id"]


async def test_a_generated_question_is_not_executed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """鳴らなかった質問で、目標を実行済みにしない（ISSUE-011）。

    設計書 6：「質問文を生成しただけで、相手から感想を聞けたとは記録しない」。
    """
    conversation_id, goal_id = await _setup(client, session_factory)
    await _ask(client, fake_llm, conversation_id)

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is None
        assert goal.completed_at is None


async def test_a_delivered_question_is_executed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """相手へ届いたら実行済みにする。**達成とは分ける。**"""
    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "completed"}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is not None
        # 質問が届いただけ。相手が答えたかは別で、完了は PR9 が扱う。
        assert goal.completed_at is None
        assert goal.status == GoalStatus.ACTIVE.value

        revisions = list(
            (
                await session.execute(
                    select(GoalRevision).where(GoalRevision.goal_id == goal_id)
                )
            ).scalars()
        )
        executed = [r for r in revisions if r.action == "executed"]
        assert len(executed) == 1
        assert "completed" in executed[0].reason


async def test_an_aborted_question_is_executed_but_recorded_as_aborted(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """**鳴り始めてから**止めた質問は、途中までは届いている。

    単純に除外すると逆に不正確になる（ISSUE-011 の修正方針）。実行済みには
    するが、中断だったことを履歴に残して、完了と混ぜない。
    """
    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "playing"}
    )
    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "aborted"}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is not None
        executed = (
            await session.execute(
                select(GoalRevision).where(
                    GoalRevision.goal_id == goal_id, GoalRevision.action == "executed"
                )
            )
        ).scalar_one()
        assert "aborted" in executed.reason


async def test_starting_to_play_is_not_enough(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """鳴り始めただけでは実行済みにしない。最後まで、または中断まで待つ。"""
    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "playing"}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is None


async def test_only_the_goal_that_was_actually_raised_is_executed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """持ち出した目標だけを実行済みにする。

    渡しただけの目標（待機のときも渡る）を実行済みにすると、聞いていない
    ことを聞いたことにする。
    """
    first = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = first.json()["user_message"]["speaker_id"]
    async with session_factory() as session:
        asked = await create_goal(
            session,
            content="土曜に見た映画の感想を聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        other = await create_goal(
            session,
            content="買ったカメラを使ったか聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()
        asked_id, other_id = asked.id, other.id

    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "こちらを聞く"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    reply = await client.post("/api/chat", json={"text": "ただいま"})
    message_id = reply.json()["reply"]["id"]

    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "completed"}
    )

    async with session_factory() as session:
        assert (await session.get(Goal, asked_id)).last_executed_at is not None
        assert (await session.get(Goal, other_id)).last_executed_at is None


async def test_replaying_does_not_execute_twice(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """聞き直しで実行済みの記録を増やさない。

    すでに話し終えた発言への通知は、開発者が聞き直しているだけとみなす
    （delivery.py の既存の扱い）。実行の履歴も増やさない。
    """
    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    for _ in range(3):
        await client.post(
            f"/api/conversations/messages/{message_id}/delivery",
            json={"state": "completed"},
        )

    async with session_factory() as session:
        executed = list(
            (
                await session.execute(
                    select(GoalRevision).where(
                        GoalRevision.goal_id == goal_id, GoalRevision.action == "executed"
                    )
                )
            ).scalars()
        )
    assert len(executed) == 1


async def test_a_question_stopped_before_playing_is_not_executed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """鳴り始める前に止めた質問は、届いていない（第1回レビューの指摘2）。

    中断を一律で「途中まで届いた」と扱うと、一度も鳴っていない質問まで
    実行済みになる。PR8 の目的そのものに反する。開始時刻の有無で見分ける。
    """
    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    # playing を送らずに中断する。
    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "aborted"}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is None


async def test_a_goal_named_by_a_waiting_decision_is_not_executed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """待機の判断に目標の番号が付いていても、実行済みにしない（指摘4）。

    行動の記録に番号があることと、その目標を話したことは別である。待機・回答・
    調査では、目標を持ち出していない。
    """
    conversation_id, goal_id = await _setup(client, session_factory)

    # 待機なのに目標の番号を返す出力。
    fake_llm.push_action('{"action": "wait", "goal": 1, "reason": "疲れている"}')
    fake_llm.push("お疲れさまでした。ゆっくり休んでくださいね。")
    reply = await client.post(
        "/api/chat", json={"text": "今日は疲れた", "conversation_id": conversation_id}
    )
    message_id = reply.json()["reply"]["id"]

    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "completed"}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is None


async def test_a_failed_goal_update_leaves_the_notice_retryable(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch,
) -> None:
    """目標の更新に失敗したら、再生の状態も進めない（指摘1）。

    先に状態だけ確定させると、送り直しても「もう進んだ」と判断されて、
    目標の更新が永久に飛ぶ。同じトランザクションで確定する。
    """
    from app.agent import delivery as delivery_module

    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    async def failing(*args, **kwargs):
        raise RuntimeError("目標の更新に失敗した")

    monkeypatch.setattr(delivery_module, "_mark_goals_executed", failing)
    with pytest.raises(RuntimeError):
        await client.post(
            f"/api/conversations/messages/{message_id}/delivery",
            json={"state": "completed"},
        )

    monkeypatch.undo()
    # 送り直せば、状態も目標も反映される。
    again = await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "completed"}
    )
    assert again.status_code == 200
    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.last_executed_at is not None


async def test_a_text_only_setup_executes_without_a_notice(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker, monkeypatch
) -> None:
    """読み上げない構成では、画面に出た時点で実行済みにする（指摘3）。

    再生の通知は来ない。通知を待つと、文字だけの構成で目標が永久に実行済みに
    ならない。あとから通知が来ても、記録は増やさない。
    """
    from sqlalchemy import select

    from app.agent import get_agent
    from app.agent.conversation import ConversationAgent
    from app.config import get_settings
    from app.main import app as fastapi_app
    from app.persona import load_persona

    settings = get_settings().model_copy(update={"speech_enabled": False})
    fastapi_app.dependency_overrides[get_agent] = lambda: ConversationAgent(
        llm=fake_llm, persona=load_persona(), settings=settings
    )
    try:
        conversation_id, goal_id = await _setup(client, session_factory)
        message_id = await _ask(client, fake_llm, conversation_id)

        async with session_factory() as session:
            goal = await session.get(Goal, goal_id)
            assert goal.last_executed_at is not None

        # あとから通知が来ても、実行の履歴は増やさない。
        await client.post(
            f"/api/conversations/messages/{message_id}/delivery",
            json={"state": "completed"},
        )
        async with session_factory() as session:
            executed = list(
                (
                    await session.execute(
                        select(GoalRevision).where(
                            GoalRevision.goal_id == goal_id,
                            GoalRevision.action == "executed",
                        )
                    )
                ).scalars()
            )
        assert len(executed) == 1
    finally:
        fastapi_app.dependency_overrides.pop(get_agent, None)


async def test_a_stale_message_does_not_lose_the_execution_record(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """開始と中断の通知が重なっても、実行の記録を落とさない（第2回レビュー）。

    条件付き UPDATE は DB の最新状態を見て成功するが、手元のオブジェクトは
    読み込んだときのままである。開始時刻が入っているのに「鳴り始めていない」と
    判定すると、実際には届いた質問の記録が消える。画面の通知は前の通知の完了を
    待たないので、重なりうる。
    """
    from app.agent.delivery import apply_delivery_state
    from app.models import DeliveryState, Message, utcnow

    conversation_id, goal_id = await _setup(client, session_factory)
    message_id = await _ask(client, fake_llm, conversation_id)

    # 中断の要求が、まだ generated の発言を読み込む。
    async with session_factory() as stale_session:
        stale = await stale_session.get(Message, message_id)
        assert stale.delivery_started_at is None

        # その間に、別の要求が「鳴り始めた」を確定させる。
        async with session_factory() as other:
            playing = await other.get(Message, message_id)
            await apply_delivery_state(
                other, playing, DeliveryState.PLAYING, now=utcnow()
            )

        # 古い発言を持ったまま中断を適用する。
        await apply_delivery_state(
            stale_session, stale, DeliveryState.ABORTED, now=utcnow()
        )

    async with session_factory() as session:
        message = await session.get(Message, message_id)
        assert message.delivery_state == DeliveryState.ABORTED.value
        assert message.delivery_started_at is not None
        goal = await session.get(Goal, goal_id)
        # 鳴り始めた後の中断なので、実行済みになる。
        assert goal.last_executed_at is not None
