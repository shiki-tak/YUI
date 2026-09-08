"""目標の器（設計書 5「目標」・6 / フェーズ4 PR1）。

ここで確かめるのは、目標を保存し「いま実行してよいもの」を選ぶところまで。
どの行動を選ぶか、いつ完了にするかは後続で足す。

画面と API はまだ無いので、agent の関数を直接呼ぶ。評価と同じ考え方で、
テストのためだけの近道は作らない。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.goal import (
    active_goals,
    create_goal,
    list_goals,
    mark_for_review,
    normalize_due_at,
)
from app.config import LOCAL_TZ
from app.models import (
    Conversation,
    ConversationMode,
    Goal,
    GoalRevision,
    GoalStatus,
    GoalTrigger,
    Speaker,
    Visibility,
)


async def _speaker(session, external_id: str, display_name: str) -> Speaker:
    speaker = Speaker(source="local", external_id=external_id, display_name=display_name)
    session.add(speaker)
    await session.flush()
    return speaker


async def _conversation(session) -> Conversation:
    conversation = Conversation(mode=ConversationMode.LOCAL.value)
    session.add(conversation)
    await session.flush()
    return conversation


async def test_new_goal_is_a_candidate_and_leaves_a_revision(
    session_factory: async_sessionmaker,
) -> None:
    """既定は候補（pending）で、作った記録が残る。

    記憶・関心と同じ流れにする。振り返りが出したものを、開発者が見る前に
    実行しない。
    """
    async with session_factory() as session:
        goal = await create_goal(
            session, content="週末に見た映画の感想を聞く", reason="会話の振り返りから"
        )
        await session.commit()
        goal_id = goal.id

    async with session_factory() as session:
        stored = await session.get(Goal, goal_id)
        assert stored.status == GoalStatus.PENDING.value
        assert stored.trigger == GoalTrigger.NEXT_CONVERSATION.value
        # 実行も達成もしていない。両方 None から始まる。
        assert stored.last_executed_at is None
        assert stored.completed_at is None

        revisions = list(
            (
                await session.execute(
                    select(GoalRevision).where(GoalRevision.goal_id == goal_id)
                )
            ).scalars()
        )
        assert [r.action for r in revisions] == ["created"]
        assert revisions[0].before is None
        assert revisions[0].after["content"] == "週末に見た映画の感想を聞く"
        assert revisions[0].reason == "会話の振り返りから"


async def test_execution_and_completion_are_recorded_separately(
    session_factory: async_sessionmaker,
) -> None:
    """実行したことと、目的を達成したことを分ける（設計書 6）。

    質問を投げても相手が答えなければ、実行済みで未達成である。変更履歴にも
    両方を残し、後から「いつ聞いて、いつ答えが返ったか」を追えるようにする。
    """
    from app.agent.goal import record_revision, snapshot

    executed = datetime(2026, 9, 12, 3, 0, tzinfo=UTC)
    async with session_factory() as session:
        goal = await create_goal(
            session, content="映画の感想を聞く", status=GoalStatus.ACTIVE.value
        )
        before = snapshot(goal)
        goal.last_executed_at = executed
        await session.flush()
        record_revision(session, goal, action="executed", before=before)
        await session.commit()
        goal_id = goal.id

    async with session_factory() as session:
        stored = await session.get(Goal, goal_id)
        # 実行しただけでは達成にしない。完了条件3の判定はここに乗る。
        assert stored.last_executed_at is not None
        assert stored.completed_at is None
        assert stored.status == GoalStatus.ACTIVE.value

        revision = (
            await session.execute(
                select(GoalRevision)
                .where(GoalRevision.goal_id == goal_id, GoalRevision.action == "executed")
            )
        ).scalar_one()
        assert revision.before["last_executed_at"] is None
        assert revision.after["last_executed_at"] is not None
        assert revision.after["completed_at"] is None


async def test_after_date_requires_a_date_and_next_conversation_rejects_one() -> None:
    """実行条件と期限の組み合わせを、作る前に止める。

    after_date に日時が無いと、いつ実行してよいのかが決まらない。
    next_conversation に日時を渡した側は、効くつもりでいる。
    """
    with pytest.raises(ValueError):
        normalize_due_at(GoalTrigger.AFTER_DATE.value, None)
    with pytest.raises(ValueError):
        normalize_due_at(
            GoalTrigger.NEXT_CONVERSATION.value, datetime(2026, 9, 10, tzinfo=UTC)
        )
    with pytest.raises(ValueError):
        normalize_due_at("weekly", None)


async def test_due_at_is_stored_in_utc(session_factory: async_sessionmaker) -> None:
    """地域時刻で渡された期限を、UTC に直して保存する。

    SQLite は timezone を落として保存するため、日本時間のまま入れると、
    読み戻したときに UTC として解釈されて9時間ずれる。記憶の occurred_at で
    同じずれを踏んでいる（ISSUE-003）。
    """
    async with session_factory() as session:
        goal = await create_goal(
            session,
            content="日曜の朝に感想を聞く",
            trigger=GoalTrigger.AFTER_DATE.value,
            # 日本時間 9/13 00:30 は UTC では 9/12 15:30。
            due_at=datetime(2026, 9, 13, 0, 30, tzinfo=LOCAL_TZ),
        )
        await session.commit()
        goal_id = goal.id

    async with session_factory() as session:
        stored = await session.get(Goal, goal_id)
        naive = stored.due_at.replace(tzinfo=None)
        assert naive == datetime(2026, 9, 12, 15, 30)


async def test_only_accepted_goals_are_executable(session_factory: async_sessionmaker) -> None:
    """候補と、再評価の印が付いたものは行動選択に渡さない。"""
    async with session_factory() as session:
        pending = await create_goal(session, content="候補のままの目標")
        accepted = await create_goal(
            session, content="採用した目標", status=GoalStatus.ACTIVE.value
        )
        marked = await create_goal(
            session, content="根拠が変わった目標", status=GoalStatus.ACTIVE.value
        )
        marked.needs_review = True
        await session.commit()

        executable = await active_goals(session, speaker_id=None)
        assert [g.id for g in executable] == [accepted.id]
        assert pending.id not in {g.id for g in executable}


async def test_after_date_waits_for_the_date(session_factory: async_sessionmaker) -> None:
    """指定日以降の目標は、その日時を過ぎてから渡す。

    「予定日後に感想を聞く」を、予定の前に聞かないため。設計書の最初の小さな
    実験がこの経路になる。
    """
    async with session_factory() as session:
        await create_goal(
            session,
            content="映画を見た後で感想を聞く",
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=datetime(2026, 9, 13, 0, 0, tzinfo=UTC),
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        before = await active_goals(
            session, speaker_id=None, now=datetime(2026, 9, 12, 23, 0, tzinfo=UTC)
        )
        assert before == []

        after = await active_goals(
            session, speaker_id=None, now=datetime(2026, 9, 13, 1, 0, tzinfo=UTC)
        )
        assert [g.content for g in after] == ["映画を見た後で感想を聞く"]


async def test_a_goal_about_someone_is_not_executed_with_another_person(
    session_factory: async_sessionmaker,
) -> None:
    """相手が決まっている目標を、別の相手との会話で持ち出さない。

    「この前の映画どうだった？」を別人に聞くと、話していない人の予定を
    持ち出すことになる（設計書 3C）。相手が決まらない目標は誰にでも渡す。
    """
    async with session_factory() as session:
        alice = await _speaker(session, "alice", "アリス")
        bob = await _speaker(session, "bob", "ボブ")
        await create_goal(
            session,
            content="アリスに映画の感想を聞く",
            subject_speaker_id=alice.id,
            visible_to_speaker_id=alice.id,
            status=GoalStatus.ACTIVE.value,
        )
        await create_goal(
            session, content="相手を選ばない話題を出す", status=GoalStatus.ACTIVE.value
        )
        await session.commit()

        for_alice = await active_goals(session, speaker_id=alice.id)
        assert {g.content for g in for_alice} == {
            "アリスに映画の感想を聞く",
            "相手を選ばない話題を出す",
        }

        for_bob = await active_goals(session, speaker_id=bob.id)
        assert {g.content for g in for_bob} == {"相手を選ばない話題を出す"}


async def test_private_goals_are_not_used_in_a_stream(
    session_factory: async_sessionmaker,
) -> None:
    """配信では、非公開の目標を渡さない。記憶・状態と同じ規則。"""
    async with session_factory() as session:
        await create_goal(session, content="個人的な話の続きを聞く", status=GoalStatus.ACTIVE.value)
        await create_goal(
            session,
            content="配信で出してよい話題",
            visibility=Visibility.PUBLIC.value,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        on_stream = await active_goals(
            session, speaker_id=None, mode=ConversationMode.STREAM.value
        )
        assert [g.content for g in on_stream] == ["配信で出してよい話題"]


async def test_correcting_the_basis_marks_the_goal(session_factory: async_sessionmaker) -> None:
    """根拠にした記憶が変わったら、再評価の印を付ける（ISSUE-016）。

    フェーズ3で3回続けて指摘された穴が、目標でもそのまま当てはまる。拾う経路を
    1つずつ足すのではなく、漏れる経路が無いことをここで並べて示す。
    """
    async with session_factory() as session:
        conversation = await _conversation(session)
        other = await _conversation(session)

        by_memory = await create_goal(
            session,
            content="記憶を根拠に持つ目標",
            basis_memory_ids=[7],
            status=GoalStatus.ACTIVE.value,
        )
        without_basis = await create_goal(
            session,
            content="根拠をまだ持たない候補",
            source_conversation_id=conversation.id,
        )
        provisional = await create_goal(
            session,
            content="暫定の根拠しか持たない目標",
            basis_memory_ids=[99],
            basis_is_provisional=True,
            source_conversation_id=conversation.id,
            status=GoalStatus.ACTIVE.value,
        )
        withdrawn = await create_goal(
            session,
            content="撤回中の目標",
            basis_memory_ids=[7],
            status=GoalStatus.WITHDRAWN.value,
        )
        done = await create_goal(
            session,
            content="達成済みの目標",
            basis_memory_ids=[7],
            status=GoalStatus.DONE.value,
        )
        unrelated = await create_goal(
            session,
            content="別の会話から作った目標",
            source_conversation_id=other.id,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        affected = await mark_for_review(
            session,
            memory_id=7,
            reason="根拠にした記憶が訂正されました。",
            source_conversation_id=conversation.id,
        )
        await session.commit()

        marked = {g.id for g in affected}
        # 記憶ID・会話単位（根拠なし）・暫定の根拠・撤回中の4つを拾う。
        assert marked == {by_memory.id, without_basis.id, provisional.id, withdrawn.id}
        # 終わった目標と、別の会話の目標には付けない。
        assert done.id not in marked
        assert unrelated.id not in marked

        # 印が付いている間は行動選択へ渡さない。自動では消さない。
        # 波及しなかった目標は、そのまま実行できる。
        executable = await active_goals(session, speaker_id=None)
        assert [g.id for g in executable] == [unrelated.id]
        pending_review = await list_goals(session, needs_review=True)
        assert len(pending_review) == 4
        assert all(g.review_reason == "根拠にした記憶が訂正されました。" for g in pending_review)

        history = list(
            (
                await session.execute(
                    select(GoalRevision).where(GoalRevision.goal_id == by_memory.id)
                )
            ).scalars()
        )
        assert [r.action for r in history] == ["created", "needs_review"]
        assert history[1].before["needs_review"] is False
        assert history[1].after["needs_review"] is True


async def test_the_same_moment_is_judged_the_same_in_any_timezone(
    session_factory: async_sessionmaker,
) -> None:
    """同じ瞬間なら、渡した時刻の表記が違っても実行してよいかは変わらない。

    第1回レビューの指摘1。due_at は保存時に UTC へ直していたが、比較に使う now
    はそのまま SQL へ渡していた。SQLite は timezone を落とすため、日本時間で
    渡すと9時間先の時刻として比較され、まだ実行してはいけない目標が選ばれた。
    """
    from zoneinfo import ZoneInfo

    due = datetime(2026, 9, 13, 15, 30, tzinfo=UTC)
    async with session_factory() as session:
        await create_goal(
            session,
            content="映画を見た後で感想を聞く",
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=due,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        # 基準日時より前の同じ瞬間。どの表記でも実行してはいけない。
        before = datetime(2026, 9, 13, 15, 0, tzinfo=UTC)
        for moment in (
            before,
            before.astimezone(ZoneInfo("Asia/Tokyo")),
            before.astimezone(ZoneInfo("America/Los_Angeles")),
        ):
            assert await active_goals(session, speaker_id=None, now=moment) == [], moment

        # 基準日時より後の同じ瞬間。どの表記でも実行してよい。
        after = datetime(2026, 9, 13, 16, 0, tzinfo=UTC)
        for moment in (
            after,
            after.astimezone(ZoneInfo("Asia/Tokyo")),
            after.astimezone(ZoneInfo("America/Los_Angeles")),
        ):
            assert len(await active_goals(session, speaker_id=None, now=moment)) == 1, moment

        # 境界。基準の日時ちょうどは「以降」に含める。
        assert len(await active_goals(session, speaker_id=None, now=due)) == 1
        # timezone を持たない時刻は UTC として扱う（保存が UTC のため）。
        assert len(await active_goals(session, speaker_id=None, now=due.replace(tzinfo=None))) == 1


async def test_a_public_goal_about_someone_stays_with_that_person(
    session_factory: async_sessionmaker,
) -> None:
    """配信でも、相手が決まっている目標は、その相手との会話でだけ渡す。

    公開してよいことと、誰に対する目標かは別の軸である。公開の印だけを見ると、
    配信で本人がいないのに「この前の映画どうだった？」を持ち出す。
    """
    async with session_factory() as session:
        alice = await _speaker(session, "alice", "アリス")
        bob = await _speaker(session, "bob", "ボブ")
        await create_goal(
            session,
            content="アリスに配信で感想を聞く",
            subject_speaker_id=alice.id,
            visibility=Visibility.PUBLIC.value,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        on_stream = ConversationMode.STREAM.value
        assert (
            await active_goals(session, speaker_id=None, mode=on_stream)
        ) == []
        assert (
            await active_goals(session, speaker_id=bob.id, mode=on_stream)
        ) == []
        for_alice = await active_goals(session, speaker_id=alice.id, mode=on_stream)
        assert [g.content for g in for_alice] == ["アリスに配信で感想を聞く"]
