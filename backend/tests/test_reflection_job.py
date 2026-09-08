"""振り返りを会話後のジョブへ分けた（フェーズ4 PR5）。

ここで固定するのは、同期で処理していたときとの違いである。

- 終了は待たずに返る。進み具合は記録から見る。
- 失敗は HTTP の応答ではなくジョブの中で起きる。開始権は必ず解放する。
- プロセスが落ちて開始だけが残った会話を、起動で戻す（ISSUE-025）。
"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent import reflection_job
from app.models import Conversation, ReflectionStep, utcnow
from tests.conftest import FakeLLM, end_and_wait, reflection_progress


async def _say(client: AsyncClient, text: str) -> int:
    response = await client.post("/api/chat", json={"text": text})
    return response.json()["conversation_id"]


async def test_end_returns_without_waiting_for_the_model(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """終了は、モデルの応答を待たずに返る。

    振り返りは LLM を3回以上呼ぶため20秒以上かかる。同期で処理すると、その間
    画面には何も出ない（設計書「重い振り返りは会話後のジョブへ分離します」）。
    """
    conversation_id = await _say(client, "今日は写真の話をした")
    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()

    accepted = await client.post(f"/api/conversations/{conversation_id}/end")

    # モデルの応答を待たずに返っている。
    assert accepted.status_code == 202
    body = accepted.json()
    assert body["state"] == "running"
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    fake_llm.gate.set()
    fake_llm.gate = None
    await reflection_job.wait(conversation_id)


async def test_progress_is_readable_while_it_runs(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """どこまで進んだかを、処理中に読める。

    段階ごとの時間差が大きい（実測で拾う3.9秒・選ぶ16.5秒）。まとめて
    「処理中」とだけ出すと、止まっているのか進んでいるのかが分からない。
    """
    conversation_id = await _say(client, "今日は写真の話をした")
    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    # 2回目の呼び出し（選ぶ）で止める。
    fake_llm.gate_on = None
    fake_llm.scripted = ["[]"]

    await client.post(f"/api/conversations/{conversation_id}/end")
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)
    await asyncio.wait_for(fake_llm.held.wait(), timeout=5)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "running"
    assert progress["step"] == ReflectionStep.PICKING.value

    fake_llm.gate.set()
    fake_llm.gate = None
    await reflection_job.wait(conversation_id)

    done = await reflection_progress(client, conversation_id)
    assert done["state"] == "completed"
    # 終わったら段階の表示は消す。残すと、終わったのに処理中に見える。
    assert done["step"] is None
    assert done["error"] is None


async def test_a_failed_job_releases_the_claim_and_keeps_the_reason(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """失敗しても処理中のまま残さない。理由は残す。

    ジョブには呼び出し元がいない。握りつぶすと、会話が「処理中」の表示で
    止まったままになり、開始権も残る。
    """
    conversation_id = await _say(client, "読み取れない出力になる会話")
    fake_llm.push("これは JSON ではありません")

    await end_and_wait(client, conversation_id)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    assert progress["error"]
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        # 開始権は解放され、終了済みにもしない。やり直せる状態。
        assert conversation.reflection_started_at is None
        assert conversation.reflection_completed_at is None
        assert conversation.ended_at is None


async def test_an_unexpected_failure_does_not_leave_it_running(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """想定していない失敗でも、処理中のままにしない。

    抽出の失敗（LLMError・解析失敗）だけを捕まえていると、それ以外の不具合で
    会話が永久に「処理中」になる。実際、発言者の先読みを忘れて同じ状態に
    なった（この PR の実装中）。
    """
    conversation_id = await _say(client, "想定外の失敗をする会話")

    async def broken_chat(*args, **kwargs):
        raise RuntimeError("想定していない不具合")

    fake_llm.chat = broken_chat  # type: ignore[method-assign]
    await end_and_wait(client, conversation_id)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    assert "想定していない不具合" in progress["error"]
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.reflection_started_at is None


async def test_startup_recovers_a_reflection_that_was_left_running(
    client: AsyncClient, session_factory: async_sessionmaker
) -> None:
    """開始だけが残った会話を、起動で戻す（ISSUE-025 の修正方針2）。

    プロセスが落ちると走っていたジョブは消えるが、開始権は DB に残る。放って
    おくと reflection_stale_seconds（既定180秒）を過ぎるまでやり直せない。
    """
    conversation_id = await _say(client, "処理中のまま落ちた会話")
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        conversation.reflection_started_at = utcnow()
        conversation.reflection_step = ReflectionStep.SELECTING.value
        await session.commit()

    recovered = await reflection_job.recover_orphaned(session_factory)
    assert recovered == 1

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    assert "停止" in progress["error"]
    # 期限切れを待たずにやり直せる。
    assert (await client.post(f"/api/conversations/{conversation_id}/end")).status_code == 202
    await reflection_job.wait(conversation_id)


async def test_a_completed_reflection_is_not_recovered(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """終わった会話は、起動の回収で戻さない。"""
    conversation_id = await _say(client, "終わっている会話")
    fake_llm.push("[]")
    await end_and_wait(client, conversation_id)

    assert await reflection_job.recover_orphaned(session_factory) == 0
    assert (await reflection_progress(client, conversation_id))["state"] == "completed"


async def test_a_failure_while_saving_does_not_leave_it_running(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch,
) -> None:
    """保存の段階で落ちても、処理中のまま残さない（第1回レビューの指摘1）。

    抽出だけを try に入れていたため、保存の直前・最中で落ちると開始権も
    「処理中」の表示も残った。画面はポーリングし続け、再試行は回収期限まで
    409 になる。
    """
    conversation_id = await _say(client, "保存で落ちる会話")
    fake_llm.push("[]")

    original = reflection_job._set_step

    async def failing_step(factory, cid: int, step: str) -> None:
        if step == ReflectionStep.SAVING.value:
            raise RuntimeError("保存の直前で落ちた")
        await original(factory, cid, step)

    monkeypatch.setattr(reflection_job, "_set_step", failing_step)
    await end_and_wait(client, conversation_id)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    assert "保存の直前で落ちた" in progress["error"]
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.reflection_started_at is None
        assert conversation.reflection_completed_at is None


async def test_stopping_during_the_save_releases_the_claim(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch,
) -> None:
    """保存の最中に停止されても、開始権を解放する（第1回レビューの指摘1）。

    停止は保存の途中にも起きる。抽出の待ち合わせだけを中断の対象にしていると、
    ここで止まったジョブが処理中のまま残る。
    """
    conversation_id = await _say(client, "保存中に止められる会話")
    fake_llm.push("[]")

    reached = asyncio.Event()
    hold = asyncio.Event()
    original = reflection_job._save

    async def slow_save(*args, **kwargs):
        reached.set()
        await hold.wait()
        await original(*args, **kwargs)

    monkeypatch.setattr(reflection_job, "_save", slow_save)
    accepted = await client.post(f"/api/conversations/{conversation_id}/end")
    assert accepted.status_code == 202
    await asyncio.wait_for(reached.wait(), timeout=5)

    await reflection_job.shutdown(session_factory)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.reflection_started_at is None
        assert conversation.reflection_completed_at is None
    # 期限切れを待たずにやり直せる。
    hold.set()
    monkeypatch.undo()
    fake_llm.scripted = ["[]"]
    assert (await end_and_wait(client, conversation_id)).status_code == 202
    assert (await reflection_progress(client, conversation_id))["state"] == "completed"


async def test_a_completed_reflection_is_not_overwritten_by_a_stop(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch,
) -> None:
    """保存が終わった直後に停止されても、完了を失敗で上書きしない。

    レビューの修正方針にある「すでに完了した結果を失敗へ上書きしない」。
    """
    conversation_id = await _say(client, "保存直後に止められる会話")
    fake_llm.push("[]")

    saved = asyncio.Event()
    hold = asyncio.Event()
    original = reflection_job._save

    async def save_then_wait(*args, **kwargs):
        await original(*args, **kwargs)
        saved.set()
        await hold.wait()

    monkeypatch.setattr(reflection_job, "_save", save_then_wait)
    await client.post(f"/api/conversations/{conversation_id}/end")
    await asyncio.wait_for(saved.wait(), timeout=5)

    await reflection_job.shutdown(session_factory)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "completed"
    assert progress["error"] is None
    hold.set()
