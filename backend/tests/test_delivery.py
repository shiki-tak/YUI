"""音声の取得・再生状態・待ち時間の記録（v0.1）。

v0.1 の完了条件のうち、次を対象にする。

- 文字入力から返答生成、音声の取得まで一通り動く。
- 返答を重複再生せず、停止操作で音声を止められる（記録の側から確かめる）。
- 読み上げた内容と字幕が一致している（合成に渡す文章が発言の本文と同じ）。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app import main
from app.agent.delivery import _delivered_char_count, apply_delivery_state
from app.models import DeliveryState, Message, utcnow
from tests.conftest import FakeSpeech


async def _say(client: AsyncClient, text: str, conversation_id: int | None = None) -> dict:
    payload: dict = {"text": text}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


async def _delivery(
    client: AsyncClient, message_id: int, state: str, *, progress: float | None = None
):
    payload: dict = {"state": state}
    if progress is not None:
        payload["progress"] = progress
    return await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json=payload
    )


async def test_reply_starts_as_generated_when_speech_is_on(client: AsyncClient):
    """読み上げる構成では、生成しただけの状態から始まる。"""
    result = await _say(client, "こんばんは")
    assert result["reply"]["delivery_state"] == "generated"
    assert result["reply"]["delivery_started_at"] is None
    assert result["reply"]["delivery_finished_at"] is None
    # 相手の発言は届いている。読み上げの対象ではない。
    assert result["user_message"]["delivery_state"] == "completed"


async def test_reply_is_delivered_immediately_without_speech(
    client: AsyncClient, monkeypatch
):
    """読み上げない構成では、画面に出た時点で届いた扱いにする。"""
    monkeypatch.setattr(main.settings, "speech_enabled", False)

    result = await _say(client, "こんばんは")
    assert result["reply"]["delivery_state"] == "completed"
    assert result["reply"]["delivery_finished_at"] is not None


async def test_speech_is_synthesized_from_the_reply_text(
    client: AsyncClient, fake_llm, fake_speech: FakeSpeech
):
    """読み上げる文章は発言の本文そのもの。字幕と一致させるため。"""
    fake_llm.push("こんばんは。今日はどんな一日でしたか？")
    result = await _say(client, "ただいま")

    response = await client.get(
        f"/api/conversations/messages/{result['reply']['id']}/speech"
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.content
    assert fake_speech.calls == [("こんばんは。今日はどんな一日でしたか？", None)]


async def test_user_message_is_not_spoken(client: AsyncClient):
    result = await _say(client, "こんばんは")
    response = await client.get(
        f"/api/conversations/messages/{result['user_message']['id']}/speech"
    )
    assert response.status_code == 400


async def test_missing_message_has_no_speech(client: AsyncClient):
    assert (await client.get("/api/conversations/messages/999/speech")).status_code == 404


async def test_speech_failure_is_reported_without_losing_the_reply(
    client: AsyncClient, fake_speech: FakeSpeech
):
    """音声が出せなくても、返答そのものは残る。"""
    fake_speech.fail = True
    result = await _say(client, "こんばんは")

    response = await client.get(
        f"/api/conversations/messages/{result['reply']['id']}/speech"
    )
    assert response.status_code == 503

    stored = await client.get(f"/api/conversations/messages/{result['reply']['id']}")
    assert stored.status_code == 200
    assert stored.json()["content"] == result["reply"]["content"]


async def test_speech_is_refused_when_disabled(client: AsyncClient, monkeypatch):
    monkeypatch.setattr(main.settings, "speech_enabled", False)
    result = await _say(client, "こんばんは")
    response = await client.get(
        f"/api/conversations/messages/{result['reply']['id']}/speech"
    )
    assert response.status_code == 503


async def test_playback_is_recorded_from_start_to_finish(client: AsyncClient):
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    playing = await _delivery(client, message_id, "playing")
    assert playing.status_code == 200, playing.text
    assert playing.json()["delivery_state"] == "playing"
    started = playing.json()["delivery_started_at"]
    assert started is not None
    assert playing.json()["delivery_finished_at"] is None

    completed = await _delivery(client, message_id, "completed")
    assert completed.json()["delivery_state"] == "completed"
    # 開始時刻は最初の通知のまま残る。
    assert completed.json()["delivery_started_at"] == started
    assert completed.json()["delivery_finished_at"] is not None


async def test_stopping_playback_is_recorded_as_aborted(client: AsyncClient):
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await _delivery(client, message_id, "playing")
    aborted = await _delivery(client, message_id, "aborted")
    assert aborted.json()["delivery_state"] == "aborted"
    assert aborted.json()["delivery_finished_at"] is not None


async def test_stopping_before_playback_does_not_record_a_start(client: AsyncClient):
    """鳴っていない音声を「再生した」ことにしない。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    aborted = await _delivery(client, message_id, "aborted")
    assert aborted.json()["delivery_state"] == "aborted"
    assert aborted.json()["delivery_started_at"] is None
    assert aborted.json()["delivery_finished_at"] is not None


@pytest.mark.parametrize("first", ["completed", "aborted"])
async def test_replay_does_not_overwrite_the_first_delivery(
    client: AsyncClient, first: str
):
    """聞き直しでは記録を動かさない。実際に届いた1回目を残すため。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await _delivery(client, message_id, "playing")
    finished = (await _delivery(client, message_id, first)).json()

    replayed = await _delivery(client, message_id, "playing")
    assert replayed.json()["delivery_state"] == first
    assert replayed.json()["delivery_started_at"] == finished["delivery_started_at"]
    assert replayed.json()["delivery_finished_at"] == finished["delivery_finished_at"]

    # 聞き直しの終了通知でも上書きしない。
    again = await _delivery(client, message_id, "completed")
    assert again.json()["delivery_finished_at"] == finished["delivery_finished_at"]


async def test_repeated_playing_notice_keeps_the_first_start(client: AsyncClient):
    """同じ発言の再生通知が重なっても、開始時刻は最初のものを残す。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    first = (await _delivery(client, message_id, "playing")).json()
    second = (await _delivery(client, message_id, "playing")).json()
    assert second["delivery_started_at"] == first["delivery_started_at"]


async def test_delivery_is_only_for_character_messages(client: AsyncClient):
    result = await _say(client, "こんばんは")
    response = await _delivery(client, result["user_message"]["id"], "playing")
    assert response.status_code == 400


async def test_delivery_rejects_unknown_state(client: AsyncClient):
    result = await _say(client, "こんばんは")
    response = await _delivery(client, result["reply"]["id"], "generated")
    assert response.status_code == 422


async def test_delivery_state_survives_reload(client: AsyncClient):
    """再生の記録は会話履歴から読み直せる。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]
    await _delivery(client, message_id, "playing")
    await _delivery(client, message_id, "completed")

    detail = (
        await client.get(f"/api/conversations/{result['conversation_id']}")
    ).json()
    reply = next(m for m in detail["messages"] if m["id"] == message_id)
    assert reply["delivery_state"] == "completed"
    assert reply["delivery_started_at"] is not None


async def test_speech_run_is_recorded(client: AsyncClient):
    """合成の時間を区間ごとに残す。どこが遅いかを後から見るため。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await client.get(f"/api/conversations/messages/{message_id}/speech")

    runs = (
        await client.get(f"/api/conversations/messages/{message_id}/speech-runs")
    ).json()
    assert len(runs) == 1
    run = runs[0]
    assert run["provider"] == "fake-voice"
    assert run["engine_version"] == "test"
    # 合成用データの作成と音声生成を分けて残す。
    assert run["query_ms"] == 1
    assert run["synthesis_ms"] == 2
    # 音声そのものの長さは、合成の速さと別に持つ。
    assert run["audio_ms"] == 500
    assert run["byte_size"] > 0


async def test_each_synthesis_adds_a_record(client: AsyncClient):
    """聞き直すたびに 1 行増える。同じ文章の合成にかかる時間の変化を見るため。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    for _ in range(3):
        await client.get(f"/api/conversations/messages/{message_id}/speech")

    runs = (
        await client.get(f"/api/conversations/messages/{message_id}/speech-runs")
    ).json()
    assert len(runs) == 3
    # 新しい順に返す。
    assert [r["id"] for r in runs] == sorted((r["id"] for r in runs), reverse=True)


async def test_failed_synthesis_leaves_no_record(client: AsyncClient, fake_speech):
    """合成できなかった試行を、かかった時間として残さない。"""
    fake_speech.fail = True
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await client.get(f"/api/conversations/messages/{message_id}/speech")

    runs = (
        await client.get(f"/api/conversations/messages/{message_id}/speech-runs")
    ).json()
    assert runs == []


async def test_retrieval_time_is_recorded_separately(client: AsyncClient):
    """記憶検索の時間を、生成の時間と分けて残す。"""
    result = await _say(client, "こんばんは")
    run = (
        await client.get(f"/api/conversations/messages/{result['reply']['id']}/run")
    ).json()
    assert run["retrieval_ms"] is not None
    assert run["retrieval_ms"] >= 0
    assert run["latency_ms"] is not None


async def test_stale_notice_cannot_reopen_a_finished_message(
    client: AsyncClient, session_factory
):
    """古い状態を読んだ通知が、話し終えた記録を上書きしない。

    再生の通知は待たずに送られるため、短い再生や直後の停止では通知が
    並行する。反映の判定を読み込んだオブジェクト上で行うと、先に確定した
    完了を、あとから届いた再生開始が押し戻せてしまう。
    """
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    async with session_factory() as session:
        # 通知が届く前の状態を読み込んでおく。
        stale = await session.get(Message, message_id)
        assert stale is not None and stale.delivery_state == "generated"

        # 別の経路で最後まで再生され、記録が確定する。
        await _delivery(client, message_id, "playing")
        finished = (await _delivery(client, message_id, "completed")).json()

        # 古い読み込みのまま、再生開始を反映しようとする。
        await apply_delivery_state(
            session, stale, DeliveryState.PLAYING, now=utcnow()
        )

    stored = (await client.get(f"/api/conversations/messages/{message_id}")).json()
    assert stored["delivery_state"] == "completed"
    assert stored["delivery_finished_at"] == finished["delivery_finished_at"]


async def test_stale_finish_notice_does_not_move_the_first_record(
    client: AsyncClient, session_factory
):
    """遅れて届いた中断の通知でも、確定した完了を上書きしない。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    async with session_factory() as session:
        stale = await session.get(Message, message_id)
        assert stale is not None

        await _delivery(client, message_id, "playing")
        finished = (await _delivery(client, message_id, "completed")).json()

        await apply_delivery_state(
            session, stale, DeliveryState.ABORTED, now=utcnow()
        )

    stored = (await client.get(f"/api/conversations/messages/{message_id}")).json()
    assert stored["delivery_state"] == "completed"
    assert stored["delivery_finished_at"] == finished["delivery_finished_at"]


async def test_aborted_playback_records_the_delivered_char_count(
    client: AsyncClient, fake_llm
):
    """中断したとき、再生位置の比率から届いた文字数を近似する（ISSUE-051）。"""
    fake_llm.push("こんばんは。今日は良い一日でしたか？また明日も話しましょう。")
    result = await _say(client, "ただいま")
    message_id = result["reply"]["id"]
    content = result["reply"]["content"]

    await _delivery(client, message_id, "playing")
    aborted = (await _delivery(client, message_id, "aborted", progress=0.5)).json()

    assert aborted["delivery_state"] == "aborted"
    assert aborted["delivered_char_count"] == round(len(content) * 0.5)


async def test_completed_playback_never_sets_delivered_char_count(client: AsyncClient):
    """完了は「全文届いた」を NULL のままで表す。全長を書き直して二重に表現しない。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await _delivery(client, message_id, "playing")
    completed = (await _delivery(client, message_id, "completed")).json()

    assert completed["delivery_state"] == "completed"
    assert completed["delivered_char_count"] is None


async def test_aborted_without_progress_leaves_delivered_char_count_unset(
    client: AsyncClient,
):
    """進捗が送られない中断（旧クライアント等）では、量を勝手に決めない。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await _delivery(client, message_id, "playing")
    aborted = (await _delivery(client, message_id, "aborted")).json()

    assert aborted["delivery_state"] == "aborted"
    assert aborted["delivered_char_count"] is None


async def test_delivered_char_count_at_full_progress_equals_the_content_length(
    client: AsyncClient,
):
    """比率が1（全部届いた）なら、本文の長さと同じ文字数になる。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]
    content = result["reply"]["content"]

    await _delivery(client, message_id, "playing")
    aborted = (await _delivery(client, message_id, "aborted", progress=1.0)).json()

    assert aborted["delivered_char_count"] == len(content)


def test_delivered_char_count_helper_clamps_progress_above_one():
    """API 側は1を超える比率を422で弾くため（`ge=0, le=1`）、この境界は
    `_delivered_char_count` を直接呼んで確かめる。壊れた呼び出し元が
    範囲外の値を渡しても、本文の長さを超える文字数を作らない。
    """
    assert _delivered_char_count("こんばんは", 1.5) == len("こんばんは")


def test_delivered_char_count_helper_clamps_progress_below_zero():
    assert _delivered_char_count("こんばんは", -0.5) == 0


async def test_delivery_rejects_progress_outside_zero_to_one(client: AsyncClient):
    result = await _say(client, "こんばんは")
    response = await _delivery(client, result["reply"]["id"], "aborted", progress=1.5)
    assert response.status_code == 422


async def test_stale_finish_notice_with_progress_does_not_move_the_first_record(
    client: AsyncClient, session_factory
):
    """遅れて届いた中断の通知（進捗つき）でも、確定した完了を上書きしない。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    async with session_factory() as session:
        stale = await session.get(Message, message_id)
        assert stale is not None

        await _delivery(client, message_id, "playing")
        finished = (await _delivery(client, message_id, "completed")).json()

        await apply_delivery_state(
            session, stale, DeliveryState.ABORTED, now=utcnow(), progress=0.3
        )

    stored = (await client.get(f"/api/conversations/messages/{message_id}")).json()
    assert stored["delivery_state"] == "completed"
    assert stored["delivered_char_count"] is None


async def test_delivery_returns_the_confirmed_state(client: AsyncClient):
    """反映できなかった通知でも、DB 上の確定した状態を返す。"""
    result = await _say(client, "こんばんは")
    message_id = result["reply"]["id"]

    await _delivery(client, message_id, "playing")
    completed = (await _delivery(client, message_id, "completed")).json()

    replayed = (await _delivery(client, message_id, "playing")).json()
    assert replayed == completed
