"""音声の取得と再生状態（フェーズ2 Step 2）。

フェーズ2の完了条件のうち、次を対象にする。

- 文字入力から返答生成、音声の取得まで一通り動く。
- 返答を重複再生せず、停止操作で音声を止められる（記録の側から確かめる）。
- 読み上げた内容と字幕が一致している（合成に渡す文章が発言の本文と同じ）。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app import main
from tests.conftest import FakeSpeech


async def _say(client: AsyncClient, text: str, conversation_id: int | None = None) -> dict:
    payload: dict = {"text": text}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


async def _delivery(client: AsyncClient, message_id: int, state: str):
    return await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": state}
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
