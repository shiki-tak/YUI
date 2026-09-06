"""音声合成の接続（フェーズ2 Step 1）。

エンジンを起動していなくても実行できるよう、VOICEVOX の HTTP 応答は
差し替えて検証する。実際のエンジンでの確認とは別物として扱う。
"""

from __future__ import annotations

import io
import wave

import httpx
import pytest
from httpx import AsyncClient

from app import main
from app.voice.base import SpeechError, wav_duration_ms
from app.voice.voicevox_client import VoicevoxClient

AUDIO_QUERY = {"accent_phrases": [], "speedScale": 1.0, "outputSamplingRate": 24000}


def make_wav(duration_ms: int, rate: int = 24000) -> bytes:
    """無音の WAV。長さの計算を確かめるために使う。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"\x00\x00" * int(rate * duration_ms / 1000))
    return buffer.getvalue()


def build_client(handler, *, speaker_id: int = 3) -> VoicevoxClient:
    transport = httpx.MockTransport(handler)
    return VoicevoxClient(
        host="http://voicevox.test",
        speaker_id=speaker_id,
        client=httpx.AsyncClient(base_url="http://voicevox.test", transport=transport),
    )


async def test_synthesize_calls_query_then_synthesis():
    """合成用データを作ってから音声を作る。話者は両方に渡す。"""
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, dict(request.url.params)))
        if request.url.path == "/audio_query":
            return httpx.Response(200, json=AUDIO_QUERY)
        if request.url.path == "/synthesis":
            return httpx.Response(200, content=make_wav(500))
        return httpx.Response(200, json="0.99.0")

    result = await build_client(handler).synthesize("こんばんは")

    assert [path for path, _ in calls[:2]] == ["/audio_query", "/synthesis"]
    assert calls[0][1] == {"text": "こんばんは", "speaker": "3"}
    assert calls[1][1] == {"speaker": "3"}
    assert result.media_type == "audio/wav"
    assert result.provider == "voicevox"
    assert result.speaker_id == 3
    # 読み上げた文章を結果に持つ。字幕との一致を後から確認できる。
    assert result.text == "こんばんは"
    assert result.engine_version == "0.99.0"
    assert result.query_ms is not None and result.synthesis_ms is not None


async def test_audio_length_is_measured():
    """合成にかかった時間と、音声そのものの長さを区別する。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio_query":
            return httpx.Response(200, json=AUDIO_QUERY)
        if request.url.path == "/synthesis":
            return httpx.Response(200, content=make_wav(1500))
        return httpx.Response(200, json="0.99.0")

    result = await build_client(handler).synthesize("長めの文章です")
    assert result.audio_ms == 1500


def test_broken_audio_is_not_reported_as_a_length():
    assert wav_duration_ms(b"not a wav") is None


async def test_speaker_can_be_overridden():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("speaker", ""))
        if request.url.path == "/audio_query":
            return httpx.Response(200, json=AUDIO_QUERY)
        if request.url.path == "/synthesis":
            return httpx.Response(200, content=make_wav(100))
        return httpx.Response(200, json="0.99.0")

    result = await build_client(handler, speaker_id=3).synthesize("はい", speaker_id=8)
    assert result.speaker_id == 8
    assert seen[:2] == ["8", "8"]


async def test_empty_text_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("空の文章でエンジンを呼んではいけない")

    with pytest.raises(SpeechError):
        await build_client(handler).synthesize("   \n ")


async def test_engine_error_becomes_speech_error():
    """エンジンのエラーは SpeechError にそろえる。会話側で縮退できるようにする。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio_query":
            return httpx.Response(422, text="unknown speaker")
        return httpx.Response(200, json="0.99.0")

    with pytest.raises(SpeechError) as exc:
        await build_client(handler).synthesize("こんばんは")
    assert "422" in str(exc.value)


async def test_connection_failure_becomes_speech_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SpeechError) as exc:
        await build_client(handler).synthesize("こんばんは")
    assert "接続できませんでした" in str(exc.value)


async def test_empty_audio_is_rejected():
    """再生できない結果を成功として返さない。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio_query":
            return httpx.Response(200, json=AUDIO_QUERY)
        if request.url.path == "/synthesis":
            return httpx.Response(200, content=b"")
        return httpx.Response(200, json="0.99.0")

    with pytest.raises(SpeechError):
        await build_client(handler).synthesize("こんばんは")


async def test_health_reports_engine_version():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/version"
        return httpx.Response(200, json="0.99.0")

    status = await build_client(handler).health()
    assert status == {
        "ok": True,
        "provider": "voicevox",
        "engine_version": "0.99.0",
        "speaker": 3,
    }


async def test_health_does_not_raise_when_engine_is_down():
    """接続できないことを例外ではなく状態として返す。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    status = await build_client(handler).health()
    assert status["ok"] is False
    assert "error" in status


async def test_api_health_includes_voice(client: AsyncClient):
    body = (await client.get("/api/health")).json()
    assert body["voice"]["enabled"] is True
    assert body["voice"]["ok"] is True
    assert body["voice"]["provider"] == "fake-voice"


async def test_api_health_carries_the_credit_line(client: AsyncClient):
    """読み上げた音声を出す場では表記が要る。画面がそれを出せるように返す。"""
    body = (await client.get("/api/health")).json()
    assert body["voice"]["credit"] == main.settings.voicevox_credit
    assert body["voice"]["credit"]


async def test_api_health_reports_disabled_voice(client: AsyncClient, monkeypatch):
    """音声を切っていても、会話ができる状態なら全体は ok のまま。"""
    monkeypatch.setattr(main.settings, "speech_enabled", False)

    body = (await client.get("/api/health")).json()
    assert body["voice"]["enabled"] is False
    assert body["voice"]["ok"] is False
    assert body["ok"] is True
