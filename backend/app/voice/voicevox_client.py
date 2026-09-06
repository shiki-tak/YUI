"""VOICEVOX Engine による音声合成。

合成は 2 段階に分かれる。

1. /audio_query  読み上げる文章から合成用データ（アクセント・音の長さ）を作る
2. /synthesis    合成用データから WAV を作る

区間を分けて時間を測れるよう、この 2 つは別々に記録する。
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.voice.base import SpeechClient, SpeechError, SpeechResult, wav_duration_ms


class VoicevoxClient(SpeechClient):
    provider = "voicevox"

    def __init__(
        self,
        *,
        host: str,
        speaker_id: int,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.speaker_id = speaker_id
        self._client = client or httpx.AsyncClient(base_url=self.host, timeout=timeout)
        self._engine_version: str | None = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise SpeechError(
                f"VOICEVOX に接続できませんでした（{self.host}）: {exc}"
            ) from exc
        if response.is_error:
            # 本文は原因の手がかりになるが、長い HTML が返ることもあるため切る。
            detail = response.text[:200].replace("\n", " ")
            raise SpeechError(
                f"VOICEVOX がエラーを返しました（{path} / {response.status_code}）: {detail}"
            )
        return response

    async def _version(self) -> str | None:
        """エンジンの版。どの版で合成した音声かを後から追えるように記録する。"""
        if self._engine_version is not None:
            return self._engine_version
        try:
            response = await self._request("GET", "/version")
        except SpeechError:
            return None
        self._engine_version = _as_version(response)
        return self._engine_version

    async def synthesize(self, text: str, *, speaker_id: int | None = None) -> SpeechResult:
        body = text.strip()
        if not body:
            raise SpeechError("読み上げる文章が空です。")
        speaker = self.speaker_id if speaker_id is None else speaker_id

        started = time.perf_counter()
        query = await self._request(
            "POST", "/audio_query", params={"text": body, "speaker": speaker}
        )
        query_ms = int((time.perf_counter() - started) * 1000)
        try:
            audio_query = query.json()
        except ValueError as exc:
            raise SpeechError("VOICEVOX の合成用データを読み取れませんでした。") from exc

        started = time.perf_counter()
        synthesis = await self._request(
            "POST", "/synthesis", params={"speaker": speaker}, json=audio_query
        )
        synthesis_ms = int((time.perf_counter() - started) * 1000)

        audio = synthesis.content
        if not audio:
            raise SpeechError("VOICEVOX から空の音声が返りました。")

        return SpeechResult(
            audio=audio,
            media_type="audio/wav",
            provider=self.provider,
            speaker_id=speaker,
            text=body,
            engine_version=await self._version(),
            query_ms=query_ms,
            synthesis_ms=synthesis_ms,
            audio_ms=wav_duration_ms(audio),
            options={"speaker": speaker},
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._request("GET", "/version")
        except SpeechError as exc:
            return {"ok": False, "provider": self.provider, "error": str(exc)}
        self._engine_version = _as_version(response)
        return {
            "ok": True,
            "provider": self.provider,
            "engine_version": self._engine_version,
            "speaker": self.speaker_id,
        }

    async def aclose(self) -> None:
        await self._client.aclose()


def _as_version(response: httpx.Response) -> str | None:
    """/version は JSON 文字列で返る。読めない場合は本文をそのまま使う。"""
    try:
        value = response.json()
    except ValueError:
        value = response.text
    text = str(value).strip()
    return text or None
