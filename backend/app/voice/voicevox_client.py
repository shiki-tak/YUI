"""VOICEVOX Engine による音声合成。

合成は 2 段階に分かれる。

1. /audio_query  読み上げる文章から合成用データ（アクセント・音の長さ）を作る
2. /synthesis    合成用データから WAV を作る

区間を分けて時間を測れるよう、この 2 つは別々に記録する。

読みの調整はエンジンのユーザー辞書（/user_dict）で行う。合成に渡す文章を
書き換えると、字幕と読み上げが別の文字列になる（フェーズ2の完了条件
「読み上げた内容と字幕が一致していることを確認できる」に反する）。
"""

from __future__ import annotations

import logging
import time
import unicodedata
from typing import Any

import httpx

from app.voice.base import (
    Reading,
    SpeechClient,
    SpeechError,
    SpeechResult,
    wav_duration_ms,
)

logger = logging.getLogger(__name__)


class VoicevoxClient(SpeechClient):
    provider = "voicevox"

    def __init__(
        self,
        *,
        host: str,
        speaker_id: int,
        timeout: float = 30.0,
        readings: list[Reading] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.speaker_id = speaker_id
        self._client = client or httpx.AsyncClient(base_url=self.host, timeout=timeout)
        self._engine_version: str | None = None
        self._readings = readings or []
        # 辞書を入れ終わったか。エンジンは再起動で入れ直しが要ることがあるため、
        # 失敗したら次の合成でもう一度試す。
        self._readings_ready = not self._readings

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

    async def register_readings(self) -> None:
        """読み替えをエンジンのユーザー辞書へ入れる。

        同じ表記が既にあれば、読みが違うときだけ書き換える。毎回登録し直すと
        辞書が同じ語で埋まる。

        エンジンは表記を全角へ直して保存する（"YUI" → "ＹＵＩ"）。そのまま
        比べると毎回「未登録」と判断し、起動のたびに同じ語が増える。
        """
        if not self._readings:
            return
        response = await self._request("GET", "/user_dict")
        try:
            current: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SpeechError("VOICEVOX のユーザー辞書を読み取れませんでした。") from exc

        by_surface = {
            _normalize(str(word.get("surface", ""))): (uuid, word)
            for uuid, word in current.items()
            if isinstance(word, dict)
        }
        for reading in self._readings:
            params = {
                "surface": reading.surface,
                "pronunciation": reading.pronunciation,
                "accent_type": reading.accent,
                "word_type": "PROPER_NOUN",
            }
            found = by_surface.get(_normalize(reading.surface))
            if found is None:
                await self._request("POST", "/user_dict_word", params=params)
                continue
            uuid, word = found
            if (
                word.get("pronunciation") == reading.pronunciation
                and word.get("accent_type") == reading.accent
            ):
                continue
            await self._request("PUT", f"/user_dict_word/{uuid}", params=params)

    async def _ensure_readings(self) -> None:
        """合成の前に辞書を整える。失敗しても合成は止めない。

        辞書が入らない理由は読みが直らないことだけで、音声は出せる。読み上げ
        自体を落とすほうが重い。
        """
        if self._readings_ready:
            return
        try:
            await self.register_readings()
        except SpeechError as exc:
            logger.warning("VOICEVOX のユーザー辞書を登録できませんでした: %s", exc)
            return
        self._readings_ready = True

    async def synthesize(self, text: str, *, speaker_id: int | None = None) -> SpeechResult:
        body = text.strip()
        if not body:
            raise SpeechError("読み上げる文章が空です。")
        speaker = self.speaker_id if speaker_id is None else speaker_id
        await self._ensure_readings()

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


def _normalize(surface: str) -> str:
    """表記を比べるための形。全角と半角の違いを無視する。

    エンジンは登録時に半角英数を全角へ直す。比較の側で揃えないと、登録済みの
    語を毎回「無い」と判断してしまう。
    """
    return unicodedata.normalize("NFKC", surface)


def _as_version(response: httpx.Response) -> str | None:
    """/version は JSON 文字列で返る。読めない場合は本文をそのまま使う。"""
    try:
        value = response.json()
    except ValueError:
        value = response.text
    text = str(value).strip()
    return text or None
