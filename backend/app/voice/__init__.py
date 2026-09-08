"""音声合成モジュール。"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.voice.base import (
    Reading,
    SpeechClient,
    SpeechError,
    SpeechResult,
    parse_readings,
)
from app.voice.voicevox_client import VoicevoxClient

__all__ = [
    "Reading",
    "SpeechClient",
    "SpeechError",
    "SpeechResult",
    "get_speech_client",
    "parse_readings",
]


@lru_cache
def get_speech_client() -> SpeechClient:
    """読み上げに使うクライアント。

    エンジンを差し替えるときは、ここで別の SpeechClient を返す。
    """
    settings = get_settings()
    return VoicevoxClient(
        host=settings.voicevox_host,
        speaker_id=settings.voicevox_speaker,
        timeout=settings.voicevox_timeout_seconds,
        readings=parse_readings(settings.speech_readings),
    )
