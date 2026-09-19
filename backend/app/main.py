"""FastAPI アプリ。当面は 1 プロセスにまとめる。

イベント駆動の構成（設計書 docs/design/yui_stream_first_design.md）を
S0 から積み上げるための最小の土台。入力アダプター・対話ランタイム・
会話・人格・表現の各モジュールは未実装。
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.voice import get_speech_client
from app.voice.base import SpeechClient

settings = get_settings()

app = FastAPI(
    title="yui backend",
    version="0.1.0",
    description="YUI：対話・YouTube Live配信システム",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
async def health(speech: SpeechClient = Depends(get_speech_client)) -> dict:
    """音声合成に接続できるかを含めた状態確認。"""
    voice_status: dict[str, Any] = {
        "ok": False,
        "enabled": settings.speech_enabled,
        "provider": speech.provider,
    }
    if settings.speech_enabled:
        voice_status.update(await speech.health())
        # 読み上げた音声を画面に出す場では、この表記が必要になる。
        voice_status["credit"] = settings.voicevox_credit

    return {"ok": True, "voice": voice_status}
