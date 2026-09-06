"""FastAPI アプリ。フェーズ1は 1 プロセスにまとめる。"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, conversations, memories
from app.config import get_settings
from app.llm import get_llm_client
from app.llm.base import LLMClient
from app.persona import BASE_PERSONA
from app.voice import get_speech_client
from app.voice.base import SpeechClient

settings = get_settings()

app = FastAPI(
    title="yui backend",
    version="0.1.0",
    description="AI VTuber「YUI」：人格・記憶・会話進行（フェーズ1）",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(memories.router, prefix="/api")


@app.get("/api/health", tags=["system"])
async def health(
    llm: LLMClient = Depends(get_llm_client),
    speech: SpeechClient = Depends(get_speech_client),
) -> dict:
    """LLM と音声合成に接続できるかを含めた状態確認。"""
    llm_status = await llm.health()

    voice_status: dict[str, Any] = {
        "ok": False,
        "enabled": settings.speech_enabled,
        "provider": speech.provider,
    }
    if settings.speech_enabled:
        voice_status.update(await speech.health())

    return {
        # 音声が使えなくても文字での会話は続けられるため、全体の状態には含めない。
        "ok": llm_status.get("ok", False),
        "persona": {"name": BASE_PERSONA.name, "version": BASE_PERSONA.version},
        "llm": llm_status,
        "voice": voice_status,
    }


@app.get("/api/persona", tags=["system"])
async def persona() -> dict:
    """いま適用されている基本の人格。開発画面で確認する。"""
    return {
        "name": BASE_PERSONA.name,
        "version": BASE_PERSONA.version,
        "traits": BASE_PERSONA.traits,
        "speech": BASE_PERSONA.speech,
        "rules": BASE_PERSONA.rules,
        "prompt": BASE_PERSONA.to_prompt(),
    }
