"""FastAPI アプリ。当面は 1 プロセスにまとめる。

重い振り返りや学習を別プロセスへ分けるのは、必要が出てからにする。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, conversations, memories
from app.config import get_settings
from app.llm import get_llm_client
from app.llm.base import LLMClient
from app.persona import PersonaError, available_versions, load_persona
from app.schemas import PersonaOut
from app.voice import get_speech_client
from app.voice.base import SpeechClient

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """起動時に既定の人格を確かめる。

    人格が読めない状態で会話を受け付けると、どの人格で話したのかを後から
    追えない。設定の誤りは、最初のリクエストではなく起動で分かるようにする。
    """
    load_persona()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="yui backend",
    version="0.1.0",
    description="AIキャラクター「YUI」：人格・記憶・会話進行",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(PersonaError)
async def persona_error_handler(_: Request, exc: PersonaError) -> JSONResponse:
    """人格を読めないときの応答。

    起動時に確かめているため、ここへ来るのは起動後に版のファイルが
    壊れた場合など。原因の分かる本文を返し、500 で潰さない。
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
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
    persona_def = load_persona()

    voice_status: dict[str, Any] = {
        "ok": False,
        "enabled": settings.speech_enabled,
        "provider": speech.provider,
    }
    if settings.speech_enabled:
        voice_status.update(await speech.health())
        # 読み上げた音声を画面に出す場では、この表記が必要になる。
        voice_status["credit"] = settings.voicevox_credit

    return {
        # 音声が使えなくても文字での会話は続けられるため、全体の状態には含めない。
        "ok": llm_status.get("ok", False),
        "persona": {"name": persona_def.name, "version": persona_def.version},
        "llm": llm_status,
        "voice": voice_status,
    }


@app.get("/api/persona", response_model=PersonaOut, tags=["system"])
async def persona(version: str | None = None) -> PersonaOut:
    """固定人格の内容。version を指定すると、その版の定義を返す。

    版を比べるときは、この出力と実行記録の persona_version を突き合わせる。
    """
    versions = available_versions()
    if version is not None and version not in versions:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"人格の版が見つかりません: {version}。"
            f"用意されている版: {'、'.join(versions) or 'なし'}",
        )
    persona_def = load_persona(version)
    return PersonaOut(
        name=persona_def.name,
        version=persona_def.version,
        traits=persona_def.traits,
        speech=persona_def.speech,
        rules=persona_def.rules,
        identity=persona_def.identity,
        curiosity=persona_def.curiosity,
        boundaries=persona_def.boundaries,
        prompt=persona_def.to_prompt(),
        available_versions=versions,
    )
