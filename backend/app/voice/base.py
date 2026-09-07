"""音声合成の共通インターフェース。

設計書「3. 技術と採用範囲」の音声合成にあたる。LLM 接続と同じ考え方で、
呼び出し側はこの窓口だけを使い、合成エンジンを差し替えても影響を受けない
ようにする。

合成にかかった時間を結果に含める。設計書フェーズ2の実装内容6「送信から
音声の再生開始までの時間を測り、待ち時間の大きい部分を確認する」で、
どの区間が遅いかを分けて見るために使う。
"""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpeechResult:
    audio: bytes
    media_type: str
    provider: str
    speaker_id: int
    # 読み上げた文章。字幕と読み上げの一致を確認できるように結果へ持たせる。
    text: str
    engine_version: str | None = None
    # 合成用データの作成にかかった時間。
    query_ms: int | None = None
    # 音声そのものの生成にかかった時間。
    synthesis_ms: int | None = None
    # 生成された音声の長さ。合成の速さと区別する。
    audio_ms: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


class SpeechError(RuntimeError):
    """音声合成の失敗。呼び出し側はこれを捕まえて縮退する。

    音声が出せなくても会話は続けられる状態を保つ。
    """


class SpeechClient(ABC):
    provider: str

    @abstractmethod
    async def synthesize(self, text: str, *, speaker_id: int | None = None) -> SpeechResult: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """接続確認。エンジンが使える状態かを返す。"""

    async def aclose(self) -> None:
        return None


def wav_duration_ms(audio: bytes) -> int | None:
    """WAV の長さ。読み取れない形式なら None を返す。

    合成にかかった時間とは別物で、再生にかかる時間の見積もりに使う。
    """
    try:
        with wave.open(io.BytesIO(audio)) as source:
            rate = source.getframerate()
            if rate <= 0:
                return None
            return int(source.getnframes() / rate * 1000)
    except (wave.Error, EOFError):
        return None
