"""アプリ設定。環境変数（接頭辞 YUI_）と .env から読む。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YUI_",
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama：通常の推論
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_temperature: float = 0.8
    ollama_num_ctx: int = 8192
    # 思考出力の制御。False で切る（応答が大幅に速くなる）。
    # None にすると think を送らず、モデルの既定に任せる。
    ollama_think: bool | None = False
    llm_timeout_seconds: float = 120.0

    # VOICEVOX Engine：音声合成
    voicevox_host: str = "http://localhost:50021"
    # 猫使ビィ／おちつき。読み上げた音声を公開する場合は
    # 「VOICEVOX:猫使ビィ」のクレジット表記が必要（話者を変えるときは規約も確認する）。
    voicevox_speaker: int = 59
    voicevox_timeout_seconds: float = 30.0
    # 生成した音声を公開する場に出す表記。話者を変えるときは必ず一緒に直す。
    voicevox_credit: str = "VOICEVOX:猫使ビィ"
    # 音声を切っても会話は続けられる。エンジンが無い環境で使う。
    speech_enabled: bool = True

    # DB
    database_url: str = "sqlite+aiosqlite:///./data/yui.db"

    # 記憶検索・会話履歴
    # 振り返りが中断された場合に、やり直しを許すまでの時間。
    # LLM 自体のタイムアウトより長くとる。
    reflection_stale_seconds: float = 180.0

    memory_retrieval_limit: int = 8
    recent_message_limit: int = 12

    # カンマ区切り。.env に JSON を書かせないため文字列で受ける。
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlite_path(self) -> Path | None:
        """SQLite の実ファイル位置。ディレクトリ作成に使う。"""
        prefix = "sqlite+aiosqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        raw = self.database_url[len(prefix) :]
        path = Path(raw)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
