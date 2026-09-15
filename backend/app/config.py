"""アプリ設定。環境変数（接頭辞 YUI_）と .env から読む。"""

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# 表示と、会話に出てくる日付の解釈に使う地域時刻。保存は UTC のまま。
LOCAL_TZ = ZoneInfo("Asia/Tokyo")


def to_local(value: datetime) -> datetime:
    """保存された日時を、地域時刻へ直す。

    SQLite は timezone を落として返すため、読み戻した値は timezone を持たない。
    そのまま astimezone を呼ぶと、実行環境の時刻として解釈される。保存は UTC
    なので、UTC を補ってから直す。補わないと、日本時間の 9/8 00:30 の発言が
    9/7 の発言として扱われる（v0.1 レビューの指摘）。
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(LOCAL_TZ)


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
    # 読み上げの読み替え。「表記:カタカナの読み:アクセント位置」を「,」で並べる。
    # 合成に渡す文章は変えず、エンジン側の辞書で発音だけを直す。字幕と読み上げを
    # 同じ文字列に保つため（v0.1 の完了条件）。
    # 既定のままだと YUI は「ワイユウアイ」と読まれる。
    speech_readings: str = "YUI:ユイ:2"

    # 固定人格。personas/<版>.toml を読む。版を増やして比較できるようにし、
    # 生成に使った版は実行記録に残す（人格の版管理。v0.1）。
    persona_version: str = "2026-09-07.1"
    persona_dir: str = "personas"

    # DB
    database_url: str = "sqlite+aiosqlite:///./data/yui.db"

    # 記憶検索・会話履歴
    # 振り返りが中断された場合に、やり直しを許すまでの時間。
    # LLM 自体のタイムアウトより長くとる。
    reflection_stale_seconds: float = 180.0

    memory_retrieval_limit: int = 8
    recent_message_limit: int = 12

    # v0.2 PR3：会話状態の解釈（LLM）。既定は無効のまま
    # （計画 §9・§11 PR5 で判断。docs/result/v0.2.md に測定と根拠を記録）。
    # 待ち時間が1ターンあたり+77%〜+184%（測定条件で変動）増えるうえ、
    # 実モデル（qwen3.5:9b）では判定精度が実用に耐えない
    # （食い違いの検出・連続する訂正が特に弱い）。
    conversation_state_llm: bool = False
    conversation_state_llm_timeout_seconds: float = 10.0
    conversation_state_context_messages: int = 6
    # 食い違いの確認候補を「確かめてよいこと」として渡す回数の上限。
    conversation_state_confirm_offer_limit: int = 2
    # 1ターン全体（解釈＋生成＋再生成1回）の時間上限の候補（計画 §9）。
    # 解釈を有効にした実測（平均11.9〜12.0秒・最大12.7〜14.3秒、測定回で変動）
    # に対して十分な余裕が
    # あることを確認した（v0.2 PR5）。**まだ何も強制しない**——解釈が既定で
    # 無効なので、実際にこの値に迫る場面が今は無い。解釈を有効化するときに
    # 上限処理を作るための記録値。
    turn_time_budget_seconds: float = 45.0

    # カンマ区切り。.env に JSON を書かせないため文字列で受ける。
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def persona_path(self) -> Path:
        """人格の版を置くディレクトリ。相対指定は backend/ からとして解く。"""
        path = Path(self.persona_dir)
        return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()

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
