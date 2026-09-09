"""アプリ設定。環境変数（接頭辞 YUI_）と .env から読む。"""

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# 表示と、会話に出てくる日付の解釈に使う地域時刻。保存は UTC のまま。
LOCAL_TZ = ZoneInfo("Asia/Tokyo")

# 自動採用の対象にできる種類（フェーズ4 PR11）。記憶・状態の種類と、目標。
#
# **models を import しない。** 設定は起動のいちばん外側で読むもので、ここから
# モデルへ依存を張ると輪になる。代わりに、この一覧が MemoryKind / StateKind と
# ずれていないことをテストで見る（test_auto_adopt.py）。種類を増やしたときに
# ここを直し忘れると、有効にできない種類が黙って生まれる。
AUTO_ADOPTABLE_KINDS = frozenset(
    {
        # 記憶
        "experience",
        "about_person",
        "promise",
        "impression",
        "fact",
        # 変化する状態
        "interest",
        "relationship",
        # 目標。種類を持たないので、そのまま1つの名前にする。
        "goal",
    }
)


def to_local(value: datetime) -> datetime:
    """保存された日時を、地域時刻へ直す。

    SQLite は timezone を落として返すため、読み戻した値は timezone を持たない。
    そのまま astimezone を呼ぶと、実行環境の時刻として解釈される。保存は UTC
    なので、UTC を補ってから直す。補わないと、日本時間の 9/8 00:30 の発言が
    9/7 の発言として扱われる（フェーズ3再レビューの指摘2）。
    """
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(LOCAL_TZ)


def _parse_kinds(value: str) -> set[str]:
    return {name.strip() for name in value.split(",") if name.strip()}


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
    # 同じ文字列に保つため（フェーズ2の完了条件）。
    # 既定のままだと YUI は「ワイユウアイ」と読まれる。
    speech_readings: str = "YUI:ユイ:2"

    # 固定人格。personas/<版>.toml を読む。版を増やして比較できるようにし、
    # 生成に使った版は実行記録に残す（設計書フェーズ3の3A）。
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

    # 目標（フェーズ4）
    # 一度聞いた目標を、次に持ち出せるようになるまでの間隔。**恒久的な禁止では
    # ない。** 届いたが答えてもらえなかった質問を二度と聞けなくしないため
    # （PR8 から渡した条件2・3）。繰り返し防止そのものは達成（done）で行う。
    goal_reask_interval_hours: float = 12.0
    # 実行してよくなる日から、これだけ過ぎても実行していない目標は期限切れに
    # する。予定の話題は時間が経つと持ち出しにくくなるため。
    goal_expiry_days: float = 14.0

    # 自動採用する種類（フェーズ4 PR11、設計書の実装内容9）。
    #
    # **既定は空＝すべて手動のまま。** 設計書は「更新前後を比較し、評価できた
    # 種類から自動採用へ移す」としている。移してよい根拠は種類ごとに要る。
    #
    # **既定で有効にしたのは promise と goal だけ**（2026-09-09 の測定、
    # `docs/result/phase4.md`）。全種類を有効にして全シナリオ×3回を流し、
    # 入った中身を読んで決めた。
    #
    # | 種類 | 全体 | 残すべきでない筋書き | 誤った入手経路 |
    # | --- | ---: | ---: | ---: |
    # | promise | 15 | 0 | 0 |
    # | goal | 47 | 0 | — |
    # | relationship | 6 | 0 | — |
    # | impression | 8 | 1 | 0 |
    # | about_person | 23 | 1 | 2 |
    # | experience | 44 | 1 | 4 |
    # | interest | 30 | **4** | — |
    # | fact | 0 | — | — |
    #
    # 「残すべきでない筋書き」は、冗談・雑談・人格変更の要求から候補を作って
    # しまった回のこと。**件数の少なさではなく、中身を読んで決めている**
    # （設計書「自動化の合否を単なる採用件数で測りません」）。
    #
    # - `interest` は冗談から採用した（「10 億円当てた時の本当の感情を知りたい」）。
    # - `impression` は入手経路の誤りが 0/8 だが、冗談から**架空の観察**を採用した
    #   （「開発者の表情が立派だったと評価した」。YUI は表情を見られない）。
    #   **数字だけ見ていたら見落とす。**
    # - `relationship` は 0 だが 6 件しか観測しておらず、根拠が薄い。
    # - `fact` は観測が無い。**根拠が無いという意味であり、安全という意味ではない。**
    #
    # 書ける名前：記憶（experience / about_person / promise / impression /
    # fact）、状態（interest / relationship）、目標（goal）。カンマ区切り。
    # **綴りを間違えたら起動時に止める。** 黙って無効になると、自動採用を
    # 有効にしたつもりの測定が、無効の測定になる。
    auto_adopt: str = "promise,goal"

    # カンマ区切り。.env に JSON を書かせないため文字列で受ける。
    cors_origins: str = "http://localhost:5173"

    @field_validator("auto_adopt")
    @classmethod
    def _check_auto_adopt(cls, value: str) -> str:
        """書き間違いを**設定を作った時点で**落とす（PR11 レビューの指摘7）。

        以前は `auto_adopt_kinds` を読むまで検証しなかった。本番で最初に読むのは
        振り返りの保存の直前なので、**4回の抽出を終えた後に設定エラーになり、
        候補を保存できない**。起動時に止まるほうがよい。

        黙って無効へ倒さないのは、**自動採用を有効にしたつもりの測定が、無効の
        測定になる**ため。PR12 は有無を比べる測定で、そこが狂うと結論が変わる。
        """
        unknown = _parse_kinds(value) - AUTO_ADOPTABLE_KINDS
        if unknown:
            raise ValueError(
                f"auto_adopt に知らない種類があります: {'、'.join(sorted(unknown))}"
                f"（書けるのは {'、'.join(sorted(AUTO_ADOPTABLE_KINDS))}）"
            )
        return value

    @property
    def auto_adopt_kinds(self) -> set[str]:
        """自動採用する種類。検証は生成時に済んでいる。"""
        return _parse_kinds(self.auto_adopt)

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
