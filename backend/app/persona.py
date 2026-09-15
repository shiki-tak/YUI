"""固定人格の定義と、版の読み込み。

人格の版管理（v0.1）にあたる。人格の定義はコードではなく `personas/<版>.toml` に置き、
どの版で生成したかを実行記録（run_records.persona_version）に残す。

可変の状態（関心・関係性・目標）はここでは扱わない。固定人格と可変状態を
分け、保存先は DB になる（ISSUE-015）。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

# 版の名前はファイル名になる。ディレクトリを抜け出す指定を受け付けない。
_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


class PersonaError(RuntimeError):
    """人格の版を読み込めなかった。

    既定の版は起動時に確かめる（app.main の lifespan）。人格が不明なまま
    会話を始めると、どの人格で話したのかを後から追えなくなるため、
    既定の人格へ黙って落とさない。

    起動後に出るのは、会話ごとに指定された版のファイルが壊れている場合など。
    API 層で 503 に変換する。
    """


@dataclass(frozen=True)
class Persona:
    name: str
    version: str
    traits: list[str] = field(default_factory=list)
    speech: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    # 以下は設計書 4.3 の設定群のうち、3A で具体化するもの。
    # 空の項目はプロンプトに出さない。基準版との差分を、実際に渡した文面で
    # 比べられるようにするため。
    identity: list[str] = field(default_factory=list)
    curiosity: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"あなたは「{self.name}」という名前のキャラクターです。"]
        for heading, items in (
            ("自分について", self.identity),
            ("性格", self.traits),
            ("話し方", self.speech),
            ("興味の向け方", self.curiosity),
            ("守ること", self.rules),
            ("知識と体験の境界", self.boundaries),
        ):
            if not items:
                continue
            lines.append("")
            lines.append(f"# {heading}")
            lines += [f"- {item}" for item in items]
        return "\n".join(lines)


def _persona_dir() -> Path:
    return get_settings().persona_path


def _read_list(raw: dict, key: str, *, version: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PersonaError(f"人格 {version} の {key} は文字列の配列で書いてください。")
    return [item for item in value]


def _persona_file(directory: Path, version: str) -> Path | None:
    """personas 内の実ファイルとして解決できた場合だけパスを返す。

    版の名前はファイル名になる。名前の文字種だけを見ても、シンボリック
    リンクがディレクトリの外を指していれば外のファイルを読めてしまう。
    解決した先がこのディレクトリの直下にあることまで確かめる。
    """
    path = directory / f"{version}.toml"
    if not path.is_file():
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != directory.resolve():
        return None
    return resolved


def _validate_version(version: str) -> str:
    if not _VERSION_PATTERN.match(version) or ".." in version:
        raise PersonaError(f"人格の版の名前として使えません: {version!r}")
    return version


@lru_cache
def load_persona(version: str | None = None) -> Persona:
    """人格の版を読み込む。version を省略すると設定の版を使う。"""
    # 空文字を既定の版として扱わない。指定した版と実際に使った版が
    # 食い違うと、実行記録の persona_version から文面を引けなくなる。
    if version is None:
        version = get_settings().persona_version
    version = _validate_version(version)
    directory = _persona_dir()
    path = _persona_file(directory, version)
    if path is None:
        available = "、".join(available_versions()) or "（1つもありません）"
        raise PersonaError(
            f"人格の版が見つかりません: {directory / f'{version}.toml'}。"
            f"ディレクトリの外を指している場合も読み込みません。"
            f"用意されている版: {available}"
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PersonaError(f"人格 {version} を読み取れません: {exc}") from exc

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PersonaError(f"人格 {version} に name がありません。")
    # ファイル名と中身の版が食い違うと、記録した版から定義を引けなくなる。
    declared = raw.get("version")
    if declared is not None and declared != version:
        raise PersonaError(
            f"人格 {version} の中の version が {declared!r} になっています。"
            "ファイル名と揃えてください。"
        )

    return Persona(
        name=name,
        version=version,
        traits=_read_list(raw, "traits", version=version),
        speech=_read_list(raw, "speech", version=version),
        rules=_read_list(raw, "rules", version=version),
        identity=_read_list(raw, "identity", version=version),
        curiosity=_read_list(raw, "curiosity", version=version),
        boundaries=_read_list(raw, "boundaries", version=version),
    )


def available_versions() -> list[str]:
    """置いてある人格の版。会話ごとの指定は、この一覧の中だけを許す。"""
    directory = _persona_dir()
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.glob("*.toml")
        # 一覧と読み込みで同じ条件を使う。一覧に出るのに読めない版があると、
        # 会話ごとの指定が通ったり通らなかったりする。
        if _VERSION_PATTERN.match(path.stem)
        and ".." not in path.stem
        and _persona_file(directory, path.stem) is not None
    )
