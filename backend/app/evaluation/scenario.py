"""評価用会話の定義（ISSUE-006 / 設計書フェーズ3）。

pytest とは役割が違う。pytest は実装の振る舞いを固定するもので、こちらは
人格・記憶・根拠の「質」を実モデルで測るためのもの。設計書が「ルール・DB・
キャンセル等は隔離DBとモックで検証し、モデルの判断は実モデルで別に確認する」
と分けているとおり、合否を自動で決めきらず、機械で判定できる観点と人が読む
観点を分けて出す。

シナリオは TOML で書く（人格の版と同じ形式）。1ファイルに複数のシナリオを
並べられる。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.models import Certainty, MemoryKind, SourceKind, Visibility

# 設計書 ISSUE-006 が挙げている観点。ファイルの分け方と対応させる。
ASPECTS = {
    "memory": "記憶：保存した経験・約束を、別の会話で引けるか",
    "grounding": "根拠：返答が渡した記憶の範囲に収まっているか",
    "persona": "人格：口調と「知らないことは正直に言う」が保たれているか",
    "reflection": "振り返り：残すべき経験・約束が候補に含まれるか",
    "repetition": "繰り返し：同じ話を繰り返していないか",
}


class ScenarioError(RuntimeError):
    """シナリオを読み取れなかった。書き間違いは実行前に止める。"""


class SpeakerSpec(BaseModel):
    """会話の相手。表示名だけでなく、入力元と識別子で同定する。"""

    key: str = Field(min_length=1)
    source: str = SourceKind.LOCAL_TEXT.value
    external_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)


class MemorySpec(BaseModel):
    """会話を始める前に入れておく記憶。"""

    key: str = Field(min_length=1)
    kind: str = MemoryKind.EXPERIENCE.value
    content: str = Field(min_length=1)
    certainty: str = Certainty.FACT.value
    visibility: str = Visibility.PRIVATE.value
    keywords: str = ""
    # 誰についての記憶か。相手を指定しなければ、YUI 自身の経験として扱う。
    subject: str | None = None
    # 誰との会話で参照してよいか。未指定はすべての相手に渡る（ISSUE-010）。
    visible_to: str | None = None

    @model_validator(mode="after")
    def _check_enums(self) -> MemorySpec:
        if self.kind not in {k.value for k in MemoryKind}:
            raise ValueError(f"kind が不正です: {self.kind}")
        if self.certainty not in {c.value for c in Certainty}:
            raise ValueError(f"certainty が不正です: {self.certainty}")
        if self.visibility not in {v.value for v in Visibility}:
            raise ValueError(f"visibility が不正です: {self.visibility}")
        return self


class TurnSpec(BaseModel):
    """1回の発言と、その返答に期待すること。"""

    speaker: str | None = None
    text: str = Field(min_length=1)
    # 渡されるべき記憶（MemorySpec.key）。実行記録の参照記憶と突き合わせる。
    expect_memories: list[str] = Field(default_factory=list)
    # 渡してはいけない記憶。関係のない記憶を引き寄せていないかを見る。
    expect_not_memories: list[str] = Field(default_factory=list)
    # 返答に含まれてほしい語。いずれか1つでも含まれれば通す。
    expect_any: list[str] = Field(default_factory=list)
    # 返答に含まれてはいけない語。記憶に無いことを作っていないかを見る。
    expect_none: list[str] = Field(default_factory=list)
    # 直前までの返答をそのまま繰り返していないか。
    expect_not_repeating: bool = False
    # 機械で判定しない観点。レポートに欄として出し、人が読んで書き込む。
    human_check: str | None = None


class ReflectionSpec(BaseModel):
    """会話の終わりに振り返りを流し、候補を確かめる。"""

    run: bool = True
    # 出てほしい候補の種別。並べたものはすべて出ること。1つの項目に
    # "experience|about_person" と書くと、そのどちらかが出れば通す。
    # 分類が割れても内容が残っていればよい場合に使う。
    expect_kinds: list[str] = Field(default_factory=list)
    # 候補の本文に含まれてほしい語。いずれか1つでも含まれれば通す。
    expect_any: list[str] = Field(default_factory=list)
    # 候補が1件も出ないこと。取りこぼしを直すつもりで、何でも記憶にする方向へ
    # 倒れていないかを見る。
    expect_empty: bool = False
    # 機械で判定しない観点。レポートに欄として出す。
    human_check: str | None = None

    @model_validator(mode="after")
    def _check_kinds(self) -> ReflectionSpec:
        known = {k.value for k in MemoryKind}
        for entry in self.expect_kinds:
            for kind in entry.split("|"):
                if kind not in known:
                    raise ValueError(f"expect_kinds が不正です: {kind}")
        if self.expect_empty and (self.expect_kinds or self.expect_any):
            raise ValueError("expect_empty と、出てほしい候補の指定は両立しません。")
        return self


class Scenario(BaseModel):
    id: str = Field(min_length=1)
    aspect: str
    description: str = ""
    speakers: list[SpeakerSpec] = Field(default_factory=list)
    memories: list[MemorySpec] = Field(default_factory=list)
    turns: list[TurnSpec] = Field(min_length=1)
    reflection: ReflectionSpec | None = None

    @model_validator(mode="after")
    def _check_references(self) -> Scenario:
        if self.aspect not in ASPECTS:
            raise ValueError(f"aspect が不正です: {self.aspect}（{'、'.join(ASPECTS)}）")

        if not self.speakers:
            # 相手を書かないシナリオは、開発者ひとりとの会話として扱う。
            self.speakers = [
                SpeakerSpec(
                    key="dev",
                    external_id=f"eval-{self.id}",
                    display_name="開発者",
                )
            ]
        keys = {s.key for s in self.speakers}
        if len(keys) != len(self.speakers):
            raise ValueError("speakers の key が重複しています。")

        memory_keys = {m.key for m in self.memories}
        if len(memory_keys) != len(self.memories):
            raise ValueError("memories の key が重複しています。")

        for memory in self.memories:
            for field, value in (("subject", memory.subject), ("visible_to", memory.visible_to)):
                if value is not None and value not in keys:
                    raise ValueError(f"memories.{field} が speakers にありません: {value}")

        default_speaker = self.speakers[0].key
        for turn in self.turns:
            if turn.speaker is None:
                turn.speaker = default_speaker
            elif turn.speaker not in keys:
                raise ValueError(f"turns.speaker が speakers にありません: {turn.speaker}")
            for field_name in ("expect_memories", "expect_not_memories"):
                for key in getattr(turn, field_name):
                    if key not in memory_keys:
                        raise ValueError(f"turns.{field_name} が memories にありません: {key}")
                    if key in turn.expect_memories and key in turn.expect_not_memories:
                        raise ValueError(f"渡す・渡さないの両方に書かれています: {key}")
        return self


def load_scenarios(path: Path) -> list[Scenario]:
    """ディレクトリまたは1ファイルからシナリオを読む。

    書き間違いは実行前に止める。実モデルを何度も呼んでから気づくと高くつく。
    """
    files = sorted(path.glob("*.toml")) if path.is_dir() else [path]
    if not files:
        raise ScenarioError(f"シナリオが見つかりません: {path}")

    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for file in files:
        try:
            raw = tomllib.loads(file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ScenarioError(f"{file} を読み取れません: {exc}") from exc
        entries = raw.get("scenario", [])
        if not isinstance(entries, list):
            raise ScenarioError(f"{file}: scenario は [[scenario]] で並べてください。")
        for index, entry in enumerate(entries, start=1):
            try:
                scenario = Scenario.model_validate(entry)
            except Exception as exc:
                raise ScenarioError(f"{file} の {index} 件目を読み取れません: {exc}") from exc
            if scenario.id in seen:
                raise ScenarioError(
                    f"シナリオ id が重複しています: {scenario.id}"
                    f"（{seen[scenario.id]} と {file}）"
                )
            seen[scenario.id] = file
            scenarios.append(scenario)
    return scenarios
