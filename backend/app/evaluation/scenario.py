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
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models import Certainty, MemoryKind, SourceKind, StateKind, Visibility

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


class StateSpec(BaseModel):
    """会話を始める前に採用しておく、変化する状態（関心・相手との関係）。

    固定人格とは別に渡されることと、相手の意見でその場で反転しないことを
    見るために使う（設計書フェーズ3の3B）。
    """

    kind: str = StateKind.INTEREST.value
    content: str = Field(min_length=1)
    topic: str | None = None
    # 関係性のとき：誰との関係か。
    subject: str | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> StateSpec:
        if self.kind not in {k.value for k in StateKind}:
            raise ValueError(f"状態の種類が不正です: {self.kind}")
        if self.kind == StateKind.RELATIONSHIP.value and self.subject is None:
            raise ValueError("関係性には subject が要ります。")
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
    # 出来事の日付が入ること（ISSUE-004）。会話に日付の手がかりがある
    # シナリオで使う。
    expect_occurred_at: bool = False
    # 候補に「近い既存の記憶」が印として付くこと。同じ出来事を二重に
    # 覚えないための手がかりが働いているかを見る（ISSUE-018）。
    expect_similar_marked: bool = False
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


class StepSpec(BaseModel):
    """会話の途中で行う操作（ISSUE-006 の通し評価）。

    完了条件1・2は、記憶が**作られてから**使われるまでを見る必要がある。
    記憶を事前に入れて1会話を流すだけでは、抽出と採用の経路を通らない。

    - say：発言する（turns と同じ）。
    - reflect：会話を振り返り、候補を出す。accept で採用まで行う。
    - new_conversation：会話を分ける。別のセッションとして続ける。
    - restart：DBへの接続を作り直し、会話の担当も作り直す。**実際のプロセス
      再起動ではない**。保存から読み直されることは確かめられるが、起動時の
      処理（lifespan、人格の読み込み）は通らない。
    - correct_memory / delete_memory：開発者が記憶を訂正・削除する。
    """

    kind: Literal[
        "say", "reflect", "new_conversation", "restart", "correct_memory", "delete_memory"
    ] = "say"

    # say
    speaker: str | None = None
    text: str | None = None
    expect_memories: list[str] = Field(default_factory=list)
    expect_not_memories: list[str] = Field(default_factory=list)
    # 渡された可変状態の本文に含まれる語。返答の語ではなく、プロンプトへ
    # 実際に渡ったものを見る。訂正が状態へ届いたかは、返答の言い回しでは
    # 決められない（第6回レビューの「測っていない観点」）。
    expect_states: list[str] = Field(default_factory=list)
    # 渡ってはいけない状態。根拠の記憶を訂正・削除した後に、古い状態が
    # 会話へ入らないことを見る。
    expect_not_states: list[str] = Field(default_factory=list)
    expect_any: list[str] = Field(default_factory=list)
    expect_none: list[str] = Field(default_factory=list)
    expect_not_repeating: bool = False
    human_check: str | None = None

    # reflect
    accept: bool = False
    # 本文にこの語を含む候補だけ採用する。空なら出た候補をすべて採用する。
    accept_contains: list[str] = Field(default_factory=list)
    # 採用した記憶に付ける名前。後の say の expect_memories で、
    # 「振り返りで作った記憶が実際に渡ったか」を指せるようにする。
    accept_key: str | None = None
    # 関心・関係性の候補も採用する。記憶を訂正したときに、そこから作った状態へ
    # 波及するかを測るために使う。
    accept_states: bool = False
    expect_kinds: list[str] = Field(default_factory=list)
    expect_candidate_any: list[str] = Field(default_factory=list)
    # 関心・関係性の候補に含まれてほしい語。状態が1件も出ていないのに、
    # 後の「古い状態が渡っていない」が空振りで通るのを防ぐ。
    expect_state_any: list[str] = Field(default_factory=list)
    expect_empty: bool = False
    expect_occurred_at: bool = False
    expect_similar_marked: bool = False

    # correct_memory / delete_memory：本文にこの語を含む記憶を選ぶ
    match: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> StepSpec:
        if self.kind == "say" and not self.text:
            raise ValueError("say には text が要ります。")
        if self.kind in {"correct_memory", "delete_memory"} and not self.match:
            raise ValueError(f"{self.kind} には match が要ります。")
        if self.kind == "correct_memory" and not self.content:
            raise ValueError("correct_memory には content が要ります。")
        return self


class Scenario(BaseModel):
    id: str = Field(min_length=1)
    aspect: str
    description: str = ""
    speakers: list[SpeakerSpec] = Field(default_factory=list)
    memories: list[MemorySpec] = Field(default_factory=list)
    states: list[StateSpec] = Field(default_factory=list)
    # turns は1会話を流すだけの書き方。steps は採用・訂正・再起動を挟める。
    turns: list[TurnSpec] = Field(default_factory=list)
    steps: list[StepSpec] = Field(default_factory=list)
    reflection: ReflectionSpec | None = None

    @property
    def effective_steps(self) -> list[StepSpec]:
        """turns 形式も steps へ揃えて返す。実行側は steps だけを見る。"""
        if self.steps:
            return self.steps
        steps = [
            StepSpec(
                kind="say",
                speaker=turn.speaker,
                text=turn.text,
                expect_memories=turn.expect_memories,
                expect_not_memories=turn.expect_not_memories,
                expect_any=turn.expect_any,
                expect_none=turn.expect_none,
                expect_not_repeating=turn.expect_not_repeating,
                human_check=turn.human_check,
            )
            for turn in self.turns
        ]
        spec = self.reflection
        if spec is not None and spec.run:
            steps.append(
                StepSpec(
                    kind="reflect",
                    expect_kinds=spec.expect_kinds,
                    expect_candidate_any=spec.expect_any,
                    expect_empty=spec.expect_empty,
                    expect_occurred_at=spec.expect_occurred_at,
                    expect_similar_marked=spec.expect_similar_marked,
                    human_check=spec.human_check,
                )
            )
        return steps

    @property
    def has_machine_checks(self) -> bool:
        """機械で判定する項目が宣言されているか。

        実行できたチェックの数ではなく、シナリオの書き方から決める。モデルの
        呼び出しが失敗して1つも判定できなかった場合に、人手専用のシナリオと
        取り違えて集計から落とさないため（フェーズ3再レビューの指摘3）。
        """
        for step in self.effective_steps:
            # 訂正・削除は、対象が見つかったかどうかを機械で判定する
            # （第6回レビューの指摘2）。
            if step.kind in {"correct_memory", "delete_memory"}:
                return True
            if (
                step.expect_memories
                or step.expect_not_memories
                or step.expect_states
                or step.expect_not_states
                or step.expect_any
                or step.expect_none
                or step.expect_not_repeating
                or step.expect_kinds
                or step.expect_candidate_any
                or step.expect_state_any
                or step.expect_empty
                or step.expect_similar_marked
                or step.expect_occurred_at
            ):
                return True
        return False

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
        # 振り返りで採用した記憶にも名前を付けられる。事前に入れた記憶と
        # 同じように expect_memories から指せるようにする。
        memory_keys |= {s.accept_key for s in self.steps if s.accept_key}

        for memory in self.memories:
            for field, value in (("subject", memory.subject), ("visible_to", memory.visible_to)):
                if value is not None and value not in keys:
                    raise ValueError(f"memories.{field} が speakers にありません: {value}")

        for state in self.states:
            if state.subject is not None and state.subject not in keys:
                raise ValueError(f"states.subject が speakers にありません: {state.subject}")

        if not self.turns and not self.steps:
            raise ValueError("turns か steps のどちらかが要ります。")
        if self.turns and self.steps:
            raise ValueError("turns と steps は両方書けません。")

        default_speaker = self.speakers[0].key
        for item in [*self.turns, *self.steps]:
            if getattr(item, "kind", "say") != "say":
                continue
            if item.speaker is None:
                item.speaker = default_speaker
            elif item.speaker not in keys:
                raise ValueError(f"speaker が speakers にありません: {item.speaker}")
            for field_name in ("expect_memories", "expect_not_memories"):
                for key in getattr(item, field_name):
                    if key not in memory_keys:
                        raise ValueError(f"{field_name} が memories にありません: {key}")
                    if key in item.expect_memories and key in item.expect_not_memories:
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
