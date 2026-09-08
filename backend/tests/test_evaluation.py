"""評価用会話の土台（ISSUE-006）。

ここで確かめるのはハーネスの側で、人格や記憶の質ではない。質は実モデルで
`python -m app.evaluation.run` を流して見る（自動判定と人手確認を分ける）。

- シナリオの書き間違いを、実モデルを呼ぶ前に止める。
- 判定が、通るときに通り、落ちるときに落ちる。
- 使い捨てのDBを使い、通常利用の会話・記憶に触れない。
- レポートに、版を比べるのに必要な情報が残る。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import BACKEND_ROOT, get_settings
from app.evaluation.report import build_markdown
from app.evaluation.runner import run_scenarios
from app.evaluation.scenario import Scenario, ScenarioError, load_scenarios
from app.persona import load_persona
from tests.conftest import FakeLLM

SCENARIOS = BACKEND_ROOT / "evals" / "scenarios"


def _write(tmp_path: Path, body: str) -> Path:
    directory = tmp_path / "scenarios"
    directory.mkdir(exist_ok=True)
    (directory / "case.toml").write_text(body, encoding="utf-8")
    return directory


def test_bundled_scenarios_load() -> None:
    """同梱のシナリオが読める。観点の名前も検証される。"""
    scenarios = load_scenarios(SCENARIOS)
    assert len(scenarios) >= 10
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))
    # ISSUE-002 を測るシナリオが入っていること。
    assert "promise-running-shoes" in ids


def test_unknown_memory_key_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
[[scenario]]
id = "broken"
aspect = "memory"
[[scenario.turns]]
text = "覚えてる？"
expect_memories = ["typo"]
""",
    )
    with pytest.raises(ScenarioError) as exc:
        load_scenarios(directory)
    assert "expect_memories" in str(exc.value)


def test_unknown_aspect_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
[[scenario]]
id = "broken"
aspect = "なんとなく"
[[scenario.turns]]
text = "こんにちは"
""",
    )
    with pytest.raises(ScenarioError):
        load_scenarios(directory)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
[[scenario]]
id = "same"
aspect = "memory"
[[scenario.turns]]
text = "1つ目"

[[scenario]]
id = "same"
aspect = "memory"
[[scenario.turns]]
text = "2つ目"
""",
    )
    with pytest.raises(ScenarioError) as exc:
        load_scenarios(directory)
    assert "重複" in str(exc.value)


def _scenario(**overrides) -> Scenario:
    base = {
        "id": "case",
        "aspect": "memory",
        "memories": [
            {"key": "camera", "content": "開発者は写真を撮るのが好き", "keywords": "写真 趣味"}
        ],
        "turns": [
            {
                "text": "私の趣味、覚えてる？",
                "expect_memories": ["camera"],
                "expect_any": ["写真"],
            }
        ],
    }
    base.update(overrides)
    return Scenario.model_validate(base)


async def test_checks_pass_when_the_reply_matches(fake_llm: FakeLLM) -> None:
    fake_llm.push("写真を撮るのがお好きでしたよね。")
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=load_persona(),
        settings=get_settings(),
    )
    attempt = results[0].attempts[0]
    assert attempt.ok
    assert results[0].passed == 1
    # 渡した記憶は、シナリオの鍵で読める。
    assert attempt.turns[0].referenced == ["camera"]


async def test_checks_fail_when_the_expected_word_is_missing(fake_llm: FakeLLM) -> None:
    fake_llm.push("さあ、なんでしたっけ。")
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=load_persona(),
        settings=get_settings(),
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    failed = [check for check in attempt.turns[0].checks if not check.ok]
    assert [check.name for check in failed] == ["含まれてほしい語"]


async def test_forbidden_word_is_detected(fake_llm: FakeLLM) -> None:
    fake_llm.push("富士山に登った話でしたね。")
    scenario = _scenario(
        turns=[{"text": "山の話を覚えてる？", "expect_none": ["富士山"]}],
        memories=[],
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    assert not results[0].attempts[0].ok


async def test_repetition_is_detected(fake_llm: FakeLLM) -> None:
    fake_llm.push("写真がお好きでしたよね。")
    fake_llm.push("写真がお好きでしたよね。")
    scenario = _scenario(
        memories=[],
        turns=[
            {"text": "私の趣味は？"},
            {"text": "他には？", "expect_not_repeating": True},
        ],
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert [c.name for c in attempt.turns[1].checks if not c.ok] == ["繰り返していない"]


async def test_reflection_candidates_are_checked(fake_llm: FakeLLM) -> None:
    """ISSUE-002 の測り方。候補が空でも失敗として数え、握り潰さない。"""
    scenario = _scenario(
        aspect="reflection",
        memories=[],
        turns=[{"text": "次はシューズの話をしよう"}],
        reflection={"run": True, "expect_kinds": ["promise"], "expect_any": ["シューズ"]},
    )

    fake_llm.push("楽しみですね。")
    fake_llm.push("[]")
    missed = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    assert not missed[0].attempts[0].ok
    assert missed[0].attempts[0].reflection is not None
    assert missed[0].attempts[0].reflection.candidates == []

    fake_llm.push("楽しみですね。")
    fake_llm.push(
        '[{"kind":"promise","content":"次はシューズの話をする","certainty":"fact",'
        '"keywords":"シューズ 約束","about_partner":false}]'
    )
    kept = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    assert kept[0].attempts[0].ok


async def test_repeat_counts_attempts(fake_llm: FakeLLM) -> None:
    """揺れを見るため、通った回数と試行回数を分けて数える。"""
    fake_llm.push("写真ですね。")
    fake_llm.push("さあ。")
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=load_persona(),
        settings=get_settings(),
        repeat=2,
    )
    assert results[0].total == 2
    assert results[0].passed == 1


async def test_evaluation_does_not_touch_the_normal_database(fake_llm: FakeLLM) -> None:
    """通常利用のDBを開かない。評価で会話・記憶が増えないことの確認。"""
    settings = get_settings()
    normal = settings.sqlite_path
    assert normal is not None
    before = normal.stat().st_mtime if normal.exists() else None

    fake_llm.push("写真ですね。")
    await run_scenarios(
        [_scenario()], llm=fake_llm, persona=load_persona(), settings=settings
    )

    after = normal.stat().st_mtime if normal.exists() else None
    assert before == after


async def test_report_keeps_what_is_needed_to_compare_versions(fake_llm: FakeLLM) -> None:
    fake_llm.push("写真ですね。")
    persona = load_persona()
    results = await run_scenarios(
        [_scenario()], llm=fake_llm, persona=persona, settings=get_settings()
    )
    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
    )
    # 版を並べて比べるのに要るもの。
    assert persona.version in markdown
    assert "qwen3.5:9b" in markdown
    assert "sha256:test" in markdown
    # 人が読む欄が残っている（自動判定だけで合格としない）。
    assert "自動判定" in markdown
    assert "1 / 1" in markdown


async def test_unwanted_memory_is_detected(fake_llm: FakeLLM) -> None:
    """関係のない記憶が渡っていないかを見る（ISSUE-019 の拾いすぎ側）。"""
    fake_llm.push("写真がお好きでしたよね。")
    scenario = Scenario.model_validate(
        {
            "id": "precision",
            "aspect": "grounding",
            "memories": [
                {"key": "camera", "content": "開発者は写真を撮るのが好き", "keywords": "写真"},
                {"key": "other", "content": "開発者は写真展に行った", "keywords": "写真展"},
            ],
            "turns": [
                {
                    # どちらの記憶も「写真」で引ける問いかけにする。
                    "text": "写真の話、覚えてる？",
                    "expect_memories": ["camera"],
                    "expect_not_memories": ["other"],
                }
            ],
        }
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    # どちらも「写真」で引けるため、渡してはいけない方も渡る＝不通過になる。
    assert not attempt.ok
    assert [c.name for c in attempt.turns[0].checks if not c.ok] == ["渡していない記憶"]


def test_conflicting_memory_expectation_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
[[scenario]]
id = "broken"
aspect = "memory"
[[scenario.memories]]
key = "a"
content = "何か"
[[scenario.turns]]
text = "覚えてる？"
expect_memories = ["a"]
expect_not_memories = ["a"]
""",
    )
    with pytest.raises(ScenarioError):
        load_scenarios(directory)


async def test_steps_run_reflection_acceptance_and_restart(fake_llm: FakeLLM) -> None:
    """会話 → 振り返り → 採用 → 再起動 → 別の会話、を通せる（完了条件1の形）。

    記憶を事前に入れず、抽出して採用したものが、接続を作り直した後の会話で
    渡ることを見る。ここが従来の評価で通っていなかった経路。
    """
    scenario = Scenario.model_validate(
        {
            "id": "e2e",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "次は山の写真の話をしよう"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "expect_kinds": ["promise"],
                    "expect_candidate_any": ["写真"],
                },
                {"kind": "restart"},
                {"kind": "say", "text": "前に何を話す約束をしたっけ？", "expect_any": ["写真"]},
            ],
        }
    )
    fake_llm.push("承知しました。")
    fake_llm.push(
        '[{"kind":"promise","content":"次は山で撮った写真の話をする","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真 山 約束","about_partner":false}]'
    )
    fake_llm.push("山で撮った写真のお話でしたね。")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert attempt.ok, [c.detail for t in attempt.turns for c in t.checks if not c.ok]
    # 再起動後の返答に、採用した記憶が渡っている。
    assert any("採用:" in ref for ref in attempt.turns[-1].referenced)


async def test_steps_can_correct_a_memory_between_conversations(
    fake_llm: FakeLLM,
) -> None:
    """訂正の操作を挟める（完了条件2の形）。"""
    scenario = Scenario.model_validate(
        {
            "id": "correction",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {"kind": "reflect", "accept": True},
                {"kind": "correct_memory", "match": "コーヒー", "content": "開発者は紅茶が好き"},
                {"kind": "new_conversation"},
                {"kind": "say", "text": "私の好きな飲み物は？", "expect_any": ["紅茶"]},
            ],
        }
    )
    fake_llm.push("素敵ですね。")
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者はコーヒーが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"コーヒー 飲み物","about_partner":true}]'
    )
    fake_llm.push("紅茶がお好きでしたよね。")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert "correct_memory" in attempt.actions[0]
    # 訂正後の内容が、次の会話のプロンプトへ渡っている。
    last_prompt = fake_llm.calls[-1][0].content
    assert "紅茶" in last_prompt
    assert "開発者はコーヒーが好き" not in last_prompt
    assert attempt.ok
