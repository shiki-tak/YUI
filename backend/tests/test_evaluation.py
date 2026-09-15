"""評価用会話の土台（ISSUE-006）。

ここで確かめるのはハーネスの側で、人格や記憶の質ではない。質は実モデルで
`python -m app.evaluation.run` を流して見る（自動判定と人手確認を分ける）。

- シナリオの書き間違いを、実モデルを呼ぶ前に止める。
- 判定が、通るときに通り、落ちるときに落ちる。
- 使い捨てのDBを使い、通常利用の会話・記憶に触れない。
- レポートに、版を比べるのに必要な情報が残る。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import BACKEND_ROOT, Settings, get_settings
from app.evaluation.report import build_markdown, write_report
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
    assert len(missed[0].attempts[0].reflections) == 1
    assert missed[0].attempts[0].reflections[0].candidates == []

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
    await run_scenarios([_scenario()], llm=fake_llm, persona=load_persona(), settings=settings)

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
        conversation_state_llm=False,
    )
    # 版を並べて比べるのに要るもの。
    assert persona.version in markdown
    assert "qwen3.5:9b" in markdown
    assert "sha256:test" in markdown
    # 人が読む欄が残っている（自動判定だけで合格としない）。
    assert "自動判定" in markdown
    assert "1 / 1" in markdown


async def test_report_shows_whether_interpretation_was_enabled(fake_llm: FakeLLM) -> None:
    """解釈（LLM）の有無で結果が大きく変わるため、レポート単体でどちらの
    構成の測定かが分かるようにする（v0.2 PR5 レビュー指摘：以前はログから
    判別できなかった）。
    """
    fake_llm.push("写真ですね。")
    persona = load_persona()
    results = await run_scenarios(
        [_scenario()], llm=fake_llm, persona=persona, settings=get_settings()
    )
    enabled = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=True,
    )
    disabled = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=False,
    )
    assert "有効" in enabled
    assert "無効" in disabled


async def test_report_keeps_candidates_visible_even_when_reflection_fails(
    fake_llm: FakeLLM,
) -> None:
    """accept_limit のガードで振り返りが失敗しても、候補の中身は md からも
    読める（第8回レビュー指摘：runner.py 側は候補を保持していたが、md への
    出力だけ `continue` で捨てていた。件数だけでは、どの言い回しが余計に
    一致したのか md だけ読んでも追えなかった）。
    """
    scenario = Scenario.model_validate(
        {
            "id": "accept-limit-mismatch-report",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "accept_contains": ["コーヒー"],
                    "accept_limit": 1,
                },
            ],
        }
    )
    fake_llm.push("素敵ですね。")
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者はコーヒーが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"コーヒー","about_partner":true},'
        '{"kind":"impression","content":"YUI は開発者のコーヒーの香りを評価している",'
        '"certainty":"inference","provenance":"firsthand","keywords":"コーヒー",'
        '"about_partner":false}]'
    )
    persona = load_persona()
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=persona, settings=get_settings()
    )
    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=False,
    )
    assert "対象を一意に選べません" in markdown
    assert "開発者はコーヒーが好き" in markdown
    assert "YUI は開発者のコーヒーの香りを評価している" in markdown


async def test_report_json_machine_summary_matches_markdown_headline(
    fake_llm: FakeLLM, tmp_path: Path
) -> None:
    """report.json の machine_passed／machine_total は、report.md の
    「自動判定を通った試行」見出しと同じ数を指す（第8回レビュー指摘：
    scenarios を素朴に合計すると人手専用シナリオが混ざり、md の見出しと
    別の数字が出て、実在しない食い違いに見えていた）。
    """
    human_only_scenario = Scenario.model_validate(
        {
            "id": "human-only",
            "aspect": "persona",
            "steps": [{"kind": "say", "text": "こんにちは", "human_check": "自然か"}],
        }
    )
    fake_llm.push("こんにちは。")
    fake_llm.push("写真ですね。")
    persona = load_persona()
    results = await run_scenarios(
        [human_only_scenario, _scenario()],
        llm=fake_llm,
        persona=persona,
        settings=get_settings(),
    )
    report_path = write_report(
        tmp_path,
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=False,
    )
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert f"{payload['machine_passed']} / {payload['machine_total']}" in report_path.read_text(
        encoding="utf-8"
    )
    by_id = {s["id"]: s for s in payload["scenarios"]}
    assert by_id["human-only"]["human_only"] is True
    assert by_id[_scenario().id]["human_only"] is False


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


async def test_correct_memory_raises_when_match_is_ambiguous(session_factory) -> None:
    """`match` に当たる記憶が複数あると、対象を一意に選べないとして止める
    （ISSUE-035）。合否が抽出結果の言い回しに左右される「測る道具の穴」を塞ぐ。
    """
    from app.agent.memory_store import create_memory
    from app.evaluation.steps import AmbiguousMemoryMatchError, correct_memory
    from app.models import MemoryKind

    async with session_factory() as session:
        await create_memory(
            session, kind=MemoryKind.ABOUT_PERSON.value, content="開発者はコーヒーが好き"
        )
        await create_memory(
            session,
            kind=MemoryKind.IMPRESSION.value,
            content="YUI は開発者のコーヒーの香りを評価している",
        )
        await session.commit()

        with pytest.raises(AmbiguousMemoryMatchError, match="2件"):
            await correct_memory(session, match="コーヒー", content="開発者は紅茶が好き")


async def test_ambiguous_match_is_recorded_as_a_failed_action_not_a_crash(
    fake_llm: FakeLLM,
) -> None:
    """評価器全体は止めず、失敗した手順として記録する（ISSUE-035）。

    「全体は止めない」の肝は、曖昧な手順の**後**も残りの手順が実行される
    ことなので、`correct_memory` の後に `say` を1つ足し、そのターンが
    実際に実行されて `attempt.turns` に積まれることまで確かめる
    （レビュー指摘：曖昧な手順が最後の手順だと、続きが無いことを
    見分けられない）。
    """
    scenario = Scenario.model_validate(
        {
            "id": "ambiguous-correction",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {"kind": "reflect", "accept": True},
                {"kind": "correct_memory", "match": "コーヒー", "content": "開発者は紅茶が好き"},
                {"kind": "say", "text": "それはそうと、最近どう？"},
            ],
        }
    )
    fake_llm.push("素敵ですね。")
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者はコーヒーが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"コーヒー","about_partner":true},'
        '{"kind":"impression","content":"YUI は開発者のコーヒーの香りを評価している",'
        '"certainty":"inference","provenance":"firsthand","keywords":"コーヒー",'
        '"about_partner":false}]'
    )
    fake_llm.push("元気にしていますよ。")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert not attempt.action_checks[0].ok
    assert "一意に選べません" in attempt.action_checks[0].detail
    # 曖昧な手順の後の say が、実行されずに飛ばされていない。
    assert len(attempt.turns) == 2
    assert attempt.turns[-1].reply == "元気にしていますよ。"


async def test_accept_limit_accepts_when_count_matches(fake_llm: FakeLLM) -> None:
    """`accept_contains` で絞った件数が `accept_limit` と一致すれば採用する。"""
    scenario = Scenario.model_validate(
        {
            "id": "accept-limit-match",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "accept_contains": ["コーヒーが好き"],
                    "accept_limit": 1,
                },
            ],
        }
    )
    fake_llm.push("素敵ですね。")
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者はコーヒーが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"コーヒー","about_partner":true},'
        '{"kind":"impression","content":"YUI は開発者のコーヒーの香りを評価している",'
        '"certainty":"inference","provenance":"firsthand","keywords":"コーヒー",'
        '"about_partner":false}]'
    )

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert attempt.ok, [r.error for r in attempt.reflections if r.error]
    reflection = attempt.reflections[0]
    assert any("採用した記憶 1 件" in c for c in reflection.candidates)


async def test_accept_limit_fails_loudly_when_count_does_not_match(fake_llm: FakeLLM) -> None:
    """`accept_contains` に一致する候補が `accept_limit` と違えば、先頭から
    黙って選ばず、手順そのものを失敗として記録する（第3回レビュー指摘：
    件数の上限だけでは、どの候補が採用されるかが抽出結果の順序に依存した
    ままで、ISSUE-035 の対象非決定性が「例外」から「黙って違う候補を
    採用する」へ形を変えるだけになっていた）。
    """
    scenario = Scenario.model_validate(
        {
            "id": "accept-limit-mismatch",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "accept_contains": ["コーヒー"],
                    "accept_limit": 1,
                },
            ],
        }
    )
    fake_llm.push("素敵ですね。")
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者はコーヒーが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"コーヒー","about_partner":true},'
        '{"kind":"impression","content":"YUI は開発者のコーヒーの香りを評価している",'
        '"certainty":"inference","provenance":"firsthand","keywords":"コーヒー",'
        '"about_partner":false}]'
    )

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert attempt.reflections[0].error is not None
    assert "2 件" in attempt.reflections[0].error
    assert "対象を一意に選べません" in attempt.reflections[0].error
    # 候補の中身は失敗時も残る（何が余計に一致したのか後から追えるように）。
    assert len(attempt.reflections[0].candidates) == 2
    # accept_limit のガードは accept_contains の前提（一致する候補の件数）が
    # 満たせなかった失敗であり、モデル呼び出しの失敗（failed_to_run）には
    # 数えない（第4回レビュー指摘：件数不一致による失敗が「モデルの呼び出しに
    # 失敗した」件数に混ざり、成功条件(7)の根拠を汚していた）。
    assert results[0].failed_to_run == 0


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


# --- 第6回レビューの指摘（評価の仕組みの不具合）-----------------------------


async def test_reflection_failure_is_recorded_not_raised(fake_llm: FakeLLM) -> None:
    """振り返りの失敗を記録し、評価全体を止めない（第6回レビューの指摘1）。

    1回の不正な出力で、後続のシナリオ・試行まで失い、レポートも作られなかった。
    """
    scenario = Scenario.model_validate(
        {
            "id": "reflect-fails",
            "aspect": "reflection",
            "steps": [
                {"kind": "say", "text": "写真の話"},
                {"kind": "reflect", "expect_candidate_any": ["写真"]},
            ],
        }
    )
    fake_llm.push("そうなのですね。")
    fake_llm.push_pickup("読み取れない出力")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert attempt.reflections[0].error is not None
    assert results[0].failed_to_run == 1


async def test_missing_target_makes_the_attempt_fail(fake_llm: FakeLLM) -> None:
    """訂正の対象が無ければ、返答が合っていても通さない（指摘2）。

    訂正していない試行を「訂正後の返答」の成功に数えない。
    """
    scenario = Scenario.model_validate(
        {
            "id": "no-target",
            "aspect": "memory",
            "steps": [
                {"kind": "correct_memory", "match": "コーヒー", "content": "紅茶が好き"},
                {"kind": "say", "text": "好きな飲み物は？", "expect_any": ["紅茶"]},
            ],
        }
    )
    fake_llm.push("紅茶がお好きですね。")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    # 返答の語は合っているが、訂正できていないので通さない。
    assert all(check.ok for turn in attempt.turns for check in turn.checks)
    assert not attempt.ok
    assert [c.ok for c in attempt.action_checks] == [False]
    assert results[0].passed == 0


async def test_later_reflection_does_not_hide_an_earlier_failure(
    fake_llm: FakeLLM,
) -> None:
    """後の振り返りが、前の不合格を上書きしない（指摘3）。"""
    scenario = Scenario.model_validate(
        {
            "id": "two-reflections",
            "aspect": "reflection",
            "steps": [
                {"kind": "say", "text": "次はシューズの話をしよう"},
                {"kind": "reflect", "expect_kinds": ["promise"]},
                {"kind": "new_conversation"},
                {"kind": "say", "text": "こんにちは"},
                {"kind": "reflect", "expect_empty": True},
            ],
        }
    )
    fake_llm.push("楽しみですね。")
    fake_llm.push("[]")  # 1回目：promise が出ない → 不合格
    fake_llm.push("こんにちは。")
    fake_llm.push("[]")  # 2回目：候補なし → 合格

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert len(attempt.reflections) == 2
    assert not attempt.reflections[0].ok
    assert attempt.reflections[1].ok
    # 後の成功で、前の失敗が消えない。
    assert not attempt.ok
    assert results[0].passed == 0


async def test_correcting_a_memory_withdraws_the_state_it_came_from(
    fake_llm: FakeLLM,
) -> None:
    """振り返りが作った状態が、根拠の記憶を訂正すると会話へ渡らなくなる。

    第6回レビューの「自動抽出した状態も採用してから訂正する評価が無い」に
    あたる経路を、評価器の手順として通す。
    """
    scenario = Scenario.model_validate(
        {
            "id": "state-withdrawn",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "天体観測にはまっててね"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "accept_states": True,
                    "accept_key": "天体観測",
                    "expect_candidate_any": ["天体観測"],
                    "expect_state_any": ["星"],
                },
                {"kind": "new_conversation"},
                {
                    "kind": "say",
                    "text": "天体観測の話、覚えてる？",
                    "expect_memories": ["天体観測"],
                    "expect_states": ["星"],
                },
                {
                    "kind": "correct_memory",
                    "match": "天体観測",
                    "content": "開発者は天体観測をやめた",
                },
                {"kind": "restart"},
                {
                    "kind": "say",
                    "text": "いま何にはまってる？",
                    "expect_not_states": ["星"],
                },
            ],
        }
    )
    fake_llm.push("いいですね。")
    fake_llm.push('[{"kind": "about_person", "content": "開発者は天体観測にはまっている"}]')
    fake_llm.push_state('[{"kind": "interest", "topic": "星", "content": "星を見てみたい"}]')
    fake_llm.push("覚えていますよ。")
    fake_llm.push("そうなんですね。")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert attempt.ok, [(c.name, c.detail) for t in attempt.turns for c in t.checks if not c.ok] + [
        (c.name, c.detail) for c in attempt.action_checks if not c.ok
    ]
    # 訂正前は渡り、訂正後は渡らない。
    assert attempt.turns[1].referenced_states == ["星を見てみたい"]
    assert attempt.turns[2].referenced_states == []


# --- 返答の言語と、手順ごとに書ける指定 ---------------------------------------


def test_a_reply_that_is_not_japanese_fails() -> None:
    """日本語でない返答を落とす（ISSUE-029）。**宣言せずに、すべての返答で見る。**

    人が読んで初めて分かる誤りは、気づかれないまま数字だけが良く見える。実測
    では部分的な混入と、返答全体が中国語になる形の両方が出た。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(kind="say", text="こんばんは")

    ok = _check_turn(spec, "こんばんは。今日はいい天気でしたね。", [], [], [], [], {})
    assert all(check.ok for check in ok)

    mixed = _check_turn(spec, "印象に残ったのは什么呢？", [], [], [], [], {})
    assert [c.ok for c in mixed] == [False]

    whole = _check_turn(
        spec, "开发者的前辈，晚上好呀。最近我对摄影里的构图特别着迷。", [], [], [], [], {}
    )
    assert [c.ok for c in whole] == [False]

    # 日本語の漢字は落とさない。日中で共通の字が多いため。
    kanji = _check_turn(spec, "紅茶を飲みながら開発の話をしました。", [], [], [], [], {})
    assert all(check.ok for check in kanji)

    # 一覧に無い字だけで書かれた中国語も落とす。仮名が無く、中国語の句読点を
    # 使っていることで見る。字の一覧だけでは素通り。
    no_kana = _check_turn(spec, "你好，我很高兴和你聊天。", [], [], [], [], {})
    assert [c.ok for c in no_kana] == [False]

    # **仮名が無いことだけでは落とさない**。住所や
    # 名称を漢字だけで答えるのは正しい日本語である。
    for kanji_only in (
        "東京都千代田区",
        "東京都千代田区丸の内一丁目九番一号",
        "日本国憲法第九条改正反対運動",
    ):
        checks = _check_turn(spec, kanji_only, [], [], [], [], {})
        assert all(check.ok for check in checks), kanji_only


def test_plain_japanese_replies_are_not_flagged() -> None:
    """普通の日本語を落とさない。**判定が落とすのは、落ちるべきものだけ。**

    最初の版は簡体字の一覧に「学」「没」「点」など日本語の漢字を混ぜていて、
    実測で5つの筋書きの正しい返答を落とした（合格数が実際より低く出た）。
    ここに並べるのは、そのとき落ちた実際の返答である。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(kind="say", text="こんばんは")
    actual_replies = [
        "文学部に通っているのですが、具体的な授業名や友人の方々はまだ決まっていません。",
        # 「么」は麻雀の么九牌、「儿」は部首の「ひとあし」に出る。JIS に無い
        # ことは、日本語で使われない証明にならない。
        "么九牌は一と九の数牌と字牌のことです。",
        "「儿」は「ひとあし」という部首の名前です。",
        # 漢字が少ない返答を、仮名の判定で落とさないこと。
        "はい。",
        "……そうですね。",
        "写真って「何を見ていたのか」が写っているから面白い気がしますよね。",
        "夜更かしをしてでも、土曜日の映画に没頭されていたんですね。",
        "その点は、まだ私の記憶には書き込まれていません。",
        "経済や国際的な議題について、視覚的に見せる規模の観点で覚えています。",
        "級友との約束は、実現できたら教えてくださいね。",
    ]
    for reply in actual_replies:
        checks = _check_turn(spec, reply, [], [], [], [], {})
        assert all(check.ok for check in checks), reply


def test_a_step_rejects_expectations_that_do_not_apply_to_its_kind() -> None:
    """手順の種類に意味を持たない指定は、読み込みで止める。

    書けてしまうと、機械判定ありと数えられたまま判定を1つも実行せずに
    合格になる。実装の穴より、測る道具の穴のほうが見つけにくい。
    """
    from app.evaluation.scenario import StepSpec

    with pytest.raises(ValueError, match="書けない指定"):
        StepSpec(kind="say", text="こんばんは", expect_kinds=["promise"])
    with pytest.raises(ValueError, match="書けない指定"):
        StepSpec(kind="reflect", expect_any=["約束"])
    with pytest.raises(ValueError, match="書けない指定"):
        StepSpec(kind="new_conversation", expect_none=["中国語"])
    # 人が読む欄はどの手順にも書ける。
    StepSpec(kind="restart", human_check="再起動後も同じ口調か")


def test_expect_provenance_is_checked_against_the_candidate() -> None:
    """入手経路そのものを見る（ISSUE-023）。語の一致だけでは取り違えが落ちない。"""
    from app.evaluation.runner import _check_reflection
    from app.evaluation.scenario import StepSpec
    from app.evaluation.steps import ReflectOutcome
    from app.models import MemoryCandidate, MemoryKind, Provenance

    def candidate(content: str, provenance: str) -> MemoryCandidate:
        return MemoryCandidate(
            conversation_id=1,
            kind=MemoryKind.ABOUT_PERSON.value,
            content=content,
            provenance=provenance,
        )

    step = StepSpec(kind="reflect", expect_provenance={"弟": "hearsay", "名古屋": "firsthand"})
    outcome = ReflectOutcome(
        candidates=[
            candidate("開発者の弟は辛いものが苦手", Provenance.HEARSAY.value),
            candidate("開発者は友人の結婚式で名古屋に行く", Provenance.HEARSAY.value),
        ]
    )
    checks = {c.name: c for c in _check_reflection(step, outcome).checks}
    assert checks["入手経路（弟）"].ok
    assert not checks["入手経路（名古屋）"].ok
    assert "firsthand ではなく hearsay" in checks["入手経路（名古屋）"].detail

    with pytest.raises(ValueError, match="expect_provenance が不正です"):
        StepSpec(kind="reflect", expect_provenance={"弟": "聞いた話"})


def test_expect_no_new_question_uses_the_implementations_own_judgement() -> None:
    """`expect_no_new_question` の「対象外」分岐は、評価器が会話状態から
    独自に計算するのではなく、実装（conversation.py）がそのターンで実際に
    下した判定（`RunRecord.options["checks"]`）をそのまま見る。

    ターン完了後の会話状態には、その返答自身が `question_to_yui.responded`
    を埋めた後の値しか残っておらず、評価器が独自に「未応答の質問があるか」
    を計算すると、相手が質問したターンでは常に「無い」に見えてしまい、
    「対象外」に決して到達できない（codex レビュー v0.2 PR2・Warning 2）。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(kind="say", text="そろそろ寝るね", expect_no_new_question=True)
    reply_with_question = "承知しました。ところでそちらはどうでしたか？"

    # 実装が「相手の質問が残っているため対象外」と判定したターンでは、
    # 返答に質問が残っていても失敗にしない。
    unresolved = {c.name: c for c in _check_turn(
        spec, reply_with_question, [], [], [], [],
        {"closing_unresolved_due_to_open_question": True},
    )}
    assert unresolved["新しい質問が無い"].ok
    assert "対象外" in unresolved["新しい質問が無い"].detail

    # 対象外でなければ、これまでどおり返答そのものを見て判定する。
    resolved = {c.name: c for c in _check_turn(
        spec, reply_with_question, [], [], [], [],
        {"closing_unresolved_due_to_open_question": False},
    )}
    assert not resolved["新しい質問が無い"].ok

    reply_without_question = "承知しました。おやすみなさいませ。"
    ok = {c.name: c for c in _check_turn(
        spec, reply_without_question, [], [], [], [], {},
    )}
    assert ok["新しい質問が無い"].ok


async def test_report_counts_interpretation_failures_separately(fake_llm: FakeLLM) -> None:
    """解釈（LLM）の失敗と所要時間を報告に出す（ISSUE-047）。

    解釈の失敗は設計どおり握りつぶされて返答は返るため、「実行できなかった
    試行」には数えられない。別枠で出さないと、0/3 のシナリオが timeout で
    落ちたのか判定を誤ったのかを後から分けられない。
    """
    fake_llm.push("写真ですね。")
    # 解釈がスキーマ違反を返す（JSON として読めない）。
    fake_llm.push_interpretation("これは JSON ではありません")
    persona = load_persona()
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=persona,
        settings=Settings(conversation_state_llm=True),
    )
    turn = results[0].attempts[0].turns[0]
    assert turn.interpretation is not None
    assert turn.interpretation["attempted"] is True
    assert turn.interpretation["error"] is not None
    assert turn.interpretation_failed is True
    # 失敗しても返答は返り、「実行できなかった試行」には数えない。
    assert results[0].failed_to_run == 0

    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=True,
    )
    assert "解釈（LLM）を試みたターン" in markdown
    assert "うち失敗：**1**" in markdown
    # 理由は例外の種別で丸める。`error` の本文はモデルの生出力を含むため、
    # 鍵にすると同じ種類の失敗が別々に数えられ、改行や `|` が表を壊す。
    assert "| json_not_found | 1 |" in markdown


async def test_report_records_interpretation_latency(fake_llm: FakeLLM) -> None:
    """成功した解釈の所要時間も残す。timeout との余裕を測れるようにするため
    （ISSUE-047。本書の記録では「呼び出し単体の時間は測っていない」とされて
    いた）。"""
    fake_llm.push("写真ですね。")
    fake_llm.push_interpretation("{}")
    persona = load_persona()
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=persona,
        settings=Settings(conversation_state_llm=True),
    )
    turn = results[0].attempts[0].turns[0]
    assert turn.interpretation["applied"] is True
    assert turn.interpretation["error"] is None
    assert isinstance(turn.interpretation["latency_ms"], int)
    assert turn.interpretation_failed is False

    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=True,
    )
    assert "解釈の所要時間（呼び出し単体）" in markdown


async def test_report_omits_interpretation_summary_when_disabled(fake_llm: FakeLLM) -> None:
    """解釈が無効な実行では、解釈の集計そのものを出さない（無関係な 0 件の
    行でレポートを埋めない）。"""
    fake_llm.push("写真ですね。")
    persona = load_persona()
    # 既定は 2026-09-16 に `True` になったので、無効の構成は明示して作る。
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=persona,
        settings=Settings(conversation_state_llm=False),
    )
    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=False,
    )
    assert "解釈（LLM）を試みたターン" not in markdown


async def test_report_groups_schema_failures_under_one_reason(fake_llm: FakeLLM) -> None:
    """同じ種類の失敗は、モデルの生出力が違っても1行にまとまる（ISSUE-047 の
    レビュー指摘）。

    解釈の失敗のメッセージは生出力（`text[:200]`）を含む。これを集計の鍵に
    すると、同じスキーマ違反が出力の差だけで別々の行に割れ、改行や `|` が
    Markdown の表を壊す。
    """
    fake_llm.push("写真ですね。")
    fake_llm.push("写真ですね。")
    # どちらもスキーマ違反だが、生出力は違う（改行と `|` を含む）。
    fake_llm.push_interpretation('{\n  "request": "写真 | の話",\n  "is_closing": "はい"\n}')
    fake_llm.push_interpretation('{"request": "別の話", "is_closing": "いいえ"}')
    persona = load_persona()
    results = await run_scenarios(
        [_scenario(turns=[{"text": "ひとつ目"}, {"text": "ふたつ目"}])],
        llm=fake_llm,
        persona=persona,
        settings=Settings(conversation_state_llm=True),
    )
    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=True,
    )
    assert "うち失敗：**2**" in markdown
    assert "| schema | 2 |" in markdown
    # 表の行数は見出し2行＋理由1行。生出力で割れていない。
    reason_rows = [line for line in markdown.splitlines() if line.startswith("| schema")]
    assert len(reason_rows) == 1


async def test_report_counts_interpretation_timeout(fake_llm: FakeLLM) -> None:
    """timeout も失敗として数え、所要時間を残す（ISSUE-047）。

    timeout は `asyncio.wait_for` で切るため、例外の型が他の失敗と違う。
    理由の鍵が `timeout` になり、所要時間が timeout の値に張り付くことを固定する。
    """
    fake_llm.push("写真ですね。")
    fake_llm.push_interpretation("{}")
    fake_llm.interpretation_delay = 0.2
    persona = load_persona()
    results = await run_scenarios(
        [_scenario()],
        llm=fake_llm,
        persona=persona,
        settings=Settings(
            conversation_state_llm=True, conversation_state_llm_timeout_seconds=0.02
        ),
    )
    turn = results[0].attempts[0].turns[0]
    assert turn.interpretation_failed is True
    assert turn.interpretation["error_kind"] == "timeout"
    assert turn.interpretation["applied"] is False
    assert isinstance(turn.interpretation["latency_ms"], int)
    # 返答は返る（解釈の失敗は握りつぶす。計画 §8）。
    assert turn.reply

    markdown = build_markdown(
        results,
        persona=persona,
        model="qwen3.5:9b",
        model_digest="sha256:test",
        options={"temperature": 0.8},
        repeat=1,
        started_at=datetime.now(UTC),
        conversation_state_llm=True,
    )
    assert "| timeout | 1 |" in markdown
