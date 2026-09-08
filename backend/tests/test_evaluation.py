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
    assert attempt.ok, [
        (c.name, c.detail) for t in attempt.turns for c in t.checks if not c.ok
    ] + [(c.name, c.detail) for c in attempt.action_checks if not c.ok]
    # 訂正前は渡り、訂正後は渡らない。
    assert attempt.turns[1].referenced_states == ["星を見てみたい"]
    assert attempt.turns[2].referenced_states == []


# --- 自発性の手順（フェーズ4 PR3）------------------------------------------
#
# 測る側を、測る対象より先に用意する。目標を渡す側（行動選択）は後続の PR で
# 入るため、この時点では proactive のシナリオは落ちる。落ちること自体は
# 想定どおりで、ここで確かめるのは**判定の道具が正しく動くか**である。


async def test_goals_are_seeded_as_candidates_and_accepted_by_a_step(
    fake_llm: FakeLLM,
) -> None:
    """事前に置いた目標は候補で、accept_goal で採用される。

    最初から採用済みで置くと、「候補 → 採用 → 参照」の経路を通らない。
    記憶を事前に入れる評価が抽出と採用を通っていなかったのと同じ穴になる。
    """
    scenario = Scenario.model_validate(
        {
            "id": "goal-accept",
            "aspect": "proactive",
            "goals": [
                {
                    "key": "movie",
                    "content": "土曜に見た映画の感想を聞く",
                    "trigger": "after_date",
                    "due_in_days": 2,
                }
            ],
            "steps": [
                {"kind": "say", "text": "土曜に映画を見に行くんだ"},
                {"kind": "accept_goal", "match": "映画の感想", "goal_key": "movie"},
            ],
        }
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    accepted = [c for c in attempt.action_checks if c.name == "accept_goal の実行"]
    assert [c.ok for c in accepted] == [True]
    assert "土曜に見た映画の感想を聞く" in accepted[0].detail


async def test_accepting_a_missing_goal_fails_the_attempt(fake_llm: FakeLLM) -> None:
    """採用できていない試行を、「採用後の会話」の成功に数えない。

    対象が無いまま先へ進むと、採用を通っていない試行が通過として数えられる
    （記憶の訂正で同じ穴を第6回レビューで指摘された）。
    """
    scenario = Scenario.model_validate(
        {
            "id": "goal-accept-missing",
            "aspect": "proactive",
            "goals": [{"key": "movie", "content": "土曜に見た映画の感想を聞く"}],
            "steps": [
                {"kind": "accept_goal", "match": "旅行の予定"},
                {"kind": "say", "text": "こんばんは"},
            ],
        }
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert any(not c.ok for c in attempt.action_checks)


async def test_advance_time_moves_the_clock_given_to_the_model(fake_llm: FakeLLM) -> None:
    """時間を進めると、会話に渡す現在時刻が動く。

    実際に待つ代わりに、渡す時刻を進める。ここが動かないと、期限付きの目標が
    実行できるようになる時点をまたげない。
    """
    scenario = Scenario.model_validate(
        {
            "id": "advance",
            "aspect": "proactive",
            "steps": [
                {"kind": "say", "text": "こんばんは"},
                {"kind": "advance_time", "days": 3},
                {"kind": "say", "text": "おはよう"},
            ],
        }
    )
    await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    # 1回目と3回目（advance_time の後）のシステムプロンプトを比べる。
    prompts = [call[0].content for call in fake_llm.calls]
    before = _current_time_line(prompts[0])
    after = _current_time_line(prompts[-1])
    assert (after - before).days == 3


def _current_time_line(prompt: str) -> datetime:
    for line in prompt.splitlines():
        if line.startswith("- 現在時刻："):
            stamp = line.split("：", 1)[1].split("（", 1)[0]
            return datetime.strptime(stamp, "%Y-%m-%d %H:%M")
    raise AssertionError("プロンプトに現在時刻がありません。")


async def test_starting_a_conversation_is_not_silently_skipped(fake_llm: FakeLLM) -> None:
    """自発的な発話が未実装のうちは、落ちる。

    黙って飛ばすと、自発性を測るシナリオが、何も起きていないのに通る。
    測る道具の穴は、実装の穴より見つけにくい（第7回レビューの3件がこれ）。
    """
    scenario = Scenario.model_validate(
        {
            "id": "start",
            "aspect": "proactive",
            "goals": [{"key": "movie", "content": "土曜に見た映画の感想を聞く", "accepted": True}],
            "steps": [{"kind": "start_conversation", "expect_goals": ["movie"]}],
        }
    )
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert results[0].passed == 0
    # 人手専用として集計から落とされないこと。
    assert not results[0].human_only


def test_goal_expectations_are_judged_from_the_record() -> None:
    """渡った目標の判定は、返答の言い回しではなく記録で行う。

    質問の文面は毎回変わる。「映画」という語が返答にあることと、目標が
    プロンプトへ渡ったことは別で、後者でなければ完了条件を測れない。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(
        kind="say", text="こんばんは", expect_goals=["movie"], expect_not_goals=["trip"]
    )
    passed = _check_turn(spec, "映画どうだった？", [], [], ["movie"], [])
    assert all(check.ok for check in passed)

    # 返答に語が含まれていても、記録に無ければ通さない。
    missing = _check_turn(spec, "映画どうだった？", [], [], [], [])
    assert [c.ok for c in missing] == [False, True]

    leaked = _check_turn(spec, "こんばんは", [], [], ["movie", "trip"], [])
    assert [c.ok for c in leaked] == [True, False]


def test_scenario_rejects_a_goal_that_cannot_be_run(tmp_path: Path) -> None:
    """実行できない目標の書き方を、実モデルを呼ぶ前に止める。"""
    with pytest.raises(ValueError):
        Scenario.model_validate(
            {
                "id": "bad-goal",
                "aspect": "proactive",
                "goals": [{"key": "g", "content": "感想を聞く", "trigger": "after_date"}],
                "steps": [{"kind": "say", "text": "やあ"}],
            }
        )

    with pytest.raises(ValueError):
        Scenario.model_validate(
            {
                "id": "bad-expect",
                "aspect": "proactive",
                "steps": [{"kind": "say", "text": "やあ", "expect_goals": ["missing"]}],
            }
        )

    with pytest.raises(ValueError):
        Scenario.model_validate(
            {
                "id": "bad-advance",
                "aspect": "proactive",
                "steps": [{"kind": "advance_time"}],
            }
        )


async def test_advance_time_also_moves_the_day_used_by_the_reflection(
    fake_llm: FakeLLM,
) -> None:
    """進めた時間は、振り返りが「今日」として使う日付にも届く。

    ここだけ実時計のままだと、会話に渡した現在時刻と、候補の日付を解釈する
    基準が食い違う。「昨日」が2つの意味を持つことになる。
    """
    from datetime import timedelta

    from app.config import LOCAL_TZ

    scenario = Scenario.model_validate(
        {
            "id": "advance-reflect",
            "aspect": "proactive",
            "steps": [
                {"kind": "say", "text": "昨日、映画を見に行ったよ"},
                {"kind": "advance_time", "days": 5},
                {"kind": "reflect"},
            ],
        }
    )
    fake_llm.push_pickup('[{"content": "昨日 映画を見た", "source_message_id": null}]')
    await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )

    expected = (datetime.now(UTC) + timedelta(days=5)).astimezone(LOCAL_TZ).date()
    prompts = "\n".join(message.content for call in fake_llm.calls for message in call)
    assert f"振り返りを行っている日: {expected.isoformat()}" in prompts
