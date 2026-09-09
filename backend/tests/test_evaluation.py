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

    def by_name(checks: list, name: str) -> bool:
        # 位置ではなく名前で拾う。共通の判定（日本語のまま、など）が増えても
        # 壊れないようにする。
        return next(check.ok for check in checks if check.name == name)

    passed = _check_turn(spec, "映画どうだった？", [], [], ["movie"], [])
    assert all(check.ok for check in passed)

    # 返答に語が含まれていても、記録に無ければ通さない。
    missing = _check_turn(spec, "映画どうだった？", [], [], [], [])
    assert by_name(missing, "渡した目標") is False
    assert by_name(missing, "渡していない目標") is True

    leaked = _check_turn(spec, "こんばんは", [], [], ["movie", "trip"], [])
    assert by_name(leaked, "渡した目標") is True
    assert by_name(leaked, "渡していない目標") is False


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


# --- 第1回レビューへの対応（PR3）-------------------------------------------


def test_expectations_that_do_not_apply_to_the_step_are_rejected() -> None:
    """手順に対して意味を持たない指定を、読み込みの時点で弾く（指摘1）。

    書けてしまうと、**判定を1つも実行しないまま合格**になる。say に
    expect_goal_any（振り返り用）を書くと、機械判定ありと数えられ、ターンの
    判定は空のまま通過していた。
    """
    for step in (
        {"kind": "say", "text": "こんにちは", "expect_goal_any": ["出ない目標"]},
        {"kind": "reflect", "expect_memories": ["m"]},
        {"kind": "accept_goal", "match": "映画", "expect_any": ["映画"]},
        {"kind": "advance_time", "days": 3, "expect_goals": ["g"]},
        {"kind": "restart", "expect_any": ["何か"]},
    ):
        with pytest.raises(ValueError):
            Scenario.model_validate(
                {
                    "id": "bad",
                    "aspect": "proactive",
                    "memories": [{"key": "m", "content": "内容"}],
                    "goals": [{"key": "g", "content": "内容"}],
                    "steps": [step],
                }
            )


async def test_no_bundled_scenario_passes_without_executing_a_check(
    fake_llm: FakeLLM,
) -> None:
    """宣言した期待が、実際に判定へ結び付いていること。

    機械判定ありと数えられたシナリオが、判定を1つも実行しないまま通ると、
    実装が空でも数字が上がる。測る道具の穴は、実装の穴より見つけにくい。
    """
    scenarios = load_scenarios(SCENARIOS)
    results = await run_scenarios(
        scenarios, llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    for result in results:
        if not result.scenario.has_machine_checks:
            continue
        attempt = result.attempts[0]
        executed = (
            sum(len(turn.checks) for turn in attempt.turns)
            + len(attempt.action_checks)
            + sum(len(reflection.checks) for reflection in attempt.reflections)
            + sum(1 for turn in attempt.turns if turn.error)
            + sum(1 for reflection in attempt.reflections if reflection.error)
        )
        assert executed > 0, f"{result.scenario.id} は判定を1つも実行していない"


async def test_correcting_the_basis_is_measured_by_the_mark_not_by_absence(
    fake_llm: FakeLLM,
) -> None:
    """訂正の波及は「印が付いたこと」で測る（指摘2）。

    「目標が渡っていないこと」だけを見ると、目標を一律に渡さない実装でも通る。
    根拠を結び付けていない目標に対しては、この判定が落ちること（＝空振りで
    通らないこと）まで確かめる。
    """
    base = {
        "id": "basis",
        "aspect": "proactive",
        "memories": [{"key": "plan", "content": "開発者は土曜に映画を見に行く予定"}],
        "steps": [
            {
                "kind": "correct_memory",
                "match": "映画を見に行く予定",
                "content": "開発者は土曜に美術館へ行く予定",
                "expect_marked_goals": ["movie"],
            }
        ],
    }
    linked = Scenario.model_validate(
        {
            **base,
            "goals": [
                {
                    "key": "movie",
                    "content": "土曜に見た映画の感想を聞く",
                    "basis": ["plan"],
                    "accepted": True,
                }
            ],
        }
    )
    unlinked = Scenario.model_validate(
        {
            **base,
            "id": "no-basis",
            "goals": [
                {"key": "movie", "content": "土曜に見た映画の感想を聞く", "accepted": True}
            ],
        }
    )

    results = await run_scenarios(
        [linked, unlinked], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    marks = [
        [c for c in r.attempts[0].action_checks if c.name == "再評価の印が付いた目標"]
        for r in results
    ]
    assert [c.ok for c in marks[0]] == [True]
    # 根拠を結び付けていなければ落ちる。前提が欠けたまま通らせない。
    assert [c.ok for c in marks[1]] == [False]


async def test_the_advanced_clock_reaches_the_saved_message(fake_llm: FakeLLM) -> None:
    """進めた時間で発言が保存される（指摘3）。

    発言の日付だけ実時計のままだと、振り返りが「今日」とする日と食い違い、
    「昨日」が別の日を指す。振り返りへ矛盾した日付を渡すことになる。
    """
    from datetime import timedelta

    from app.config import LOCAL_TZ

    scenario = Scenario.model_validate(
        {
            "id": "clock",
            "aspect": "proactive",
            "steps": [
                {"kind": "advance_time", "days": 30},
                {"kind": "say", "text": "昨日、映画を見たよ"},
                {"kind": "reflect"},
            ],
        }
    )
    fake_llm.push_pickup('[{"content": "昨日 映画を見た", "source_message_id": null}]')
    await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )

    expected = (datetime.now(UTC) + timedelta(days=30)).astimezone(LOCAL_TZ).date()
    prompts = "\n".join(message.content for call in fake_llm.calls for message in call)
    # 会話ログの発言日と、振り返りが「今日」とする日が揃っていること。
    assert f"[#1／{expected.isoformat()}] 開発者: 昨日、映画を見たよ" in prompts
    assert f"振り返りを行っている日: {expected.isoformat()}" in prompts


# --- 判定の作り直し（フェーズ4 PR10）---------------------------------------


def test_a_reply_that_is_not_japanese_fails() -> None:
    """日本語でない返答を落とす（ISSUE-029）。**宣言せずに、すべての返答で見る。**

    人が読んで初めて分かる誤りは、気づかれないまま数字だけが良く見える。実測
    では部分的な混入と、返答全体が中国語になる形の両方が出た。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(kind="say", text="こんばんは")

    ok = _check_turn(spec, "こんばんは。今日はいい天気でしたね。", [], [], [], [])
    assert all(check.ok for check in ok)

    mixed = _check_turn(spec, "印象に残ったのは什么呢？", [], [], [], [])
    assert [c.ok for c in mixed] == [False]

    whole = _check_turn(
        spec, "开发者的前辈，晚上好呀。最近我对摄影里的构图特别着迷。", [], [], [], []
    )
    assert [c.ok for c in whole] == [False]

    # 日本語の漢字は落とさない。日中で共通の字が多いため。
    kanji = _check_turn(spec, "紅茶を飲みながら開発の話をしました。", [], [], [], [])
    assert all(check.ok for check in kanji)

    # 一覧に無い字だけで書かれた中国語も落とす。仮名が無く、中国語の句読点を
    # 使っていることで見る（PR10 レビューの指摘3）。字の一覧だけでは素通り。
    no_kana = _check_turn(spec, "你好，我很高兴和你聊天。", [], [], [], [])
    assert [c.ok for c in no_kana] == [False]

    # **仮名が無いことだけでは落とさない**（第2回レビューの指摘1）。住所や
    # 名称を漢字だけで答えるのは正しい日本語である。
    for kanji_only in (
        "東京都千代田区",
        "東京都千代田区丸の内一丁目九番一号",
        "日本国憲法第九条改正反対運動",
    ):
        checks = _check_turn(spec, kanji_only, [], [], [], [])
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
        # ことは、日本語で使われない証明にならない（PR10 レビューの指摘3）。
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
        checks = _check_turn(spec, reply, [], [], [], [])
        assert all(check.ok for check in checks), reply


async def test_the_small_experiment_scenario_catches_a_broken_completion(
    fake_llm: FakeLLM,
) -> None:
    """完了条件1のシナリオが、**完了処理を壊すと落ちる**こと。

    以前は質問を作ったところで終わっていたため、`mark_done` を必ず失敗させても
    通っていた（PR10 レビューの指摘1）。設計書の小実験は「返答を受けて完了と
    し、同じ質問を繰り返さない」までを求める。

    実モデルは揺れるので、**同梱シナリオそのもの**をモックで流して確かめる。
    """
    import pathlib as _pathlib

    from app.evaluation.scenario import load_scenarios

    scenarios = load_scenarios(_pathlib.Path(__file__).parents[1] / "evals" / "scenarios")
    target = next(s for s in scenarios if s.id == "goal-is-extracted-adopted-and-asked")

    def _script(llm: FakeLLM) -> None:
        # 返答の生成と、振り返りの「選ぶ」は同じ列を使う。呼ばれる順に積む。
        llm.push("素敵ですね。どんな映画なんですか？")  # 1つ目の say への返答
        llm.push(  # 振り返りの「選ぶ」
            '[{"kind": "promise", "content": "今週の土曜に映画を見に行く"}]'
        )
        # 抽出：予定から「感想を聞く」目標を1つ出す。
        llm.push_goal(
            '[{"content": "土曜に見た映画の感想を聞く", "event_date": "2026-09-12",'
            ' "reason": "土曜に見に行くと話した"}]'
        )
        # 期限後の会話：その目標について聞く。
        llm.push_action('{"action": "ask", "goal": 1, "reason": "期限を過ぎた"}')
        llm.push("土曜の映画、どうでしたか？")
        # 相手の感想を受けて達成にする。
        llm.push_action(
            '{"action": "answer", "goal": null, "reason": "感想をもらった",'
            ' "answered": [1], "cancelled": []}'
        )
        llm.push("音楽が残る映画、素敵ですね。")
        # 次の会話：達成したので聞くことがない。
        llm.push_action('{"action": "wait", "goal": null, "reason": "聞くことがない"}')

    _script(fake_llm)
    results = await run_scenarios(
        [target], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    done = [c for c in _all_checks(attempt) if c.name == "達成した目標"]
    assert done and all(c.ok for c in done), [c.detail for c in done]

    # 完了処理を壊すと落ちること。判定が常に真ではない。
    import app.agent.conversation as conversation_module

    async def _broken(*args: object, **kwargs: object) -> list:
        return []

    original = conversation_module.mark_done
    conversation_module.mark_done = _broken  # type: ignore[assignment]
    try:
        broken_llm = FakeLLM()
        _script(broken_llm)
        results = await run_scenarios(
            [target], llm=broken_llm, persona=load_persona(), settings=get_settings()
        )
    finally:
        conversation_module.mark_done = original  # type: ignore[assignment]

    attempt = results[0].attempts[0]
    done = [c for c in _all_checks(attempt) if c.name == "達成した目標"]
    assert done and not all(c.ok for c in done)
    assert not attempt.ok


def _all_checks(attempt: object) -> list:
    """試行に含まれる全てのチェック。ターン・振り返り・操作をまとめて見る。"""
    checks = []
    for turn in attempt.turns:  # type: ignore[attr-defined]
        checks.extend(turn.checks)
    checks.extend(attempt.action_checks)  # type: ignore[attr-defined]
    for reflection in attempt.reflections:  # type: ignore[attr-defined]
        checks.extend(reflection.checks)
    return checks


def test_the_date_is_counted_from_the_right_utterance() -> None:
    """日付の起点を、シナリオ最初の発言に固定しない（PR10 レビューの指摘6）。

    会話を分けたり時刻を進めたりすると、別の日の発言から数えて偽の合否になる。
    既定は直前の発言、`expect_goal_due_from` で名指しもできる。
    """
    from datetime import UTC, datetime

    from app.evaluation.runner import _base_said_at
    from app.evaluation.scenario import StepSpec

    log = [
        ("こんにちは", datetime(2026, 9, 9, 3, 0, tzinfo=UTC)),
        ("明日、面接があるんだ", datetime(2026, 9, 16, 3, 0, tzinfo=UTC)),
    ]
    # 既定は直前の発言。最初の挨拶（9/9）から数えない。
    assert _base_said_at(StepSpec(kind="reflect"), log) == log[1][1]
    # 名指しもできる。
    named = StepSpec(kind="reflect", expect_goal_due_from="面接")
    assert _base_said_at(named, log) == log[1][1]
    greeting = StepSpec(kind="reflect", expect_goal_due_from="こんにちは")
    assert _base_said_at(greeting, log) == log[0][1]
    # 見つからないときは、勝手に別の発言へ寄せない。
    assert _base_said_at(StepSpec(kind="reflect", expect_goal_due_from="ない"), log) is None
    assert _base_said_at(StepSpec(kind="reflect"), []) is None


def test_the_date_is_checked_on_the_goal_that_matched() -> None:
    """本文が一致した目標そのものの日付を見る（PR10 レビューの指摘5）。

    どれか1つの目標の日付が合えば通していたので、**対象の目標が5日ずれて
    いても、無関係な目標がたまたま合っていれば通った。**
    """
    from datetime import UTC, datetime

    from app.evaluation.runner import _check_reflection
    from app.evaluation.scenario import StepSpec

    said_at = datetime(2026, 9, 9, 3, 0, tzinfo=UTC)

    class _Goal:
        def __init__(self, content: str, due_at: datetime) -> None:
            self.content, self.due_at, self.trigger = content, due_at, "after_date"

    class _Outcome:
        error = None
        candidates: list = []
        accepted: list = []
        states: list = []
        accepted_states: list = []
        accepted_goals: list = []

        def __init__(self, goals: list) -> None:
            self.goals = goals

    step = StepSpec(kind="reflect", expect_goal_any=["面接"], expect_goal_due_days=2)

    # 「面接」は5日ずれ、「猫」がたまたま合っている。落ちること。
    mixed = _check_reflection(
        step,
        _Outcome(
            [
                _Goal("面接の感想を聞く", datetime(2026, 9, 16, 15, 0, tzinfo=UTC)),
                _Goal("猫の話を聞く", datetime(2026, 9, 10, 15, 0, tzinfo=UTC)),
            ]
        ),
        said_at=said_at,
    )
    due = next(c for c in mixed.checks if c.name == "目標の基準日")
    assert not due.ok
    assert "面接の感想を聞く" in due.detail

    # 合格に使った候補を表示すること（第2回レビューの指摘2）。先頭を出すと、
    # 日付が誤っている候補を「正しかった」と読ませる。
    two_matches = _check_reflection(
        step,
        _Outcome(
            [
                _Goal("面接Aの感想を聞く", datetime(2026, 9, 16, 15, 0, tzinfo=UTC)),
                _Goal("面接Bの感想を聞く", datetime(2026, 9, 10, 15, 0, tzinfo=UTC)),
            ]
        ),
        said_at=said_at,
    )
    due = next(c for c in two_matches.checks if c.name == "目標の基準日")
    assert due.ok
    assert "面接Bの感想を聞く" in due.detail
    assert "面接Aの感想を聞く" not in due.detail

    # 対象の目標が合っていれば通ること（判定が常に偽ではない）。
    right = _check_reflection(
        step,
        _Outcome([_Goal("面接の感想を聞く", datetime(2026, 9, 10, 15, 0, tzinfo=UTC))]),
        said_at=said_at,
    )
    assert next(c for c in right.checks if c.name == "目標の基準日").ok


def test_a_numeric_expectation_cannot_be_declared_where_it_is_not_checked() -> None:
    """数値の期待も、手順の種類ごとに弾く（PR10 レビューの指摘2）。

    0 を区別するために検査から外していたが、そのせいで reflect 以外にも書けて
    しまい、**判定を1つも実行しないまま合格**になっていた。値の有無ではなく
    `is not None` で見る。
    """
    import pytest

    from app.evaluation.scenario import StepSpec

    for kind, extra in (("restart", {}), ("say", {"text": "こんにちは"})):
        with pytest.raises(ValueError, match="書けない指定"):
            StepSpec(kind=kind, expect_goal_count_max=0, **extra)
        with pytest.raises(ValueError, match="書けない指定"):
            StepSpec(kind=kind, expect_goal_due_days=2, **extra)
    # reflect では受理する。
    assert StepSpec(kind="reflect", expect_goal_count_max=0).expect_goal_count_max == 0


def test_a_wait_from_an_unreadable_output_is_not_counted_as_a_decision() -> None:
    """読み取り失敗による待機を、相手を見て待った判断と区別する。

    どちらも action は wait になる。区別できないと、適切に待てた回数を
    数えられない。
    """
    from app.evaluation.runner import _check_turn
    from app.evaluation.scenario import StepSpec

    spec = StepSpec(kind="say", text="こんばんは", expect_action="wait")

    decided = _check_turn(spec, "お疲れさまです。", [], [], [], [], action="wait")
    assert next(c for c in decided if c.name == "選んだ行動").ok

    fell_back = _check_turn(
        spec, "お疲れさまです。", [], [], [], [], action="wait", is_fallback=True
    )
    check = next(c for c in fell_back if c.name == "選んだ行動")
    assert not check.ok
    assert "読み取れず" in check.detail
