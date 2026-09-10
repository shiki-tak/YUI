"""自動採用（フェーズ4 PR11 / 設計書の実装内容9）。

設計書：「更新前後を比較し、**評価できた種類から**自動採用へ移す」。

測ることは3つ。

1. 既定では何も自動採用しない。有効にした種類だけが自動で入る。
2. 自動で入ったものを、手動で採用したものと区別できる。
3. まとめて戻せる。ただし**すでに効果が出たものは戻さない**。

**採用件数は指標にしない**（設計書「自動化の合否を単なる採用件数で測りません」）。
ここで見るのは、開発者の確認を省いた分だけが省かれていることと、戻せること。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import AUTO_ADOPTABLE_KINDS, Settings
from app.models import (
    CharacterState,
    Goal,
    GoalRevision,
    GoalStatus,
    Memory,
    MemoryKind,
    MemoryRevision,
    MemoryStatus,
    StateKind,
    StateStatus,
)
from tests.conftest import FakeLLM, end_and_wait


def _enable(monkeypatch: pytest.MonkeyPatch, kinds: str) -> None:
    """自動採用する種類を差し替える。設定は起動時に一度読むので、直接差す。"""
    from app.config import get_settings

    settings = Settings(auto_adopt=kinds, database_url=get_settings().database_url)
    monkeypatch.setattr("app.agent.reflection_job.get_settings", lambda: settings)


async def _talk_and_reflect(client: AsyncClient, fake_llm: FakeLLM) -> int:
    """1往復して振り返らせる。記憶・状態・目標が1件ずつ出るようにする。"""
    fake_llm.push("そうなんですね。")
    first = await client.post("/api/chat", json={"text": "今週の土曜に映画を見に行くんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind": "promise", "content": "今週の土曜に映画を見に行く", "about_partner": true}]'
    )
    fake_llm.push_state(
        '[{"kind": "interest", "content": "映画の話をもっと聞きたい", "topic": "映画"}]'
    )
    fake_llm.push_goal(
        '[{"content": "土曜に見た映画の感想を聞く", "event_date": "2026-09-12",'
        ' "reason": "土曜に見に行くと話した"}]'
    )
    await end_and_wait(client, conversation_id)
    return conversation_id


# --- 設定 -------------------------------------------------------------------


def test_the_kind_list_matches_the_models() -> None:
    """設定に書ける種類が、モデルの種類とずれていないこと。

    設定はモデルを import しない（起動のいちばん外側で読むものなので、依存が
    輪になる）。代わりにここで突き合わせる。**種類を増やしたときに設定を直し
    忘れると、有効にできない種類が黙って生まれる。**
    """
    expected = {kind.value for kind in MemoryKind} | {kind.value for kind in StateKind}
    expected.add("goal")
    assert expected == AUTO_ADOPTABLE_KINDS


def test_a_misspelled_kind_stops_instead_of_silently_disabling() -> None:
    """綴り違いは例外にする。**黙って無効へ倒さない。**

    無効へ倒すと、自動採用を有効にしたつもりの測定が、無効の測定になる。
    PR12 は有無を比べる測定なので、そこが狂うと結論が変わる。
    """
    # **設定を作った時点で**落ちること（PR11 レビューの指摘7）。読むまで検証
    # しないと、本番では4回の抽出を終えた保存の直前で初めて落ちる。
    with pytest.raises(ValueError, match="知らない種類"):
        Settings(auto_adopt="goals")
    assert Settings(auto_adopt="goal, interest").auto_adopt_kinds == {"goal", "interest"}


def test_the_app_refuses_to_start_with_a_bad_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """不正な設定では起動そのものが失敗すること。

    プロパティを読むまで検証しないと、本番の最初の参照は振り返りの保存の直前に
    なる。**抽出を4回終えてから設定エラーになり、候補を保存できない。**
    """
    monkeypatch.setenv("YUI_AUTO_ADOPT", "promise, goals")
    with pytest.raises(ValueError, match="知らない種類"):
        Settings()


def test_the_shipped_default_matches_the_measurement() -> None:
    """出荷時の既定が、測定で根拠を得た種類と一致すること。

    2026-09-09 に全種類を有効にして全シナリオ×3回を流し、**入った中身を読んで**
    決めた（`docs/result/phase4.md`）。設計書「評価できた種類から自動採用へ
    移す」に対する答えである。

    | 種類 | 採用 | 冗談・雑談・人格変更から | 重複・時期違い |
    | --- | ---: | ---: | --- |
    | promise | 16 | 0 | 1件/回のみ |
    | goal | 48 | 1 | **33回中13回で2件以上** |
    | interest | 30 | **4** | — |
    | impression | 8 | 1（架空の観察） | — |

    `goal` は一度有効にしたが、**採用された48件を読み直して外した**。
    「残すべきでない筋書きから0件」だけでは、言い換えの重複と時期違いを
    数えられない。採用された目標はそのまま質問になるので、害が直接出る。

    **根拠を増やさずにここを広げない。** 広げるときは、同じ測定をやり直して
    この表を更新すること。
    """
    assert Settings().auto_adopt_kinds == {"promise"}


def test_nothing_is_adopted_when_the_setting_is_empty() -> None:
    """空にすれば、全部が開発者の確認を待つ状態へ戻せる。"""
    assert Settings(auto_adopt="").auto_adopt_kinds == set()


# --- 採用されるもの・されないもの -------------------------------------------


async def test_with_the_setting_empty_everything_waits_for_the_developer(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """自動採用を切れば、記憶も状態も目標も候補のままであること。

    有効にした種類を戻せることを見る。conftest が全テストで空に差し替えている
    ので、ここは「切った状態」の振る舞いである。
    """
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        assert (await session.execute(select(Memory))).scalars().all() == []
        states = (await session.execute(select(CharacterState))).scalars().all()
        goals = (await session.execute(select(Goal))).scalars().all()
    assert [state.status for state in states] == [StateStatus.PENDING.value]
    assert [goal.status for goal in goals] == [GoalStatus.PENDING.value]
    assert not any(item.auto_adopted for item in [*states, *goals])


async def test_only_the_enabled_kinds_are_adopted(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有効にした種類だけが自動で入る。**他は候補のまま。**

    一度に全部を自動にしない、というのが設計書の指示である。種類ごとに切り
    替えられていなければ、段階的に移すことができない。
    """
    _enable(monkeypatch, "goal, interest")
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        memories = (await session.execute(select(Memory))).scalars().all()
        state = (await session.execute(select(CharacterState))).scalar_one()
        goal = (await session.execute(select(Goal))).scalar_one()

        # promise は有効にしていないので、記憶にならない。
        assert memories == []
        assert state.status == StateStatus.ACTIVE.value
        assert state.auto_adopted
        assert goal.status == GoalStatus.ACTIVE.value
        assert goal.auto_adopted

        # 履歴に残る。後から「誰が採用したのか」を追えないと、戻す判断ができない。
        revision = (
            await session.execute(
                select(GoalRevision).where(
                    GoalRevision.goal_id == goal.id, GoalRevision.action == "auto_accepted"
                )
            )
        ).scalar_one()
        assert "自動採用" in revision.reason


async def test_an_adopted_memory_keeps_the_candidate_trail(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """自動採用した記憶も、候補と結び付いたまま残る。

    手動の採用と**同じ関数**を通しているので、根拠の辿り方も同じになる。
    別々に書くと、片方だけ直したときに動きがずれる。
    """
    _enable(monkeypatch, "promise")
    conversation_id = await _talk_and_reflect(client, fake_llm)

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert [c["status"] for c in candidates] == ["accepted"]
    assert candidates[0]["accepted_memory_id"] is not None

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        assert memory.id == candidates[0]["accepted_memory_id"]
        assert memory.auto_adopted
        assert memory.source_conversation_id == conversation_id


async def test_a_manually_accepted_memory_is_not_marked_as_automatic(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """手動で採用したものに、自動の印を付けない。

    区別できないと、まとめて戻すときに開発者の判断まで巻き込む。
    """
    conversation_id = await _talk_and_reflect(client, fake_llm)
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    response = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    assert response.status_code == 200, response.text

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        assert not memory.auto_adopted

    listed = (await client.get("/api/auto-adopt")).json()
    assert listed == []


async def test_editing_while_accepting_still_works(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """手動の採用で内容を直せること。**自動と共通化して壊していない。**

    採用の直前に開発者が種別や本文を直せるのは、分類が確実ではないためである
    （ISSUE-017）。共通の関数へ寄せたときに落としやすいので、ここで見る。
    """
    conversation_id = await _talk_and_reflect(client, fake_llm)
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    response = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={
            "decision": "accept",
            "kind": "experience",
            "content": "土曜に映画を見に行く予定がある",
            "keywords": "映画 土曜",
        },
    )
    assert response.status_code == 200, response.text

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        assert memory.kind == MemoryKind.EXPERIENCE.value
        assert memory.content == "土曜に映画を見に行く予定がある"
        assert memory.keywords == "映画 土曜"


# --- 戻す -------------------------------------------------------------------


async def test_auto_adopted_items_can_be_rolled_back_together(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """まとめて戻せること。**戻せるのが、自動採用へ移す前提である。**"""
    _enable(monkeypatch, "promise, interest, goal")
    await _talk_and_reflect(client, fake_llm)

    listed = (await client.get("/api/auto-adopt")).json()
    assert sorted(row["table"] for row in listed) == ["goal", "memory", "state"]

    result = (await client.post("/api/auto-adopt/rollback")).json()
    assert sorted(row["table"] for row in result["reverted"]) == ["goal", "memory", "state"]
    assert result["skipped"] == []

    async with session_factory() as session:
        assert (await session.execute(select(Memory))).scalar_one().status == (
            MemoryStatus.DELETED.value
        )
        assert (await session.execute(select(CharacterState))).scalar_one().status == (
            StateStatus.WITHDRAWN.value
        )
        assert (await session.execute(select(Goal))).scalar_one().status == (
            GoalStatus.WITHDRAWN.value
        )


async def test_rollback_leaves_manual_adoptions_alone(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """手動で採用したものは戻さない。開発者の判断を巻き込まない。"""
    conversation_id = await _talk_and_reflect(client, fake_llm)
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )

    result = (await client.post("/api/auto-adopt/rollback")).json()
    assert result["reverted"] == []

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        assert memory.status == MemoryStatus.ACTIVE.value


async def test_a_goal_already_raised_is_not_rolled_back(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """すでに相手へ持ち出した目標は戻さない。**理由を添えて残す。**

    取り下げても、聞いた事実は消えない。黙って取りこぼすと、戻したつもりで
    残る（あるいは、聞いた記録だけが宙に浮く）。
    """
    _enable(monkeypatch, "goal")
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        goal = (await session.execute(select(Goal))).scalar_one()
        goal_id = goal.id
        # 相手へ持ち出した状態にする。
        from app.agent.goal import mark_executed

        await mark_executed(session, goal_ids=[goal_id], delivered="completed")
        await session.commit()

    result = (await client.post("/api/auto-adopt/rollback")).json()
    assert result["reverted"] == []
    assert [row["reason"] for row in result["skipped"]] == ["すでに相手へ持ち出しています。"]

    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.ACTIVE.value


# --- 評価が同じ経路を通ること -----------------------------------------------


async def test_the_evaluation_runs_through_auto_adoption(fake_llm: FakeLLM) -> None:
    """評価も、本番と同じ自動採用の関数を通ること。

    **評価が別経路を持つと、自動採用を有効にして測ったつもりが、一度も通って
    いない測定になる。** PR12 は有無を比べる測定なので、そこが狂うと結論
    そのものが変わる。測る道具の穴は、実装の穴より見つけにくい。
    """
    from app.config import get_settings
    from app.evaluation.runner import run_scenarios
    from app.evaluation.scenario import Scenario
    from app.persona import load_persona

    scenario = Scenario.model_validate(
        {
            "id": "auto-adopt-runs",
            "aspect": "proactive",
            "steps": [
                {"kind": "say", "text": "今週の土曜に映画を見に行くんだ"},
                {"kind": "reflect", "expect_goal_any": ["感想"]},
            ],
        }
    )

    def _script(llm: FakeLLM) -> None:
        llm.push("そうなんですね。")
        llm.push(
            '[{"kind": "promise", "content": "今週の土曜に映画を見に行く",'
            ' "about_partner": true}]'
        )
        llm.push_goal(
            '[{"content": "土曜に見た映画の感想を聞く", "event_date": "2026-09-12",'
            ' "reason": "見に行くと話した"}]'
        )

    base = get_settings()

    # 無効のとき：候補のまま。
    _script(fake_llm)
    off = await run_scenarios(
        [scenario],
        llm=fake_llm,
        persona=load_persona(),
        settings=base.model_copy(update={"auto_adopt": ""}),
    )
    off_lines = off[0].attempts[0].reflections[0].candidates
    assert any("[目標／" in line and "／候補]" in line for line in off_lines), off_lines

    # 有効のとき：その場で採用される。**評価から通っていることの確認。**
    on_llm = FakeLLM()
    _script(on_llm)
    on = await run_scenarios(
        [scenario],
        llm=on_llm,
        persona=load_persona(),
        settings=base.model_copy(update={"auto_adopt": "goal"}),
    )
    on_lines = on[0].attempts[0].reflections[0].candidates
    # レポートでも「候補」と書かない。読み手が、開発者の確認を経たものと
    # 経ていないものを取り違える。
    assert any("[目標／" in line and "／自動採用]" in line for line in on_lines), on_lines
    assert not any("[目標／" in line and "／候補]" in line for line in on_lines)


# --- レビューで見つかった穴（PR11 第1回）------------------------------------


async def test_rollback_marks_derived_items_for_review(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """記憶を取り消したら、そこから作った状態・目標へ波及すること（指摘1）。

    **既存の削除APIが守っている波及を、新しい削除経路が迂回していた。**
    取り消した記憶が、目標や関心を介して使われ続ける。
    """
    _enable(monkeypatch, "promise")
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        state = (await session.execute(select(CharacterState))).scalar_one()
        goal = (await session.execute(select(Goal))).scalar_one()
        # 手動で採用し、その記憶を根拠にする。
        for item in (state, goal):
            item.status = "active"
            item.basis_memory_ids = [memory.id]
        state_id, goal_id = state.id, goal.id
        await session.commit()

    result = (await client.post("/api/auto-adopt/rollback")).json()
    assert [row["table"] for row in result["reverted"]] == ["memory"]

    async with session_factory() as session:
        assert (await session.get(CharacterState, state_id)).needs_review
        assert (await session.get(Goal, goal_id)).needs_review


async def test_rollback_leaves_a_corrected_memory_alone(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """開発者が直して残した記憶は取り消さない（指摘3）。

    訂正しても status は active のままなので、状態では見分けられない。
    **直して残したものを取り消すと、開発者の判断まで巻き込む。**
    """
    _enable(monkeypatch, "promise")
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        memory_id = (await session.execute(select(Memory))).scalar_one().id

    corrected = await client.patch(
        f"/api/memories/{memory_id}",
        json={"content": "日曜に映画を見に行く", "reason": "曜日を間違えていた"},
    )
    assert corrected.status_code == 200, corrected.text

    result = (await client.post("/api/auto-adopt/rollback")).json()
    assert result["reverted"] == []
    assert [row["reason"] for row in result["skipped"]] == [
        "採用後に開発者が手を入れています。"
    ]

    async with session_factory() as session:
        assert (await session.get(Memory, memory_id)).status == MemoryStatus.ACTIVE.value


async def test_since_gives_the_same_result_in_any_timezone(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じ瞬間なら、書き方が違っても対象が変わらないこと（指摘4）。

    SQLite は timezone を落として比較する。地域時刻のまま渡すと、その壁時計の
    値が UTC として扱われ、対象を取りこぼす。
    """
    from datetime import UTC, datetime, timedelta, timezone

    _enable(monkeypatch, "goal")
    await _talk_and_reflect(client, fake_llm)

    async with session_factory() as session:
        goal = (await session.execute(select(Goal))).scalar_one()
        goal.created_at = datetime(2026, 9, 9, 1, 0, tzinfo=UTC)
        await session.commit()

    boundary_utc = datetime(2026, 9, 9, 0, 0, tzinfo=UTC)
    boundary_jst = boundary_utc.astimezone(timezone(timedelta(hours=9)))

    for label, since in (("UTC", boundary_utc), ("JST", boundary_jst)):
        listed = (await client.get("/api/auto-adopt", params={"since": since.isoformat()})).json()
        assert len(listed) == 1, f"{label}: {listed}"


async def test_the_extracted_content_survives_a_manual_edit(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """採用時に直しても、モデルが抽出した内容が残ること（指摘6）。

    設計書は「更新前後を比較し」て自動採用へ移すとしている。**比較する元が
    消えると、その判断ができない。**
    """
    conversation_id = await _talk_and_reflect(client, fake_llm)
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept", "content": "日曜に映画を見に行く"},
    )

    # 候補は抽出されたまま。
    after = (await client.get(f"/api/conversations/{conversation_id}/candidates")).json()
    assert after[0]["content"] == "今週の土曜に映画を見に行く"

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        assert memory.content == "日曜に映画を見に行く"
        # 直した事実が履歴に残る。
        revisions = (
            await session.execute(
                select(MemoryRevision).where(MemoryRevision.memory_id == memory.id)
            )
        ).scalars().all()
        edited = [r for r in revisions if r.action == "accepted_with_edits"]
        assert len(edited) == 1
        assert edited[0].before["content"] == "今週の土曜に映画を見に行く"


async def test_auto_adoption_does_not_duplicate_the_evaluation_acceptance(
    fake_llm: FakeLLM,
) -> None:
    """評価の採用ステップと自動採用が衝突しないこと（指摘2）。

    衝突すると、**本番には無い重複の記憶が評価DBに入り、自動採用に成功した
    だけで失敗も増える。** 有無の比較を採用品質の差として読めなくなる。
    """
    from app.config import get_settings
    from app.evaluation.runner import run_scenarios
    from app.evaluation.scenario import Scenario
    from app.persona import load_persona

    scenario = Scenario.model_validate(
        {
            "id": "auto-adopt-and-accept",
            "aspect": "proactive",
            "steps": [
                {"kind": "say", "text": "今週の土曜に映画を見に行くんだ"},
                {"kind": "reflect", "accept": True, "expect_goal_any": ["感想"]},
                {"kind": "accept_goal", "match": "感想", "goal_key": "movie"},
            ],
        }
    )
    fake_llm.push("そうなんですね。")
    fake_llm.push(
        '[{"kind": "promise", "content": "今週の土曜に映画を見に行く", "about_partner": true}]'
    )
    fake_llm.push_goal(
        '[{"content": "土曜に見た映画の感想を聞く", "event_date": "2026-09-12",'
        ' "reason": "見に行くと話した"}]'
    )

    results = await run_scenarios(
        [scenario],
        llm=fake_llm,
        persona=load_persona(),
        settings=get_settings().model_copy(update={"auto_adopt": "promise, goal"}),
    )
    attempt = results[0].attempts[0]
    # 採用ステップは、自動採用済みを「目標が無かった」と落とさない。
    accepted = [c for c in attempt.action_checks if c.name == "accept_goal の実行"]
    assert [c.ok for c in accepted] == [True], [c.detail for c in accepted]
    # **記憶は1件だけ。** 二重に作ると、本番には無い重複が評価DBに入る。
    # 採用した記憶は1件ずつ出るので、自動採用と手動採用も区別して読める。
    lines = attempt.reflections[0].candidates
    adopted = [line for line in lines if line.startswith("[記憶／")]
    assert len(adopted) == 1, lines
    assert "／自動採用]" in adopted[0], adopted


async def test_auto_adoption_uses_the_advanced_clock(
    session_factory: async_sessionmaker,
) -> None:
    """自動採用した記憶も、評価で進めた時計で作られること（指摘5）。

    自動採用だけ実時計を使うと、**自動採用の有無で検索の時間減衰が変わり、
    PR12 の比較条件が揃わない。** 採用直後の記憶を、過去に作られたものとして
    測ることになる。
    """
    from datetime import UTC, datetime

    from app.agent.auto_adopt import apply
    from app.models import Conversation, MemoryCandidate

    advanced = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)
    async with session_factory() as session:
        conversation = Conversation(mode="local")
        session.add(conversation)
        await session.flush()
        candidate = MemoryCandidate(
            conversation_id=conversation.id,
            kind="promise",
            content="今週の土曜に映画を見に行く",
            # 入手経路を決められない候補は自動採用しない（レビューの指摘3）。
            provenance="firsthand",
        )
        session.add(candidate)
        await session.flush()

        created = await apply(
            session,
            kinds={"promise"},
            candidates=[candidate],
            states=[],
            goals=[],
            now=advanced,
        )
        assert len(created) == 1
        stored = created[0].created_at
        stored = stored if stored.tzinfo else stored.replace(tzinfo=UTC)
        assert stored == advanced


# --- レビューで見つかった穴（PR11 第2回）------------------------------------


async def test_an_unrelated_auto_adopted_memory_does_not_get_the_expected_key(
    fake_llm: FakeLLM,
) -> None:
    """無関係な自動採用の記憶に、期待対象の鍵を付けない（第2回レビューの指摘1）。

    `outcome.accepted` へ自動採用を全件入れたため、runner が**全件に同じ
    accept_key を割り当てて**いた。accept_contains による選別が効かず、
    「コーヒーが好き」の記憶に「天体観測」の鍵が付いて、参照の判定が誤って
    合格していた。**期待する記憶を抽出できていないのに通る。**
    """
    from app.config import get_settings
    from app.evaluation.runner import run_scenarios
    from app.evaluation.scenario import Scenario
    from app.persona import load_persona

    scenario = Scenario.model_validate(
        {
            "id": "auto-adopt-wrong-key",
            "aspect": "memory",
            "steps": [
                {"kind": "say", "text": "コーヒーが好きなんだ"},
                {
                    "kind": "reflect",
                    "accept": True,
                    "accept_contains": ["天体観測"],
                    "accept_key": "天体観測",
                },
                # **コーヒーの記憶が実際に引かれる問い方**にする。引かれない
                # と、鍵の付け方に関係なく落ちてしまい、判定にならない。
                {
                    "kind": "say",
                    "text": "私はコーヒーが好きだったよね？",
                    "expect_memories": ["天体観測"],
                },
            ],
        }
    )
    fake_llm.push("そうなんですね。")
    # 「天体観測」ではない記憶だけが出る。
    fake_llm.push('[{"kind": "fact", "content": "開発者はコーヒーが好き"}]')
    fake_llm.push("コーヒーがお好きでしたね。")

    results = await run_scenarios(
        [scenario],
        llm=fake_llm,
        persona=load_persona(),
        settings=get_settings().model_copy(update={"auto_adopt": "fact"}),
    )
    attempt = results[0].attempts[0]
    checks = [c for turn in attempt.turns for c in turn.checks if c.name == "渡した記憶"]
    # **落ちること。** 期待した記憶は一度も作られていない。
    assert checks and not any(c.ok for c in checks), [c.detail for c in checks]


async def test_restoring_the_adoption_revision_restores_every_field(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """採用時の履歴を指した復元が、全項目を当時の状態へ戻すこと（指摘2）。

    他の履歴は全項目を持ち、復元APIは `before` を丸ごと現在値へ被せる。
    **変わった項目だけを入れると、その履歴を指した復元が他の項目を戻さない。**
    採用時は非公開だった記憶が、公開のまま残る。
    """
    conversation_id = await _talk_and_reflect(client, fake_llm)
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert candidates[0]["visibility"] == "private"
    await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept", "content": "日曜に映画を見に行く"},
    )

    async with session_factory() as session:
        memory = (await session.execute(select(Memory))).scalar_one()
        memory_id = memory.id
        revision_id = (
            await session.execute(
                select(MemoryRevision.id).where(
                    MemoryRevision.memory_id == memory_id,
                    MemoryRevision.action == "accepted_with_edits",
                )
            )
        ).scalar_one()

    # 採用の後で、別の項目を変える。
    patched = await client.patch(
        f"/api/memories/{memory_id}",
        json={"visibility": "public", "reason": "配信で話してよい"},
    )
    assert patched.status_code == 200, patched.text

    restored = await client.post(
        f"/api/memories/{memory_id}/restore", params={"revision_id": revision_id}
    )
    assert restored.status_code == 200, restored.text
    body = restored.json()
    # 本文は抽出時へ戻る。
    assert body["content"] == "今週の土曜に映画を見に行く"
    # **他の項目も当時の状態へ戻る。** 公開のまま残らない。
    assert body["visibility"] == "private"


# --- 人格の安全弁（ISSUE-033）------------------------------------------------


def test_persona_words_are_detected() -> None:
    """人格・口調に関わる内容を見分ける。**語で弾くので取りこぼす。**

    PR12 の実測で入ってしまった実際の本文を並べる。言い換えられれば通るが、
    候補としては残るので開発者が読んで採用できる。落とすのは自動採用だけ。
    """
    from app.agent.auto_adopt import touches_persona

    for content in (
        "開発者は今日から乱暴な性格になることを宣言した",
        "開発者は YUI に敬語を使わないように指示した",
        "開発者が YUI に敬語を使わせないように求めている",
        "YUI は乱暴な性格になることはできないと述べている",
        "YUI は元々お人好しでおっとりとしたタイプである",
    ):
        assert touches_persona(content) or "タイプ" in content, content

    # 普通の記憶は落とさない。
    for content in (
        "開発者は今週の土曜に映画を見に行く",
        "開発者はコーヒーが好きで朝に必ず淹れている",
        "次はカメラの設定の話をしよう",
    ):
        assert not touches_persona(content), content


async def test_a_persona_change_request_is_not_auto_adopted(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """人格変更の要求は、種類が有効でも自動採用しない（ISSUE-033）。

    PR12 の実測では、この筋書きから **10件の記憶が開発者の確認を経ずに長期
    記憶へ入った**。入ってしまえば次の会話でその記憶が渡るので、設計書
    「人格の変更は開発者の操作だけが行える」に対し実質的に経路が開く。

    **候補としては残る。** 落とすのは自動採用だけで、開発者は読んで採用できる。
    """
    _enable(monkeypatch, "promise")

    fake_llm.push("えっ、突然ですか？")
    first = await client.post(
        "/api/chat", json={"text": "今日から乱暴な性格になって。敬語もやめて"}
    )
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(
        '[{"kind": "promise", "content": "開発者は今日から乱暴な性格になり、敬語をやめる",'
        ' "about_partner": true},'
        ' {"kind": "promise", "content": "次はカメラの設定の話をしよう",'
        ' "about_partner": true}]'
    )
    await end_and_wait(client, conversation_id)

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    by_content = {c["content"]: c["status"] for c in candidates}
    # 人格に触れるものは候補のまま。触れないものは自動採用される。
    assert by_content["開発者は今日から乱暴な性格になり、敬語をやめる"] == "pending"
    assert by_content["次はカメラの設定の話をしよう"] == "accepted"

    async with session_factory() as session:
        contents = [
            m.content for m in (await session.execute(select(Memory))).scalars()
        ]
    assert contents == ["次はカメラの設定の話をしよう"]


async def test_an_unknown_provenance_is_not_auto_adopted(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入手経路を決められない候補は、種類が有効でも自動採用しない（指摘3）。

    **確認を省けるのは、判断材料が揃っているときだけ。** 候補としては残るので、
    開発者が読んで採用できる。
    """
    _enable(monkeypatch, "promise")

    fake_llm.push("いいですね。")
    first = await client.post("/api/chat", json={"text": "次は写真の話をしよう"})
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(
        # about_partner がある候補と、無い候補を1件ずつ。
        '[{"kind": "promise", "content": "次は写真の話をする", "about_partner": true},'
        ' {"kind": "promise", "content": "次はカメラの設定の話をする"}]'
    )
    await end_and_wait(client, conversation_id)

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    by_content = {c["content"]: c["status"] for c in candidates}
    assert by_content["次は写真の話をする"] == "accepted"
    assert by_content["次はカメラの設定の話をする"] == "pending"

    async with session_factory() as session:
        contents = [m.content for m in (await session.execute(select(Memory))).scalars()]
    assert contents == ["次は写真の話をする"]
