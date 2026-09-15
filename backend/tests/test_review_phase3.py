"""v0.1 レビューの指摘に対する再発検知。

対象は旧 docs/review/codex/phase3_review.md の指摘1〜5
（`b1afd20` で削除済み。原文は git 履歴：
`git log --diff-filter=D -- docs/review/codex/phase3_review.md`）。
どれも「別の経路に同じ問題が残っていた」形なので、経路ごとに固定する。
"""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import CharacterState, MemoryCandidate
from tests.conftest import FakeLLM

A = {"source": "local_text", "external_id": "rev-a", "display_name": "Aさん"}
B = {"source": "local_text", "external_id": "rev-b", "display_name": "Bさん"}


async def _two_speaker_conversation(client: AsyncClient) -> tuple[int, int, int]:
    first = await client.post("/api/chat", json={"text": "私は写真が好きなんだ", "speaker": A})
    conversation_id = first.json()["conversation_id"]
    a_id = first.json()["user_message"]["speaker_id"]
    second = await client.post(
        "/api/chat",
        json={"text": "私は登山が好き", "conversation_id": conversation_id, "speaker": B},
    )
    return conversation_id, a_id, second.json()["user_message"]["speaker_id"]


async def test_transcript_names_each_speaker(client: AsyncClient, fake_llm: FakeLLM) -> None:
    """振り返りの会話ログにも、誰の発言かを書く（指摘1）。

    通常の返答では発言者を付けるようにしたが（ISSUE-020）、振り返りの経路には
    届いておらず、全員の発言が同じ名前で並んでいた。
    """
    conversation_id, _, _ = await _two_speaker_conversation(client)
    fake_llm.push("[]")
    await client.post(f"/api/conversations/{conversation_id}/end")

    sent = next(call[-1].content for call in fake_llm.calls if "会話:" in call[-1].content)
    assert "Aさん: 私は写真が好きなんだ" in sent
    assert "Bさん: 私は登山が好き" in sent


async def test_candidate_belongs_to_the_speaker_of_its_source(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """候補は、根拠の発言をした人のものにする（指摘1）。

    最後に話した人へ寄せると、Aさんの好みがBさんの情報として保存される。
    """
    conversation_id, a_id, b_id = await _two_speaker_conversation(client)
    async with session_factory() as session:
        from app.models import Message  # noqa: PLC0415

        messages = list(
            (
                await session.execute(
                    select(Message).where(Message.conversation_id == conversation_id)
                )
            ).scalars()
        )
    a_message = next(m for m in messages if m.speaker_id == a_id)

    fake_llm.push(
        f'[{{"kind":"about_person","content":"Aさんは写真が好き","certainty":"fact",'
        f'"provenance":"firsthand","keywords":"写真","about_partner":true,'
        f'"source_message_id":{a_message.id}}}]'
    )
    candidates = (await client.post(f"/api/conversations/{conversation_id}/end")).json()

    # 最後に話したのは B さんだが、根拠は A さんの発言。
    assert candidates[0]["subject_speaker_id"] == a_id
    assert candidates[0]["visible_to_speaker_id"] == a_id


async def test_state_scope_cannot_exceed_its_basis(client: AsyncClient) -> None:
    """根拠の記憶より広い参照範囲の状態は作れない（指摘3）。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは", "speaker": A})
    a_id = turn.json()["user_message"]["speaker_id"]
    memory = (
        await client.post(
            "/api/memories",
            json={
                "kind": "about_person",
                "content": "Aさんは写真が好き",
                "keywords": "写真",
                "visible_to_speaker_id": a_id,
            },
        )
    ).json()

    leaked = await client.post(
        "/api/states",
        json={
            "kind": "interest",
            "content": "写真を撮る人の見方に興味がある",
            "basis_memory_ids": [memory["id"]],
            "visible_to_all": True,
        },
    )
    assert leaked.status_code == 400
    assert "限定しています" in leaked.json()["detail"]

    ok = await client.post(
        "/api/states",
        json={
            "kind": "interest",
            "content": "写真を撮る人の見方に興味がある",
            "basis_memory_ids": [memory["id"]],
            "visible_to_speaker_id": a_id,
        },
    )
    assert ok.status_code == 201


async def test_state_scope_is_required(client: AsyncClient) -> None:
    """状態にも参照範囲を必ず指定させる（指摘3、ISSUE-010 と同じ考え方）。"""
    response = await client.post(
        "/api/states", json={"kind": "interest", "content": "範囲の無い関心"}
    )
    assert response.status_code == 422
    assert "参照範囲" in response.text


async def test_failed_reflection_leaves_no_candidates(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """途中で失敗したら、記憶の候補も残さない（指摘4）。

    残ると、やり直したときに同じ候補が二重に保存される。
    """
    first = await client.post("/api/chat", json={"text": "写真を撮るのが好きなんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    fake_llm.push_state("読み取れない出力")
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503

    async with session_factory() as session:
        stored = list((await session.execute(select(MemoryCandidate))).scalars())
        states = list((await session.execute(select(CharacterState))).scalars())
    assert stored == []
    assert states == []

    # やり直せて、二重にならない。
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    fake_llm.push_state("[]")
    retried = await client.post(f"/api/conversations/{conversation_id}/end")
    assert retried.status_code == 200
    assert len(retried.json()) == 1


async def test_other_conversations_can_write_while_reflecting(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """振り返りでモデルを待つ間、別の会話の書き込みを止めない（指摘5）。"""
    import asyncio  # noqa: PLC0415

    first = await client.post("/api/chat", json={"text": "写真の話"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    gate = asyncio.Event()
    fake_llm.gate = gate
    fake_llm.entered.clear()

    reflecting = asyncio.create_task(client.post(f"/api/conversations/{conversation_id}/end"))
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=2)

    # 抽出の途中でも、別の記憶を書ける。
    written = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "振り返り中に書いた記憶",
            "keywords": "確認",
            "visible_to_all": True,
        },
    )
    assert written.status_code == 201

    fake_llm.gate = None
    gate.set()
    assert (await reflecting).status_code == 200


# --- 再レビューの指摘 -------------------------------------------------------


async def test_correction_reaches_states_made_by_reflection(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """振り返りが作った状態にも、記憶の訂正が届く（再レビューの指摘1）。

    候補の時点では記憶がまだ採用されておらず、記憶IDを持てない。会話単位でも
    印が届くようにした。届かないまま古い内容を話すほうが重い。
    """
    first = await client.post("/api/chat", json={"text": "写真を撮るのが好きなんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    fake_llm.push_state(
        '[{"kind":"interest","topic":"写真","content":"人が撮った写真に興味がある"}]'
    )
    candidates = (await client.post(f"/api/conversations/{conversation_id}/end")).json()

    memory_id = (
        await client.post(
            f"/api/conversations/candidates/{candidates[0]['id']}/decide",
            json={"decision": "accept"},
        )
    ).json()["accepted_memory_id"]

    states = (await client.get("/api/states?state_status=pending")).json()
    accepted = (
        await client.post(f"/api/states/{states[0]['id']}/decide", json={"decision": "accept"})
    ).json()
    # 採用の時点で、その会話から採用された記憶を根拠として結び付ける。
    assert accepted["basis_memory_ids"] == [memory_id]

    await client.patch(
        f"/api/memories/{memory_id}",
        json={"content": "開発者は絵を描くのが好き", "reason": "聞き違いだった"},
    )
    marked = (await client.get("/api/states?needs_review=true")).json()
    assert [s["id"] for s in marked] == [accepted["id"]]


async def test_correction_reaches_states_accepted_before_the_memory(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """記憶より先に状態を採用しても、訂正が届く（再レビューの指摘1）。"""
    first = await client.post("/api/chat", json={"text": "写真を撮るのが好きなんだ"})
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    fake_llm.push_state(
        '[{"kind":"interest","topic":"写真","content":"人が撮った写真に興味がある"}]'
    )
    candidates = (await client.post(f"/api/conversations/{conversation_id}/end")).json()

    # 記憶を採用する前に、状態を採用する。根拠は結び付かない。
    states = (await client.get("/api/states?state_status=pending")).json()
    accepted = (
        await client.post(f"/api/states/{states[0]['id']}/decide", json={"decision": "accept"})
    ).json()
    assert accepted["basis_memory_ids"] is None

    memory_id = (
        await client.post(
            f"/api/conversations/candidates/{candidates[0]['id']}/decide",
            json={"decision": "accept"},
        )
    ).json()["accepted_memory_id"]
    await client.delete(f"/api/memories/{memory_id}?reason=誤りのため")

    # 会話が同じなので、根拠を持たない状態にも印が届く。
    marked = (await client.get("/api/states?needs_review=true")).json()
    assert [s["id"] for s in marked] == [accepted["id"]]


def test_message_date_is_read_back_as_utc() -> None:
    """SQLite から返る timezone 無しの日時に、UTC を補ってから直す（指摘2）。

    補わないと、日本時間 9/8 00:30 の発言が 9/7 の発言として渡る。
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from app.config import to_local  # noqa: PLC0415

    naive = datetime(2026, 9, 7, 15, 30)  # UTC で保存された値
    assert to_local(naive).strftime("%Y-%m-%d %H:%M") == "2026-09-08 00:30"
    # すでに timezone を持つ値は、そのまま直す。
    aware = datetime(2026, 9, 7, 15, 30, tzinfo=UTC)
    assert to_local(aware) == to_local(naive)


def test_machine_checked_is_decided_by_the_scenario(tmp_path) -> None:
    """機械判定の有無は、実行できた件数ではなくシナリオの書き方で決める（指摘3）。

    モデルの呼び出しが全部失敗したシナリオを、人手専用として集計から
    落とさないため。
    """
    from app.evaluation.scenario import Scenario  # noqa: PLC0415

    machine = Scenario.model_validate(
        {
            "id": "machine",
            "aspect": "memory",
            "turns": [{"text": "覚えてる？", "expect_any": ["写真"]}],
        }
    )
    human = Scenario.model_validate(
        {
            "id": "human",
            "aspect": "persona",
            "turns": [{"text": "こんにちは", "human_check": "口調はどうか"}],
        }
    )
    assert machine.has_machine_checks is True
    assert human.has_machine_checks is False


async def test_failed_runs_are_not_counted_as_human_only(fake_llm: FakeLLM) -> None:
    """全試行がモデルの失敗でも、機械判定のシナリオは集計に残す（指摘3）。"""
    from app.config import get_settings  # noqa: PLC0415
    from app.evaluation.runner import run_scenarios  # noqa: PLC0415
    from app.evaluation.scenario import Scenario  # noqa: PLC0415
    from app.llm.base import LLMError  # noqa: PLC0415
    from app.persona import load_persona  # noqa: PLC0415

    scenario = Scenario.model_validate(
        {
            "id": "broken",
            "aspect": "memory",
            "turns": [{"text": "覚えてる？", "expect_any": ["写真"]}],
        }
    )

    async def fail(*_args, **_kwargs):
        raise LLMError("接続できません（テスト）")

    fake_llm.chat = fail  # type: ignore[method-assign]
    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    assert results[0].human_only is False
    assert results[0].passed == 0
    assert results[0].failed_to_run == 1


def test_pickup_accepts_the_shapes_the_model_actually_returns() -> None:
    """拾い出しが返してくる形を受け入れる（実行時に観測した失敗）。

    拾い出しは形式の作業で、判断はしない。読み取れる形は受け入れる。
    内容が失われるわけではないため、ここで失敗させる意味は薄い。
    """
    from app.agent.reflection import _parse_pickups  # noqa: PLC0415

    # ["内容", "#1"] の形
    nested = _parse_pickups('[["今日はいい天気", "#1"], ["空を眺めている", "#2"]]')
    assert [(p.content, p.source_message_id) for p in nested] == [
        ("今日はいい天気", 1),
        ("空を眺めている", 2),
    ]

    # 配列を分けて並べてくる形
    split = _parse_pickups('["今日はいい天気だね", "#1"] ["空を眺めている", "#2"]')
    assert len(split) == 2

    # 通常の形も、これまでどおり読める。
    normal = _parse_pickups('[{"content": "山に登った", "source_message_id": 3}]')
    assert normal[0].source_message_id == 3


# --- 再々レビューの指摘 -----------------------------------------------------


async def test_correction_reaches_states_adopted_between_memories(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """記憶を一部だけ採用した時点で状態を採用しても、訂正が届く（再々レビューの指摘1）。

    採用のときに自動で並べた根拠は、後から採用される記憶が入らない。完全に
    特定した根拠として扱うと、そこから漏れた記憶の訂正が届かなくなる。
    """
    first = await client.post("/api/chat", json={"text": "写真と紅茶が好きなんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true},'
        '{"kind":"about_person","content":"開発者は紅茶が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"紅茶","about_partner":true}]'
    )
    fake_llm.push_state(
        '[{"kind":"interest","topic":"飲み物","content":"紅茶の話をもっと聞きたい"}]'
    )
    candidates = (await client.post(f"/api/conversations/{conversation_id}/end")).json()
    photo = next(c for c in candidates if "写真" in c["content"])
    tea = next(c for c in candidates if "紅茶" in c["content"])

    # 写真の記憶だけ採用 → 状態を採用 → 紅茶の記憶を採用、の順。
    await client.post(
        f"/api/conversations/candidates/{photo['id']}/decide", json={"decision": "accept"}
    )
    states = (await client.get("/api/states?state_status=pending")).json()
    accepted = (
        await client.post(f"/api/states/{states[0]['id']}/decide", json={"decision": "accept"})
    ).json()
    # 自動で並べた根拠なので、暫定として記録する。
    assert accepted["basis_is_provisional"] is True

    tea_memory_id = (
        await client.post(
            f"/api/conversations/candidates/{tea['id']}/decide", json={"decision": "accept"}
        )
    ).json()["accepted_memory_id"]
    assert tea_memory_id not in (accepted["basis_memory_ids"] or [])

    # 後から採用した記憶を削除しても、印が届く。
    await client.delete(f"/api/memories/{tea_memory_id}?reason=誤りのため")
    marked = (await client.get("/api/states?needs_review=true")).json()
    assert [s["id"] for s in marked] == [accepted["id"]]


async def test_withdrawn_state_is_checked_when_brought_back(client: AsyncClient) -> None:
    """撤回中に根拠が消えたら、有効へ戻すときに気づく（再々レビューの指摘2）。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = turn.json()["user_message"]["speaker_id"]
    memory = (
        await client.post(
            "/api/memories",
            json={
                "kind": "about_person",
                "content": "開発者は紅茶が好き",
                "keywords": "紅茶",
                "visible_to_speaker_id": speaker_id,
            },
        )
    ).json()
    state = (
        await client.post(
            "/api/states",
            json={
                "kind": "interest",
                "topic": "飲み物",
                "content": "紅茶の話をもっと聞きたい",
                "basis_memory_ids": [memory["id"]],
                "visible_to_speaker_id": speaker_id,
            },
        )
    ).json()
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})

    # 撤回してから、根拠を削除する。
    await client.patch(f"/api/states/{state['id']}", json={"status": "withdrawn"})
    await client.delete(f"/api/memories/{memory['id']}?reason=誤りのため")

    # 撤回中でも印は付く。
    withdrawn = (await client.get("/api/states?needs_review=true")).json()
    assert [s["id"] for s in withdrawn] == [state["id"]]

    # 印を無視して有効へ戻しても、根拠が消えていることに気づく。
    await client.patch(f"/api/states/{state['id']}", json={"reviewed": True})
    back = (await client.patch(f"/api/states/{state['id']}", json={"status": "active"})).json()
    assert back["needs_review"] is True
    assert "有効でなくなっている" in back["review_reason"]


async def test_third_model_call_does_not_hold_the_write_lock(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """3回目の呼び出し（関心・関係性）の間も、別の会話が書ける。

    以前は、記憶候補を flush してから3回目を呼んでいたため、その間 SQLite の
    書き込みロックを保持していた。前のテストは最初の呼び出しで止めており、
    この経路の証拠になっていなかった（再レビューでの指摘）。
    """
    import asyncio  # noqa: PLC0415

    from app.agent.state_reflection import _INSTRUCTION as STATE_INSTRUCTION  # noqa: PLC0415

    first = await client.post("/api/chat", json={"text": "写真の話"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"about_person","content":"開発者は写真が好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真","about_partner":true}]'
    )
    gate = asyncio.Event()
    fake_llm.gate = gate
    # 3回目（関心・関係性の抽出）だけを止める。
    fake_llm.gate_on = STATE_INSTRUCTION
    fake_llm.held.clear()

    reflecting = asyncio.create_task(client.post(f"/api/conversations/{conversation_id}/end"))
    await asyncio.wait_for(fake_llm.held.wait(), timeout=2)
    # 止まっているのが振り返りの3回目であることを確かめる。
    # （1回目は会話の返答、そのあと 拾う → 選ぶ → 関心・関係性 と続く）
    from app.agent.reflection import _INSTRUCTION as SELECT  # noqa: PLC0415
    from app.agent.reflection import _PICKUP_INSTRUCTION as PICKUP  # noqa: PLC0415

    assert [call[0].content for call in fake_llm.calls[-3:]] == [PICKUP, SELECT, STATE_INSTRUCTION]

    written = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "3回目の推論中に書いた記憶",
            "keywords": "確認",
            "visible_to_all": True,
        },
    )
    assert written.status_code == 201

    fake_llm.gate = None
    fake_llm.gate_on = None
    gate.set()
    assert (await reflecting).status_code == 200


# --- 第5回レビューの指摘 -----------------------------------------------------


def test_broken_pickup_output_is_not_partially_accepted() -> None:
    """壊れた部分があれば失敗させる（第5回レビューの指摘2）。

    「読める別形式を許す」ことと「壊れた部分を捨てる」ことは別。捨てると、
    拾い損ねた内容に気づけない。
    """
    import pytest  # noqa: PLC0415

    from app.agent.reflection import ReflectionParseError, _parse_pickups  # noqa: PLC0415

    with pytest.raises(ReflectionParseError) as exc:
        _parse_pickups('[["写真が好き", "#1"], ["紅茶が好き", broken]]')
    assert "読み取れない部分" in str(exc.value)

    # 読める形は、これまでどおり受け入れる。
    assert len(_parse_pickups('["写真が好き", "#1"] ["紅茶が好き", "#2"]')) == 2
    assert len(_parse_pickups('[{"content": "山に登った", "source_message_id": 3}]')) == 1


def test_migration_marks_old_auto_filled_basis_as_provisional(tmp_path) -> None:
    """旧版が自動で並べた根拠を、移行で暫定として扱う（第5回レビューの指摘1）。

    確定扱いのまま残すと、そこから漏れた記憶の訂正が届かない。手で指定した
    根拠は確定のままにする。
    """
    import sqlite3  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from app.config import BACKEND_ROOT  # noqa: PLC0415

    db = tmp_path / "migrate.db"
    url = f"sqlite+aiosqlite:///{db}"
    env = {"YUI_DATABASE_URL": url, "PATH": "/usr/bin:/bin"}

    def alembic(*args: str) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            cwd=BACKEND_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    # 暫定フラグが入る前の版まで上げ、その時点のデータを入れる。
    alembic("upgrade", "3187f6b349d3")
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        INSERT INTO conversations (id, mode, started_at) VALUES (1, 'local', '2026-09-01');
        -- 振り返りが作り、採用のときに自動で根拠が入った状態
        INSERT INTO character_states
            (id, kind, content, basis_memory_ids, needs_review, status, visibility,
             source_conversation_id, created_at, updated_at)
        VALUES (1, 'interest', '自動で根拠が入った関心', '[1]', 0, 'active', 'private',
                1, '2026-09-01', '2026-09-01');
        -- APIから手で作り、根拠を明示した状態
        INSERT INTO character_states
            (id, kind, content, basis_memory_ids, needs_review, status, visibility,
             source_conversation_id, created_at, updated_at)
        VALUES (2, 'interest', '手で根拠を指定した関心', '[2]', 0, 'active', 'private',
                NULL, '2026-09-01', '2026-09-01');
        """
    )
    connection.commit()
    connection.close()

    alembic("upgrade", "head")

    connection = sqlite3.connect(db)
    rows = dict(
        connection.execute("SELECT id, basis_is_provisional FROM character_states").fetchall()
    )
    connection.close()
    assert rows[1] == 1, "振り返り由来の自動根拠は暫定として扱う"
    assert rows[2] == 0, "手で指定した根拠は確定のまま"
