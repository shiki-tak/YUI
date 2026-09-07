"""会話履歴を画面から読み直せることを確認する（ISSUE-003）。

保存されているだけでは足りず、次を満たす必要がある。

- 過去の会話を一覧から選び、発言を順番どおりに読み直せる。
- 終了した会話は、読み取り専用として開けると分かる。
- 過去の返答が参照した記憶を、実行記録のIDから引き直せる。
"""

from __future__ import annotations

import json
from datetime import datetime

from httpx import AsyncClient

from tests.conftest import FakeLLM


async def _say(client: AsyncClient, text: str, conversation_id: int | None = None) -> dict:
    payload: dict = {"text": text}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


async def test_conversations_are_listed_newest_first(client: AsyncClient):
    first = await _say(client, "こんばんは")
    second = await _say(client, "べつの話をしよう")

    listed = await client.get("/api/conversations")
    assert listed.status_code == 200, listed.text
    ids = [c["id"] for c in listed.json()]

    assert ids[:2] == [second["conversation_id"], first["conversation_id"]]
    # 一覧から会話を選べるように、見出しが付いている。
    titles = {c["id"]: c["title"] for c in listed.json()}
    assert titles[first["conversation_id"]] == "こんばんは"


async def test_past_conversation_can_be_read_back(client: AsyncClient, fake_llm: FakeLLM):
    fake_llm.push("こんばんは。今日はどうでしたか？")
    first = await _say(client, "ただいま")
    fake_llm.push("それは大変でしたね。")
    await _say(client, "今日は疲れたよ", first["conversation_id"])

    detail = await client.get(f"/api/conversations/{first['conversation_id']}")
    assert detail.status_code == 200, detail.text
    body = detail.json()

    # 発言は交互に、話した順で並ぶ。
    assert [(m["speaker_kind"], m["content"]) for m in body["messages"]] == [
        ("user", "ただいま"),
        ("character", "こんばんは。今日はどうでしたか？"),
        ("user", "今日は疲れたよ"),
        ("character", "それは大変でしたね。"),
    ]


async def test_ended_conversation_is_marked_as_read_only(
    client: AsyncClient, fake_llm: FakeLLM
):
    """終了した会話は、開く前に読み取り専用だと分かる。"""
    first = await _say(client, "そろそろ終わろうか")

    before = (await client.get(f"/api/conversations/{first['conversation_id']}")).json()
    assert before["ended_at"] is None
    assert before["reflection_completed_at"] is None

    fake_llm.push("[]")
    await client.post(f"/api/conversations/{first['conversation_id']}/end")

    after = (await client.get(f"/api/conversations/{first['conversation_id']}")).json()
    assert after["ended_at"] is not None
    assert after["reflection_completed_at"] is not None
    # 終了しても発言は残り、読み直せる。
    assert [m["content"] for m in after["messages"]][0] == "そろそろ終わろうか"


async def test_missing_conversation_is_reported(client: AsyncClient):
    assert (await client.get("/api/conversations/999")).status_code == 404


async def test_referenced_memory_can_be_read_back_by_id(client: AsyncClient):
    """過去の返答の根拠を、実行記録の記憶IDから辿れる。

    その場で返す used_memories と違い、過去の会話には点数・理由が無い。
    残っているのは記憶IDだけなので、IDから内容を引けないと根拠を確認できない。
    """
    first = await _say(client, "山登りが趣味なんだ")
    created = await client.post(
        "/api/memories",
        json={
            "visible_to_all": True,
            "kind": "about_person",
            "content": "開発者の趣味は山登り",
            "keywords": "山登り 趣味",
            "subject_speaker_id": first["user_message"]["speaker_id"],
            "source_message_id": first["user_message"]["id"],
        },
    )
    memory_id = created.json()["id"]

    reply = await _say(client, "山登りはどう？", first["conversation_id"])
    run = (
        await client.get(f"/api/conversations/messages/{reply['reply']['id']}/run")
    ).json()
    assert memory_id in run["referenced_memory_ids"]

    fetched = await client.get(f"/api/memories/{memory_id}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["content"] == "開発者の趣味は山登り"

    # 後から削除した記憶も引ける。当時それを渡していた事実は変わらないため。
    await client.delete(f"/api/memories/{memory_id}?reason=誤りのため")
    deleted = await client.get(f"/api/memories/{memory_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["status"] == "deleted"


async def test_memory_id_route_does_not_shadow_search(client: AsyncClient):
    """/memories/search が ID として解釈されないことを確認する。"""
    response = await client.get("/api/memories/search?q=山")
    assert response.status_code == 200, response.text
    assert response.json()["query"] == "山"


async def test_missing_memory_is_reported(client: AsyncClient):
    assert (await client.get("/api/memories/999")).status_code == 404


async def test_api_datetimes_carry_a_timezone(client: AsyncClient, fake_llm: FakeLLM):
    """日時に timezone を付けて返す。

    保存は UTC だが SQLite は timezone を落とす。そのまま返すと、画面が
    自分の地域の時刻として解釈し、表示が実際の時刻からずれる。
    """
    fake_llm.push("おかえりなさい。")
    first = await _say(client, "ただいま")

    detail = (await client.get(f"/api/conversations/{first['conversation_id']}")).json()
    assert datetime.fromisoformat(detail["started_at"]).tzinfo is not None
    assert datetime.fromisoformat(detail["messages"][0]["created_at"]).tzinfo is not None

    listed = (await client.get("/api/conversations")).json()
    assert datetime.fromisoformat(listed[0]["started_at"]).tzinfo is not None

    fake_llm.push(
        json.dumps(
            [
                {
                    "kind": "experience",
                    "content": "帰宅の挨拶をした",
                    "certainty": "fact",
                    "keywords": "帰宅",
                    "about_partner": True,
                }
            ],
            ensure_ascii=False,
        )
    )
    candidates = (
        await client.post(f"/api/conversations/{first['conversation_id']}/end")
    ).json()
    assert datetime.fromisoformat(candidates[0]["created_at"]).tzinfo is not None

    memory = (
        await client.post(
            f"/api/conversations/candidates/{candidates[0]['id']}/decide",
            json={"decision": "accept"},
        )
    ).json()
    stored = (await client.get(f"/api/memories/{memory['accepted_memory_id']}")).json()
    assert datetime.fromisoformat(stored["created_at"]).tzinfo is not None
    assert datetime.fromisoformat(stored["updated_at"]).tzinfo is not None
