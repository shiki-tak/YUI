"""v0.2 PR4：会話状態の API と `/end` での失効（docs/plan/v0.2.md §5・§6）。

`GET /conversations/{id}/states`・`POST /conversations/{id}/states/{state_id}/decide`
と、会話終了で開いている会話状態がすべて `expired` になること（`resolved` には
しない）、抽出が失敗した場合は会話状態を変えないことを確かめる。
"""

from __future__ import annotations

from httpx import AsyncClient

from tests.conftest import FakeLLM


async def _say(client: AsyncClient, text: str, conversation_id: int | None = None) -> dict:
    payload: dict = {"text": text}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --- GET /conversations/{id}/states -----------------------------------------


async def test_chat_response_includes_open_conversation_states(client: AsyncClient) -> None:
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    kinds = {state["kind"] for state in turn["conversation_states"]}
    assert "question_to_yui" in kinds


async def test_chat_response_excludes_presented(client: AsyncClient) -> None:
    """`presented` は返答ごとに1件でき、解決・取消が無いので open のまま
    残り続ける。画面はこの種類を使わないので、`/chat` の応答に含めて会話が
    長くなるほど肥大させない（計画 §6。レビューで実測）。
    """
    turn = await _say(client, "こんにちは")
    kinds = {state["kind"] for state in turn["conversation_states"]}
    assert "presented" not in kinds

    # 会話状態としては作られている（GET で全件は見える）。
    all_states = await client.get(f"/api/conversations/{turn['conversation_id']}/states")
    assert "presented" in {s["kind"] for s in all_states.json()}


async def test_list_states_can_be_filtered_by_kind_and_status(client: AsyncClient) -> None:
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]
    await _say(client, "映画の話はまた今度にしよう", conversation_id)

    all_states = await client.get(f"/api/conversations/{conversation_id}/states")
    assert all_states.status_code == 200, all_states.text
    all_kinds = {s["kind"] for s in all_states.json()}
    assert {"question_to_yui", "deferral"} <= all_kinds

    only_deferral = await client.get(
        f"/api/conversations/{conversation_id}/states", params={"kind": "deferral"}
    )
    assert {s["kind"] for s in only_deferral.json()} == {"deferral"}

    combined = await client.get(
        f"/api/conversations/{conversation_id}/states",
        params={"kind": "deferral,closing"},
    )
    assert {s["kind"] for s in combined.json()} <= {"deferral", "closing"}

    # status で実際に絞り込みが効くことを、1件だけ決着させてから確かめる
    # （レビュー指摘：全件 open のままだと、フィルタが無くても通ってしまう）。
    question = next(s for s in all_states.json() if s["kind"] == "question_to_yui")
    resolved = await client.post(
        f"/api/conversations/{conversation_id}/states/{question['id']}/decide",
        json={"decision": "resolved"},
    )
    assert resolved.status_code == 200

    only_open = await client.get(
        f"/api/conversations/{conversation_id}/states", params={"status": "open"}
    )
    open_ids = {s["id"] for s in only_open.json()}
    assert question["id"] not in open_ids
    assert all(s["status"] == "open" for s in only_open.json())

    only_resolved = await client.get(
        f"/api/conversations/{conversation_id}/states", params={"status": "resolved"}
    )
    assert [s["id"] for s in only_resolved.json()] == [question["id"]]


async def test_list_states_can_be_filtered_by_target_speaker(client: AsyncClient) -> None:
    """`/chat` の応答は話している相手だけに絞る。`GET` も同じ絞り方を選べる
    （レビュー指摘：経路によって絞り方が違うと表示が食い違う）。
    """
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]
    target_speaker_id = turn["user_message"]["speaker_id"]

    matched = await client.get(
        f"/api/conversations/{conversation_id}/states",
        params={"target_speaker_id": target_speaker_id},
    )
    assert any(s["kind"] == "question_to_yui" for s in matched.json())

    other = await client.get(
        f"/api/conversations/{conversation_id}/states",
        params={"target_speaker_id": target_speaker_id + 999},
    )
    assert other.json() == []


async def test_list_states_rejects_unknown_kind_and_status(client: AsyncClient) -> None:
    turn = await _say(client, "こんにちは")
    conversation_id = turn["conversation_id"]

    bad_kind = await client.get(
        f"/api/conversations/{conversation_id}/states", params={"kind": "not_a_kind"}
    )
    assert bad_kind.status_code == 400

    bad_status = await client.get(
        f"/api/conversations/{conversation_id}/states", params={"status": "not_a_status"}
    )
    assert bad_status.status_code == 400


# --- POST /conversations/{id}/states/{state_id}/decide -----------------------


async def test_decide_resolves_a_state_as_operator(client: AsyncClient) -> None:
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    question = next(s for s in states if s["kind"] == "question_to_yui")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{question['id']}/decide",
        json={"decision": "resolved"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "resolved"
    assert body["decided_by"] == "operator"
    # 作成の経路は変えない。
    assert body["detected_by"] == "rule"


async def test_decide_withdraws_a_state_with_a_reason(client: AsyncClient) -> None:
    turn = await _say(client, "映画の話はまた今度にしよう")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    deferral = next(s for s in states if s["kind"] == "deferral")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{deferral['id']}/decide",
        json={"decision": "withdrawn", "withdraw_reason": "misdetected"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "withdrawn"
    assert body["withdraw_reason"] == "misdetected"
    assert body["decided_by"] == "operator"


async def test_decide_withdrawn_requires_a_reason(client: AsyncClient) -> None:
    turn = await _say(client, "映画の話はまた今度にしよう")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    deferral = next(s for s in states if s["kind"] == "deferral")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{deferral['id']}/decide",
        json={"decision": "withdrawn"},
    )
    assert response.status_code == 422


async def test_decide_rejects_a_reason_that_does_not_apply_to_the_kind(
    client: AsyncClient,
) -> None:
    """種類ごとに使える取消理由は決まっている（計画 3節）。解釈（LLM）の
    経路と同じ検証を operator 操作にも通す（レビューで実測：以前は検証
    無しで `question_to_yui` に `reopened` を受け入れていた）。
    """
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    question = next(s for s in states if s["kind"] == "question_to_yui")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{question['id']}/decide",
        json={"decision": "withdrawn", "withdraw_reason": "reopened"},
    )
    assert response.status_code == 422

    unchanged = await client.get(f"/api/conversations/{conversation_id}/states")
    refreshed = next(s for s in unchanged.json() if s["id"] == question["id"])
    assert refreshed["status"] == "open"


async def test_decide_never_accepts_superseded(client: AsyncClient) -> None:
    """`superseded` は置き換え先の作成と同じトランザクションでだけ付く理由
    で、`decide`（operator 操作）からは常に拒否する。
    """
    turn = await _say(client, "映画の話はまた今度にしよう")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    deferral = next(s for s in states if s["kind"] == "deferral")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{deferral['id']}/decide",
        json={"decision": "withdrawn", "withdraw_reason": "superseded"},
    )
    assert response.status_code == 422

    unchanged = await client.get(f"/api/conversations/{conversation_id}/states")
    refreshed = next(s for s in unchanged.json() if s["id"] == deferral["id"])
    assert refreshed["status"] == "open"


async def test_decide_rejects_resolved_for_a_kind_with_no_resolution(
    client: AsyncClient,
) -> None:
    """`deferral` に「解決」の概念は無い（計画 3節の表：取消でしか終わらない）。
    以前は `decide` がどの種類の `resolved` も無条件に受け付けていた
    （レビュー指摘）。
    """
    turn = await _say(client, "映画の話はまた今度にしよう")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    deferral = next(s for s in states if s["kind"] == "deferral")

    response = await client.post(
        f"/api/conversations/{conversation_id}/states/{deferral['id']}/decide",
        json={"decision": "resolved"},
    )
    assert response.status_code == 422

    unchanged = await client.get(f"/api/conversations/{conversation_id}/states")
    refreshed = next(s for s in unchanged.json() if s["id"] == deferral["id"])
    assert refreshed["status"] == "open"


async def test_decide_rejects_a_state_from_another_conversation(client: AsyncClient) -> None:
    turn_a = await _say(client, "土曜と日曜、どちらが空いてる？")
    turn_b = await _say(client, "こんにちは")
    states_a = (await client.get(f"/api/conversations/{turn_a['conversation_id']}/states")).json()
    question = next(s for s in states_a if s["kind"] == "question_to_yui")

    response = await client.post(
        f"/api/conversations/{turn_b['conversation_id']}/states/{question['id']}/decide",
        json={"decision": "resolved"},
    )
    assert response.status_code == 404


async def test_decide_rejects_a_state_that_is_already_decided(client: AsyncClient) -> None:
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]
    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    question = next(s for s in states if s["kind"] == "question_to_yui")

    first = await client.post(
        f"/api/conversations/{conversation_id}/states/{question['id']}/decide",
        json={"decision": "resolved"},
    )
    assert first.status_code == 200

    second = await client.post(
        f"/api/conversations/{conversation_id}/states/{question['id']}/decide",
        json={"decision": "resolved"},
    )
    assert second.status_code == 409


async def test_decide_missing_state_is_reported(client: AsyncClient) -> None:
    turn = await _say(client, "こんにちは")
    response = await client.post(
        f"/api/conversations/{turn['conversation_id']}/states/999999/decide",
        json={"decision": "resolved"},
    )
    assert response.status_code == 404


# --- /end：open な会話状態を expired にする -----------------------------------


async def test_ending_a_conversation_expires_open_states_not_resolves_them(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]

    fake_llm.push("[]")  # 記憶候補の抽出（select）。状態抽出は既定の "[]"。
    ended = await client.post(f"/api/conversations/{conversation_id}/end")
    assert ended.status_code == 200, ended.text

    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    question = next(s for s in states if s["kind"] == "question_to_yui")
    assert question["status"] == "expired"
    # 会話が終わったことと問題が解決したことは別（計画 §5）。
    assert question["decided_by"] is None


async def test_extraction_failure_leaves_conversation_states_open(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """抽出が失敗したら会話状態は閉じない（既存どおり 503 で再試行できる）。"""
    turn = await _say(client, "土曜と日曜、どちらが空いてる？")
    conversation_id = turn["conversation_id"]

    fake_llm.push("これはJSONではありません")  # 記憶候補の抽出（select）が失敗する
    ended = await client.post(f"/api/conversations/{conversation_id}/end")
    assert ended.status_code == 503

    states = (await client.get(f"/api/conversations/{conversation_id}/states")).json()
    question = next(s for s in states if s["kind"] == "question_to_yui")
    assert question["status"] == "open"
