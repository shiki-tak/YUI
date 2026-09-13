"""v0.2 PR3：解釈（LLM）の呼び出し（docs/plan/v0.2.md §4 手順3・4）。

規則（PR1・PR2）は「作った」「応答があった」までしか決めない。ここでは
解釈が返した JSON を検証してから適用する経路——答えたか・確認できたか・
明示的な訂正か——と、失敗時に規則の結果だけで進むことを確かめる。

既定では解釈は無効（`Settings.conversation_state_llm = False`）なので、
`interpretation_client` フィクスチャ（`conftest.py`）で有効にした構成を使う。
"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient
from sqlalchemy import select

from app.agent.conversation_state import (
    create_state,
    discrepancy_offer_ledger,
    find_judgment_candidate,
    get_open_states,
    select_discrepancy_offers,
)
from app.agent.memory_store import get_or_create_speaker
from app.models import (
    Conversation,
    ConversationState,
    ConversationStateKind,
    ConversationStateRefKind,
    ConversationStateStatus,
    ConversationStateWithdrawReason,
    DetectionSource,
    Message,
    SourceKind,
)
from tests.conftest import FakeLLM


async def _states(session_factory, conversation_id: int) -> list[ConversationState]:
    async with session_factory() as session:
        return await get_open_states(session, conversation_id=conversation_id)


# --- answered / unanswered と再判定 ------------------------------------------


async def test_answered_resolves_the_question_and_drops_followup(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """判定対象の質問が実際に答えられていれば、解決して followup_needed が
    下りる（計画 §3・§4 手順4）。
    """
    fake_llm.push_interpretation("{}")  # 質問はまだ無いので、この時点は何もしない
    fake_llm.push("土曜も日曜も空いてるよ。")
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "土曜と日曜、どちらが空いてる？"}
    )
    conversation_id = turn1.json()["conversation_id"]
    reply1_id = turn1.json()["reply"]["id"]

    states = await _states(session_factory, conversation_id)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)
    assert question.responded_message_id == reply1_id
    assert question.status == ConversationStateStatus.OPEN.value

    # 2ターン目：解釈が「実際に答えている」と判定する。
    fake_llm.push_interpretation(f'{{"answered_state_ids": [{question.id}]}}')
    fake_llm.push("了解、じゃあ土曜にしよう。")
    await interpretation_client.post(
        "/api/chat", json={"text": "じゃあ決めよう", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        assert refreshed.status == ConversationStateStatus.RESOLVED.value
        assert refreshed.followup_needed is False
        assert refreshed.judged_message_id == reply1_id


async def test_unanswered_sets_followup_and_a_later_reply_resolves_it(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """答え損ねた質問は `followup_needed` になり、後の返答が答えていれば
    `resolved` になって `followup_needed` が下りる（計画 §3。設計シナリオ11）。
    """
    fake_llm.push_interpretation("{}")
    fake_llm.push("そういえば今日は天気が良かったね。")  # 質問には答えない
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "土曜と日曜、どちらが空いてる？"}
    )
    conversation_id = turn1.json()["conversation_id"]
    reply1_id = turn1.json()["reply"]["id"]
    states = await _states(session_factory, conversation_id)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)

    # 2ターン目：解釈が「答えていない」と判定する。
    fake_llm.push_interpretation(f'{{"unanswered_state_ids": [{question.id}]}}')
    fake_llm.push("うん、そうだね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "本当だね", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        assert refreshed.status == ConversationStateStatus.OPEN.value
        assert refreshed.followup_needed is True
        assert refreshed.judged_message_id == reply1_id

    # 3ターン目：新しい応答候補（2ターン目の返答）が答えていれば resolved。
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        candidate = await find_judgment_candidate(
            session, refreshed, conversation_id=conversation_id
        )
        assert candidate is not None
        assert candidate.id != reply1_id

    fake_llm.push_interpretation(f'{{"answered_state_ids": [{question.id}]}}')
    fake_llm.push("そうそう、土曜にしよう。")
    await interpretation_client.post(
        "/api/chat", json={"text": "決まった？", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        assert refreshed.status == ConversationStateStatus.RESOLVED.value
        assert refreshed.followup_needed is False


# --- 食い違いの確認・解消・訂正 ----------------------------------------------


async def test_asked_and_resolved_discrepancy_creates_correction_in_one_turn(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """確認待ちの食い違いを、直前の返答が実際に確かめていたと解釈したとき
    `asked`・`responded` が入り、今の発言で解消したなら `correction` が
    同じトランザクションで作られる（計画 §3・§4 手順4）。
    """
    turn1 = await interpretation_client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn1.json()["conversation_id"]
    speaker_id = turn1.json()["user_message"]["speaker_id"]
    ref_message_id = turn1.json()["user_message"]["id"]

    async with session_factory() as session:
        discrepancy = await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="土曜と聞いていたが、前回は日曜と言っていた",
            source_message_id=ref_message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=ref_message_id,
            detected_by=DetectionSource.LLM.value,
        )
        await session.commit()
        discrepancy_id = discrepancy.id

    # このターンの返答が食い違いを渡した、として offered_discrepancy_ids に
    # 記録させる（本来は select_discrepancy_offers が選ぶが、直接作ったので
    # 台帳を1件分そのまま作る）。
    async with session_factory() as session:
        ledger = await discrepancy_offer_ledger(
            session, conversation_id=conversation_id, target_speaker_id=speaker_id
        )
        assert discrepancy_id in ledger
        assert ledger[discrepancy_id].offer_count == 0
        offers = select_discrepancy_offers(ledger, limit=2)
        assert [state.id for state in offers] == [discrepancy_id]

    fake_llm.push_interpretation("{}")
    fake_llm.push("念のため確認ですが、今度の予定は日曜で合っていますか？")
    turn2 = await interpretation_client.post(
        "/api/chat", json={"text": "それより最近どう？", "conversation_id": conversation_id}
    )
    offering_reply_id = turn2.json()["reply"]["id"]

    async with session_factory() as session:
        ledger = await discrepancy_offer_ledger(
            session, conversation_id=conversation_id, target_speaker_id=speaker_id
        )
        assert ledger[discrepancy_id].offer_count == 1
        assert ledger[discrepancy_id].pending_reply is not None
        assert ledger[discrepancy_id].pending_reply.id == offering_reply_id
        # 照合待ちの間は、次の確認候補として選ばれない。
        assert select_discrepancy_offers(ledger, limit=2) == []

    fake_llm.push_interpretation(
        f'{{"asked_discrepancy_ids": [{discrepancy_id}], '
        f'"resolved_discrepancy_ids": [{discrepancy_id}], '
        f'"correction": {{"ref_kind": "message", "ref_id": {ref_message_id}, '
        f'"content": "日曜"}}}}'
    )
    fake_llm.push("了解、日曜ですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "あ、日曜だった", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        refreshed = await session.get(ConversationState, discrepancy_id)
        assert refreshed.asked_message_id == offering_reply_id
        assert refreshed.status == ConversationStateStatus.RESOLVED.value
        stmt = select(ConversationState).where(
            ConversationState.conversation_id == conversation_id,
            ConversationState.kind == ConversationStateKind.CORRECTION.value,
        )
        correction = (await session.execute(stmt)).scalars().first()
        assert correction is not None
        assert correction.content == "日曜"
        assert correction.status == ConversationStateStatus.OPEN.value


async def test_discrepancy_offer_stops_after_reaching_the_limit(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """渡した回数が上限に達しても `asked` が入らなければ、渡すのをやめて
    確認できなかったとして残す（計画 §3・§8）。上限は
    `Settings.conversation_state_confirm_offer_limit`（既定2）。
    """
    turn1 = await interpretation_client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn1.json()["conversation_id"]
    speaker_id = turn1.json()["user_message"]["speaker_id"]
    ref_message_id = turn1.json()["user_message"]["id"]

    async with session_factory() as session:
        discrepancy = await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="土曜と聞いていたが、前回は日曜と言っていた",
            source_message_id=ref_message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=ref_message_id,
            detected_by=DetectionSource.LLM.value,
        )
        await session.commit()
        discrepancy_id = discrepancy.id

    for _ in range(2):
        fake_llm.push_interpretation("{}")
        fake_llm.push("それは大丈夫そうです。")
        await interpretation_client.post(
            "/api/chat", json={"text": "それより最近どう？", "conversation_id": conversation_id}
        )

    async with session_factory() as session:
        ledger = await discrepancy_offer_ledger(
            session, conversation_id=conversation_id, target_speaker_id=speaker_id
        )
        assert ledger[discrepancy_id].offer_count == 2
        # 上限（既定2）に達したので、`interpretation_ran=True`（照合待ちを
        # 無視してよい状況）でも渡さない——止めているのは上限そのものだと
        # 確かめる（レビュー指摘：`interpretation_ran` 省略＝False のままでは
        # 照合待ちで弾かれただけでも同じ結果になり、上限の検証にならない）。
        assert select_discrepancy_offers(ledger, limit=2, interpretation_ran=True) == []
        # 上限を上げれば、同じ状態でも候補として選ばれる。
        assert [
            state.id
            for state in select_discrepancy_offers(ledger, limit=99, interpretation_ran=True)
        ] == [discrepancy_id]
        refreshed = await session.get(ConversationState, discrepancy_id)
        assert refreshed.status == ConversationStateStatus.OPEN.value
        assert refreshed.asked_message_id is None


# --- withdrawals（reopened / cancelled）--------------------------------------


async def test_reopened_withdraws_a_deferral_and_cancelled_withdraws_a_question(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    # deferral は最後の文だけを見るため（PR1・PR2 レビュー参照）、
    # 延期と質問は別のターンに分けて作る。
    fake_llm.push_interpretation("{}")
    fake_llm.push("承知しました。")
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "映画の話はまた今度にしよう"}
    )
    conversation_id = turn1.json()["conversation_id"]

    fake_llm.push_interpretation("{}")
    fake_llm.push("いいですよ。")
    await interpretation_client.post(
        "/api/chat", json={"text": "明日は何時に集合？", "conversation_id": conversation_id}
    )

    states = await _states(session_factory, conversation_id)
    deferral = next(s for s in states if s.kind == ConversationStateKind.DEFERRAL.value)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)

    fake_llm.push_interpretation(
        f'{{"withdrawals": ['
        f'{{"state_id": {deferral.id}, "reason": "reopened"}}, '
        f'{{"state_id": {question.id}, "reason": "cancelled"}}]}}'
    )
    fake_llm.push("映画の話、いいですよ。集合時間はもう気にしなくて平気です。")
    await interpretation_client.post(
        "/api/chat",
        json={
            "text": "やっぱり今映画の話をしたい。あと集合時間はもう聞かなくていいや",
            "conversation_id": conversation_id,
        },
    )

    async with session_factory() as session:
        refreshed_deferral = await session.get(ConversationState, deferral.id)
        assert refreshed_deferral.status == ConversationStateStatus.WITHDRAWN.value
        assert refreshed_deferral.withdraw_reason == ConversationStateWithdrawReason.REOPENED.value

        refreshed_question = await session.get(ConversationState, question.id)
        assert refreshed_question.status == ConversationStateStatus.WITHDRAWN.value
        assert (
            refreshed_question.withdraw_reason == ConversationStateWithdrawReason.CANCELLED.value
        )


async def test_withdrawal_reason_must_apply_to_the_states_kind(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """`cancelled` は質問にだけ使える。`deferral` に対して返っても捨てる
    （計画 §4 検証：「操作できない kind・status・target」の項目だけ捨てる）。
    """
    fake_llm.push_interpretation("{}")
    fake_llm.push("承知しました。")
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "映画の話はまた今度にしよう"}
    )
    conversation_id = turn1.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    deferral = next(s for s in states if s.kind == ConversationStateKind.DEFERRAL.value)

    fake_llm.push_interpretation(
        f'{{"withdrawals": [{{"state_id": {deferral.id}, "reason": "cancelled"}}]}}'
    )
    fake_llm.push("そうですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "また別の話だけど", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        refreshed = await session.get(ConversationState, deferral.id)
        assert refreshed.status == ConversationStateStatus.OPEN.value


# --- is_closing / confirms_presented_id --------------------------------------


async def test_is_closing_creates_a_closing_state_when_rules_missed_it(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """規則が拾えない終了の言い方でも、解釈が `is_closing: true` を返せば
    `closing` を作る（計画 3節：「解釈が補う」）。
    """
    fake_llm.push_interpretation('{"is_closing": true}')
    fake_llm.push("はい、今日はここまでにしましょう。")
    turn = await interpretation_client.post(
        "/api/chat", json={"text": "今日はもう満足したから終わろうか"}
    )
    conversation_id = turn.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    assert any(s.kind == ConversationStateKind.CLOSING.value for s in states)


async def test_confirms_presented_id_is_not_created_from_an_acknowledgement_only_reply(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """相づちだけの発言からは `confirmed` を作らない（計画 3節）。"""
    fake_llm.push_interpretation("{}")
    fake_llm.push("最近、近所に新しいパン屋ができたんですよ。")
    turn1 = await interpretation_client.post("/api/chat", json={"text": "最近どう？"})
    conversation_id = turn1.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    presented = next(s for s in states if s.kind == ConversationStateKind.PRESENTED.value)

    fake_llm.push_interpretation(f'{{"confirms_presented_id": {presented.id}}}')
    fake_llm.push("そうなんです。")
    await interpretation_client.post(
        "/api/chat", json={"text": "うん", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        confirmed = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.CONFIRMED.value
        )
        assert confirmed == []


async def test_confirms_presented_id_creates_confirmed_for_a_real_reply(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    fake_llm.push_interpretation("{}")
    fake_llm.push("最近、近所に新しいパン屋ができたんですよ。")
    turn1 = await interpretation_client.post("/api/chat", json={"text": "最近どう？"})
    conversation_id = turn1.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    presented = next(s for s in states if s.kind == ConversationStateKind.PRESENTED.value)

    fake_llm.push_interpretation(f'{{"confirms_presented_id": {presented.id}}}')
    fake_llm.push("それは楽しみですね。")
    await interpretation_client.post(
        "/api/chat",
        json={
            "text": "パン屋さんができたんだ、今度行ってみようかな",
            "conversation_id": conversation_id,
        },
    )
    async with session_factory() as session:
        confirmed = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.CONFIRMED.value
        )
        assert len(confirmed) == 1
        assert confirmed[0].ref_id == presented.id


# --- 検証：範囲外の id・両立しない操作・correction 優先 -----------------------


async def test_out_of_range_ids_are_dropped_without_crashing(
    interpretation_client: AsyncClient, fake_llm: FakeLLM
) -> None:
    fake_llm.push_interpretation(
        '{"answered_state_ids": [999999], "confirms_presented_id": 999999}'
    )
    fake_llm.push("こんにちは。")
    response = await interpretation_client.post("/api/chat", json={"text": "こんにちは"})
    assert response.status_code == 200
    reply_id = response.json()["reply"]["id"]
    run = await interpretation_client.get(f"/api/conversations/messages/{reply_id}/run")
    # 落ちるだけで無視されたのではなく、実際に候補外として捨てたことを見る
    # （レビュー指摘：200 が返るだけでは、検証を素通りしていないか分からない）。
    dropped = run.json()["options"]["interpretation"]["summary"]["dropped"]
    assert len(dropped) == 2


async def test_conflicting_answered_and_unanswered_drops_both(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    fake_llm.push_interpretation("{}")
    fake_llm.push("それは分からないですね。")
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "土曜と日曜、どちらが空いてる？"}
    )
    conversation_id = turn1.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)

    fake_llm.push_interpretation(
        f'{{"answered_state_ids": [{question.id}], "unanswered_state_ids": [{question.id}]}}'
    )
    fake_llm.push("うーん。")
    await interpretation_client.post(
        "/api/chat", json={"text": "どうかな", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        assert refreshed.status == ConversationStateStatus.OPEN.value
        assert refreshed.followup_needed is False


async def test_conflicting_answered_and_cancelled_drops_both(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """解決（answered）と取消（withdrawals）に同じ id が返ったら、その id
    への操作を両方とも捨てる（計画 §4 手順3・§8）。片方が先に適用されて
    後発に上書きされる「早い者勝ち」にはしない（レビューで実測）。
    """
    fake_llm.push_interpretation("{}")
    fake_llm.push("それは分からないですね。")
    turn1 = await interpretation_client.post(
        "/api/chat", json={"text": "土曜と日曜、どちらが空いてる？"}
    )
    conversation_id = turn1.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)

    fake_llm.push_interpretation(
        f'{{"answered_state_ids": [{question.id}], '
        f'"withdrawals": [{{"state_id": {question.id}, "reason": "cancelled"}}]}}'
    )
    fake_llm.push("うーん。")
    await interpretation_client.post(
        "/api/chat", json={"text": "どうかな", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, question.id)
        assert refreshed.status == ConversationStateStatus.OPEN.value


async def test_withdrawals_cannot_supersede_a_correction(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """`superseded` は `create_or_supersede_correction`／`_request` が置き
    換え先の作成と同じトランザクションで自分で行うもので、`withdrawals`
    からは受け付けない。置き換え先が無いのに有効な訂正が消えてはいけない
    （計画 §4 手順3。レビューで実測）。
    """
    turn1 = await interpretation_client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn1.json()["conversation_id"]
    ref_message_id = turn1.json()["user_message"]["id"]

    fake_llm.push_interpretation(
        f'{{"correction": {{"ref_kind": "message", "ref_id": {ref_message_id}, '
        f'"content": "日曜"}}}}'
    )
    fake_llm.push("承知しました、日曜ですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "あ、日曜でした", "conversation_id": conversation_id}
    )
    states = await _states(session_factory, conversation_id)
    correction = next(s for s in states if s.kind == ConversationStateKind.CORRECTION.value)

    # 置き換え先（新しい correction）が無いまま superseded を返す。
    fake_llm.push_interpretation(
        f'{{"withdrawals": [{{"state_id": {correction.id}, "reason": "superseded"}}]}}'
    )
    fake_llm.push("そうですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "映画、楽しみだね", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, correction.id)
        assert refreshed.status == ConversationStateStatus.OPEN.value


async def test_correction_is_preferred_over_discrepancy_for_the_same_ref(
    interpretation_client: AsyncClient, fake_llm: FakeLLM, session_factory
) -> None:
    turn1 = await interpretation_client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn1.json()["conversation_id"]
    ref_message_id = turn1.json()["user_message"]["id"]

    fake_llm.push_interpretation(
        f'{{"correction": {{"ref_kind": "message", "ref_id": {ref_message_id}, '
        f'"content": "日曜"}}, '
        f'"discrepancy": {{"ref_kind": "message", "ref_id": {ref_message_id}, '
        f'"note": "土曜と日曜で食い違っている"}}}}'
    )
    fake_llm.push("承知しました、日曜ですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "あ、日曜でした", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        correction = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.CORRECTION.value
        )
        discrepancy = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.DISCREPANCY.value
        )
        assert len(correction) == 1
        assert discrepancy == []


# --- 失敗時：規則の結果だけで進む --------------------------------------------


async def test_a_pending_interpretation_call_does_not_block_another_conversations_write(
    session_factory, fake_llm: FakeLLM
) -> None:
    """解釈の呼び出しを待っている間、書き込みトランザクションを開いた
    ままにしない。別の会話の `/chat` が `database is locked` にならない
    ことを確かめる（レビューで、解釈中に別会話の書き込みが失敗することを
    実測。修正：解釈の直前に一度 commit し、モデルを呼んでいる間は
    書き込みトランザクションを開かない）。
    """
    from httpx import ASGITransport
    from httpx import AsyncClient as HttpxAsyncClient

    from app.agent import ConversationAgent, get_agent
    from app.agent.interpretation import _INSTRUCTION as INTERPRETATION_INSTRUCTION
    from app.config import Settings
    from app.db import get_session
    from app.main import app
    from app.persona import load_persona

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # timeout は長くとる。ここで確かめたいのは timeout の打ち切りではなく、
    # 解釈を待っている「間」に別会話の書き込みが通ることそのもの。
    settings = Settings(conversation_state_llm=True, conversation_state_llm_timeout_seconds=30.0)
    agent = ConversationAgent(llm=fake_llm, persona=load_persona(), settings=settings)
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_agent] = lambda: agent

    fake_llm.gate = asyncio.Event()
    fake_llm.gate_on = INTERPRETATION_INSTRUCTION
    fake_llm.push("こんにちは、Aさん。")  # A の生成（門を抜けた後）
    fake_llm.push("こんにちは、Bさん。")  # B の生成

    held_gate = fake_llm.gate
    try:
        transport = ASGITransport(app=app)
        async with HttpxAsyncClient(transport=transport, base_url="http://test") as client:
            task_a = asyncio.create_task(client.post("/api/chat", json={"text": "Aです"}))
            # A が解釈の呼び出しで実際に止まるまで待つ（門に到達したことの確認）。
            await asyncio.wait_for(fake_llm.held.wait(), timeout=5)

            # 門はここで外す——B の解釈まで同じ門で止めたいのではなく、
            # A だけを「解釈から戻らない」状態のまま固定したい。既に
            # `await held_gate.wait()` に入っている A の呼び出しは、
            # 個別の Event オブジェクトを直接待っているため、`fake_llm.gate`
            # を外してもここでは影響を受けない（B の新しい呼び出しだけが
            # 門を素通りするようになる）。
            fake_llm.gate = None

            # A がまだ解釈を待っている間に、別会話 B が書き込みを行う。
            # 直前の実装では、A の respond() が commit していない書き込み
            # トランザクションを開いたままだったため、これが
            # `database is locked` で失敗した。
            response_b = await asyncio.wait_for(
                client.post("/api/chat", json={"text": "Bです"}), timeout=5
            )
            assert response_b.status_code == 200

            held_gate.set()
            response_a = await asyncio.wait_for(task_a, timeout=5)
            assert response_a.status_code == 200
    finally:
        held_gate.set()
        app.dependency_overrides.clear()


async def test_interpretation_timeout_falls_back_to_rules_only(
    session_factory, fake_llm: FakeLLM
) -> None:
    """解釈が timeout しても、応答は返り、解決・取消は起きない（計画 §8）。

    `asyncio.wait_for` の timeout（設定で短く切る）が、解釈の呼び出し自体を
    確実に打ち切ることを、FakeLLM の門（`gate_on`）で解釈だけを止めて確かめる。
    """
    from httpx import ASGITransport
    from httpx import AsyncClient as HttpxAsyncClient

    from app.agent import ConversationAgent, get_agent
    from app.agent.interpretation import _INSTRUCTION as INTERPRETATION_INSTRUCTION
    from app.config import Settings
    from app.db import get_session
    from app.main import app
    from app.persona import load_persona

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    settings = Settings(conversation_state_llm=True, conversation_state_llm_timeout_seconds=0.05)
    agent = ConversationAgent(llm=fake_llm, persona=load_persona(), settings=settings)
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_agent] = lambda: agent

    # 解釈の呼び出しだけを門で止め、絶対に返ってこない状態を作る。
    # timeout（0.05秒）が実際にこの呼び出しを打ち切ることを確かめたい。
    fake_llm.gate = asyncio.Event()
    fake_llm.gate_on = INTERPRETATION_INSTRUCTION
    fake_llm.push("こんにちは。")

    try:
        transport = ASGITransport(app=app)
        async with HttpxAsyncClient(transport=transport, base_url="http://test") as client:
            response = await asyncio.wait_for(
                client.post("/api/chat", json={"text": "こんにちは"}), timeout=5
            )
            assert response.status_code == 200
            conversation_id = response.json()["conversation_id"]
            run = await client.get(
                f"/api/conversations/messages/{response.json()['reply']['id']}/run"
            )
    finally:
        fake_llm.gate.set()
        app.dependency_overrides.clear()

    interpretation_status = run.json()["options"]["interpretation"]
    assert interpretation_status["applied"] is False
    # str(TimeoutError()) は空文字になるので、理由が空欄でないことまで見る
    # （レビュー指摘：is not None だけでは "" でも通ってしまう）。
    assert "timeout" in interpretation_status["error"]
    # 「こんにちは」は規則も何も検出しない発言。返答由来の presented だけが
    # できて、解釈でしか作れない種類（request/correction/discrepancy/
    # confirmed）は1件もできていないことを確かめる（計画 §8）。
    states = await _states(session_factory, conversation_id)
    interpretation_only_kinds = {
        ConversationStateKind.REQUEST.value,
        ConversationStateKind.CORRECTION.value,
        ConversationStateKind.DISCREPANCY.value,
        ConversationStateKind.CONFIRMED.value,
    }
    assert [s for s in states if s.kind in interpretation_only_kinds] == []


async def test_interpretation_failure_does_not_change_states(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """解釈が壊れたJSONを返しても、規則の結果はそのまま残り、返答は返る。"""
    fake_llm.push_interpretation("これはJSONではありません")
    fake_llm.push("それは分からないですね。")
    response = await interpretation_client.post(
        "/api/chat", json={"text": "土曜と日曜、どちらが空いてる？"}
    )
    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    states = await _states(session_factory, conversation_id)
    question = next(s for s in states if s.kind == ConversationStateKind.QUESTION_TO_YUI.value)
    assert question.status == ConversationStateStatus.OPEN.value
    assert question.followup_needed is False


async def test_judgment_candidate_is_none_before_any_response(
    session_factory,
) -> None:
    """まだ一度も応答が無い質問には、判定対象の候補が無い。"""
    async with session_factory() as session:
        speaker = await get_or_create_speaker(
            session, source=SourceKind.LOCAL_TEXT.value, external_id="x", display_name="X"
        )
        conversation = Conversation(mode="private")
        session.add(conversation)
        await session.flush()
        message = Message(
            conversation_id=conversation.id,
            speaker_kind="user",
            speaker_id=speaker.id,
            content="質問？",
        )
        session.add(message)
        await session.flush()
        state = await create_state(
            session,
            conversation_id=conversation.id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
            content="質問？",
            source_message_id=message.id,
            speaker_id=speaker.id,
            target_speaker_id=speaker.id,
        )
        await session.commit()
        candidate = await find_judgment_candidate(
            session, state, conversation_id=conversation.id
        )
        assert candidate is None


async def test_a_failed_confirmation_attempt_keeps_the_discrepancy_pending_until_it_succeeds(
    interpretation_client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """設計 §7 シナリオ24：確認を発話した直後の解釈が1回失敗しても、
    再確認（同じ食い違いをもう一度渡す）はしない。照合待ちのまま保持し、
    次のターンで解釈が成功すれば `asked`・`responded` が入る（計画 §8）。
    """
    turn1 = await interpretation_client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn1.json()["conversation_id"]
    speaker_id = turn1.json()["user_message"]["speaker_id"]
    ref_message_id = turn1.json()["user_message"]["id"]

    async with session_factory() as session:
        discrepancy = await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="土曜と聞いていたが、前回は日曜と言っていた",
            source_message_id=ref_message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=ref_message_id,
            detected_by=DetectionSource.LLM.value,
        )
        await session.commit()
        discrepancy_id = discrepancy.id

    # このターンの返答が確認候補として渡される。
    fake_llm.push_interpretation("{}")
    fake_llm.push("念のためですが、今度の予定は日曜で合っていますか？")
    turn2 = await interpretation_client.post(
        "/api/chat", json={"text": "それより最近どう？", "conversation_id": conversation_id}
    )
    offering_reply_id = turn2.json()["reply"]["id"]

    # 直後のターンで、解釈そのものが失敗する（壊れたJSON）。
    fake_llm.push_interpretation("解釈に失敗しました（壊れたJSON）")
    fake_llm.push("うーん、そうですね。")
    await interpretation_client.post(
        "/api/chat", json={"text": "うーん", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, discrepancy_id)
        assert refreshed.asked_message_id is None
        ledger = await discrepancy_offer_ledger(
            session, conversation_id=conversation_id, target_speaker_id=speaker_id
        )
        # 失敗したので、まだ照合待ちのまま——再提示しない。
        assert ledger[discrepancy_id].pending_reply is not None
        assert ledger[discrepancy_id].pending_reply.id == offering_reply_id
        assert select_discrepancy_offers(ledger, limit=2, interpretation_ran=False) == []

    # 次のターンで解釈が成功し、直前に渡した確認候補（offering_reply）を
    # 照合して asked・responded が入る。
    fake_llm.push_interpretation(f'{{"asked_discrepancy_ids": [{discrepancy_id}]}}')
    fake_llm.push("承知しました。")
    await interpretation_client.post(
        "/api/chat", json={"text": "あ、日曜で合ってます", "conversation_id": conversation_id}
    )
    async with session_factory() as session:
        refreshed = await session.get(ConversationState, discrepancy_id)
        assert refreshed.asked_message_id == offering_reply_id
        assert refreshed.responded_message_id is not None
