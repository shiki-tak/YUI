"""v0.2「会話理解と修復」PR1：会話状態の器と規則（docs/plan/v0.2.md）。

**発言があったことと解決したことを分ける。** 相手の次の発言が来ただけでは
質問も食い違いも解決にならない。この PR1 では規則（辞書・正規表現）だけを
使い、解釈（LLM）でしか起きない解決・取消（discrepancy・correction・
request・confirmed）はモジュール関数を直接呼んで確かめる。
"""

from __future__ import annotations

from httpx import AsyncClient

from app.agent.conversation_state import (
    build_conversation_state_section,
    can_create_discrepancy,
    create_or_supersede_correction,
    create_state,
    detect_closing,
    detect_deferral_topic,
    detect_question,
    detect_request_question,
    get_open_states,
    is_acknowledgement_only,
)
from app.models import (
    ConversationState,
    ConversationStateKind,
    ConversationStateRefKind,
    ConversationStateStatus,
    ConversationStateWithdrawReason,
    DecisionSource,
)
from tests.conftest import FakeLLM

# --- 規則の検出（純粋関数） --------------------------------------------------


def test_detect_question_matches_sentence_ending_forms() -> None:
    assert detect_question("土曜と日曜、どちらですか？")
    assert detect_question("それでよろしいでしょうか")
    # 句点で終わる丁寧体（レビューで非対称を実測。依頼の文末と同じ形にした）。
    assert detect_question("明日の天気はどうですか。")
    assert detect_question("行きますか。")
    assert not detect_question("それでいいと思う")
    # 「でしょう」単独は推量の平叙文、「かな」は独り言でも使うため質問に含めない
    # （計画 §3 の語彙は ？／?／〜ますか／〜でしょうか／〜かしら に限る。
    # レビューで「明日は雨でしょう」「たぶん来るかな」の誤検出を実測）。
    assert not detect_question("明日は雨でしょう")
    assert not detect_question("たぶん来るかな")
    # 「そう(なん)ですか」は相づちで、質問ではない（コードレビューで
    # 「なるほど、そうですか！」等の誤検出を実測）。
    assert not detect_question("そうですか。")
    assert not detect_question("そうですか")
    assert not detect_question("なるほど、そうですか！")
    assert not detect_question("へえ、そうなんですか")
    # 「ですか」を含んでいても、相づちの定型文でなければ質問として扱う。
    assert detect_question("山は好きですか")


def test_detect_request_question_matches_request_forms_too() -> None:
    assert detect_request_question("明日の集合時間を教えて")
    assert detect_request_question("手伝ってほしい")
    assert not detect_request_question("今日は晴れているね")


def test_detect_deferral_topic_requires_topic_and_sentence_ending() -> None:
    assert detect_deferral_topic("映画の話はまた今度にしよう") == "映画"
    assert detect_deferral_topic("今日はいい天気だね") is None
    # 未来の提案（「今度は」で始まる）は延期ではないので拾わない。
    assert detect_deferral_topic("今度は映画の話をしよう") is None
    # 「については」は「は」を含むため、二重に「は」を要求すると一致しなく
    # なる（レビューで実測した死んだ分岐）。
    assert detect_deferral_topic("映画についてはまた今度") == "映画"
    # 前の文の内容を topic が飲み込まない（最後の文だけを見る）。
    assert detect_deferral_topic("疲れた。映画の話はまた今度ね") == "映画"
    assert detect_deferral_topic("今日は映画の話はまた今度にしよう") == "映画"
    # 語尾のバリエーション。
    assert detect_deferral_topic("映画の話はまた今度にしようね") == "映画"
    # 改行区切りでも、前の行を topic が飲み込まない（画面の入力欄は textarea
    # で Enter が改行になるため、通常操作で複数行の発言が届く。コード
    # レビューで「今日は疲れた\n映画の話はまた今度」の取りこぼしを実測）。
    assert detect_deferral_topic("今日は疲れた\n映画の話はまた今度") == "映画"


def test_detect_closing_requires_sentence_ending_form() -> None:
    assert detect_closing("そろそろ寝るね")
    assert detect_closing("今日はここまでにしよう")
    # 単なる感謝は終了の合図ではない（前回レビューの例）。
    assert not detect_closing("ありがとう")
    assert not detect_closing("友達に「またね」と言った")
    # 最後の文**全体**が定型句であることを求める。前置きのある平叙文
    # （用事の報告）を終了の合図として拾わない（レビューで誤検出を実測）。
    assert not detect_closing("明日は実家に帰ります")
    assert not detect_closing("12時に寝る予定です")
    assert not detect_closing("成績が落ちる")
    # 前の文に用事があっても、最後の文が定型句なら検出する。
    assert detect_closing("楽しかった。そろそろ寝るね")
    # 改行区切りでも同様（textarea の Enter は改行。コードレビューで
    # 「今日は楽しかった\nそろそろ寝るね」の取りこぼしを実測）。
    assert detect_closing("今日は楽しかった\nそろそろ寝るね")


def test_is_acknowledgement_only_catches_short_backchannel_words() -> None:
    assert is_acknowledgement_only("うん")
    assert is_acknowledgement_only("そうですね。")
    assert not is_acknowledgement_only("15時にしましょう")


# --- respond() への統合：規則だけで状態が作られる ----------------------------


async def test_partner_question_creates_question_to_yui(
    client: AsyncClient, session_factory
) -> None:
    """相手の質問から `question_to_yui` が作られる。

    このターンの YUI の返答自体が最初の応答候補になる（設計 §3：応答＝YUI の
    次の返答）ため、ターン完了後は `responded_message_id` が入っている。
    生成前の時点で「まだ答えていない質問」として渡ることは
    ``test_conversation_state_section_appears_in_prompt`` で確かめる。
    """
    turn = await client.post("/api/chat", json={"text": "明日の天気はどうですか？"})
    conversation_id = turn.json()["conversation_id"]
    reply_id = turn.json()["reply"]["id"]

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    assert len(states) == 1
    assert states[0].responded_message_id == reply_id
    assert states[0].status == ConversationStateStatus.OPEN.value


async def test_answering_marks_response_but_does_not_resolve(
    client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """規則だけでは、何ターン経っても解決にしない（解決は解釈が決める。PR3）。"""
    first = await client.post("/api/chat", json={"text": "明日の天気はどうですか？"})
    conversation_id = first.json()["conversation_id"]
    reply_id = first.json()["reply"]["id"]

    # さらに関係ない発言（質問ではない）を重ねても、規則だけでは resolved に
    # ならない。
    await client.post(
        "/api/chat", json={"text": "今日は天気がいいね", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    assert len(states) == 1
    # 応答はあったが、状態はまだ open のまま（未判定）。
    assert states[0].status == ConversationStateStatus.OPEN.value
    assert states[0].responded_message_id == reply_id


async def test_partner_asking_again_does_not_double_mark_response(
    client: AsyncClient, session_factory
) -> None:
    """聞き返し（相手のさらなる発言）で、最初の応答記録を上書きしない。"""
    first = await client.post("/api/chat", json={"text": "明日の予定はどうですか？"})
    conversation_id = first.json()["conversation_id"]
    first_reply_id = first.json()["reply"]["id"]

    await client.post(
        "/api/chat", json={"text": "何のことか教えて？", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    original = next(s for s in states if s.content == "明日の予定はどうですか？")
    # 最初の応答（1回目の YUI の返答）のまま。2回目の応答では上書きしない。
    assert original.responded_message_id == first_reply_id
    assert original.status == ConversationStateStatus.OPEN.value


async def test_deferral_is_detected_from_partner_message(
    client: AsyncClient, session_factory
) -> None:
    """規則は文末の形だけを見るので、延期と終了は別々のターンで確かめる。"""
    turn = await client.post("/api/chat", json={"text": "映画の話はまた今度にしよう"})
    conversation_id = turn.json()["conversation_id"]

    async with session_factory() as session:
        deferrals = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.DEFERRAL.value
        )
    assert len(deferrals) == 1
    assert deferrals[0].content == "映画"


async def test_closing_is_detected_from_partner_message(
    client: AsyncClient, session_factory
) -> None:
    turn = await client.post("/api/chat", json={"text": "そろそろ寝るね"})
    conversation_id = turn.json()["conversation_id"]

    async with session_factory() as session:
        closings = await get_open_states(
            session, conversation_id=conversation_id, kind=ConversationStateKind.CLOSING.value
        )
    assert len(closings) == 1


async def test_character_question_creates_question_to_partner_and_presented(
    client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    fake_llm.push("承知しました。ところで、山にはよく登られますか？")
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    conversation_id = turn.json()["conversation_id"]

    async with session_factory() as session:
        questions = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
        )
        presented = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.PRESENTED.value,
        )
    assert len(questions) == 1
    assert len(presented) == 1


async def test_partner_answering_yuis_question_marks_response_without_resolving(
    client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    fake_llm.push("山にはよく登られますか？")
    first = await client.post("/api/chat", json={"text": "こんにちは"})
    conversation_id = first.json()["conversation_id"]

    answer = await client.post(
        "/api/chat", json={"text": "たまに登ります", "conversation_id": conversation_id}
    )
    answer_message_id = answer.json()["user_message"]["id"]

    async with session_factory() as session:
        questions = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
        )
    assert len(questions) == 1
    assert questions[0].responded_message_id == answer_message_id
    # 「一度聞いた」だけが分かる。解決は次のターンの解釈が決める（PR3）。
    assert questions[0].status == ConversationStateStatus.OPEN.value


async def test_conversation_state_section_appears_in_prompt(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    await client.post("/api/chat", json={"text": "明日の予定を教えて"})
    assert "# この会話で" in fake_llm.last_system_prompt
    assert "まだ答えていない相手の質問" in fake_llm.last_system_prompt


async def test_long_question_is_truncated_in_content_and_prompt(
    client: AsyncClient, session_factory
) -> None:
    """質問の本文は `presented` と同じ上限で切る。

    `ChatRequest.text` は最大4000文字を許すため、切らないと `# この会話で`
    節だけで数千文字になり、ローカルモデルのコンテキストを圧迫する
    （コードレビューで実測）。
    """
    # ChatRequest.text の上限（4000文字）に収まる範囲で長文を作る。
    long_question = "明日の予定は" + "あ" * 3900 + "教えて"
    turn = await client.post("/api/chat", json={"text": long_question})
    conversation_id = turn.json()["conversation_id"]

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    assert len(states) == 1
    assert len(states[0].content) <= 200


# --- 別の相手では遷移しない（target_speaker_id の一致） ---------------------


async def test_state_targeted_at_one_speaker_is_not_affected_by_another(
    client: AsyncClient, session_factory
) -> None:
    """A への質問を B の発言・B へのYUIの返答で更新しない（複数話者の適用範囲）。"""
    a_turn = await client.post(
        "/api/chat",
        json={
            "text": "明日の予定はどうですか？",
            "speaker": {"source": "local", "external_id": "a", "display_name": "Aさん"},
        },
    )
    conversation_id = a_turn.json()["conversation_id"]
    # A 自身のターンの返答が、A 宛の質問の最初の応答候補になる（設計どおり）。
    a_reply_id = a_turn.json()["reply"]["id"]

    b_turn = await client.post(
        "/api/chat",
        json={
            "text": "こんにちは、はじめまして",
            "conversation_id": conversation_id,
            "speaker": {"source": "local", "external_id": "b", "display_name": "Bさん"},
        },
    )
    b_reply_id = b_turn.json()["reply"]["id"]

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    assert len(states) == 1
    # A 宛の質問は A 自身のターンの返答で応答済みになったまま、
    # B のターン（B の発言・B 向けの返答）では変わらない。
    assert states[0].target_speaker_id != b_turn.json()["user_message"]["speaker_id"]
    assert states[0].responded_message_id == a_reply_id
    assert states[0].responded_message_id != b_reply_id


async def test_yuis_question_to_a_is_not_answered_by_b(
    client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """YUI が A に聞いた質問を、B の発言で応答済みにしない。

    前のテストは A 自身のターンの返答が最初の応答候補になる形で target の
    絞り込みを間接的にしか確かめていなかった（レビュー指摘）。ここでは
    YUI が A へ質問した**後に**、無関係な B のターンが挟まっても
    `question_to_partner`（target=A）が応答済みにならないことを直接見る。
    """
    fake_llm.push("山にはよく登られますか？")
    a_turn = await client.post(
        "/api/chat",
        json={
            "text": "こんにちは",
            "speaker": {"source": "local", "external_id": "a3", "display_name": "Aさん"},
        },
    )
    conversation_id = a_turn.json()["conversation_id"]

    await client.post(
        "/api/chat",
        json={
            "text": "はじめまして",
            "conversation_id": conversation_id,
            "speaker": {"source": "local", "external_id": "b3", "display_name": "Bさん"},
        },
    )

    async with session_factory() as session:
        questions = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
        )
    assert len(questions) == 1
    # B の発言では応答済みにならない。
    assert questions[0].responded_message_id is None


async def test_yuis_reply_failure_then_b_does_not_answer_a_question_to_yui(
    client: AsyncClient, session_factory, fake_llm: FakeLLM
) -> None:
    """A の質問への YUI の返答が失敗し、次に B のターンが来ても、
    A 宛の `question_to_yui` を応答済みにしない（apply_character_message_rules
    の target 絞り込み。レビュー指摘）。

    A 自身のターンの返答が成功すると、その返答自体が最初の応答候補になって
    しまい（設計どおり）、target の絞り込みが効いているかを検出できない。
    生成を失敗させて A 自身の返答由来の状態が作られない状況を作ることで、
    絞り込みが実際に効いているかを区別する。
    """
    from app.llm.base import LLMError

    original_chat = fake_llm.chat

    async def failing_chat(*args, **kwargs):
        raise LLMError("接続できません（テスト）")

    fake_llm.chat = failing_chat  # type: ignore[method-assign]

    a_turn = await client.post(
        "/api/chat",
        json={
            "text": "明日の予定はどうですか？",
            "speaker": {"source": "local", "external_id": "a4", "display_name": "Aさん"},
        },
    )
    assert a_turn.status_code == 503
    conversation_id = (
        await client.get("/api/conversations")
    ).json()[0]["id"]

    fake_llm.chat = original_chat  # type: ignore[method-assign]

    b_turn = await client.post(
        "/api/chat",
        json={
            "text": "こんにちは、はじめまして",
            "conversation_id": conversation_id,
            "speaker": {"source": "local", "external_id": "b4", "display_name": "Bさん"},
        },
    )
    assert b_turn.status_code == 200

    async with session_factory() as session:
        states = await get_open_states(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
        )
    assert len(states) == 1
    # A への質問は、B のターン（B の発言・B 向けの返答）では応答済みにならない。
    assert states[0].responded_message_id is None


async def test_get_open_states_filters_by_target_speaker(
    client: AsyncClient, session_factory
) -> None:
    a_turn = await client.post(
        "/api/chat",
        json={
            "text": "明日の予定はどうですか？",
            "speaker": {"source": "local", "external_id": "a2", "display_name": "Aさん"},
        },
    )
    conversation_id = a_turn.json()["conversation_id"]
    b_turn = await client.post(
        "/api/chat",
        json={
            "text": "来週の話をしたいです",
            "conversation_id": conversation_id,
            "speaker": {"source": "local", "external_id": "b2", "display_name": "Bさん"},
        },
    )
    a_speaker_id = a_turn.json()["user_message"]["speaker_id"]
    b_speaker_id = b_turn.json()["user_message"]["speaker_id"]
    assert a_speaker_id != b_speaker_id

    async with session_factory() as session:
        for_a = await get_open_states(
            session, conversation_id=conversation_id, target_speaker_id=a_speaker_id
        )
        for_b = await get_open_states(
            session, conversation_id=conversation_id, target_speaker_id=b_speaker_id
        )
    assert {s.target_speaker_id for s in for_a} == {a_speaker_id}
    assert {s.target_speaker_id for s in for_b} == {b_speaker_id}


# --- discrepancy / correction：解釈でしか作らない経路を直接呼ぶ --------------


async def test_discrepancy_without_asked_does_not_get_a_response(
    client: AsyncClient, session_factory
) -> None:
    """`asked` が入っていない discrepancy は、規則からは応答を受け取らない
    （応答の記録は PR3 の解釈による照合でしか起きない）。
    """
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    source_message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        discrepancy = await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="土曜と聞いていたが日曜と言っている",
            source_message_id=source_message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=source_message_id,
            detected_by="llm",
        )
        discrepancy_id = discrepancy.id
        await session.commit()

    # 相手がさらに発言しても、規則は asked の無い discrepancy を応答済みにしない。
    await client.post(
        "/api/chat", json={"text": "そういえば元気？", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        stored = await session.get(ConversationState, discrepancy_id)
        assert stored is not None
        assert stored.asked_message_id is None
        assert stored.responded_message_id is None
        assert stored.status == ConversationStateStatus.OPEN.value


async def test_can_create_discrepancy_blocks_when_open_row_exists(
    client: AsyncClient, session_factory
) -> None:
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="食い違い",
            source_message_id=message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            detected_by="llm",
        )
        allowed = await can_create_discrepancy(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
        )
        await session.commit()
    assert allowed is False


async def test_can_create_discrepancy_ignores_misdetected_rows(
    client: AsyncClient, session_factory
) -> None:
    """誤検出として取り消した行は、同じ対象の再登録を妨げない。"""
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        from app.agent.conversation_state import withdraw_state

        row = await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="食い違い",
            source_message_id=message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            detected_by="llm",
        )
        await withdraw_state(
            session,
            row,
            reason=ConversationStateWithdrawReason.MISDETECTED.value,
            decided_by=DecisionSource.LLM.value,
        )
        allowed = await can_create_discrepancy(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
        )
        await session.commit()
    assert allowed is True


async def test_correction_blocks_new_discrepancy_on_same_target(
    client: AsyncClient, session_factory
) -> None:
    """有効な訂正のある対象には、食い違いを作らない（蒸し返さない）。"""
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        await create_or_supersede_correction(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            content="日曜が正しい",
            source_message_id=message_id,
            speaker_id=speaker_id,
            decided_by=DecisionSource.LLM.value,
        )
        allowed = await can_create_discrepancy(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
        )
        await session.commit()
    assert allowed is False


async def test_re_correction_supersedes_the_old_row(
    client: AsyncClient, session_factory
) -> None:
    """「日曜」の訂正の後に「やっぱり月曜」→ 古い行が superseded、新しい行が有効。"""
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        first = await create_or_supersede_correction(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            content="日曜が正しい",
            source_message_id=message_id,
            speaker_id=speaker_id,
            decided_by=DecisionSource.LLM.value,
        )
        first_id = first.id

        second = await create_or_supersede_correction(
            session,
            conversation_id=conversation_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            content="月曜が正しい",
            source_message_id=message_id,
            speaker_id=speaker_id,
            decided_by=DecisionSource.LLM.value,
        )
        await session.commit()

    async with session_factory() as session:
        old = await session.get(ConversationState, first_id)
        new = await session.get(ConversationState, second.id)
        assert old is not None and new is not None
        assert old.status == ConversationStateStatus.WITHDRAWN.value
        assert old.withdraw_reason == ConversationStateWithdrawReason.SUPERSEDED.value
        assert new.status == ConversationStateStatus.OPEN.value
        assert new.content == "月曜が正しい"


# --- プロンプトの節（純粋関数） ----------------------------------------------


def test_prompt_section_omits_followup_and_asked_states_left_to_pr3() -> None:
    """PR1 は question_to_yui/question_to_partner/deferral/closing/presented だけ扱う。

    `responded` が入って未判定の `question_to_yui` は、答え損ねと決まって
    いない（`followup_needed` は PR3 が立てる）ので節に渡さない。
    """

    def make(**kwargs) -> ConversationState:
        state = ConversationState(
            conversation_id=1,
            source_message_id=1,
            status=ConversationStateStatus.OPEN.value,
            detected_by="rule",
        )
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    states = [
        make(kind=ConversationStateKind.QUESTION_TO_YUI.value, content="明日は何時？"),
        make(
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
            content="お昼は何を食べた？",
            responded_message_id=99,
        ),
        make(
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
            content="山は好き？",
            responded_message_id=None,
        ),
        make(kind=ConversationStateKind.DEFERRAL.value, content="映画"),
        make(kind=ConversationStateKind.CLOSING.value, content="そろそろ寝るね"),
    ]
    section = build_conversation_state_section(states, window_message_ids=set())
    assert "# この会話で" in section
    assert "明日は何時？" in section
    assert "映画" in section
    assert "終わりにしようとしている" in section
    # 未判定（responded 済みだが followup_needed ではない）の質問は渡さない。
    assert "お昼は何を食べた？" not in section


def test_prompt_section_is_empty_when_nothing_open() -> None:
    assert build_conversation_state_section([], window_message_ids=set()) == ""


def test_prompt_section_deduplicates_repeated_closing_and_deferral() -> None:
    """規則は検出のたびに行を作るので、節では同じ内容をまとめる（レビュー指摘）。"""

    def make(**kwargs) -> ConversationState:
        state = ConversationState(
            conversation_id=1,
            source_message_id=1,
            status=ConversationStateStatus.OPEN.value,
            detected_by="rule",
        )
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    states = [
        make(kind=ConversationStateKind.CLOSING.value, content="そろそろ寝るね"),
        make(kind=ConversationStateKind.CLOSING.value, content="今日はここまで"),
        make(kind=ConversationStateKind.DEFERRAL.value, content="映画"),
        make(kind=ConversationStateKind.DEFERRAL.value, content="映画"),
    ]
    section = build_conversation_state_section(states, window_message_ids=set())
    assert section.count("終わりにしようとしている") == 1
    assert section.count("相手が後にすると言った話題：映画") == 1


def test_prompt_section_deduplicates_repeated_questions() -> None:
    """生成失敗（503）後に相手が同じ質問を再送すると、規則は重複制約を
    置かないため同じ本文の `question_to_yui` が複数件できうる（設計 §4
    末尾で受容済み）。節では同じ本文をまとめる（レビュー指摘）。
    """

    def make(**kwargs) -> ConversationState:
        state = ConversationState(
            conversation_id=1,
            source_message_id=1,
            status=ConversationStateStatus.OPEN.value,
            detected_by="rule",
        )
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    states = [
        make(kind=ConversationStateKind.QUESTION_TO_YUI.value, content="明日の予定は？"),
        make(kind=ConversationStateKind.QUESTION_TO_YUI.value, content="明日の予定は？"),
    ]
    section = build_conversation_state_section(states, window_message_ids=set())
    assert section.count("まだ答えていない相手の質問：「明日の予定は？」") == 1


def test_prompt_section_limits_answered_questions_to_recent_ones() -> None:
    """応答済みの question_to_partner は上限を超えて渡さない（レビュー指摘）。"""

    def make(index: int) -> ConversationState:
        state = ConversationState(
            conversation_id=1,
            source_message_id=1,
            status=ConversationStateStatus.OPEN.value,
            detected_by="rule",
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
            content=f"質問{index}",
            responded_message_id=index,
        )
        return state

    states = [make(i) for i in range(10)]
    section = build_conversation_state_section(states, window_message_ids=set())
    answered_lines = [line for line in section.split("\n") if "一度答えのあった質問" in line]
    assert len(answered_lines) == 5
    # 直近（新しいもの）が残る。
    assert "質問9" in section
    assert "質問0" not in section
