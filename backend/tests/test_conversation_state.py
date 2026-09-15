"""v0.2「会話理解と修復」PR1：会話状態の器と規則（docs/plan/v0.2.md）。

**発言があったことと解決したことを分ける。** 相手の次の発言が来ただけでは
質問も食い違いも解決にならない。この PR1 では規則（辞書・正規表現）だけを
使い、解釈（LLM）でしか起きない解決・取消（discrepancy・correction・
request・confirmed）はモジュール関数を直接呼んで確かめる。
"""

from __future__ import annotations

from httpx import AsyncClient

from app.agent.conversation_state import (
    CLOSING_FALLBACK_SENTENCE,
    InterpretationCandidates,
    _format_open_states,
    apply_interpretation_result,
    build_conversation_state_section,
    can_create_discrepancy,
    create_or_supersede_correction,
    create_state,
    detect_closing,
    detect_correction_marker,
    detect_deferral_topic,
    detect_question,
    detect_request_question,
    gather_interpretation_context,
    get_open_states,
    is_acknowledgement_only,
)
from app.agent.interpretation import (
    CorrectionPayload,
    InterpretationResult,
    WithdrawalPayload,
)
from app.models import (
    Conversation,
    ConversationState,
    ConversationStateKind,
    ConversationStateRefKind,
    ConversationStateStatus,
    ConversationStateWithdrawReason,
    DecisionSource,
    DetectionSource,
    Message,
    Speaker,
    SpeakerKind,
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


def test_detect_deferral_topic_keeps_demonstratives_whole() -> None:
    """指示語（その／あの／この）＋「話／件」は、`topic` の非貪欲マッチが
    「そ」＋「の話」のように壊れた1文字を返さない（ISSUE-043）。

    「その」は「そ」＋「の」で構成されるため、`_DEFERRAL_RE` 単独では
    非貪欲マッチが最短一致（1文字）を優先し、続く「の話」を任意グループの
    リテラル一致に譲ってしまう。指示語を1文字に壊すのではなく、フルの
    語をそのまま content にする。
    """
    assert detect_deferral_topic("その話はまた今度にしよう") == "その話"
    assert detect_deferral_topic("あの話はまた今度にしよう") == "あの話"
    assert detect_deferral_topic("この話はまた今度にしよう") == "この話"
    # 「件」でも同様。
    assert detect_deferral_topic("その件はまた今度にしよう") == "その件"
    # 指示語の直後に具体的な語が続く場合は、既存どおり一般パターンで
    # 「の話」を剥ぎ取る（指示語専用パターンは「その／あの／この」の直後が
    # 「話／件」で終わる場合にしか一致しない）。
    assert detect_deferral_topic("この前の話はまた今度にしよう") == "この前"
    # 対照：具体的な話題（指示語ではない）は既存どおり「の話」を剥ぎ取る。
    assert detect_deferral_topic("写真の話はまた今度にしよう") == "写真"


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


def test_detect_closing_matches_before_form_with_a_trailing_request() -> None:
    """「寝る前に」のように、終了語が文末以外の位置に付く形も終了の意図として
    拾う（計画 §7 シナリオ4：終了の合図と最後の依頼が同時に来る）。
    """
    assert detect_closing("寝る前に明日の集合時間だけ教えて")
    assert detect_closing("そろそろ帰る前に一つ聞いてもいい？")
    # 「前に」が付かない単なる予定の報告は、引き続き終了として拾わない。
    assert not detect_closing("明日は実家に帰る予定です")
    # 「〜前に」が依頼を伴わない習慣・伝聞の報告は終了の合図ではない
    # （レビューで誤検出を実測。要求するのは「最後の文が依頼の形」であること）。
    assert not detect_closing("毎晩寝る前にストレッチしてるんだ")
    assert not detect_closing("寝る前に本を読むのが習慣なんだよね")
    assert not detect_closing("友達が帰る前に一緒に写真を撮ったよ")
    assert not detect_closing("明日は帰る前に買い物して来るつもり")
    # 「最後の文が依頼の形」まで絞っても、終了の意図が無い一般的な質問・依頼は
    # 拾ってしまう。「前に」の直後に「だけ／一つ／ひとつ／最後に」を求めて
    # 除く（レビューで誤検出を実測）。
    assert not detect_closing("寝る前にストレッチするといいって本当？")
    assert not detect_closing("寝る前に飲むといい薬を教えて")
    assert not detect_closing("成績が落ちる前に対策を教えてほしい")
    assert not detect_closing("帰る前にやることリストを教えてくれる？")
    assert not detect_closing("おやすみ前に読む絵本、おすすめある？")


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


# --- 出力検査（v0.2 PR2） ----------------------------------------------------


async def test_closing_reply_with_no_new_question_passes_without_regeneration(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """closing が開いていても、返答が新しい質問を含まなければ何もしない。"""
    turn = await client.post("/api/chat", json={"text": "そろそろ寝るね"})
    reply_id = turn.json()["reply"]["id"]
    run = (await client.get(f"/api/conversations/messages/{reply_id}/run")).json()

    assert run["options"]["checks"]["closing_open"] is True
    assert run["options"]["checks"]["closing_new_question_detected"] is False
    assert run["options"]["checks"]["regenerated_for_closing"] is False
    assert len(fake_llm.calls) == 1


async def test_closing_reply_with_new_question_is_regenerated_once(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """closing が開いているのに新しい質問が出たら、1回だけ再生成する。"""
    fake_llm.push("承知しました。ちなみに明日は何をご予定ですか？")  # 1回目：質問が残る
    fake_llm.push("承知しました。おやすみなさいませ。")  # 2回目：質問が無い

    turn = await client.post("/api/chat", json={"text": "そろそろ寝るね"})
    reply = turn.json()["reply"]
    reply_id = reply["id"]
    run = (await client.get(f"/api/conversations/messages/{reply_id}/run")).json()

    assert reply["content"] == "承知しました。おやすみなさいませ。"
    assert len(fake_llm.calls) == 2
    checks = run["options"]["checks"]
    assert checks["closing_new_question_detected"] is True
    assert checks["regenerated_for_closing"] is True
    assert checks["closing_fell_back_to_template"] is False
    assert checks["closing_unresolved_due_to_open_question"] is False


async def test_closing_reply_falls_back_to_template_when_regeneration_still_asks(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """再生成でも質問が残り、答えを待っている相手の質問も無ければ、定型文へ落とす。"""
    fake_llm.push("承知しました。明日は何をご予定ですか？")
    fake_llm.push("では、明後日はいかがですか？")  # 2回目も質問が残る

    turn = await client.post("/api/chat", json={"text": "そろそろ寝るね"})
    reply = turn.json()["reply"]
    reply_id = reply["id"]
    run = (await client.get(f"/api/conversations/messages/{reply_id}/run")).json()

    assert reply["content"] == CLOSING_FALLBACK_SENTENCE
    assert len(fake_llm.calls) == 2
    checks = run["options"]["checks"]
    assert checks["closing_fell_back_to_template"] is True
    assert checks["closing_unresolved_due_to_open_question"] is False


async def test_closing_reply_does_not_fall_back_when_partners_question_is_open(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """相手の質問（question_to_yui）が未応答のまま残っているターンでは、答えと
    新しい質問を機械判定で分けられないため、定型文へは落とさず記録だけする。
    """
    fake_llm.push("えっと、寝る前に一つ確認してもいいですか？")  # 1回目
    fake_llm.push("承知しました。ところでそちらはどうでしたか？")  # 2回目も質問が残る

    # 「寝る前に」の1文にすることで closing と question_to_yui を同時に作る
    # （終了語が最後の文以外にある形は closing の判定対象外。計画 §7 シナリオ4）。
    turn = await client.post(
        "/api/chat", json={"text": "寝る前に明日の集合時間だけ教えて"}
    )
    reply = turn.json()["reply"]
    reply_id = reply["id"]
    run = (await client.get(f"/api/conversations/messages/{reply_id}/run")).json()

    # 定型文には落ちていない（2回目の生成結果がそのまま使われる）。
    assert reply["content"] == "承知しました。ところでそちらはどうでしたか？"
    checks = run["options"]["checks"]
    assert checks["closing_fell_back_to_template"] is False
    assert checks["closing_unresolved_due_to_open_question"] is True


async def test_closing_falls_back_even_if_an_earlier_question_was_already_answered(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """相手の質問に一度答えると `status` は `open` のまま残る（PR3 の解釈まで
    解決しない）。answered な question_to_yui まで「未応答」として数えると、
    一度でも質問された会話では定型への差し替えがずっと働かなくなる
    （レビューで実測。判定は `responded_message_id` の有無で見る）。
    """
    fake_llm.push("はい、元気ですよ。")
    first = await client.post("/api/chat", json={"text": "元気？"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push("承知しました。明日は何をご予定ですか？")
    fake_llm.push("では、明後日はいかがですか？")
    turn = await client.post(
        "/api/chat", json={"text": "そろそろ寝るね", "conversation_id": conversation_id}
    )
    reply = turn.json()["reply"]
    run = (await client.get(f"/api/conversations/messages/{reply['id']}/run")).json()

    assert reply["content"] == CLOSING_FALLBACK_SENTENCE
    checks = run["options"]["checks"]
    assert checks["closing_fell_back_to_template"] is True
    assert checks["closing_unresolved_due_to_open_question"] is False


async def test_deferral_mention_is_recorded_but_reply_is_not_changed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory
) -> None:
    """延期の話題の語が返答に含まれても、記録するだけで再生成しない。"""
    await client.post("/api/chat", json={"text": "映画の話はまた今度にしよう"})
    first_conversation = (await client.get("/api/conversations")).json()[0]
    conversation_id = first_conversation["id"]

    calls_before = len(fake_llm.calls)
    fake_llm.push("承知しました。映画、楽しみですね。")
    turn = await client.post(
        "/api/chat", json={"text": "そういえば元気？", "conversation_id": conversation_id}
    )
    reply = turn.json()["reply"]
    run = (await client.get(f"/api/conversations/messages/{reply['id']}/run")).json()

    # 語が含まれていても、そのまま通す（再生成しない）。
    assert reply["content"] == "承知しました。映画、楽しみですね。"
    assert len(fake_llm.calls) - calls_before == 1
    assert run["options"]["checks"]["deferral_topics_mentioned"] == ["映画"]


async def test_discrepancy_open_records_whether_reply_looks_assertive(
    client: AsyncClient, fake_llm: FakeLLM, session_factory
) -> None:
    """discrepancy が開いているとき、返答が断定的に見えるかを記録する
    （行動は変えない。設計 §4 手順6「記録のみ」）。discrepancy 自体は解釈
    （PR3）でしか作られないため、ここではテスト用に直接作る。
    """
    turn = await client.post("/api/chat", json={"text": "土曜に映画に行くよ"})
    conversation_id = turn.json()["conversation_id"]
    message_id = turn.json()["user_message"]["id"]
    speaker_id = turn.json()["user_message"]["speaker_id"]

    async with session_factory() as session:
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DISCREPANCY.value,
            content="土曜と聞いていたが日曜と言っている",
            source_message_id=message_id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=message_id,
            detected_by="llm",
        )
        await session.commit()

    fake_llm.push("日曜日ですね、承知しました。")  # 断定
    assertive_turn = await client.post(
        "/api/chat", json={"text": "そういえば元気？", "conversation_id": conversation_id}
    )
    assertive_run = (
        await client.get(f"/api/conversations/messages/{assertive_turn.json()['reply']['id']}/run")
    ).json()
    assert assertive_run["options"]["checks"]["discrepancy_open"] is True
    assert assertive_run["options"]["checks"]["discrepancy_possibly_assertive"] is True

    fake_llm.push("日曜日で合っていますでしょうか？")  # 確認する形（ヘッジあり）
    hedged_turn = await client.post(
        "/api/chat", json={"text": "うん", "conversation_id": conversation_id}
    )
    hedged_run = (
        await client.get(f"/api/conversations/messages/{hedged_turn.json()['reply']['id']}/run")
    ).json()
    assert hedged_run["options"]["checks"]["discrepancy_possibly_assertive"] is False


# --- 解釈への入力：訂正の対象を見せる（ISSUE-044） ----------------------------


def test_open_states_prompt_shows_correction_target() -> None:
    """訂正・食い違いは対象（ref）付きで解釈に見せる（ISSUE-044）。

    `create_or_supersede_correction` は (ref_kind, ref_id) の完全一致で旧訂正を
    探す。対象が見えないと、同じ事実を訂正し直す発言で解釈が別の対象（直前の
    発言）を指し、superseded が起きない（実モデルで 3 回中 2 回、月曜の訂正は
    作られるのに日曜が open のまま残った）。対象を持たない種類は従来どおり。
    """
    correction = ConversationState(
        id=7,
        conversation_id=1,
        kind=ConversationStateKind.CORRECTION.value,
        content="日曜",
        source_message_id=2,
        ref_kind=ConversationStateRefKind.MESSAGE.value,
        ref_id=1,
        status=ConversationStateStatus.OPEN.value,
        detected_by=DetectionSource.LLM.value,
    )
    presented = ConversationState(
        id=8,
        conversation_id=1,
        kind=ConversationStateKind.PRESENTED.value,
        content="日曜ですね",
        source_message_id=3,
        status=ConversationStateStatus.OPEN.value,
        detected_by=DetectionSource.RULE.value,
    )
    text = _format_open_states([correction, presented])
    assert "- [7] correction（対象: message 1）: 日曜" in text
    assert "- [8] presented: 日曜ですね" in text


async def test_interpretation_candidates_accept_open_correction_target(session_factory) -> None:
    """開いている訂正の対象は、直近の窓の外でも有効な ref として受け付ける
    （ISSUE-044）。

    `_ref_is_valid` は message の ref を直近の窓＋今の発言に限る。元の発言が
    窓から出た後に同じ事実を訂正し直すと、解釈が正しく元の対象を指しても検証で
    捨てられ、superseded が起きない。解釈に見せた対象は受け付ける。
    """
    async with session_factory() as session:
        speaker = Speaker(source="local_text", external_id="dev", display_name="開発者")
        conversation = Conversation()
        session.add_all([speaker, conversation])
        await session.flush()

        texts = [
            "土曜に映画に行くよ",
            "土曜ですね",
            "ごめん、土曜じゃなくて日曜だった",
            "日曜ですね",
            "映画館は混むかな",
            "週末は混みますね",
            "ああ、日曜でもなくて、やっぱり月曜だった",
        ]
        messages: list[Message] = []
        for index, text in enumerate(texts):
            is_user = index % 2 == 0
            message = Message(
                conversation_id=conversation.id,
                speaker_kind=SpeakerKind.USER.value if is_user else SpeakerKind.CHARACTER.value,
                speaker_id=speaker.id if is_user else None,
                content=text,
            )
            session.add(message)
            messages.append(message)
        await session.flush()
        original, current = messages[0], messages[-1]

        await create_state(
            session,
            conversation_id=conversation.id,
            kind=ConversationStateKind.CORRECTION.value,
            content="日曜",
            source_message_id=messages[2].id,
            speaker_id=speaker.id,
            target_speaker_id=speaker.id,
            ref_kind=ConversationStateRefKind.MESSAGE.value,
            ref_id=original.id,
            detected_by=DetectionSource.LLM.value,
        )
        open_states = await get_open_states(
            session, conversation_id=conversation.id, target_speaker_id=speaker.id
        )
        history = messages[:-1]
        context, candidates = await gather_interpretation_context(
            session,
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            current_message=current,
            history=history,
            memory_items=[],
            open_states=open_states,
            discrepancy_ledger={},
            context_messages=2,
        )

    # 元の発言は直近2件の窓の外にある
    assert original.id not in {message.id for message in history[-2:]}
    # それでも、開いている訂正の対象として受け付ける・解釈にも見せる
    assert original.id in candidates.message_ids
    assert f"（対象: message {original.id}）" in context


async def _correction_fixture(session):
    """土曜→日曜の訂正が1件開いている会話。

    返り値は (speaker, conversation, messages, correction)。
    """
    speaker = Speaker(source="local_text", external_id="dev", display_name="開発者")
    conversation = Conversation()
    session.add_all([speaker, conversation])
    await session.flush()
    messages: list[Message] = []
    for text, is_user in [
        ("土曜に映画に行くよ", True),
        ("土曜ですね", False),
        ("ごめん、土曜じゃなくて日曜だった", True),
        ("日曜ですね、覚えておきます", False),
        ("ああ、日曜でもなくて、やっぱり月曜だった", True),
    ]:
        message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.USER.value if is_user else SpeakerKind.CHARACTER.value,
            speaker_id=speaker.id if is_user else None,
            content=text,
        )
        session.add(message)
        messages.append(message)
    await session.flush()
    correction = await create_state(
        session,
        conversation_id=conversation.id,
        kind=ConversationStateKind.CORRECTION.value,
        content="日曜",
        source_message_id=messages[2].id,
        speaker_id=speaker.id,
        target_speaker_id=speaker.id,
        ref_kind=ConversationStateRefKind.MESSAGE.value,
        ref_id=messages[0].id,
        detected_by=DetectionSource.LLM.value,
    )
    return speaker, conversation, messages, correction


def _candidates_for(
    states: list[ConversationState], messages: list[Message]
) -> InterpretationCandidates:
    return InterpretationCandidates(
        open_states={state.id: state for state in states},
        message_ids={message.id for message in messages},
    )


async def test_explicit_supersede_replaces_older_correction(session_factory) -> None:
    """同じ事実の訂正し直しは、解釈が置き換える訂正を `withdrawals`（superseded）で
    名指しし、同じ結果に検証を通った新しい correction があるときだけ置き換える
    （ISSUE-044）。

    実モデルは2回目の訂正で元の対象（土曜の発言）ではなく直前の発言や YUI の
    復唱を指すため、`(ref_kind, ref_id)` の一致では旧訂正が残る。source を辿って
    「同じ事実」と推測する経路は、同じ発言に含まれていた別の事実の訂正まで
    消すため置かない（レビューで再現）。
    """
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        current = messages[-1]
        result = InterpretationResult(
            correction=CorrectionPayload(
                ref_kind=ConversationStateRefKind.MESSAGE.value,
                ref_id=messages[3].id,  # YUI の復唱を指した（元の対象ではない）
                content="月曜",
            ),
            withdrawals=[WithdrawalPayload(state_id=old.id, reason="superseded")],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=_candidates_for([old], messages),
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=current,
        )
        await session.refresh(old)
        assert old.status == ConversationStateStatus.WITHDRAWN.value
        assert old.withdraw_reason == ConversationStateWithdrawReason.SUPERSEDED.value
        assert summary["created_correction"] is True
        assert summary["withdrawn_state_ids"] == [old.id]
        open_corrections = await get_open_states(
            session,
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            kind=ConversationStateKind.CORRECTION.value,
        )
        assert [state.content for state in open_corrections] == ["月曜"]


async def test_supersede_without_replacement_is_dropped(session_factory) -> None:
    """置き換え先の correction が無い superseded は受け付けない（有効な訂正が
    無言で消えないため。従来どおり）。"""
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        result = InterpretationResult(
            withdrawals=[WithdrawalPayload(state_id=old.id, reason="superseded")],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=_candidates_for([old], messages),
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=messages[-1],
        )
        await session.refresh(old)
        assert old.status == ConversationStateStatus.OPEN.value
        assert summary["withdrawn_state_ids"] == []
        assert any("withdrawals" in item for item in summary["dropped"])


async def test_supersede_only_applies_to_corrections(session_factory) -> None:
    """新しい correction があっても、correction 以外の状態への superseded は
    受け付けない（質問や延期が訂正の置き換えとして消えないため）。"""
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        question = await create_state(
            session,
            conversation_id=conversation.id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
            content="何時から？",
            source_message_id=messages[2].id,
            speaker_id=speaker.id,
            target_speaker_id=speaker.id,
        )
        result = InterpretationResult(
            correction=CorrectionPayload(
                ref_kind=ConversationStateRefKind.MESSAGE.value,
                ref_id=messages[2].id,
                content="月曜",
            ),
            withdrawals=[WithdrawalPayload(state_id=question.id, reason="superseded")],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=_candidates_for([old, question], messages),
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=messages[-1],
        )
        await session.refresh(question)
        await session.refresh(old)
        assert question.status == ConversationStateStatus.OPEN.value
        # 名指しされていない旧訂正も、ref が違うので置き換わらない（推測しない）
        assert old.status == ConversationStateStatus.OPEN.value
        assert summary["created_correction"] is True
        assert summary["withdrawn_state_ids"] == []


async def test_memory_ref_colliding_with_withdrawn_state_id_is_dropped(session_factory) -> None:
    """同じ番号を「取り消す状態の id」と「記憶の ref_id」の両方に使った出力は、
    どちらも適用しない（ISSUE-044 の残件で観測した、状態行の [N] を memory N と
    書く取り違え）。記憶と状態は id の空間が重なるため、偶然一致した記憶に
    訂正が付き、さらに旧訂正が消える経路を塞ぐ。"""
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        candidates = _candidates_for([old], messages)
        # 取り違えが実害になる条件：同じ番号の記憶が今回の候補に載っている
        candidates.memory_ids.add(old.id)
        result = InterpretationResult(
            correction=CorrectionPayload(
                ref_kind=ConversationStateRefKind.MEMORY.value,
                ref_id=old.id,
                content="月曜",
            ),
            withdrawals=[WithdrawalPayload(state_id=old.id, reason="superseded")],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=candidates,
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=messages[-1],
        )
        await session.refresh(old)
        assert old.status == ConversationStateStatus.OPEN.value
        assert summary["created_correction"] is False
        assert summary["superseded_correction_ids"] == []
        assert any("取り違え" in item for item in summary["dropped"])


async def test_memory_ref_colliding_with_a_cancelled_state_id_is_kept(session_factory) -> None:
    """同じ番号でも、取消の理由が `superseded` でなければ訂正を捨てない。

    取り違えが観測されたのは旧訂正の置き換え（`superseded`）のときだけ。
    `cancelled` 等まで含めてガードすると、**正当な訂正を捨てる**——質問の状態 N
    を取り消しつつ記憶 N を訂正する場合、新しい DB では状態と記憶の id が同じ
    番号から始まるため偶然一致しやすい（レビューで再現）。
    """
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        candidates = _candidates_for([old], messages)
        candidates.memory_ids.add(old.id)
        result = InterpretationResult(
            correction=CorrectionPayload(
                ref_kind=ConversationStateRefKind.MEMORY.value,
                ref_id=old.id,
                content="月曜",
            ),
            withdrawals=[WithdrawalPayload(state_id=old.id, reason="cancelled")],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=candidates,
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=messages[-1],
        )
        assert summary["created_correction"] is True
        assert not any("取り違え" in item for item in summary["dropped"])


async def test_duplicate_withdrawals_apply_once(session_factory) -> None:
    """同じ (state_id, reason) が重複して返っても 1 回だけ適用し、記録も 1 件。"""
    async with session_factory() as session:
        speaker, conversation, messages, old = await _correction_fixture(session)
        result = InterpretationResult(
            correction=CorrectionPayload(
                ref_kind=ConversationStateRefKind.MESSAGE.value,
                ref_id=messages[3].id,
                content="月曜",
            ),
            withdrawals=[
                WithdrawalPayload(state_id=old.id, reason="superseded"),
                WithdrawalPayload(state_id=old.id, reason="superseded"),
            ],
        )
        summary = await apply_interpretation_result(
            session,
            result,
            candidates=_candidates_for([old], messages),
            conversation_id=conversation.id,
            target_speaker_id=speaker.id,
            speaker_id=speaker.id,
            current_message=messages[-1],
        )
        assert summary["withdrawn_state_ids"] == [old.id]
        assert summary["superseded_correction_ids"] == [old.id]


# --- 言い直しの語を事実として渡す（ISSUE-048） -------------------------------


def test_detect_correction_marker_finds_explicit_rewording() -> None:
    """明示的な言い直しの語だけを拾う（ISSUE-048）。

    語の有無だけを見て、訂正かどうかの判断はしない。判断は解釈の側に残す。
    """
    assert detect_correction_marker("ごめん、土曜じゃなくて日曜だった") == "じゃなくて"
    assert detect_correction_marker("すみません、間違えました") == "間違え"
    assert detect_correction_marker("ああ、日曜でもなくて、やっぱり月曜だった") == "やっぱり"
    # 内容が食い違っていても、言い直しの語が無ければ None。
    assert detect_correction_marker("あの映画、日曜の回がすごく混みそうだね") is None
    assert detect_correction_marker("クロワッサンがすごく美味しかった") is None


async def test_interpretation_context_states_whether_a_rewording_marker_exists(
    session_factory,
) -> None:
    """解釈への入力に、言い直しの語の有無を**事実として**添える（ISSUE-048）。

    指示文で「語が無ければ discrepancy」と書くだけでは、実モデルは記憶と
    食い違う発言を 8/8 で `correction` にした。同じ入力にこの1行を足すと
    discrepancy 側へ変わる（実測）。語の有無は規則で決まるので、LLM に
    探させずコードが判定して渡す。
    """
    async with session_factory() as session:
        speaker = Speaker(source="local_text", external_id="dev", display_name="開発者")
        conversation = Conversation()
        session.add_all([speaker, conversation])
        await session.flush()

        async def context_for(text: str) -> str:
            message = Message(
                conversation_id=conversation.id,
                speaker_kind=SpeakerKind.USER.value,
                speaker_id=speaker.id,
                content=text,
            )
            session.add(message)
            await session.flush()
            context, _ = await gather_interpretation_context(
                session,
                conversation_id=conversation.id,
                target_speaker_id=speaker.id,
                current_message=message,
                history=[],
                memory_items=[],
                open_states=[],
                discrepancy_ledger={},
                context_messages=6,
            )
            return context

        implicit = await context_for("あの映画、日曜の回がすごく混みそうだね")
        assert "言い直しの語" in implicit
        assert "**含まれていない**" in implicit

        explicit = await context_for("ごめん、土曜じゃなくて日曜だった")
        assert "言い直しの語「じゃなくて」が含まれている" in explicit
