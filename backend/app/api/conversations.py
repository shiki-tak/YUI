"""会話履歴・振り返り・記憶候補・理想の返答の API。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.character_state import active_states, create_state
from app.agent.conversation_state import (
    RESOLVABLE_KINDS,
    expire_open_states,
    resolve_state,
    withdraw_reason_applies,
    withdraw_state,
)
from app.agent.delivery import apply_delivery_state
from app.agent.memory_store import create_memory
from app.agent.reflection import ReflectionParseError, extract_candidates, format_transcript
from app.agent.state_reflection import StateReflectionError, propose_state_candidates
from app.agent.turn_lock import conversation_locks
from app.config import Settings, get_settings
from app.db import get_session
from app.llm import get_llm_client
from app.llm.base import LLMClient, LLMError
from app.models import (
    CandidateStatus,
    Conversation,
    ConversationState,
    ConversationStateKind,
    ConversationStateStatus,
    DecisionSource,
    DeliveryState,
    IdealResponse,
    MemoryCandidate,
    Message,
    RunRecord,
    Speaker,
    SpeakerKind,
    SpeechRun,
    StateKind,
    StateStatus,
    utcnow,
)
from app.persona import load_persona
from app.schemas import (
    CandidateDecision,
    ConversationDetail,
    ConversationOut,
    ConversationStateDecision,
    ConversationStateOut,
    DeliveryUpdate,
    IdealResponseCreate,
    IdealResponseOut,
    MemoryCandidateOut,
    MessageOut,
    RunRecordDetail,
    SpeechRunOut,
)
from app.voice import get_speech_client
from app.voice.base import SpeechClient, SpeechError

router = APIRouter(prefix="/conversations", tags=["conversations"])


async def _release_reflection(session: AsyncSession, conversation_id: int) -> None:
    """振り返りの開始を取り消し、やり直せる状態に戻す。"""
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(reflection_started_at=None)
    )
    await session.commit()


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    limit: int = 50, session: AsyncSession = Depends(get_session)
) -> list[Conversation]:
    stmt = select(Conversation).order_by(Conversation.id.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


@router.get("/candidates/pending", response_model=list[MemoryCandidateOut])
async def list_pending_candidates(
    limit: int = 100, session: AsyncSession = Depends(get_session)
) -> list[MemoryCandidate]:
    """未判断の記憶候補を会話をまたいで取得する。

    画面を再読み込みしても、抽出済みの候補を採用できるようにするため。
    """
    stmt = (
        select(MemoryCandidate)
        .where(MemoryCandidate.status == CandidateStatus.PENDING.value)
        .order_by(MemoryCandidate.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int, session: AsyncSession = Depends(get_session)
) -> Conversation:
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = (await session.execute(stmt)).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会話が見つかりません。")
    return conversation


async def _conversation_messages(session: AsyncSession, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        .options(selectinload(Message.speaker))
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars())


async def _sole_partner(session: AsyncSession, conversation_id: int) -> Speaker | None:
    """その会話にひとりだけいる相手。複数いれば None を返す。

    「最後に話した人」を相手として扱うと、複数の相手がいる会話で、別人の
    情報をその人のものとして保存する（v0.1 レビューの指摘）。
    """
    stmt = (
        select(Speaker)
        .join(Message, Message.speaker_id == Speaker.id)
        .where(
            Message.conversation_id == conversation_id,
            Message.speaker_kind == SpeakerKind.USER.value,
        )
        .distinct()
        .limit(2)
    )
    found = list((await session.execute(stmt)).scalars())
    return found[0] if len(found) == 1 else None


@router.post("/{conversation_id}/end", response_model=list[MemoryCandidateOut])
async def end_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> list[MemoryCandidate]:
    """会話を終了し、長期記憶の候補を抽出する。採用は別途 /candidates で行う。"""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会話が見つかりません。")

    # 開始権の確定はロックの内側で行う。外で確定すると、ロック待ちの時間が
    # 振り返りの開始権を取る前に人格を解決する。開始権を取った後に失敗すると、
    # 解放されないまま reflection_stale_seconds を過ぎるまで再試行できない。
    # 人格の読み込みはファイルを読むため、ここで失敗しうる。
    character_name = load_persona().name

    # 回収期限に含まれ、稼働中の振り返りまで期限切れとみなされてしまう。
    async with conversation_locks.hold(conversation_id):
        await session.refresh(conversation)
        if conversation.reflection_completed_at is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, "この会話はすでに終了しています。")

        # 同一プロセスではロックが排他を保証する。ここでの期限判定は、
        # プロセスが落ちて開始だけが残った場合を回収するためのもの。
        now = utcnow()
        stale_before = now - timedelta(seconds=settings.reflection_stale_seconds)
        # UPDATE の結果は CursorResult。rowcount で更新できたかを判定する。
        claimed = cast(
            "CursorResult[Any]",
            await session.execute(
                update(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.reflection_completed_at.is_(None),
                    (Conversation.reflection_started_at.is_(None))
                    | (Conversation.reflection_started_at < stale_before),
                )
                .values(reflection_started_at=now)
                # SQLite は timezone を落として返すため、条件の評価を Python 側で
                # 行わせない。判定は SQL に任せる。
                .execution_options(synchronize_session=False)
            ),
        )
        if claimed.rowcount == 0:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "この会話の振り返りは実行中です。しばらく待って再試行してください。",
            )

        sole_partner = await _sole_partner(session, conversation_id)
        # 生成に入る前に DB の書き込みロックを手放す（会話 API と同じ理由）。
        await session.commit()

        # 相手がひとりの会話でだけ、関係性の候補を作る。複数いる会話で
        # 「その相手との関係」を最後の話者へ寄せると、別人の関係になる。
        partner_id = sole_partner.id if sole_partner else None
        messages = await _conversation_messages(session, conversation_id)
        current_states = await active_states(session, speaker_id=partner_id, mode=conversation.mode)

        try:
            # モデルを呼んでいる間は、書き込みのトランザクションを開かない。
            # 開いたまま待つと、別の会話の書き込みが「database is locked」で
            # 失敗する。抽出はここでは保存せず、すべて成功してからまとめて
            # 保存する（v0.1 レビューの指摘）。
            candidates = await extract_candidates(
                session,
                llm=llm,
                conversation=conversation,
                character_name=character_name,
            )
            # 関心・関係性の更新候補は、別の呼び出しで作る。同じ指示文へ項目を
            # 足すと記憶の抽出が落ちるため（ISSUE-017 で実測）。
            state_payloads = await propose_state_candidates(
                llm=llm,
                transcript=format_transcript(messages, character_name),
                partner_speaker_id=partner_id,
                current_states=current_states,
            )
        except (LLMError, ReflectionParseError, StateReflectionError) as exc:
            # 抽出できなかった会話を処理中・終了済みのまま残すと、やり直せない。
            # 特に出力の解析失敗は「候補なしの成功」と区別する必要がある。
            # 途中まで作ったものは捨てる。残すと、再試行で二重に保存される。
            await session.rollback()
            await _release_reflection(session, conversation_id)
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        # ここから保存。すべて成功したものだけを、1つのトランザクションで書く。
        session.add_all(candidates)
        for payload in state_payloads:
            is_relationship = payload.kind == StateKind.RELATIONSHIP.value
            await create_state(
                session,
                kind=payload.kind,
                content=payload.content,
                topic=None if is_relationship else payload.topic,
                subject_speaker_id=partner_id if is_relationship else None,
                # 非公開の会話から作った状態は、その相手との会話に限る。
                visible_to_speaker_id=partner_id,
                source_conversation_id=conversation.id,
                status=StateStatus.PENDING.value,
                reason=payload.reason or "会話の振り返りから",
            )
        await session.flush()

        completed = utcnow()
        # 開いている会話状態はすべて expired にする（resolved にはしない。
        # 会話が終わったことと問題が解決したことは別。計画 §5）。抽出が
        # 失敗した場合はここへ到達しないため、会話状態は開いたまま残る
        # （既存どおり 503 で再試行できる）。
        await expire_open_states(session, conversation_id=conversation_id, at=completed)
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(reflection_completed_at=completed, ended_at=completed)
        )
        await session.commit()

    return candidates


@router.get("/{conversation_id}/candidates", response_model=list[MemoryCandidateOut])
async def list_candidates(
    conversation_id: int, session: AsyncSession = Depends(get_session)
) -> list[MemoryCandidate]:
    stmt = (
        select(MemoryCandidate)
        .where(MemoryCandidate.conversation_id == conversation_id)
        .order_by(MemoryCandidate.id)
    )
    return list((await session.execute(stmt)).scalars())


@router.post("/candidates/{candidate_id}/decide", response_model=MemoryCandidateOut)
async def decide_candidate(
    candidate_id: int,
    payload: CandidateDecision,
    session: AsyncSession = Depends(get_session),
) -> MemoryCandidate:
    """候補を採用して長期記憶にするか、却下する。"""
    candidate = await session.get(MemoryCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "候補が見つかりません。")
    if candidate.status != CandidateStatus.PENDING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "この候補はすでに判断済みです。")

    if payload.decision == "reject":
        candidate.status = CandidateStatus.REJECTED.value
        return candidate

    if payload.subject_to_none and payload.subject_speaker_id is not None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "subject_speaker_id と subject_to_none は同時に指定できません。",
        )
    if payload.subject_to_none:
        subject_speaker_id = None
    elif payload.subject_speaker_id is not None:
        subject_speaker_id = payload.subject_speaker_id
    else:
        subject_speaker_id = candidate.subject_speaker_id

    memory = await create_memory(
        session,
        kind=(payload.kind.value if payload.kind else candidate.kind),
        content=(payload.content or candidate.content),
        subject_speaker_id=subject_speaker_id,
        visible_to_speaker_id=candidate.visible_to_speaker_id,
        certainty=(payload.certainty.value if payload.certainty else candidate.certainty),
        provenance=(payload.provenance.value if payload.provenance else candidate.provenance),
        visibility=(payload.visibility.value if payload.visibility else candidate.visibility),
        keywords=(payload.keywords if payload.keywords is not None else candidate.keywords),
        occurred_at=candidate.occurred_at,
        source_message_id=candidate.source_message_id,
        source_conversation_id=candidate.conversation_id,
        reason=payload.reason or "会話の振り返りから採用",
    )
    candidate.status = CandidateStatus.ACCEPTED.value
    candidate.accepted_memory_id = memory.id
    return candidate


@router.get("/{conversation_id}/states", response_model=list[ConversationStateOut])
async def list_conversation_states(
    conversation_id: int,
    status_: str | None = Query(default=None, alias="status"),
    kind: str | None = None,
    target_speaker_id: int | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[ConversationState]:
    """会話状態の一覧（計画 §6）。`status`・`kind`・`target_speaker_id` で絞れる。

    `kind` はカンマ区切りで複数指定できる（例：`kind=correction,discrepancy`。
    会話終了後の「訂正の候補」表示が使う）。`target_speaker_id` を指定しないと
    会話にいる全員宛の状態が返る——`/chat` の応答（`ChatResponse
    .conversation_states`）は話している相手だけに絞っているため、画面側で
    同じ絞り方をしたい場合はここも指定する（レビュー指摘：絞り方が経路ごとに
    違うと、複数話者の会話で「誰の未回答質問か」が食い違って見える）。
    空文字は「指定しない」として扱う（未入力のセレクトボックス等から
    そのまま渡っても絞り込みが外れないようにする）。
    """
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id
    )
    if status_:
        if status_ not in {s.value for s in ConversationStateStatus}:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"status が不正です: {status_}")
        stmt = stmt.where(ConversationState.status == status_)
    if kind:
        kinds = [k.strip() for k in kind.split(",") if k.strip()]
        known_kinds = {k.value for k in ConversationStateKind}
        unknown = [k for k in kinds if k not in known_kinds]
        if unknown:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"kind が不正です: {'、'.join(unknown)}"
            )
        stmt = stmt.where(ConversationState.kind.in_(kinds))
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    stmt = stmt.order_by(ConversationState.id)
    return list((await session.execute(stmt)).scalars())


@router.post(
    "/{conversation_id}/states/{state_id}/decide", response_model=ConversationStateOut
)
async def decide_conversation_state(
    conversation_id: int,
    state_id: int,
    payload: ConversationStateDecision,
    session: AsyncSession = Depends(get_session),
) -> ConversationState:
    """開発者が会話状態を解決・取消にする（計画 §6）。

    誤検出の是正と、解釈（LLM）を有効にしていない構成での手動解決に使う。
    `decided_by = operator` にし、作成の経路（`detected_by`）は変えない。
    """
    # 会話ロックの内側で行う。解釈（PR3）が同じ行を更新するターンと重なると
    # 後勝ちになるため（レビュー指摘）、`/chat` と同じロックで直列化する。
    async with conversation_locks.hold(conversation_id):
        state = await session.get(ConversationState, state_id)
        if state is None or state.conversation_id != conversation_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会話状態が見つかりません。")
        if state.status != ConversationStateStatus.OPEN.value:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "この会話状態は open ではありません（判断済みか、失効しています）。",
            )

        if payload.decision == "resolved":
            # 「解決」の概念がある種類だけに限る（計画 3節の表。レビュー
            # 指摘：以前はどの種類の resolved も無条件に受け付けていた）。
            if state.kind not in RESOLVABLE_KINDS:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"「{state.kind}」に解決の概念はありません。",
                )
            await resolve_state(
                session, state, resolved_message_id=None, decided_by=DecisionSource.OPERATOR.value
            )
        else:
            assert payload.withdraw_reason is not None
            # 解釈（LLM）の経路と同じ検証を通す。種類ごとに使える取消理由は
            # 決まっており（計画 3節）、operator 操作でも例外にしない
            # （レビュー指摘：以前は検証無しで `question_to_yui` に `reopened`
            # のような不整合な組み合わせを受け入れていた。`superseded` は
            # どの種類にも紐付かないため、この経路では常に拒否される）。
            if not withdraw_reason_applies(state.kind, payload.withdraw_reason):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"「{state.kind}」に「{payload.withdraw_reason}」は使えません。",
                )
            await withdraw_state(
                session,
                state,
                reason=payload.withdraw_reason,
                decided_by=DecisionSource.OPERATOR.value,
                resolved_message_id=None,
            )
        await session.commit()
        return state


@router.get("/messages/{message_id}", response_model=MessageOut)
async def get_message(message_id: int, session: AsyncSession = Depends(get_session)) -> Message:
    """根拠の発言の本文を確認する。記憶がどの発言から作られたかを追うため。"""
    message = await session.get(Message, message_id)
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "発言が見つかりません。")
    return message


@router.get("/messages/{message_id}/run", response_model=RunRecordDetail)
async def get_run_record(
    message_id: int, session: AsyncSession = Depends(get_session)
) -> RunRecord:
    """返答の実行記録。参照した記憶・モデル・設定・応答時間を確認する。"""
    stmt = select(RunRecord).where(RunRecord.message_id == message_id)
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "実行記録が見つかりません。")
    return run


async def _character_message(session: AsyncSession, message_id: int) -> Message:
    message = await session.get(Message, message_id)
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "発言が見つかりません。")
    if message.speaker_kind != SpeakerKind.CHARACTER.value:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "読み上げの対象はキャラクターの発言です。")
    return message


@router.get(
    "/messages/{message_id}/speech",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def get_speech(
    message_id: int,
    session: AsyncSession = Depends(get_session),
    speech: SpeechClient = Depends(get_speech_client),
    settings: Settings = Depends(get_settings),
) -> Response:
    """返答を読み上げた音声。字幕は発言の本文をそのまま使う。

    音声は都度合成する。保存や先読みは、どの区間が遅いかを測ってから判断する。
    """
    if not settings.speech_enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "音声合成は無効になっています。")
    message = await _character_message(session, message_id)
    try:
        result = await speech.synthesize(message.content)
    except SpeechError as exc:
        # 音声が出せなくても会話は続けられる。失敗として返し、文字は残す。
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    # 区間ごとの時間を残す。聞き直すたびに 1 行増やし、同じ文章の合成が
    # 毎回どれだけかかるかを見られるようにする。
    session.add(
        SpeechRun(
            message_id=message.id,
            provider=result.provider,
            speaker_id=result.speaker_id,
            engine_version=result.engine_version,
            query_ms=result.query_ms,
            synthesis_ms=result.synthesis_ms,
            audio_ms=result.audio_ms,
            byte_size=len(result.audio),
        )
    )
    return Response(content=result.audio, media_type=result.media_type)


@router.get("/messages/{message_id}/speech-runs", response_model=list[SpeechRunOut])
async def list_speech_runs(
    message_id: int, session: AsyncSession = Depends(get_session)
) -> list[SpeechRun]:
    """この発言を合成したときの記録。新しい順に返す。"""
    stmt = select(SpeechRun).where(SpeechRun.message_id == message_id).order_by(SpeechRun.id.desc())
    return list((await session.execute(stmt)).scalars())


@router.post("/messages/{message_id}/delivery", response_model=MessageOut)
async def update_delivery(
    message_id: int,
    payload: DeliveryUpdate,
    session: AsyncSession = Depends(get_session),
) -> Message:
    """再生の開始・完了・中断を記録する。

    すでに話し終えた発言への通知は、聞き直しとみなして記録を変えない。
    """
    message = await _character_message(session, message_id)
    return await apply_delivery_state(session, message, DeliveryState(payload.state), now=utcnow())


@router.post("/messages/{message_id}/ideal", response_model=IdealResponseOut)
async def create_ideal_response(
    message_id: int,
    payload: IdealResponseCreate,
    session: AsyncSession = Depends(get_session),
) -> IdealResponse:
    """理想の返答を記録する。並行改善（学習）の教師データと比較評価に使う。"""
    message = await session.get(Message, message_id)
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "発言が見つかりません。")
    if message.speaker_kind != SpeakerKind.CHARACTER.value:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "理想の返答はキャラクターの発言に対して記録します。"
        )
    ideal = IdealResponse(message_id=message_id, ideal_text=payload.ideal_text, note=payload.note)
    session.add(ideal)
    await session.flush()
    return ideal


@router.get("/messages/{message_id}/ideal", response_model=list[IdealResponseOut])
async def list_ideal_responses(
    message_id: int, session: AsyncSession = Depends(get_session)
) -> list[IdealResponse]:
    stmt = (
        select(IdealResponse)
        .where(IdealResponse.message_id == message_id)
        .order_by(IdealResponse.id.desc())
    )
    return list((await session.execute(stmt)).scalars())
