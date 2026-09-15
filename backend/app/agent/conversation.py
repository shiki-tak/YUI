"""会話進行。設計書「6. 会話・自発的行動の流れ」のうち、現在実装している範囲。

手順番号は設計書 §6 の表に対応する（版が付いていない手順は v0.1）。

1. 入力を受け取り、相手・セッション・公開範囲を確定する。会話単位のロックで順序を守る。
2. 直近の会話・関連記憶・採用済みの状態を参照する。
3. 会話状態を更新する：未回答の質問への答えか、延期・終了の合図か、食い違いか（v0.2）。
7. 固定人格・状態・記憶・根拠・会話状態を使ってローカル LLM が返答を生成する。
8. 出力を検査し、字幕・表情・音声へ送る。修正・再生成は回数を制限する。

手順4（感情の更新）は v0.3、手順5（発話候補の発火判定）は v0.5〜v0.6、
手順6（調査等の追加能力の実行）は並行トラック（設計書 §7）でまだ入らない。
手順9（再生完了・中断の記録）は voice 層、手順10（振り返り）は reflection.py
が別途実装している。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent import prompt as prompt_builder
from app.agent.character_state import active_states
from app.agent.conversation_state import (
    CLOSING_FALLBACK_SENTENCE,
    CLOSING_REINFORCEMENT_INSTRUCTION,
    apply_character_message_rules,
    apply_interpretation_result,
    apply_partner_message_rules,
    build_conversation_state_section,
    detect_question,
    discrepancy_offer_ledger,
    gather_interpretation_context,
    get_open_states,
    has_open_state,
    has_unanswered_question_to_yui,
    looks_assertive,
    mentioned_deferral_topics,
    select_discrepancy_offers,
)
from app.agent.interpretation import InterpretationError, interpret_conversation
from app.agent.memory_store import RetrievedMemory, search_memories
from app.config import Settings
from app.llm.base import ChatMessage, LLMClient, LLMError
from app.models import (
    Conversation,
    ConversationStateKind,
    DeliveryState,
    Message,
    RunRecord,
    SourceKind,
    Speaker,
    SpeakerKind,
    utcnow,
)
from app.persona import Persona


@dataclass
class ReplyResult:
    user_message: Message
    reply_message: Message
    run: RunRecord
    memories: list[RetrievedMemory]


class ConversationAgent:
    def __init__(self, *, llm: LLMClient, persona: Persona, settings: Settings) -> None:
        self._llm = llm
        self._persona = persona
        self._settings = settings

    async def _recent_history(
        self, session: AsyncSession, conversation_id: int, *, exclude_id: int
    ) -> list[Message]:
        stmt = (
            select(Message)
            # 誰の発言かをプロンプトに残すため、話者を一緒に読む。遅延読み込みは
            # 非同期セッションでは使えない。
            .options(selectinload(Message.speaker))
            .where(Message.conversation_id == conversation_id, Message.id != exclude_id)
            .order_by(Message.id.desc())
            .limit(self._settings.recent_message_limit)
        )
        return list(reversed(list((await session.execute(stmt)).scalars())))

    async def respond(
        self,
        session: AsyncSession,
        *,
        conversation: Conversation,
        speaker: Speaker,
        text: str,
        source: str = SourceKind.LOCAL_TEXT.value,
        persona: Persona | None = None,
    ) -> ReplyResult:
        # 人格は会話ごとに差し替えられる。3A で同じ入力を別の版へ通し、
        # 着眼点や口調の違いを比べるため。指定が無ければ設定の版を使う。
        persona = persona or self._persona
        user_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.USER.value,
            speaker_id=speaker.id,
            source=source,
            content=text,
            delivery_state=DeliveryState.COMPLETED.value,
        )
        session.add(user_message)
        await session.flush()

        # v0.2：規則（辞書・正規表現）だけで会話状態を作る／応答を記録する。
        # 解釈（LLM）が要る解決・取消は PR3 まで行わない（計画 docs/plan/v0.2.md）。
        await apply_partner_message_rules(
            session,
            conversation_id=conversation.id,
            message=user_message,
            speaker_id=speaker.id,
        )

        history = await self._recent_history(session, conversation.id, exclude_id=user_message.id)
        # 記憶検索の時間は生成と分けて残す。どちらが待ち時間の大半かを
        # 見分けられるようにするため。
        retrieval_started = time.perf_counter()
        memories = await search_memories(
            session,
            query=text,
            speaker_id=speaker.id,
            mode=conversation.mode,
            limit=self._settings.memory_retrieval_limit,
        )
        # 可変状態（関心・関係性）も、記憶と同じ区間で読む。固定人格とは
        # 分けて渡し、どれを使ったかを実行記録に残す。
        states = await active_states(session, speaker_id=speaker.id, mode=conversation.mode)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        conversation_states = await get_open_states(
            session, conversation_id=conversation.id, target_speaker_id=speaker.id
        )

        # 解釈（LLM。設計 §4 手順3・v0.2 PR3）。既定は無効（計画 §9）。
        # 失敗（例外・timeout・スキーマ違反）は規則の結果だけで進む——
        # 解決・取消は起こさない（計画 §8）。
        interpretation_status: dict[str, object] = {
            "enabled": self._settings.conversation_state_llm,
            "attempted": False,
            "applied": False,
            "error": None,
            "summary": None,
        }
        if self._settings.conversation_state_llm:
            pending_ledger = await discrepancy_offer_ledger(
                session, conversation_id=conversation.id, target_speaker_id=speaker.id
            )
            context, candidates = await gather_interpretation_context(
                session,
                conversation_id=conversation.id,
                target_speaker_id=speaker.id,
                current_message=user_message,
                history=history,
                memory_items=[(m.memory.id, m.memory.content) for m in memories],
                open_states=conversation_states,
                discrepancy_ledger=pending_ledger,
                context_messages=self._settings.conversation_state_context_messages,
            )
            interpretation_status["attempted"] = True

            # 解釈の呼び出し中は書き込みトランザクションを開いたままにしない。
            # SQLite は書き込みロックを1つしか持てず、相手の発言・規則由来の
            # 状態を flush しただけの状態で解釈（timeout 既定10秒）を待つと、
            # 別の会話の /chat や記憶の訂正が database is locked で失敗する
            # （既存の「生成前に commit する」と同じ理由。レビューで実測）。
            # 相手の発言由来の状態はここで確定させてよい（計画 §8）。
            await session.commit()
            try:
                result = await asyncio.wait_for(
                    interpret_conversation(self._llm, context=context),
                    timeout=self._settings.conversation_state_llm_timeout_seconds,
                )
            except TimeoutError:
                # str(TimeoutError()) は空文字になる（Python 3.11）。timeout の
                # 秒数を残さないと、失敗の理由が空欄になって分からない
                # （計画 §8「失敗はログに残す」。レビュー指摘）。
                interpretation_status["error"] = (
                    f"timeout ({self._settings.conversation_state_llm_timeout_seconds}s)"
                )
            except (LLMError, InterpretationError) as exc:
                interpretation_status["error"] = str(exc)
            else:
                interpretation_status["summary"] = await apply_interpretation_result(
                    session,
                    result,
                    candidates=candidates,
                    conversation_id=conversation.id,
                    target_speaker_id=speaker.id,
                    speaker_id=speaker.id,
                    current_message=user_message,
                )
                interpretation_status["applied"] = True
                # 解釈の適用は短い書き込みトランザクションで確定させる
                # （モデルを呼んでいる間だけ開けないようにする、が要点。
                # ここは DB だけの操作なので開いていても他会話を止めない）。
                await session.commit()
                # 解決・取消・作成が反映された状態で節を作る。
                conversation_states = await get_open_states(
                    session, conversation_id=conversation.id, target_speaker_id=speaker.id
                )

        # 食い違いの確認候補（照合待ちでなく、渡す回数の上限に達していない
        # もの）を選ぶ。渡した ID は、この返答の RunRecord.options に残し、
        # 次のターンの解釈が「照合待ち」を判定する台帳にする（計画 §4 手順3）。
        discrepancy_offers = select_discrepancy_offers(
            await discrepancy_offer_ledger(
                session, conversation_id=conversation.id, target_speaker_id=speaker.id
            ),
            limit=self._settings.conversation_state_confirm_offer_limit,
            interpretation_ran=bool(interpretation_status["applied"]),
        )
        discrepancy_offer_ids = [state.id for state in discrepancy_offers]

        window_message_ids = {message.id for message in history} | {user_message.id}
        conversation_state_section = build_conversation_state_section(
            conversation_states,
            window_message_ids=window_message_ids,
            discrepancy_offers=discrepancy_offers,
        )

        messages, system_prompt = prompt_builder.build_messages(
            persona=persona,
            memories=memories,
            speaker=speaker,
            history=history,
            user_text=text,
            states=states,
            conversation_state_section=conversation_state_section,
        )

        # 生成に入る前にトランザクションを閉じる。SQLite は書き込みロックを
        # 1 つしか持てないため、応答を待つ間ロックを保持すると、他の会話や
        # 記憶の訂正が「database is locked」で失敗する。
        # 相手の発言はこの時点で確定させ、生成に失敗しても履歴には残す。
        await session.commit()

        # LLMError はここでは握らず、API 層で 503 として返す。
        response = await self._llm.chat(messages)

        # 出力検査（設計 §4 手順6。v0.2 PR2）。conversation_states は commit の
        # 前（生成前）に読んだもので、この相手宛に絞り込み済み。生成前後で
        # 内容は変わらないので、そのまま使う（expire_on_commit=False）。
        checks: dict[str, object] = {
            "closing_open": has_open_state(
                conversation_states, kind=ConversationStateKind.CLOSING.value
            ),
            "closing_new_question_detected": False,
            "regenerated_for_closing": False,
            "regeneration_failed": False,
            "closing_fell_back_to_template": False,
            "closing_unresolved_due_to_open_question": False,
            "deferral_topics_mentioned": [],
            "discrepancy_open": has_open_state(
                conversation_states, kind=ConversationStateKind.DISCREPANCY.value
            ),
            "discrepancy_possibly_assertive": False,
        }

        if checks["closing_open"] and detect_question(response.text):
            checks["closing_new_question_detected"] = True
            # 1回だけ再生成する。強めた指示を追加のシステムメッセージとして渡す
            # （元の system prompt は変えない。人格の弁など既存の規則は残す）。
            reinforced = [
                *messages,
                ChatMessage(role="system", content=CLOSING_REINFORCEMENT_INSTRUCTION),
            ]
            try:
                response = await self._llm.chat(reinforced)
            except LLMError:
                # 1回目の応答はすでに生成できている。品質向上のための
                # 再生成が失敗しただけで相手の発言への応答自体は失われて
                # いないので、503 にはせず1回目の応答をそのまま使う
                # （レビューで指摘。計画 §8 は生成そのものの失敗を想定）。
                checks["regeneration_failed"] = True
            else:
                checks["regenerated_for_closing"] = True
                if detect_question(response.text):
                    open_question_to_yui = has_unanswered_question_to_yui(conversation_states)
                    if open_question_to_yui:
                        # 答えと新しい質問を機械判定で分けられないので、定型へは
                        # 落とさず、人手判定に回す記録だけを残す（設計 §4 手順6）。
                        checks["closing_unresolved_due_to_open_question"] = True
                    else:
                        response.text = CLOSING_FALLBACK_SENTENCE
                        checks["closing_fell_back_to_template"] = True

        # deferral・discrepancy の検査は closing の結果と独立に行う
        # （設計 §4 手順6の3項目は互いに排他ではない）。
        deferral_hits = mentioned_deferral_topics(conversation_states, reply_text=response.text)
        if deferral_hits:
            checks["deferral_topics_mentioned"] = deferral_hits

        if checks["discrepancy_open"] and looks_assertive(response.text):
            checks["discrepancy_possibly_assertive"] = True

        now = utcnow()
        # 読み上げる構成では、生成しただけの状態から始める。実際に鳴ったかは
        # 再生側の通知で進める。読み上げない構成では、画面に出た時点で届く。
        spoken = self._settings.speech_enabled
        reply_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.CHARACTER.value,
            speaker_id=None,
            source=source,
            content=response.text,
            delivery_state=(
                DeliveryState.GENERATED.value if spoken else DeliveryState.COMPLETED.value
            ),
            delivery_started_at=None if spoken else now,
            delivery_finished_at=None if spoken else now,
        )
        session.add(reply_message)
        await session.flush()

        # 返答と同じトランザクションの中で行う（設計 §4 手順7）。相手の
        # 発言由来の状態は手順1・2の commit（上）で確定済みで、ここで
        # 作るのは YUI の返答由来の状態だけ。
        await apply_character_message_rules(
            session,
            conversation_id=conversation.id,
            message=reply_message,
            target_speaker_id=speaker.id,
        )

        # 検査結果は options に checks の鍵で残す（設計 §4 手順6）。生成設定
        # （response.options）を上書きしないよう、コピーへ足す。
        # `offered_discrepancy_ids` は次のターンの解釈が「照合待ち」を
        # 判定する台帳そのもの（新しい表を作らない。計画 §4 手順3）。
        options = dict(response.options)
        options["checks"] = checks
        options["offered_discrepancy_ids"] = discrepancy_offer_ids
        options["interpretation"] = interpretation_status

        run = RunRecord(
            message_id=reply_message.id,
            provider=response.provider,
            model=response.model,
            model_digest=response.model_digest,
            persona_version=persona.version,
            options=options,
            referenced_memory_ids=[m.memory.id for m in memories],
            referenced_state_ids=[state.id for state in states] or None,
            system_prompt=system_prompt,
            retrieval_ms=retrieval_ms,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        session.add(run)
        await session.flush()

        return ReplyResult(
            user_message=user_message,
            reply_message=reply_message,
            run=run,
            memories=memories,
        )
