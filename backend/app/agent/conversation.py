"""会話進行。設計書「6. 会話・自発的行動の流れ」のうち、現在実装している範囲。

1. 発言を受け取り、会話履歴に保存する。
2. 相手・話題に合う長期記憶を検索する。
3. 人格・記憶・直近の会話を LLM へ渡す。
4. 返答を保存し、実行記録（モデル・設定・参照した記憶・応答時間）を残す。

検索・クラウド分析・出力検査はフェーズ5Aで 2 と 3 の間に入る。行動選択
（回答・確認質問・話題提案・調査・待機）はフェーズ4で 2 の後に入る。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent import prompt as prompt_builder
from app.agent.action_selector import ActionChoice, select_action
from app.agent.character_state import active_states
from app.agent.goal import (
    active_goals,
    expire_overdue,
    mark_cancelled,
    mark_done,
    mark_executed,
)
from app.agent.memory_store import RetrievedMemory, search_memories
from app.config import Settings
from app.llm.base import LLMClient
from app.models import (
    Action,
    ActionRecord,
    Conversation,
    DeliveryState,
    Goal,
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
    # 選んだ行動と、その対象の目標。完了条件「行動と状態変化の根拠を追える」。
    choice: ActionChoice | None = None


@dataclass
class OpenResult:
    """YUI の側から会話を始めた結果。

    **話しかけないこともある。** そのときは reply_message が None になる。
    設計書「常に話しかけることを自律性の達成条件にはしません」。
    """

    choice: ActionChoice
    reply_message: Message | None = None
    run: RunRecord | None = None
    goals: list[Goal] = field(default_factory=list)


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
        now: datetime | None = None,
    ) -> ReplyResult:
        # 人格は会話ごとに差し替えられる。3A で同じ入力を別の版へ通し、
        # 着眼点や口調の違いを比べるため。指定が無ければ設定の版を使う。
        persona = persona or self._persona
        # 現在時刻は差し替えられる。評価で期限の到来をまたぐために使う
        # （実際に待つ代わりに、渡す時刻を進める）。記憶検索の減衰と、
        # プロンプトへ書く現在時刻を同じ値で揃える。ばらばらに utcnow() を
        # 呼ぶと、進めた時刻と実時計が混ざる。
        reference_time = now or utcnow()
        user_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.USER.value,
            speaker_id=speaker.id,
            source=source,
            content=text,
            delivery_state=DeliveryState.COMPLETED.value,
            # 発言の時刻は、渡した現在時刻と同じ基準にする。ここだけ実時計だと、
            # 評価で時間を進めたときに、振り返りが「今日」とする日と会話ログの
            # 発言日がずれ、「昨日」が別の日を指す（第1回レビューの指摘3）。
            created_at=reference_time,
        )
        session.add(user_message)
        await session.flush()

        history = await self._recent_history(
            session, conversation.id, exclude_id=user_message.id
        )
        # 記憶検索の時間は生成と分けて残す。どちらが待ち時間の大半かを
        # 見分けられるようにするため。
        retrieval_started = time.perf_counter()
        memories = await search_memories(
            session,
            query=text,
            speaker_id=speaker.id,
            mode=conversation.mode,
            limit=self._settings.memory_retrieval_limit,
            now=reference_time,
        )
        # 可変状態（関心・関係性）も、記憶と同じ区間で読む。固定人格とは
        # 分けて渡し、どれを使ったかを実行記録に残す。
        states = await active_states(session, speaker_id=speaker.id, mode=conversation.mode)
        # 実行してよい目標。期限が来ていないもの、再評価の印が付いたものは
        # ここで落ちる（agent/goal.py）。
        # 実行しないまま時期を過ぎた目標を終わらせる。**行動選択に渡す前に
        # 走らせる。** 起動点が増えても通る位置に置くため、目標を読む直前で行う。
        await expire_overdue(
            session, after_days=self._settings.goal_expiry_days, now=reference_time
        )
        # 判定に渡す目標と、質問してよい目標を分ける。
        #
        # 判定用は、**実行条件で絞らない**。予定の取消は実行してよくなる日より
        # 前にも起こる（第1回レビューの指摘2）。間隔でも絞らない。相手が答えた
        # ばかりの目標を隠すと、いつまでも達成にならない。
        candidates = await active_goals(
            session,
            speaker_id=speaker.id,
            mode=conversation.mode,
            now=reference_time,
            ignore_trigger=True,
        )
        askable = {
            goal.id
            for goal in await active_goals(
                session,
                speaker_id=speaker.id,
                mode=conversation.mode,
                now=reference_time,
                reask_interval_hours=self._settings.goal_reask_interval_hours,
            )
        }
        # この相手に質問した記録がある目標。再生の通知の有無では決めない。
        asked = await self._raised_goal_ids(session, conversation.id, speaker_id=speaker.id)
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        # **モデルを呼ぶ前にトランザクションを閉じる。** SQLite は書き込みロックを
        # 1 つしか持てないため、応答を待つ間ロックを保持すると、他の会話や記憶の
        # 訂正が「database is locked」で失敗する。相手の発言はこの時点で確定させ、
        # 生成に失敗しても履歴には残す。
        #
        # 行動選択を足したときにここを動かし忘れ、1回目のモデル呼び出しが
        # ロックを握ったままになっていた（第1回レビューの指摘1）。**モデルを
        # 呼ぶ処理を足すときは、その前に閉じているかを必ず確かめる。**
        await session.commit()

        # いま目標に触れてよいかを選ぶ。目標が無ければモデルを呼ばない
        # （設計書「常時LLMを呼び続けず」）。
        choice = await select_action(
            llm=self._llm,
            goals=candidates,
            history=history,
            character_name=persona.name,
            user_text=text,
            asked_ids=asked,
            askable_ids=askable,
        )

        # 相手の返答で終わった目標を片付ける。**達成と取消は別の終わり方**で、
        # どちらも「実行した」こととは違う（PR8 から渡した条件1）。
        done = (
            await mark_done(
                session,
                # この会話で実際に持ち出した目標だけを達成にする。聞いていない
                # 目標を「答えた」と判定されても達成にしない。
                goal_ids=[gid for gid in choice.answered_goal_ids if gid in asked],
                reason=f"相手が答えた：{choice.reason or '返答を受けた'}",
                at=reference_time,
            )
            if choice.answered_goal_ids
            else []
        )
        cancelled = (
            await mark_cancelled(
                session,
                goal_ids=choice.cancelled_goal_ids,
                reason=f"前提が無くなった：{choice.reason or '相手が取り消した'}",
            )
            if choice.cancelled_goal_ids
            else []
        )
        # 終わった目標は、この返答では持ち出さない。
        finished = {goal.id for goal in done} | {goal.id for goal in cancelled}
        goals = [goal for goal in candidates if goal.id not in finished]
        choice.goal_ids = [gid for gid in choice.goal_ids if gid not in finished]

        # 聞くと決めた目標が、いま聞いてよいものかを確かめる。
        #
        # - 直前に聞いたばかり（間隔の内側）なら聞かない。
        # - 終わらせた目標を選んでいた場合、**別の目標へ勝手に振り替えない**。
        #   選び直しは行動選択の仕事で、ここで補うと、渡す目標と実行の記録が
        #   食い違う（第1回レビューの指摘3）。
        if choice.executes_a_goal and (
            not choice.goal_ids or not set(choice.goal_ids) <= askable
        ):
            choice.action = Action.ANSWER.value
            choice.goal_ids = []
            choice.reason = "いま聞いてよい目標が残っていないので、聞かない。"

        # 行動の判断を、**生成より前に**残す。目標の状態を変えた判断が、返答の
        # 生成に失敗しただけで記録から消えると、何がその状態にしたのかを追え
        # なくなる（第2回レビューの指摘1）。発言との結び付きは、生成が成功して
        # から入れる。
        action_record: ActionRecord | None = None
        if candidates:
            action_record = ActionRecord(
                conversation_id=conversation.id,
                speaker_id=speaker.id,
                action=choice.action,
                reason=choice.reason,
                # 判定に渡した候補。**終わらせた目標も含めて残す。**
                # 更新後の残りで決めると、最後の1件を完了・取消したときに
                # 判断そのものが記録から消える（第1回レビューの指摘5）。
                candidate_goal_ids=[goal.id for goal in candidates],
                selected_goal_ids=choice.goal_ids or None,
                message_id=None,
                is_proactive=False,
                is_fallback=choice.is_fallback,
                provider=choice.provider,
                model=choice.model,
                model_digest=choice.model_digest,
                options=choice.options,
            )
            session.add(action_record)

        # **書き込みを確定させてから生成へ入る。** 開いたまま応答を待つと、
        # 他の会話や記憶の訂正が「database is locked」で失敗する。1回目の
        # 呼び出しの前では閉じていたのに、2回目の前で同じ形を作っていた
        # （第1回レビューの指摘1）。
        await session.commit()

        # 聞くと決めたときは、選んだ目標だけを渡す。全部渡して生成側に選び
        # 直させると、相手の様子を見て選んだ判断が伝わらない
        # （第1回レビューの指摘3）。触れないときは、いま聞いてよい目標を
        # 持っているものとして渡す（渡さないことで待機させない）。
        if choice.executes_a_goal:
            passed_goals = [goal for goal in goals if goal.id in choice.goal_ids]
        else:
            passed_goals = [goal for goal in goals if goal.id in askable]

        messages, system_prompt = prompt_builder.build_messages(
            persona=persona,
            memories=memories,
            speaker=speaker,
            history=history,
            user_text=text,
            states=states,
            goals=passed_goals,
            action=choice.action,
            now=reference_time,
        )

        # LLMError はここでは握らず、API 層で 503 として返す。
        response = await self._llm.chat(messages)

        recorded_at = utcnow()
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
            # 再生の記録は実時計のまま。いつ鳴らしたかは監査の記録で、
            # 会話の中の時間の進み方とは別のもの。
            delivery_started_at=None if spoken else recorded_at,
            delivery_finished_at=None if spoken else recorded_at,
            created_at=reference_time,
        )
        session.add(reply_message)
        await session.flush()

        run = RunRecord(
            message_id=reply_message.id,
            provider=response.provider,
            model=response.model,
            model_digest=response.model_digest,
            persona_version=persona.version,
            options=response.options,
            referenced_memory_ids=[m.memory.id for m in memories],
            referenced_state_ids=[state.id for state in states] or None,
            # 渡した目標と、選んだ行動。待機を選んだことも残す。合格は
            # 「よく喋ること」ではないため（完了条件5）。
            referenced_goal_ids=[goal.id for goal in passed_goals] or None,
            selected_action=choice.action,
            system_prompt=system_prompt,
            retrieval_ms=retrieval_ms,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        session.add(run)
        # 読み上げない構成では、画面に出た時点で相手に届く。再生の通知は
        # 来ないので、ここで実行済みにする。通知を待つと、文字だけの構成で
        # 目標が永久に実行済みにならない（第1回レビューの指摘3）。
        if not spoken and choice.executes_a_goal:
            await mark_executed(
                session,
                goal_ids=[goal.id for goal in passed_goals],
                delivered=DeliveryState.COMPLETED.value,
                at=recorded_at,
            )
        # 行動の判断を残す。目標が無いときは判断していないので作らない
        # （すべての返答に1件ずつ増やしても、読む材料にならない）。
        if action_record is not None:
            # 話せたので、発言と結び付ける。**発言のない判断は「持ち出した」
            # ことにしない**（下の _raised_goal_ids と、実行済みの判定が
            # message_id で見る）。
            action_record.message_id = reply_message.id
        await session.flush()

        return ReplyResult(
            user_message=user_message,
            reply_message=reply_message,
            run=run,
            memories=memories,
            choice=choice,
        )

    async def _raised_goal_ids(
        self,
        session: AsyncSession,
        conversation_id: int,
        *,
        speaker_id: int | None = None,
    ) -> set[int]:
        """実際に持ち出した目標。

        行動の記録から引く。渡しただけの目標（待機のときも渡る）は含めない。
        **再生の通知の有無では決めない。** 通知は欠けることがあり、話した記録が
        あれば聞いてはいる（ISSUE-013）。

        speaker_id を渡すと、その相手に対して持ち出したものを会話をまたいで
        拾う。**達成の判定も相手単位で行う。** 「昨日聞かれたやつだけど」と
        後から答えることがあり、会話に閉じると達成にできない。

        **発言が残っている記録だけを見る。** 判断は生成の前に保存するため、
        生成に失敗した判断も記録に残る。それを「持ち出した」と数えると、
        一度も話していない質問を聞いたことにする（第2回レビューの指摘1）。

        **再生の状態では絞らない。** 一度「鳴り始める前に止めた発言（aborted で
        開始時刻が無いもの）」を除いたが、戻した（PR10 レビューの指摘4）。
        開始時刻が無いことは未到達の証拠にならない。

        - `playing` の通知だけが落ちて `aborted` が届くと、鳴っていても開始時刻
          は NULL になる（[ISSUE-013](../../../docs/issues/issues.md)）。
        - ローカルの会話では文字が画面に出ている。音が鳴らなくても相手は読める。
          実際、除いた状態では「質問は画面で読んだよ。映画は面白かった」と
          具体的に答えても達成にできなかった。

        観測された誤りは、相手の「ごめん、聞こえなかった」を行動選択が
        `answered` と判定したことである。**それは判定の誤りで、再生の状態から
        は区別できない。** ISSUE-034 として分けて記録した。
        """
        stmt = select(ActionRecord).where(
            ActionRecord.action.in_([Action.ASK.value, Action.SUGGEST.value]),
            ActionRecord.message_id.is_not(None),
        )
        if speaker_id is not None:
            stmt = stmt.where(ActionRecord.speaker_id == speaker_id)
        else:
            stmt = stmt.where(ActionRecord.conversation_id == conversation_id)
        raised: set[int] = set()
        for record in (await session.execute(stmt)).scalars():
            raised.update(record.selected_goal_ids or [])
        return raised

    async def open(
        self,
        session: AsyncSession,
        *,
        conversation: Conversation,
        speaker: Speaker,
        source: str = SourceKind.LOCAL_TEXT.value,
        persona: Persona | None = None,
        now: datetime | None = None,
    ) -> OpenResult:
        """YUI の側から会話を始める（設計書 6「自発的行動」）。

        起動点は会話開始だけに絞っている。話題の区切りと、許可された待機時間は
        ISSUE-024 に記録した。

        **話しかけないこともある。** 目標が無い、いま持ち出す場面ではない、と
        判断したら発言を作らない。「常に話しかけることを自律性の達成条件には
        しません」（設計書）。
        """
        persona = persona or self._persona
        reference_time = now or utcnow()

        history = await self._recent_history(session, conversation.id, exclude_id=0)
        await expire_overdue(
            session, after_days=self._settings.goal_expiry_days, now=reference_time
        )
        # 自分から始める場面では、聞いてよい目標だけを見る。相手の発言が無いので、
        # 答えた・取り消したの判定はここでは行わない。
        goals = await active_goals(
            session,
            speaker_id=speaker.id,
            mode=conversation.mode,
            now=reference_time,
            reask_interval_hours=self._settings.goal_reask_interval_hours,
        )
        # モデルを呼ぶ前に、開いている書き込みを閉じる。呼び出し側が話者を
        # 作った直後に来ることがあり、そのまま待つとロックを握り続ける
        # （respond と同じ扱い）。
        await session.commit()
        choice = await select_action(
            llm=self._llm,
            goals=goals,
            history=history,
            character_name=persona.name,
            user_text=None,
        )
        if not choice.executes_a_goal:
            # 待機と調査は、発言を作らない。調査はフェーズ5A が未実装で、
            # 「調べた」と言わせないため（設計書 6）。
            #
            # **話しかけなかったことも記録する。** 発言が無いので run_records は
            # 作れない。設計書は「適切な待機も確認する」としており、残さないと
            # 適切に待てたのかを測れない（第1回レビューの指摘5）。
            session.add(
                ActionRecord(
                    conversation_id=conversation.id,
                    speaker_id=speaker.id,
                    action=choice.action,
                    reason=choice.reason,
                    candidate_goal_ids=[goal.id for goal in goals] or None,
                    selected_goal_ids=choice.goal_ids or None,
                    message_id=None,
                    is_proactive=True,
                    is_fallback=choice.is_fallback,
                    provider=choice.provider,
                    model=choice.model,
                    model_digest=choice.model_digest,
                    options=choice.options,
                )
            )
            await session.flush()
            return OpenResult(choice=choice, goals=goals)

        # 話しかけると決めた目標だけを渡す。持っている目標を全部渡すと、
        # 1回の発話に複数の用件が混ざる。
        chosen = [goal for goal in goals if goal.id in choice.goal_ids] or goals[:1]
        # 目標の本文で記憶を引く。渡さないと「この相手について思い出せることは
        # ありません」と書かれた直後に「映画の感想を聞け」と指示することになり、
        # 矛盾した入力になる（PR7 の実測で、あいさつだけで終わっていた）。
        states = await active_states(session, speaker_id=speaker.id, mode=conversation.mode)
        memories = await search_memories(
            session,
            query=chosen[0].content,
            speaker_id=speaker.id,
            mode=conversation.mode,
            limit=self._settings.memory_retrieval_limit,
            now=reference_time,
        )
        messages, system_prompt = prompt_builder.build_messages(
            persona=persona,
            memories=memories,
            speaker=speaker,
            history=history,
            # 相手の発言はまだない。自分から始める場面であることを伝える。
            user_text="（相手はまだ何も言っていません。あなたから話しかけてください）",
            states=states,
            goals=chosen,
            action=choice.action,
            opening=True,
            now=reference_time,
        )
        await session.commit()

        response = await self._llm.chat(messages)

        spoken = self._settings.speech_enabled
        recorded_at = utcnow()
        reply_message = Message(
            conversation_id=conversation.id,
            speaker_kind=SpeakerKind.CHARACTER.value,
            speaker_id=None,
            source=source,
            content=response.text,
            delivery_state=(
                DeliveryState.GENERATED.value if spoken else DeliveryState.COMPLETED.value
            ),
            delivery_started_at=None if spoken else recorded_at,
            delivery_finished_at=None if spoken else recorded_at,
            created_at=reference_time,
        )
        session.add(reply_message)
        await session.flush()

        run = RunRecord(
            message_id=reply_message.id,
            provider=response.provider,
            model=response.model,
            model_digest=response.model_digest,
            persona_version=persona.version,
            options=response.options,
            referenced_memory_ids=[item.memory.id for item in memories],
            referenced_state_ids=[state.id for state in states] or None,
            referenced_goal_ids=[goal.id for goal in chosen] or None,
            selected_action=choice.action,
            system_prompt=system_prompt,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        session.add(run)
        if not spoken:
            await mark_executed(
                session,
                goal_ids=[goal.id for goal in chosen],
                delivered=DeliveryState.COMPLETED.value,
                at=recorded_at,
            )
        session.add(
            ActionRecord(
                conversation_id=conversation.id,
                speaker_id=speaker.id,
                action=choice.action,
                reason=choice.reason,
                candidate_goal_ids=[goal.id for goal in goals],
                selected_goal_ids=[goal.id for goal in chosen],
                message_id=reply_message.id,
                is_proactive=True,
                is_fallback=choice.is_fallback,
                provider=choice.provider,
                model=choice.model,
                model_digest=choice.model_digest,
                options=choice.options,
            )
        )
        await session.flush()

        return OpenResult(
            choice=choice, reply_message=reply_message, run=run, goals=chosen
        )
