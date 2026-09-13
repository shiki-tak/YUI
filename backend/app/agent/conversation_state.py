"""v0.2 の会話状態：現在の用件・未回答の質問・提示済みの内容・延期・終了・
食い違いを扱う（設計書「6. 会話・自発的行動の流れ」、計画 docs/plan/v0.2.md）。

**発言があったことと解決したことを分ける。** 相手の次の発言が来ただけでは
質問も食い違いも解決にならない。解決・取消は解釈（LLM）か開発者の操作でだけ
起き、規則は「作る」と「応答があった」しか記録しない。

PR1 は規則（辞書・正規表現）だけを使う。PR3 で解釈（LLM）の呼び出し
（`interpretation.py`）と、その結果の検証・適用（`apply_interpretation_result`）
をここに足した。`request` / `confirmed` / `correction` / `discrepancy` の
作成と、すべての解決・取消は解釈か開発者の操作でしか起きない
（計画 docs/plan/v0.2.md 3節）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.interpretation import InterpretationResult, RefPayload
from app.models import (
    ConversationState,
    ConversationStateKind,
    ConversationStateRefKind,
    ConversationStateStatus,
    ConversationStateWithdrawReason,
    DecisionSource,
    DetectionSource,
    Message,
    RunRecord,
    SpeakerKind,
    utcnow,
)

# 本文の保存上限。`content` は開発者向けの本文だが（計画 §3）、質問の全文が
# system prompt の「この会話で」節にそのまま入るため、`presented` と同じ上限で
# 切る（レビューで、4000文字の発言1件だけで節が4000文字を超えることを実測）。
CONTENT_LIMIT = 200

# --- 規則：文末の形だけを見る。曖昧な内容判定はしない -----------------------

# 質問の文末（相手の発言・YUI の発言の両方に使う）。「でしょうか」は疑問だが
# 「でしょう」単独は推量の平叙文（「明日は雨でしょう」）でも使うため含めない。
# 「かな」も「たぶん来るかな」のような独り言に出るため含めない
# （レビューで誤検出を実測。計画 §3 の語彙も ？／?／〜ますか／〜でしょうか に限る）。
# 句点で終わる丁寧体（「〜ますか。」）も拾えるよう、閉じ括弧の後に句読点を許す
# ——依頼の文末と同じ形にする（レビューで非対称を実測）。
_QUESTION_ENDING_RE = re.compile(
    r"(？|\?|ですか|ますか|でしょうか|かしら)[」』】\)]*[。.！!]*\s*$"
)

# 依頼の文末（相手の発言からの question_to_yui にだけ使う。設計 §3）。
# 「〜してほしい」に限らず、「手伝ってほしい」のような動詞＋補助動詞の形も
# 拾えるよう、直前の活用形は問わずに末尾の語だけで判定する。
_REQUEST_ENDING_RE = re.compile(
    r"(教えて|教えてください|ほしい|欲しい|てくれる|てくれない|"
    r"てもらえる|お願い)[」』】\)]*[。.！!]*\s*$"
)

# 延期：話題つきの文末に限る（誤検出を避けるため広い一致はしない）。
# 「の話／について」は「は」の直前に付く修飾としてだけ許す
# （「については」は「は」を含むので、その後にもう一つ「は」を要求すると
# 一致しない。レビューで実測した死んだ分岐を直した）。
# 「今度は映画の話をしよう」のような未来の提案は対象外にする（「また今度」を
# 含まず、文末が延期の定型句でないため一致しない）。
# **判定対象は最後の文（。！!で区切った末尾）だけ**にする。前の文の内容を
# topic が飲み込むのを防ぐ（レビューで「疲れた。映画の話は…」が
# 「疲れた。映画」になる例を実測）。
# topic は「は」自体を含めない。1文に「AはBの話はまた今度」のように「は」が
# 2つあるとき、topic に含めることを許すと最初の「は」の手前までを topic に
# してしまい（「今日は」を飲み込んで「今日は映画」になる）、その組み合わせで
# 一致してしまう。「は」を境界の外に置くと、その位置からは先に進めなくなり、
# `re.search` が次の「は」（＝実際に延期の対象を指す方）から始まる一致を
# 拾い直す。トピックの先頭が仮名の「は」で始まる語（稀）は切り詰められる
# トレードオフを受け入れる（レビューで実測した「今日は映画の話は…」の誤り）。
_DEFERRAL_RE = re.compile(
    r"(?P<topic>[^。！!、,　\nは]{1,20}?)(の話|について)?は\s*"
    r"(また今度|また後で|後で|あとで)"
    r"(ね|にする|にします|にしよう|にしようね|にしますね|話そう|話します)?"
    r"[。.！!]*\s*$"
)

# 終了：文末の定型句に限る。「ありがとう」等の単独の発言では検出しない。
# **最後の文全体**に対して fullmatch する（`^`〜`$`）。「そろそろ」等の前置きは
# 任意だが、それ以外の内容が混ざった文（「明日は実家に帰ります」「12時に寝る」）
# は終了の合図として扱わない（レビューで誤検出を実測）。
_CLOSING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(そろそろ)?(寝る|寝ます|おやすみ)(ね|なさい)?[。.！!]*$"),
    re.compile(
        r"(今日はここまで|今日はこの辺で|今日はこれで)"
        r"(にしよう|にします|にしましょう)?[。.！!]*$"
    ),
    re.compile(r"(そろそろ)?(落ちる|落ちます|帰る|帰ります)(ね)?[。.！!]*$"),
    re.compile(r"また(今度|ね|明日)[。.！!]*$"),
]

# 「寝る前に明日の集合時間だけ教えて」のように、終了語が文末以外の位置に
# 「〜前に」の形で付く場合も終了の意図として拾う（計画 §7 シナリオ4：
# 終了の合図と最後の依頼が同時に来る形）。`_CLOSING_PATTERNS` は最後の文
# 全体の一致を求めるため、依頼が続く文はそこでは拾えない。
#
# **「前に」の直後に「だけ／一つ／ひとつ／最後に」のいずれかを求める。**
# 「最後の文が依頼の形であること」まで絞っても、「寝る前にストレッチする
# といいって本当？」「帰る前にやることリストを教えてくれる？」のような、
# 終了の意図が無い一般的な質問・依頼が拾われてしまう（レビューで実測）。
# シナリオ4の言い方（「〜前に◯◯だけ」）は「終わる前にこれだけ済ませたい」
# という限定を伴うので、その語を要求すると誤検出を防ぎつつ実際の言い方は
# 拾える。
_CLOSING_BEFORE_RE = re.compile(
    r"(そろそろ)?(寝る|寝ます|おやすみ|落ちる|落ちます|帰る|帰ります)前に"
    r".{0,20}?(だけ|ひとつ|一つ|最後に)"
)

# 文の区切り。最後の文だけを終了・延期の判定対象にする。改行・疑問符も
# 区切りに含める——画面の入力欄は textarea で Enter は改行のため、通常操作で
# 複数行の発言が届く（レビューで「今日は楽しかった\nそろそろ寝るね」の
# 取りこぼしを実測。区切らないと前の行を延期の話題が飲み込み、終了の合図は
# 逆に検出されなくなる）。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！!？?\n])")


def _last_sentence(text: str) -> str:
    """判定対象を最後の文に絞る（前の文の内容を巻き込まないため）。"""
    parts = [part for part in _SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    return parts[-1].strip() if parts else text.strip()

# 相づち（confirmed の作成を抑止するのに使う。PR3 が参照する）。
ACKNOWLEDGEMENT_MAX_LENGTH = 6
_ACKNOWLEDGEMENT_WORDS = {
    "うん", "はい", "ええ", "そう", "そうだね", "そうですね", "へえ", "なるほど",
}

# 「そうですか」「そうなんですか」は相づちで、質問ではない。前に「なるほど、」
# 「へえ、」等が付く形もあるため、is_acknowledgement_only の完全一致（短い
# 発言限定）では弾けない。文末だけを見て、質問扱いから明示的に除外する
# （レビューで「なるほど、そうですか！」等の誤検出を実測）。
_ACKNOWLEDGEMENT_QUESTION_RE = re.compile(r"そう(なん)?ですか[。.！!]*\s*$")


def is_acknowledgement_only(text: str) -> bool:
    """短い相づちだけの発言か（confirmed を作らない条件。設計 §3）。"""
    stripped = text.strip("。.！!　 ")
    return len(stripped) <= ACKNOWLEDGEMENT_MAX_LENGTH and stripped in _ACKNOWLEDGEMENT_WORDS


def detect_question(text: str) -> bool:
    """文末が質問の形か。「そう(なん)ですか」は相づちなので質問に数えない。"""
    stripped = text.strip()
    if _ACKNOWLEDGEMENT_QUESTION_RE.search(stripped):
        return False
    return bool(_QUESTION_ENDING_RE.search(stripped))


def detect_request_question(text: str) -> bool:
    """相手の発言から question_to_yui を作るときだけ使う（質問＋依頼の形）。"""
    stripped = text.strip()
    return detect_question(stripped) or bool(_REQUEST_ENDING_RE.search(stripped))


def detect_deferral_topic(text: str) -> str | None:
    """延期の文末があれば、話題らしき語を返す。無ければ None。

    判定は最後の文だけを見る（前の文を topic が飲み込まないため）。
    """
    match = _DEFERRAL_RE.search(_last_sentence(text))
    if not match:
        return None
    topic = match.group("topic").strip("、, 　")
    return topic or None


def detect_closing(text: str) -> bool:
    """終了の合図があるか。

    最後の文**全体**が定型句であることを求める（`fullmatch`）。「そろそろ」等の
    前置きは任意だが、それ以外の内容が混ざった平叙文（「明日は実家に帰ります」）
    を終了の合図として拾わない。加えて、「寝る前に」のような「〜前に」の形は、
    **最後の文が依頼の形であるときに限って**拾う（依頼と同時に終了の意図が出て
    くる形。計画 §7 シナリオ4）。依頼を求めないと、「毎晩寝る前にストレッチ
    してる」のような習慣・伝聞の報告まで終了の合図になる（レビューで実測）。
    """
    last = _last_sentence(text)
    if any(pattern.fullmatch(last) for pattern in _CLOSING_PATTERNS):
        return True
    return bool(_CLOSING_BEFORE_RE.search(last)) and detect_request_question(last)


# --- 低レベルの作成・遷移 ----------------------------------------------------


async def create_state(
    session: AsyncSession,
    *,
    conversation_id: int,
    kind: str,
    content: str,
    source_message_id: int,
    speaker_id: int | None = None,
    target_speaker_id: int | None = None,
    ref_kind: str | None = None,
    ref_id: int | None = None,
    detected_by: str = DetectionSource.RULE.value,
) -> ConversationState:
    """状態を1件作る。重複の制約はここでは見ない（呼び出し側か can_create_* で見る）。"""
    state = ConversationState(
        conversation_id=conversation_id,
        kind=kind,
        content=content,
        speaker_id=speaker_id,
        target_speaker_id=target_speaker_id,
        source_message_id=source_message_id,
        ref_kind=ref_kind,
        ref_id=ref_id,
        status=ConversationStateStatus.OPEN.value,
        detected_by=detected_by,
    )
    session.add(state)
    await session.flush()
    return state


async def get_open_states(
    session: AsyncSession,
    *,
    conversation_id: int,
    target_speaker_id: int | None = None,
    kind: str | None = None,
) -> list[ConversationState]:
    """開いている会話状態。`target_speaker_id` を指定すると、その相手に適用する
    ものだけを返す（別の相手の状態を混ぜない）。
    """
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.status == ConversationStateStatus.OPEN.value,
    )
    if kind is not None:
        stmt = stmt.where(ConversationState.kind == kind)
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    return list((await session.execute(stmt.order_by(ConversationState.id))).scalars())


async def get_open_state_for_ref(
    session: AsyncSession,
    *,
    conversation_id: int,
    kind: str,
    ref_kind: str,
    ref_id: int,
    target_speaker_id: int | None,
) -> ConversationState | None:
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.kind == kind,
        ConversationState.ref_kind == ref_kind,
        ConversationState.ref_id == ref_id,
        ConversationState.status == ConversationStateStatus.OPEN.value,
    )
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    return (await session.execute(stmt)).scalars().first()


def mark_first_response(state: ConversationState, *, message_id: int) -> None:
    """最初に応答があったことだけを記録する。解決にはしない（PR1 は規則のみ）。"""
    if state.responded_message_id is None:
        state.responded_message_id = message_id


async def resolve_state(
    session: AsyncSession,
    state: ConversationState,
    *,
    resolved_message_id: int | None,
    decided_by: str,
) -> None:
    state.status = ConversationStateStatus.RESOLVED.value
    state.resolved_message_id = resolved_message_id
    state.decided_by = decided_by
    state.followup_needed = False
    await session.flush()


async def withdraw_state(
    session: AsyncSession,
    state: ConversationState,
    *,
    reason: str,
    decided_by: str,
    resolved_message_id: int | None = None,
) -> None:
    state.status = ConversationStateStatus.WITHDRAWN.value
    state.withdraw_reason = reason
    state.resolved_message_id = resolved_message_id
    state.decided_by = decided_by
    await session.flush()


async def mark_followup_needed(
    session: AsyncSession,
    state: ConversationState,
    *,
    judged_message_id: int,
    decided_by: str,
) -> None:
    """「応答したが答えていない」の判定（PR3）。答え損ねであって、解決ではない。"""
    state.followup_needed = True
    state.judged_message_id = judged_message_id
    state.decided_by = decided_by
    await session.flush()


async def expire_open_states(
    session: AsyncSession, *, conversation_id: int, at: datetime | None = None
) -> list[ConversationState]:
    """会話の終了時、開いているものをすべて `expired` にする（計画 5節）。

    `resolved` にはしない。会話が終わったことと問題が解決したことは別。
    呼び出し（`/end` への配線）は PR4 で行う。
    """
    states = await get_open_states(session, conversation_id=conversation_id)
    for state in states:
        state.status = ConversationStateStatus.EXPIRED.value
        state.updated_at = at or utcnow()
    if states:
        await session.flush()
    return states


# --- discrepancy / correction の重複制約（PR3 が使う。ここに置いて先に固める） ---


async def can_create_discrepancy(
    session: AsyncSession,
    *,
    conversation_id: int,
    target_speaker_id: int | None,
    ref_kind: str,
    ref_id: int,
) -> bool:
    """同じ会話・同じ適用相手・同じ対象について、新しい discrepancy を
    作ってよいか（計画 3節の `can_create_discrepancy`）。

    まず誤検出（`misdetected`）で取り消した行を除外し、残りに
    `open` の行、`asked` が入った行（状態を問わない）、有効な `correction`
    のどれかがあれば作らない。
    """
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.kind == ConversationStateKind.DISCREPANCY.value,
        ConversationState.ref_kind == ref_kind,
        ConversationState.ref_id == ref_id,
    )
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    rows = list((await session.execute(stmt)).scalars())
    remaining = [
        row
        for row in rows
        if not (
            row.status == ConversationStateStatus.WITHDRAWN.value
            and row.withdraw_reason == ConversationStateWithdrawReason.MISDETECTED.value
        )
    ]
    if any(row.status == ConversationStateStatus.OPEN.value for row in remaining):
        return False
    if any(row.asked_message_id is not None for row in remaining):
        return False

    correction_stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.kind == ConversationStateKind.CORRECTION.value,
        ConversationState.ref_kind == ref_kind,
        ConversationState.ref_id == ref_id,
        ConversationState.status == ConversationStateStatus.OPEN.value,
    )
    if target_speaker_id is not None:
        correction_stmt = correction_stmt.where(
            ConversationState.target_speaker_id == target_speaker_id
        )
    has_correction = (await session.execute(correction_stmt)).scalars().first()
    return has_correction is None


async def create_or_supersede_correction(
    session: AsyncSession,
    *,
    conversation_id: int,
    target_speaker_id: int | None,
    ref_kind: str,
    ref_id: int,
    content: str,
    source_message_id: int,
    speaker_id: int | None,
    decided_by: str,
) -> ConversationState:
    """訂正を採用する。同じ対象に有効な訂正があれば `superseded` にしてから
    新しい行を作る（計画 3節）。
    """
    existing = await get_open_state_for_ref(
        session,
        conversation_id=conversation_id,
        kind=ConversationStateKind.CORRECTION.value,
        ref_kind=ref_kind,
        ref_id=ref_id,
        target_speaker_id=target_speaker_id,
    )
    if existing is not None:
        await withdraw_state(
            session,
            existing,
            reason=ConversationStateWithdrawReason.SUPERSEDED.value,
            decided_by=decided_by,
            resolved_message_id=source_message_id,
        )
    return await create_state(
        session,
        conversation_id=conversation_id,
        kind=ConversationStateKind.CORRECTION.value,
        content=content,
        source_message_id=source_message_id,
        speaker_id=speaker_id,
        target_speaker_id=target_speaker_id,
        ref_kind=ref_kind,
        ref_id=ref_id,
        detected_by=DetectionSource.LLM.value,
    )


async def get_open_request(
    session: AsyncSession, *, conversation_id: int, target_speaker_id: int | None
) -> ConversationState | None:
    """開いている `request`（今の用件）。`ref_kind`／`ref_id` を持たないため、
    `get_open_state_for_ref` ではなく専用に引く（計画 3節）。
    """
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.kind == ConversationStateKind.REQUEST.value,
        ConversationState.status == ConversationStateStatus.OPEN.value,
    )
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    return (await session.execute(stmt)).scalars().first()


async def create_or_supersede_request(
    session: AsyncSession,
    *,
    conversation_id: int,
    target_speaker_id: int | None,
    content: str,
    source_message_id: int,
    speaker_id: int | None,
    decided_by: str,
) -> ConversationState:
    """今の用件を採用する。用件が変わっていれば古い行を `superseded` にする
    （計画 3節：「用件が変わった（新しい request を作り、古いものを
    superseded）」）。**同じ内容なら作り直さない**——解釈のたびに同じ用件が
    返ると行が積み上がるため（設計に明記は無いが、`presented` と同じ理由で
    無駄な行を増やさない判断）。
    """
    existing = await get_open_request(
        session, conversation_id=conversation_id, target_speaker_id=target_speaker_id
    )
    truncated = content[:CONTENT_LIMIT]
    if existing is not None:
        if existing.content == truncated:
            return existing
        await withdraw_state(
            session,
            existing,
            reason=ConversationStateWithdrawReason.SUPERSEDED.value,
            decided_by=decided_by,
            resolved_message_id=source_message_id,
        )
    return await create_state(
        session,
        conversation_id=conversation_id,
        kind=ConversationStateKind.REQUEST.value,
        content=truncated,
        source_message_id=source_message_id,
        speaker_id=speaker_id,
        target_speaker_id=target_speaker_id,
        detected_by=DetectionSource.LLM.value,
    )


async def _has_confirmed(
    session: AsyncSession, *, conversation_id: int, target_speaker_id: int | None, presented_id: int
) -> bool:
    """同じ相手・同じ `presented` への `confirmed` が既にあるか（重複制約。
    計画 3節）。`confirmed` は解決・取消が無いため、状態を問わず全件を見る。
    """
    stmt = select(ConversationState).where(
        ConversationState.conversation_id == conversation_id,
        ConversationState.kind == ConversationStateKind.CONFIRMED.value,
        ConversationState.ref_kind == ConversationStateRefKind.STATE.value,
        ConversationState.ref_id == presented_id,
    )
    if target_speaker_id is not None:
        stmt = stmt.where(ConversationState.target_speaker_id == target_speaker_id)
    return (await session.execute(stmt)).scalars().first() is not None


# --- respond() から呼ぶ、規則だけの適用 --------------------------------------


async def apply_partner_message_rules(
    session: AsyncSession,
    *,
    conversation_id: int,
    message: Message,
    speaker_id: int,
) -> None:
    """相手の発言を保存した直後に呼ぶ（設計 §4 手順1・2）。

    - 質問・依頼の文末があれば `question_to_yui` を作る。
    - 延期・終了の文末があれば `deferral` / `closing` を作る。
    - 開いている `question_to_partner`（この相手宛）に、最初の応答なら記録する。

    `discrepancy` の応答はここでは扱わない——`asked` が入っているかどうかは
    PR3 の解釈が決めるため、規則だけでは判定できない（計画 3節・4節）。
    """
    text = message.content

    if detect_request_question(text):
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_YUI.value,
            content=text[:CONTENT_LIMIT],
            source_message_id=message.id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
        )

    topic = detect_deferral_topic(text)
    if topic is not None:
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DEFERRAL.value,
            content=topic,
            source_message_id=message.id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
        )

    if detect_closing(text):
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.CLOSING.value,
            content=text[:CONTENT_LIMIT],
            source_message_id=message.id,
            speaker_id=speaker_id,
            target_speaker_id=speaker_id,
        )

    open_questions = await get_open_states(
        session,
        conversation_id=conversation_id,
        target_speaker_id=speaker_id,
        kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
    )
    for question in open_questions:
        mark_first_response(question, message_id=message.id)
    if open_questions:
        await session.flush()


async def apply_character_message_rules(
    session: AsyncSession,
    *,
    conversation_id: int,
    message: Message,
    target_speaker_id: int,
) -> None:
    """YUI の返答を保存した直後に呼ぶ（設計 §4 手順7）。返答と同じ
    トランザクションの中で行う。

    - 返答の文末が質問なら `question_to_partner` を作る。
    - 返答ごとに `presented` を1件作る。
    - 開いている `question_to_yui`（この相手宛）に、最初の応答なら記録する。

    `discrepancy` の `asked` はここでは入れない——返答に質問があっても、
    その食い違いを確認したとは限らない（別の話題の質問かもしれない）。
    次のターンの解釈が照合して決める（計画 3節・4節）。
    """
    text = message.content

    if detect_question(text):
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.QUESTION_TO_PARTNER.value,
            content=text[:CONTENT_LIMIT],
            source_message_id=message.id,
            speaker_id=None,
            target_speaker_id=target_speaker_id,
        )

    await create_state(
        session,
        conversation_id=conversation_id,
        kind=ConversationStateKind.PRESENTED.value,
        content=text[:CONTENT_LIMIT],
        source_message_id=message.id,
        speaker_id=None,
        target_speaker_id=target_speaker_id,
    )

    open_questions = await get_open_states(
        session,
        conversation_id=conversation_id,
        target_speaker_id=target_speaker_id,
        kind=ConversationStateKind.QUESTION_TO_YUI.value,
    )
    for question in open_questions:
        mark_first_response(question, message_id=message.id)
    if open_questions:
        await session.flush()


# --- 出力検査（設計 §4 手順6。v0.2 PR2） -------------------------------------

# 再生成でも通らなかった場合に使う、人格判定済みの固定文（計画 §9：1文固定、
# 設定にしない）。おしとやかな口調（`personas/*.toml` の speech）に合わせる。
CLOSING_FALLBACK_SENTENCE = (
    "はい、今日はここまでにいたしましょう。またお話しできるのを楽しみにしていますね。"
)

# closing が開いているのに YUI から新しい質問が出たときの、再生成用の追加指示。
CLOSING_REINFORCEMENT_INSTRUCTION = (
    "相手は会話を終えようとしています。新しい質問や話題を自分から出さず、"
    "相手の依頼にはきちんと答えたうえで、短い言葉で締めくくってください。"
)

# discrepancy が開いているときに、返答が断定に見えるかを見る弱いヒューリスティック。
# 行動は変えず記録するだけなので、精度は求めない（設計 §4 手順6「記録のみ」）。
_HEDGE_MARKERS = ("かもしれ", "でしょうか", "たぶん", "確認", "念のため", "合ってい", "でしたっけ")


def has_open_state(states: list[ConversationState], *, kind: str) -> bool:
    """渡された状態一覧（既に target で絞り込み済みの前提）に、その種類が
    開いているものがあるか。
    """
    return any(state.kind == kind for state in states)


def has_unanswered_question_to_yui(states: list[ConversationState]) -> bool:
    """まだ答えるべき相手の質問があるか（`open` かつ、一度も応答していない
    か、答え損ねたと判定済み（`followup_needed`）のもの）。

    PR1・PR2 は解決を作らないため、一度答えた question_to_yui も `status` は
    `open` のまま残る（PR3 の解釈まで）。`has_open_state(kind=QUESTION_TO_YUI)`
    のように `status` だけを見ると、会話のどこかで一度でも質問された時点で
    ずっと真になり、closing の定型への差し替えが実運用でほぼ働かなくなる
    （レビューで実測）。

    **`followup_needed` も「答えるべき質問」に含める。** PR3 で解釈が
    「答え損ねた」と判定した質問（`responded_message_id` は入っている）を
    含めないと、closing の定型文がその答えを潰してしまう
    （計画 §3「答え損ねた質問（次で答える）」。レビュー指摘）。

    `status` も明示的に見る。呼び出し元（`conversation.py`）は `get_open_states`
    で絞り込み済みの一覧しか渡さないため実害は無いが、`withdrawn`／`expired`
    を含む一覧を渡す呼び出し元が増えても壊れないようにする（レビュー指摘）。
    """
    return any(
        state.kind == ConversationStateKind.QUESTION_TO_YUI.value
        and state.status == ConversationStateStatus.OPEN.value
        and (state.responded_message_id is None or state.followup_needed)
        for state in states
    )


def mentioned_deferral_topics(states: list[ConversationState], *, reply_text: str) -> list[str]:
    """返答に含まれている延期の話題の本文一覧。

    語の一致は「YUI が自分から再開したか」の判定には使わない（誤検出しうる。
    設計 §4 手順6）。記録だけに使う。
    """
    return [
        state.content
        for state in states
        if state.kind == ConversationStateKind.DEFERRAL.value
        and state.content
        and state.content in reply_text
    ]


def looks_assertive(text: str) -> bool:
    """断定的な言い切りに見えるか（弱いヒューリスティック）。

    discrepancy が開いているのに断定していないかを記録するためだけに使う。
    行動を変える判定には使わない。
    """
    return not any(marker in text for marker in _HEDGE_MARKERS)


# --- 画面へ渡す種類（v0.2 PR4） ----------------------------------------------

# `presented`（返答ごとに1件でき、解決・取消が無いので open のまま残る）と
# `confirmed`（作られるだけで参照されない）は、画面のどちらの表示にも使わない
# （`ChatPanel` の「この会話で」、`CandidatePanel` の訂正の候補）。全件を
# `/chat` の応答へ同梱すると、会話が長くなるほど応答が肥大する（レビューで
# 40ターン・44KBを実測）。画面が実際に使う種類だけに絞る。
DISPLAYED_STATE_KINDS = frozenset(
    {
        ConversationStateKind.REQUEST.value,
        ConversationStateKind.QUESTION_TO_YUI.value,
        ConversationStateKind.QUESTION_TO_PARTNER.value,
        ConversationStateKind.DEFERRAL.value,
        ConversationStateKind.CLOSING.value,
        ConversationStateKind.CORRECTION.value,
    }
)


# --- プロンプトへの節 ---------------------------------------------------------

PRESENTED_LIMIT = 5  # 節に出す presented の件数上限（本文の長さ上限は CONTENT_LIMIT）
# PR1 は解決・取消をしないため、応答済みの question_to_partner は解釈が
# 「答えた」と返す（PR3）まで open のまま残り続ける。上限を置かないと、
# 長い会話でこの1種類だけで節の大半を占める（レビューで36ターン30行を実測）。
RESPONDED_QUESTION_LIMIT = 5


def build_conversation_state_section(
    states: list[ConversationState],
    *,
    window_message_ids: set[int],
    discrepancy_offers: list[ConversationState] | None = None,
) -> str:
    """「# この会話で」の節（設計 §4 手順5）。

    `request`／`correction`（解釈が作る）と、`discrepancy` の確認候補
    （`discrepancy_offers`。呼び出し側が渡す回数の上限とあわせて選ぶ。
    PR3）を、規則だけの種類（question_to_yui / question_to_partner /
    deferral / closing / presented）に足す。**訂正は「思い出せること」より
    前に置く**（設計 §4 手順5の例の並びどおり。記憶より優先すると書く）。
    """
    lines: list[str] = []

    request = next(
        (state for state in states if state.kind == ConversationStateKind.REQUEST.value), None
    )
    if request is not None:
        lines.append(f"- 相手の今の用件：{request.content}")

    for state in states:
        if state.kind == ConversationStateKind.CORRECTION.value:
            lines.append(
                f"- 相手の訂正（記憶より優先する）：この会話では「{state.content}」として扱う"
            )

    # 生成失敗（503）後に相手が同じ質問を再送すると、規則は重複制約を
    # 置かないため同じ本文の question_to_yui が複数件できうる。節では
    # 同じ本文をまとめる（deferral・closing と同じ扱い。レビュー指摘）。
    seen_unanswered: set[str] = set()
    for state in states:
        if (
            state.kind == ConversationStateKind.QUESTION_TO_YUI.value
            and state.responded_message_id is None
            and not state.followup_needed
            and state.content not in seen_unanswered
        ):
            seen_unanswered.add(state.content)
            lines.append(f"- まだ答えていない相手の質問：「{state.content}」")

    seen_followup: set[str] = set()
    for state in states:
        if (
            state.kind == ConversationStateKind.QUESTION_TO_YUI.value
            and state.followup_needed
            and state.content not in seen_followup
        ):
            seen_followup.add(state.content)
            lines.append(f"- 答え損ねた相手の質問（次で答える）：「{state.content}」")

    seen_waiting: set[str] = set()
    for state in states:
        if (
            state.kind == ConversationStateKind.QUESTION_TO_PARTNER.value
            and state.responded_message_id is None
            and state.content not in seen_waiting
        ):
            seen_waiting.add(state.content)
            lines.append(f"- 自分が聞いて答えを待っている質問：「{state.content}」")

    # 応答済みは PR3 の解決が付くまで open のまま溜まり続けるので、
    # 直近だけに絞る（古いものは「同じ会話では繰り返さない」効果が薄い）。
    # 上限を切ってから重複を除く（新しいものを優先して残すため）。
    answered_once = [
        state
        for state in states
        if state.kind == ConversationStateKind.QUESTION_TO_PARTNER.value
        and state.responded_message_id is not None
    ][-RESPONDED_QUESTION_LIMIT:]
    seen_answered: set[str] = set()
    for state in answered_once:
        if state.content in seen_answered:
            continue
        seen_answered.add(state.content)
        lines.append(
            f"- 自分が聞いて一度答えのあった質問：「{state.content}」"
            "（同じ会話では繰り返し聞かない）"
        )

    # 同じ話題・同じ文言の重複行を作らない（規則は検出のたびに作成するため、
    # 同じ延期・終了が会話中に繰り返し検出されると内容が重複しうる）。
    seen_topics: set[str] = set()
    for state in states:
        if state.kind == ConversationStateKind.DEFERRAL.value and state.content not in seen_topics:
            seen_topics.add(state.content)
            lines.append(
                f"- 相手が後にすると言った話題：{state.content}"
                "（自分からは持ち出さない。相手が出したら応じる）"
            )

    if any(state.kind == ConversationStateKind.CLOSING.value for state in states):
        lines.append(
            "- 相手は終わりにしようとしている。"
            "自分から新しい質問や用件を出さず、相手の依頼には答えて短く締める"
        )

    for state in discrepancy_offers or []:
        lines.append(
            f"- 確かめてよいこと（一度だけ）：「{state.content}」。断定せず確かめる"
        )

    presented = [
        state
        for state in states
        if state.kind == ConversationStateKind.PRESENTED.value
        and state.source_message_id not in window_message_ids
    ][-PRESENTED_LIMIT:]
    for state in presented:
        lines.append(f"- 既に伝えたこと（求められていなければ繰り返さない）：{state.content}")

    if not lines:
        return ""
    return "\n".join(["# この会話で", *lines])


# --- 解釈（LLM）への入力集め（v0.2 PR3） ------------------------------------
#
# 規則だけでは「作った」「応答があった」までしか分からない。ここから先の
# 「答えたか」「確認できたか」「明示的な訂正か」は、次の情報を渡して解釈に
# 決めてもらう：①判定対象の質問とその応答候補、②確認待ちの食い違いと、
# それを渡したときの YUI の返答、③開いている会話状態、④参照できる記憶。


async def find_judgment_candidate(
    session: AsyncSession, state: ConversationState, *, conversation_id: int
) -> Message | None:
    """この質問の、まだ判定していない応答候補（計画 §4 手順3）。

    `responded_message_id` は**最初の**応答しか記録しない（規則は答えの
    質を判定できないため）。`judged_message_id` が空なら、その最初の応答
    自体がまだ判定していない候補になる。答え損ねの後に新しい返答が来た
    場合など、2件目以降の候補は `judged_message_id` より新しい、応答した
    側の発言を履歴から探す。

    `question_to_yui`（相手が聞き、YUI が答える）は YUI の発言、
    `question_to_partner`（YUI が聞き、相手が答える）はその相手の発言を見る。
    """
    if state.judged_message_id is None:
        if state.responded_message_id is None:
            return None
        return await session.get(Message, state.responded_message_id)

    stmt = select(Message).where(
        Message.conversation_id == conversation_id, Message.id > state.judged_message_id
    )
    if state.kind == ConversationStateKind.QUESTION_TO_YUI.value:
        stmt = stmt.where(Message.speaker_kind == SpeakerKind.CHARACTER.value)
    else:
        stmt = stmt.where(
            Message.speaker_kind == SpeakerKind.USER.value,
            Message.speaker_id == state.target_speaker_id,
        )
    stmt = stmt.order_by(Message.id.desc()).limit(1)
    return (await session.execute(stmt)).scalars().first()


@dataclass
class DiscrepancyOffer:
    """1件の食い違いについて、これまで何回・いつ「確かめてよいこと」として
    渡したか（計画 3節：渡す回数の上限、照合待ちの扱い）。
    """

    state: ConversationState
    offer_count: int
    # 渡した後、解釈が一度も成功して走っていなければ、渡したときの YUI の
    # 返答。解釈が成功して走っていれば（確かめられなかった場合も含め）
    # None——確かめる機会は既にあったので、渡す回数の上限内なら再び渡せる。
    pending_reply: Message | None


async def discrepancy_offer_ledger(
    session: AsyncSession, *, conversation_id: int, target_speaker_id: int | None
) -> dict[int, DiscrepancyOffer]:
    """開いていて未確認（`asked` が空）の食い違いごとに、渡した回数と
    「直前に渡したばかりか」を、会話の実行記録（`RunRecord.options` の
    `offered_discrepancy_ids`）から数える。**新しい表は作らない**
    （計画 §4 手順3：「`RunRecord.options` の記録から引く」）。
    """
    unasked = [
        state
        for state in await get_open_states(
            session,
            conversation_id=conversation_id,
            target_speaker_id=target_speaker_id,
            kind=ConversationStateKind.DISCREPANCY.value,
        )
        if state.asked_message_id is None
    ]
    if not unasked:
        return {}
    tracked_ids = {state.id for state in unasked}

    # `options` だけを読む。`RunRecord.system_prompt` は Text で数KB/行あり、
    # 未確認の食い違いが残る長い会話ではターンごとに全件読み直すことになる
    # ため、要らない列を取ってこない（レビュー指摘）。
    stmt = (
        select(RunRecord.options, Message.id)
        .join(Message, RunRecord.message_id == Message.id)
        .where(Message.conversation_id == conversation_id)
        .order_by(RunRecord.id)
    )
    rows = list((await session.execute(stmt)).all())

    counts: dict[int, int] = {}
    last_offer_index: dict[int, int] = {}
    for index, (options, _message_id) in enumerate(rows):
        for state_id in (options or {}).get("offered_discrepancy_ids") or []:
            if state_id in tracked_ids:
                counts[state_id] = counts.get(state_id, 0) + 1
                last_offer_index[state_id] = index

    def pending_message_id(state_id: int) -> int | None:
        """渡した後、解釈が実際に確かめる機会を一度も得ていなければ、その
        渡した返答の id を返す（照合待ち）。**「直前の返答だけ」ではない**
        ——渡した後の解釈が失敗し続ける限り、何ターン後でも照合待ちのまま
        （計画 §8「例外・timeout・失敗のときは照合待ちを保持し、再提示
        しない」）。解釈が一度でも成功して走れば（結果に含まれなくても）、
        確かめる機会はあったとして照合待ちを解く。
        """
        index = last_offer_index.get(state_id)
        if index is None:
            return None
        for options, _ in rows[index + 1 :]:
            if ((options or {}).get("interpretation") or {}).get("applied"):
                return None
        return rows[index][1]

    ledger: dict[int, DiscrepancyOffer] = {}
    for state in unasked:
        message_id = pending_message_id(state.id)
        pending_reply = await session.get(Message, message_id) if message_id is not None else None
        ledger[state.id] = DiscrepancyOffer(
            state=state, offer_count=counts.get(state.id, 0), pending_reply=pending_reply
        )
    return ledger


def select_discrepancy_offers(
    ledger: dict[int, DiscrepancyOffer], *, limit: int, interpretation_ran: bool = False
) -> list[ConversationState]:
    """今のターンで新しく「確かめてよいこと」として渡してよい食い違い。

    渡す回数の上限に達したものは除く（計画 3節・8節）。**照合待ち**
    （直前の返答で渡したばかりで、まだ確かめられていない）ものも、今の
    ターンの解釈が実際に走って確かめる機会があった場合は除かない——
    `interpretation_ran` が真のとき、解釈は既にこの食い違いを候補として
    受け取り、確認できなかったと分かっている（計画 §8「例外・timeout・
    解析失敗のときだけ照合待ちのまま保持し、再提示しない」。成功して
    確かめられなかった場合はこの限りではない）。
    """
    return [
        offer.state
        for offer in ledger.values()
        if offer.offer_count < limit and (offer.pending_reply is None or interpretation_ran)
    ]


@dataclass
class InterpretationCandidates:
    """解釈に渡した候補。結果の ID・kind・target を検証するのに使う
    （計画 §4 手順3の検証）。"""

    open_states: dict[int, ConversationState] = field(default_factory=dict)
    judgment_candidates: dict[int, Message] = field(default_factory=dict)
    pending_discrepancies: dict[int, Message] = field(default_factory=dict)
    memory_ids: set[int] = field(default_factory=set)
    message_ids: set[int] = field(default_factory=set)


def _format_open_states(states: list[ConversationState]) -> str:
    """解釈へ渡す「開いている会話状態」の本文。

    `confirmed` はどの操作の対象にもならない（作られるだけで、以後 ID で
    参照されない）ため出さない。`presented` は返答のたびに増え、解決・
    取消が無いので開いたまま溜まり続ける——`confirms_presented_id` の
    候補として直近の一部だけ見せれば足りる（節と同じ上限。他の種類は件数が
    自然に絞られるため上限を置かない。レビュー指摘：無制限だと長い会話で
    解釈への入力が肥大する）。
    """
    visible = [
        state
        for state in states
        if state.kind != ConversationStateKind.CONFIRMED.value
    ]
    presented = [s for s in visible if s.kind == ConversationStateKind.PRESENTED.value]
    if len(presented) > PRESENTED_LIMIT:
        drop = set(presented[:-PRESENTED_LIMIT])
        visible = [state for state in visible if state not in drop]
    if not visible:
        return "開いている会話状態: まだありません。"
    lines = ["開いている会話状態（id・種類・本文）:"]
    for state in visible:
        lines.append(f"- [{state.id}] {state.kind}: {state.content}")
    return "\n".join(lines)


def _format_judgment_candidates(candidates: dict[int, tuple[ConversationState, Message]]) -> str:
    if not candidates:
        return "判定対象の質問: まだありません。"
    lines = ["判定対象の質問（id・質問・応答候補）:"]
    for state_id, (state, message) in candidates.items():
        lines.append(f"- [{state_id}] 質問「{state.content}」→ 応答候補「{message.content}」")
    return "\n".join(lines)


def _format_pending_discrepancies(pending: dict[int, tuple[ConversationState, Message]]) -> str:
    if not pending:
        return "確認待ちの食い違い: まだありません。"
    lines = ["確認待ちの食い違い（id・内容・直前に確かめようとした返答）:"]
    for state_id, (state, message) in pending.items():
        lines.append(f"- [{state_id}] 「{state.content}」→ 直前の返答「{message.content}」")
    return "\n".join(lines)


def _format_memories(memories: list[tuple[int, str]]) -> str:
    if not memories:
        return "参照できる記憶: まだありません。"
    lines = ["参照できる記憶（id・内容）:"]
    for memory_id, content in memories:
        lines.append(f"- [{memory_id}] {content}")
    return "\n".join(lines)


def _format_transcript(history: list[Message], current: Message) -> str:
    """直近の会話。**各行に発言の id を付ける**——`correction`／`discrepancy`
    が `ref_kind="message"` を返すとき、対象を指すのに使う id は、記憶・
    会話状態と同じく本文の前に `[id]` として渡す以外に知る手段が無い
    （レビューで、id が渡っておらず実モデルでは訂正が検証を通れないことを
    実測）。
    """
    lines = ["直近の会話（id・話者・本文）:"]
    for message in [*history, current]:
        who = "YUI" if message.speaker_kind == SpeakerKind.CHARACTER.value else "相手"
        lines.append(f"- [{message.id}] {who}: {message.content}")
    return "\n".join(lines)


async def gather_interpretation_context(
    session: AsyncSession,
    *,
    conversation_id: int,
    target_speaker_id: int,
    current_message: Message,
    history: list[Message],
    memory_items: list[tuple[int, str]],
    open_states: list[ConversationState],
    discrepancy_ledger: dict[int, DiscrepancyOffer],
    context_messages: int,
) -> tuple[str, InterpretationCandidates]:
    """解釈への入力（本文）と、結果の検証に使う候補集合を組み立てる。

    `history` は生成と同じ直近12件だが、解釈の直近文脈は
    `Settings.conversation_state_context_messages`（計画 §9・既定6）に
    絞る——生成より短い窓でよいという設計判断で、生成の窓をそのまま流用
    すると設定を変えても何も変わらなくなる（レビュー指摘）。`ref_kind
    = "message"` の検証対象も、実際に見せた発言だけに絞る。
    """
    recent_history = history[-context_messages:] if context_messages > 0 else history
    judgment_candidates: dict[int, tuple[ConversationState, Message]] = {}
    for state in open_states:
        if state.kind not in {
            ConversationStateKind.QUESTION_TO_YUI.value,
            ConversationStateKind.QUESTION_TO_PARTNER.value,
        }:
            continue
        candidate = await find_judgment_candidate(
            session, state, conversation_id=conversation_id
        )
        if candidate is not None:
            judgment_candidates[state.id] = (state, candidate)

    pending_discrepancies: dict[int, tuple[ConversationState, Message]] = {
        state_id: (offer.state, offer.pending_reply)
        for state_id, offer in discrepancy_ledger.items()
        if offer.pending_reply is not None
    }

    context = "\n\n".join(
        [
            _format_transcript(recent_history, current_message),
            _format_open_states(open_states),
            _format_judgment_candidates(judgment_candidates),
            _format_pending_discrepancies(pending_discrepancies),
            _format_memories(memory_items),
        ]
    )

    candidates = InterpretationCandidates(
        open_states={state.id: state for state in open_states},
        judgment_candidates={
            state_id: message for state_id, (_, message) in judgment_candidates.items()
        },
        pending_discrepancies={
            state_id: message for state_id, (_, message) in pending_discrepancies.items()
        },
        memory_ids={memory_id for memory_id, _ in memory_items},
        message_ids={message.id for message in recent_history} | {current_message.id},
    )
    return context, candidates


# --- 解釈の結果を検証して適用する（v0.2 PR3） --------------------------------

# 「解決」の概念がある種類（計画 3節の表：`request` は「解決は無い」、
# `presented`／`confirmed`／`deferral`／`closing`／`correction` は解決の
# 欄が無く取消でしか終わらない）。開発者の手動操作
# （`POST /conversations/{id}/states/{state_id}/decide`）が `resolved` を
# 受け付けてよい種類をここに限定する（v0.2 PR4 レビュー指摘：以前は
# `decide` がどの種類の `resolved` も無条件に受け付けていた）。
RESOLVABLE_KINDS = frozenset(
    {
        ConversationStateKind.QUESTION_TO_YUI.value,
        ConversationStateKind.QUESTION_TO_PARTNER.value,
        ConversationStateKind.DISCREPANCY.value,
    }
)

WITHDRAW_REASON_KINDS: dict[str, set[str] | None] = {
    ConversationStateWithdrawReason.REOPENED.value: {
        ConversationStateKind.DEFERRAL.value,
        ConversationStateKind.CLOSING.value,
    },
    ConversationStateWithdrawReason.CANCELLED.value: {
        ConversationStateKind.QUESTION_TO_YUI.value,
        ConversationStateKind.QUESTION_TO_PARTNER.value,
    },
    # superseded は `withdrawals` からは受け付けない。置き換え先の新しい
    # request／correction の作成が検証を通った場合だけ適用するもので
    # （計画 §4 手順3）、`create_or_supersede_request`／
    # `create_or_supersede_correction` が置き換え先の作成と同じトランザ
    # クションで自分で行う。ここで無条件に受け付けると、置き換え先が
    # 無い（＝ result.request／result.correction が無いか検証で落ちた）の
    # に有効な訂正・用件だけが消える（レビューで実測）。
    ConversationStateWithdrawReason.SUPERSEDED.value: set(),
    # misdetected は検出（規則・解釈）が誤って作った行を取り消すためのもの。
    # `presented` は YUI が実際にそう言った事実そのもの、`confirmed` は
    # 一度作ったら参照されない記録、`request` は superseded でしか置き換え
    # ないため、誤検出の余地・意味が無い（計画 3節の表に取消欄が無い。
    # レビューで指摘）。
    ConversationStateWithdrawReason.MISDETECTED.value: {
        ConversationStateKind.QUESTION_TO_YUI.value,
        ConversationStateKind.QUESTION_TO_PARTNER.value,
        ConversationStateKind.DEFERRAL.value,
        ConversationStateKind.CLOSING.value,
        ConversationStateKind.DISCREPANCY.value,
        ConversationStateKind.CORRECTION.value,
    },
}


def withdraw_reason_applies(kind: str, reason: str) -> bool:
    """この種類にこの取消理由が使えるか（計画 3節の表のとおり）。

    解釈（LLM）の `withdrawals` だけでなく、開発者の手動操作
    （`POST /conversations/{id}/states/{state_id}/decide`）からも呼ぶ。
    経路によって許す組み合わせを変えない（レビュー指摘：operator 経路が
    この検証を通さず、`question_to_yui` に `reopened` のような不整合な
    組み合わせを受け入れていた）。
    """
    allowed = WITHDRAW_REASON_KINDS.get(reason)
    return allowed is None or kind in allowed


def _ref_is_valid(payload: RefPayload, candidates: InterpretationCandidates) -> bool:
    if payload.ref_kind == ConversationStateRefKind.MEMORY.value:
        return payload.ref_id in candidates.memory_ids
    if payload.ref_kind == ConversationStateRefKind.MESSAGE.value:
        return payload.ref_id in candidates.message_ids
    return False


async def apply_interpretation_result(
    session: AsyncSession,
    result: InterpretationResult,
    *,
    candidates: InterpretationCandidates,
    conversation_id: int,
    target_speaker_id: int,
    speaker_id: int,
    current_message: Message,
) -> dict[str, object]:
    """解釈の結果を検証し、通った項目だけ適用する（計画 §4 手順4・検証）。

    **ID・kind・target が候補の範囲外の項目は、その項目だけ捨てる。**
    **同じ状態に両立しない操作（answered と unanswered に同じ id）が
    返ったら、その状態への操作だけ両方捨てる。**
    """
    decided_by = DecisionSource.LLM.value
    summary: dict[str, object] = {
        "resolved_state_ids": [],
        "followup_state_ids": [],
        "withdrawn_state_ids": [],
        "asked_discrepancy_ids": [],
        "resolved_discrepancy_ids": [],
        "created_request": False,
        "created_correction": False,
        "created_discrepancy": False,
        "created_deferral": False,
        "created_closing": False,
        "created_confirmed": False,
        "dropped": [],
    }
    dropped: list[str] = summary["dropped"]  # type: ignore[assignment]

    # 両立しない操作は、その状態への操作を**両方とも**捨てる（計画 §4 手順3・
    # §8：「解決と取消に同じ id」も対象。answered/unanswered だけでなく、
    # 解決系（answered・resolved_discrepancy）と withdrawals の競合も見る。
    # 片方が先に適用されて後発が弾かれる「早い者勝ち」にしない
    # （レビューで、resolved の直後に withdrawn へ上書きされる例を実測）。
    withdrawal_ids = {withdrawal.state_id for withdrawal in result.withdrawals}
    resolving_ids = set(result.answered_state_ids) | set(result.resolved_discrepancy_ids)
    conflicting = (
        (set(result.answered_state_ids) & set(result.unanswered_state_ids))
        | (resolving_ids & withdrawal_ids)
    )
    for state_id in conflicting:
        dropped.append(f"両立しない操作が競合: {state_id}")

    for state_id in dict.fromkeys(result.answered_state_ids):
        if state_id in conflicting:
            continue
        candidate = candidates.judgment_candidates.get(state_id)
        state = candidates.open_states.get(state_id)
        if candidate is None or state is None:
            dropped.append(f"answered_state_ids: 判定対象ではない id {state_id}")
            continue
        await resolve_state(
            session, state, resolved_message_id=candidate.id, decided_by=decided_by
        )
        state.judged_message_id = candidate.id
        summary["resolved_state_ids"].append(state_id)  # type: ignore[union-attr]

    for state_id in dict.fromkeys(result.unanswered_state_ids):
        if state_id in conflicting:
            continue
        candidate = candidates.judgment_candidates.get(state_id)
        state = candidates.open_states.get(state_id)
        if candidate is None or state is None:
            dropped.append(f"unanswered_state_ids: 判定対象ではない id {state_id}")
            continue
        await mark_followup_needed(
            session, state, judged_message_id=candidate.id, decided_by=decided_by
        )
        summary["followup_state_ids"].append(state_id)  # type: ignore[union-attr]

    for state_id in dict.fromkeys(result.asked_discrepancy_ids):
        offered_reply = candidates.pending_discrepancies.get(state_id)
        state = candidates.open_states.get(state_id)
        if offered_reply is None or state is None:
            dropped.append(f"asked_discrepancy_ids: 確認待ちではない id {state_id}")
            continue
        state.asked_message_id = offered_reply.id
        state.responded_message_id = current_message.id
        await session.flush()
        summary["asked_discrepancy_ids"].append(state_id)  # type: ignore[union-attr]

    correction_ref: tuple[str, int] | None = None
    if result.correction is not None:
        if _ref_is_valid(result.correction, candidates):
            correction_ref = (result.correction.ref_kind, result.correction.ref_id)
        else:
            dropped.append("correction: ref が候補にありません")

    for state_id in dict.fromkeys(result.resolved_discrepancy_ids):
        if state_id in conflicting:
            continue
        state = candidates.open_states.get(state_id)
        if state is None or state.kind != ConversationStateKind.DISCREPANCY.value:
            dropped.append(f"resolved_discrepancy_ids: 開いている食い違いではない id {state_id}")
            continue
        await resolve_state(
            session, state, resolved_message_id=current_message.id, decided_by=decided_by
        )
        summary["resolved_discrepancy_ids"].append(state_id)  # type: ignore[union-attr]

    for withdrawal in result.withdrawals:
        if withdrawal.state_id in conflicting:
            continue
        state = candidates.open_states.get(withdrawal.state_id)
        if state is None or not withdraw_reason_applies(state.kind, withdrawal.reason):
            dropped.append(f"withdrawals: 対象外 id {withdrawal.state_id}")
            continue
        await withdraw_state(
            session,
            state,
            reason=withdrawal.reason,
            decided_by=decided_by,
            resolved_message_id=current_message.id,
        )
        summary["withdrawn_state_ids"].append(withdrawal.state_id)  # type: ignore[union-attr]

    if result.deferral_topic:
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.DEFERRAL.value,
            content=result.deferral_topic[:CONTENT_LIMIT],
            source_message_id=current_message.id,
            speaker_id=speaker_id,
            target_speaker_id=target_speaker_id,
            detected_by=DetectionSource.LLM.value,
        )
        summary["created_deferral"] = True

    if result.is_closing and not any(
        state.kind == ConversationStateKind.CLOSING.value
        for state in candidates.open_states.values()
    ):
        await create_state(
            session,
            conversation_id=conversation_id,
            kind=ConversationStateKind.CLOSING.value,
            content=current_message.content[:CONTENT_LIMIT],
            source_message_id=current_message.id,
            speaker_id=speaker_id,
            target_speaker_id=target_speaker_id,
            detected_by=DetectionSource.LLM.value,
        )
        summary["created_closing"] = True

    if result.confirms_presented_id is not None:
        presented = candidates.open_states.get(result.confirms_presented_id)
        if presented is None or presented.kind != ConversationStateKind.PRESENTED.value:
            dropped.append("confirms_presented_id: 対象外")
        elif is_acknowledgement_only(current_message.content):
            dropped.append("confirms_presented_id: 相づちのみ")
        elif await _has_confirmed(
            session,
            conversation_id=conversation_id,
            target_speaker_id=target_speaker_id,
            presented_id=presented.id,
        ):
            dropped.append("confirms_presented_id: 重複")
        else:
            await create_state(
                session,
                conversation_id=conversation_id,
                kind=ConversationStateKind.CONFIRMED.value,
                content=presented.content,
                source_message_id=current_message.id,
                speaker_id=speaker_id,
                target_speaker_id=target_speaker_id,
                ref_kind=ConversationStateRefKind.STATE.value,
                ref_id=presented.id,
                detected_by=DetectionSource.LLM.value,
            )
            summary["created_confirmed"] = True

    if result.request:
        # 既存の open な request と同じ内容なら、create_or_supersede_request は
        # 何も変えずにそれを返す。summary の created_request を「実際に行を
        # 作った・置き換えた」の意味に保つため、そのケースは False のままに
        # する（レビュー指摘：常に True だと台帳を読む側が誤読する）。
        existing_request = next(
            (
                state
                for state in candidates.open_states.values()
                if state.kind == ConversationStateKind.REQUEST.value
            ),
            None,
        )
        unchanged = (
            existing_request is not None
            and existing_request.content == result.request[:CONTENT_LIMIT]
        )
        await create_or_supersede_request(
            session,
            conversation_id=conversation_id,
            target_speaker_id=target_speaker_id,
            content=result.request,
            source_message_id=current_message.id,
            speaker_id=speaker_id,
            decided_by=decided_by,
        )
        summary["created_request"] = not unchanged

    if correction_ref is not None:
        assert result.correction is not None
        await create_or_supersede_correction(
            session,
            conversation_id=conversation_id,
            target_speaker_id=target_speaker_id,
            ref_kind=correction_ref[0],
            ref_id=correction_ref[1],
            content=result.correction.content[:CONTENT_LIMIT],
            source_message_id=current_message.id,
            speaker_id=speaker_id,
            decided_by=decided_by,
        )
        summary["created_correction"] = True

    if result.discrepancy is not None:
        discrepancy_ref = (result.discrepancy.ref_kind, result.discrepancy.ref_id)
        if correction_ref is not None and discrepancy_ref == correction_ref:
            # 同じ対象を correction と discrepancy の両方が指したら correction を
            # 優先する（計画 3節）。
            dropped.append("discrepancy: 同じ対象の correction を優先")
        elif not _ref_is_valid(result.discrepancy, candidates):
            dropped.append("discrepancy: ref が候補にありません")
        elif not await can_create_discrepancy(
            session,
            conversation_id=conversation_id,
            target_speaker_id=target_speaker_id,
            ref_kind=result.discrepancy.ref_kind,
            ref_id=result.discrepancy.ref_id,
        ):
            dropped.append("discrepancy: 重複制約により作成しない")
        else:
            await create_state(
                session,
                conversation_id=conversation_id,
                kind=ConversationStateKind.DISCREPANCY.value,
                content=result.discrepancy.note[:CONTENT_LIMIT],
                source_message_id=current_message.id,
                speaker_id=speaker_id,
                target_speaker_id=target_speaker_id,
                ref_kind=result.discrepancy.ref_kind,
                ref_id=result.discrepancy.ref_id,
                detected_by=DetectionSource.LLM.value,
            )
            summary["created_discrepancy"] = True

    return summary
