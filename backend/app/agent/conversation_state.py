"""v0.2 の会話状態：現在の用件・未回答の質問・提示済みの内容・延期・終了・
食い違いを扱う（設計書「6. 会話・自発的行動の流れ」、計画 docs/plan/v0.2.md）。

**発言があったことと解決したことを分ける。** 相手の次の発言が来ただけでは
質問も食い違いも解決にならない。解決・取消は解釈（LLM）か開発者の操作でだけ
起き、規則は「作る」と「応答があった」しか記録しない。

この PR1 では規則（辞書・正規表現）だけを使う。`request` / `confirmed` /
`correction` / `discrepancy` の作成、解決、取消は解釈でしか起きないため、
この版の respond() からは呼ばれない。ただし種類ごとの重複制約
（`can_create_discrepancy`、`create_or_supersede_correction`）はここに置き、
PR3 が呼び出す形にしておく。
"""

from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ConversationState,
    ConversationStateKind,
    ConversationStateStatus,
    ConversationStateWithdrawReason,
    DetectionSource,
    Message,
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
    を終了の合図として拾わない。
    """
    last = _last_sentence(text)
    return any(pattern.fullmatch(last) for pattern in _CLOSING_PATTERNS)


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


# --- プロンプトへの節 ---------------------------------------------------------

PRESENTED_LIMIT = 5  # 節に出す presented の件数上限（本文の長さ上限は CONTENT_LIMIT）
# PR1 は解決・取消をしないため、応答済みの question_to_partner は解釈が
# 「答えた」と返す（PR3）まで open のまま残り続ける。上限を置かないと、
# 長い会話でこの1種類だけで節の大半を占める（レビューで36ターン30行を実測）。
RESPONDED_QUESTION_LIMIT = 5


def build_conversation_state_section(
    states: list[ConversationState], *, window_message_ids: set[int]
) -> str:
    """「# この会話で」の節（設計 §4 手順5）。

    PR1 の時点では規則だけが作る種類（question_to_yui / question_to_partner /
    deferral / closing / presented）だけを扱う。`request` / `correction` /
    `discrepancy`（訂正・確認）は解釈が要るため PR3 で節に足す。
    """
    lines: list[str] = []

    # 生成失敗（503）後に相手が同じ質問を再送すると、規則は重複制約を
    # 置かないため同じ本文の question_to_yui が複数件できうる。節では
    # 同じ本文をまとめる（deferral・closing と同じ扱い。レビュー指摘）。
    seen_unanswered: set[str] = set()
    for state in states:
        if (
            state.kind == ConversationStateKind.QUESTION_TO_YUI.value
            and state.responded_message_id is None
            and state.content not in seen_unanswered
        ):
            seen_unanswered.add(state.content)
            lines.append(f"- まだ答えていない相手の質問：「{state.content}」")
            # responded が入っていて未判定のものは、答え損ねと決まっていない
            # ので渡さない（followup_needed は PR3 が立てる）。

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
