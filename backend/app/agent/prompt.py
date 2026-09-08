"""LLM へ渡すプロンプトの組み立て。

設計書「6. 会話・自発的行動の流れ」の 5 に対応する。基本の人格、現在の状態、
関連する記憶、直近の会話をまとめて渡す。事実と推測は分けて示し、
記憶にない事実を作らないよう指示する。
"""

from __future__ import annotations

from datetime import datetime

from app.agent.character_state import KIND_LABEL as STATE_LABEL
from app.agent.memory_store import KIND_LABEL, RetrievedMemory
from app.config import LOCAL_TZ, to_local
from app.llm.base import ChatMessage
from app.models import (
    Certainty,
    CharacterState,
    Message,
    Provenance,
    Speaker,
    SpeakerKind,
    utcnow,
)
from app.persona import Persona

JST = LOCAL_TZ

def _format_memory(item: RetrievedMemory) -> str:
    memory = item.memory
    label = KIND_LABEL.get(memory.kind, memory.kind)
    when = memory.occurred_at or memory.created_at
    stamp = to_local(when).strftime("%Y-%m-%d") if when else "日付不明"
    # 伝聞は、本人から聞いたことと区別して渡す。区別せずに渡すと、別の人から
    # 聞いた話を、目の前の相手が言ったこととして扱う（設計書 3C）。
    source = "／人づてに聞いた" if memory.provenance == Provenance.HEARSAY.value else ""
    return f"- [{label}／{stamp}{source}] {memory.content}"


def build_memory_section(memories: list[RetrievedMemory]) -> str:
    """記憶を事実と推測に分けて並べる。"""
    if not memories:
        return (
            "# 思い出せること\n"
            "この相手・話題について思い出せることはありません。"
            "過去に話した内容を、覚えているふりをして作らないでください。"
        )

    facts = [m for m in memories if m.memory.certainty == Certainty.FACT.value]
    guesses = [m for m in memories if m.memory.certainty == Certainty.INFERENCE.value]

    lines = ["# 思い出せること"]
    if facts:
        lines.append("## 確かなこと")
        lines += [_format_memory(m) for m in facts]
    if guesses:
        lines.append("## そう思っているだけのこと（断定しない）")
        lines += [_format_memory(m) for m in guesses]
    lines.append("")
    lines.append("ここに書かれていないことは、思い出せないこととして扱ってください。")
    return "\n".join(lines)


def build_state_section(states: list[CharacterState]) -> str:
    """いまの関心と、相手との関係。

    固定人格とは別の見出しで渡す。人格は変わらない基準、こちらは経験で
    変わっていくもので、混ぜると「いま思っていること」が人格の一部として
    扱われる（設計書 4.1）。
    """
    if not states:
        return ""
    lines = ["# いまの自分", ""]
    for state in states:
        label = STATE_LABEL.get(state.kind, state.kind)
        topic = f"／{state.topic}" if state.topic else ""
        lines.append(f"- [{label}{topic}] {state.content}")
    lines.append("")
    lines.append(
        "これは経験から変わっていくものです。相手の意見に合わせて、"
        "その場で反転させないでください。"
    )
    return "\n".join(lines)


def build_system_prompt(
    *,
    persona: Persona,
    memories: list[RetrievedMemory],
    speaker: Speaker | None,
    states: list[CharacterState] | None = None,
    now: datetime | None = None,
) -> str:
    now = (now or utcnow()).astimezone(JST)
    sections = [persona.to_prompt(), ""]

    partner = speaker.display_name if speaker else "相手"
    sections.append("# いまの状況")
    sections.append(f"- 現在時刻：{now.strftime('%Y-%m-%d %H:%M')}（日本時間）")
    sections.append(f"- 話している相手：{partner}")
    sections.append("- 場所：開発者との個人的な会話。配信ではありません。")
    sections.append("")
    state_section = build_state_section(states or [])
    if state_section:
        sections.append(state_section)
        sections.append("")
    sections.append(build_memory_section(memories))
    return "\n".join(sections)


def build_messages(
    *,
    persona: Persona,
    memories: list[RetrievedMemory],
    speaker: Speaker | None,
    history: list[Message],
    user_text: str,
    states: list[CharacterState] | None = None,
    now: datetime | None = None,
) -> tuple[list[ChatMessage], str]:
    """LLM へ渡すメッセージ列と、記録用のシステムプロンプトを返す。"""
    system_prompt = build_system_prompt(
        persona=persona, memories=memories, speaker=speaker, states=states, now=now
    )
    messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
    # 相手が複数いる会話では、発言に誰のものかを付ける。付けないと、履歴の
    # 発言がすべて同じ「user」に見え、直前の相手の発言を目の前の相手のものと
    # して扱う（Bさんに「Aさんの場合でも」と返すなど）。設計書の3Cが求める
    # 「別人の相反する好みを共存させる」「人物を特定できない場合は同一人物と
    # 決めつけない」に必要な情報である。
    speaker_ids = {
        message.speaker_id
        for message in history
        if message.speaker_kind != SpeakerKind.CHARACTER.value
    }
    if speaker is not None:
        speaker_ids.add(speaker.id)
    label_speakers = len(speaker_ids) > 1

    for message in history:
        if message.speaker_kind == SpeakerKind.CHARACTER.value:
            messages.append(ChatMessage(role="assistant", content=message.content))
            continue
        content = message.content
        if label_speakers:
            name = message.speaker.display_name if message.speaker else "相手"
            content = f"{name}：{content}"
        messages.append(ChatMessage(role="user", content=content))

    if label_speakers and speaker is not None:
        user_text = f"{speaker.display_name}：{user_text}"
    messages.append(ChatMessage(role="user", content=user_text))
    return messages, system_prompt
