"""LLM へ渡すプロンプトの組み立て。

設計書「6. 会話・自発的行動の流れ」の 5 に対応する。基本の人格、現在の状態、
関連する記憶、直近の会話をまとめて渡す。事実と推測は分けて示し、
記憶にない事実を作らないよう指示する。
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent.memory_store import KIND_LABEL, RetrievedMemory
from app.llm.base import ChatMessage
from app.models import Certainty, Message, Speaker, SpeakerKind, utcnow
from app.persona import Persona

JST = ZoneInfo("Asia/Tokyo")

def _format_memory(item: RetrievedMemory) -> str:
    memory = item.memory
    label = KIND_LABEL.get(memory.kind, memory.kind)
    when = memory.occurred_at or memory.created_at
    stamp = when.astimezone(JST).strftime("%Y-%m-%d") if when else "日付不明"
    return f"- [{label}／{stamp}] {memory.content}"


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


def build_system_prompt(
    *,
    persona: Persona,
    memories: list[RetrievedMemory],
    speaker: Speaker | None,
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
    sections.append(build_memory_section(memories))
    return "\n".join(sections)


def build_messages(
    *,
    persona: Persona,
    memories: list[RetrievedMemory],
    speaker: Speaker | None,
    history: list[Message],
    user_text: str,
    now: datetime | None = None,
) -> tuple[list[ChatMessage], str]:
    """LLM へ渡すメッセージ列と、記録用のシステムプロンプトを返す。"""
    system_prompt = build_system_prompt(
        persona=persona, memories=memories, speaker=speaker, now=now
    )
    messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
    for message in history:
        role = "assistant" if message.speaker_kind == SpeakerKind.CHARACTER.value else "user"
        messages.append(ChatMessage(role=role, content=message.content))
    messages.append(ChatMessage(role="user", content=user_text))
    return messages, system_prompt
