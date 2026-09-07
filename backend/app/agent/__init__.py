"""自作エージェント：人格適用、記憶選択、会話進行、振り返り。"""

from __future__ import annotations

from functools import lru_cache

from app.agent.conversation import ConversationAgent, ReplyResult
from app.config import get_settings
from app.llm import get_llm_client
from app.persona import load_persona

__all__ = ["ConversationAgent", "ReplyResult", "get_agent"]


@lru_cache
def get_agent() -> ConversationAgent:
    return ConversationAgent(
        llm=get_llm_client(),
        persona=load_persona(),
        settings=get_settings(),
    )
