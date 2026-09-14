"""1ターンの待ち時間を、会話状態の解釈（LLM）の有無で比べる（v0.2 PR5）。

    .venv/bin/python -m app.evaluation.measure_turn_latency

評価用会話（`run_scenarios`）とは別に、単純な固定会話を解釈あり／なしで
1回ずつ流し、ターンごとの壁時計時間と生成時間を比べる。あわせて、
解釈ありの側で作られた会話状態の `detected_by`／`decided_by` の内訳も出す。

**この測定は、他に Ollama を叩くプロセスが無い状態で行うこと。** 評価器を
同時に複数走らせると、解釈の呼び出しが timeout に間に合わなくなり、
数字が実態より悪く出る（v0.2 PR5 で実測。詳細は docs/result/v0.2.md）。

数字は `docs/result/v0.2.md` に記録している。人格の版やモデルを変えて
測り直すときは、ここを再実行してその記録を更新する。
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agent import ConversationAgent
from app.agent.memory_store import get_or_create_speaker
from app.config import BACKEND_ROOT, Settings
from app.llm.ollama_client import OllamaClient
from app.models import Base, Conversation, ConversationMode, ConversationState
from app.persona import load_persona

# docs/result/v0.2.md の「待ち時間」節はこのファイルの出力を引用している。
# 標準出力だけだと後から突き合わせる手段が無いため（第4回レビュー指摘）、
# 実行のたびにここへも書き出す。
OUT_DIR = BACKEND_ROOT.parent / "logs" / "evals"

# 質問・延期・訂正・終了の合図を一通り含む、固定の5ターン会話。
TURNS = [
    "土曜と日曜、どちらが空いてる？",
    "映画の話はまた今度にしよう",
    "そういえば最近仕事が忙しくてさ",
    "ごめん、土曜じゃなくて日曜だった",
    "そろそろ寝るね",
]


async def _run_once(*, conversation_state_llm: bool, directory: str) -> dict[str, Any]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{directory}/measure.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(conversation_state_llm=conversation_state_llm)
    llm = OllamaClient(
        host=settings.ollama_host,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
        timeout=settings.llm_timeout_seconds,
        think=settings.ollama_think,
    )
    agent = ConversationAgent(llm=llm, persona=load_persona(), settings=settings)

    session = factory()
    speaker = await get_or_create_speaker(
        session, source="local", external_id="pr5-measure", display_name="開発者"
    )
    conversation = Conversation(mode=ConversationMode.LOCAL.value)
    session.add(conversation)
    await session.flush()
    await session.commit()

    turn_wall_ms: list[float] = []
    gen_latency_ms: list[int | None] = []
    for text in TURNS:
        started = time.perf_counter()
        result = await agent.respond(
            session, conversation=conversation, speaker=speaker, text=text
        )
        turn_wall_ms.append((time.perf_counter() - started) * 1000)
        # 生成が失敗した回は latency_ms が None になりうる。turn_wall_ms と
        # 添字を揃えて None を積むことで、表示側の [i - 1] が turn とずれて
        # 別のターンの値を指したり範囲外になったりしないようにする。
        gen_latency_ms.append(result.run.latency_ms)

    # detected_by／decided_by の内訳は、解釈を有効にしたときだけ意味を持つ
    # （無効な構成では request／correction／discrepancy／confirmed が
    # 作られず、decided_by は常に空のため）。
    breakdown: Counter[tuple[str, str, str]] = Counter()
    if conversation_state_llm:
        stmt = select(
            ConversationState.kind, ConversationState.detected_by, ConversationState.decided_by
        )
        for kind, detected_by, decided_by in (await session.execute(stmt)).all():
            breakdown[(kind, detected_by, decided_by or "-")] += 1

    await session.close()
    await engine.dispose()
    return {"turn_wall_ms": turn_wall_ms, "gen_latency_ms": gen_latency_ms, "breakdown": breakdown}


async def main() -> None:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    persona = load_persona()
    settings = Settings()
    emit(f"人格の版 {persona.version} ／ モデル {settings.ollama_model}")
    emit(
        f"conversation_state_llm_timeout_seconds={settings.conversation_state_llm_timeout_seconds}"
    )

    emit("=== 解釈 無効 ===")
    with tempfile.TemporaryDirectory(prefix="pr5-measure-") as directory:
        without = await _run_once(conversation_state_llm=False, directory=directory)
    for i, ms in enumerate(without["turn_wall_ms"], start=1):
        gen = without["gen_latency_ms"][i - 1]
        emit(f"  turn {i}: {ms:.0f}ms (生成 {gen if gen is not None else '不明'}ms)")
    avg_without = sum(without["turn_wall_ms"]) / len(without["turn_wall_ms"])
    emit(f"  平均: {avg_without:.0f}ms")

    emit("\n=== 解釈 有効 ===")
    with tempfile.TemporaryDirectory(prefix="pr5-measure-") as directory:
        with_interp = await _run_once(conversation_state_llm=True, directory=directory)
    for i, ms in enumerate(with_interp["turn_wall_ms"], start=1):
        gen = with_interp["gen_latency_ms"][i - 1]
        emit(f"  turn {i}: {ms:.0f}ms (生成 {gen if gen is not None else '不明'}ms)")
    avg_with = sum(with_interp["turn_wall_ms"]) / len(with_interp["turn_wall_ms"])
    emit(f"  平均: {avg_with:.0f}ms")

    diff = avg_with - avg_without
    emit(f"\n差分: +{diff:.0f}ms/ターン（{(avg_with / avg_without - 1) * 100:.0f}%増）")

    emit("\n=== detected_by / decided_by の内訳（解釈 有効）===")
    for (kind, detected_by, decided_by), count in sorted(with_interp["breakdown"].items()):
        emit(f"  {kind:20s} detected={detected_by:5s} decided={decided_by:9s} count={count}")

    started_at = datetime.now(UTC)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{started_at.strftime('%Y%m%dT%H%M%S')}-turn-latency.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n記録: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
