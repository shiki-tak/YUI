"""評価用会話を流すコマンド（ISSUE-006）。

    .venv/bin/python -m app.evaluation.run --repeat 3
    .venv/bin/python -m app.evaluation.run --persona-version <別の版> --repeat 3

実モデル（Ollama）を呼ぶ。pytest とは別に、人格・記憶の質を測るために使う。
通常利用のDBは使わない（シナリオごとに使い捨てのDBを作る）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import BACKEND_ROOT, get_settings
from app.evaluation.report import write_report
from app.evaluation.runner import run_scenarios
from app.evaluation.scenario import Scenario, ScenarioError, load_scenarios
from app.llm import get_llm_client
from app.llm.base import LLMClient  # noqa: TC001
from app.persona import PersonaError, load_persona

DEFAULT_SCENARIOS = BACKEND_ROOT / "evals" / "scenarios"
DEFAULT_OUT = BACKEND_ROOT.parent / "logs" / "evals"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.run",
        description="評価用会話を実モデルで流し、結果をレポートに出す。",
    )
    parser.add_argument("--persona-version", default=None, help="人格の版。既定は設定の版")
    parser.add_argument("--repeat", type=int, default=1, help="各シナリオを流す回数")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=None, help="レポートの出力先")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="ID",
        help="流すシナリオを id で絞る（複数指定できる）",
    )
    parser.add_argument(
        "--aspect", default=None, help="観点で絞る（memory / grounding / persona / …）"
    )
    parser.add_argument(
        "--conversation-state-llm",
        action="store_true",
        help=(
            "会話状態の解釈（LLM）を有効にして流す（v0.2 PR3。既定は無効のまま。"
            "計画 §7 のシナリオ 5・7・9〜18・20〜24 はこれが無いと解釈が起きない）"
        ),
    )
    return parser.parse_args(argv)


def _select(scenarios: list[Scenario], args: argparse.Namespace) -> list[Scenario]:
    selected = scenarios
    if args.only:
        wanted = set(args.only)
        selected = [s for s in selected if s.id in wanted]
        missing = wanted - {s.id for s in selected}
        if missing:
            raise ScenarioError(f"シナリオが見つかりません: {'、'.join(sorted(missing))}")
    if args.aspect:
        selected = [s for s in selected if s.aspect == args.aspect]
    if not selected:
        raise ScenarioError("流すシナリオがありません。")
    return selected


def _progress(scenario: Scenario, attempt: int, repeat: int) -> None:
    print(f"  {scenario.id} … {attempt}/{repeat}", flush=True)


async def _run(args: argparse.Namespace, llm: LLMClient) -> Path:
    settings = get_settings()
    if args.conversation_state_llm:
        settings = settings.model_copy(update={"conversation_state_llm": True})
    persona = load_persona(args.persona_version)
    scenarios = _select(load_scenarios(args.scenarios), args)

    print(
        f"人格の版 {persona.version} ／ モデル {settings.ollama_model} ／ "
        f"{len(scenarios)} シナリオ × {args.repeat} 回",
        flush=True,
    )
    started_at = datetime.now(UTC)
    results = await run_scenarios(
        scenarios,
        llm=llm,
        persona=persona,
        settings=settings,
        repeat=args.repeat,
        on_progress=_progress,
    )

    # 実際に応答したモデルの版。実行記録と同じものをレポートにも残す。
    model = settings.ollama_model
    model_digest: str | None = None
    for result in results:
        for attempt in result.attempts:
            if attempt.model is not None:
                model = attempt.model
                model_digest = attempt.model_digest
                break
        if model_digest is not None:
            break

    directory = args.out or (
        DEFAULT_OUT / f"{started_at.strftime('%Y%m%dT%H%M%S')}-{persona.version}"
    )
    return write_report(
        directory,
        results,
        persona=persona,
        model=model,
        model_digest=model_digest,
        options={
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
            "think": settings.ollama_think,
        },
        repeat=args.repeat,
        started_at=started_at,
        conversation_state_llm=settings.conversation_state_llm,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    llm = get_llm_client()
    try:
        report = asyncio.run(_run(args, llm))
    except (ScenarioError, PersonaError) as exc:
        print(f"実行できません: {exc}", file=sys.stderr)
        return 1
    print(f"レポート: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
