"""評価結果のレポート（ISSUE-006）。

版を変えて2回流し、2つのレポートを並べて比べられるようにする。そのため
先頭に、比較に必要なもの（人格版、モデル、生成設定、実行日時）を必ず残す。

機械で判定した結果と、人が読む欄を混ぜない。設計書が「人手評価を含め、
LLM自身の評価だけで更新を採用しない」としているため、自動判定だけで
「合格」と書かない。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation.runner import ScenarioResult
from app.evaluation.scenario import ASPECTS
from app.persona import Persona


def _escape(text: str) -> str:
    """表のセルに入れる。改行と縦棒だけ潰す。"""
    return text.replace("|", "\\|").replace("\n", " ")


def build_markdown(
    results: list[ScenarioResult],
    *,
    persona: Persona,
    model: str,
    model_digest: str | None,
    options: dict[str, Any],
    repeat: int,
    started_at: datetime,
) -> str:
    lines: list[str] = ["# 評価用会話の結果", ""]
    lines += [
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| 実行日時 | {started_at.isoformat(timespec='seconds')} |",
        f"| 人格の版 | {persona.version} |",
        f"| モデル | {model} |",
        f"| モデルの版 | {model_digest or '不明'} |",
        f"| 生成設定 | {_escape(json.dumps(options, ensure_ascii=False))} |",
        f"| 試行回数 | 各シナリオ {repeat} 回 |",
        "",
        "自動判定は、機械で確かめられる観点だけを見ている。口調・着眼点・"
        "迎合していないかは人が読んで判断する（各ターンの「人が見る点」）。",
        "",
    ]

    machine = [r for r in results if not r.human_only]
    human_only = [r for r in results if r.human_only]
    total = sum(r.total for r in machine)
    passed = sum(r.passed for r in machine)
    lines += [
        "## まとめ",
        "",
        f"自動判定を通った試行：**{passed} / {total}**"
        f"（機械で判定した {len(machine)} シナリオ）",
        "",
        f"人が読んで判断するシナリオ：**{len(human_only)}**。"
        "機械の判定項目を書いていないため、上の数には含めない。読んで判断する"
        "まで、通ったとも通らなかったとも言えない。",
        "",
        f"実行できなかった試行：**{sum(r.failed_to_run for r in results)}**"
        "（モデルの呼び出しや出力の読み取りに失敗したもの）。",
        "",
        "| シナリオ | 観点 | 判定 |",
        "| --- | --- | --- |",
    ]
    for result in results:
        aspect = ASPECTS.get(result.scenario.aspect, result.scenario.aspect).split("：")[0]
        verdict = (
            "人手（未判定）"
            if result.human_only
            else f"{result.passed} / {result.total}"
        )
        lines.append(f"| {result.scenario.id} | {aspect} | {verdict} |")
    lines.append("")

    for result in results:
        scenario = result.scenario
        lines += [
            f"## {scenario.id}",
            "",
            f"観点：{ASPECTS.get(scenario.aspect, scenario.aspect)}",
            "",
        ]
        if scenario.description:
            lines += [scenario.description, ""]

        for index, attempt in enumerate(result.attempts, start=1):
            # 人手だけのシナリオは、機械では判定していない。「通過」と書かない。
            mark = (
                "未判定（人が読む）"
                if result.human_only
                else ("通過" if attempt.ok else "不通過")
            )
            lines += [f"### {index} 回目（{mark}）", ""]
            for turn in attempt.turns:
                lines.append(f"- 入力：{turn.text}")
                if turn.error:
                    lines.append(f"  - **失敗：{_escape(turn.error)}**")
                    continue
                lines.append(f"  - 返答：{_escape(turn.reply)}")
                if turn.referenced:
                    lines.append(f"  - 渡した記憶：{'、'.join(turn.referenced)}")
                if turn.referenced_states:
                    lines.append(
                        f"  - 渡した状態：{_escape('、'.join(turn.referenced_states))}"
                    )
                for check in turn.checks:
                    lines.append(
                        f"  - {'○' if check.ok else '×'} {check.name}：{_escape(check.detail)}"
                    )
                if turn.human_check:
                    lines.append(f"  - 人が見る点：{turn.human_check} → ")
            # 会話以外に行った操作。どの操作の後の返答かを追えるようにする。
            for action in attempt.actions:
                lines.append(f"- 操作：{_escape(action)}")
            for check in attempt.action_checks:
                lines.append(
                    f"  - {'○' if check.ok else '×'} {check.name}：{_escape(check.detail)}"
                )
            for index, reflection in enumerate(attempt.reflections, start=1):
                label = "振り返り" if len(attempt.reflections) == 1 else f"振り返り{index}"
                lines.append(f"- {label}")
                if reflection.error:
                    lines.append(f"  - **失敗：{_escape(reflection.error)}**")
                    continue
                if reflection.candidates:
                    for candidate in reflection.candidates:
                        lines.append(f"  - 候補：{_escape(candidate)}")
                else:
                    lines.append("  - 候補：なし")
                for check in reflection.checks:
                    lines.append(
                        f"  - {'○' if check.ok else '×'} {check.name}："
                        f"{_escape(check.detail)}"
                    )
                if reflection.human_check:
                    lines.append(f"  - 人が見る点：{reflection.human_check} → ")
            lines.append("")

    return "\n".join(lines)


def write_report(
    directory: Path,
    results: list[ScenarioResult],
    *,
    persona: Persona,
    model: str,
    model_digest: str | None,
    options: dict[str, Any],
    repeat: int,
    started_at: datetime,
) -> Path:
    """Markdown と JSON を書き出し、Markdown の位置を返す。

    JSON は、版どうしを機械で突き合わせるときに使う。
    """
    directory.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        results,
        persona=persona,
        model=model,
        model_digest=model_digest,
        options=options,
        repeat=repeat,
        started_at=started_at,
    )
    report_path = directory / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "persona_version": persona.version,
        "model": model,
        "model_digest": model_digest,
        "options": options,
        "repeat": repeat,
        "scenarios": [
            {
                "id": result.scenario.id,
                "aspect": result.scenario.aspect,
                "passed": result.passed,
                "total": result.total,
                "attempts": [asdict(attempt) for attempt in result.attempts],
            }
            for result in results
        ],
    }
    (directory / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path
