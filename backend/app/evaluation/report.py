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
from statistics import median
from typing import Any

from app.evaluation.runner import ScenarioResult
from app.evaluation.scenario import ASPECTS
from app.persona import Persona


def _escape(text: str) -> str:
    """表のセルに入れる。改行と縦棒だけ潰す。"""
    return text.replace("|", "\\|").replace("\n", " ")


def _interpretation_summary(results: list[ScenarioResult]) -> list[str]:
    """解釈（LLM）の失敗と所要時間（ISSUE-047）。

    解釈の失敗は設計どおり握りつぶされ、返答は返る。上の「実行できなかった
    試行」には数えられないため、別枠で出さないと 0/3 の内訳が timeout か
    判定の誤りかを分けられない。解釈が無効な実行では何も出さない。
    """
    turns = [
        turn
        for result in results
        for attempt in result.attempts
        for turn in attempt.turns
        if (turn.interpretation or {}).get("attempted")
    ]
    if not turns:
        return []

    failed = [turn for turn in turns if turn.interpretation_failed]
    reasons: dict[str, int] = {}
    for turn in failed:
        status = turn.interpretation or {}
        # 集計の鍵は `error_kind`（timeout / json_not_found / json_decode /
        # not_an_object / schema / llm_error）。`error` の本文にはモデルの生出力が
        # 入るため、鍵にすると同じ種類の失敗が出力の差だけで別々に数えられ、
        # 改行や `|` が表を壊す（ISSUE-047 のレビュー指摘）。
        key = str(status.get("error_kind") or "不明")
        reasons[key] = reasons.get(key, 0) + 1

    latencies = [
        ms
        for turn in turns
        if isinstance(ms := (turn.interpretation or {}).get("latency_ms"), int)
    ]
    lines = [
        f"解釈（LLM）を試みたターン：**{len(turns)}**、"
        f"うち失敗：**{len(failed)}**"
        "（失敗しても返答は返るため、上の「実行できなかった試行」には入らない）。",
        "",
    ]
    if reasons:
        lines += ["| 失敗の理由 | 回数 |", "| --- | ---: |"]
        lines += [f"| {_escape(reason)} | {count} |" for reason, count in sorted(reasons.items())]
        lines.append("")
    if latencies:
        lines += [
            f"解釈の所要時間（呼び出し単体）：最小 {min(latencies) / 1000:.1f}s ／ "
            f"中央 {median(latencies) / 1000:.1f}s ／ 最大 {max(latencies) / 1000:.1f}s。",
            "",
        ]
    return lines


def build_markdown(
    results: list[ScenarioResult],
    *,
    persona: Persona,
    model: str,
    model_digest: str | None,
    options: dict[str, Any],
    repeat: int,
    started_at: datetime,
    conversation_state_llm: bool,
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
        # v0.2 PR5：解釈（LLM）の有無で結果が大きく変わるため、レポート単体で
        # どちらの構成の測定かが分かるようにする（レビュー指摘：以前は
        # ログから判別できなかった）。
        f"| 会話状態の解釈（LLM） | {'有効' if conversation_state_llm else '無効'} |",
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
        f"自動判定を通った試行：**{passed} / {total}**（機械で判定した {len(machine)} シナリオ）",
        "",
        f"人が読んで判断するシナリオ：**{len(human_only)}**。"
        "機械の判定項目を書いていないため、上の数には含めない。読んで判断する"
        "まで、通ったとも通らなかったとも言えない。",
        "",
        f"実行できなかった試行：**{sum(r.failed_to_run for r in results)}**"
        "（モデルの呼び出しや出力の読み取りに失敗したもの）。",
        "",
    ]
    lines += _interpretation_summary(results)
    lines += [
        "| シナリオ | 観点 | 判定 |",
        "| --- | --- | --- |",
    ]
    for result in results:
        aspect = ASPECTS.get(result.scenario.aspect, result.scenario.aspect).split("：")[0]
        verdict = "人手（未判定）" if result.human_only else f"{result.passed} / {result.total}"
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
                "未判定（人が読む）" if result.human_only else ("通過" if attempt.ok else "不通過")
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
                    lines.append(f"  - 渡した状態：{_escape('、'.join(turn.referenced_states))}")
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
                    # 候補の中身は失敗時も残す。件数だけでは、どの言い回しが
                    # 余計に一致したのかを md だけ読んでも追えない
                    # （レビュー指摘：runner.py 側は候補を保持していたが、
                    # md への出力だけここで捨てていた）。
                    for candidate in reflection.candidates:
                        lines.append(f"  - 候補：{_escape(candidate)}")
                    continue
                if reflection.candidates:
                    for candidate in reflection.candidates:
                        lines.append(f"  - 候補：{_escape(candidate)}")
                else:
                    lines.append("  - 候補：なし")
                for check in reflection.checks:
                    lines.append(
                        f"  - {'○' if check.ok else '×'} {check.name}：{_escape(check.detail)}"
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
    conversation_state_llm: bool,
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
        conversation_state_llm=conversation_state_llm,
    )
    report_path = directory / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    machine_results = [r for r in results if not r.human_only]
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "persona_version": persona.version,
        "model": model,
        "model_digest": model_digest,
        "options": options,
        "conversation_state_llm": conversation_state_llm,
        "repeat": repeat,
        # report.md の「自動判定を通った試行」見出しと同じ集計（人手専用の
        # シナリオを除く）。json から scenarios を素朴に合計すると、人手専用
        # シナリオが混ざって md の数字と食い違う（レビュー指摘）。
        "machine_passed": sum(r.passed for r in machine_results),
        "machine_total": sum(r.total for r in machine_results),
        "scenarios": [
            {
                "id": result.scenario.id,
                "aspect": result.scenario.aspect,
                "human_only": result.human_only,
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
