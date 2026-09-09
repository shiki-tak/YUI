/**
 * 目標の画面（フェーズ4 PR4）。
 *
 * ここで固定するのは、目標だけが持つ区別が画面に出ることである。
 *
 * - 候補は採用するまで「採用済み」に入らない。
 * - 質問を投げたこと（実行）と、相手が答えたこと（達成）を混ぜない。
 * - 根拠が変わったものは、確認するまで行動に渡さないと分かる形で出す。
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GoalPanel } from "./GoalPanel";
import { api } from "../api";
import type { Goal } from "../types";

function goal(overrides: Partial<Goal> = {}): Goal {
  return {
    id: 1,
    content: "土曜に見た映画の感想を聞く",
    subject_speaker_id: null,
    trigger: "next_conversation",
    due_at: null,
    basis_memory_ids: null,
    basis_is_provisional: false,
    needs_review: false,
    review_reason: null,
    status: "active",
    auto_adopted: false,
    visibility: "private",
    visible_to_speaker_id: null,
    source_conversation_id: null,
    last_executed_at: null,
    completed_at: null,
    created_at: "2026-09-08T00:00:00Z",
    updated_at: "2026-09-08T00:00:00Z",
    ...overrides,
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  cleanup();
});

describe("GoalPanel", () => {
  it("候補は採用するまで採用済みに入らない", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ id: 1, status: "pending", content: "映画の感想を聞く" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText("目標の候補（1）")).toBeInTheDocument();
    expect(screen.getByText("採用済み（0）")).toBeInTheDocument();
    expect(screen.getByText("まだありません。")).toBeInTheDocument();
  });

  it("実行した時刻と、達成した時刻を分けて出す", async () => {
    // 質問は投げたが、まだ答えは返っていない状態。
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ last_executed_at: "2026-09-12T03:00:00Z" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText("採用済み（1）")).toBeInTheDocument();
    // 実行済みの印は出るが、終わったものには入らない。
    expect(screen.getByText(/実行済み/)).toBeInTheDocument();
    expect(screen.queryByText(/終わったもの/)).not.toBeInTheDocument();
  });

  it("達成にすると、終わり方を区別して残す", async () => {
    const update = vi
      .spyOn(api, "updateGoal")
      .mockResolvedValue(goal({ status: "done", completed_at: "2026-09-13T00:00:00Z" }));
    vi.spyOn(api, "goals")
      .mockResolvedValueOnce([goal()])
      .mockResolvedValue([goal({ status: "done", completed_at: "2026-09-13T00:00:00Z" })]);

    render(<GoalPanel refreshKey={0} />);
    fireEvent.click(await screen.findByRole("button", { name: "達成にする" }));

    await waitFor(() => expect(update).toHaveBeenCalledWith(1, expect.objectContaining({ status: "done" })));
    expect(await screen.findByText("終わったもの（1）")).toBeInTheDocument();
    expect(screen.getByText("達成")).toBeInTheDocument();
  });

  it("根拠が変わったものは、確認するまで渡さないと分かる形で出す", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({
        needs_review: true,
        review_reason: "根拠にした記憶 #3 が訂正された",
        basis_memory_ids: [3],
      }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText("根拠が変わったもの（1）")).toBeInTheDocument();
    expect(
      screen.getByText(/確認するまで行動には渡しません/),
    ).toBeInTheDocument();
    expect(screen.getByText("根拠にした記憶 #3 が訂正された")).toBeInTheDocument();
    // 印が付いている間は「採用済み」に数えない。
    expect(screen.getByText("採用済み（0）")).toBeInTheDocument();
  });

  it("指定日以降の目標は、その日付を出す", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ trigger: "after_date", due_at: "2026-09-13T00:00:00Z" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText(/指定した日以降/)).toBeInTheDocument();
  });
});

describe("GoalPanel の操作（第1回レビューの指摘4）", () => {
  it("撤回中は終わったものと分け、戻せる", async () => {
    const update = vi.spyOn(api, "updateGoal").mockResolvedValue(goal());
    vi.spyOn(api, "goals").mockResolvedValue([goal({ status: "withdrawn" })]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText("撤回中（1）")).toBeInTheDocument();
    expect(screen.queryByText(/終わったもの/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "戻す" }));
    await waitFor(() =>
      expect(update).toHaveBeenCalledWith(1, expect.objectContaining({ status: "active" })),
    );
  });

  it("終わったものには状態を変える操作を出さない", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ status: "done", completed_at: "2026-09-13T00:00:00Z" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    expect(await screen.findByText("終わったもの（1）")).toBeInTheDocument();
    expect(screen.getByText(/状態を戻せません/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "戻す" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "達成にする" })).not.toBeInTheDocument();
  });

  it("指定日以降の目標は、日付そのものを出す", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ trigger: "after_date", due_at: "2026-09-13T00:00:00+09:00" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    // ラベルだけでなく、日付の値が出ていること。
    expect(await screen.findByText(/2026\/09\/13/)).toBeInTheDocument();
  });

  it("API が失敗したら、そのまま画面に出す", async () => {
    vi.spyOn(api, "goals").mockResolvedValue([goal()]);
    vi.spyOn(api, "updateGoal").mockRejectedValue(
      new Error("終わった目標の状態は変えられません。"),
    );
    render(<GoalPanel refreshKey={0} />);

    fireEvent.click(await screen.findByRole("button", { name: "達成にする" }));
    expect(
      await screen.findByText("終わった目標の状態は変えられません。"),
    ).toBeInTheDocument();
  });
});

describe("GoalPanel の遷移の網羅（第2回レビュー）", () => {
  /** API が許す遷移（backend/app/api/goals.py の _ALLOWED_TRANSITIONS）。 */
  const ALLOWED: Record<string, string[]> = {
    active: ["done", "withdrawn", "cancelled", "expired"],
    withdrawn: ["active", "cancelled", "expired"],
  };

  it("要確認の目標を、印を下ろさずに外せる", async () => {
    // 根拠が変わったまま「確認済み」にしないと外せないのでは、確認の意味が
    // 無くなる。撤回・取消・期限切れは、根拠の正しさと関係なく選べる。
    const update = vi.spyOn(api, "updateGoal").mockResolvedValue(goal());
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ needs_review: true, review_reason: "根拠にした記憶 #3 が訂正された" }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    await screen.findByText("根拠が変わったもの（1）");
    for (const [label, status] of [
      ["撤回する", "withdrawn"],
      ["取り消す", "cancelled"],
      ["期限切れにする", "expired"],
    ] as const) {
      update.mockClear();
      fireEvent.click(screen.getByRole("button", { name: label }));
      await waitFor(() =>
        expect(update).toHaveBeenCalledWith(1, expect.objectContaining({ status })),
      );
    }
  });

  it("撤回中から、API が許す遷移をすべて起こせる", async () => {
    const update = vi.spyOn(api, "updateGoal").mockResolvedValue(goal());
    vi.spyOn(api, "goals").mockResolvedValue([goal({ status: "withdrawn" })]);
    render(<GoalPanel refreshKey={0} />);

    await screen.findByText("撤回中（1）");
    const sent: string[] = [];
    for (const label of ["戻す", "取り消す", "期限切れにする"]) {
      update.mockClear();
      fireEvent.click(screen.getByRole("button", { name: label }));
      await waitFor(() => expect(update).toHaveBeenCalled());
      sent.push(String(update.mock.calls[0][1].status));
    }
    expect(sent.sort()).toEqual([...ALLOWED.withdrawn].sort());
  });

  it("採用済みから、API が許す遷移をすべて起こせる", async () => {
    const update = vi.spyOn(api, "updateGoal").mockResolvedValue(goal());
    vi.spyOn(api, "goals").mockResolvedValue([goal()]);
    render(<GoalPanel refreshKey={0} />);

    await screen.findByText("採用済み（1）");
    const sent: string[] = [];
    for (const label of ["達成にする", "撤回する", "取り消す", "期限切れにする"]) {
      update.mockClear();
      fireEvent.click(screen.getByRole("button", { name: label }));
      await waitFor(() => expect(update).toHaveBeenCalled());
      sent.push(String(update.mock.calls[0][1].status));
    }
    expect(sent.sort()).toEqual([...ALLOWED.active].sort());
  });
  it("自動採用したものを、開発者が採用したものと区別して見せる", async () => {
    // 区別が付かないと、確認を経ていないものを経たものとして読む。まとめて
    // 戻す判断もできない（フェーズ4 PR11）。
    vi.spyOn(api, "goals").mockResolvedValue([
      goal({ id: 1, content: "土曜に見た映画の感想を聞く", auto_adopted: true }),
      goal({ id: 2, content: "カメラの設定の話をする", auto_adopted: false }),
    ]);
    render(<GoalPanel refreshKey={0} />);

    const auto = await screen.findByText("土曜に見た映画の感想を聞く");
    expect(auto.closest(".memory")).toHaveTextContent("自動採用");

    const manual = screen.getByText("カメラの設定の話をする");
    expect(manual.closest(".memory")).not.toHaveTextContent("自動採用");
  });
});
