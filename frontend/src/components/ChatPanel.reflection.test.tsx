/**
 * 会話終了と振り返りの分離（フェーズ4 PR5）。
 *
 * 終了は待たずに返るようになった。画面が確かめるのは次の3つ。
 *
 * - 待っている間、どこを処理しているかが出る（26秒無表示だった）。
 * - 終わってから候補の取り直しへ進む。途中で進めると候補が空になる。
 * - 失敗はやり直せる。理由を出し、終了済みには切り替えない。
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatPanel } from "./ChatPanel";
import { api } from "../api";
import type { ReflectionProgress } from "../types";
import type React from "react";

function progress(overrides: Partial<ReflectionProgress> = {}): ReflectionProgress {
  return {
    conversation_id: 1,
    state: "running",
    step: null,
    error: null,
    started_at: "2026-09-08T00:00:00Z",
    completed_at: null,
    ...overrides,
  };
}

const player = {
  play: vi.fn(),
  stop: vi.fn(),
  playing: null,
  timings: {},
} as never;

function panel(props: Partial<React.ComponentProps<typeof ChatPanel>> = {}) {
  const base = {
    conversationId: 1,
    messages: [],
    liveEntries: [],
    state: "open" as const,
    loading: false,
    speechAvailable: false,
    player,
    speaker: { source: "local_text", external_id: "dev", display_name: "開発者" },
    onSpeakerChange: vi.fn(),
    speakerNames: {},
    view: 0,
    onEntry: () => true,
    onEnded: vi.fn(),
    onReflectionFailed: vi.fn(),
    onNewConversation: vi.fn(),
  };
  return { ...base, ...props };
}

function renderPanel(onEnded = vi.fn(), onReflectionFailed = vi.fn()) {
  render(<ChatPanel {...panel({ onEnded, onReflectionFailed })} />);
  return { onEnded, onReflectionFailed };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  cleanup();
});

describe("会話の終了と振り返り", () => {
  it("待っている間、どこを処理しているかを出す", async () => {
    vi.spyOn(api, "endConversation").mockResolvedValue(progress());
    vi.spyOn(api, "reflection")
      .mockResolvedValueOnce(progress({ step: "picking" }))
      .mockResolvedValueOnce(progress({ step: "selecting" }))
      // 4回目の呼び出し（目標の抽出）も段階として出す。ラベルが無いと
      // 「振り返っています：」だけになる（PR6 の第1回レビューの指摘4）。
      .mockResolvedValueOnce(progress({ step: "goals" }))
      .mockResolvedValue(progress({ state: "completed", step: null }));

    const { onEnded } = renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));

    expect(await screen.findByText(/話に出たことを拾っています/)).toBeInTheDocument();
    expect(await screen.findByText(/覚えておくものを選んでいます/)).toBeInTheDocument();
    expect(await screen.findByText(/次に話したいことを考えています/)).toBeInTheDocument();
    // 終わってから、候補の取り直しへ進む。
    await waitFor(() => expect(onEnded).toHaveBeenCalled());
  });

  it("終わるまで、候補の取り直しへ進まない", async () => {
    vi.spyOn(api, "endConversation").mockResolvedValue(progress());
    vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "selecting" }));

    const { onEnded } = renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));

    await screen.findByText(/覚えておくものを選んでいます/);
    // 処理中の間は呼ばない。呼ぶと、まだできていない候補を取りに行く。
    expect(onEnded).not.toHaveBeenCalled();
  });

  it("失敗したら理由を出し、終了済みにはしない", async () => {
    vi.spyOn(api, "endConversation").mockResolvedValue(progress());
    vi.spyOn(api, "reflection").mockResolvedValue(
      progress({ state: "failed", error: "振り返りの出力を読み取れませんでした。" }),
    );

    const { onEnded } = renderPanel();
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));

    expect(
      await screen.findByText("振り返りの出力を読み取れませんでした。"),
    ).toBeInTheDocument();
    // やり直せる。終了済みへ切り替えない。
    expect(onEnded).not.toHaveBeenCalled();
  });
});

describe("監視の寿命（第1回レビューの指摘2・3）", () => {
  it("振り返り中の会話を開き直したら、終了ボタンを押さなくても監視する", async () => {
    // 別のタブで終了した会話や、処理中に画面を再読み込みした場合。
    // 監視が終了ボタンの中だけにあると、終わっても画面が変わらない。
    const end = vi.spyOn(api, "endConversation");
    vi.spyOn(api, "reflection")
      .mockResolvedValueOnce(progress({ step: "states" }))
      .mockResolvedValue(progress({ state: "completed" }));

    const onEnded = vi.fn();
    render(<ChatPanel {...panel({ state: "reflecting", onEnded })} />);

    expect(await screen.findByText(/関心と関係を見直しています/)).toBeInTheDocument();
    await waitFor(() => expect(onEnded).toHaveBeenCalledWith(0));
    // 終了はもう一度呼ばない。開き直しただけで積み直さない。
    expect(end).not.toHaveBeenCalled();
  });

  it("会話を切り替えたら監視を止め、前の会話の結果を出さない", async () => {
    vi.spyOn(api, "endConversation").mockResolvedValue(progress());
    const watch = vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "picking" }));

    const { rerender } = render(
      <ChatPanel {...panel({ state: "reflecting" })} />,
    );
    await screen.findByText(/話に出たことを拾っています/);

    // 別の会話（世代が進む）へ切り替える。
    rerender(
      <ChatPanel {...panel({ conversationId: 2, view: 1 })} />,
    );

    // 前の会話の進行を残さない。
    await waitFor(() =>
      expect(screen.queryByText(/話に出たことを拾っています/)).not.toBeInTheDocument(),
    );
    const calls = watch.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 700));
    // 止まっているので、これ以上は見に行かない。
    expect(watch.mock.calls.length).toBe(calls);
  });
});

describe("失敗からの復帰と、遅れて返る終了（第2回レビュー）", () => {
  it("振り返り中として開いた会話が失敗したら、続けられる状態へ戻す", async () => {
    // 読み取り専用のままだと、その場でやり直せない。会話は終わっていない。
    vi.spyOn(api, "reflection").mockResolvedValue(
      progress({ state: "failed", error: "接続できません" }),
    );
    const onEnded = vi.fn();
    const onReflectionFailed = vi.fn();
    render(
      <ChatPanel {...panel({ state: "reflecting", onEnded, onReflectionFailed })} />,
    );

    expect(await screen.findByText("接続できません")).toBeInTheDocument();
    await waitFor(() => expect(onReflectionFailed).toHaveBeenCalledWith(0));
    // 失敗は「終わった」ではない。候補の取り直しへは進まない。
    expect(onEnded).not.toHaveBeenCalled();
  });

  it("終了の応答が遅れて返っても、切り替え先の会話で監視を始めない", async () => {
    // 会話Aの終了POSTが返る前にBへ移り、その後Aの応答が届く状況。
    let resolveEnd: (value: ReflectionProgress) => void = () => {};
    vi.spyOn(api, "endConversation").mockReturnValue(
      new Promise<ReflectionProgress>((resolve) => {
        resolveEnd = resolve;
      }),
    );
    const watch = vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "picking" }));

    const { rerender } = render(<ChatPanel {...panel()} />);
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));

    // 応答を待たずに別の会話へ移る（世代が進む）。
    rerender(<ChatPanel {...panel({ conversationId: 2, view: 1 })} />);
    resolveEnd(progress());
    await new Promise((resolve) => setTimeout(resolve, 50));

    // 切り替え先の会話で、前の会話の進行を出さない・監視も始めない。
    expect(screen.queryByText(/話に出たことを拾っています/)).not.toBeInTheDocument();
    expect(watch).not.toHaveBeenCalled();
  });
});

describe("切り替え先で操作できること（第3回レビュー）", () => {
  it("終了の応答を待っている間に会話を切り替えても、切り替え先で操作できる", async () => {
    // 旧要求の後始末は世代が違うので走らない。busy を解かないままだと、
    // 切り替え先の会話で送信も終了もできなくなる。
    let resolveEnd: (value: ReflectionProgress) => void = () => {};
    vi.spyOn(api, "endConversation").mockReturnValue(
      new Promise<ReflectionProgress>((resolve) => {
        resolveEnd = resolve;
      }),
    );
    vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "picking" }));

    const { rerender } = render(<ChatPanel {...panel()} />);
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));
    // 応答を待っている間は押せない。
    expect(screen.getByRole("button", { name: "終了して振り返る" })).toBeDisabled();

    rerender(<ChatPanel {...panel({ conversationId: 2, view: 1 })} />);
    resolveEnd(progress());
    await new Promise((resolve) => setTimeout(resolve, 50));

    // 切り替え先では、入力も終了もできる（送信ボタンは本文が空だと無効なので、
    // busy をそのまま映す入力欄で見る）。
    expect(screen.getByRole("button", { name: "終了して振り返る" })).not.toBeDisabled();
    expect(screen.getByRole("textbox")).not.toBeDisabled();
  });

  it("終了が失敗した場合も、切り替え先で操作できる", async () => {
    let rejectEnd: (reason: Error) => void = () => {};
    vi.spyOn(api, "endConversation").mockReturnValue(
      new Promise<ReflectionProgress>((_resolve, reject) => {
        rejectEnd = reject;
      }),
    );
    vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "picking" }));

    const { rerender } = render(<ChatPanel {...panel()} />);
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));
    rerender(<ChatPanel {...panel({ conversationId: 2, view: 1 })} />);
    rejectEnd(new Error("接続できません"));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(screen.getByRole("button", { name: "終了して振り返る" })).not.toBeDisabled();
    // 切り替え先に、前の会話の失敗を出さない。
    expect(screen.queryByText("接続できません")).not.toBeInTheDocument();
  });
});

describe("送信と終了で busy を共有していること（第4回レビュー）", () => {
  it("遅れて返った旧送信が、切り替え先で始めた要求の busy を解かない", async () => {
    // busy は送信と終了で共有している。旧送信の後始末に世代の検査が無いと、
    // 切り替え先の要求が終わっていないのにボタンが有効になり、二重に送れる。
    let resolveChat: (value: unknown) => void = () => {};
    vi.spyOn(api, "chat").mockReturnValue(
      new Promise((resolve) => {
        resolveChat = resolve;
      }) as never,
    );
    let resolveEnd: (value: ReflectionProgress) => void = () => {};
    vi.spyOn(api, "endConversation").mockReturnValue(
      new Promise<ReflectionProgress>((resolve) => {
        resolveEnd = resolve;
      }),
    );
    vi.spyOn(api, "reflection").mockResolvedValue(progress({ step: "picking" }));

    // 会話Aで送信する（返答は保留）。
    const { rerender } = render(<ChatPanel {...panel()} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Aへの発言" } });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    // 会話Bへ移り、Bで終了を始める（こちらも保留）。
    rerender(<ChatPanel {...panel({ conversationId: 2, view: 1 })} />);
    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));
    expect(screen.getByRole("button", { name: "終了して振り返る" })).toBeDisabled();

    // ここでAの送信が返る。Bの要求はまだ終わっていない。
    resolveChat({
      conversation_id: 1,
      user_message: { id: 1, speaker_id: 1 },
      reply: { id: 2 },
      run: {},
      used_memories: [],
    });
    await new Promise((resolve) => setTimeout(resolve, 50));

    // Bの要求が終わるまで、押せないままであること。
    expect(screen.getByRole("button", { name: "終了して振り返る" })).toBeDisabled();

    resolveEnd(progress());
    await new Promise((resolve) => setTimeout(resolve, 50));
  });

  it("遅れて返った旧送信の失敗を、切り替え先に出さない", async () => {
    let rejectChat: (reason: Error) => void = () => {};
    vi.spyOn(api, "chat").mockReturnValue(
      new Promise((_resolve, reject) => {
        rejectChat = reject;
      }) as never,
    );

    const { rerender } = render(<ChatPanel {...panel()} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Aへの発言" } });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));
    rerender(<ChatPanel {...panel({ conversationId: 2, view: 1 })} />);

    rejectChat(new Error("接続できません"));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(screen.queryByText("接続できません")).not.toBeInTheDocument();
    // 打ち直しの文も、移った先の入力欄へ書き戻さない。
    expect(screen.getByRole("textbox")).toHaveValue("");
  });
});
