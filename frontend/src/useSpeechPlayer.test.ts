/**
 * 再生器の振る舞いを固定する。
 *
 * v0.1 の完了条件「返答を重複再生せず、停止操作で音声を止められる」は
 * この画面側の制御で成り立っている。バックエンドのテストでは守れないため、
 * ここで固定する。
 *
 * 記録の通知も対象にする。鳴っていないのに「再生中」のまま残る、開始だけ
 * 通知して終わりを通知しない、といった食い違いを防ぐ。
 */

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import type { DeliveryNotice } from "./types";
import { useSpeechPlayer } from "./useSpeechPlayer";

/** 音声の取得を、テスト側の任意のタイミングで完了させる。 */
class PendingSpeech {
  resolve!: (blob: Blob) => void;
  reject!: (error: Error) => void;
  readonly promise: Promise<Blob>;

  constructor() {
    this.promise = new Promise((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}

class FakeAudio {
  static instances: FakeAudio[] = [];
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  playCount = 0;
  pauseCount = 0;
  /** play() が返す約束。自動再生の失敗を作るときに差し替える。 */
  playResult: Promise<void> = Promise.resolve();

  constructor(readonly src: string) {
    FakeAudio.instances.push(this);
  }

  play(): Promise<void> {
    this.playCount += 1;
    return this.playResult;
  }

  pause(): void {
    this.pauseCount += 1;
  }
}

let pending: PendingSpeech[] = [];
let notices: { messageId: number; state: DeliveryNotice }[] = [];

function speechBlob(): Blob {
  return new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" });
}

/** 取得中の音声を1つ完了させ、再生の開始まで進める。 */
async function deliverSpeech(index = 0) {
  await act(async () => {
    pending[index].resolve(speechBlob());
    await pending[index].promise;
  });
}

beforeEach(() => {
  pending = [];
  notices = [];
  FakeAudio.instances = [];

  vi.spyOn(api, "speech").mockImplementation(() => {
    const next = new PendingSpeech();
    pending.push(next);
    return next.promise;
  });
  vi.spyOn(api, "notifyDelivery").mockImplementation(async (messageId, state) => {
    notices.push({ messageId, state });
    return {
      id: messageId,
      conversation_id: 1,
      speaker_kind: "character",
      speaker_id: null,
      source: "local_text",
      content: "",
      delivery_state: state,
      delivery_started_at: null,
      delivery_finished_at: null,
      created_at: "2026-09-07T00:00:00Z",
    };
  });

  vi.stubGlobal("Audio", FakeAudio);
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: () => "blob:test",
    revokeObjectURL: () => undefined,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("停止と追い越し", () => {
  it("取得中に止めると、あとから届いた音声は鳴らない", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    act(() => result.current.stop());
    await deliverSpeech();

    expect(FakeAudio.instances).toHaveLength(0);
    expect(result.current.playingId).toBeNull();
  });

  it("続けて再生すると、鳴るのは最後の1つだけ", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    act(() => result.current.play(2));
    // 先に始めた 1 の音声が、あとから届く。
    await deliverSpeech(0);
    await deliverSpeech(1);

    expect(FakeAudio.instances).toHaveLength(1);
    await waitFor(() => expect(result.current.playingId).toBe(2));
  });

  it("再生中に別の返答を再生すると、前の再生を中断として記録する", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech(0);
    await waitFor(() => expect(result.current.playingId).toBe(1));

    act(() => result.current.play(2));
    await deliverSpeech(1);
    await waitFor(() => expect(result.current.playingId).toBe(2));

    expect(notices).toEqual([
      { messageId: 1, state: "playing" },
      { messageId: 1, state: "aborted" },
      { messageId: 2, state: "playing" },
    ]);
    // 前の音声は止まっている。重ねて鳴らさない。
    expect(FakeAudio.instances[0].pauseCount).toBeGreaterThan(0);
  });

  it("停止すると中断として記録する", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech();
    await waitFor(() => expect(result.current.playingId).toBe(1));

    act(() => result.current.stop());

    expect(notices).toEqual([
      { messageId: 1, state: "playing" },
      { messageId: 1, state: "aborted" },
    ]);
    expect(result.current.playingId).toBeNull();
  });
});

describe("画面を離れたとき", () => {
  it("取得中に離れると、あとから届いた音声は鳴らない", async () => {
    const { result, unmount } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    unmount();
    await deliverSpeech();

    // 所有元が消えたあとに鳴り出さない。
    expect(FakeAudio.instances.some((audio) => audio.playCount > 0)).toBe(false);
    expect(notices).toEqual([]);
  });

  it("再生中に離れると、中断として記録する", async () => {
    const { result, unmount } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech();
    await waitFor(() => expect(result.current.playingId).toBe(1));

    unmount();

    // 開始を通知した発言には、必ず終わりも通知する。
    expect(notices).toEqual([
      { messageId: 1, state: "playing" },
      { messageId: 1, state: "aborted" },
    ]);
    expect(FakeAudio.instances[0].pauseCount).toBeGreaterThan(0);
  });
});

describe("再生の終わり方", () => {
  it("最後まで鳴ると、話し終えたとして記録する", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech();
    await waitFor(() => expect(result.current.playingId).toBe(1));

    act(() => FakeAudio.instances[0].onended?.());

    expect(notices).toEqual([
      { messageId: 1, state: "playing" },
      { messageId: 1, state: "completed" },
    ]);
    expect(result.current.playingId).toBeNull();
  });

  it("鳴り始めたあとの異常終了を、中断として記録する", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech();
    await waitFor(() => expect(result.current.playingId).toBe(1));

    act(() => FakeAudio.instances[0].onerror?.());

    // 鳴っていないのに記録が「再生中」のまま残らない。
    expect(notices).toEqual([
      { messageId: 1, state: "playing" },
      { messageId: 1, state: "aborted" },
    ]);
    expect(result.current.error).not.toBeNull();
  });

  it("鳴り始める前に失敗した場合は、開始も終わりも記録しない", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await act(async () => {
      pending[0].reject(new Error("音声合成は無効になっています。"));
      await pending[0].promise.catch(() => undefined);
    });

    expect(notices).toEqual([]);
    expect(result.current.playingId).toBeNull();
    expect(result.current.error).toContain("音声合成は無効");
  });
});

describe("待ち時間の計測", () => {
  it("再生した返答の待ち時間を残す", async () => {
    const { result } = renderHook(() => useSpeechPlayer());

    act(() => result.current.play(1));
    await deliverSpeech();
    await waitFor(() => expect(result.current.playingId).toBe(1));

    const timing = result.current.timings[1];
    expect(timing).toBeDefined();
    expect(timing.fetchMs).toBeGreaterThanOrEqual(0);
    expect(timing.startMs).toBeGreaterThanOrEqual(0);
  });
});
