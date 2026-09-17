/**
 * 会話を切り替えたときに、前の会話の結果が混ざらないことを固定する。
 *
 * 返答は数秒かかる。待っている間に履歴から別の会話を開けるため、遅れて
 * 届いた結果をそのまま反映すると、表示している履歴と次の送信先が食い違う。
 */

import "@testing-library/jest-dom/vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DeliveryNotice, DeliveryState } from "./types";
import App from "./App";
import { api } from "./api";
import type {
  ChatResponse,
  Conversation,
  ConversationDetail,
  ConversationStateRecord,
  Message,
} from "./types";
import { shouldApplyDelivery } from "./types";

function message(id: number, kind: "user" | "character", content: string): Message {
  return {
    id,
    conversation_id: 1,
    speaker_kind: kind,
    speaker_id: kind === "user" ? 1 : null,
    source: "local_text",
    content,
    delivery_state: kind === "user" ? "completed" : "generated",
    delivery_started_at: null,
    delivery_finished_at: null,
    delivered_char_count: null,
    created_at: "2026-09-07T00:00:00Z",
  };
}

const CONVERSATIONS: Conversation[] = [
  {
    id: 2,
    mode: "local",
    title: "べつの会話",
    started_at: "2026-09-07T00:00:00Z",
    ended_at: null,
    reflection_started_at: null,
    reflection_completed_at: null,
  },
];

const CONVERSATION_B: ConversationDetail = {
  ...CONVERSATIONS[0],
  messages: [message(10, "user", "べつの会話の発言")],
};

function chatResponse(): ChatResponse {
  return {
    conversation_id: 1,
    user_message: message(20, "user", "さきほどの質問"),
    reply: message(21, "character", "さきほどの返答"),
    run: {
      id: 1,
      message_id: 21,
      provider: "fake",
      model: "fake",
      model_digest: null,
  persona_version: "2026-09-06.2",
      options: null,
      referenced_memory_ids: [],
  referenced_state_ids: null,
      retrieval_ms: 1,
      latency_ms: 10,
      prompt_tokens: null,
      completion_tokens: null,
      created_at: "2026-09-07T00:00:00Z",
    },
    used_memories: [],
    conversation_states: [],
  };
}

// 明示的に片付ける。前のテストの画面が残ると、要素が二重に見つかる。
afterEach(cleanup);

beforeEach(() => {
  vi.spyOn(api, "health").mockResolvedValue({
    ok: true,
    persona: { name: "YUI", version: "test" },
    llm: { ok: true, provider: "fake", model: "fake" },
    // 音声は使わない。切り替えの経路だけを見る。
    voice: { ok: false, enabled: false, provider: "fake" },
  });
  vi.spyOn(api, "speakers").mockResolvedValue([]);
  vi.spyOn(api, "pendingCandidates").mockResolvedValue([]);
  vi.spyOn(api, "memories").mockResolvedValue([]);
  vi.spyOn(api, "conversations").mockResolvedValue(CONVERSATIONS);
  vi.spyOn(api, "conversation").mockResolvedValue(CONVERSATION_B);
  vi.spyOn(api, "conversationStates").mockResolvedValue([]);
});

describe("会話の切り替え", () => {
  it("返答を待つ間に別の会話を開くと、遅れて届いた返答を画面に出さない", async () => {
    let resolveChat: (value: ChatResponse) => void = () => undefined;
    vi.spyOn(api, "chat").mockImplementation(
      () =>
        new Promise<ChatResponse>((resolve) => {
          resolveChat = resolve;
        }),
    );

    render(<App />);
    await screen.findByText(/人格 YUI/);

    // 会話 A へ送信する。返答はまだ返らない。
    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "さきほどの質問" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    // 待っている間に、履歴から会話 B を開く。
    fireEvent.click(screen.getByRole("button", { name: "会話履歴" }));
    fireEvent.click(await screen.findByRole("button", { name: "開いて続ける" }));
    await screen.findByText("べつの会話の発言");

    // ここで A の返答が返る。
    await act(async () => {
      resolveChat(chatResponse());
    });

    // B の履歴に A のやり取りが混ざらない。
    await waitFor(() =>
      expect(screen.queryByText("さきほどの返答")).not.toBeInTheDocument(),
    );
    expect(screen.queryByText("さきほどの質問")).not.toBeInTheDocument();
    expect(screen.getByText("べつの会話の発言")).toBeInTheDocument();

    // いま開いているのは B のまま。次の送信先が入れ替わらない。
    expect(screen.getByText("会話 #2")).toBeInTheDocument();

    // 返答が消えたわけではないことを画面で伝える。
    expect(screen.getByText(/別の会話へ移ったため/)).toBeInTheDocument();
  });

  it("会話を切り替えなければ、返答をそのまま表示する", async () => {
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse());

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "さきほどの質問" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    expect(await screen.findByText("さきほどの返答")).toBeInTheDocument();
    expect(screen.getByText("さきほどの質問")).toBeInTheDocument();
  });
});

describe("履歴の読み込み中", () => {
  it("読み込みが終わるまで送信できない", async () => {
    // 送信先が決まる前に送ると、開こうとしている会話ではなく
    // 新しい会話へ発言が入ってしまう。
    let resolveDetail: (value: ConversationDetail) => void = () => undefined;
    vi.spyOn(api, "conversation").mockImplementation(
      () =>
        new Promise<ConversationDetail>((resolve) => {
          resolveDetail = resolve;
        }),
    );
    const chat = vi.spyOn(api, "chat").mockResolvedValue(chatResponse());

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.click(screen.getByRole("button", { name: "会話履歴" }));
    fireEvent.click(await screen.findByRole("button", { name: "開いて続ける" }));

    // 読み込み中は入力も送信もできない。
    const composer = screen.getByPlaceholderText("会話を読み込んでいます…");
    expect(composer).toBeDisabled();
    expect(screen.getByRole("button", { name: "送信" })).toBeDisabled();

    // それでも送信が走らないことを確かめる。
    fireEvent.keyDown(composer, { key: "Enter", metaKey: true });
    expect(chat).not.toHaveBeenCalled();

    // 読み込みが終われば送信できる。
    await act(async () => {
      resolveDetail(CONVERSATION_B);
    });
    await screen.findByText("べつの会話の発言");
    expect(screen.getByPlaceholderText(/話しかける/)).not.toBeDisabled();
  });

  it("会話状態の取得が遅れている間も送信できない", async () => {
    // 送信可能になった直後に送ると、その返答の一覧が、開いたときに投げた
    // 古い取得の遅延応答で上書きされることがある（レビューで実測）。
    // 両方が確定するまで読み込み中のままにする。
    let resolveStates: (value: never[]) => void = () => undefined;
    vi.spyOn(api, "conversation").mockResolvedValue(CONVERSATION_B);
    vi.spyOn(api, "conversationStates").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStates = resolve;
        }),
    );

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.click(screen.getByRole("button", { name: "会話履歴" }));
    fireEvent.click(await screen.findByRole("button", { name: "開いて続ける" }));

    // 会話状態の取得が終わるまでは、発言も出さず読み込み中のままにする
    // （両方が確定してから一度に反映する。片方だけ先に出すと、送信直後の
    // 一覧が遅れて届いた古い取得で上書きされる余地が残る）。
    expect(screen.getByPlaceholderText(/会話を読み込んでいます/)).toBeDisabled();
    expect(screen.queryByText("べつの会話の発言")).not.toBeInTheDocument();

    await act(async () => {
      resolveStates([]);
    });
    await screen.findByText("べつの会話の発言");
    expect(screen.getByPlaceholderText(/話しかける/)).not.toBeDisabled();
  });
});

/** 音声を鳴らさずに、再生の流れだけを追えるようにする。 */
class FakeAudio {
  static instances: FakeAudio[] = [];
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(readonly src: string) {
    FakeAudio.instances.push(this);
  }
  play(): Promise<void> {
    return Promise.resolve();
  }
  pause(): void {}
}

describe("再生の記録の反映", () => {
  it("開始の通知が遅れて届いても、話し終えた表示が再生中へ戻らない", async () => {
    // 通知の応答は、サーバーが作った時点の確定状態。開始の応答だけ遅れると
    // 順番が入れ替わって届く。
    FakeAudio.instances = [];
    vi.stubGlobal("Audio", FakeAudio);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: () => "blob:test",
      revokeObjectURL: () => undefined,
    });

    vi.spyOn(api, "health").mockResolvedValue({
      ok: true,
      persona: { name: "YUI", version: "test" },
      llm: { ok: true, provider: "fake", model: "fake" },
      voice: { ok: true, enabled: true, provider: "fake", speaker: 1 },
    });
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse());
    vi.spyOn(api, "speech").mockResolvedValue(
      new Blob([new Uint8Array([1])], { type: "audio/wav" }),
    );

    let resolvePlaying: (value: Message) => void = () => undefined;
    vi.spyOn(api, "notifyDelivery").mockImplementation(
      (messageId: number, state: DeliveryNotice) => {
        const reply = (delivery: DeliveryState): Message => ({
          ...message(messageId, "character", "さきほどの返答"),
          delivery_state: delivery,
        });
        if (state === "playing") {
          return new Promise<Message>((resolve) => {
            resolvePlaying = () => resolve(reply("playing"));
          });
        }
        return Promise.resolve(reply(state));
      },
    );

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "さきほどの質問" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));
    // 会話欄と字幕の両方に出る。
    await screen.findAllByText("さきほどの返答");
    await waitFor(() => expect(FakeAudio.instances).toHaveLength(1));

    // 最後まで鳴り、完了の応答が先に届く。
    await act(async () => {
      FakeAudio.instances[0].onended?.();
    });
    await waitFor(() => expect(screen.queryByText("再生中")).not.toBeInTheDocument());

    // 遅れて開始の応答が届く。
    await act(async () => {
      resolvePlaying(message(21, "character", "さきほどの返答"));
    });

    expect(screen.queryByText("再生中")).not.toBeInTheDocument();
    expect(screen.queryByText("生成のみ")).not.toBeInTheDocument();
  });

  it("進んだ状態からは戻さない", () => {
    // 開始の通知の応答だけ遅れて届くと、話し終えた表示が再生中へ戻る。
    expect(shouldApplyDelivery("completed", "playing")).toBe(false);
    expect(shouldApplyDelivery("aborted", "playing")).toBe(false);
    expect(shouldApplyDelivery("completed", "generated")).toBe(false);
  });

  it("進む更新は反映する", () => {
    expect(shouldApplyDelivery("generated", "playing")).toBe(true);
    expect(shouldApplyDelivery("playing", "completed")).toBe(true);
    expect(shouldApplyDelivery("playing", "aborted")).toBe(true);
    // 同じ状態の通知は反映してよい（時刻が埋まることがある）。
    expect(shouldApplyDelivery("playing", "playing")).toBe(true);
  });
});

describe("誰として話すか", () => {
  /** 会話欄に出ている発言者の名前。選択肢の名前と混ざらないように絞る。 */
  const shownNames = () =>
    Array.from(document.querySelectorAll(".speaker-name")).map(
      (el) => el.textContent,
    );

  it("選んだ相手として送り、発言者の名前を会話欄に出す", async () => {
    // alice はまだ保存されていない。初回の送信で作られる。
    vi.spyOn(api, "speakers").mockResolvedValue([
      {
        id: 1,
        source: "local",
        external_id: "developer",
        display_name: "shiki",
      },
    ]);
    const chat = vi.spyOn(api, "chat").mockResolvedValue({
      ...chatResponse(),
      user_message: { ...message(20, "user", "はじめまして"), speaker_id: 2 },
    });

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "alice" } });
    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "はじめまして" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    await screen.findByText("はじめまして");
    // 既定の相手ではなく、選んだ相手として送る。
    expect(chat).toHaveBeenCalledWith("はじめまして", null, {
      source: "local",
      external_id: "alice",
      display_name: "alice",
    });
    // 初めて話した相手も、名前で表示できる。
    await waitFor(() => expect(shownNames()).toEqual(["alice"]));
  });

  it("保存済みの会話でも、発言ごとに誰の発言かを出す", async () => {
    vi.spyOn(api, "speakers").mockResolvedValue([
      {
        id: 1,
        source: "local",
        external_id: "developer",
        display_name: "shiki",
      },
      {
        id: 2,
        source: "local",
        external_id: "bob",
        display_name: "bob",
      },
    ]);
    vi.spyOn(api, "conversation").mockResolvedValue({
      ...CONVERSATIONS[0],
      messages: [
        message(10, "user", "shiki の発言"),
        { ...message(11, "user", "bob の発言"), speaker_id: 2 },
      ],
    });

    render(<App />);
    await screen.findByText(/人格 YUI/);
    fireEvent.click(screen.getByRole("button", { name: "会話履歴" }));
    fireEvent.click(await screen.findByRole("button", { name: "開いて続ける" }));

    await screen.findByText("bob の発言");
    // 同じ会話に2人いても、どちらの発言かが順に読める。
    await waitFor(() => expect(shownNames()).toEqual(["shiki", "bob"]));
  });
});

function conversationState(
  overrides: Partial<ConversationStateRecord>,
): ConversationStateRecord {
  return {
    id: 100,
    conversation_id: 1,
    kind: "question_to_yui",
    content: "土曜と日曜、どちらが空いてる？",
    speaker_id: 1,
    target_speaker_id: 1,
    source_message_id: 20,
    ref_kind: null,
    ref_id: null,
    status: "open",
    withdraw_reason: null,
    asked_message_id: null,
    responded_message_id: null,
    judged_message_id: null,
    followup_needed: false,
    resolved_message_id: null,
    detected_by: "rule",
    decided_by: null,
    created_at: "2026-09-07T00:00:00Z",
    updated_at: "2026-09-07T00:00:00Z",
    ...overrides,
  };
}

describe("v0.2：この会話での表示", () => {
  it("答えていない質問を、状態の種類の名前を出さずに一文で示す", async () => {
    vi.spyOn(api, "chat").mockResolvedValue({
      ...chatResponse(),
      conversation_states: [conversationState({})],
    });

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "土曜と日曜、どちらが空いてる？" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    expect(
      await screen.findByText(/まだ答えていない質問があります/),
    ).toBeInTheDocument();
    // 種類の名前（question_to_yui）はどこにも出さない。
    expect(screen.queryByText(/question_to_yui/)).not.toBeInTheDocument();
  });

  it("答え損ねた質問は、未回答とは別の一文で示す", async () => {
    vi.spyOn(api, "chat").mockResolvedValue({
      ...chatResponse(),
      conversation_states: [conversationState({ followup_needed: true })],
    });

    render(<App />);
    await screen.findByText(/人格 YUI/);
    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "土曜と日曜、どちらが空いてる？" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));

    expect(
      await screen.findByText(/答えきれていない質問があります/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^まだ答えていない質問/)).not.toBeInTheDocument();
  });
});

describe("v0.2：会話終了後の訂正の候補", () => {
  it("correction・discrepancy を読み取り専用で示し、採用ボタンは付けない", async () => {
    vi.spyOn(api, "endConversation").mockResolvedValue([]);
    vi.spyOn(api, "conversationStates").mockImplementation(async (_id, filter) => {
      if (filter?.kind === "correction,discrepancy") {
        return [
          conversationState({
            id: 200,
            kind: "correction",
            content: "日曜",
            status: "expired",
          }),
        ];
      }
      return [];
    });
    vi.spyOn(api, "chat").mockResolvedValue(chatResponse());

    render(<App />);
    await screen.findByText(/人格 YUI/);

    fireEvent.change(screen.getByPlaceholderText(/話しかける/), {
      target: { value: "さきほどの質問" },
    });
    fireEvent.click(screen.getByRole("button", { name: "送信" }));
    await screen.findByText("さきほどの返答");

    fireEvent.click(screen.getByRole("button", { name: "終了して振り返る" }));
    await screen.findByRole("button", { name: /記憶の候補/ });

    expect(await screen.findByText("この会話であった訂正の候補")).toBeInTheDocument();
    expect(screen.getByText("日曜")).toBeInTheDocument();
    expect(screen.getByText("明示的な訂正")).toBeInTheDocument();
    // 読み取り専用。記憶の候補のような採用・却下は付けない。
    expect(screen.queryByRole("button", { name: "採用" })).not.toBeInTheDocument();
  });
});
