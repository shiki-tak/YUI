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
import App from "./App";
import { api } from "./api";
import type { ChatResponse, Conversation, ConversationDetail, Message } from "./types";

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
      options: null,
      referenced_memory_ids: [],
      retrieval_ms: 1,
      latency_ms: 10,
      prompt_tokens: null,
      completion_tokens: null,
      created_at: "2026-09-07T00:00:00Z",
    },
    used_memories: [],
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
