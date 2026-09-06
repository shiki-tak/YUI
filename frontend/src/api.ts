// FastAPI への呼び出し。開発中は Vite の proxy 経由で同一オリジンになる。

import { SELF_SPEAKER } from "./types";
import type {
  ChatResponse,
  Health,
  IdealResponse,
  Memory,
  MemoryCandidate,
  MemoryRevision,
  Message,
  RetrievedMemory,
  RunRecord,
  Speaker,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // JSON でない応答はそのまま扱う。
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  chat: (text: string, conversationId: number | null) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      // 誰として話しているかを明示する。バックエンドの既定に頼らない。
      body: JSON.stringify({
        text,
        conversation_id: conversationId,
        speaker: SELF_SPEAKER,
      }),
    }),

  speakers: () => request<Speaker[]>("/speakers"),

  message: (messageId: number) =>
    request<Message>(`/conversations/messages/${messageId}`),

  endConversation: (conversationId: number) =>
    request<MemoryCandidate[]>(`/conversations/${conversationId}/end`, {
      method: "POST",
    }),

  candidates: (conversationId: number) =>
    request<MemoryCandidate[]>(`/conversations/${conversationId}/candidates`),

  decideCandidate: (
    candidateId: number,
    body: { decision: "accept" | "reject"; content?: string; reason?: string },
  ) =>
    request<MemoryCandidate>(`/conversations/candidates/${candidateId}/decide`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  run: (messageId: number) =>
    request<RunRecord>(`/conversations/messages/${messageId}/run`),

  saveIdeal: (messageId: number, idealText: string, note: string | null) =>
    request<IdealResponse>(`/conversations/messages/${messageId}/ideal`, {
      method: "POST",
      body: JSON.stringify({ ideal_text: idealText, note }),
    }),

  memories: (includeInactive: boolean) =>
    request<Memory[]>(`/memories?include_inactive=${includeInactive}`),

  // 会話と同じ経路で検索するため、相手のIDを必ず渡す。
  searchMemories: (q: string, speakerId: number | null) => {
    const params = new URLSearchParams({ q });
    if (speakerId !== null) params.set("speaker_id", String(speakerId));
    return request<{ query: string; results: RetrievedMemory[] }>(
      `/memories/search?${params.toString()}`,
    );
  },

  correctMemory: (
    memoryId: number,
    body: { content?: string; keywords?: string; reason?: string },
  ) =>
    request<Memory>(`/memories/${memoryId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteMemory: (memoryId: number, reason: string) =>
    request<Memory>(`/memories/${memoryId}?reason=${encodeURIComponent(reason)}`, {
      method: "DELETE",
    }),

  restoreMemory: (memoryId: number) =>
    request<Memory>(`/memories/${memoryId}/restore`, { method: "POST" }),

  revisions: (memoryId: number) =>
    request<MemoryRevision[]>(`/memories/${memoryId}/revisions`),
};
