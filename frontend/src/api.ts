// FastAPI への呼び出し。開発中は Vite の proxy 経由で同一オリジンになる。

import { SELF_SPEAKER } from "./types";
import type {
  ChatResponse,
  DeliveryNotice,
  Conversation,
  ConversationDetail,
  ConversationStateRecord,
  Health,
  IdealResponse,
  Memory,
  MemoryCandidate,
  MemoryRevision,
  Message,
  CharacterState,
  RetrievedMemory,
  RunRecord,
  Speaker,
  SpeakerRef,
  SpeechRun,
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

  chat: (
    text: string,
    conversationId: number | null,
    speaker: SpeakerRef = SELF_SPEAKER,
  ) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      // 誰として話しているかを明示する。バックエンドの既定に頼らない。
      body: JSON.stringify({
        text,
        conversation_id: conversationId,
        speaker,
      }),
    }),

  speakers: () => request<Speaker[]>("/speakers"),

  message: (messageId: number) =>
    request<Message>(`/conversations/messages/${messageId}`),

  conversations: (limit = 50) =>
    request<Conversation[]>(`/conversations?limit=${limit}`),

  conversation: (conversationId: number) =>
    request<ConversationDetail>(`/conversations/${conversationId}`),

  endConversation: (conversationId: number) =>
    request<MemoryCandidate[]>(`/conversations/${conversationId}/end`, {
      method: "POST",
    }),

  candidates: (conversationId: number) =>
    request<MemoryCandidate[]>(`/conversations/${conversationId}/candidates`),

  // v0.2：今の用件・未回答の質問・延期・終了・食い違い。target_speaker_id を
  // 指定しないと、会話にいる全員宛の状態が返る（/chat の応答は話している
  // 相手だけに絞っているため、同じ絞り方にしたい場合は指定する）。
  conversationStates: (
    conversationId: number,
    filter?: { status?: string; kind?: string; target_speaker_id?: number },
  ) => {
    const params = new URLSearchParams();
    if (filter?.status) params.set("status", filter.status);
    if (filter?.kind) params.set("kind", filter.kind);
    if (filter?.target_speaker_id !== undefined) {
      params.set("target_speaker_id", String(filter.target_speaker_id));
    }
    const query = params.toString();
    return request<ConversationStateRecord[]>(
      `/conversations/${conversationId}/states${query ? `?${query}` : ""}`,
    );
  },

  decideConversationState: (
    conversationId: number,
    stateId: number,
    body: { decision: "resolved" | "withdrawn"; withdraw_reason?: string },
  ) =>
    request<ConversationStateRecord>(
      `/conversations/${conversationId}/states/${stateId}/decide`,
      { method: "POST", body: JSON.stringify(body) },
    ),

  // 再読み込みしても採用できるよう、未判断の候補を会話をまたいで取り直す。
  pendingCandidates: () =>
    request<MemoryCandidate[]>("/conversations/candidates/pending"),

  decideCandidate: (
    candidateId: number,
    body: { decision: "accept" | "reject"; content?: string; reason?: string },
  ) =>
    request<MemoryCandidate>(`/conversations/candidates/${candidateId}/decide`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** 返答を読み上げた音声。JSON ではないので request() を通さない。 */
  speech: async (messageId: number): Promise<Blob> => {
    const response = await fetch(
      `/api/conversations/messages/${messageId}/speech`,
    );
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
    return await response.blob();
  },

  // 再生の開始・完了・中断を記録する。聞き直しでは記録は変わらない。
  // progress（0〜1）は中断のときだけ使う。再生位置から測った比率で、
  // サーバー側が届いた文字数の近似に使う（ISSUE-051）。
  notifyDelivery: (messageId: number, state: DeliveryNotice, progress?: number) =>
    request<Message>(`/conversations/messages/${messageId}/delivery`, {
      method: "POST",
      body: JSON.stringify(progress === undefined ? { state } : { state, progress }),
    }),

  run: (messageId: number) =>
    request<RunRecord>(`/conversations/messages/${messageId}/run`),

  speechRuns: (messageId: number) =>
    request<SpeechRun[]>(`/conversations/messages/${messageId}/speech-runs`),

  saveIdeal: (messageId: number, idealText: string, note: string | null) =>
    request<IdealResponse>(`/conversations/messages/${messageId}/ideal`, {
      method: "POST",
      body: JSON.stringify({ ideal_text: idealText, note }),
    }),

  // 過去の返答が参照した記憶を ID から引く。訂正・削除済みでも返る。
  memory: (memoryId: number) => request<Memory>(`/memories/${memoryId}`),

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

  // 変化する状態（関心・相手との関係）。採用したものだけが会話に渡る。
  states: () => request<CharacterState[]>("/states"),

  decideState: (stateId: number, body: { decision: "accept" | "reject"; reason?: string }) =>
    request<CharacterState>(`/states/${stateId}/decide`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  updateState: (
    stateId: number,
    body: { content?: string; status?: "active" | "withdrawn"; reviewed?: boolean; reason?: string },
  ) =>
    request<CharacterState>(`/states/${stateId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  revisions: (memoryId: number) =>
    request<MemoryRevision[]>(`/memories/${memoryId}/revisions`),
};
