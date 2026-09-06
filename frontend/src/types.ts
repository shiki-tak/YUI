// バックエンド（app/schemas.py）と対応する型。

export type MemoryKind =
  | "experience"
  | "about_person"
  | "promise"
  | "impression"
  | "fact";

export type Certainty = "fact" | "inference";
export type Visibility = "private" | "public";
export type MemoryStatus = "active" | "corrected" | "deleted";

export interface Message {
  id: number;
  conversation_id: number;
  speaker_kind: "user" | "character";
  speaker_id: number | null;
  source: string;
  content: string;
  delivery_state: string;
  created_at: string;
}

export interface Memory {
  id: number;
  kind: MemoryKind;
  content: string;
  subject_speaker_id: number | null;
  certainty: Certainty;
  visibility: Visibility;
  status: MemoryStatus;
  keywords: string;
  occurred_at: string | null;
  source_message_id: number | null;
  source_conversation_id: number | null;
  superseded_by_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface RetrievedMemory {
  memory: Memory;
  score: number;
  reason: string;
}

export interface RunRecord {
  id: number;
  message_id: number;
  provider: string;
  model: string;
  model_digest: string | null;
  options: Record<string, unknown> | null;
  referenced_memory_ids: number[] | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  created_at: string;
  system_prompt?: string | null;
}

export interface ChatResponse {
  conversation_id: number;
  user_message: Message;
  reply: Message;
  run: RunRecord;
  used_memories: RetrievedMemory[];
}

export interface MemoryCandidate {
  id: number;
  conversation_id: number;
  kind: MemoryKind;
  content: string;
  subject_speaker_id: number | null;
  certainty: Certainty;
  visibility: Visibility;
  keywords: string;
  source_message_id: number | null;
  status: "pending" | "accepted" | "rejected";
  accepted_memory_id: number | null;
  created_at: string;
}

export interface MemoryRevision {
  id: number;
  memory_id: number;
  action: string;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  reason: string | null;
  created_at: string;
}

export interface IdealResponse {
  id: number;
  message_id: number;
  ideal_text: string;
  note: string | null;
  created_at: string;
}

export interface Health {
  ok: boolean;
  persona: { name: string; version: string };
  llm: { ok: boolean; provider: string; model?: string; error?: string };
}

export const KIND_LABEL: Record<MemoryKind, string> = {
  experience: "経験",
  about_person: "相手について",
  promise: "約束",
  impression: "受け止め方",
  fact: "確認した事実",
};

export const CERTAINTY_LABEL: Record<Certainty, string> = {
  fact: "確かなこと",
  inference: "推測",
};

export const VISIBILITY_LABEL: Record<Visibility, string> = {
  private: "非公開",
  public: "配信で使える",
};
