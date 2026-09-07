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

export interface Speaker {
  id: number;
  source: string;
  external_id: string;
  display_name: string;
}

/** 発話の状態。生成しただけの文章と、実際に話し終えた内容を区別する。 */
export type DeliveryState = "generated" | "playing" | "completed" | "aborted";

/** 再生側から通知できる状態。generated は生成時の状態なので送らない。 */
export type DeliveryNotice = "playing" | "completed" | "aborted";

export interface Message {
  id: number;
  conversation_id: number;
  speaker_kind: "user" | "character";
  speaker_id: number | null;
  source: string;
  content: string;
  delivery_state: DeliveryState;
  delivery_started_at: string | null;
  delivery_finished_at: string | null;
  created_at: string;
}

export interface Conversation {
  id: number;
  mode: string;
  title: string | null;
  started_at: string;
  ended_at: string | null;
  reflection_started_at: string | null;
  reflection_completed_at: string | null;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface Memory {
  id: number;
  kind: MemoryKind;
  content: string;
  subject_speaker_id: number | null;
  visible_to_speaker_id: number | null;
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
  /** 生成に使った固定人格の版。版を変えた前後を見分けるために出す。 */
  persona_version: string | null;
  options: Record<string, unknown> | null;
  referenced_memory_ids: number[] | null;
  /** 返答に渡した可変状態。記憶と分けて残す。 */
  referenced_state_ids: number[] | null;
  retrieval_ms: number | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  created_at: string;
  system_prompt?: string | null;
}

/** 音声合成の実行記録。待ち時間の内訳を見るために使う。 */
export interface SpeechRun {
  id: number;
  message_id: number;
  provider: string;
  speaker_id: number;
  engine_version: string | null;
  query_ms: number | null;
  synthesis_ms: number | null;
  audio_ms: number | null;
  byte_size: number | null;
  created_at: string;
}

export interface ChatResponse {
  conversation_id: number;
  user_message: Message;
  reply: Message;
  run: RunRecord;
  used_memories: RetrievedMemory[];
}

export type Provenance = "firsthand" | "hearsay" | "unknown";

export interface MemoryCandidate {
  id: number;
  conversation_id: number;
  kind: MemoryKind;
  content: string;
  /** どうやって知ったか。伝聞を本人の発言と区別する。 */
  provenance: Provenance;
  subject_speaker_id: number | null;
  visible_to_speaker_id: number | null;
  certainty: Certainty;
  visibility: Visibility;
  keywords: string;
  source_message_id: number | null;
  /** 内容が近い既存の記憶。二重に覚えないための手がかりとして出す。 */
  similar_memory_ids: number[] | null;
  status: "pending" | "accepted" | "rejected";
  accepted_memory_id: number | null;
  created_at: string;
}

/** 変化する状態：YUI の関心と、相手との関係。固定人格とは別に扱う。 */
export interface CharacterState {
  id: number;
  kind: "interest" | "relationship";
  subject_speaker_id: number | null;
  topic: string | null;
  content: string;
  /** 根拠にした記憶。訂正・削除されると要確認の印が付く。 */
  basis_memory_ids: number[] | null;
  needs_review: boolean;
  review_reason: string | null;
  status: "pending" | "active" | "rejected" | "superseded" | "withdrawn";
  visibility: Visibility;
  visible_to_speaker_id: number | null;
  superseded_by_id: number | null;
  source_conversation_id: number | null;
  created_at: string;
  updated_at: string;
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
  voice: {
    ok: boolean;
    enabled: boolean;
    provider: string;
    engine_version?: string | null;
    speaker?: number;
    /** 音声を公開する場に出す表記。 */
    credit?: string;
    error?: string;
  };
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

/** この画面を使っている相手。表示名ではなく source と external_id で同定する。 */
export const SELF_SPEAKER = {
  source: "local",
  external_id: "developer",
  display_name: "開発者",
} as const;

/** 会話が続けられるか。終了済み・振り返り中は読み取り専用として開く。 */
export type ConversationState = "open" | "reflecting" | "ended";

export function conversationState(
  conversation: Conversation,
): ConversationState {
  if (conversation.reflection_completed_at !== null || conversation.ended_at !== null) {
    return "ended";
  }
  if (conversation.reflection_started_at !== null) return "reflecting";
  return "open";
}

export const CONVERSATION_STATE_LABEL: Record<ConversationState, string> = {
  open: "続きを話せる",
  reflecting: "振り返り中",
  ended: "終了",
};

/** API が返す UTC の日時を、この端末の時刻で表示する。 */
export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export const DELIVERY_LABEL: Record<DeliveryState, string> = {
  generated: "生成のみ",
  playing: "再生中",
  completed: "話し終えた",
  aborted: "中断",
};

/**
 * 状態の進み具合。generated → playing → completed／aborted と一方向に進む。
 * completed と aborted はどちらも終わりで、そこから戻らない。
 */
const DELIVERY_RANK: Record<DeliveryState, number> = {
  generated: 0,
  playing: 1,
  completed: 2,
  aborted: 2,
};

/**
 * 受け取った状態を表示へ反映してよいか。
 *
 * 通知の応答は、サーバーが作った時点の確定状態である。開始の通知が先に
 * 反映され、その応答だけ遅れて届くと、すでに話し終えた表示が「再生中」へ
 * 戻ってしまう。進んだ状態からは戻さない。
 */
export function shouldApplyDelivery(
  current: DeliveryState,
  incoming: DeliveryState,
): boolean {
  return DELIVERY_RANK[incoming] >= DELIVERY_RANK[current];
}
