import { useEffect, useState } from "react";
import { api } from "../api";
import type { Conversation } from "../types";
import {
  CONVERSATION_STATE_LABEL,
  conversationState,
  formatDateTime,
} from "../types";

interface Props {
  /** 会話の開始・終了で数を上げ、一覧を取り直す。 */
  refreshKey: number;
  /** いま開いている会話。一覧のどれを見ているかを示す。 */
  currentId: number | null;
  onOpen: (conversationId: number) => void;
}

/** 過去の会話の一覧。保存した会話履歴を読み直すための入り口。 */
export function HistoryPanel({ refreshKey, currentId, onOpen }: Props) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .conversations()
      .then((list) => {
        setConversations(list);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [refreshKey]);

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>会話履歴</h2>
        <span className="muted small">新しい順に最大50件</span>
      </header>

      {error && <p className="error">{error}</p>}

      <div className="memory-list">
        {conversations.length === 0 && !error && (
          <p className="muted center">まだ会話がありません。</p>
        )}
        {conversations.map((conversation) => {
          const state = conversationState(conversation);
          return (
            <div
              key={conversation.id}
              className={`history ${conversation.id === currentId ? "current" : ""}`}
            >
              <div className="memory-head">
                <span className={`tag ${state === "open" ? "" : "subtle"}`}>
                  {CONVERSATION_STATE_LABEL[state]}
                </span>
                <span className="muted small">#{conversation.id}</span>
                <span className="muted small">
                  {formatDateTime(conversation.started_at)}
                </span>
              </div>
              <p className="memory-content">
                {conversation.title ?? <span className="muted">（無題）</span>}
              </p>
              <div className="row">
                <button
                  type="button"
                  className="link"
                  onClick={() => onOpen(conversation.id)}
                  disabled={conversation.id === currentId}
                >
                  {conversation.id === currentId
                    ? "表示中"
                    : state === "open"
                      ? "開いて続ける"
                      : "開く（読み取り専用）"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
