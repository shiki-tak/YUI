import { useState } from "react";
import { api } from "../api";
import type { Message } from "../types";

/** 記憶の根拠になった発言。番号だけでなく本文まで辿れるようにする。 */
export function SourceMessage({ messageId }: { messageId: number }) {
  const [message, setMessage] = useState<Message | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function toggle() {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (message !== null) return;
    try {
      setMessage(await api.message(messageId));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <span className="source-message">
      <button type="button" className="link" onClick={toggle}>
        根拠の発言 #{messageId}
      </button>
      {open && (
        <span className="source-message-body">
          {error ? (
            <span className="error small">{error}</span>
          ) : message ? (
            <>
              「{message.content}」
              <span className="muted small">
                （{message.speaker_kind === "character" ? "ゆい" : "相手"}の発言）
              </span>
            </>
          ) : (
            <span className="muted small">読み込み中…</span>
          )}
        </span>
      )}
    </span>
  );
}
