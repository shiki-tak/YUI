import { useState } from "react";
import { api } from "../api";
import type { MemoryCandidate } from "../types";
import { CERTAINTY_LABEL, KIND_LABEL } from "../types";
import { SourceMessage } from "./SourceMessage";

interface Props {
  candidates: MemoryCandidate[];
  onDecided: (candidate: MemoryCandidate) => void;
}

/** 会話の振り返りで出た記憶候補。採用するまで長期記憶にはしない。 */
export function CandidatePanel({ candidates, onDecided }: Props) {
  const pending = candidates.filter((c) => c.status === "pending");

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>記憶の候補</h2>
        <span className="muted small">採用したものだけが長期記憶になります</span>
      </header>
      {pending.length === 0 ? (
        <p className="muted center">
          未判断の候補はありません。会話を終了すると候補が出ます。
        </p>
      ) : (
        <div className="memory-list">
          {pending.map((candidate) => (
            <CandidateRow key={candidate.id} candidate={candidate} onDecided={onDecided} />
          ))}
        </div>
      )}
    </section>
  );
}

interface RowProps {
  candidate: MemoryCandidate;
  onDecided: (candidate: MemoryCandidate) => void;
}

function CandidateRow({ candidate, onDecided }: RowProps) {
  const [content, setContent] = useState(candidate.content);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(decision: "accept" | "reject") {
    setBusy(true);
    try {
      const updated = await api.decideCandidate(candidate.id, {
        decision,
        ...(decision === "accept" && content !== candidate.content ? { content } : {}),
      });
      onDecided(updated);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="memory">
      <div className="memory-head">
        <span className="tag">{KIND_LABEL[candidate.kind]}</span>
        <span className="tag subtle">{CERTAINTY_LABEL[candidate.certainty]}</span>
        <span className="muted small">会話 #{candidate.conversation_id}</span>
      </div>
      <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={2} />
      <div className="muted small">
        {candidate.keywords && `キーワード: ${candidate.keywords}`}
        {candidate.source_message_id !== null ? (
          <>
            {candidate.keywords && " · "}
            <SourceMessage messageId={candidate.source_message_id} />
          </>
        ) : (
          <span className="tag warn"> 根拠未確認</span>
        )}
      </div>
      {error && <p className="error small">{error}</p>}
      <div className="row">
        <button type="button" onClick={() => decide("accept")} disabled={busy}>
          採用
        </button>
        <button type="button" className="link" onClick={() => decide("reject")} disabled={busy}>
          却下
        </button>
      </div>
    </div>
  );
}
