import { useEffect, useState } from "react";
import { api } from "../api";
import type { ConversationStateRecord, MemoryCandidate } from "../types";
import { CERTAINTY_LABEL, KIND_LABEL } from "../types";
import { SourceMessage } from "./SourceMessage";

interface Props {
  candidates: MemoryCandidate[];
  /** 直近に開いた会話。終了後、訂正の候補を読むために使う。 */
  conversationId: number | null;
  conversationEnded: boolean;
  onDecided: (candidate: MemoryCandidate) => void;
}

/** 会話の振り返りで出た記憶候補。採用するまで長期記憶にはしない。 */
export function CandidatePanel({
  candidates,
  conversationId,
  conversationEnded,
  onDecided,
}: Props) {
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
      {conversationEnded && conversationId !== null && (
        <CorrectionCandidates conversationId={conversationId} />
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
        {candidate.provenance === "hearsay" && (
          <span className="tag subtle">人づてに聞いた</span>
        )}
        <span className="muted small">会話 #{candidate.conversation_id}</span>
      </div>
      {candidate.similar_memory_ids && candidate.similar_memory_ids.length > 0 && (
        // 同じ出来事を二重に覚えないための手がかり。自動では捨てないので、
        // 採用するか、既存の記憶を訂正するかは開発者が決める。
        <p className="tag warn">
          既存の記憶と近い（
          {candidate.similar_memory_ids.map((id) => `#${id}`).join("、")}
          ）。二重に覚えるか、既存を訂正するか確かめてください。
        </p>
      )}
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

/**
 * v0.2：会話終了後の訂正の候補（計画 §5・§6）。`correction`（明示的な訂正）と
 * 未解決の `discrepancy`（訂正か不明な食い違い）を読み取り専用で見せる。
 * **採用ボタンは付けない**——記憶の訂正は既存の MemoryPanel から行う。
 * `/end` はすべての開いている状態を `expired` にするため、ここでは
 * `open`／`expired` の両方を対象にする（`resolved`／`withdrawn` は
 * 開発者がすでに判断済みなので出さない）。
 */
function CorrectionCandidates({ conversationId }: { conversationId: number }) {
  const [states, setStates] = useState<ConversationStateRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStates(null);
    api
      .conversationStates(conversationId, { kind: "correction,discrepancy" })
      .then((all) => {
        if (!cancelled) {
          setStates(all.filter((s) => s.status === "open" || s.status === "expired"));
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  if (error) return null;
  if (states === null || states.length === 0) return null;

  return (
    <div className="memory-list correction-candidates">
      <h3 className="muted small">この会話であった訂正の候補</h3>
      {states.map((s) => (
        <div className="memory" key={s.id}>
          <div className="memory-head">
            <span className="tag subtle">
              {s.kind === "correction" ? "明示的な訂正" : "訂正かどうか不明な食い違い"}
            </span>
          </div>
          <p>{s.content}</p>
          <div className="muted small">
            <SourceMessage messageId={s.source_message_id} />
          </div>
        </div>
      ))}
    </div>
  );
}
