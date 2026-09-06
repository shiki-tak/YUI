import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChatResponse, MemoryCandidate, RunRecord } from "../types";
import { CERTAINTY_LABEL, KIND_LABEL } from "../types";
import { SourceMessage } from "./SourceMessage";

interface Props {
  conversationId: number | null;
  entries: ChatResponse[];
  onEntry: (entry: ChatResponse) => void;
  onCandidates: (candidates: MemoryCandidate[]) => void;
  onReset: () => void;
}

export function ChatPanel({
  conversationId,
  entries,
  onEntry,
  onCandidates,
  onReset,
}: Props) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length, busy]);

  async function send() {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const entry = await api.chat(trimmed, conversationId);
      onEntry(entry);
      setText("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function endConversation() {
    if (conversationId === null || busy) return;
    setBusy(true);
    setError(null);
    try {
      onCandidates(await api.endConversation(conversationId));
      onReset();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel chat">
      <header className="panel-header">
        <h2>会話</h2>
        <span className="muted">
          {conversationId === null ? "新しい会話" : `会話 #${conversationId}`}
        </span>
        <button
          type="button"
          onClick={endConversation}
          disabled={conversationId === null || busy}
          title="会話を終了し、長期記憶の候補を抽出します"
        >
          終了して振り返る
        </button>
      </header>

      <div className="messages">
        {entries.length === 0 && (
          <p className="muted center">まだ会話がありません。話しかけてください。</p>
        )}
        {entries.map((entry) => (
          <Turn key={entry.reply.id} entry={entry} />
        ))}
        {busy && <p className="muted center">考えています…</p>}
        <div ref={bottomRef} />
      </div>

      {error && <p className="error">{error}</p>}

      <div className="composer">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void send();
          }}
          placeholder="話しかける（⌘/Ctrl + Enter で送信）"
          rows={3}
          disabled={busy}
        />
        <button type="button" onClick={send} disabled={busy || !text.trim()}>
          送信
        </button>
      </div>
    </section>
  );
}

function Turn({ entry }: { entry: ChatResponse }) {
  const [showBasis, setShowBasis] = useState(false);
  const [showIdeal, setShowIdeal] = useState(false);

  return (
    <div className="turn">
      <div className="bubble user">{entry.user_message.content}</div>
      <div className="bubble character">{entry.reply.content}</div>
      <div className="turn-actions">
        <button type="button" className="link" onClick={() => setShowBasis((v) => !v)}>
          {showBasis ? "根拠を隠す" : `根拠（記憶 ${entry.used_memories.length} 件）`}
        </button>
        <button type="button" className="link" onClick={() => setShowIdeal((v) => !v)}>
          理想の返答を記録
        </button>
        <span className="muted small">
          {entry.run.model} / {entry.run.latency_ms ?? "-"} ms
        </span>
      </div>
      {showBasis && <Basis entry={entry} />}
      {showIdeal && <IdealForm messageId={entry.reply.id} />}
    </div>
  );
}

function Basis({ entry }: { entry: ChatResponse }) {
  const [run, setRun] = useState<RunRecord | null>(null);

  useEffect(() => {
    api.run(entry.reply.id).then(setRun).catch(() => setRun(null));
  }, [entry.reply.id]);

  return (
    <div className="basis">
      <h4>渡した記憶</h4>
      {entry.used_memories.length === 0 ? (
        <p className="muted small">この返答に記憶は渡していません。</p>
      ) : (
        <ul>
          {entry.used_memories.map((item) => (
            <li key={item.memory.id}>
              <span className="tag">{KIND_LABEL[item.memory.kind]}</span>
              <span className="tag subtle">{CERTAINTY_LABEL[item.memory.certainty]}</span>
              {item.memory.content}
              <div className="muted small">
                #{item.memory.id} · {item.reason} · 点数 {item.score}
                {item.memory.source_message_id !== null ? (
                  <>
                    {" · "}
                    <SourceMessage messageId={item.memory.source_message_id} />
                  </>
                ) : (
                  <span className="tag warn"> 根拠未確認</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      <h4>実行記録</h4>
      <div className="muted small">
        {entry.run.provider} / {entry.run.model}
        {entry.run.model_digest && ` (${entry.run.model_digest.slice(0, 16)}…)`} ·
        トークン {entry.run.prompt_tokens ?? "-"} / {entry.run.completion_tokens ?? "-"} ·
        設定 {JSON.stringify(entry.run.options ?? {})}
      </div>
      {run?.system_prompt && (
        <details>
          <summary>実際に渡したプロンプト</summary>
          <pre>{run.system_prompt}</pre>
        </details>
      )}
    </div>
  );
}

function IdealForm({ messageId }: { messageId: number }) {
  const [ideal, setIdeal] = useState("");
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    if (!ideal.trim()) return;
    try {
      await api.saveIdeal(messageId, ideal.trim(), note.trim() || null);
      setSaved(true);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (saved) return <p className="muted small">理想の返答を記録しました。</p>;

  return (
    <div className="ideal-form">
      <textarea
        value={ideal}
        onChange={(e) => setIdeal(e.target.value)}
        placeholder="こう返してほしかった、という文章"
        rows={2}
      />
      <input
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="何が問題だったか（任意）"
      />
      {error && <p className="error small">{error}</p>}
      <button type="button" onClick={save} disabled={!ideal.trim()}>
        記録する
      </button>
    </div>
  );
}
