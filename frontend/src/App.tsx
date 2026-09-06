import { useEffect, useState } from "react";
import { api } from "./api";
import { CandidatePanel } from "./components/CandidatePanel";
import { ChatPanel } from "./components/ChatPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import type { ChatResponse, Health, MemoryCandidate } from "./types";

type Tab = "memories" | "candidates";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [entries, setEntries] = useState<ChatResponse[]>([]);
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([]);
  const [tab, setTab] = useState<Tab>("memories");
  const [memoryRefresh, setMemoryRefresh] = useState(0);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const pendingCount = candidates.filter((c) => c.status === "pending").length;

  return (
    <div className="app">
      <header className="app-header">
        <h1>ゆい：開発用操作画面</h1>
        <span className="muted small">
          {health
            ? `人格 ${health.persona.name}（${health.persona.version}） · ${
                health.llm.ok
                  ? `${health.llm.provider} / ${health.llm.model}`
                  : `LLM に接続できません: ${health.llm.error ?? "不明"}`
              }`
            : "バックエンドに接続できません"}
        </span>
      </header>

      <main className="layout">
        <ChatPanel
          conversationId={conversationId}
          entries={entries}
          onEntry={(entry) => {
            setConversationId(entry.conversation_id);
            setEntries((prev) => [...prev, entry]);
          }}
          onCandidates={(list) => {
            setCandidates(list);
            setTab("candidates");
          }}
          onReset={() => {
            // 終了した会話には続けられないので、次は新しい会話として始める。
            setConversationId(null);
            setEntries([]);
          }}
        />

        <div className="side">
          <nav className="tabs">
            <button
              type="button"
              className={tab === "memories" ? "active" : ""}
              onClick={() => setTab("memories")}
            >
              長期記憶
            </button>
            <button
              type="button"
              className={tab === "candidates" ? "active" : ""}
              onClick={() => setTab("candidates")}
            >
              記憶の候補{pendingCount > 0 ? `（${pendingCount}）` : ""}
            </button>
          </nav>

          {tab === "memories" ? (
            <MemoryPanel refreshKey={memoryRefresh} />
          ) : (
            <CandidatePanel
              candidates={candidates}
              onDecided={(updated) => {
                setCandidates((prev) =>
                  prev.map((c) => (c.id === updated.id ? updated : c)),
                );
                if (updated.status === "accepted") {
                  setMemoryRefresh((n) => n + 1);
                }
              }}
            />
          )}
        </div>
      </main>
    </div>
  );
}
