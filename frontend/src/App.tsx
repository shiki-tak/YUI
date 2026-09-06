import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { AvatarPanel } from "./components/AvatarPanel";
import { CandidatePanel } from "./components/CandidatePanel";
import { ChatPanel } from "./components/ChatPanel";
import { HistoryPanel } from "./components/HistoryPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import type {
  ChatResponse,
  ConversationState,
  Health,
  MemoryCandidate,
  Message,
} from "./types";
import { SELF_SPEAKER, conversationState } from "./types";
import { useSpeechPlayer } from "./useSpeechPlayer";

type Tab = "memories" | "candidates" | "history";

function voiceLabel(health: Health): string {
  if (!health.voice.enabled) return "音声：無効";
  if (!health.voice.ok) return "音声：エンジンに接続できません";
  return `音声：${health.voice.provider} 話者 ${health.voice.speaker ?? "-"}`;
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(null);
  // 表示している発言。過去の会話を開いた場合は保存済みの履歴が入る。
  const [messages, setMessages] = useState<Message[]>([]);
  // この画面で生成した返答の根拠。過去の会話には無いので、実行記録から引き直す。
  const [liveEntries, setLiveEntries] = useState<Record<number, ChatResponse>>({});
  const [conversation, setConversation] = useState<ConversationState | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [candidates, setCandidates] = useState<MemoryCandidate[]>([]);
  const [tab, setTab] = useState<Tab>("memories");
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [historyError, setHistoryError] = useState<string | null>(null);
  // 記憶検索を会話と同じ条件で行うために、自分の speaker id を解決する。
  const [selfSpeakerId, setSelfSpeakerId] = useState<number | null>(null);

  // 再生の記録が変わったら、画面の発言にも反映する。
  const applyDelivery = useCallback((updated: Message) => {
    setMessages((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
  }, []);
  // 会話欄とアバターの両方が見るため、再生器はここで持つ。
  const player = useSpeechPlayer(applyDelivery);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .speakers()
      .then((speakers) => {
        const self = speakers.find(
          (s) =>
            s.source === SELF_SPEAKER.source &&
            s.external_id === SELF_SPEAKER.external_id,
        );
        setSelfSpeakerId(self ? self.id : null);
      })
      .catch(() => setSelfSpeakerId(null));
    // 前回の会話で出た未判断の候補を拾い直す。
    api.pendingCandidates().then(setCandidates).catch(() => setCandidates([]));
  }, []);

  // 判断済みの候補は一覧から外し、残りだけを見せる。
  const refreshCandidates = () => {
    api.pendingCandidates().then(setCandidates).catch(() => undefined);
  };

  /** 保存済みの会話を開く。終了済みなら読み取り専用として表示する。 */
  async function openConversation(id: number) {
    setLoadingConversation(true);
    setHistoryError(null);
    try {
      const detail = await api.conversation(id);
      setConversationId(detail.id);
      setMessages(detail.messages);
      setLiveEntries({});
      setConversation(conversationState(detail));
    } catch (e) {
      setHistoryError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingConversation(false);
    }
  }

  function startNewConversation() {
    setConversationId(null);
    setMessages([]);
    setLiveEntries({});
    setConversation(null);
  }

  const pendingCount = candidates.filter((c) => c.status === "pending").length;
  // 音声は、有効で、かつエンジンに接続できているときだけ使う。
  const speechAvailable = health?.voice.enabled === true && health.voice.ok;
  // 字幕は、読み上げている発言の本文そのもの。合成に渡すのと同じ文字列を使う。
  const speakingText =
    messages.find((m) => m.id === player.playingId)?.content ?? null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>YUI：開発用操作画面</h1>
        <span className="muted small">
          {health
            ? `人格 ${health.persona.name}（${health.persona.version}） · ${
                health.llm.ok
                  ? `${health.llm.provider} / ${health.llm.model}`
                  : `LLM に接続できません: ${health.llm.error ?? "不明"}`
              } · ${voiceLabel(health)}`
            : "バックエンドに接続できません"}
        </span>
      </header>

      <main className="layout">
        <AvatarPanel
          levelRef={player.levelRef}
          speaking={player.playingId !== null}
          subtitle={speakingText}
          credit={health?.voice.credit}
        />

        <ChatPanel
          conversationId={conversationId}
          messages={messages}
          liveEntries={liveEntries}
          state={conversation}
          loading={loadingConversation}
          speechAvailable={speechAvailable}
          player={player}
          onEntry={(entry) => {
            const isNew = conversationId === null;
            setConversationId(entry.conversation_id);
            setMessages((prev) => [...prev, entry.user_message, entry.reply]);
            setLiveEntries((prev) => ({ ...prev, [entry.reply.id]: entry }));
            setConversation("open");
            // 新しい会話が作られたときだけ一覧を取り直す。
            if (isNew) setHistoryRefresh((n) => n + 1);
            if (selfSpeakerId === null) {
              // 初回の会話で相手が作られるので、ここで解決しておく。
              setSelfSpeakerId(entry.user_message.speaker_id);
            }
          }}
          onEnded={() => {
            // 終了しても画面からは消さない。読み取り専用に切り替えるだけにして、
            // 何を話した結果の候補なのかを見比べられるようにする。
            setConversation("ended");
            setHistoryRefresh((n) => n + 1);
            refreshCandidates();
            setTab("candidates");
          }}
          onNewConversation={startNewConversation}
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
            <button
              type="button"
              className={tab === "history" ? "active" : ""}
              onClick={() => setTab("history")}
            >
              会話履歴
            </button>
          </nav>

          {tab === "memories" && (
            <MemoryPanel refreshKey={memoryRefresh} speakerId={selfSpeakerId} />
          )}
          {tab === "candidates" && (
            <CandidatePanel
              candidates={candidates}
              onDecided={(updated) => {
                setCandidates((prev) => prev.filter((c) => c.id !== updated.id));
                if (updated.status === "accepted") {
                  setMemoryRefresh((n) => n + 1);
                }
              }}
            />
          )}
          {tab === "history" && (
            <>
              {historyError && <p className="error">{historyError}</p>}
              <HistoryPanel
                refreshKey={historyRefresh}
                currentId={conversationId}
                onOpen={openConversation}
              />
            </>
          )}
        </div>
      </main>
    </div>
  );
}
