import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { AvatarPanel } from "./components/AvatarPanel";
import { CandidatePanel } from "./components/CandidatePanel";
import { ChatPanel } from "./components/ChatPanel";
import { HistoryPanel } from "./components/HistoryPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import { StatePanel } from "./components/StatePanel";
import type {
  ChatResponse,
  ConversationState,
  ConversationStateRecord,
  Health,
  MemoryCandidate,
  Message,
  SpeakerRef,
} from "./types";
import {
  DISPLAYED_STATE_KINDS,
  SELF_SPEAKER,
  conversationState,
  shouldApplyDelivery,
} from "./types";
import { useSpeechPlayer } from "./useSpeechPlayer";

type Tab = "memories" | "candidates" | "states" | "history";

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
  // v0.2：いまの会話で開いている状態（今の用件・未回答の質問・延期・終了・
  // 訂正）。ChatPanel の「この会話で」表示に使う。
  const [conversationStates, setConversationStates] = useState<ConversationStateRecord[]>([]);
  const [tab, setTab] = useState<Tab>("memories");
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const [historyRefresh, setHistoryRefresh] = useState(0);
  const [historyError, setHistoryError] = useState<string | null>(null);
  // いま誰として話すか。同じ会話へ別の相手を入れて、記憶を別々に持てて
  // いるかを確かめられるようにする（v0.1）。
  const [speaker, setSpeaker] = useState<SpeakerRef>(SELF_SPEAKER);
  // 相手の識別子（source/external_id）と、保存された speaker id の対応。
  // 記憶検索を会話と同じ条件で行うために使う。
  const [speakerIds, setSpeakerIds] = useState<Record<string, number>>({});
  // 発言者IDと表示名の対応。誰の発言かを会話欄に出すために使う。
  const [speakerNames, setSpeakerNames] = useState<Record<number, string>>({});

  const speakerKey = (ref: { source: string; external_id: string }) =>
    `${ref.source}:${ref.external_id}`;
  // 選んでいる相手の speaker id。まだ一度も話していない相手は null。
  const selfSpeakerId = speakerIds[speakerKey(speaker)] ?? null;

  // 表示している会話の世代。切り替えるたびに進める。送信・履歴の読み込み・
  // 終了の結果は、始めたときの世代がいまも一致するときだけ反映する。
  // そうしないと、返答を待つ間に別の会話を開いたとき、前の会話の結果が
  // いまの画面へ混ざる。
  const viewRef = useRef(0);
  const [view, setView] = useState(0);
  const switchView = useCallback(() => {
    viewRef.current += 1;
    setView(viewRef.current);
    return viewRef.current;
  }, []);

  // 再生の記録が変わったら、画面の発言にも反映する。
  // 通知の応答が入れ替わって届くことがあるため、進んだ状態からは戻さない。
  const applyDelivery = useCallback((updated: Message) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === updated.id &&
        shouldApplyDelivery(m.delivery_state, updated.delivery_state)
          ? updated
          : m,
      ),
    );
  }, []);
  // 会話欄とアバターの両方が見るため、再生器はここで持つ。
  const player = useSpeechPlayer(applyDelivery);

  // 保存済みの相手を読み直す。過去の会話を開いたときにも、誰の発言かを
  // 名前で出せるようにする。
  const loadSpeakers = useCallback(async () => {
    try {
      const speakers = await api.speakers();
      setSpeakerIds(
        Object.fromEntries(
          speakers.map((s) => [`${s.source}:${s.external_id}`, s.id]),
        ),
      );
      setSpeakerNames(
        Object.fromEntries(speakers.map((s) => [s.id, s.display_name])),
      );
    } catch {
      // 読めなくても会話はできる。名前の代わりに「相手」と出す。
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    void loadSpeakers();
    // 前回の会話で出た未判断の候補を拾い直す。
    api.pendingCandidates().then(setCandidates).catch(() => setCandidates([]));
  }, [loadSpeakers]);

  // 判断済みの候補は一覧から外し、残りだけを見せる。
  const refreshCandidates = () => {
    api.pendingCandidates().then(setCandidates).catch(() => undefined);
  };

  /** 保存済みの会話を開く。終了済みなら読み取り専用として表示する。 */
  async function openConversation(id: number) {
    const token = switchView();
    setConversationId(null);
    setMessages([]);
    setLiveEntries({});
    setConversation(null);
    setConversationStates([]);
    setLoadingConversation(true);
    setHistoryError(null);
    try {
      // 会話状態も同じ await の中で待つ。片方だけ先に確定させて送信できる
      // ようにすると、送信直後の一覧が、開いたときに投げていた古い取得の
      // 遅延応答で上書きされることがある（レビューで実測）。
      // /chat の応答（entry.conversation_states）は話している相手だけに
      // 絞っているため、ここも同じ絞り方にする（レビュー指摘：絞り方が
      // 経路ごとに違うと、複数話者の会話で表示が食い違う）。
      const [detail, states] = await Promise.all([
        api.conversation(id),
        // 終了済みの会話は expired になっているので、ここでは何も出ない。
        // 読めなくても会話は開ける。
        api
          .conversationStates(id, {
            status: "open",
            // 画面が使う種類だけに絞る。presented・confirmed を含む全件は
            // 会話が長くなるほど肥大する（/chat の同梱と同じ理由）。
            kind: DISPLAYED_STATE_KINDS,
            ...(selfSpeakerId !== null ? { target_speaker_id: selfSpeakerId } : {}),
          })
          .catch(() => []),
      ]);
      // 続けて別の会話を開いた場合、遅れて届いたこちらは捨てる。
      if (token !== viewRef.current) return;
      setConversationId(detail.id);
      setMessages(detail.messages);
      setConversation(conversationState(detail));
      setConversationStates(states);
      // この会話に出てくる相手の名前を出せるようにする。
      void loadSpeakers();
    } catch (e) {
      if (token !== viewRef.current) return;
      setHistoryError(e instanceof Error ? e.message : String(e));
    } finally {
      if (token === viewRef.current) setLoadingConversation(false);
    }
  }

  function startNewConversation() {
    switchView();
    setConversationId(null);
    setMessages([]);
    setLiveEntries({});
    setConversation(null);
    setConversationStates([]);
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
          conversationStates={conversationStates}
          loading={loadingConversation}
          speechAvailable={speechAvailable}
          player={player}
          speaker={speaker}
          onSpeakerChange={setSpeaker}
          speakerNames={speakerNames}
          view={view}
          onEntry={(entry, token) => {
            // 待っている間に別の会話へ移っていたら、この画面には出さない。
            // 発言そのものは保存済みで、会話履歴から読み直せる。
            if (token !== viewRef.current) {
              setHistoryRefresh((n) => n + 1);
              return false;
            }
            const isNew = conversationId === null;
            setConversationId(entry.conversation_id);
            setMessages((prev) => [...prev, entry.user_message, entry.reply]);
            setLiveEntries((prev) => ({ ...prev, [entry.reply.id]: entry }));
            setConversation("open");
            setConversationStates(entry.conversation_states);
            // 新しい会話が作られたときだけ一覧を取り直す。
            if (isNew) setHistoryRefresh((n) => n + 1);
            // 初めて話した相手はこの時点で作られる。ここで対応を覚えておく。
            const id = entry.user_message.speaker_id;
            if (id !== null) {
              setSpeakerIds((prev) => ({ ...prev, [speakerKey(speaker)]: id }));
              setSpeakerNames((prev) => ({ ...prev, [id]: speaker.display_name }));
            }
            return true;
          }}
          onEnded={(token) => {
            // 終了しても画面からは消さない。読み取り専用に切り替えるだけにして、
            // 何を話した結果の候補なのかを見比べられるようにする。
            // 別の会話へ移っていた場合、いまの表示は終了扱いにしない。
            if (token === viewRef.current) {
              setConversation("ended");
              // 開いていた状態はすべて expired になった（計画 §5）。
              // 「この会話で」は今の会話向けの表示なので消す
              // （訂正・食い違いは会話終了後、記憶の候補欄に出る）。
              setConversationStates([]);
            }
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
              className={tab === "states" ? "active" : ""}
              onClick={() => setTab("states")}
            >
              いまの自分
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
              conversationId={conversationId}
              conversationEnded={conversation === "ended"}
              onDecided={(updated) => {
                setCandidates((prev) => prev.filter((c) => c.id !== updated.id));
                if (updated.status === "accepted") {
                  setMemoryRefresh((n) => n + 1);
                }
              }}
            />
          )}
          {tab === "states" && <StatePanel refreshKey={memoryRefresh} />}
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
