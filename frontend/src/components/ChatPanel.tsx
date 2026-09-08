import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type {
  ChatResponse,
  ConversationState,
  Memory,
  Message,
  RetrievedMemory,
  RunRecord,
  SpeakerRef,
  SpeechRun,
} from "../types";
import {
  CERTAINTY_LABEL,
  DELIVERY_LABEL,
  KIND_LABEL,
  SPEAKERS,
  formatDateTime,
} from "../types";
import type { ClientSpeechTiming, SpeechPlayer } from "../useSpeechPlayer";
import { SourceMessage } from "./SourceMessage";

interface Props {
  conversationId: number | null;
  /** 表示している発言。過去の会話を開いた場合は保存済みの履歴。 */
  messages: Message[];
  /** この画面で生成した返答の根拠。キーは返答の発言ID。 */
  liveEntries: Record<number, ChatResponse>;
  /** 会話の状態。新しい会話（未作成）は null。 */
  state: ConversationState | null;
  loading: boolean;
  /** 読み上げを使える状態か。エンジンに接続できないときは自動再生しない。 */
  speechAvailable: boolean;
  /** 読み上げの再生器。アバターと共有するため App が持つ。 */
  player: SpeechPlayer;
  /** いま誰として話すか。1つの会話に複数の相手を入れられるようにする。 */
  speaker: SpeakerRef;
  onSpeakerChange: (speaker: SpeakerRef) => void;
  /** 発言者IDと表示名の対応。誰の発言かを会話欄に出すために使う。 */
  speakerNames: Record<number, string>;
  /** 表示している会話の世代。結果を反映してよいかの判定に使う。 */
  view: number;
  /** 返答を画面へ反映する。世代が変わっていれば false を返す。 */
  onEntry: (entry: ChatResponse, view: number) => boolean;
  /** 終了して振り返った直後。候補の取り直しと、読み取り専用への切り替えに使う。 */
  onEnded: (view: number) => void;
  onNewConversation: () => void;
}

export function ChatPanel({
  conversationId,
  messages,
  liveEntries,
  state,
  loading,
  speechAvailable,
  player,
  speaker,
  onSpeakerChange,
  speakerNames,
  view,
  onEntry,
  onEnded,
  onNewConversation,
}: Props) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // 送信から返答が返るまで。待ち時間の内訳を見るために測る。
  const [chatMs, setChatMs] = useState<Record<number, number>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  // 終了済み・振り返り中の会話は読み取り専用で開く。送っても 409 になる。
  const readOnly = state === "ended" || state === "reflecting";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, busy]);

  // 別の会話を開いたら、前の会話の音声を鳴らし続けない。表示も残さない。
  // 会話 ID ではなく世代で判定する。新しい会話に ID が付いただけのときは
  // 世代が変わらないため、1 通目で鳴らし始めた音声を打ち消さない。
  const stopSpeech = player.stop;
  const firstViewRef = useRef(true);
  useEffect(() => {
    if (firstViewRef.current) {
      firstViewRef.current = false;
      return;
    }
    setError(null);
    setNotice(null);
    stopSpeech();
  }, [view, stopSpeech]);

  async function send() {
    const trimmed = text.trim();
    // 読み込み中は送信先が決まっていない。ここで送ると、開こうとしている
    // 会話ではなく新しい会話へ発言が入る。
    if (!trimmed || busy || readOnly || loading) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    // 返答を待つ間に別の会話へ移ることがある。始めたときの世代を覚えておく。
    const sentView = view;
    try {
      const sentAt = performance.now();
      const entry = await api.chat(trimmed, conversationId, speaker);
      const roundTrip = Math.round(performance.now() - sentAt);
      setChatMs((prev) => ({ ...prev, [entry.reply.id]: roundTrip }));
      setText("");
      if (!onEntry(entry, sentView)) {
        // 別の会話へ移ったあとの返答。画面には出さず、読み上げもしない。
        setNotice(
          "別の会話へ移ったため、いまの返答はこの画面に出していません。会話履歴から読めます。",
        );
        return;
      }
      // 返答が出たら読み上げる。生成しただけの状態から、再生の通知で進む。
      if (speechAvailable) player.play(entry.reply.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function endConversation() {
    if (conversationId === null || busy || readOnly) return;
    setBusy(true);
    setError(null);
    const endedView = view;
    try {
      await api.endConversation(conversationId);
      onEnded(endedView);
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
        <div className="header-actions">
          <button
            type="button"
            onClick={onNewConversation}
            disabled={conversationId === null || busy}
            title="いまの会話から離れ、新しい会話を始めます"
          >
            新しい会話
          </button>
          <button
            type="button"
            onClick={player.stop}
            disabled={player.playingId === null && player.loadingId === null}
            title="再生中の音声を止めます"
          >
            音声を止める
          </button>
          <button
            type="button"
            onClick={endConversation}
            disabled={conversationId === null || busy || readOnly}
            title="会話を終了し、長期記憶の候補を抽出します"
          >
            終了して振り返る
          </button>
        </div>
      </header>

      {readOnly && (
        <p className="notice">
          {state === "ended"
            ? "終了した会話です。読み取り専用で表示しています。"
            : "振り返り中の会話です。終わるまで発言を追加できません。"}
        </p>
      )}

      <div className="messages">
        {loading && <p className="muted center">読み込み中…</p>}
        {!loading && messages.length === 0 && (
          <p className="muted center">まだ会話がありません。話しかけてください。</p>
        )}
        {messages.map((message) =>
          message.speaker_kind === "user" ? (
            <div key={message.id} className="turn user">
              {/* 誰の発言かを出す。複数人の会話では本文だけでは追えない。 */}
              <div className="speaker-name">
                {(message.speaker_id !== null && speakerNames[message.speaker_id]) ||
                  "相手"}
              </div>
              <div className="bubble user" title={formatDateTime(message.created_at)}>
                {message.content}
              </div>
            </div>
          ) : (
            <ReplyTurn
              key={message.id}
              message={message}
              live={liveEntries[message.id]}
              player={player}
              speechAvailable={speechAvailable}
              chatMs={chatMs[message.id]}
            />
          ),
        )}
        {busy && <p className="muted center">考えています…</p>}
        <div ref={bottomRef} />
      </div>

      {error && <p className="error">{error}</p>}
      {notice && <p className="notice">{notice}</p>}
      {player.error && (
        <p className="error small">
          {player.error}{" "}
          <button type="button" className="link" onClick={player.clearError}>
            閉じる
          </button>
        </p>
      )}

      <div className="composer">
        <label className="speaker-select">
          <span className="muted small">話しているのは</span>
          <select
            value={speaker.external_id}
            onChange={(e) => {
              const next = SPEAKERS.find((s) => s.external_id === e.target.value);
              if (next) onSpeakerChange(next);
            }}
            disabled={busy || readOnly || loading}
            title="この発言を誰のものとして送るかを選びます"
          >
            {SPEAKERS.map((option) => (
              <option key={option.external_id} value={option.external_id}>
                {option.display_name}
              </option>
            ))}
          </select>
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void send();
          }}
          placeholder={
            loading
              ? "会話を読み込んでいます…"
              : readOnly
                ? "この会話には発言を追加できません"
                : "話しかける（⌘/Ctrl + Enter で送信）"
          }
          rows={3}
          disabled={busy || readOnly || loading}
        />
        <button
          type="button"
          onClick={send}
          disabled={busy || readOnly || loading || !text.trim()}
        >
          送信
        </button>
      </div>
    </section>
  );
}

/** キャラクターの返答と、その根拠への入り口。 */
function ReplyTurn({
  message,
  live,
  player,
  speechAvailable,
  chatMs,
}: {
  message: Message;
  live?: ChatResponse;
  player: SpeechPlayer;
  speechAvailable: boolean;
  chatMs?: number;
}) {
  const [showBasis, setShowBasis] = useState(false);
  const [showIdeal, setShowIdeal] = useState(false);

  const playing = player.playingId === message.id;
  const loading = player.loadingId === message.id;

  return (
    <div className="turn">
      <div
        className={playing ? "bubble character speaking" : "bubble character"}
        title={formatDateTime(message.created_at)}
      >
        {message.content}
      </div>
      <div className="turn-actions">
        {speechAvailable && (
          <button
            type="button"
            className="link"
            onClick={() => (playing ? player.stop() : player.play(message.id))}
            disabled={loading}
          >
            {loading ? "音声を用意中…" : playing ? "止める" : "再生"}
          </button>
        )}
        <button type="button" className="link" onClick={() => setShowBasis((v) => !v)}>
          {showBasis
            ? "根拠を隠す"
            : live
              ? `根拠（記憶 ${live.used_memories.length} 件）`
              : "根拠"}
        </button>
        <button type="button" className="link" onClick={() => setShowIdeal((v) => !v)}>
          理想の返答を記録
        </button>
        {message.delivery_state !== "completed" && (
          <span className="tag subtle">{DELIVERY_LABEL[message.delivery_state]}</span>
        )}
        {live && (
          <span className="muted small">
            {live.run.model} / {live.run.latency_ms ?? "-"} ms
          </span>
        )}
      </div>
      {showBasis && (
        <Basis
          messageId={message.id}
          used={live ? live.used_memories : null}
          chatMs={chatMs}
          clientTiming={player.timings[message.id]}
        />
      )}
      {showIdeal && <IdealForm messageId={message.id} />}
    </div>
  );
}

function Basis({
  messageId,
  used,
  chatMs,
  clientTiming,
}: {
  messageId: number;
  /** この画面で生成した返答だけが持つ、選ばれた理由と点数。 */
  used: RetrievedMemory[] | null;
  /** 送信から返答が返るまで（この画面で送った場合だけ分かる）。 */
  chatMs?: number;
  /** 音声の受け取りと再生開始（この画面で鳴らした場合だけ分かる）。 */
  clientTiming?: ClientSpeechTiming;
}) {
  const [run, setRun] = useState<RunRecord | null>(null);
  const [speechRuns, setSpeechRuns] = useState<SpeechRun[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .run(messageId)
      .then((record) => {
        setRun(record);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    api
      .speechRuns(messageId)
      .then(setSpeechRuns)
      .catch(() => setSpeechRuns([]));
  }, [messageId]);

  return (
    <div className="basis">
      <h4>渡した記憶</h4>
      {used ? (
        used.length === 0 ? (
          <p className="muted small">この返答に記憶は渡していません。</p>
        ) : (
          <ul>
            {used.map((item) => (
              <li key={item.memory.id}>
                <MemoryLine
                  memory={item.memory}
                  hint={`${item.reason} · 点数 ${item.score}`}
                />
              </li>
            ))}
          </ul>
        )
      ) : error ? (
        <p className="muted small">
          実行記録を読めないため、渡した記憶を確認できません。
        </p>
      ) : (
        // 過去の会話は、実行記録に残した記憶IDから引き直す。選ばれた理由と
        // 点数は保存していないため、当時の内容だけを示す。
        <ReferencedMemories ids={run ? (run.referenced_memory_ids ?? []) : null} />
      )}

      <h4>実行記録</h4>
      {error && <p className="error small">{error}</p>}
      {!run && !error && <p className="muted small">読み込み中…</p>}
      {run && (
        <>
          <div className="muted small">
            {run.provider} / {run.model}
            {run.model_digest && ` (${run.model_digest.slice(0, 16)}…)`} · 人格{" "}
            {run.persona_version ?? "版の記録なし"} · 応答{" "}
            {run.latency_ms ?? "-"} ms · トークン {run.prompt_tokens ?? "-"} /{" "}
            {run.completion_tokens ?? "-"} · 設定 {JSON.stringify(run.options ?? {})}
          </div>
          {run.system_prompt && (
            <details>
              <summary>実際に渡したプロンプト</summary>
              <pre>{run.system_prompt}</pre>
            </details>
          )}
        </>
      )}

      <h4>待ち時間</h4>
      <Timings
        run={run}
        speechRun={speechRuns[0]}
        chatMs={chatMs}
        clientTiming={clientTiming}
      />
    </div>
  );
}

/**
 * 送信から再生開始までの内訳。どの区間が待ち時間の大半かを見る。
 *
 * 画面側で測った値は、この画面で送信・再生した返答にだけ付く。過去の会話を
 * 開いた場合はサーバー側に残した記録だけを出す。
 */
function Timings({
  run,
  speechRun,
  chatMs,
  clientTiming,
}: {
  run: RunRecord | null;
  speechRun?: SpeechRun;
  chatMs?: number;
  clientTiming?: ClientSpeechTiming;
}) {
  const synthesisMs =
    speechRun && (speechRun.query_ms ?? 0) + (speechRun.synthesis_ms ?? 0);
  const total =
    chatMs !== undefined && clientTiming
      ? chatMs + clientTiming.fetchMs + clientTiming.startMs
      : undefined;

  const rows: { label: string; value: number | undefined; note?: string }[] = [
    { label: "記憶検索", value: run?.retrieval_ms ?? undefined },
    { label: "返答の生成", value: run?.latency_ms ?? undefined },
    { label: "送信 → 返答", value: chatMs, note: "上の 2 つを含む往復" },
    {
      label: "音声合成",
      value: synthesisMs === undefined ? undefined : synthesisMs,
      note: speechRun
        ? `合成用データ ${speechRun.query_ms ?? "-"} / 音声生成 ${
            speechRun.synthesis_ms ?? "-"
          }`
        : undefined,
    },
    {
      label: "返答 → 音声の受け取り",
      value: clientTiming?.fetchMs,
      note: "上の合成を含む",
    },
    { label: "受け取り → 再生開始", value: clientTiming?.startMs },
  ];

  return (
    <div className="timings">
      <table>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <th>{row.label}</th>
              <td>{row.value === undefined ? "-" : `${row.value} ms`}</td>
              <td className="muted small">{row.note ?? ""}</td>
            </tr>
          ))}
          <tr className="total">
            <th>送信 → 再生開始</th>
            <td>{total === undefined ? "-" : `${total} ms`}</td>
            <td className="muted small">
              {speechRun?.audio_ms ? `音声の長さ ${speechRun.audio_ms} ms` : ""}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** 実行記録に残った記憶IDから、当時渡した記憶を引き直す。 */
function ReferencedMemories({ ids }: { ids: number[] | null }) {
  const [memories, setMemories] = useState<Record<number, Memory | null>>({});
  const key = (ids ?? []).join(",");

  useEffect(() => {
    if (!ids || ids.length === 0) return;
    let cancelled = false;
    // 1件が取れなくても残りは見せる。取れなかったことは行として示す。
    Promise.all(
      ids.map(async (id) => [id, await api.memory(id).catch(() => null)] as const),
    ).then((pairs) => {
      if (!cancelled) setMemories(Object.fromEntries(pairs));
    });
    return () => {
      cancelled = true;
    };
    // ids は毎回別の配列になるため、中身で比較する。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (ids === null) return <p className="muted small">読み込み中…</p>;
  if (ids.length === 0)
    return <p className="muted small">この返答に記憶は渡していません。</p>;

  return (
    <ul>
      {ids.map((id) => {
        const memory = memories[id];
        if (memory === undefined) {
          return (
            <li key={id} className="muted small">
              #{id} 読み込み中…
            </li>
          );
        }
        if (memory === null) {
          return (
            <li key={id}>
              <span className="tag warn">取得できません</span>
              <span className="muted small"> #{id}</span>
            </li>
          );
        }
        return (
          <li key={id}>
            <MemoryLine
              memory={memory}
              hint={memory.status !== "active" ? `現在は ${memory.status}` : undefined}
            />
          </li>
        );
      })}
    </ul>
  );
}

function MemoryLine({ memory, hint }: { memory: Memory; hint?: string }) {
  return (
    <>
      <span className="tag">{KIND_LABEL[memory.kind]}</span>
      <span className="tag subtle">{CERTAINTY_LABEL[memory.certainty]}</span>
      {memory.content}
      <div className="muted small">
        #{memory.id}
        {hint && ` · ${hint}`}
        {memory.source_message_id !== null ? (
          <>
            {" · "}
            <SourceMessage messageId={memory.source_message_id} />
          </>
        ) : (
          <span className="tag warn"> 根拠未確認</span>
        )}
      </div>
    </>
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
