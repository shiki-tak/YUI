import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import { api } from "./api";
import type { DeliveryNotice, Message } from "./types";

declare global {
  interface Window {
    // Safari の旧実装。無い環境もあるため任意にしておく。
    webkitAudioContext?: typeof AudioContext;
  }
}

/** 画面側で測った待ち時間。サーバー側の記録と合わせて内訳を見る。 */
export interface ClientSpeechTiming {
  /** 音声を要求してから受け取るまで（サーバーでの合成を含む）。 */
  fetchMs: number;
  /** 受け取ってから鳴り始めるまで。 */
  startMs: number;
}

/** 返答の読み上げ。同時に鳴らすのは常に1つだけにする。 */
export interface SpeechPlayer {
  /** いま鳴っている発言。 */
  playingId: number | null;
  /** 音声を取りに行っている発言。 */
  loadingId: number | null;
  error: string | null;
  /**
   * 鳴っている音声の大きさ（0〜1）。口パクに使う。
   *
   * 毎フレームの値を state で持つと画面全体が再描画されるため、ref で渡して
   * 読む側が自分の描画周期で見る。
   */
  levelRef: MutableRefObject<number>;
  /** 発言ごとの、画面側で測った待ち時間。 */
  timings: Record<number, ClientSpeechTiming>;
  play: (messageId: number) => void;
  stop: () => void;
  clearError: () => void;
}

export function useSpeechPlayer(
  /** 再生の記録が返ってきたときに、画面の状態へ反映する。 */
  onDelivered?: (message: Message) => void,
): SpeechPlayer {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);
  // 音量の解析。使えない環境では解析なしで再生だけ行う（口は閉じたまま）。
  const contextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frameRef = useRef<number | null>(null);
  const levelRef = useRef(0);
  // 進行中の再生要求。新しい要求が来たら古い結果は捨てる。
  const requestRef = useRef(0);
  const playingRef = useRef<number | null>(null);

  const [playingId, setPlayingId] = useState<number | null>(null);
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [timings, setTimings] = useState<Record<number, ClientSpeechTiming>>({});

  // 呼び出し側が毎回新しい関数を渡しても、play / stop の同一性を保つ。
  // 変わると再生中に後片付けの副作用が走り、音声が止まってしまう。
  const onDeliveredRef = useRef(onDelivered);
  onDeliveredRef.current = onDelivered;

  const notify = useCallback((messageId: number, state: DeliveryNotice) => {
    api
      .notifyDelivery(messageId, state)
      .then((message) => onDeliveredRef.current?.(message))
      .catch(() => {
        // 記録に失敗しても再生は続ける。会話を止める理由にはしない。
      });
  }, []);

  /** 鳴っている音声を止めて後片付けする。中断として記録するかは呼び出し側が決める。 */
  const teardown = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    levelRef.current = 0;
    sourceRef.current?.disconnect();
    sourceRef.current = null;
    analyserRef.current?.disconnect();
    analyserRef.current = null;

    const audio = audioRef.current;
    if (audio) {
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audioRef.current = null;
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
  }, []);

  /** 再生中の音声を解析につなぐ。つなげない場合は普通に鳴らすだけにする。 */
  const connectAnalyser = useCallback(async (audio: HTMLAudioElement) => {
    try {
      const context =
        contextRef.current ??
        new (window.AudioContext ?? window.webkitAudioContext)();
      contextRef.current = context;
      if (context.state === "suspended") await context.resume();
      // 動いていない状態でつなぐと音そのものが出なくなる。解析はあきらめ、
      // 再生を優先する。
      if (context.state !== "running") return;

      // 音の経路を先に作る。解析はそこから枝分かれさせるだけにして、
      // 解析側で失敗しても音が消えないようにする。
      const source = context.createMediaElementSource(audio);
      source.connect(context.destination);
      sourceRef.current = source;

      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;

      const samples = new Uint8Array(analyser.fftSize);
      const tick = () => {
        const node = analyserRef.current;
        if (!node) return;
        node.getByteTimeDomainData(samples);
        // 中央（128）からのずれの二乗平均。無音なら 0 に近づく。
        let sum = 0;
        for (const value of samples) {
          const centered = (value - 128) / 128;
          sum += centered * centered;
        }
        levelRef.current = Math.sqrt(sum / samples.length);
        frameRef.current = requestAnimationFrame(tick);
      };
      frameRef.current = requestAnimationFrame(tick);
    } catch {
      // 解析できない環境。再生は続け、口は閉じたままにする。
      levelRef.current = 0;
    }
  }, []);

  const stop = useCallback(() => {
    // 取得中の要求も無効にする。あとから届いた音声が鳴り出さないようにする。
    requestRef.current += 1;
    const stopped = playingRef.current;
    teardown();
    playingRef.current = null;
    setPlayingId(null);
    setLoadingId(null);
    if (stopped !== null) notify(stopped, "aborted");
  }, [notify, teardown]);

  const play = useCallback(
    (messageId: number) => {
      const request = ++requestRef.current;
      // 直前の再生は中断として記録する。重ねて鳴らさないため。
      const interrupted = playingRef.current;
      teardown();
      playingRef.current = null;
      setPlayingId(null);
      if (interrupted !== null) notify(interrupted, "aborted");

      setLoadingId(messageId);
      setError(null);

      void (async () => {
        const requestedAt = performance.now();
        let blob: Blob;
        try {
          blob = await api.speech(messageId);
        } catch (e) {
          if (request !== requestRef.current) return;
          setError(e instanceof Error ? e.message : String(e));
          setLoadingId(null);
          return;
        }
        // 待っている間に別の再生が始まっていたら、この音声は捨てる。
        if (request !== requestRef.current) return;
        const receivedAt = performance.now();

        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioRef.current = audio;
        urlRef.current = url;

        audio.onended = () => {
          if (request !== requestRef.current) return;
          teardown();
          playingRef.current = null;
          setPlayingId(null);
          notify(messageId, "completed");
        };
        audio.onerror = () => {
          if (request !== requestRef.current) return;
          teardown();
          playingRef.current = null;
          setPlayingId(null);
          setLoadingId(null);
          setError("音声を再生できませんでした。");
        };

        try {
          await audio.play();
        } catch (e) {
          if (request !== requestRef.current) return;
          teardown();
          setLoadingId(null);
          // 自動再生を止めるブラウザがある。画面から再生し直せることを伝える。
          setError(
            `音声を再生できませんでした（${
              e instanceof Error ? e.message : String(e)
            }）。「再生」から鳴らせます。`,
          );
          return;
        }

        if (request !== requestRef.current) return;
        const startedAt = performance.now();
        playingRef.current = messageId;
        setLoadingId(null);
        setPlayingId(messageId);
        setTimings((prev) => ({
          ...prev,
          [messageId]: {
            fetchMs: Math.round(receivedAt - requestedAt),
            startMs: Math.round(startedAt - receivedAt),
          },
        }));
        notify(messageId, "playing");
        await connectAnalyser(audio);
      })();
    },
    [connectAnalyser, notify, teardown],
  );

  // 画面を離れるときに鳴らしっぱなしにしない。
  useEffect(() => teardown, [teardown]);

  return {
    playingId,
    loadingId,
    error,
    levelRef,
    timings,
    play,
    stop,
    clearError: useCallback(() => setError(null), []),
  };
}
