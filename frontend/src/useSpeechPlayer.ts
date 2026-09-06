import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { DeliveryNotice, Message } from "./types";

/** 返答の読み上げ。同時に鳴らすのは常に1つだけにする。 */
export interface SpeechPlayer {
  /** いま鳴っている発言。 */
  playingId: number | null;
  /** 音声を取りに行っている発言。 */
  loadingId: number | null;
  error: string | null;
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
  // 進行中の再生要求。新しい要求が来たら古い結果は捨てる。
  const requestRef = useRef(0);
  const playingRef = useRef<number | null>(null);

  const [playingId, setPlayingId] = useState<number | null>(null);
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        playingRef.current = messageId;
        setLoadingId(null);
        setPlayingId(messageId);
        notify(messageId, "playing");
      })();
    },
    [notify, teardown],
  );

  // 画面を離れるときに鳴らしっぱなしにしない。
  useEffect(() => teardown, [teardown]);

  return {
    playingId,
    loadingId,
    error,
    play,
    stop,
    clearError: useCallback(() => setError(null), []),
  };
}
