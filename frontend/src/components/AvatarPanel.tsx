import { useEffect, useState } from "react";
import type { MutableRefObject } from "react";

/** 口と目の開閉で 4 状態。切り替えても髪や服が動かない素材を使う。 */
const STATES = [
  { mouthOpen: false, eyesOpen: true, file: "mouth_closed_eyes_open" },
  { mouthOpen: true, eyesOpen: true, file: "mouth_open_eyes_open" },
  { mouthOpen: false, eyesOpen: false, file: "mouth_closed_eyes_closed" },
  { mouthOpen: true, eyesOpen: false, file: "mouth_open_eyes_closed" },
] as const;

// 音量から口の開閉を決める。開く・閉じるでしきい値を分け、境目での
// ちらつきを防ぐ。
const OPEN_LEVEL = 0.08;
const CLOSE_LEVEL = 0.05;
/** 一度開いたら最低これだけ保つ。細かく震えて見えないようにする。 */
const HOLD_MS = 70;

const BLINK_CLOSED_MS = 120;
const BLINK_MIN_MS = 2500;
const BLINK_MAX_MS = 6000;

interface Props {
  /** 鳴っている音声の大きさ（0〜1）。 */
  levelRef: MutableRefObject<number>;
  /** 読み上げ中か。止まったら口を閉じる。 */
  speaking: boolean;
  /** 音声を公開する場に出す表記。 */
  credit?: string;
}

export function AvatarPanel({ levelRef, speaking, credit }: Props) {
  const [mouthOpen, setMouthOpen] = useState(false);
  const [eyesOpen, setEyesOpen] = useState(true);

  // 口パク。毎フレーム見て、変わったときだけ描き直す。
  useEffect(() => {
    if (!speaking) {
      setMouthOpen(false);
      return;
    }
    let frame = 0;
    let open = false;
    let changedAt = 0;

    const tick = (now: number) => {
      const level = levelRef.current;
      const next = open ? level > CLOSE_LEVEL : level > OPEN_LEVEL;
      if (next !== open && now - changedAt >= HOLD_MS) {
        open = next;
        changedAt = now;
        setMouthOpen(open);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [levelRef, speaking]);

  // まばたき。読み上げの有無に関係なく続ける。
  useEffect(() => {
    let closeTimer: number | undefined;
    let nextTimer: number | undefined;

    const schedule = () => {
      const wait = BLINK_MIN_MS + Math.random() * (BLINK_MAX_MS - BLINK_MIN_MS);
      nextTimer = window.setTimeout(() => {
        setEyesOpen(false);
        closeTimer = window.setTimeout(() => {
          setEyesOpen(true);
          schedule();
        }, BLINK_CLOSED_MS);
      }, wait);
    };
    schedule();

    return () => {
      window.clearTimeout(nextTimer);
      window.clearTimeout(closeTimer);
    };
  }, []);

  return (
    <section className="panel avatar-panel">
      <header className="panel-header">
        <h2>アバター</h2>
        <span className="muted small">{speaking ? "発話中" : "待機中"}</span>
      </header>

      {/* 4枚とも置いて表示だけ切り替える。切り替えた瞬間の読み込み待ちを避ける。 */}
      <div className="avatar-stage">
        {STATES.map((state) => (
          <img
            key={state.file}
            src={`/avatar/${state.file}.png`}
            alt=""
            className={
              state.mouthOpen === mouthOpen && state.eyesOpen === eyesOpen
                ? "avatar-image shown"
                : "avatar-image"
            }
          />
        ))}
      </div>

      {credit && <p className="avatar-credit">{credit}</p>}
    </section>
  );
}
