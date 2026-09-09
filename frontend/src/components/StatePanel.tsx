import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { CharacterState } from "../types";

const KIND_LABEL: Record<CharacterState["kind"], string> = {
  interest: "関心",
  relationship: "相手との関係",
};

/**
 * 変化する状態（関心・相手との関係）。
 *
 * 固定人格とは別に扱う。人格は版として管理し、ここでの更新では動かさない。
 * 候補は採用するまで会話に使われない。
 */
export function StatePanel({ refreshKey }: { refreshKey: number }) {
  const [states, setStates] = useState<CharacterState[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setStates(await api.states());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  async function run(action: () => Promise<CharacterState>) {
    setBusy(true);
    try {
      await action();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const pending = states.filter((s) => s.status === "pending");
  const review = states.filter((s) => s.status === "active" && s.needs_review);
  const active = states.filter((s) => s.status === "active" && !s.needs_review);

  return (
    <section className="panel">
      <h2>いまの自分</h2>
      <p className="muted small">
        経験から変わっていくもの。固定人格（口調・価値観・自己設定）とは別に扱い、
        採用したものだけが会話に渡ります。
      </p>
      {error && <p className="error">{error}</p>}

      {review.length > 0 && (
        <>
          <h3 className="small">根拠が変わったもの（{review.length}）</h3>
          <p className="muted small">
            もとにした記憶が訂正・削除されました。確認するまで会話には渡しません。
          </p>
          {review.map((state) => (
            <div key={state.id} className="memory">
              <div className="memory-head">
                <span className="tag">{KIND_LABEL[state.kind]}</span>
                {state.topic && <span className="tag subtle">{state.topic}</span>}
                <span className="tag warn">要確認</span>
              </div>
              <p>{state.content}</p>
              <p className="muted small">{state.review_reason}</p>
              <div className="row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateState(state.id, {
                        reviewed: true,
                        reason: "根拠を確認した",
                      }),
                    )
                  }
                >
                  このままでよい
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateState(state.id, {
                        status: "withdrawn",
                        reason: "根拠が無くなったため取り消した",
                      }),
                    )
                  }
                >
                  取り消す
                </button>
              </div>
            </div>
          ))}
        </>
      )}

      {pending.length > 0 && (
        <>
          <h3 className="small">更新の候補（{pending.length}）</h3>
          {pending.map((state) => (
            <div key={state.id} className="memory">
              <div className="memory-head">
                <span className="tag">{KIND_LABEL[state.kind]}</span>
                {state.topic && <span className="tag subtle">{state.topic}</span>}
                {state.source_conversation_id !== null && (
                  <span className="muted small">
                    会話 #{state.source_conversation_id}
                  </span>
                )}
              </div>
              <p>{state.content}</p>
              <div className="row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    run(() => api.decideState(state.id, { decision: "accept" }))
                  }
                >
                  採用
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() => api.decideState(state.id, { decision: "reject" }))
                  }
                >
                  却下
                </button>
              </div>
            </div>
          ))}
        </>
      )}

      <h3 className="small">採用済み（{active.length}）</h3>
      {active.length === 0 && <p className="muted small">まだありません。</p>}
      {active.map((state) => (
        <div key={state.id} className="memory">
          <div className="memory-head">
            <span className="tag">{KIND_LABEL[state.kind]}</span>
            {state.topic && <span className="tag subtle">{state.topic}</span>}
            {/* 開発者の確認を経ていないもの（フェーズ4 PR11）。 */}
            {state.auto_adopted && <span className="tag warn">自動採用</span>}
          </div>
          <p>{state.content}</p>
          {state.basis_memory_ids && state.basis_memory_ids.length > 0 && (
            <p className="muted small">
              根拠の記憶: {state.basis_memory_ids.map((id) => `#${id}`).join("、")}
            </p>
          )}
          <button
            type="button"
            className="link"
            disabled={busy}
            onClick={() =>
              run(() =>
                api.updateState(state.id, {
                  status: "withdrawn",
                  reason: "使わないことにした",
                }),
              )
            }
          >
            取り消す
          </button>
        </div>
      ))}
    </section>
  );
}
