import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { Goal } from "../types";

const TRIGGER_LABEL: Record<Goal["trigger"], string> = {
  next_conversation: "次に話すとき",
  after_date: "指定した日以降",
};

/** 終わり方は区別して残す。達成と、取消・前提の消滅・期限切れは意味が違う。
 * 撤回（withdrawn）はここに入れない。戻せる状態なので、別の一覧で扱う。 */
const FINISHED_LABEL: Partial<Record<Goal["status"], string>> = {
  done: "達成",
  cancelled: "前提が消えた",
  expired: "期限切れ",
  rejected: "却下",
};

function formatDate(value: string | null): string {
  if (!value) return "";
  return new Date(value).toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * 目標（次に何を話したいか）。
 *
 * 記憶・関心とは別に、実行条件と期限を持つ。候補は採用するまで行動に使わない。
 * 実行（質問を投げた）と達成（相手が答えた）は分けて表示する。混ぜると、
 * 聞いただけの目標を終わったものとして扱うことになる。
 */
export function GoalPanel({ refreshKey }: { refreshKey: number }) {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setGoals(await api.goals());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  async function run(action: () => Promise<Goal>) {
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

  const pending = goals.filter((g) => g.status === "pending");
  const review = goals.filter((g) => g.status === "active" && g.needs_review);
  const active = goals.filter((g) => g.status === "active" && !g.needs_review);
  // 撤回は開発者が一時的に外した状態で、戻せる。終わったものと分けて出す。
  const withdrawn = goals.filter((g) => g.status === "withdrawn");
  const finished = goals.filter((g) =>
    ["done", "cancelled", "expired", "rejected"].includes(g.status),
  );

  function schedule(goal: Goal) {
    const when = goal.trigger === "after_date" ? formatDate(goal.due_at) : "";
    return when ? `${TRIGGER_LABEL[goal.trigger]}（${when}）` : TRIGGER_LABEL[goal.trigger];
  }

  return (
    <section className="panel">
      <h2>目標</h2>
      <p className="muted small">
        経験をもとに「次に何を話したいか」を持ちます。採用したものだけが行動の候補に
        なります。質問したことと、相手が答えたことは分けて記録します。
      </p>
      {error && <p className="error">{error}</p>}

      {review.length > 0 && (
        <>
          <h3 className="small">根拠が変わったもの（{review.length}）</h3>
          <p className="muted small">
            もとにした記憶が訂正・削除されました。確認するまで行動には渡しません。
          </p>
          {review.map((goal) => (
            <div key={goal.id} className="memory">
              <div className="memory-head">
                <span className="tag">{schedule(goal)}</span>
                <span className="tag warn">要確認</span>
              </div>
              <p>{goal.content}</p>
              <p className="muted small">{goal.review_reason}</p>
              <div className="row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
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
                      api.updateGoal(goal.id, {
                        status: "withdrawn",
                        reason: "根拠を確かめるまで外しておく",
                      }),
                    )
                  }
                >
                  撤回する
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
                        status: "cancelled",
                        reason: "前提が無くなったため取り消した",
                      }),
                    )
                  }
                >
                  取り消す
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
                        status: "expired",
                        reason: "実行しないまま時期を過ぎた",
                      }),
                    )
                  }
                >
                  期限切れにする
                </button>
              </div>
              {/*
                「達成にする」はここに置かない。達成は「相手が答えた」という
                記録で、根拠が変わったまま付けると、古い前提のやりとりを
                達成として残すことになる。先に確認してから採用済みの一覧で行う。
              */}
            </div>
          ))}
        </>
      )}

      {pending.length > 0 && (
        <>
          <h3 className="small">目標の候補（{pending.length}）</h3>
          {pending.map((goal) => (
            <div key={goal.id} className="memory">
              <div className="memory-head">
                <span className="tag">{schedule(goal)}</span>
                {goal.source_conversation_id !== null && (
                  <span className="muted small">会話 #{goal.source_conversation_id}</span>
                )}
              </div>
              <p>{goal.content}</p>
              <div className="row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => run(() => api.decideGoal(goal.id, { decision: "accept" }))}
                >
                  採用
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() => run(() => api.decideGoal(goal.id, { decision: "reject" }))}
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
      {active.map((goal) => (
        <div key={goal.id} className="memory">
          <div className="memory-head">
            <span className="tag">{schedule(goal)}</span>
            {goal.last_executed_at && (
              <span className="tag subtle">
                実行済み {formatDate(goal.last_executed_at)}
              </span>
            )}
          </div>
          <p>{goal.content}</p>
          {goal.basis_memory_ids && goal.basis_memory_ids.length > 0 && (
            <p className="muted small">
              根拠の記憶: {goal.basis_memory_ids.map((id) => `#${id}`).join("、")}
              {goal.basis_is_provisional && "（暫定）"}
            </p>
          )}
          <div className="row">
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                run(() =>
                  api.updateGoal(goal.id, {
                    status: "done",
                    reason: "目的を達成した",
                  }),
                )
              }
            >
              達成にする
            </button>
            <button
              type="button"
              className="link"
              disabled={busy}
              onClick={() =>
                run(() =>
                  api.updateGoal(goal.id, {
                    status: "withdrawn",
                    reason: "いまは使わないことにした",
                  }),
                )
              }
            >
              撤回する
            </button>
            <button
              type="button"
              className="link"
              disabled={busy}
              onClick={() =>
                run(() =>
                  api.updateGoal(goal.id, {
                    status: "cancelled",
                    reason: "予定が無くなった",
                  }),
                )
              }
            >
              取り消す
            </button>
            <button
              type="button"
              className="link"
              disabled={busy}
              onClick={() =>
                run(() =>
                  api.updateGoal(goal.id, {
                    status: "expired",
                    reason: "実行しないまま時期を過ぎた",
                  }),
                )
              }
            >
              期限切れにする
            </button>
          </div>
        </div>
      ))}

      {withdrawn.length > 0 && (
        <>
          <h3 className="small">撤回中（{withdrawn.length}）</h3>
          <p className="muted small">
            一時的に外しているものです。戻すときに、根拠の記憶がまだ有効かを確かめます。
          </p>
          {withdrawn.map((goal) => (
            <div key={goal.id} className="memory">
              <div className="memory-head">
                <span className="tag subtle">撤回中</span>
              </div>
              <p>{goal.content}</p>
              <div className="row">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
                        status: "active",
                        reason: "また使うことにした",
                      }),
                    )
                  }
                >
                  戻す
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
                        status: "cancelled",
                        reason: "予定が無くなった",
                      }),
                    )
                  }
                >
                  取り消す
                </button>
                <button
                  type="button"
                  className="link"
                  disabled={busy}
                  onClick={() =>
                    run(() =>
                      api.updateGoal(goal.id, {
                        status: "expired",
                        reason: "実行しないまま時期を過ぎた",
                      }),
                    )
                  }
                >
                  期限切れにする
                </button>
              </div>
            </div>
          ))}
        </>
      )}

      {finished.length > 0 && (
        <>
          <h3 className="small">終わったもの（{finished.length}）</h3>
          <p className="muted small">
            達成したものは、もう一度は実行しません。取消・期限切れとは区別して残します。
            ここからは状態を戻せません。前提が戻ったのなら、新しい目標として作り直します。
          </p>
          {finished.map((goal) => (
            <div key={goal.id} className="memory">
              <div className="memory-head">
                <span className="tag subtle">
                  {FINISHED_LABEL[goal.status] ?? goal.status}
                </span>
                {goal.completed_at && (
                  <span className="muted small">{formatDate(goal.completed_at)}</span>
                )}
              </div>
              <p>{goal.content}</p>
            </div>
          ))}
        </>
      )}
    </section>
  );
}
