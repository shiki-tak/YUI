import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Memory, MemoryRevision, RetrievedMemory } from "../types";
import { CERTAINTY_LABEL, KIND_LABEL, VISIBILITY_LABEL } from "../types";

export function MemoryPanel({ refreshKey }: { refreshKey: number }) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [includeInactive, setIncludeInactive] = useState(false);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<RetrievedMemory[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setMemories(await api.memories(includeInactive));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [includeInactive]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  async function runSearch() {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }
    try {
      const result = await api.searchMemories(query.trim());
      setSearchResults(result.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>長期記憶</h2>
        <label className="muted small">
          <input
            type="checkbox"
            checked={includeInactive}
            onChange={(e) => setIncludeInactive(e.target.checked)}
          />
          訂正・削除済みも表示
        </label>
      </header>

      <div className="search-row">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void runSearch()}
          placeholder="会話と同じ検索を試す（例：前に何を話す約束をしたっけ）"
        />
        <button type="button" onClick={runSearch}>
          検索
        </button>
        {searchResults && (
          <button type="button" className="link" onClick={() => setSearchResults(null)}>
            一覧に戻る
          </button>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="memory-list">
        {searchResults
          ? searchResults.map((item) => (
              <MemoryRow
                key={item.memory.id}
                memory={item.memory}
                hint={`${item.reason} · 点数 ${item.score}`}
                onChanged={load}
              />
            ))
          : memories.map((memory) => (
              <MemoryRow key={memory.id} memory={memory} onChanged={load} />
            ))}
        {!searchResults && memories.length === 0 && (
          <p className="muted center">記憶はまだありません。</p>
        )}
        {searchResults?.length === 0 && (
          <p className="muted center">この問いかけで引ける記憶はありません。</p>
        )}
      </div>
    </section>
  );
}

function MemoryRow({
  memory,
  hint,
  onChanged,
}: {
  memory: Memory;
  hint?: string;
  onChanged: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [content, setContent] = useState(memory.content);
  const [keywords, setKeywords] = useState(memory.keywords);
  const [reason, setReason] = useState("");
  const [revisions, setRevisions] = useState<MemoryRevision[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function act(fn: () => Promise<unknown>) {
    try {
      await fn();
      setError(null);
      setEditing(false);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className={`memory ${memory.status !== "active" ? "inactive" : ""}`}>
      <div className="memory-head">
        <span className="tag">{KIND_LABEL[memory.kind]}</span>
        <span className="tag subtle">{CERTAINTY_LABEL[memory.certainty]}</span>
        <span className="tag subtle">{VISIBILITY_LABEL[memory.visibility]}</span>
        {memory.status !== "active" && <span className="tag warn">{memory.status}</span>}
        <span className="muted small">#{memory.id}</span>
      </div>

      {editing ? (
        <div className="memory-edit">
          <textarea value={content} onChange={(e) => setContent(e.target.value)} rows={2} />
          <input
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="検索用キーワード（空白区切り）"
          />
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="訂正の理由"
          />
          <div className="row">
            <button
              type="button"
              onClick={() =>
                act(() =>
                  api.correctMemory(memory.id, {
                    content,
                    keywords,
                    reason: reason || "訂正",
                  }),
                )
              }
            >
              訂正を保存
            </button>
            <button type="button" className="link" onClick={() => setEditing(false)}>
              やめる
            </button>
          </div>
        </div>
      ) : (
        <p className="memory-content">{memory.content}</p>
      )}

      <div className="muted small">
        {memory.keywords && `キーワード: ${memory.keywords}`}
        {memory.source_message_id !== null && ` · 根拠の発言 #${memory.source_message_id}`}
        {hint && ` · ${hint}`}
      </div>

      {error && <p className="error small">{error}</p>}

      {!editing && (
        <div className="row">
          <button type="button" className="link" onClick={() => setEditing(true)}>
            訂正
          </button>
          <button
            type="button"
            className="link"
            onClick={() => act(() => api.deleteMemory(memory.id, "誤りのため削除"))}
            disabled={memory.status === "deleted"}
          >
            削除
          </button>
          <button
            type="button"
            className="link"
            onClick={() => act(() => api.restoreMemory(memory.id))}
          >
            直前の状態に戻す
          </button>
          <button
            type="button"
            className="link"
            onClick={async () =>
              setRevisions(revisions ? null : await api.revisions(memory.id))
            }
          >
            変更履歴
          </button>
        </div>
      )}

      {revisions && (
        <ul className="revisions">
          {revisions.map((revision) => (
            <li key={revision.id}>
              <span className="tag subtle">{revision.action}</span>
              {revision.reason && <span className="muted small"> {revision.reason}</span>}
              {revision.before && (
                <div className="muted small">
                  更新前: {String(revision.before.content ?? "")}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
