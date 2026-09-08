# フェーズ4 レビュー

## PR1：目標の器 — 第1回レビュー（2026-09-08）

### 結論

PR1の承認前に修正したい不具合を1件確認した。注入した現在時刻のタイムゾーンによって、目標の実行可否が変わる。それ以外には、今回の範囲で追加の不具合は見つからなかった。フェーズ4全体の完了を評価するレビューではない。

### 対象

- `phase4-goal-store` の未コミットのPR1実装：`backend/app/agent/goal.py`、`backend/app/models.py` の目標関連定義、マイグレーション `c667f41b8e3c`、`backend/tests/test_goal.py`、`backend/tests/test_migration.py` の追加部分。
- `AGENTS.md`、`README.md`、`docs/plan/phase4.md` の決定事項とPR1の範囲、設計書の目標・フェーズ4の記述を確認した。訂正波及は `character_state.mark_for_review` と比較した。
- API・画面、訂正経路からの呼び出し配線、振り返りからの抽出、行動選択、再生状態との連動、完了・繰り返し防止・期限切れの遷移、実行記録への目標参照の追加は後続PRの範囲として扱った。

### 指摘事項

#### 1. [中] `active_goals` が現在時刻をUTCへ正規化せず、実行日時を誤判定する

- **場所：** `backend/app/agent/goal.py:173`、`backend/app/agent/goal.py:198`（レビュー時点の行番号）
- **確認区分：** 実行して再現
- **発生条件：** `now` にUTC以外のタイムゾーン付き日時を渡す。
- **原因：** `due_at` は保存時にUTCへ正規化されるが、`now` はそのままSQL比較へ渡される。SQLiteへのバインド時にタイムゾーン情報が落ち、異なる基準の時刻を比較する。
- **再現手順：** 一時DBに `due_at=2026-09-13 15:30 UTC` の有効な目標を保存し、同じ瞬間を異なるタイムゾーンで表した `now` で `active_goals` を呼び出した。既存の `session_factory` fixtureを利用する再現テストをリポジトリ外の `/private/tmp/yui_goal_pr1_review_test.py` に作成して実行した。
- **観測結果：** 次の2組で、同じ瞬間にもかかわらず選択結果が異なった。

  | 渡した現在時刻 | 期待 | 実際 |
  |---|---|---|
  | 9月13日15:00 UTC | 対象外 | 対象外 |
  | 同じ瞬間の9月14日00:00 JST | 対象外 | 選択される |
  | 9月13日16:00 UTC | 選択される | 選択される |
  | 同じ瞬間の9月13日09:00 UTC−7 | 選択される | 対象外になる |

- **影響：** 実行可能になる前の目標を選択したり、実行可能な目標を落としたりする。既定の `utcnow()` を使う経路では再現しないが、現在すでに存在する `now` 引数で発生する。
- **修正方針：** SQL比較前に `now` もUTCへ正規化する。naive日時の解釈も明示し、同一瞬間のUTC・正負のオフセットで結果が一致するテストと、`now == due_at` の境界テストを追加する。

### その他の重点項目の評価

以下は追加の不具合指摘ではなく、確認した判断と検証の限界である。

- **訂正波及（コード確認）：** 明示的な記憶ID、会話単位の根拠、暫定根拠、撤回中の目標を拾う規則は `character_state.mark_for_review` と一致している。終了状態の `done`・`rejected`・`cancelled`・`expired` は実行対象にならないため、現在の実装で対象外にする判断は妥当。
- **配信と相手の絞り込み（コード確認）：** 配信では公開目標だけが対象。さらに、相手が不明なら相手指定の目標を除外し、相手Aなら相手なし・Aの目標だけを選ぶ。コード上、別人を対象にした目標を選ぶ経路は見つからなかった。
- **作成時の検査（コード確認）：** `create_goal` は条件と日時の組み合わせを検査する。ORM直接操作では迂回できるが、今回の正式な作成経路の不具合とは判定していない。PR4の更新処理にも同じ検査を適用する必要がある。
- **snapshot（コード確認）：** 現在変更する項目について、履歴から欠落する問題は見つからなかった。`source_conversation_id` はsnapshotに含まれないが、今回の処理では変更せず、目標本体に保持している。
- **テストの観点：** 不正な条件の拒否や対象外の除外も確認しており、成功経路だけのテストではない。ただし、相手の指定と公開範囲を独立に組み合わせたケース、配信で相手を指定するケースは補強できる。
- **schema一致テストの限界（コード確認）：** `backend/tests/test_migration.py:101` の比較対象は列名・型・NULL可否と索引名。外部キー、既定値、索引の構成列などまで一致を保証するテストではない。今回、実際の定義不一致は見つからなかった。
- **マイグレーション：** 既存テーブルを書き換えず、目標とその履歴のテーブルを追加する。downgradeは履歴側から削除する順序になっている。一時DBでupgrade・downgrade・再upgradeが成功した。`ix_goals_status_trigger` について、今回の確認で不具合と判断する根拠は得られなかった。

### 実行した確認

バックエンドディレクトリで実行した。一時DBを使い、通常利用のDBは変更していない。

```sh
YUI_DATABASE_URL=sqlite+aiosqlite:////private/tmp/yui-phase4-pr1-unused.db PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
.venv/bin/ruff check app tests --no-cache
```

- 既存テスト：**206件成功**。
- ruff：成功。

追加の日時再現テスト：

```sh
YUI_DATABASE_URL=sqlite+aiosqlite:////private/tmp/yui-goal-pr1-repro-unused.db PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -c pyproject.toml -p no:cacheprovider /private/tmp/yui_goal_pr1_review_test.py -q
```

- 同一瞬間で選択結果が一致するという検査が、UTC+9とUTC−7の**2ケースで失敗**。上記指摘の再現結果である。

別途、一時ディレクトリ内のDBを `YUI_DATABASE_URL` に指定して以下を実行した。

```sh
.venv/bin/alembic upgrade head
.venv/bin/alembic check
.venv/bin/alembic downgrade f76fb885504f
.venv/bin/alembic upgrade head
```

- すべて成功。`alembic check` は `No new upgrade operations detected.`。
- 既存テストには、旧形式のデータを投入してheadへ移行する確認も含まれる。

### 対象外・未確認

- 実モデル（Ollama）での評価は行っていない。PR1は会話経路に接続されていない。
- データを投入した状態でのPR1 downgrade後の保持確認は未実施。
- 大量データでの索引性能は未計測。
- 後続PRで実装するAPI・画面・会話経路との統合動作は対象外。
- レビュー時にソースコード、設定、ドキュメント、通常利用のDBは変更していない。その後のユーザーの保存依頼により、本レビュー資料のみを新規作成した。

## PR1：目標の器 — 第2回レビュー・対応確認（2026-09-08）

### 結論

**PR1「目標の器」は承認してよく、次のPR2へ進める。** 前回の日時誤判定は解消しており、今回の再レビューで進行を妨げる新たな不具合は見つからなかった。この承認はPR1の範囲であり、フェーズ4全体の完了承認ではない。

### 対象

- [Claudeの対応記録](../claude/phase4_review_response.md)と、修正後の `backend/app/agent/goal.py`、`backend/tests/test_goal.py` を照合した。
- 前回の指摘の修正、再発防止テスト、配信時の相手指定に関する追加テストを確認した。

### 前回の指摘への評価

#### 指摘1：現在時刻のタイムゾーンによる実行日時の誤判定 — 解消

- **確認区分：** 実行して再現確認（修正後は不一致が解消）
- **コード確認：** `as_utc()` を保存側の `normalize_due_at` と比較側の `active_goals` の両方で使用している。タイムゾーン付き日時はUTCへ変換し、naive日時はUTCとして解釈する。
- **独立した再確認：** 前回こちらで作成した `/private/tmp/yui_goal_pr1_review_test.py` を修正後のコードに対して再実行した。UTC+9で基準日時より前の目標を選んでしまうケースと、UTC−7で基準日時を過ぎた目標を落とすケースの両方が成功し、同じ瞬間で選択結果が一致した。
- **再発防止テスト：** 追加された `test_the_same_moment_is_judged_the_same_in_any_timezone` はUTC・UTC+9・UTC−7で基準日時の前後を検査し、`now == due_at` の境界とnaive日時も検査している。前回指摘への対応として適切。

### 追加テストと対応記録の評価

- `test_a_public_goal_about_someone_stays_with_that_person` は、公開目標でも相手不明・別人の場合には選択せず、対象本人の場合だけ選択することを確認している。配信時の公開範囲と相手指定を区別する検証になっている。
- 今回再実行した既存テストの件数とruffの結果は、対応記録の「208件成功・ruff成功」と一致した。
- 対応記録にある「正規化を戻すと追加テストが失敗する」という作業自体は再実施していない。代わりに、前回失敗した独立の再現テストが修正後に成功することを確認した。

### 実行した確認

バックエンドディレクトリから、一時DBを使って実行した。

```sh
YUI_DATABASE_URL=sqlite+aiosqlite:////private/tmp/yui-pr1-rereview-unused.db PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
.venv/bin/ruff check app tests --no-cache
YUI_DATABASE_URL=sqlite+aiosqlite:////private/tmp/yui-pr1-rereview-unused.db PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -c pyproject.toml -p no:cacheprovider /private/tmp/yui_goal_pr1_review_test.py -q
```

- バックエンド既存テスト：**208件成功**（目標関連11件を含む）。
- ruff：成功。
- 前回の独立した日時再現テスト：**2件成功**。

### 次のタスク・未確認事項

- PR2では、`mark_for_review` を記憶の訂正・削除経路へ接続する作業へ進める。
- 実モデル評価は未実施。PR1は会話経路に未接続であり、今回の承認を妨げる未確認事項とは判断しない。
- フェーズ4全体の統合動作、後続PRの機能は今回の承認対象外。
- ソースコード、設定、通常利用のDBは変更していない。ユーザーの保存依頼により、本資料に再レビュー結果のみを追記した。
