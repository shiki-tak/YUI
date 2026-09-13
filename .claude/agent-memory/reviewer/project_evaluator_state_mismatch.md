---
name: evaluator-state-mismatch
description: backend/app/evaluation/runner.py の会話状態チェックは、実装と「誰の・いつの」状態を見るかがずれて誤判定する。expect_conversation_states / expect_no_new_question を触る diff は評価器を FakeLLM で実際に流して確かめる
metadata:
  type: project
---

`backend/app/evaluation/runner.py` の `_check_turn` と `_conversation_states_for_speaker` は、
実装（`conversation.py`）と別の条件・別の時点・別の話者で会話状態を見て、実装が正しくても
シナリオを失敗にする不一致が v0.2 で 3 回続いた（2026-09-13 時点）。
- PR2 1回目：`has_open_state(kind=question_to_yui)` で見て、実装の `responded_message_id is None` と食い違う
- PR2 2回目：ターン**後**の状態で `has_unanswered_question_to_yui` を再判定し、返答自身が `responded` を埋めた後なので「対象外」分岐に到達しない → `run.options["checks"]` をそのまま使う形に修正
- PR3：`target_speaker_id = say の話者` で絞るため、A 宛の状態を B の say の後に見るシナリオ 20 が常に「行が無い」

**Why:** 評価器は PR の受け入れ条件（§7 のシナリオを 3 回）そのもの。評価器側の誤判定は
「実装が悪い」と読まれて無駄な修正を誘発するか、逆に人手判定で片付けられて欠陥を隠す。
pytest は ID や返答を直接埋め込むため、この種の不一致は pytest では見つからない。

**How to apply:** 評価器の期待値・チェックを足す diff では、
- 実装がどの時点（生成前／後）・どの話者向けの状態で判定したかを確認し、評価器は実装の記録
  （`RunRecord.options`）を再利用できるならそれを使う
- 複数話者のシナリオは、期待する状態の target が say の話者と一致するかを見る
- 該当シナリオを `run_scenarios` に FakeLLM で実際に流し、期待が満たせる形になっているかを確かめる
  （`tests/test_evaluation.py` の `run_scenarios(..., llm=fake_llm, settings=get_settings())` の形）
関連：[[surface-regex-false-positives]]
