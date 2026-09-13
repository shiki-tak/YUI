---
name: surface-regex-false-positives
description: conversation_state.py の表層規則（正規表現）は広げるたびに誤検出が出て、解釈(LLM)が無い構成では取り消されず会話の残り全部に効く。closing/deferral/question の検出器を触る diff は必ず「誤検出の側」の文で実測する
metadata:
  type: project
---

`backend/app/agent/conversation_state.py` の検出器（`detect_closing` / `detect_deferral_topic` / `detect_question`）は、
規則を広げるたびに誤検出か取りこぼしがレビューで実測されている（v0.2 PR1 で 3 回、PR2 で 2 回。2026-09-13 時点）。
- PR1 1回目：closing の `search` が「明日は実家に帰ります」に当たる → `fullmatch` 化
- PR1 2回目：改行を区切りにしない／「そうですか」が質問になる
- PR2 1回目：「〜前に」の全文 search で「毎晩寝る前にストレッチしてる」が closing → 依頼形を要求
- PR2 2回目：依頼形を要求しても「寝る前にストレッチするといいって本当？」「成績が落ちる前に対策を教えて」が closing

**Why:** 状態の取消（`withdrawn`）は解釈（LLM、PR3 以降・既定で無効）でしか起きないので、
規則の誤検出は `/end` まで open のまま残り、出力検査（再生成→定型文）が会話の残り全ターンに効く。
誤検出 1 件の被害が 1 ターンで済まない。「条件を 1 つ足せば直る」修正は、足した条件の側にも同型の穴が残ることが多い。

**How to apply:** これらの検出器を触る diff では、以下を実際に `detect_*` に通して確認する。
- 習慣・予定・一般論の報告と質問（「寝る前に本を読む」「寝る前に飲む薬を教えて」「明日は帰る予定」）
- 伝聞（「友達に『またね』と言った」）
- 最後の文以外に終了語・延期語がある複数文、改行区切り
- 「そうですか」「たぶん来るかな」等の相づち・独り言
- 設計のシナリオ文そのものだけでなく、その自然な言い換え（文を分ける等）が拾えるかも見る
また出力検査側では、`question_to_yui` は `responded_message_id` が入っても PR3 まで `open` のままなので、
「open の相手の質問」は `status == open and responded_message_id is None` で見るべき。
`has_open_state(kind=...)` だけの判定は、会話で一度でも相手が質問すると経路が死ぬ。
評価器（`runner.py`）はターン**後**の状態を見るため、実装が生成**前**に見た条件を同じ関数で再判定すると結果がずれる。
