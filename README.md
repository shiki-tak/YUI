# yui

同じ人格と記憶を持つ AI VTuber キャラクター。普段は開発者と会話し、YouTube では視聴者と交流し、
その経験を次の話題選びや行動に反映することを目指します。

設計は [design/ai_vtuber_architecture.md](design/ai_vtuber_architecture.md)（全体構成）と
[design/ai_vtuber_development_phases.md](design/ai_vtuber_development_phases.md)（開発フェーズ）にあります。

## いまの状態：フェーズ1（文字会話 MVP）

設定した人格で会話し、再起動を越えて過去の経験を使って答えられる状態までを実装しています。

- React の文字チャット画面
- FastAPI から Ollama への接続（LLM 接続モジュール経由で後から差し替え可能）
- 基本の人格設定
- 会話履歴と長期記憶の保存（別のテーブルとして分ける）
- 相手・時刻・キーワードによる記憶検索
- 記憶の確認・訂正・削除・復元と、変更履歴
- 会話終了時の振り返りによる記憶候補の抽出と、開発者による採用・却下
- 実行記録（モデルの版、生成設定、参照した記憶、応答時間、トークン数）
- 理想の返答の記録

フェーズ2以降（音声・アバター・検索・配信・学習）は未着手です。

## 構成

```
backend/    FastAPI。人格・記憶・会話進行・振り返り・実行記録
  app/
    agent/    自作エージェント（記憶検索、プロンプト構築、会話進行、振り返り）
    llm/      LLM 接続モジュール（現在は Ollama。Claude・MLX を後から足す）
    api/      HTTP API
  alembic/  DB のテーブル定義の変更履歴
frontend/   React + TypeScript + Vite。開発用の操作画面
design/     設計書
```

## 準備

### 1. Ollama

```sh
ollama pull qwen3:8b
```

別のモデルを使う場合は `backend/.env` の `YUI_OLLAMA_MODEL` を書き換えます。

### 2. バックエンド

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/alembic upgrade head      # SQLite にテーブルを作る
.venv/bin/uvicorn app.main:app --reload
```

`http://localhost:8000/docs` で API を確認できます。
`http://localhost:8000/api/health` はモデルに接続できるかを返します。

### 3. フロントエンド

```sh
cd frontend
npm install
npm run dev
```

`http://localhost:5173` を開きます。`/api` は Vite の proxy 経由でバックエンドへ渡ります。

## 動作確認

設計書の確認例をそのまま実行できます。

1. 「写真を撮るのが好き。次は山で撮った写真の話をしよう」と話しかける。
2. 「終了して振り返る」を押し、出てきた記憶の候補を採用する。
3. 新しい会話で「前に何を話す約束をしたっけ？」と尋ね、約束を踏まえて答えることを確認する。
4. 返答の「根拠」を開き、どの記憶を渡したか、どの発言が根拠かを確認する。

## テスト

```sh
cd backend && .venv/bin/python -m pytest
```

LLM は共通インターフェース越しに差し替えるため、Ollama を起動していなくても実行できます。

## 設計上の約束

実装で崩さないようにしている区別です。

- 会話履歴と長期記憶は別。会話ログをそのまま記憶に昇格させない。
- 事実と推測を分け、記憶には根拠の発言を残して訂正できるようにする。
- 生成しただけの文章と、実際に相手へ届いた文章を混同しない（`delivery_state`）。
- 人物は表示名ではなく、入力元とその識別子で同定する。
- 非公開の記憶を配信モードで参照しない。
- 記憶の更新候補は、開発者が確認してから採用する。
