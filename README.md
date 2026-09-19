# yui

対話とYouTube Live配信を行うAIキャラクター「YUI」。

設計は [docs/design/yui_stream_first_design.md](docs/design/yui_stream_first_design.md) にあります（唯一の設計基準。コンポーネントと責務、イベント配送、発話方針、記憶の3層構成、実装順、初配信の合格条件を1冊にまとめています）。人格の構成要素の参考研究は [docs/research/components-of-artificial-personhood.md](docs/research/components-of-artificial-personhood.md) にあります。

## いまの状態

設計を作り直し、実装を一から積み上げ直す段階です。旧設計（人格形成を中心とした版）の実装は削除し、イベント駆動アーキテクチャに必要な最小限の土台だけを残しています。

次に着手するのは、設計書「10. 実装順」の **S0：入出力の導通**（共通セッション・接続・人物IDのスキーマ、YouTube認証・コメント受信、模擬イベント、`/broadcast`、VOICEVOXとPNGアバター、停止）です。

## 構成

```
backend/    FastAPI。単一プロセス・単一ワーカー
  app/
    config.py   設定（VOICEVOX・DB・CORS）
    db.py       DB接続（SQLite + aiosqlite）
    main.py     FastAPIアプリの起点
    voice/      VOICEVOX Engineへの接続（音声合成ポート）
  tests/      音声合成モジュールの単体テスト
frontend/   React + TypeScript + Vite
  public/avatar/  PNGアバターの表情差分
  src/main.tsx    エントリーポイント（画面は未実装）
dev.sh      開発用のプロセスをまとめて起動する
docs/design/    設計書
docs/research/  人格の構成要素に関する参考研究
```

## 準備

### 1. バックエンド

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/uvicorn app.main:app --reload
```

`http://localhost:8000/docs` で API を確認できます。
`http://localhost:8000/api/health` は音声合成に接続できるかを返します。

### 2. フロントエンド

```sh
cd frontend
npm install
npm run dev
```

`http://localhost:5173` を開きます。`/api` は Vite の proxy 経由でバックエンドへ渡ります。

### 3. VOICEVOX Engine（任意）

読み上げに使います。起動していれば `./dev.sh` が接続を確認し、
無ければ警告を出して音声なしで起動します。

既定の接続先は `http://localhost:50021`、話者は **59（猫使ビィ／おちつき）** です。
`backend/.env` の `YUI_VOICEVOX_SPEAKER` で変えられます。エンジンを使わない環境では
`YUI_SPEECH_ENABLED=false` にします。

**生成した音声を公開する場合は「VOICEVOX:猫使ビィ」のクレジット表記が必要です。**
話者を変えるときは、その話者の利用規約と表記の条件を確認してください（規約は
起動中のエンジンの `/speaker_info` からも読めます）。

配布物はリポジトリに含めません。リポジトリのルートに展開した場合、書庫と
展開先（`macos-arm64/`、`voicevox/`）は `.gitignore` 済みです。展開には 7z が要ります。

```sh
brew install sevenzip
7zz x voicevox_engine-macos-arm64-<版>.7z.001
./macos-arm64/run                 # 127.0.0.1:50021 で待ち受ける
```

## まとめて起動する

準備が済んでいれば、リポジトリ直下のスクリプトで両方を起動できます。

```sh
./dev.sh
```

このプロジェクトのサーバーが既に動いていれば停止してから起動し直すので、
実行するたびに再起動になります。無関係なプロセスがポートを使っている場合は、
勝手に止めずにその旨を表示して終了します。

Ctrl+C で両方まとめて停止します。ログは `logs/` に出力されます。

```sh
./dev.sh --no-open           # ブラウザを開かない
BACKEND_PORT=8001 ./dev.sh   # ポートを変える
```

## テスト

```sh
cd backend && .venv/bin/python -m pytest   # バックエンド
cd frontend && npm run typecheck           # フロントエンド
```

音声合成は共通インターフェース（`SpeechClient`）越しに差し替えるため、
VOICEVOX を起動していなくてもバックエンドのテストを実行できます。
