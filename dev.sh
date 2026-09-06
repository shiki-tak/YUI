#!/usr/bin/env bash
#
# 開発用のプロセスをまとめて起動する。
#
#   ./dev.sh            バックエンドとフロントエンドを起動する
#   ./dev.sh --no-open  ブラウザを開かない
#
# 起動前に、このプロジェクトの既存プロセスがあれば停止する。
# Ctrl+C で両方まとめて止まる。ログは logs/ に残る。
# 環境変数 BACKEND_PORT / FRONTEND_PORT でポートを変えられる。
#
# VOICEVOX Engine は、このスクリプトでは起動しない（インストール方法が
# 環境によって違うため）。起動していれば使い、無ければ警告して続行する。
#
# macOS 標準の bash 3.2 で動くように書いている（wait -n や連想配列は使わない）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
LOG_DIR="$ROOT/logs"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
OPEN_BROWSER=1

for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_BROWSER=0 ;;
    -h|--help) sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "不明な引数: $arg" >&2; exit 2 ;;
  esac
done

GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
info() { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
fail() { printf '%s✗%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

# 起動したプロセスの PID。bash 3.2 では空配列の展開に注意が必要なため、
# 配列ではなく空白区切りの文字列で持つ。
PIDS=""
LAST_PID=""

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  echo
  info "停止しています..."
  for pid in $PIDS; do
    # 子プロセス（uvicorn の reloader や vite）も一緒に止める。
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  ok "停止しました"
  exit "$status"
}

# 注意：$( ) で呼ぶとサブシェルになり PID を親が管理できなくなる。
# 呼び出し側は LAST_PID を参照する。
start_process() {
  local label="$1" dir="$2"
  shift 2
  ( cd "$dir" && exec "$@" ) > "$LOG_DIR/$label.log" 2>&1 &
  LAST_PID=$!
  PIDS="$PIDS $LAST_PID"
}

port_in_use() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# そのプロセスがこのリポジトリのものか。vite のコマンドラインには絶対パスが
# 出ないため、コマンドラインと作業ディレクトリの両方で判定する。
process_is_ours() {
  local pid="$1" cmd cwd
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$cmd" in
    *"$ROOT"*) return 0 ;;
  esac
  cwd="$(lsof -a -d cwd -p "$pid" -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)"
  case "$cwd" in
    "$ROOT"|"$ROOT"/*) return 0 ;;
  esac
  return 1
}

# 起動しているポートを握っているプロセスのうち、このリポジトリのものを止める。
# 他のプロセスが使っている場合は、勝手に止めずに終了する。
stop_existing() {
  local port pid ppid pids targets="" i
  for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
    pids="$(lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    for pid in $pids; do
      if process_is_ours "$pid"; then
        targets="$targets $pid"
        # uvicorn --reload は親が子を再起動するため、親も対象にする。
        ppid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')"
        if [ -n "$ppid" ] && [ "$ppid" != "1" ] && process_is_ours "$ppid"; then
          targets="$targets $ppid"
        fi
      else
        fail "ポート $port を、このプロジェクト以外のプロセスが使っています:
    $(ps -p "$pid" -o command= 2>/dev/null)
  停止するか、BACKEND_PORT / FRONTEND_PORT で別のポートを指定してください。"
      fi
    done
  done

  [ -n "$targets" ] || return 0
  warn "既存のプロセスを停止します"
  for pid in $targets; do
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  done

  for i in $(seq 1 20); do
    if ! port_in_use "$BACKEND_PORT" && ! port_in_use "$FRONTEND_PORT"; then
      return 0
    fi
    sleep 0.25
  done
  for pid in $targets; do kill -9 "$pid" 2>/dev/null || true; done
  sleep 1
}

wait_for_http() {
  local url="$1" label="$2" pid="$3" i
  for i in $(seq 1 60); do
    if curl -fs -m 2 -o /dev/null "$url" 2>/dev/null; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      fail "$label の起動に失敗しました。logs/$label.log を確認してください。"
    fi
    sleep 0.5
  done
  fail "$label が応答しません（$url）。logs/ を確認してください。"
}

# --- 事前確認 ---------------------------------------------------------------

mkdir -p "$LOG_DIR"
info "起動前の確認"

[ -x "$BACKEND/.venv/bin/python" ] || fail \
  "backend/.venv がありません。次を実行してください:
    cd backend && python3 -m venv .venv && .venv/bin/pip install -e \".[dev]\""

[ -d "$FRONTEND/node_modules" ] || fail \
  "frontend/node_modules がありません。次を実行してください:
    cd frontend && npm install"

if [ ! -f "$BACKEND/.env" ]; then
  cp "$BACKEND/.env.example" "$BACKEND/.env"
  warn "backend/.env が無かったため .env.example から作成しました"
fi

read_env() {
  grep -E "^$1=" "$BACKEND/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true
}
OLLAMA_HOST="$(read_env YUI_OLLAMA_HOST)"
OLLAMA_MODEL="$(read_env YUI_OLLAMA_MODEL)"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"

curl -fs -m 3 -o /dev/null "$OLLAMA_HOST/api/tags" 2>/dev/null \
  || fail "Ollama に接続できません（$OLLAMA_HOST）。\`ollama serve\` を起動してください。"
curl -fs -m 5 "$OLLAMA_HOST/api/tags" 2>/dev/null | grep -q "\"${OLLAMA_MODEL%%:*}" \
  || fail "モデル $OLLAMA_MODEL がありません。次を実行してください:
    ollama pull $OLLAMA_MODEL"
ok "Ollama：$OLLAMA_MODEL"

# VOICEVOX は任意。無くても文字での会話は続けられるため、止めずに続行する。
VOICEVOX_HOST="$(read_env YUI_VOICEVOX_HOST)"
VOICEVOX_HOST="${VOICEVOX_HOST:-http://localhost:50021}"
SPEECH_ENABLED="$(read_env YUI_SPEECH_ENABLED)"
if [ "${SPEECH_ENABLED:-true}" = "false" ]; then
  warn "音声合成：無効（YUI_SPEECH_ENABLED=false）"
elif VOICEVOX_VERSION="$(curl -fs -m 3 "$VOICEVOX_HOST/version" 2>/dev/null)"; then
  ok "VOICEVOX：$(printf '%s' "$VOICEVOX_VERSION" | tr -d '\"')（$VOICEVOX_HOST）"
else
  warn "VOICEVOX に接続できません（$VOICEVOX_HOST）。音声なしで起動します。
  エンジンを起動してから開き直すと音声を使えます。"
fi

stop_existing

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if port_in_use "$port"; then
    fail "ポート $port を解放できませんでした:
    $(ps -p "$(lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -1)" -o command= 2>/dev/null)"
  fi
done

# DB のテーブル定義を最新にする（適用済みなら何もしない）。
( cd "$BACKEND" && .venv/bin/alembic upgrade head ) > "$LOG_DIR/alembic.log" 2>&1 \
  || fail "DB のマイグレーションに失敗しました。logs/alembic.log を確認してください。"
ok "DB：最新の状態"

# --- 起動 -------------------------------------------------------------------

trap cleanup INT TERM EXIT

info ""
info "起動しています"

start_process backend "$BACKEND" \
  .venv/bin/uvicorn app.main:app --reload --port "$BACKEND_PORT" --log-level warning
BACKEND_PID="$LAST_PID"
wait_for_http "http://127.0.0.1:$BACKEND_PORT/api/health" backend "$BACKEND_PID"
ok "バックエンド    http://localhost:$BACKEND_PORT/docs"

start_process frontend "$FRONTEND" \
  node_modules/.bin/vite --port "$FRONTEND_PORT" --strictPort
FRONTEND_PID="$LAST_PID"
wait_for_http "http://127.0.0.1:$FRONTEND_PORT/" frontend "$FRONTEND_PID"
ok "フロントエンド  http://localhost:$FRONTEND_PORT"

if [ "$OPEN_BROWSER" -eq 1 ] && command -v open >/dev/null 2>&1; then
  open "http://localhost:$FRONTEND_PORT"
fi

info ""
info "${DIM}Ctrl+C で両方を停止します。${OFF}"
info ""

# ログを流しながら、どちらかが落ちるまで待つ。
tail -n 0 -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" &
PIDS="$PIDS $!"

while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done
warn "いずれかのプロセスが終了しました"
