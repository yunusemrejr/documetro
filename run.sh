#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
PID_FILE="$RUNTIME_DIR/documetro.pid"
LOG_FILE="$RUNTIME_DIR/server.log"
URL_FILE="$RUNTIME_DIR/server.url"
APP_IMPORT="documetro.app:create_app"
APP_PATTERN='documetro-server|documetro.app:create_app'
DEFAULT_HOST="${DOCUMETRO_HOST:-127.0.0.1}"
DEFAULT_PORT="${DOCUMETRO_PORT:-8421}"
MAX_PORT="${DOCUMETRO_MAX_PORT:-8499}"
COMMAND="${1:-start}"

mkdir -p "$RUNTIME_DIR"

resolve_python() {
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    return
  fi
  printf '%s\n' "${PYTHON_BIN:-python3}"
}

PYTHON_BIN="$(resolve_python)"

ensure_environment() {
  if [[ ! -x "$ROOT_DIR/.venv/bin/python" && "${DOCUMETRO_USE_VENV:-1}" == "1" ]]; then
    python3 -m venv --system-site-packages "$ROOT_DIR/.venv"
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  fi
}

python_requirements_ok() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
required = ["fastapi", "uvicorn", "numpy", "scipy", "multipart", "bs4", "lxml", "xlrd"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
PY
}

install_python_requirements() {
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
}

check_dependencies() {
  local missing=()
  for command in pdftotext unzip file; do
    if ! command -v "$command" >/dev/null 2>&1; then
      missing+=("$command")
    fi
  done
  if (( ${#missing[@]} )); then
    printf 'Missing required system tools: %s\n' "${missing[*]}" >&2
    exit 1
  fi

  if ! python_requirements_ok 2>/dev/null; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1 && "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
      printf 'Installing missing Python packages from requirements.txt...\n'
      install_python_requirements
      python_requirements_ok
    else
      printf 'Python dependencies are missing and pip is unavailable.\n' >&2
      exit 1
    fi
  fi
}

find_available_port() {
  "$PYTHON_BIN" - <<PY
import socket
host = "${DEFAULT_HOST}"
start = int("${DEFAULT_PORT}")
end = int("${MAX_PORT}")
for port in range(start, end + 1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit("No available port found in range")
PY
}

wait_for_health() {
  local port="$1"
  DOCUMETRO_HEALTH_PORT="$port" "$PYTHON_BIN" - <<'PY'
import os
import time
import urllib.request

port = os.environ["DOCUMETRO_HEALTH_PORT"]
url = f"http://127.0.0.1:{port}/api/health"
for _ in range(80):
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.25)
raise SystemExit("Server did not become healthy in time")
PY
}

stop_running_instances() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      sleep 0.5
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$PID_FILE"
  fi

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "$pid" >/dev/null 2>&1 || true
  done < <(ps -eo pid=,args= | awk '/documetro-server|documetro\.app:create_app/ && !/awk/ {print $1}')
}

cleanup_runtime() {
  rm -f "$URL_FILE"
  find /tmp -maxdepth 1 -type d -name 'documetro-session-*' -exec rm -rf {} + 2>/dev/null || true
}

start_app() {
  ensure_environment
  check_dependencies
  stop_running_instances
  cleanup_runtime

  local port
  port="$(find_available_port)"
  : > "$LOG_FILE"
  (
    cd "$ROOT_DIR"
    export DOCUMETRO_HOST="$DEFAULT_HOST"
    export DOCUMETRO_PORT="$port"
    exec -a documetro-server "$PYTHON_BIN" -m uvicorn "$APP_IMPORT" --factory --host "$DEFAULT_HOST" --port "$port" --log-level warning
  ) >>"$LOG_FILE" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  wait_for_health "$port"
  printf 'http://%s:%s\n' "$DEFAULT_HOST" "$port" > "$URL_FILE"
  printf 'Documetro is running at http://%s:%s\n' "$DEFAULT_HOST" "$port"
  printf 'Log file: %s\n' "$LOG_FILE"
}

stop_app() {
  stop_running_instances
  cleanup_runtime
  printf 'Documetro stopped.\n'
}

status_app() {
  if [[ -f "$URL_FILE" ]]; then
    printf 'URL: %s\n' "$(cat "$URL_FILE")"
  else
    printf 'URL: not running\n'
  fi
  if [[ -f "$PID_FILE" ]]; then
    printf 'PID: %s\n' "$(cat "$PID_FILE")"
  else
    printf 'PID: none\n'
  fi
  printf 'Log file: %s\n' "$LOG_FILE"
}

case "$COMMAND" in
  start)
    start_app
    ;;
  stop)
    stop_app
    ;;
  restart)
    stop_app
    start_app
    ;;
  status)
    status_app
    ;;
  *)
    printf 'Usage: %s [start|stop|restart|status]\n' "$0" >&2
    exit 1
    ;;
esac

