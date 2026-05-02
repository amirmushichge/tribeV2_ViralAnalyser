#!/usr/bin/env bash
set -u

NO_BROWSER=0
if [[ "${1:-}" == "--no-browser" || "${1:-}" == "-n" ]]; then
  NO_BROWSER=1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$APP_DIR/.venv/bin/python"
REQUIREMENTS_FILE="$APP_DIR/requirements.txt"
BOOTSTRAP_SCRIPT="$APP_DIR/bootstrap_models.py"
BOOTSTRAP_READY_FILE="$APP_DIR/.bootstrap/models-ready.json"
HOST_ADDRESS="127.0.0.1"
PREFERRED_PORT=8000

cd "$APP_DIR" || exit 1

stop_with_message() {
  printf "\n%s\n\n" "$1" >&2
  exit 1
}

find_python() {
  if [[ -n "${TRIBE_PYTHON:-}" ]] && command -v "$TRIBE_PYTHON" >/dev/null 2>&1; then
    printf "%s" "$TRIBE_PYTHON"
    return 0
  fi

  for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        printf "%s" "$candidate"
        return 0
      fi
    fi
  done

  return 1
}

test_port_free() {
  "$VENV_PYTHON" - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

host = "127.0.0.1"
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, port))
    except OSError:
        raise SystemExit(1)
PY
}

test_tribe_app_url() {
  "$VENV_PYTHON" - "$1" <<'PY' >/dev/null 2>&1
import sys
from urllib.error import URLError
from urllib.request import urlopen

port = int(sys.argv[1])
try:
    with urlopen(f"http://127.0.0.1:{port}", timeout=2) as response:
        content = response.read().decode("utf-8", errors="replace")
except (OSError, URLError):
    raise SystemExit(1)

if "TRIBE Review MVP" in content or "Predict virality with Meta TRIBE v2" in content:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

get_launch_port() {
  if test_tribe_app_url "$PREFERRED_PORT"; then
    printf "%s" "$PREFERRED_PORT"
    return 0
  fi

  if test_port_free "$PREFERRED_PORT"; then
    printf "%s" "$PREFERRED_PORT"
    return 0
  fi

  for port in {8001..8010}; do
    if test_port_free "$port"; then
      printf "%s" "$port"
      return 0
    fi
  done

  return 1
}

open_when_ready() {
  local launch_url="$1"
  "$VENV_PYTHON" - "$launch_url" <<'PY' &
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1]
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with urlopen(url, timeout=2) as response:
            if 200 <= response.status < 500:
                subprocess.Popen(["open", url])
                raise SystemExit(0)
    except (OSError, URLError):
        time.sleep(0.75)
PY
}

if [[ ! -x "$VENV_PYTHON" ]]; then
  PYTHON_CMD="$(find_python)" || stop_with_message "Python 3.11 or newer is required. Install it with Homebrew or set TRIBE_PYTHON to a compatible interpreter."
  printf "\nCreating local Python environment: .venv\n"
  "$PYTHON_CMD" -m venv "$APP_DIR/.venv" || stop_with_message "Could not create .venv."
fi

if ! "$VENV_PYTHON" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  [[ -f "$REQUIREMENTS_FILE" ]] || stop_with_message "requirements.txt not found. Download the full repository archive again."
  printf "\nInstalling Python dependencies. First run can take several minutes.\n"
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel || stop_with_message "Could not upgrade pip inside .venv."
  "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE" || stop_with_message "Could not install dependencies from requirements.txt."
fi

if [[ ! -f "$BOOTSTRAP_READY_FILE" ]]; then
  [[ -f "$BOOTSTRAP_SCRIPT" ]] || stop_with_message "bootstrap_models.py not found. Download the full repository archive again."
  "$VENV_PYTHON" "$BOOTSTRAP_SCRIPT" || stop_with_message "Initial model setup failed. Fix the issue above, then run ./start_macos.sh again."
  printf "\nInitial setup finished successfully.\nRun ./start_macos.sh again to start the app.\n"
  exit 0
fi

PORT="$(get_launch_port)" || stop_with_message "Could not find a free port from 8000 to 8010."
URL="http://$HOST_ADDRESS:$PORT"

if test_tribe_app_url "$PORT"; then
  printf "\nTRIBE Review already running on %s\n" "$URL"
  if [[ "$NO_BROWSER" -eq 0 ]]; then
    open "$URL"
  fi
  exit 0
fi

if [[ "$(uname -s)" == "Darwin" && "${TRIBE_ENABLE_MPS:-}" =~ ^(1|true|yes)$ ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

printf "\nStarting TRIBE Review on %s\n" "$URL"
if [[ "$NO_BROWSER" -eq 0 ]]; then
  open_when_ready "$URL"
fi

exec "$VENV_PYTHON" -m uvicorn app:app --host "$HOST_ADDRESS" --port "$PORT"
