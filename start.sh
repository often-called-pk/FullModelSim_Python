#!/usr/bin/env sh
# One command setup for Linux. Extra arguments go to start.py.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then exec python3 start.py "$@"; fi
if command -v uv >/dev/null 2>&1; then UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"
else
    echo "error: Python 3 and uv not found"
    echo "fix: install Python 3.10 or newer (Debian/Ubuntu: sudo apt install python3), then run this script again"
    exit 1
fi
exec "$UV" run --no-project python start.py "$@"
