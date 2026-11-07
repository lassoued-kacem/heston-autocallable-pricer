#!/usr/bin/env bash
# Stop-hook test runner. Exit 0 = allow stop, exit 2 = block stop and feed stderr back.
set -u
cd "$(dirname "$0")/.." || exit 0

payload="$(cat 2>/dev/null || true)"
active="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
print("true" if d.get("stop_hook_active") else "false")
' 2>/dev/null || echo false)"

if [ "$active" = "true" ]; then
    echo "check.sh: stop_hook_active is true; skipping tests to avoid an infinite loop." >&2
    exit 0
fi

if ! ls tests/test_*.py >/dev/null 2>&1; then
    exit 0
fi

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    PY=python3
fi

out="$("$PY" -m pytest -q 2>&1)"
status=$?
if [ $status -ne 0 ]; then
    printf 'Test suite failed (pytest exit %s):\n%s\n' "$status" "$out" >&2
    exit 2
fi
exit 0
