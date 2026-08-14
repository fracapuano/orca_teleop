#!/usr/bin/env bash
# Preflight for a powered TRON 2 + ORCA run.
#
#   ./preflight.sh                 check against the mock (default)
#   ./preflight.sh --robot HOST    check against real hardware
#   ./preflight.sh --skip-tests    skip the upstream suite (slow)
#
# Read-only: pings, opens a WebSocket, lists symlinks, runs tests. Commands no
# motion. Exit 0 = all green.
set -uo pipefail

TRON_ARM="${TRON_ARM:-$HOME/tron_arm}"
ORCA_TELEOP="${ORCA_TELEOP:-$HOME/orca_teleop}"
ROBOT_HOST=""
SKIP_TESTS=0
fail=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot) ROBOT_HOST="${2:-}"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

green() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
red()   { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=1; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if [[ -n "$ROBOT_HOST" ]]; then
    TARGET_HOST="$ROBOT_HOST"; TARGET_URL="ws://$ROBOT_HOST:5000"; MODE="REAL HARDWARE"
else
    TARGET_HOST="127.0.0.1"; TARGET_URL="ws://127.0.0.1:5000"; MODE="mock"
fi

echo "Preflight — target: $TARGET_URL  [$MODE]"

# 1. Network -------------------------------------------------------------
head_ "1. Network"
if [[ -n "$ROBOT_HOST" ]]; then
    if ping -c 2 -W 2 "$TARGET_HOST" >/dev/null 2>&1; then
        rtt=$(ping -c 3 -W 2 "$TARGET_HOST" 2>/dev/null | tail -1 | awk -F'/' '{print $5}')
        green "ping $TARGET_HOST OK (avg ${rtt:-?} ms)"
    else
        red "cannot ping $TARGET_HOST — check the cable, the subnet and the static IP"
    fi
else
    green "mock target, no ping needed"
fi

# 2. WebSocket + notify_robot_info ---------------------------------------
head_ "2. Robot WebSocket"
PY_BIN="$TRON_ARM/.venv/bin/python"
[[ -x "$PY_BIN" ]] || PY_BIN="python3"
"$PY_BIN" - "$TARGET_URL" <<'PY'
import asyncio, json, sys
sys.path.insert(0, __import__("os").environ.get("TRON_ARM", ""))
url = sys.argv[1]

async def main() -> int:
    try:
        from websockets.asyncio.client import connect
    except Exception as exc:
        print(f"  ! websockets not importable ({exc}); is the venv built?")
        return 1
    try:
        async with connect(url, open_timeout=5) as ws:
            print(f"  \033[32m✓\033[0m connected to {url}")
            # accid is learned from notify_robot_info, which the robot
            # broadcasts at 1 Hz. Nothing works without it.
            try:
                for _ in range(12):
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    msg = json.loads(raw)
                    if msg.get("title") == "notify_robot_info":
                        print(f"  \033[32m✓\033[0m notify_robot_info seen "
                              f"(accid={msg.get('accid')!r}, "
                              f"state={msg.get('data', {}).get('state', '?')})")
                        return 0
                print("  \033[31m✗\033[0m connected but no notify_robot_info in 12 messages")
                return 1
            except asyncio.TimeoutError:
                print("  \033[31m✗\033[0m connected but silent — no notify_robot_info")
                return 1
    except Exception as exc:
        print(f"  \033[31m✗\033[0m cannot open {url}: {exc}")
        return 1

raise SystemExit(asyncio.run(main()))
PY
[[ $? -eq 0 ]] || fail=1

# 4. Config sanity -------------------------------------------------------
head_ "3. tron_arm config"
"$PY_BIN" - <<PY
import sys
sys.path.insert(0, "$TRON_ARM")
try:
    from tron_arm.config import load_config
    c = load_config()
except Exception as exc:
    print(f"  \033[31m✗\033[0m config will not load: {exc}"); raise SystemExit(1)
print(f"  \033[32m✓\033[0m config loads: scale={c.scale}, format={c.servop.format}, "
      f"send_both={c.servop.send_both}, rate={c.servop.rate_hz} Hz")
print(f"  \033[32m✓\033[0m clamps: margin={c.workspace.margin_m} m, "
      f"max step={c.max_step[0]:.4f} m / {c.max_step[1]:.4f} rad per tick")
if c.scale != 0.5:
    print(f"  \033[33m!\033[0m scale is {c.scale}, not the rehearsed 0.5")
if c.mapping.translation_frame != "body":
    print(f"  \033[33m!\033[0m translation_frame={c.mapping.translation_frame}")
PY
[[ $? -eq 0 ]] || fail=1

# 5. Upstream suite ------------------------------------------------------
head_ "4. Test suites"
if [[ $SKIP_TESTS -eq 1 ]]; then
    warn "skipped (--skip-tests)"
else
    # Call each venv's python DIRECTLY, never `uv run`: uv run performs a
    # dependency sync first, so a preflight would mutate the environment and
    # block on a download. A preflight must be read-only and offline-safe.
    if [[ -d "$ORCA_TELEOP" ]]; then
        up_py="$ORCA_TELEOP/.venv/bin/python"
        if [[ -x "$up_py" ]]; then
            # The --ignore is mandatory: that test hangs (pre-existing, CLAUDE.md).
            if (cd "$ORCA_TELEOP" && "$up_py" -m pytest tests/ \
                    --ignore=tests/test_recording_ready.py -q \
                    >/tmp/preflight_upstream.log 2>&1); then
                green "orca_teleop suite green ($(tail -1 /tmp/preflight_upstream.log))"
            else
                red "orca_teleop suite FAILED — see /tmp/preflight_upstream.log"
            fi
        else
            red "no venv at $up_py — run 'uv sync --extra test --extra metaquest' there first"
        fi
    else
        red "no orca_teleop checkout at $ORCA_TELEOP"
    fi
    ta_py="$TRON_ARM/.venv/bin/python"
    if [[ -x "$ta_py" ]]; then
        if (cd "$TRON_ARM" && "$ta_py" -m pytest -q -m "not slow" \
                >/tmp/preflight_tron.log 2>&1); then
            green "tron_arm suite green ($(tail -1 /tmp/preflight_tron.log))"
        else
            red "tron_arm suite FAILED — see /tmp/preflight_tron.log"
        fi
    else
        red "no venv at $ta_py"
    fi
fi

# Verdict ----------------------------------------------------------------
head_ "Verdict"
if [[ $fail -eq 0 ]]; then
    green "preflight PASSED"
    echo
    echo "Next: README.md, \"Running teleoperation\"."
else
    red "preflight FAILED — do not power the arm"
fi
exit $fail
