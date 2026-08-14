#!/usr/bin/env bash
# Rehearsal: mock robot + ingress/pipeline with the arm sink, all local.
# Nothing physical moves. This is the session to practise the drills in.
#
#   ./tmux-sim.sh              mock robot + run_arm --sim + mock publisher
#   ./tmux-sim.sh --headset    same, but no mock publisher (use a real Quest)
set -euo pipefail

SESSION=orca-sim
TRON_ARM="${TRON_ARM:-$HOME/tron_arm}"
ORCA_TELEOP="${ORCA_TELEOP:-$HOME/orca_teleop}"
MODEL_PATH="${MODEL_PATH:-}"
HEADSET=0
[[ "${1:-}" == "--headset" ]] && HEADSET=1

tmux has-session -t "$SESSION" 2>/dev/null && { tmux attach -t "$SESSION"; exit 0; }

tmux new-session  -d -s "$SESSION" -n robot  -c "$TRON_ARM"
tmux send-keys    -t "$SESSION:robot" \
    "uv run python -m tron_arm.mock_robot --port 5000" C-m

# Give the mock a moment to bind before the sink tries to connect.
tmux new-window   -t "$SESSION" -n arm -c "$TRON_ARM"
tmux send-keys    -t "$SESSION:arm" \
    "sleep 2 && uv run python tools/run_arm.py --sim ${MODEL_PATH:+--model-path $MODEL_PATH} --clutch keyboard" C-m

if [[ $HEADSET -eq 0 ]]; then
    tmux new-window -t "$SESSION" -n publisher -c "$ORCA_TELEOP"
    tmux send-keys  -t "$SESSION:publisher" \
        "sleep 4 && uv run python -m orca_teleop.ingress.metaquest.mock_publisher --server localhost:50051" C-m
fi

# A spare shell for tron2_cli pokes (pose, joints, notify tailing).
tmux new-window -t "$SESSION" -n shell -c "$TRON_ARM"

tmux select-window -t "$SESSION:arm"
echo "Session '$SESSION': robot | arm | ${HEADSET:+}publisher | shell"
echo "The clutch is HOLD-to-engage: nothing moves unless a key is held in the 'arm' window."
tmux attach -t "$SESSION"
