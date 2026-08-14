#!/usr/bin/env bash
# POWERED RUN: ingress/pipeline with the arm sink against REAL HARDWARE.
#
#   ./tmux-robot.sh 10.192.1.2
#
# Asks for an explicit acknowledgement first. run_arm.py then asks for its own
# typed confirmation (CLAUDE.md hard rule 1) and refuses outright without
# --no-wrist.
set -euo pipefail

SESSION=orca-robot
TRON_ARM="${TRON_ARM:-$HOME/tron_arm}"
MODEL_PATH="${MODEL_PATH:-}"
ROBOT="${1:-}"

if [[ -z "$ROBOT" ]]; then
    echo "usage: $0 <robot-host>   e.g. $0 10.192.1.2" >&2
    exit 2
fi

echo
echo "  *** POWERED RUN against $ROBOT ***"
echo
echo "  Before starting:"
echo "    - workspace clear, hardware remote in a named person's hand"
echo "    - arms inside the workspace (tron2_cli movej-ready)"
echo "    - --no-wrist confirmed"
echo "    - the vendor e-stop is idle-only: the remote is the abort"
echo
read -r -p "  All of the above done? Type 'yes' to continue: " ack
[[ "$ack" == "yes" ]] || { echo "  aborted."; exit 2; }

echo
echo "  Running preflight..."
"$(dirname "$0")/preflight.sh" --robot "$ROBOT" --skip-tests || {
    echo "  preflight FAILED — not starting." >&2; exit 1; }

tmux has-session -t "$SESSION" 2>/dev/null && { tmux attach -t "$SESSION"; exit 0; }
tmux new-session -d -s "$SESSION" -n arm -c "$TRON_ARM"
# --no-wrist is mandatory on hardware: a live ORCA wrist makes T_FH wrong.
tmux send-keys -t "$SESSION:arm" \
    "uv run python tools/run_arm.py --robot $ROBOT --no-wrist ${MODEL_PATH:+--model-path $MODEL_PATH} --clutch keyboard" C-m
tmux new-window -t "$SESSION" -n shell -c "$TRON_ARM"
tmux select-window -t "$SESSION:arm"

echo "  SPACE = force hold both arms.  o = orientation freeze.  q = clean exit."
tmux attach -t "$SESSION"
