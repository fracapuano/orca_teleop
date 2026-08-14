# Launching

| File | What | Physical motion? |
|---|---|---|
| `preflight.sh` | Read-only checks before anything is powered | no |
| `tmux-sim.sh` | Mock robot + pipeline + mock publisher | no |
| `tmux-robot.sh` | Powered run, with an acknowledgement gate | **yes** |
| `orca-mock-robot.service` | Mock robot as a service | no |

Plus, from `tools/`:

| Command | What |
|---|---|
| `run_arm.py --step-test` | Runbook step 4: format discovery, PASS/FAIL per format |
| `run_arm.py --arm-only` | Runbook step 5: ingress + arm worker, no hand, no torch |
| `report_session.py logs/<id>` | Plain-text session report for an issue |

## Order

```bash
./preflight.sh                    # 1. against the mock
./tmux-sim.sh                     # 2. rehearse the drills here
./preflight.sh --robot 10.192.1.2 # 3. against the real robot
./tmux-robot.sh 10.192.1.2        # 5. powered
```

Between 3 and 4, settle the ServoP encoding (runbook step 4):

```bash
uv run python tools/run_arm.py --robot 10.192.1.2 --no-wrist --step-test
```

It prints PASS/FAIL for each candidate format and names the one to put in
`configs/default.yaml`. Exit 0 means a format was accepted; exit 1 means none
was, and teleop must not proceed.

Afterwards, every run leaves `logs/<session>/`; summarise it with

```bash
uv run python tools/report_session.py logs/<session>
```

## Why there is no boot-time hardware unit

`run_arm.py` demands an interactive confirmation (`CLAUDE.md` hard rule 1)
and systemd has no tty, so any such unit would only ever refuse to start.
The reason is recorded in `orca-mock-robot.service`'s header.

`tmux-sim.sh` passes `--clutch none` for the same tty reason: with no
terminal a hold-to-engage clutch reads as permanently released. Fine
against the mock, never against hardware.

## tmux windows

`tmux-sim.sh` opens: `robot` (mock), `arm` (pipeline + status line + hotkeys),
`publisher` (synthetic operator), `shell` (spare, for `tron2_cli`).

Hotkeys live in the `arm` window: **x/ESC** force-hold both arms, **o**
orientation freeze, **q** clean exit. The clutch is hold-to-engage, so nothing
moves unless a key is held there.
