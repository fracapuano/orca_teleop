# tron_arm

Teleoperate a LimX TRON 2 dual-arm robot from operator wrist poses produced by
`orca_teleop`.

Everything else — what the code does, config, hardware findings, development —
is in [REFERENCE.md](REFERENCE.md).

## Setup (once per laptop)

`tron_arm` lives on the `feat/tron-arm` branch of `orca_teleop`, in this folder.

```bash
git clone -b feat/tron-arm git@github.com:orcahand/orca_teleop.git
cd orca_teleop
uv sync --extra test --extra metaquest      # the orca_teleop side

cd tron_arm
uv venv --python 3.10
uv pip install -e '.[dev,teleop]'
uv pip install --no-deps -e ..              # orca_teleop, from the parent folder
```

Run everything below from the `tron_arm` folder.

## Teleoperate

**1.** Join Wi-Fi `Tron2A_191_5G`. Turn any VPN off.

**2.** Put the arms in position:

```bash
uv run python tools/tron2_cli.py --robot 10.192.1.2 movej-ready
```

**3.** Start the arm:

```bash
uv run python tools/run_arm.py --robot 10.192.1.2 --no-wrist --arm-only
```

To run the hands too, drop `--arm-only` and add
`--model-path <path to the ORCA hand config.yaml>` — that tells the hand
retargeter which hand it is driving.

**4.** Hold SPACE. The arm follows only while it is held.

| key | does |
|---|---|
| **SPACE held** | engaged |
| **release** | both arms hold |
| **`x` / ESC** | force-hold |
| **`o`** | orientation freeze (position-only) |
| **`q`** | clean exit |

`--no-wrist` is mandatory on hardware: it holds the ORCA wrist at 0 so the
flange→palm offset stays rigid.

## Where to put your hand

The instant you press SPACE, wherever your hand is becomes the robot's starting
point, and whichever way it points becomes the robot's forward.

Stand in the same place each session, hand out in front, centred in your
comfortable reach. Point your knuckles the way the flange points.

If the arm runs out of room: release SPACE, bring your hand back to the middle,
press again — like lifting a mouse and putting it down. Repeated engage-release
in one direction slowly walks the arm that way; `movej-ready` re-centres it.

---

# EMERGENCY

**The vendor e-stop does not stop a moving arm.** `request_emgy_stop` is accepted
only while the robot is idle.

To stop a moving arm, in order:

1. **Hardware remote** — the real abort. A named person holds it all session and
   watches the arm, not the screen.
2. **`x` or ESC** — force-hold both arms.
3. **Release SPACE** — both arms hold.
4. **`q`** — clean exit: freezes 1 s so the arm settles, then drops the socket.

Unsure and the arm is moving: **remote, then `x`.** Don't deliberate.

**Connection dies mid-motion:** both arms FAULT and stop being commanded. No
automatic reconnect. Fix the link, restart, re-engage.

**Shutdown order:** release the clutch → `q` → hands to neutral and powered down
→ stop processes → power down the arm. Nobody enters the workspace until the arms
are visibly stationary.

**After any crash:** recalibrate the affected hand before the next run (see
`orca_core`).
