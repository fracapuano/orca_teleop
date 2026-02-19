Guide the user through the one-time initial setup for the Manus Quantum Metagloves on the Linux machine (orca-pc). Work through each step interactively, checking results before proceeding.

Read `docs/manus_glove_setup.md` for the full reference, then walk through these steps:

## Steps

1. **SSH connectivity** — Verify `ssh orca-pc` works
2. **Check USB dongles** — Run `lsusb | grep -E '1915|3325'` to see if both dongles are connected (Wireless Transceiver 1915:83fd and Sensor Dongle 3325:0049)
3. **System packages** — Install build-essential, libzmq3-dev, libncurses-dev, libusb-1.0-0-dev, libudev-dev
4. **udev rules** — Set up /etc/udev/rules.d/99-manus.rules for both dongle vendor IDs, reload rules
5. **cppzmq** — Clone and install headers from /tmp/cppzmq
6. **Compile SDK** — Build SDKClient_Linux.out from ~/dev/orca/Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux/
7. **Python environment** — Create venv, install orca_teleop, orca_core, pyzmq, verify imports

After each step, verify it worked before moving on. If something fails, troubleshoot with the user.
