# Manus Quantum Metagloves Setup Guide

## Hardware Overview

The Manus Quantum Metagloves ship with two USB dongles:

| Device | Purpose |
|---|---|
| **License Dongle** (Wireless Transceiver) | Contains the Feature license required by the SDK |
| **Sensor Dongle** | Handles wireless 2.4GHz communication with the gloves |

Both must be plugged into the Linux PC. Keep them on **separate USB ports** (ideally separate USB buses) to avoid interference.

To identify them, run `lsusb` and look for Manus/Nordic Semiconductor entries. For example:
```
Bus 001 Device 012: ID 1915:83fd Nordic Semiconductor ASA Wireless Transceiver
Bus 001 Device 013: ID 3325:0049 Manus VR (https://www.manus-vr.com) Sensor Dongle
```

The vendor/product IDs (`1915:83fd` and `3325:0049`) are the same across all Quantum Metaglove kits — they identify the device model, not your specific unit. Your serial numbers and bus/device numbers will differ.

### Glove Power Switch

The gloves have a 3-position switch:
- **Off** — powered down
- **On** — powered on for wired USB-C mode only (no radio)
- **WiFi symbol** — powered on with 2.4GHz wireless radio active (for dongle communication)

Use **On** when connected via USB-C cable. Use the **WiFi symbol** when using the wireless dongles.

### LED Status

- **Blue solid** — connected and transmitting
- **Blue blinking** — pairing mode / searching for dongle
- Check [Manus LED docs](https://docs.manus-meta.com/3.1.0/Products/Quantum%20Mocap%20Metagloves/) for full reference

---

## Initial Setup (One-Time)

### 1. System packages

```bash
sudo apt-get install -y build-essential libzmq3-dev libncurses-dev libusb-1.0-0-dev libudev-dev
```

### 2. udev rules

Create rules so the SDK can access both dongles without root. The vendor/product IDs below are standard for Manus Quantum Metaglove dongles:

```bash
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-manus.rules
# Manus Quantum Metagloves dongles
SUBSYSTEM=="usb", ATTR{idVendor}=="1915", ATTR{idProduct}=="83fd", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="3325", ATTR{idProduct}=="0049", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

If you have a different Manus product (e.g., Metagloves Pro), your dongle IDs may differ. Run `lsusb` with the dongles plugged in to find the correct vendor/product IDs and update the rules accordingly.

**Note:** If the dongles were already plugged in before creating the rules, unplug and replug them.

### 3. Install cppzmq

```bash
cd /tmp
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq && mkdir build && cd build
cmake ..
sudo cmake --install .
```

(Test compilation may fail — that's fine, the headers install correctly.)

### 4. Clone and compile the SDK

The [Bidex_Manus_Teleop](https://github.com/leap-hand/Bidex_Manus_Teleop) repo contains a modified Manus SDK 2.4.0 with ZMQ bindings that bridge glove data to Python:

```bash
cd ~/dev/orca
git clone https://github.com/leap-hand/Bidex_Manus_Teleop.git
cd Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux
make
```

This produces `SDKClient_Linux.out` (~1.2MB).

### 5. Python environment

```bash
cd ~/dev/orca/orca_teleop
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e ../orca_core
pip install pyzmq
```

Verify:
```bash
python -c "from orca_teleop import MediaPipeIngress, Retargeter; print('OK')"
```

---

## Running the Gloves

### 1. Verify dongles are connected

```bash
lsusb | grep -iE 'manus|1915|3325'
```

You should see two Manus-related devices (the license dongle and sensor dongle). If you only see one, check your USB connections.

### 2. Start the SDK client

The SDK uses ncurses and must be run in an interactive terminal (not through a non-interactive SSH pipe):

```bash
cd ~/dev/orca/Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux
sudo TERM=xterm-256color ./SDKClient_Linux.out
```

`sudo` is required for direct USB device access. The `TERM` override prevents ncurses errors if your terminal type isn't recognized on the Linux machine.

Select **`1` — Core Integrated** from the menu.

### 3. Power on the gloves

Flip the switch to the **WiFi symbol** position on each glove. Wait for the blue LED.

### 4. Verify connection

Press `G` to go to the Gloves & Dongles menu. You should see:
- A dongle entry with a non-zero hex ID and `License: Feature`
- Left Glove and Right Glove entries with non-zero hex IDs (not `0x0`)
- Finger angle values that change as you move your fingers

### 5. If gloves don't appear

- Make sure both dongles were plugged in **before** starting the SDK — restart it if you plugged them in after
- Turn gloves off, wait 5 seconds, turn back on to the WiFi symbol position
- Press `Q` to go back to the main menu, then `P` for the Pairing menu, then `P` to pair
- Gloves are factory pre-paired with their dongle, but if they were re-paired elsewhere, you may need to pair again

### 6. Troubleshooting

| Problem | Solution |
|---|---|
| `Error opening terminal: xterm-ghostty` (or similar) | Use `TERM=xterm-256color` prefix when launching |
| Dongle shows `0x0` for gloves | Gloves not connected — check they're on WiFi mode, restart SDK |
| "No glove available for pairing" | Turn gloves off, wait 5s, turn back on; check dongle USB connection |
| SDK doesn't detect dongles | Verify with `lsusb`, check udev rules, try different USB port |
| All angle values are zero | Gloves detected but not worn or need calibration (`C` menu) |
| Permission errors | Run with `sudo`, or verify udev rules are active |

### 7. Wired mode (debugging)

Connect a glove directly via USB-C cable and flip the switch to **On** (not WiFi). The glove should appear in the SDK without needing the wireless dongle. Useful for isolating wireless vs. hardware issues.

---

## SDK Menu Reference

| Key | Menu | Purpose |
|---|---|---|
| `G` | Gloves & Dongles | View connected gloves, dongle info, live ergonomics data |
| `S` | Skeleton | Raw skeleton data (25 nodes per hand, position + quaternion) |
| `P` | Pairing | Pair/unpair gloves to dongle |
| `C` | Calibration | Calibrate glove sensors |
| `J` | Gestures | Gesture recognition data |
| `D` | Landscape Time | Timestamp and system info |
| `ESC` | Quit | Exit the SDK client |
