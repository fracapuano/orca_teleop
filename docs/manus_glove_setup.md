# Manus Quantum Metagloves Setup Guide

## Hardware Overview

You need these USB devices connected to the Linux PC:

| Device | lsusb ID | Name | Purpose |
|---|---|---|---|
| Wireless Transceiver | `1915:83fd` | Nordic Semiconductor ASA | License dongle (contains Feature license) |
| Sensor Dongle | `3325:0049` | Manus VR Sensor Dongle | Wireless communication with gloves |

Keep them on **separate USB ports** (ideally separate USB buses).

### Glove Power Switch

The gloves have a 3-position switch:
- **Off** — powered down
- **On** — powered on, wired USB-C mode only (no radio)
- **WiFi symbol** — powered on with 2.4GHz wireless radio active (for dongle communication)

### LED Status

- **Blue solid** — connected and transmitting
- **Blue blinking** — pairing mode
- Check [Manus LED docs](https://docs.manus-meta.com/3.1.0/Products/Quantum%20Mocap%20Metagloves/) for full reference

---

## Initial Setup (One-Time)

### 1. System packages

```bash
sudo apt-get install -y build-essential libzmq3-dev libncurses-dev libusb-1.0-0-dev libudev-dev
```

### 2. udev rules

Create rules so the SDK can access both dongles without root:

```bash
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-manus.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="1915", ATTR{idProduct}=="83fd", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="3325", ATTR{idProduct}=="0049", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Note:** If the dongles were already plugged in, unplug and replug them after setting rules.

### 3. Install cppzmq

```bash
cd /tmp
git clone https://github.com/zeromq/cppzmq.git
cd cppzmq && mkdir build && cd build
cmake ..
sudo cmake --install .
```

(Test compilation may fail — that's fine, the headers install correctly.)

### 4. Compile the SDK

```bash
cd ~/dev/orca/Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux
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
lsusb | grep -E '1915|3325'
```

You should see both:
```
Bus 001 Device XXX: ID 1915:83fd Nordic Semiconductor ASA Wireless Transceiver
Bus 001 Device XXX: ID 3325:0049 Manus VR (https://www.manus-vr.com) Sensor Dongle
```

### 2. Start the SDK client

```bash
cd ~/dev/orca/Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux
TERM=xterm-256color ./SDKClient_Linux.out
```

Select **`1` — Core Integrated** from the menu.

### 3. Power on the gloves

Flip the switch to the **WiFi symbol** position on each glove. Wait for the blue LED to turn on.

### 4. Verify connection

Press `G` to go to the Gloves & Dongles menu. You should see:
- Dongle with ID (e.g., `0x71bbd9a8`) and `License: Feature`
- Left Glove and Right Glove with non-zero IDs
- Finger angle values changing as you move your fingers

### 5. If gloves don't appear

- Press `Q` to go back, then `P` for the Pairing menu
- Press `P` to pair first available glove
- Make sure gloves are in pairing mode (LED blinking)
- If "no glove available" — try turning gloves off and back on to WiFi symbol
- Try running with `sudo` if permission issues suspected

### 6. Troubleshooting

| Problem | Solution |
|---|---|
| `Error opening terminal: xterm-ghostty` | Use `TERM=xterm-256color` prefix |
| Dongle shows `0x0` for gloves | Restart SDK after plugging in dongles |
| "No glove available for pairing" | Turn gloves off, wait 5s, turn back on to WiFi |
| SDK doesn't detect Sensor Dongle | Check udev rules, try `sudo`, try different USB port |
| All angle values are zero | Gloves connected but not being worn / not calibrated |

### 7. Wired mode (debugging)

Connect a glove directly via USB-C, flip switch to **On** (not WiFi). It should appear without needing the wireless dongle — useful for isolating wireless issues.

---

## SDK Menu Reference

| Key | Menu | Purpose |
|---|---|---|
| `G` | Gloves & Dongles | View connected gloves, dongle info, ergonomics data |
| `S` | Skeleton | Raw skeleton data (25 nodes, position + quaternion) |
| `P` | Pairing | Pair/unpair gloves to dongle |
| `C` | Calibration | Calibrate glove sensors |
| `J` | Gestures | Gesture recognition data |
| `D` | Landscape Time | Timestamp and system info |
| `ESC` | Quit | Exit the SDK client |
