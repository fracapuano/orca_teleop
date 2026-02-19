Guide the user through starting a Manus Quantum Metagloves session. Work through each step interactively, confirming results before proceeding.

Read `docs/manus_glove_setup.md` for the full reference, then walk through these steps:

## Steps

1. **Verify dongles** — SSH to orca-pc and run `lsusb | grep -E '1915|3325'`. Confirm both the Wireless Transceiver (1915:83fd) and Sensor Dongle (3325:0049) are visible. If not, ask the user to check USB connections.

2. **Start the SDK** — Tell the user to run this in a terminal on the Linux machine (not through SSH pipe, it needs an interactive terminal):
   ```
   cd ~/dev/orca/Bidex_Manus_Teleop/MANUS_Core_2.4.0_SDK/SDKClient_Linux
   TERM=xterm-256color ./SDKClient_Linux.out
   ```
   Then select `1` for Core Integrated.

3. **Power on gloves** — Tell the user to flip the switch to the **WiFi symbol** position on each glove. Wait for the blue LED.

4. **Verify connection** — Ask the user to press `G` in the SDK to check the Gloves & Dongles menu. They should see:
   - Dongle with a non-zero ID and `License: Feature`
   - Left and Right Glove with non-zero IDs
   - Angle values that change when they move their fingers

5. **Troubleshoot if needed:**
   - Gloves show 0x0: Try restarting the SDK, make sure gloves are on WiFi mode
   - "No glove available": Turn gloves off, wait 5 seconds, turn back on
   - Terminal error: Make sure they're using `TERM=xterm-256color`
   - Permission denied: Check udev rules or try with `sudo`

6. **Confirm data streaming** — Once gloves show non-zero IDs and angle values are changing, the hardware is ready. The SDK is now streaming data via ZMQ on `tcp://127.0.0.1:8000`.
