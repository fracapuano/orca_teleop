# Manus Quantum Glove Integration Research

Reference document for integrating Manus Quantum Metagloves into the ORCA teleop pipeline.

## 1. Hardware Overview

- **Manus Quantum Metagloves**: 120Hz, <=5ms latency, proprietary 2.4GHz wireless, 15m range
- **Manus Metagloves Pro**: 120Hz, 30ms wired / 50ms wireless, Bluetooth 5.0
- **Connectivity**: USB dongle for wireless, USB-C for wired (wired bypasses pairing)
- **USB Security Dongle**: Contains Feature license required for SDK access
- **Tip**: Keep wireless dongle and security dongle on separate USB buses

## 2. SDK Architecture

### Two Modes

| | Integrated Mode | Remote Mode |
|---|---|---|
| **Init** | `CoreSdk_InitializeIntegrated()` | Connects to MANUS Core |
| **OS** | Linux + Windows | Windows (Core required) |
| **Use case** | Standalone / embedded | Desktop with Core GUI |
| **Library** | `libManusSDK_Integrated.so` | Standard SDK lib |

**No official Python SDK** — C++ only. Community bridges use ZMQ or ROS2.

### Linux Support (Integrated Mode)

- **Supported**: Ubuntu 20.04, 22.04, 24.04
- **Build deps**: `build-essential libusb-1.0-0-dev libudev-dev`
- **Runtime**: Needs udev rules for USB device access
- **Limitations**: Firmware updates require MANUS Core (Windows only)

### Linux-Only Workflow

Fully Linux-only operation is possible:
1. Gloves come **factory pre-paired** with their dongle
2. Wired USB-C mode works without any pairing
3. Pairing available programmatically: `CoreSdk_PairGlove()`, `CoreSdk_UnpairGlove()`, `CoreSdk_PairGloveToDongle()`
4. Calibration available programmatically via SDK
5. **Only firmware updates** require Windows/MANUS Core

**Confirmed**: License comes pre-loaded on the dongle. No Windows needed for initial setup.

## 3. SDK Data Formats

### Raw Skeleton (25 nodes per hand)

| Nodes | Chain |
|-------|-------|
| 0 | Wrist |
| 1-4 | Thumb (CMC, MCP, IP, Tip) |
| 5-9 | Index (MCP, PIP, DIP, Tip) |
| 10-14 | Middle (MCP, PIP, DIP, Tip) |
| 15-19 | Ring (MCP, PIP, DIP, Tip) |
| 20-24 | Pinky (MCP, PIP, DIP, Tip) |

Each node: position (x, y, z) + quaternion rotation.

### Ergonomics Data (per-joint angles in degrees)

- **Thumb**: CMC spread, CMC flex, MCP flex, IP flex
- **Index/Middle/Ring/Pinky**: MCP spread, MCP flex, PIP flex, DIP flex
- ~20-24 DOF total

### Three Stream Types

1. **Raw Skeleton** — 25 nodes, position + quaternion
2. **Ergonomics** — per-joint flex/spread angles in degrees
3. **Retargeted Skeleton** — data applied to a custom skeleton definition

## 4. SDK Code Patterns (C++)

### Initialization & Coordinate System

```cpp
// Coordinate system setup
CoordinateSystemVUH t_VUH;
CoordinateSystemVUH_Init(&t_VUH);
t_VUH.handedness = Side::Side_Right;
t_VUH.up = AxisPolarity::AxisPolarity_PositiveZ;
t_VUH.view = AxisView::AxisView_XFromViewer;
t_VUH.unitScale = 1.0f; // meters
CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true);

// Integrated mode init (Linux standalone)
CoreSdk_InitializeIntegrated();
```

### Callback Registration

```cpp
CoreSdk_RegisterCallbackForRawSkeletonStream(OnRawSkeletonStreamCallback);
CoreSdk_RegisterCallbackForSkeletonStream(OnSkeletonStreamCallback);
CoreSdk_RegisterCallbackForErgonomics(OnErgonomicsCallback);
```

### Pairing Functions

```cpp
CoreSdk_PairGlove(glove_id, &success);
CoreSdk_UnpairGlove(glove_id, &success);
CoreSdk_PairGloveToDongle(glove_id, dongle_id, &success);
```

## 5. Relevant Repositories

### Bidex_Manus_Teleop (PRIMARY)
- **URL**: https://github.com/leap-hand/Bidex_Manus_Teleop
- **Contains**: Modified MANUS SDK 2.4.0 (`MANUS_Core_2.4.0_SDK/`) with ZMQ bridge
- **Architecture**: C++ SDK -> ZMQ pub/sub -> Python consumer
- **Linux support**: Ubuntu standalone mode with "Core Integrated" selection in menu
- **Key file**: `minimal_example.py` — Python ZMQ consumer
- **Plan**: Clone this, compile `SDKClient_Linux`, use as the C++ bridge

### ASIG-X/manusGlove
- **URL**: https://github.com/ASIG-X/manusGlove
- **Contains**: Linux ROS2 driver, publishes skeleton data, Python visualization client
- **Useful for**: Reference implementation of Linux integrated mode

### iotdesignshop/manus_ros2
- **URL**: https://github.com/iotdesignshop/manus_ros2
- **Contains**: PoseArray at 50Hz, based on SDKMinimalClient_Linux
- **Useful for**: Simpler reference, ROS2 integration pattern

### iotdesignshop/dexhandv2_manus
- **URL**: https://github.com/iotdesignshop/dexhandv2_manus
- **Contains**: DexHand V2 + Manus glove integration (similar use case to ours)

## 6. ORCA Retargeter Interface Contract

### How the pipeline works

```
Ingress (landmarks) -> retargeter.retarget(data_dict) -> joint angles -> robot
```

### `retarget()` method

```python
# retargeter.py lines ~101-106
def retarget(self, data, manual_wrist_angle=None):
    if self.source == "avp":
        joints, computed_wrist_angle = retargeter_utils.preprocess_avp_data(data, self.hand_type)
    elif self.source == "mediapipe":
        joints, computed_wrist_angle = retargeter_utils.preprocess_mediapipe_data(data)
    # Need to add: elif self.source == "manus": ...
```

### Preprocessing contract

Each `preprocess_{source}_data()` must return:
- `joints`: `np.ndarray` shape `(N, 3)` — 3D positions of hand landmarks
- `wrist_angle`: `float` in degrees

### What the optimizer actually uses

Only **7 key points** matter for the IK optimization:
- 5 fingertip positions (MediaPipe indices: 4, 8, 12, 16, 20)
- 2 palm base points: thumb base (1) and pinky base (17)

These are extracted via `get_mano_joints_dict()` which needs a source-specific branch mapping indices to the finger dict structure.

### MediaPipe reference (21 points)

- joints[0] = wrist
- joints[1-4] = thumb (CMC, MCP, IP, tip)
- joints[5-8] = index (MCP, PIP, DIP, tip)
- joints[9-12] = middle
- joints[13-16] = ring
- joints[17-20] = pinky

### What a Manus preprocessor needs to do

1. Accept Manus skeleton data (25 nodes, position + quaternion)
2. Extract the 3D positions into `np.ndarray` shape `(25, 3)` or map to `(21, 3)` matching MediaPipe layout
3. Compute wrist angle from quaternion or Euler decomposition
4. Return `(joints, wrist_angle)`
5. Add index mapping in `get_mano_joints_dict()` for the Manus source

### Existing patterns to follow

- `MediaPipeIngress`: threaded capture, lock-protected shared state, callback-based output
- `preprocess_mediapipe_data()`: scales landmarks by 1.2x, adds palm offset `[0, 0, 0.015]`
- Demo script wraps data in dict: `retargeter.retarget({"hand_landmarks": landmarks})`

## 7. Integration Architecture (Planned)

```
Manus Gloves (wireless/USB-C)
    |
USB Dongles (License: 1915:83fd, Sensor: 3325:0049)
    |
SDKClient_Linux (C++ integrated mode)
    |  ZMQ pub/sub
    v
ManusIngress (Python) -- reads ZMQ, extracts positions
    |  callback
    v
Retargeter.retarget({"manus_skeleton": data})
    |  preprocess_manus_data()
    v
Joint angles -> OrcaHand -> Robot
```

### Files to create/modify

| File | Change |
|---|---|
| `orca_teleop/orca_ingress/manus/manus_ingress.py` | New: ZMQ subscriber, threaded, callback-based |
| `orca_teleop/orca_retargeter/utils/retargeter_utils.py` | Add `preprocess_manus_data()`, add Manus branch in `get_mano_joints_dict()` |
| `orca_teleop/orca_retargeter/retargeter.py` | Add `elif self.source == "manus"` routing |
| `scripts/manus_teleop_demo.py` | New: end-to-end demo with ManusIngress |
| `orca_teleop/__init__.py` | Export `ManusIngress` |

## 8. USB Dongle Info (Confirmed)

There are **two separate USB dongles** (not one combined):

| Device | Vendor:Product | lsusb Name | Role |
|---|---|---|---|
| License Dongle | `1915:83fd` | Nordic Semiconductor ASA Wireless Transceiver | Contains Feature license |
| Sensor Dongle | `3325:0049` | Manus VR Sensor Dongle | Wireless 2.4GHz communication with gloves |

- Both must be plugged in for wireless operation
- Keep on separate USB ports
- License dongle shows as HID device on `/dev/hidraw*`
- SDK requires `sudo` for USB device access (even with udev rules)

## 9. MANUS Resource Center

- **URL**: https://manus-meta.com/resources/downloads
- **Needed if**: Bidex repo's included SDK 2.4.0 is insufficient
- **Account**: User needs to create one (free registration)
- **Downloads**: Official SDK, MANUS Core (Windows), firmware tools
