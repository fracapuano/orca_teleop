"""TRON 2 dual-arm control layer for orca_teleop wrist poses.

Public surface:
  * :mod:`tron_arm.poses`        -- Pose type + sign-preserving interpolation
  * :mod:`tron_arm.config`       -- validated YAML configuration
  * :mod:`tron_arm.tron2_client` -- persistent WebSocket client for the robot
  * :mod:`tron_arm.streamer`     -- fixed-rate interpolating ServoP pacer
  * :mod:`tron_arm.mock_robot`   -- local stand-in for the real robot

This package deliberately does not import orca_teleop: the mock and CLI must run
standalone. Interop happens structurally, via the accessor names in
:mod:`tron_arm.poses` (``.matrix`` / ``.as_xyz_wxyz()``).
"""

from __future__ import annotations

__version__ = "0.1.0"
