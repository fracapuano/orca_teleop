"""ORCA teleoperation stack.

Every public name below is still importable straight from ``orca_teleop``;
they are resolved lazily (PEP 562) so that importing a leaf module does not
drag in the whole retargeting stack. That matters for the arm path: an arm
controller wants ``orca_teleop.ingress.frames``, which is numpy and stdlib
only, and should not pay ~0.7 s of torch and orca_core import to get it.
"""

from typing import TYPE_CHECKING

_LAZY_ATTRS = {
    "pipeline": "orca_teleop.pipeline",
    "CameraManager": "orca_teleop.cameras",
    "OpenCVCamera": "orca_teleop.cameras",
    "OpenCVCameraConfig": "orca_teleop.cameras",
    "list_available_cameras": "orca_teleop.cameras",
    "parse_camera_spec": "orca_teleop.cameras",
    "HandLandmarks": "orca_teleop.ingress.server",
    "IngressServer": "orca_teleop.ingress.server",
    "LatestFrame": "orca_teleop.ingress.frames",
    "Pose": "orca_teleop.ingress.frames",
    "TeleopFrame": "orca_teleop.ingress.frames",
    "ArmSink": "orca_teleop.arm",
    "arm_worker": "orca_teleop.arm",
    "OrcaHandSink": "orca_teleop.pipeline",
    "RecordableSink": "orca_teleop.pipeline",
    "SinkObservation": "orca_teleop.pipeline",
    "TeleopQueues": "orca_teleop.pipeline",
    "retargeter_worker": "orca_teleop.pipeline",
    "robot_worker": "orca_teleop.pipeline",
    "run": "orca_teleop.pipeline",
    "run_local": "orca_teleop.pipeline",
    "run_manus_local": "orca_teleop.pipeline",
    "Retargeter": "orca_teleop.retargeting.retargeter",
}

if TYPE_CHECKING:  # keep static analysis and IDE completion working
    from orca_teleop import pipeline
    from orca_teleop.arm import ArmSink, arm_worker
    from orca_teleop.cameras import (
        CameraManager,
        OpenCVCamera,
        OpenCVCameraConfig,
        list_available_cameras,
        parse_camera_spec,
    )
    from orca_teleop.ingress.frames import LatestFrame, Pose, TeleopFrame
    from orca_teleop.ingress.server import HandLandmarks, IngressServer
    from orca_teleop.pipeline import (
        OrcaHandSink,
        RecordableSink,
        SinkObservation,
        TeleopQueues,
        retargeter_worker,
        robot_worker,
        run,
        run_local,
        run_manus_local,
    )
    from orca_teleop.retargeting.retargeter import Retargeter


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = module if name == "pipeline" else getattr(module, name)
    globals()[name] = value  # resolve once, then it is a plain global
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "ArmSink",
    "CameraManager",
    "HandLandmarks",
    "IngressServer",
    "LatestFrame",
    "OpenCVCamera",
    "OpenCVCameraConfig",
    "OrcaHandSink",
    "Pose",
    "RecordableSink",
    "Retargeter",
    "SinkObservation",
    "TeleopFrame",
    "TeleopQueues",
    "arm_worker",
    "list_available_cameras",
    "parse_camera_spec",
    "pipeline",
    "retargeter_worker",
    "robot_worker",
    "run",
    "run_local",
    "run_manus_local",
]
