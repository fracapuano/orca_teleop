from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class WristPose(_message.Message):
    __slots__ = ("position", "rotation")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    position: _containers.RepeatedScalarFieldContainer[float]
    rotation: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self, position: _Iterable[float] | None = ..., rotation: _Iterable[float] | None = ...
    ) -> None: ...

class HandFrame(_message.Message):
    __slots__ = ("keypoints", "handedness", "timestamp_ns", "wrist_pose")
    KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    HANDEDNESS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    WRIST_POSE_FIELD_NUMBER: _ClassVar[int]
    keypoints: _containers.RepeatedScalarFieldContainer[float]
    handedness: str
    timestamp_ns: int
    wrist_pose: WristPose
    def __init__(
        self,
        keypoints: _Iterable[float] | None = ...,
        handedness: str | None = ...,
        timestamp_ns: int | None = ...,
        wrist_pose: WristPose | _Mapping | None = ...,
    ) -> None: ...

class StreamConfig(_message.Message):
    __slots__ = ("handedness",)
    HANDEDNESS_FIELD_NUMBER: _ClassVar[int]
    handedness: str
    def __init__(self, handedness: str | None = ...) -> None: ...

class StreamSummary(_message.Message):
    __slots__ = ("frames_received",)
    FRAMES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    frames_received: int
    def __init__(self, frames_received: int | None = ...) -> None: ...
