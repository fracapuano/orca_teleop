"""Tests for the repository-owned Quest Browser stream."""

import asyncio
import queue
import socket
import threading

import numpy as np
import pytest

from orca_teleop.ingress.metaquest.bridge import (
    QuestTelemetryBridge,
    QuestTelemetryState,
    WebXRHandSample,
)
from orca_teleop.ingress.metaquest.landmarks import (
    WEBXR_TO_RETARGETER_LANDMARK_INDICES,
    QuaternionContinuity,
)
from orca_teleop.ingress.metaquest.mock_publisher import MockQuestBridge, synthetic_payload
from orca_teleop.ingress.metaquest.publisher import MetaQuestPublisher
from orca_teleop.ingress.server import HandLandmarks, IngressServer


def _webxr_payload(side: str = "right", **extra) -> dict:
    payload = {
        "type": "telemetry",
        "client_wall_ms": 1234.5,
        "hands": {
            side: {
                "wrist": np.eye(4).T.ravel().tolist(),
                "landmarks": np.arange(75, dtype=float).reshape(25, 3).tolist(),
            }
        },
    }
    payload.update(extra)
    return payload


def _xr_matrix(position=(0.0, 0.0, 0.0)) -> list[float]:
    """A WebXR (column-major, XR basis) transform with the given translation."""
    matrix = np.eye(4)
    matrix[:3, 3] = position
    return matrix.T.ravel().tolist()


def test_telemetry_state_keeps_latest_valid_hand_sample():
    state = QuestTelemetryState()

    state.update(_webxr_payload())
    first = state.get_hand_sample("right")
    state.update(_webxr_payload())
    second = state.get_hand_sample("right")

    assert first is not None
    assert second is not None
    assert first.sequence_id == 1
    assert second.sequence_id == 2
    assert first.timestamp_ns == 1_234_500_000
    assert first.landmarks.shape == (25, 3)
    np.testing.assert_allclose(first.wrist_matrix, np.eye(4))


def test_telemetry_state_drops_incomplete_webxr_hand():
    state = QuestTelemetryState()
    state.update(_webxr_payload())
    state.update(
        {
            "type": "telemetry",
            "hands": {"right": {"landmarks": np.zeros((24, 3)).tolist()}},
        }
    )

    assert state.get_hand_sample("right") is None


def test_webxr_publisher_emits_reduced_points_and_wrist_angle():
    publisher = MetaQuestPublisher(wrist_enabled=True)
    landmarks = np.arange(75, dtype=float).reshape(25, 3)
    sample = WebXRHandSample(
        sequence_id=1,
        timestamp_ns=42,
        landmarks=landmarks,
        wrist_matrix=np.eye(4),
    )

    first = publisher._sample_to_proto(sample)
    pitched_wrist = np.eye(4)
    pitch = np.radians(-10.0)
    pitched_wrist[:3, 0] = [np.cos(pitch), 0.0, np.sin(pitch)]
    second = publisher._sample_to_proto(
        WebXRHandSample(
            sequence_id=2,
            timestamp_ns=43,
            landmarks=landmarks,
            wrist_matrix=pitched_wrist,
        )
    )

    expected = landmarks[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)].astype(np.float32)
    np.testing.assert_allclose(np.asarray(first.keypoints).reshape(21, 3), expected)
    assert first.timestamp_ns == 42
    assert first.handedness == "right"
    assert first.wrist_angle_degrees == pytest.approx(0.0)
    assert second.wrist_angle_degrees == pytest.approx(10.0)


def test_bridge_carries_head_matrix_in_flu_and_pose_epoch():
    """The headset pose was received and dropped before; it is the arm's origin
    reference for a head-relative mode."""
    state = QuestTelemetryState()
    # WebXR is X right, Y up, -Z forward; FLU is X forward, Y left, Z up.
    state.update(_webxr_payload(head=_xr_matrix((1.0, 2.0, 3.0)), session_epoch=99))

    sample = state.get_hand_sample("right")
    assert sample is not None
    assert sample.pose_epoch == 99
    np.testing.assert_allclose(sample.head_matrix[:3, 3], [-3.0, -1.0, 2.0])


def test_bridge_falls_back_to_the_connection_epoch():
    """A cached page that sends no session_epoch still gets a usable one."""
    state = QuestTelemetryState()
    state.update(_webxr_payload(), fallback_epoch=4)
    assert state.get_hand_sample("right").pose_epoch == 4


def test_bridge_tolerates_a_malformed_head_matrix():
    state = QuestTelemetryState()
    state.update(_webxr_payload(head=[1.0, 2.0]))
    sample = state.get_hand_sample("right")
    assert sample is not None and sample.head_matrix is None


def test_get_hand_sample_copies_the_head_matrix():
    """Callers must not be able to mutate the store's own array."""
    state = QuestTelemetryState()
    state.update(_webxr_payload(head=_xr_matrix((1.0, 0.0, 0.0))))
    first = state.get_hand_sample("right")
    first.head_matrix[0, 3] = 999.0
    assert state.get_hand_sample("right").head_matrix[0, 3] != 999.0


def test_sample_to_proto_sets_wrist_and_head_pose():
    publisher = MetaQuestPublisher()
    sample = WebXRHandSample(
        sequence_id=1,
        timestamp_ns=42,
        landmarks=np.arange(75, dtype=float).reshape(25, 3),
        wrist_matrix=np.eye(4),
        head_matrix=np.eye(4),
        pose_epoch=5,
    )

    frame = publisher._sample_to_proto(sample)

    assert frame.HasField("wrist_pose") and frame.HasField("head_pose")
    assert frame.HasField("tracking_valid") and frame.tracking_valid is True
    assert frame.pose_epoch == 5
    assert (frame.wrist_pose.qw, frame.wrist_pose.px) == (1.0, 0.0)


def test_no_wrist_matrix_leaves_wrist_pose_unset():
    publisher = MetaQuestPublisher()
    frame = publisher._sample_to_proto(
        WebXRHandSample(
            sequence_id=1,
            timestamp_ns=42,
            landmarks=np.arange(75, dtype=float).reshape(25, 3),
            wrist_matrix=None,
        )
    )
    assert not frame.HasField("wrist_pose")
    assert not frame.HasField("head_pose")


def test_arm_pose_can_be_disabled_without_touching_wrist_angle():
    """--no-arm-pose is independent of --wrist: the hand keeps its wrist joint."""
    pitched = np.eye(4)
    pitch = np.radians(-10.0)
    pitched[:3, 0] = [np.cos(pitch), 0.0, np.sin(pitch)]
    sample = WebXRHandSample(
        sequence_id=1,
        timestamp_ns=42,
        landmarks=np.arange(75, dtype=float).reshape(25, 3),
        wrist_matrix=pitched,
    )

    with_arm = MetaQuestPublisher(arm_pose_enabled=True)
    without_arm = MetaQuestPublisher(arm_pose_enabled=False)
    with_arm._sample_to_proto(sample)  # prime the relative wrist zero
    without_arm._sample_to_proto(sample)
    frame_with = with_arm._sample_to_proto(sample)
    frame_without = without_arm._sample_to_proto(sample)

    assert not frame_without.HasField("wrist_pose")
    assert frame_with.wrist_angle_degrees == pytest.approx(frame_without.wrist_angle_degrees)
    assert list(frame_with.keypoints) == list(frame_without.keypoints)


def test_quaternion_continuity_keeps_the_sign_and_resets_on_epoch():
    estimator = QuaternionContinuity()
    first = estimator.update(np.array([0.7071, 0.7071, 0.0, 0.0]), pose_epoch=1)
    # The same rotation, extracted with the opposite sign.
    second = estimator.update(np.array([-0.7071, -0.7071, 0.0, 0.0]), pose_epoch=1)

    assert first[0] > 0
    np.testing.assert_allclose(second, first, atol=1e-9)
    assert float(np.dot(first, second)) > 0

    reseeded = estimator.update(np.array([-0.7071, -0.7071, 0.0, 0.0]), pose_epoch=2)
    assert reseeded[0] > 0  # new reference space, re-canonicalized


def test_quaternion_continuity_survives_a_sweep_past_180_degrees():
    """The case a naive `w >= 0` canonicalization breaks.

    Rolling the wrist steadily about one axis is an operator-routine motion.
    Feeding the canonical (w >= 0) extraction each step, the tracker must keep
    following the continuous path — emitting a negative w past 180 deg —
    rather than flipping and reporting a 360 deg jump that never happened.
    """
    estimator = QuaternionContinuity()
    outputs = []
    for degrees in range(0, 360, 10):
        half = np.radians(degrees) / 2.0
        quaternion = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
        canonical = quaternion if quaternion[0] >= 0 else -quaternion
        outputs.append(estimator.update(canonical, pose_epoch=1))

    steps = [float(np.dot(a, b)) for a, b in zip(outputs[:-1], outputs[1:], strict=True)]
    assert min(steps) > 0.9, "continuity broken: a step flipped hemisphere"
    assert any(q[0] < 0 for q in outputs), "never left the w >= 0 hemisphere"


def test_mock_payload_flows_through_the_real_bridge_and_publisher():
    """The headset-free mock must exercise the production path, not a copy."""
    state = QuestTelemetryState()
    state.update(synthetic_payload(elapsed_s=1.0, side="right", pose_epoch=3))

    sample = state.get_hand_sample("right")
    assert sample is not None
    assert sample.landmarks.shape == (25, 3)
    assert sample.pose_epoch == 3

    frame = MetaQuestPublisher()._sample_to_proto(sample)
    assert len(frame.keypoints) == 63
    assert frame.HasField("wrist_pose") and frame.HasField("head_pose")
    assert frame.pose_epoch == 3
    quaternion = np.array(
        [frame.wrist_pose.qw, frame.wrist_pose.qx, frame.wrist_pose.qy, frame.wrist_pose.qz]
    )
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-9)
    # FLU: the operator's wrist is out in front (+x) and above the floor (+z).
    assert frame.wrist_pose.px > 0.0
    assert frame.wrist_pose.pz > 0.5


def test_mock_hand_is_never_degenerate_for_the_retargeter():
    """A collapsed wrist-to-middle axis would make the retargeter reject frames."""
    for elapsed_s in np.linspace(0.0, 12.0, 60):
        landmarks = np.asarray(
            synthetic_payload(elapsed_s, "right", 1)["hands"]["right"]["landmarks"]
        )
        reduced = landmarks[list(WEBXR_TO_RETARGETER_LANDMARK_INDICES)]
        centered = reduced - reduced[0]
        assert np.linalg.norm(centered[9]) > 1e-3  # wrist -> middle MCP
        assert np.isfinite(centered).all()


def test_mock_dropout_removes_the_hand_entirely():
    """Tracking loss is silence: the Quest cannot send tracking_valid=false."""
    state = QuestTelemetryState()
    state.update(synthetic_payload(0.0, "right", 1))
    assert state.get_hand_sample("right") is not None

    state.update({"type": "telemetry", "hands": {}})
    assert state.get_hand_sample("right") is None


def test_mock_bridge_satisfies_the_publisher_interface():
    """MockQuestBridge is duck-typed; a rename in the real bridge must show up.

    Checked against QuestTelemetryBridge itself rather than a hardcoded list of
    names -- a list only ever describes the mock, so it could not catch the
    upstream rename this is here to catch.
    """
    bridge = MockQuestBridge(side="right", fps=30)
    # Constructing the real bridge is inert -- no socket, no thread until start().
    real = QuestTelemetryBridge()
    for attribute in ("state", "start", "stop", "url", "ssl_context"):
        assert hasattr(real, attribute), (
            f"the real bridge no longer has {attribute!r}; the publisher contract moved"
        )
        assert hasattr(bridge, attribute), f"the mock never grew {attribute!r}"
    publisher = MetaQuestPublisher(bridge=bridge)
    assert publisher.bridge is bridge


def _unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_websocket_frame_reaches_grpc_ingress():
    aiohttp = pytest.importorskip("aiohttp")
    landmarks_q: queue.Queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    ingress = IngressServer(landmarks_q, stop_event, port=0)
    grpc_port = ingress.start()
    web_port = _unused_local_port()
    publisher = MetaQuestPublisher(
        server_address=f"localhost:{grpc_port}",
        quest_host="127.0.0.1",
        quest_port=web_port,
        fps=60,
        wrist_enabled=False,
    )
    publisher_thread = threading.Thread(target=publisher.run, daemon=True)
    publisher_thread.start()

    async def send_when_ready() -> None:
        async with aiohttp.ClientSession() as session:
            for _ in range(100):
                try:
                    async with session.ws_connect(f"http://127.0.0.1:{web_port}/ws") as ws:
                        await ws.send_json(_webxr_payload())
                        await asyncio.sleep(0.1)
                        return
                except aiohttp.ClientError:
                    await asyncio.sleep(0.02)
        raise AssertionError("Quest WebXR bridge did not become ready")

    try:
        asyncio.run(send_when_ready())
        item = landmarks_q.get(timeout=3.0)

        assert isinstance(item, HandLandmarks)
        assert item.keypoints.shape == (21, 3)
        assert item.handedness == "right"
    finally:
        publisher.stop()
        stop_event.set()
        publisher_thread.join(timeout=3.0)
        ingress.stop()

    assert not publisher_thread.is_alive()
