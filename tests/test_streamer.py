"""Tests for the orca_ui streamer pieces: the HandModel shim, the OrcaUiSink
wire protocol / coalescing, and the synthetic source. No hardware, no camera,
no orca_ui — the UI side is faked (in-memory link or a local WS server)."""

import json
import queue
import threading
import time

import pytest

from orca_teleop.hand_model import HandModel
from orca_teleop.pipeline import _SHUTDOWN
from orca_teleop.streamer import synthetic_worker
from orca_teleop.ui_sink import OrcaUiSink

CONFIG_YAML = """
type: right
joint_ids:
  - wrist
  - index_mcp
  - thumb_abd
joint_roms:
  wrist: [-30, 30]
  index_mcp: [0, 90]
  thumb_abd: [-10, 50]
neutral_position:
  wrist: 0
  index_mcp: 10
  thumb_abd: 5
"""


@pytest.fixture
def hand_model(tmp_path) -> HandModel:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG_YAML)
    return HandModel.load(str(config))


# ----- HandModel shim ---------------------------------------------------------------


def test_hand_model_parses_config(hand_model, tmp_path):
    assert hand_model.config.type == "right"
    assert hand_model.config.joint_ids == ["wrist", "index_mcp", "thumb_abd"]
    assert hand_model.config.joint_roms_dict["index_mcp"] == [0.0, 90.0]
    assert hand_model.config.neutral_position["thumb_abd"] == 5.0
    assert hand_model.config.model_path == str(tmp_path)
    # duck-type parity with OrcaHand: `.config.<field>` access
    assert hand_model.config.joint_roms_dict.keys() == set(
        hand_model.config.joint_ids)


def test_hand_model_accepts_directory(hand_model, tmp_path):
    assert HandModel.load(str(tmp_path)).config.type == "right"


def test_hand_model_rejects_bad_type(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("type: both\njoint_ids: [wrist]\njoint_roms: {wrist: [0, 1]}")
    with pytest.raises(ValueError, match="type"):
        HandModel.load(str(config))


def test_hand_model_rejects_missing_rom(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("type: left\njoint_ids: [wrist, index_mcp]\n"
                      "joint_roms: {wrist: [0, 1]}")
    with pytest.raises(ValueError, match="index_mcp"):
        HandModel.load(str(config))


# ----- OrcaUiSink over a fake link ---------------------------------------------------


class FakeLink:
    """In-memory UiLink stand-in: records outbound frames, scripted inbound."""

    def __init__(self):
        self.sent: list[dict] = []
        self.inbox: queue.Queue[dict] = queue.Queue()
        self.closed = False

    def send_json(self, message):
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(message)

    def recv_json(self, timeout=None):
        try:
            message = self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if message is None:
            raise ConnectionError("closed")
        return message

    def close(self):
        self.closed = True
        self.inbox.put(None)

    def of_type(self, kind):
        return [m for m in self.sent if m["type"] == kind]


def _make_sink(hand_model, link, **kwargs) -> OrcaUiSink:
    return OrcaUiSink(
        "ws://fake", "tok-123", "synthetic", hand_model,
        link_factory=lambda url: link, **kwargs)


def _prime_hello_ok(link, hand_model, config=None):
    link.inbox.put({"type": "hello_ok", "data": {
        "session_id": "tp-test", "proto": 1,
        "joints": list(hand_model.config.joint_ids),
        "roms": hand_model.config.joint_roms_dict,
        "engaged": False,
        "config": config or {},
    }})


def test_sink_handshake_and_inbound_messages(hand_model):
    link = FakeLink()
    _prime_hello_ok(link, hand_model)
    sink = _make_sink(hand_model, link)
    sink.connect()
    try:
        hello = link.of_type("hello")[0]["data"]
        assert hello["token"] == "tok-123"
        assert hello["proto"] == 1
        assert hello["source"] == "synthetic"
        assert hello["hand"]["side"] == "right"
        assert sink.session_id == "tp-test"

        # config messages land in ConfigState (read by the wrist provider)
        link.inbox.put({"type": "config", "data": {"manual_wrist_deg": 15.0}})
        deadline = time.time() + 2
        while time.time() < deadline and sink.config.wrist_deg() != 15.0:
            time.sleep(0.01)
        assert sink.config.wrist_deg() == 15.0

        # stop message sets the shared stop event
        link.inbox.put({"type": "stop", "data": {}})
        assert sink.stop_event.wait(timeout=2.0)
    finally:
        sink.close()


def test_sink_applies_hello_ok_config(hand_model):
    """Config accumulated before the child connected (e.g. preview toggled
    during startup) rides along in hello_ok and must be applied."""
    link = FakeLink()
    _prime_hello_ok(link, hand_model,
                    config={"preview": True, "manual_wrist_deg": -7.5})
    sink = _make_sink(hand_model, link)
    sink.connect()
    try:
        assert sink.config.get("preview") is True
        assert sink.config.wrist_deg() == -7.5
    finally:
        sink.close()


def test_sink_rejects_wrong_ack(hand_model):
    link = FakeLink()
    link.inbox.put({"type": "error", "data": {}})
    sink = _make_sink(hand_model, link)
    with pytest.raises(RuntimeError, match="hello_ok"):
        sink.connect()
    sink.close()


def test_sink_run_loop_coalesces_and_shuts_down(hand_model):
    link = FakeLink()
    _prime_hello_ok(link, hand_model)
    sink = _make_sink(hand_model, link, rate_hz=1000.0)
    sink.connect()
    try:
        actions: queue.Queue = queue.Queue()
        actions.put({"wrist": 1.0})
        actions.put({"wrist": 2.0})   # coalesced away
        actions.put({"wrist": 3.0})
        actions.put(_SHUTDOWN)
        stop = threading.Event()
        sink.run_loop(actions, stop)   # returns on _SHUTDOWN
        targets = link.of_type("targets")
        assert targets, "no targets sent"
        # latest-wins: the final pre-shutdown value must be the last sent
        assert targets[-1]["data"]["angles"]["wrist"] == 3.0
        assert targets[-1]["data"]["seq"] == len(targets)
    finally:
        sink.close()


def test_sink_link_death_stops_loop(hand_model):
    link = FakeLink()
    _prime_hello_ok(link, hand_model)
    sink = _make_sink(hand_model, link, rate_hz=1000.0)
    sink.connect()
    try:
        link.closed = True   # every send now raises
        actions: queue.Queue = queue.Queue()
        actions.put({"wrist": 1.0})
        stop = threading.Event()
        sink.run_loop(actions, stop)   # must return promptly, not spin
        assert sink.stop_event.is_set()
    finally:
        sink.close()


# ----- synthetic source ----------------------------------------------------------------


def test_synthetic_worker_stays_within_rom(hand_model):
    actions: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()
    thread = threading.Thread(
        target=synthetic_worker, args=(hand_model, actions, stop), daemon=True)
    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2.0)

    frames = []
    while not actions.empty():
        frames.append(actions.get_nowait())
    assert len(frames) >= 3
    roms = hand_model.config.joint_roms_dict
    for frame in frames:
        assert set(frame) == set(hand_model.config.joint_ids)
        for joint, value in frame.items():
            lo, hi = roms[joint]
            assert lo <= value <= hi


# ----- end-to-end over a real WebSocket -------------------------------------------------


def test_sink_against_real_websocket_server(hand_model):
    from websockets.sync.server import serve

    received: list[dict] = []
    hello_holder: list[dict] = []
    server_ready = threading.Event()

    def handler(ws):
        hello_holder.append(json.loads(ws.recv()))
        ws.send(json.dumps({"type": "hello_ok", "data": {
            "session_id": "tp-real", "proto": 1,
            "joints": list(hand_model.config.joint_ids),
            "roms": hand_model.config.joint_roms_dict,
            "engaged": False,
        }}))
        try:
            while True:
                received.append(json.loads(ws.recv()))
        except Exception:
            pass

    server = serve(handler, "127.0.0.1", 0)
    port = server.socket.getsockname()[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    server_ready.set()

    sink = OrcaUiSink(
        f"ws://127.0.0.1:{port}", "tok-real", "synthetic", hand_model,
        rate_hz=1000.0)
    try:
        sink.connect()
        actions: queue.Queue = queue.Queue()
        actions.put({"index_mcp": 42.0})
        actions.put(_SHUTDOWN)
        sink.run_loop(actions, threading.Event())

        deadline = time.time() + 2
        while time.time() < deadline and not any(
                m["type"] == "targets" for m in received):
            time.sleep(0.02)
        targets = [m for m in received if m["type"] == "targets"]
        assert targets and targets[0]["data"]["angles"]["index_mcp"] == 42.0
        assert hello_holder[0]["data"]["token"] == "tok-real"
    finally:
        sink.close()
        server.shutdown()
        server_thread.join(timeout=2.0)
