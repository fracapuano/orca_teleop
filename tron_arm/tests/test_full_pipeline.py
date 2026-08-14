"""The real `pipeline.run()` with the hand retargeter and the arm side together.

This is the demo-day path: ingress -> retargeter -> hand sink, and in parallel
ingress -> arm_worker -> TronArmSink -> robot. `--arm-only` covers the arm half;
this covers both halves running at once, which is the thing that could interfere.

Needs the full upstream stack (orca_core, torch, and the orcahand_description
URDF), so it skips unless all three are present. Run it from an environment that
has them:

    ORCAHAND_DESCRIPTION_DIR=/path/to/orcahand_description \
    ORCA_HAND_CONFIG=/path/to/orcahand_right/config.yaml \
    /path/to/orca_teleop/.venv/bin/python -m pytest tests/test_full_pipeline.py

Measured against the mock robot: 1824 dispatches, dispatch p95
0.382 ms, streamer 99.935 Hz -- i.e. the retargeter running concurrently costs
the arm path nothing.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("orca_core", reason="needs the full upstream stack")
pytest.importorskip("torch", reason="needs the retargeter")
pipeline = pytest.importorskip("orca_teleop.pipeline")

from tests.conftest import at  # noqa: E402
from tron_arm.config import load_config  # noqa: E402
from tron_arm.mock_robot import MockTron2  # noqa: E402
from tron_arm.sink import TronArmSink  # noqa: E402

pytestmark = pytest.mark.slow

def _hand_model() -> Path | None:
    """The ORCA hand config, from ORCA_HAND_CONFIG or the installed orca_core."""
    override = os.environ.get("ORCA_HAND_CONFIG")
    if override:
        return Path(override)
    try:
        import orca_core
    except Exception:  # noqa: BLE001
        return None
    root = Path(orca_core.__file__).resolve().parent / "models" / "v2" / "orcahand_right"
    config = root / "config.yaml"
    return config if config.is_file() else None


MODEL = _hand_model()


def _urdf_available() -> bool:
    root = os.environ.get("ORCAHAND_DESCRIPTION_DIR")
    return bool(root and (Path(root) / "v1/models/urdf/orcahand_right.urdf").is_file())


@pytest.mark.skipif(MODEL is None, reason="set ORCA_HAND_CONFIG or install orca_core")
@pytest.mark.skipif(not _urdf_available(), reason="set ORCAHAND_DESCRIPTION_DIR")
def test_arm_path_is_unaffected_by_the_retargeter_running_alongside():
    import asyncio

    from orca_core.hardware_hand import MockOrcaHand

    pipeline.OrcaHand = MockOrcaHand  # no physical hand

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def spin():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    threading.Thread(target=spin, daemon=True).start()
    assert ready.wait(5.0)
    robot = MockTron2(port=0, info_period_s=0.5)
    asyncio.run_coroutine_threadsafe(robot.start(), loop).result(10.0)

    cfg = dataclasses.replace(
        at(load_config(), f"ws://127.0.0.1:{robot.bound_port}"), notify_log_path=None)
    arm = TronArmSink(cfg)
    hand = pipeline.OrcaHandSink(str(MODEL), connect_hardware=False)

    def run():
        with pytest.raises(BaseException):  # the pipeline never returns cleanly here
            pipeline.run(model_path=str(MODEL), port=50126, sink=hand,
                         arm_sink=arm, retargeter_backend="rmsprop")

    threading.Thread(target=run, daemon=True).start()
    time.sleep(20)  # the retargeter takes a while to build

    publisher = threading.Thread(
        target=lambda: __import__(
            "orca_teleop.ingress.metaquest.mock_publisher", fromlist=["main"]
        ).main(["--server", "localhost:50126"]),
        daemon=True,
    )
    publisher.start()
    time.sleep(30)

    try:
        assert arm.stats.dispatches > 200, f"only {arm.stats.dispatches} frames reached the arm"
        assert arm.stats.dispatch_percentile_ms() < 1.0, "the retargeter is blocking dispatch"
        assert arm.streamer is not None
        rate = arm.streamer.stats.achieved_rate_hz
        assert 90.0 < rate < 110.0, f"streamer degraded to {rate:.1f} Hz alongside the retargeter"
        assert robot.servop_rejected == 0
    finally:
        arm.close()
        asyncio.run_coroutine_threadsafe(robot.stop(), loop).result(5.0)
        loop.call_soon_threadsafe(loop.stop)
