"""Arm-side consumer of the operator's 6-DoF wrist pose.

The hand and the arm are driven from the *same* ingress frame, but on
different clocks: the hand goes through the retargeter (~100 ms/frame), while
the arm reads the newest frame directly at ingress rate. Splitting at the
source instead would give two clocks and let the grasp close at the wrong
pose.

What lives here is only the plumbing. Arm *retargeting* — clutch/engage,
origin calibration (``T_target = T_robot0 @ inv(T_op0) @ T_op``), scaling,
workspace clamping and rate limiting — belongs to the arm controller, i.e. to
:class:`ArmSink` implementations.

An implementation sees frames that are already known-good: present wrist
pose, unit quaternion, fresh, and from the owning publisher. Everything that
would otherwise be a check the implementer has to remember is instead a
callback they cannot fail to receive::

    from orca_teleop.arm import ArmSink
    from orca_teleop.pipeline import run_metaquest_local

    class TronArmSink(ArmSink):
        def connect(self):
            self.arm = my_tron_client.connect("192.168.1.50")

        def on_reference_change(self, stream_id, pose_epoch):
            self.arm.hold()          # origin is stale; re-clutch before moving
            self.origin = None

        def dispatch(self, frame):
            # frame.wrist.as_xyz_wxyz() is a TRON left_pos/right_pos verbatim
            self.arm.servop(left_pos=frame.wrist.as_xyz_wxyz().tolist())

        def on_hold(self, reason):
            self.arm.hold()

        def close(self):
            self.arm.disable()

    run_metaquest_local(model_path=".../config.yaml", arm_sink=TronArmSink())

For bring-up without any arm, ``python -m orca_teleop.arm --port 50051``
starts a bare gRPC ingress and prints incoming poses — no retargeter, no
hand, no torch.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Literal

from orca_teleop.constants import ARM_STALE_AFTER_S, HEARTBEAT_INTERVAL
from orca_teleop.ingress.frames import LatestFrame, TeleopFrame

logger = logging.getLogger(__name__)

HoldReason = Literal["no_frames", "stale", "tracking_invalid"]

__all__ = ["ArmSink", "HoldReason", "arm_worker", "main"]


class ArmSink(ABC):
    """Consumer of operator wrist poses. Implemented by the arm controller.

    An ABC rather than a Protocol so that a misspelled or missing method
    fails loudly at construction instead of silently never being called.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the link to the arm. Raising here aborts pipeline startup."""

    @abstractmethod
    def dispatch(self, frame: TeleopFrame) -> None:
        """Command the arm from one fresh operator frame.

        Called at ingress rate (~30-60 Hz). Note the TRON's ServoJ documents a
        >=500 Hz control requirement, so an implementation is expected to
        interpolate between these frames rather than forward them one-to-one;
        ``frame.recv_monotonic_ns`` is the clock to interpolate against.
        """

    @abstractmethod
    def on_hold(self, reason: HoldReason) -> None:
        """Frames stopped arriving, or arrived unusable. Hold position.

        Edge-triggered: called once when the hold begins, not once per poll.
        """

    @abstractmethod
    def on_reference_change(self, stream_id: int, pose_epoch: int) -> None:
        """The pose reference space changed; any latched origin is invalid.

        Fires on a new publisher (``stream_id``) or a re-pinned XR reference
        space (``pose_epoch``) — the "operator took the headset off and put it
        back on" case — and once before the very first frame.
        """

    @abstractmethod
    def close(self) -> None:
        """Park or disable the arm. Always called, even on an error path."""


def arm_worker(
    frames: LatestFrame,
    sink: ArmSink,
    stop_event: threading.Event,
    stale_after_s: float = ARM_STALE_AFTER_S,
    poll_timeout_s: float = HEARTBEAT_INTERVAL,
) -> None:
    """Drive ``sink`` from the newest ingress frame until stopped.

    Owns the staleness deadline, the edge-triggered hold, the reference-change
    re-arm, and containment of exceptions raised by the sink. Never touches
    ``actions_q`` or the retargeter, so a slow arm cannot delay the hand and a
    slow retargeter cannot delay the arm.
    """
    last_seq = 0
    holding = False
    reference: tuple[int, int] | None = None

    def enter_hold(reason: HoldReason) -> None:
        nonlocal holding
        if holding:
            return
        holding = True
        try:
            sink.on_hold(reason)
        except Exception:
            logger.exception("ArmSink.on_hold(%s) raised", reason)

    try:
        while not stop_event.is_set():
            update = frames.wait_for_next(last_seq, timeout=poll_timeout_s)
            if update.closed:
                # Distinct from a hold: the ingress is gone and no frame is
                # ever coming, so park rather than wait pointing at nothing.
                logger.info("Ingress closed; stopping the arm worker.")
                break
            if not update.fresh:
                # A wakeup with no new frame is not yet a hold: poll_timeout_s
                # is only how often we look, whereas stale_after_s is the
                # actual liveness deadline. Holding on the poll interval would
                # freeze the arm after ~3 missed frames at 30 Hz.
                previous = update.frame
                if previous is None or previous.age_s > stale_after_s:
                    enter_hold("no_frames")
                continue

            last_seq = update.seq
            frame = update.frame
            if frame is None:
                enter_hold("no_frames")
                continue
            if frame.age_s > stale_after_s:
                # Arrived, but already too old to be a safe target — a stalled
                # link that just resumed, or a backed-up publisher.
                enter_hold("stale")
                continue
            if not frame.tracking_valid:
                enter_hold("tracking_invalid")
                continue

            current_reference = (frame.stream_id, frame.pose_epoch)
            if current_reference != reference:
                reference = current_reference
                try:
                    sink.on_reference_change(frame.stream_id, frame.pose_epoch)
                except Exception:
                    logger.exception("ArmSink.on_reference_change raised")

            try:
                sink.dispatch(frame)
            except Exception:
                # One bad command must not kill the arm thread and strand the
                # arm wherever it happened to be. Note `holding` is cleared
                # only on success, so a sink that raises every frame gets one
                # on_hold, not one per frame.
                logger.exception("ArmSink.dispatch raised; holding")
                enter_hold("no_frames")
            else:
                holding = False
    finally:
        try:
            sink.close()
        except Exception:
            logger.exception("ArmSink.close raised")


class _LoggingArmSink(ArmSink):
    """Bring-up sink: prints what a real arm would have been commanded."""

    def __init__(self, log_every: int = 15) -> None:
        self._log_every = max(1, log_every)
        self._count = 0

    def connect(self) -> None:
        logger.info("Logging arm sink ready — no arm is being commanded.")

    def on_reference_change(self, stream_id: int, pose_epoch: int) -> None:
        logger.info(
            "Pose reference changed (stream=%d, epoch=%d) — an arm would re-clutch here.",
            stream_id,
            pose_epoch,
        )

    def dispatch(self, frame: TeleopFrame) -> None:
        self._count += 1
        if self._count % self._log_every:
            return
        position = frame.wrist_position_m
        quaternion = frame.wrist_orientation_wxyz
        logger.info(
            "%s wrist  xyz=[%+.3f %+.3f %+.3f] m  wxyz=[%+.3f %+.3f %+.3f %+.3f]  "
            "head=%s  age=%.0f ms",
            frame.handedness,
            position[0],
            position[1],
            position[2],
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
            "yes" if frame.head is not None else "no",
            frame.age_s * 1e3,
        )

    def on_hold(self, reason: HoldReason) -> None:
        logger.warning("Hold (%s) — an arm would freeze here.", reason)

    def close(self) -> None:
        logger.info("Logging arm sink closed after %d frames.", self._count)


def main(argv: list[str] | None = None) -> None:
    """Run a bare ingress and print wrist poses. Bring-up only."""
    import argparse
    import queue

    from orca_teleop.constants import DEFAULT_PORT, QUEUES_MAXSIZE
    from orca_teleop.ingress.server import IngressServer

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="gRPC ingress port")
    parser.add_argument(
        "--log-every", type=int, default=15, help="print one in N frames (default: 15)"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stop_event = threading.Event()
    # Drained and discarded: without a consumer the queue fills, and the
    # servicer's latest-wins drop would then run on every single frame.
    landmarks_q: queue.Queue = queue.Queue(maxsize=QUEUES_MAXSIZE)
    server = IngressServer(landmarks_q, stop_event, port=args.port)
    bound_port = server.start()
    logger.info("Arm bring-up ingress listening on :%d", bound_port)

    sink = _LoggingArmSink(log_every=args.log_every)
    sink.connect()
    drain = threading.Thread(
        target=_drain_forever, args=(landmarks_q, stop_event), name="drain", daemon=True
    )
    drain.start()
    try:
        arm_worker(server.frames, sink, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.stop()


def _drain_forever(landmarks_q, stop_event: threading.Event) -> None:
    import queue as _queue

    while not stop_event.is_set():
        try:
            landmarks_q.get(timeout=HEARTBEAT_INTERVAL)
        except _queue.Empty:
            continue


if __name__ == "__main__":
    main()
