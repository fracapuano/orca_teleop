"""Demo: random wrist-pose targets → IK → meshcat arm visualization.

Drives both arms of the orcabot simultaneously. Samples reachable wrist
poses via FK at random joint configs, solves full-pose IK with pink, and
smoothly interpolates the arms between targets in a meshcat browser
viewer.  Target and current-EE poses are shown as triads (XYZ axes).

    python scripts/teleop_arm_sim.py
"""

import argparse
import logging
import time

import numpy as np

from orca_teleop.orca_arm_sink import BimanualIKSolver, OrcaArmMeshcatSink

logger = logging.getLogger(__name__)

INTERP_DURATION = 0.5  # seconds per target transition
TICK_DT = 0.02  # meshcat refresh interval
SIDES = ("left", "right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ik = BimanualIKSolver()
    sink = OrcaArmMeshcatSink()
    rng = np.random.default_rng()

    # Start from a non-singular config
    q = ik.neutral_q
    for side in SIDES:
        idx_q = ik._arm_idx_q[side]
        q[idx_q[1]] = 0.1  # joint2
        q[idx_q[3]] = 1.2  # joint4

    # Seed first targets for both arms (SE3 poses)
    targets = {side: ik.sample_reachable_target(side, rng) for side in SIDES}
    result = ik.solve(targets, q)
    q_target = result.q

    # Target triads from the SE3 targets
    target_Ts = {side: targets[side].homogeneous for side in SIDES}

    for side in SIDES:
        logger.info(
            "%s initial IK: pos=%.2fmm ori=%.2fdeg converged=%s",
            side,
            result.position_error[side] * 1000,
            np.degrees(result.orientation_error[side]),
            result.converged[side],
        )

    q_start = q.copy()
    steps_per_target = int(INTERP_DURATION / TICK_DT)
    step = 0

    sink.launch()
    logger.info("Meshcat viewer launched. Press Ctrl+C to stop.")

    try:
        while True:
            t = (step % steps_per_target) / steps_per_target
            alpha = 0.5 - 0.5 * np.cos(np.pi * t)

            q_current = q_start + alpha * (q_target - q_start)

            arm_angles = {
                side: np.array([q_current[idx] for idx in ik._arm_idx_q[side]]) for side in SIDES
            }
            sink.update(arm_angles, target_Ts=target_Ts)

            time.sleep(TICK_DT)
            step += 1

            if step % steps_per_target == 0:
                q_start = q_target.copy()
                targets = {side: ik.sample_reachable_target(side, rng) for side in SIDES}
                result = ik.solve(targets, q_start)
                q_target = result.q
                target_Ts = {side: targets[side].homogeneous for side in SIDES}
                for side in SIDES:
                    logger.info(
                        "%s: pos=%.2fmm ori=%.2fdeg converged=%s",
                        side,
                        result.position_error[side] * 1000,
                        np.degrees(result.orientation_error[side]),
                        result.converged[side],
                    )

    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
