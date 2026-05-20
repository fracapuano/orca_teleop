import numpy as np

from orca_teleop.orca_arm_ik import orca_panda_right_ik_config
from orca_teleop.orca_arm_sink import OrcaArmMeshcatSink


def test_meshcat_sink_can_load_orca_panda_joint_mapping() -> None:
    config = orca_panda_right_ik_config()
    home = {
        "right": np.array(
            [-0.1, -1.6, -0.1, -3.0718, -0.15, 2.85, -1.4027],
            dtype=np.float64,
        )
    }

    sink = OrcaArmMeshcatSink(
        urdf_path=config.urdf_path,
        sides=config.sides,
        joint_names_by_side=config.joint_names_by_side,
        ee_frame_by_side=config.ee_frame_by_side,
        home_arm_angles=home,
    )

    assert sink.arm_joint_names == {
        "right": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
    }
    assert sink._ee_links == {"right": "orcahand_right_R-Carpals_8d1f1041"}
