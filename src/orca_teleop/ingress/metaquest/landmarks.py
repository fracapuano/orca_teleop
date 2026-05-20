"""MetaQuest landmark adapters.

The MetaQuest publisher forwards HTS landmarks in the device's left-handed
native coordinate convention. This util converts them to the retargeter's
right-handed convention.
"""

import numpy as np


def retargeter_landmarks_from_quest(points: np.ndarray, handedness: str) -> np.ndarray:
    """Return MetaQuest landmarks in the convention expected by the retargeter.

    The right-hand HTS landmarks arrive mirrored relative to the retargeter's
    local hand-frame assumption; flipping the device Y axis restores the
    expected open/close direction. Left-hand landmarks are already compliant.
    """
    if handedness not in ("left", "right"):
        raise ValueError(f"Unsupported MetaQuest handedness: {handedness!r}")

    landmarks = np.asarray(points, dtype=np.float64).copy()
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"MetaQuest landmarks must have shape (N, 3); got {landmarks.shape}")

    if handedness == "right":
        landmarks[:, 1] *= -1.0  # flips Y axis

    return landmarks
