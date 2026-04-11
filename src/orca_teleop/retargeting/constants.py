KEY_VECTORS_SHAPE = (5, 3)


# TODO: replace these hand-tuned magic numbers with the centres of the per-
# fingertip collision geometry from the URDF, or adopt MJCF throughout directly.
FINGERTIP_OFFSETS: dict[str, tuple[float, float, float]] = {
    "thumb": (0.0, 0.0, 0.0305),
    "index": (0.0, 0.0, 0.0433),
    "middle": (0.0, 0.0, 0.0453),
    "ring": (0.0, 0.0, 0.0453),
    "pinky": (0.0, 0.0, 0.0383),
}

# Auto-scale calibration: number of frames to collect MANO key-vector magnitudes
# before computing the MANO→metres scale ratio.
CALIBRATION_FRAMES = 30

# Per-finger loss weights. Equal coefficients only rescale the loss (and thus
# the gradient)
DEFAULT_LOSS_COEFFS: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)

# Translation applied after rotating normalized MANO joints into the URDF base
MANO_TO_URDF_TRANSLATION = (0.0, 0.0, -0.02)
