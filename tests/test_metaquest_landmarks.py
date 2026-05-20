"""Tests for MetaQuest landmark adapters."""

import numpy as np
import pytest

from orca_teleop.ingress.metaquest.landmarks import retargeter_landmarks_from_quest


def test_right_hand_landmarks_flip_device_y_axis():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    converted = retargeter_landmarks_from_quest(points, "right")

    assert converted.tolist() == [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]


def test_left_hand_landmarks_are_unchanged():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    converted = retargeter_landmarks_from_quest(points, "left")

    assert converted.tolist() == points.tolist()


def test_landmark_conversion_does_not_mutate_input():
    points = np.array([[1.0, 2.0, 3.0]])

    converted = retargeter_landmarks_from_quest(points, "right")

    assert points.tolist() == [[1.0, 2.0, 3.0]]
    assert not np.allclose(converted, points)


def test_landmark_conversion_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        retargeter_landmarks_from_quest(np.zeros((21,)), "right")


def test_landmark_conversion_rejects_invalid_handedness():
    with pytest.raises(ValueError, match="handedness"):
        retargeter_landmarks_from_quest(np.zeros((21, 3)), "center")
