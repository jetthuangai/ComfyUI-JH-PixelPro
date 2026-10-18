"""Auto Upright core (``core/upright.py``) on synthetic building facades."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch

from core.upright import auto_upright, detect_segments


def _facade(height: int = 600, width: int = 800) -> np.ndarray:
    img = np.full((height, width, 3), 200, np.uint8)
    for x in range(60, width, 70):
        cv2.rectangle(img, (x, 80), (x + 35, height - 80), (60, 60, 60), -1)
    for y in range(100, height - 80, 90):
        cv2.line(img, (20, y), (width - 20, y), (30, 30, 30), 4)
    return img


def _tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0)[None]


def _rotated(degrees: float) -> np.ndarray:
    img = _facade()
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(200, 200, 200))


def _keystone(pitch_degrees: float) -> np.ndarray:
    img = _facade()
    h, w = img.shape[:2]
    k = np.array([[800.0, 0, w / 2], [0, 800.0, h / 2], [0, 0, 1]])
    a = math.radians(pitch_degrees)
    r = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    return cv2.warpPerspective(img, k @ r @ np.linalg.inv(k), (w, h), borderValue=(200,) * 3)


def _vertical_deviation(image: torch.Tensor) -> float:
    """Median absolute deviation (degrees) of near-vertical segments from vertical."""
    segments = detect_segments(image[0])
    vertical = np.abs(segments.angle) > 60
    dev = segments.angle[vertical] - np.sign(segments.angle[vertical]) * 90.0
    return float(np.median(np.abs(dev)))


def _rotation_degrees(matrix: list[list[float]]) -> float:
    return math.degrees(math.atan2(matrix[1][0], matrix[0][0]))


@pytest.mark.parametrize("mode", ["level", "auto", "vertical", "full"])
def test_recovers_known_tilt(mode: str) -> None:
    _, confidence, matrices = auto_upright(_tensor(_rotated(4.0)), mode=mode)
    assert confidence[0] > 0.6
    assert abs(_rotation_degrees(matrices[0]) - 4.0) < 0.5


def test_keystone_verticals_become_parallel() -> None:
    image = _tensor(_keystone(8.0))
    assert _vertical_deviation(image) > 1.0
    out, confidence, _ = auto_upright(image, mode="vertical")
    assert confidence[0] > 0.6
    assert _vertical_deviation(out) < 0.3


def test_level_mode_does_not_fix_keystone() -> None:
    out, _, _ = auto_upright(_tensor(_keystone(8.0)), mode="level")
    assert _vertical_deviation(out) > 1.0


def test_no_lines_leaves_image_unchanged() -> None:
    noise = torch.rand(1, 120, 160, 3, generator=torch.Generator().manual_seed(0))
    out, confidence, matrices = auto_upright(noise)
    assert torch.equal(out, noise)
    assert confidence == [0.0]
    assert np.allclose(matrices[0], np.eye(3))


def test_batch_is_processed_per_image() -> None:
    noise = torch.rand(1, 600, 800, 3, generator=torch.Generator().manual_seed(1))
    batch = torch.cat([_tensor(_rotated(3.0)), noise])
    out, confidence, matrices = auto_upright(batch, mode="level")
    assert confidence[0] > 0.6 and confidence[1] == 0.0
    assert torch.equal(out[1], noise[0])
    assert abs(_rotation_degrees(matrices[0]) - 3.0) < 0.5


def test_strength_and_max_angle_scale_the_correction() -> None:
    image = _tensor(_rotated(4.0))
    _, _, half = auto_upright(image, mode="level", strength=0.5)
    assert abs(_rotation_degrees(half[0]) - 2.0) < 0.3
    _, _, capped = auto_upright(image, mode="level", max_angle=1.0)
    assert abs(_rotation_degrees(capped[0]) - 1.0) < 0.1
    out, confidence, _ = auto_upright(image, mode="level", strength=0.0)
    assert torch.equal(out, image) and confidence == [0.0]


def test_valid_crop_has_no_empty_corners() -> None:
    image = _tensor(_rotated(5.0))
    none, _, _ = auto_upright(image, mode="level", crop="none")
    valid, _, _ = auto_upright(image, mode="level", crop="valid")
    assert float(none[0, :3, :3].max()) == 0.0  # rotated corner left empty
    assert float(valid.amax(-1).min()) > 0.05  # every pixel comes from the photo


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "tilt"}, "mode"),
        ({"crop": "fit"}, "crop"),
        ({"strength": 2.0}, "strength"),
        ({"max_angle": 90.0}, "max_angle"),
    ],
)
def test_validates(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        auto_upright(torch.rand(1, 16, 16, 3), **kwargs)
