from __future__ import annotations

import math

import pytest
import torch
from kornia.geometry.transform import warp_affine

from core import facial_align
from core.facial_aligner import (
    _estimate_similarity_transform,
    canonical_landmarks,
    rotation_degrees_from_affine,
)


def _make_image(batch: int, size: int, rng: torch.Generator) -> torch.Tensor:
    image = torch.rand((batch, 3, size, size), generator=rng, dtype=torch.float32)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    pattern = torch.cat([xx.expand(batch, 1, size, size), yy.expand(batch, 1, size, size)], dim=1)
    return (0.7 * image + 0.3 * pattern.repeat(1, 2, 1, 1)[:, :3]).clamp(0.0, 1.0)


def _rotate_points(points: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    rotation = torch.tensor(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ],
        dtype=points.dtype,
    )
    center = torch.tensor([0.5, 0.5], dtype=points.dtype)
    return ((points - center) @ rotation.T) + center


def test_shape_aligned_bchw(rng: torch.Generator) -> None:
    image = _make_image(2, 512, rng)
    landmarks = canonical_landmarks(target_size=512, padding=0.2).unsqueeze(0).repeat(2, 1, 1)

    aligned, inverse = facial_align(image, landmarks, target_size=512, padding=0.2)

    assert aligned.shape == (2, 3, 512, 512)
    assert inverse.shape == (2, 3, 3)


def test_roundtrip_invariant(rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)
    canonical = canonical_landmarks(target_size=512, padding=0.2)
    tilted = _rotate_points(canonical, degrees=10.0)

    aligned, inverse = facial_align(image, tilted, target_size=512, padding=0.2)
    restored = warp_affine(
        aligned,
        inverse[:, :2, :],
        dsize=(512, 512),
        align_corners=False,
    )
    error = torch.mean(torch.abs(restored - image)).item()

    assert error < 0.15


def test_canonical_identity(rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)
    canonical = canonical_landmarks(target_size=512, padding=0.2)

    aligned, inverse = facial_align(image, canonical, target_size=512, padding=0.2)

    assert torch.allclose(aligned, image, atol=1e-4)
    assert torch.allclose(inverse[0], torch.eye(3), atol=1e-4)


def test_tilt_recovery() -> None:
    canonical = canonical_landmarks(target_size=512, padding=0.2).unsqueeze(0)
    tilted = _rotate_points(canonical.squeeze(0), degrees=10.0).unsqueeze(0)

    affine = _estimate_similarity_transform(
        tilted,
        image_size=(512, 512),
        target_size=512,
        padding=0.2,
    )
    angle = rotation_degrees_from_affine(affine).item()

    assert angle == pytest.approx(-10.0, abs=0.5)


def test_landmark_format_normalized(rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)
    landmarks = canonical_landmarks(target_size=512, padding=0.2).tolist()

    aligned, inverse = facial_align(image, landmarks, target_size=512, padding=0.2)

    assert aligned.shape == (1, 3, 512, 512)
    assert inverse.shape == (1, 3, 3)


def test_landmark_format_pixel_abs(rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)
    normalized = canonical_landmarks(target_size=512, padding=0.2)
    pixel = normalized * 511.0

    aligned_norm, inverse_norm = facial_align(image, normalized, target_size=512, padding=0.2)
    aligned_px, inverse_px = facial_align(image, pixel.tolist(), target_size=512, padding=0.2)

    assert torch.allclose(aligned_px, aligned_norm, atol=1e-4)
    assert torch.allclose(inverse_px, inverse_norm, atol=1e-4)


def test_raise_on_invalid_format(rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)

    with pytest.raises(ValueError, match="5x2|Bx5x2"):
        facial_align(image, [[0.1, 0.2], [0.3, 0.4]], target_size=512, padding=0.2)


@pytest.mark.parametrize("target_size", [512, 768, 1024])
def test_target_size_variants(target_size: int, rng: torch.Generator) -> None:
    image = _make_image(1, 512, rng)
    landmarks = canonical_landmarks(target_size=512, padding=0.2)

    aligned, inverse = facial_align(image, landmarks, target_size=target_size, padding=0.2)

    assert aligned.shape == (1, 3, target_size, target_size)
    assert inverse.shape == (1, 3, 3)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str, rng: torch.Generator) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    base = _make_image(1, 512, rng)
    landmarks = _rotate_points(canonical_landmarks(target_size=512, padding=0.2), degrees=7.5)
    reference_aligned, reference_inverse = facial_align(
        base,
        landmarks,
        target_size=512,
        padding=0.2,
    )

    image = base.to(device=device_name)
    actual_aligned, actual_inverse = facial_align(
        image,
        landmarks.to(device=device_name),
        target_size=512,
        padding=0.2,
    )

    assert actual_aligned.device.type == device_name
    assert actual_inverse.device.type == device_name
    assert torch.allclose(actual_aligned.cpu(), reference_aligned, atol=1e-4)
    assert torch.allclose(actual_inverse.cpu(), reference_inverse, atol=1e-4)
