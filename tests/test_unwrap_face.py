from __future__ import annotations

import math

import pytest
import torch

from core import unwrap_face
from core.facial_aligner import canonical_landmarks, facial_align


def _gradient_image(batch: int, size: int) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    base = torch.cat(
        [
            xx.expand(batch, 1, size, size),
            yy.expand(batch, 1, size, size),
            ((xx + yy) / 2.0).expand(batch, 1, size, size),
        ],
        dim=1,
    )
    return base


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


def _translated_inverse(batch: int = 1) -> torch.Tensor:
    matrix = torch.tensor(
        [[[1.0, 0.0, 32.0], [0.0, 1.0, 32.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return matrix.repeat(batch, 1, 1)


def test_shape_composite_matches_original() -> None:
    edited = _gradient_image(1, 64)
    original = _gradient_image(1, 128)
    composited, _ = unwrap_face(edited, original, _translated_inverse())
    assert composited.shape == original.shape


def test_shape_mask_matches_original() -> None:
    edited = _gradient_image(1, 64)
    original = _gradient_image(1, 128)
    _, mask = unwrap_face(edited, original, _translated_inverse())
    assert mask.shape == (1, 128, 128)


def test_inverse_matrix_b3x3_parse() -> None:
    edited = _gradient_image(1, 64)
    original = _gradient_image(1, 128)
    matrix_3x3 = _translated_inverse()
    matrix_2x3 = matrix_3x3[:, :2, :]
    out_3x3, mask_3x3 = unwrap_face(edited, original, matrix_3x3)
    out_2x3, mask_2x3 = unwrap_face(edited, original, matrix_2x3)
    assert torch.allclose(out_3x3, out_2x3, atol=1e-6)
    assert torch.allclose(mask_3x3, mask_2x3, atol=1e-6)


def test_roundtrip_identity() -> None:
    image = _gradient_image(1, 512)
    landmarks = _rotate_points(canonical_landmarks(target_size=512, padding=0.2), degrees=9.0)
    aligned, inverse = facial_align(image, landmarks, target_size=512, padding=0.2)
    composited, _ = unwrap_face(aligned, image, inverse, feather_radius=0.0)
    assert torch.mean(torch.abs(composited - image)).item() * 255.0 < 2.0


def test_feather_zero_hard_edge() -> None:
    edited = torch.ones((1, 3, 64, 64), dtype=torch.float32)
    original = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    _, mask = unwrap_face(edited, original, _translated_inverse(), feather_radius=0.0)
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}


def test_feather_nonzero_soft_taper() -> None:
    edited = torch.ones((1, 3, 64, 64), dtype=torch.float32)
    original = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    _, mask = unwrap_face(edited, original, _translated_inverse(), feather_radius=8.0)
    assert 0.0 < mask[0, 28, 32].item() < mask[0, 32, 32].item() < 1.0
    assert mask[0, 64, 64].item() == pytest.approx(1.0, abs=1e-6)


def test_mask_override_ignores_feather() -> None:
    edited = torch.ones((1, 3, 64, 64), dtype=torch.float32)
    original = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
    override = torch.zeros((1, 128, 128), dtype=torch.float32)
    override[:, 40:88, 40:88] = 0.5
    composited, mask = unwrap_face(
        edited,
        original,
        _translated_inverse(),
        feather_radius=16.0,
        mask_override=override,
    )
    assert torch.allclose(mask, override, atol=1e-6)
    assert composited[0, 0, 60, 60].item() == pytest.approx(0.5, abs=1e-6)


def test_batch_independent_m_inv() -> None:
    edited = torch.stack(
        [
            torch.full((3, 64, 64), 0.2, dtype=torch.float32),
            torch.full((3, 64, 64), 0.5, dtype=torch.float32),
            torch.full((3, 64, 64), 0.8, dtype=torch.float32),
        ],
        dim=0,
    )
    original = torch.zeros((3, 3, 128, 128), dtype=torch.float32)
    matrices = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 16.0], [0.0, 1.0, 16.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 32.0], [0.0, 1.0, 32.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    composited, _ = unwrap_face(edited, original, matrices, feather_radius=0.0)
    assert composited[0, 0, 10, 10].item() == pytest.approx(0.2, abs=1e-6)
    assert composited[1, 0, 24, 24].item() == pytest.approx(0.5, abs=1e-6)
    assert composited[2, 0, 40, 40].item() == pytest.approx(0.8, abs=1e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    edited = _gradient_image(1, 64)
    original = _gradient_image(1, 128)
    reference_image, reference_mask = unwrap_face(
        edited,
        original,
        _translated_inverse(),
        feather_radius=8.0,
    )
    actual_image, actual_mask = unwrap_face(
        edited.to(device=device_name),
        original.to(device=device_name),
        _translated_inverse().to(device=device_name),
        feather_radius=8.0,
    )
    assert actual_image.device.type == device_name
    assert torch.allclose(actual_image.cpu(), reference_image, atol=1e-4)
    assert torch.allclose(actual_mask.cpu(), reference_mask, atol=1e-4)


def test_raise_invalid_rank() -> None:
    with pytest.raises(ValueError, match="Expected edited_aligned_bchw BCHW tensor"):
        unwrap_face(
            torch.rand((3, 64, 64), dtype=torch.float32),
            _gradient_image(1, 128),
            _translated_inverse(),
        )


def test_mask_override_shape_mismatch() -> None:
    edited = _gradient_image(1, 64)
    original = _gradient_image(1, 128)
    bad_mask = torch.ones((1, 64, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="mask_override HxW"):
        unwrap_face(edited, original, _translated_inverse(), mask_override=bad_mask)
