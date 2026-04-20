from __future__ import annotations

import math

import pytest
import torch

from core import lens_distortion
from core.lens_distortion import _forward_grid


def _gradient_image(size: int) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    return torch.cat(
        [
            xx.expand(1, 1, size, size),
            yy.expand(1, 1, size, size),
            ((xx + yy) / 2.0).expand(1, 1, size, size),
        ],
        dim=1,
    )


def _checkerboard(size: int, step: int = 64) -> torch.Tensor:
    image = torch.zeros((1, 3, size, size), dtype=torch.float32)
    for pos in range(step, size - step, step):
        image[:, :, :, pos : pos + 2] = 1.0
        image[:, :, pos : pos + 2, :] = 1.0
    return image


def _line_rms(image_bchw: torch.Tensor, *, step: int = 64) -> float:
    gray = image_bchw.mean(dim=1)[0]
    size = gray.shape[0]
    errors: list[float] = []
    for pos in range(step, size - step, step):
        for y in range(step, size - step, 16):
            window = gray[y, pos - 4 : pos + 6]
            xs = torch.arange(pos - 4, pos + 6, dtype=torch.float32)
            centroid = (window * xs).sum() / window.sum().clamp_min(1e-6)
            errors.append(float((centroid - pos).abs()))
        for x in range(step, size - step, 16):
            window = gray[pos - 4 : pos + 6, x]
            ys = torch.arange(pos - 4, pos + 6, dtype=torch.float32)
            centroid = (window * ys).sum() / window.sum().clamp_min(1e-6)
            errors.append(float((centroid - pos).abs()))
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def test_shape_inverse_bchw() -> None:
    image = _gradient_image(256)
    output = lens_distortion(image, direction="inverse", k1=-0.18, k2=0.08, k3=-0.02)
    assert output.shape == image.shape


def test_shape_forward_bchw() -> None:
    image = _gradient_image(256)
    output = lens_distortion(image, direction="forward", k1=-0.18, k2=0.08, k3=-0.02)
    assert output.shape == image.shape


def test_no_op_identity_coef() -> None:
    image = _gradient_image(256)
    output = lens_distortion(image, direction="inverse")
    assert torch.mean(torch.abs(output - image)).item() < 1e-3


def test_canon_24mm_preset_inverse() -> None:
    image = _checkerboard(512)
    distorted = lens_distortion(
        image,
        direction="forward",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
    )
    restored = lens_distortion(
        distorted,
        direction="inverse",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
    )
    assert _line_rms(restored) < 1.0


def test_roundtrip_invariant() -> None:
    image = _gradient_image(512)
    distorted = lens_distortion(
        image,
        direction="forward",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
    )
    restored = lens_distortion(
        distorted,
        direction="inverse",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
    )
    assert torch.mean(torch.abs(restored - image)).item() * 255.0 < 2.0


def test_direction_forward_monotone() -> None:
    size = 256
    grid = _forward_grid(
        batch=1,
        height=size,
        width=size,
        device=torch.device("cpu"),
        k1=0.1,
        k2=0.0,
        k3=0.0,
        p1=0.0,
        p2=0.0,
    )[0]

    def displacement(y: int, x: int) -> float:
        source_x = ((grid[y, x, 0].item() + 1.0) * size / 2.0) - 0.5
        source_y = ((grid[y, x, 1].item() + 1.0) * size / 2.0) - 0.5
        source = torch.tensor([source_x, source_y], dtype=torch.float32)
        target = torch.tensor([float(x), float(y)], dtype=torch.float32)
        return float(torch.linalg.norm(source - target))

    near = displacement(size // 2, size // 2 + 16)
    mid = displacement(size // 2, size // 2 + 48)
    far = displacement(size // 2, size - 16)
    assert near < mid < far


def test_clamp_warn_oor(caplog: pytest.LogCaptureFixture) -> None:
    image = _gradient_image(128)
    image[0, 0, 0, 0] = -0.1
    image[0, 1, 0, 1] = 1.1
    with caplog.at_level("WARNING"):
        output = lens_distortion(image, direction="inverse")
    assert "outside [0,1]" in caplog.text
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_raise_invalid_direction() -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        lens_distortion(_gradient_image(64), direction="wrong")


def test_raise_invalid_rank() -> None:
    with pytest.raises(ValueError, match="Expected BCHW tensor"):
        lens_distortion(torch.rand((3, 64, 64), dtype=torch.float32))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    image = _gradient_image(256)
    reference = lens_distortion(
        image,
        direction="forward",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
        p1=0.001,
        p2=-0.001,
    )
    actual = lens_distortion(
        image.to(device=device_name),
        direction="forward",
        k1=-0.18,
        k2=0.08,
        k3=-0.02,
        p1=0.001,
        p2=-0.001,
    )
    assert actual.device.type == device_name
    assert torch.allclose(actual.cpu(), reference, atol=5e-4)


def test_bounds_coefficient_range() -> None:
    image = _gradient_image(64)
    lens_distortion(image, k1=1.0, p1=0.1, direction="inverse")
    with pytest.raises(ValueError, match="k1 must be in"):
        lens_distortion(image, k1=1.01)
    with pytest.raises(ValueError, match="p1 must be in"):
        lens_distortion(image, p1=0.11)
