from __future__ import annotations

import pytest
import torch

from core import tone_curve

LINEAR_POINTS = torch.tensor(
    [
        [0.0, 0.0],
        [0.14, 0.14],
        [0.29, 0.29],
        [0.43, 0.43],
        [0.57, 0.57],
        [0.71, 0.71],
        [0.86, 0.86],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

S_CURVE_MILD = torch.tensor(
    [
        [0.0, 0.0],
        [0.15, 0.10],
        [0.30, 0.25],
        [0.45, 0.43],
        [0.55, 0.57],
        [0.70, 0.75],
        [0.85, 0.90],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

LIFT_SHADOWS = torch.tensor(
    [
        [0.0, 0.0],
        [0.10, 0.15],
        [0.25, 0.30],
        [0.40, 0.45],
        [0.55, 0.58],
        [0.70, 0.72],
        [0.85, 0.87],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

CRUSH_BLACKS = torch.tensor(
    [
        [0.0, 0.0],
        [0.15, 0.05],
        [0.30, 0.22],
        [0.45, 0.40],
        [0.60, 0.57],
        [0.75, 0.75],
        [0.90, 0.90],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

OVERSHOOT_POINTS = torch.tensor(
    [
        [0.0, 0.0],
        [0.10, 0.90],
        [0.20, 0.10],
        [0.40, 0.95],
        [0.60, 0.05],
        [0.80, 0.90],
        [0.90, 0.40],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)


def _gradient_image(size: int) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, 1, size)
    gray = ramp.expand(1, 1, size, size)
    return gray.repeat(1, 3, 1, 1)


def _color_image(size: int) -> torch.Tensor:
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


def test_shape_bchw() -> None:
    image = _gradient_image(64)

    output = tone_curve(image, control_points=LINEAR_POINTS)

    assert output.shape == image.shape


def test_linear_identity() -> None:
    image = _color_image(128)

    output = tone_curve(image, control_points=LINEAR_POINTS)

    assert torch.allclose(output, image, atol=1e-4)


def test_strength_zero_identity() -> None:
    image = _color_image(96)

    output = tone_curve(image, control_points=S_CURVE_MILD, strength=0.0)

    assert torch.equal(output, image)


def test_s_curve_increases_contrast() -> None:
    image = _gradient_image(256)

    output = tone_curve(image, control_points=S_CURVE_MILD)

    assert output.std().item() > image.std().item()


def test_lift_shadows_raises_blacks() -> None:
    image = _gradient_image(256)
    mask = image < 0.2

    output = tone_curve(image, control_points=LIFT_SHADOWS)

    assert output[mask].mean().item() > image[mask].mean().item()


def test_crush_blacks_deepens() -> None:
    image = _gradient_image(256)
    mask = image < 0.2

    output = tone_curve(image, control_points=CRUSH_BLACKS)

    assert output[mask].mean().item() < image[mask].mean().item()


def test_channel_rgb_master_uniform() -> None:
    image = _gradient_image(128)

    output = tone_curve(image, control_points=S_CURVE_MILD, channel="rgb_master")

    assert torch.allclose(output[:, 0], output[:, 1], atol=1e-6)
    assert torch.allclose(output[:, 1], output[:, 2], atol=1e-6)


def test_channel_r_isolates() -> None:
    image = _color_image(128)

    output = tone_curve(image, control_points=S_CURVE_MILD, channel="r")

    assert not torch.allclose(output[:, :1], image[:, :1], atol=1e-4)
    assert torch.allclose(output[:, 1:2], image[:, 1:2], atol=1e-6)
    assert torch.allclose(output[:, 2:3], image[:, 2:3], atol=1e-6)


def test_strength_half_matches_linear_blend() -> None:
    image = _color_image(64)
    full = tone_curve(image, control_points=S_CURVE_MILD, strength=1.0)
    blended = tone_curve(image, control_points=S_CURVE_MILD, strength=0.5)

    expected = (0.5 * image) + (0.5 * full)
    assert torch.allclose(blended, expected, atol=1e-5)


def test_endpoints_not_fixed_raises() -> None:
    points = LINEAR_POINTS.clone()
    points[0] = torch.tensor([0.1, 0.1], dtype=torch.float32)

    with pytest.raises(ValueError, match="start at \\(0,0\\)"):
        tone_curve(_gradient_image(32), control_points=points)


def test_non_monotone_x_raises() -> None:
    points = LINEAR_POINTS.clone()
    points[2, 0] = 0.10

    with pytest.raises(ValueError, match="strictly increasing"):
        tone_curve(_gradient_image(32), control_points=points)


@pytest.mark.parametrize("shape", [(7, 2), (9, 2)])
def test_invalid_shape_raises(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="shape \\(8, 2\\)"):
        tone_curve(_gradient_image(32), control_points=torch.zeros(shape, dtype=torch.float32))


def test_invalid_channel_raises() -> None:
    with pytest.raises(ValueError, match="channel must be one of"):
        tone_curve(_gradient_image(32), control_points=LINEAR_POINTS, channel="w")


def test_output_clamped() -> None:
    image = _gradient_image(128)

    output = tone_curve(image, control_points=OVERSHOOT_POINTS)

    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_device_cpu_cuda_parity(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    image = _color_image(128)
    expected = tone_curve(image, control_points=S_CURVE_MILD, channel="rgb_master", strength=0.8)
    actual = tone_curve(
        image.to(device=device_name),
        control_points=S_CURVE_MILD.to(device=device_name),
        channel="rgb_master",
        strength=0.8,
    )

    assert actual.device.type == device_name
    assert torch.allclose(actual.cpu(), expected, atol=1e-5)
