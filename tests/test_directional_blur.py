"""Motion / Zoom Blur core (``core/directional_blur.py``)."""

from __future__ import annotations

import pytest
import torch

from core.directional_blur import _split, directional_blur


def _dot(size: int = 81) -> torch.Tensor:
    image = torch.zeros(1, size, size, 3)
    image[:, size // 2, size // 2] = 1.0
    return image


def _spread(image: torch.Tensor) -> tuple[float, float]:
    """Standard deviation of the energy along x and y."""
    energy = image[0, ..., 0]
    ys, xs = torch.meshgrid(
        torch.arange(energy.shape[0], dtype=torch.float32),
        torch.arange(energy.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    total = energy.sum()
    mx, my = (energy * xs).sum() / total, (energy * ys).sum() / total
    sx = ((energy * (xs - mx) ** 2).sum() / total).sqrt()
    sy = ((energy * (ys - my) ** 2).sum() / total).sqrt()
    return float(sx), float(sy)


def test_split_covers_length() -> None:
    for length in (1.0, 7.5, 20.0, 333.0):
        n1, n2, step = _split(length)
        assert (n1 * n2 - 1) * step == pytest.approx(length)
        assert n1 + n2 <= 2 * (length + 2) ** 0.5 + 3


def test_zero_amount_is_identity() -> None:
    image = torch.rand(1, 24, 32, 3)
    assert torch.equal(directional_blur(image, distance=0.0), image)
    assert torch.equal(directional_blur(image, mode="zoom", zoom=0.0), image)


@pytest.mark.parametrize("mode", ["motion", "zoom"])
def test_flat_image_unchanged(mode: str) -> None:
    flat = torch.full((1, 40, 50, 3), 0.37)
    assert torch.allclose(directional_blur(flat, mode=mode, distance=15, zoom=0.3), flat, atol=1e-6)


@pytest.mark.parametrize(("angle", "axis"), [(0.0, "x"), (90.0, "y")])
def test_motion_follows_angle(angle: float, axis: str) -> None:
    out = directional_blur(_dot(), angle=angle, distance=20.0)
    sx, sy = _spread(out)
    along, across = (sx, sy) if axis == "x" else (sy, sx)
    assert along == pytest.approx(20.0 / 12**0.5, rel=0.15)  # box of length 20
    assert across < 1.0
    assert float(out.sum()) == pytest.approx(3.0, rel=0.02)  # energy kept


def test_motion_diagonal() -> None:
    sx, sy = _spread(directional_blur(_dot(), angle=45.0, distance=30.0))
    assert sx == pytest.approx(sy, rel=0.1)


def test_zoom_blurs_radially() -> None:
    image = torch.zeros(1, 101, 101, 3)
    image[:, 50, 85] = 1.0  # right of centre: zoom streak is horizontal
    sx, sy = _spread(directional_blur(image, mode="zoom", zoom=0.4))
    assert sx > 3.0 and sy < 1.0
    centre = torch.zeros(1, 101, 101, 3)
    centre[:, 50, 50] = 1.0  # the centre itself does not move
    still = directional_blur(centre, mode="zoom", zoom=0.4)
    assert float(still[0, 50, 50, 0]) == pytest.approx(1.0, abs=1e-5)
    assert max(_spread(still)) < 1.0


def test_mask_limits_blur() -> None:
    image = torch.rand(1, 40, 40, 3, generator=torch.Generator().manual_seed(2))
    mask = torch.zeros(1, 40, 40)
    mask[:, :, 20:] = 1.0
    out = directional_blur(image, distance=10.0, mask=mask)
    assert torch.equal(out[:, :, :20], image[:, :, :20])
    assert not torch.allclose(out[:, :, 20:], image[:, :, 20:])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "spin"}, "mode"),
        ({"distance": -1.0}, "distance"),
        ({"zoom": 0.95}, "zoom"),
        ({"center_x": 1.5}, "center_x"),
        ({"mask": torch.ones(1, 8, 8)}, "mask"),
    ],
)
def test_validates(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        directional_blur(torch.rand(1, 16, 16, 3), **kwargs)
