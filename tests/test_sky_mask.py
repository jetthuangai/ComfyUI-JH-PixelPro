"""Sky Mask core (``core/sky_mask.py``)."""

from __future__ import annotations

import pytest
import torch

from core.sky_mask import sky_mask


def _scene(sky_colour: tuple[float, float, float], seed: int = 0) -> torch.Tensor:
    """Smooth sky over the top half, textured green ground below."""
    gen = torch.Generator().manual_seed(seed)
    h, w = 240, 320
    image = torch.empty(1, h, w, 3)
    rows = torch.linspace(0, 0.08, h // 2).view(-1, 1, 1)
    image[0, : h // 2] = torch.tensor(sky_colour) + rows
    ground = torch.tensor([0.25, 0.45, 0.2]) + 0.12 * torch.rand(h - h // 2, w, 3, generator=gen)
    image[0, h // 2 :] = ground
    return image.clamp(0, 1)


@pytest.mark.parametrize("colour", [(0.45, 0.65, 0.92), (0.82, 0.83, 0.85)])
def test_blue_and_overcast_sky(colour: tuple[float, float, float]) -> None:
    mask = sky_mask(_scene(colour))[0]
    assert float(mask[:100].mean()) > 0.9
    assert float(mask[140:].mean()) < 0.05


def test_blue_water_low_in_frame_is_not_sky() -> None:
    gen = torch.Generator().manual_seed(5)
    image = _scene((0.45, 0.65, 0.92))  # sky on top
    image[0, 60:120] = torch.tensor([0.3, 0.45, 0.2]) + 0.12 * torch.rand(60, 320, 3, generator=gen)
    image[0, 180:] = torch.tensor([0.2, 0.4, 0.7])  # lake below the trees
    mask = sky_mask(image.clamp(0, 1))[0]
    assert float(mask[:50].mean()) > 0.9
    assert float(mask[185:].mean()) < 0.05


def test_no_sky_and_skin_are_empty() -> None:
    gen = torch.Generator().manual_seed(3)
    texture = torch.rand(1, 120, 160, 3, generator=gen)
    assert float(sky_mask(texture).max()) < 0.05
    skin = torch.tensor([0.88, 0.7, 0.6]).view(1, 1, 1, 3).expand(1, 120, 160, 3).contiguous()
    assert float(sky_mask(skin).max()) < 0.05


def test_batch_and_soft_edges() -> None:
    batch = torch.cat([_scene((0.45, 0.65, 0.92)), _scene((0.82, 0.83, 0.85), seed=1)])
    mask = sky_mask(batch)
    assert mask.shape == (2, 240, 320)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    hard = sky_mask(batch, refine_edges=False)
    assert set(hard.unique().tolist()) <= {0.0, 1.0} or float(hard.std()) > 0


@pytest.mark.parametrize("bad", [-0.1, 1.5, True])
def test_validates(bad: float) -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        sky_mask(torch.rand(1, 16, 16, 3), sensitivity=bad)


def test_warm_cloud_in_blue_sky_is_kept_but_warm_wall_is_not() -> None:
    image = _scene((0.45, 0.65, 0.92))
    image[0, 30:70, 100:200] = torch.tensor([0.93, 0.92, 0.89])  # sunlit cloud, slightly warm
    image[0, 90:120, 0:120] = torch.tensor([0.95, 0.88, 0.78])  # warm white wall touching sky
    mask = sky_mask(image, refine_edges=False)[0]
    assert float(mask[40:60, 120:180].mean()) > 0.9
    assert float(mask[100:115, 10:100].mean()) < 0.1
