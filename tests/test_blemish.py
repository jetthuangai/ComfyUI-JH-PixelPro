"""Auto Blemish core (``core/blemish.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from core.blemish import auto_blemish, remove_blemishes, skin_likelihood

_PORTRAIT = Path(__file__).resolve().parents[1] / "workflows" / "sample_portrait.jpg"
_SKIN = torch.tensor([0.86, 0.66, 0.55])


def _skin_patch(seed: int = 0) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Textured skin with five red pimples; returns the image and the pimple centres."""
    gen = torch.Generator().manual_seed(seed)
    h, w = 200, 240
    pores = 0.02 * torch.randn(1, h, w, 1, generator=gen)
    image = (_SKIN + pores).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    centres = [(50, 60), (60, 170), (120, 110), (150, 50), (160, 190)]
    for cy, cx in centres:
        d = (((yy - cy) ** 2 + (xx - cx) ** 2).float().sqrt() / 5).clamp(0, 1)
        a = ((1 - d) ** 1.5)[None, ..., None]
        image = image * (1 - 0.35 * a) + a * torch.tensor([0.25, 0.0, 0.0])
    return image.clamp(0, 1), centres


def test_skin_likelihood_separates_skin() -> None:
    skin = _SKIN.view(1, 1, 1, 3).expand(1, 8, 8, 3)
    blue = torch.tensor([0.2, 0.4, 0.8]).view(1, 1, 1, 3).expand(1, 8, 8, 3)
    assert float(skin_likelihood(skin).min()) > 0.8
    assert float(skin_likelihood(blue).max()) < 0.2


def test_pimples_are_healed_and_texture_kept() -> None:
    image, centres = _skin_patch()
    area = torch.ones(1, *image.shape[1:3])
    out, spots = remove_blemishes(image, area, spot_size=16, sensitivity=0.5)
    for cy, cx in centres:
        assert float(spots[0, cy - 1 : cy + 2, cx - 1 : cx + 2].max()) > 0
        assert float(out[0, cy, cx, 0] - out[0, cy, cx, 1]) < 0.3  # no longer red
    region = spots[0] > 0
    std_out = out[0, ..., 1][region].std()
    std_ref = image[0, ..., 1][~region].std()
    assert std_out > 0.5 * std_ref  # pores continue through the healed spots
    assert torch.equal(out[0][~region], image[0][~region])


def test_protect_and_area_are_respected() -> None:
    image, centres = _skin_patch()
    area = torch.ones(1, *image.shape[1:3])
    protect = torch.zeros_like(area)
    protect[:, :100] = 1.0
    _, spots = remove_blemishes(image, area, protect, spot_size=16)
    assert not spots[:, :100].any()
    _, none = remove_blemishes(image, torch.zeros_like(area), spot_size=16)
    assert not none.any()


def test_user_mask_skips_face_detection() -> None:
    image, _ = _skin_patch()
    out, spots, count = auto_blemish(image, mask=torch.ones(1, *image.shape[1:3]))
    assert count == 0 and spots.any() and not torch.equal(out, image)


def test_no_face_is_identity() -> None:
    gen = torch.Generator().manual_seed(2)
    image = torch.rand(1, 120, 160, 3, generator=gen) * 0.2 + torch.tensor([0.1, 0.3, 0.6])
    try:
        out, spots, count = auto_blemish(image)
    except RuntimeError as exc:
        pytest.skip(f"YuNet weights unavailable: {exc}")
    assert count == 0 and torch.equal(out, image) and not spots.any()


def test_portrait_face_found_features_untouched() -> None:
    arr = np.asarray(Image.open(_PORTRAIT).convert("RGB").resize((800, 1200)))
    image = torch.from_numpy(arr.astype(np.float32) / 255.0)[None]
    try:
        out, spots, count = auto_blemish(image)
    except RuntimeError as exc:
        pytest.skip(f"YuNet weights unavailable: {exc}")
    assert count == 1
    changed = (out - image).abs().amax(-1) > 0
    assert torch.equal(changed, spots > 0)  # only healed pixels change
    assert float(spots.mean()) < 0.02  # a retouch, not a repaint


@pytest.mark.parametrize(("kwargs", "match"), [({"spot_scale": 5.0}, "spot_scale")])
def test_validates(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        auto_blemish(torch.rand(1, 16, 16, 3), **kwargs)
