"""Auto Tone & White Balance core (``core/auto_tone.py``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from kornia.color import linear_rgb_to_rgb, rgb_to_linear_rgb

from core.auto_tone import auto_tone_params


def _scene(size: int = 192, seed: int = 0) -> torch.Tensor:
    """8x8 patches: mostly greys spanning the tonal range plus a few saturated colours."""
    gen = torch.Generator().manual_seed(seed)
    grey = torch.linspace(0.12, 0.92, 64)[torch.randperm(64, generator=gen)]
    patches = grey.view(1, 1, 8, 8).repeat(1, 3, 1, 1)
    colours = torch.tensor([[0.8, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.3, 0.8], [0.9, 0.7, 0.2]])
    for i, colour in enumerate(colours):
        patches[0, :, i * 2, 7 - i] = colour
    image = functional.interpolate(patches, size=(size, size), mode="nearest")
    noise = 0.01 * torch.randn(1, 1, size, size, generator=gen)
    return (image + noise).clamp(0, 1).permute(0, 2, 3, 1)


def _linear_gain(image: torch.Tensor, gains: tuple[float, float, float]) -> torch.Tensor:
    lin = rgb_to_linear_rgb(image.permute(0, 3, 1, 2))
    lin = (lin * torch.tensor(gains).view(1, 3, 1, 1)).clamp(0, 1)
    return linear_rgb_to_rgb(lin).permute(0, 2, 3, 1)


def test_balanced_photo_changes_little() -> None:
    params = auto_tone_params(_scene())[0]
    assert params["tone_engine"] == "local" and params["exposure_mode"] == "linear"
    assert abs(float(params.get("basic_exposure", 0.0))) <= 0.3
    assert abs(float(params.get("wb_temperature", 0.0))) < 3.0
    assert abs(float(params.get("wb_tint", 0.0))) < 3.0


def test_dark_photo_is_brightened() -> None:
    dark = _linear_gain(_scene(), (0.25, 0.25, 0.25))
    assert float(auto_tone_params(dark)[0]["basic_exposure"]) > 0.5


def test_warm_cast_is_cooled() -> None:
    warm = _linear_gain(_scene(), (1.15, 1.0, 0.85))
    params = auto_tone_params(warm)[0]
    assert float(params["wb_temperature"]) < -25.0
    cool = _linear_gain(_scene(), (0.85, 1.0, 1.15))
    assert float(auto_tone_params(cool)[0]["wb_temperature"]) > 25.0


def test_max_wb_shift_limits_and_disables() -> None:
    warm = _linear_gain(_scene(), (1.15, 1.0, 0.85))
    assert float(auto_tone_params(warm, max_wb_shift=10.0)[0]["wb_temperature"]) == -10.0
    assert "wb_temperature" not in auto_tone_params(warm, max_wb_shift=0.0)[0]
    assert "wb_temperature" not in auto_tone_params(warm, white_balance=False)[0]


def test_no_grey_areas_leaves_balance_alone() -> None:
    # A frame filled with skin tones has nothing near grey: no colour correction.
    skin = torch.tensor([0.85, 0.6, 0.48]) * torch.linspace(0.6, 1.0, 64).view(1, 64, 1, 1)
    skin = skin.expand(1, 64, 64, 3).contiguous()
    params = auto_tone_params(skin)[0]
    assert "wb_temperature" not in params and "wb_tint" not in params


def test_switches_and_bw() -> None:
    dark = _linear_gain(_scene(), (0.25, 0.25, 0.25))
    params = auto_tone_params(dark, exposure=False, tone=False, white_balance=False)[0]
    assert params == {"tone_engine": "local", "exposure_mode": "linear"}
    bw = auto_tone_params(_scene(), bw=True)[0]
    assert bw["gray_enable"] is True
    assert any(key.startswith("gray_") and key != "gray_enable" for key in bw)


def test_stable_across_image_size() -> None:
    small = _linear_gain(_scene(160), (0.5, 0.5, 0.45))
    large = functional.interpolate(small.permute(0, 3, 1, 2), scale_factor=4, mode="nearest")
    a = auto_tone_params(small)[0]
    b = auto_tone_params(large.permute(0, 2, 3, 1))[0]
    for key in ("basic_exposure", "wb_temperature"):
        assert abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) < 0.1 * max(
            1.0, abs(float(a.get(key, 0.0)))
        )


def test_batch_measures_each_image() -> None:
    batch = torch.cat([_scene(), _linear_gain(_scene(), (0.25, 0.25, 0.25))])
    first, second = auto_tone_params(batch)
    assert float(second["basic_exposure"]) > float(first.get("basic_exposure", 0.0)) + 0.5


@pytest.mark.parametrize("bad", [-1.0, 101.0, True])
def test_validates_max_wb_shift(bad: float) -> None:
    with pytest.raises(ValueError, match="max_wb_shift"):
        auto_tone_params(_scene(32), max_wb_shift=bad)


def test_white_background_is_not_darkened() -> None:
    # Product on white: 70 % near-white background, a mid-grey subject in the middle.
    image = torch.full((1, 100, 100, 3), 0.97)
    image[:, 25:75, 30:70] = 0.45
    assert float(auto_tone_params(image)[0].get("basic_exposure", 0.0)) == 0.0
    white = torch.full((1, 64, 64, 3), 1.0)
    assert "basic_exposure" not in auto_tone_params(white)[0]


def test_overexposed_photo_is_still_darkened() -> None:
    # Blown-out frame: mostly near-white, and the "subject" itself is far too bright too.
    gen = torch.Generator().manual_seed(3)
    bright = torch.full((1, 64, 64, 3), 0.97)
    bright[:, :, 45:] = 0.72 + 0.03 * torch.rand(1, 64, 19, 1, generator=gen)
    assert float(auto_tone_params(bright)[0]["basic_exposure"]) < 0.0
