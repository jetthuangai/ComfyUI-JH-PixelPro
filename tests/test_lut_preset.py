from __future__ import annotations

import torch

from core.lut import apply_lut_3d
from core.lut_preset import list_presets, load_preset

EXPECTED_PRESETS = [
    "cinematic-teal-orange",
    "cool-portrait",
    "high-contrast",
    "neutral-identity",
    "soft-pastel",
    "warm-portrait",
]


def _gradient(size: int = 24) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    xx = ramp.view(1, 1, size, 1)
    yy = ramp.view(1, size, 1, 1)
    return torch.cat(
        [
            xx.expand(1, size, size, 1),
            yy.expand(1, size, size, 1),
            ((xx + yy) / 2.0).expand(1, size, size, 1),
        ],
        dim=-1,
    )


def test_list_presets_contract() -> None:
    assert list_presets() == EXPECTED_PRESETS


def test_all_presets_parse_clean() -> None:
    for name in EXPECTED_PRESETS:
        parsed = load_preset(name)
        assert parsed["size"] == 17
        assert parsed["lut"].shape == (17, 17, 17, 3)
        assert torch.isfinite(parsed["lut"]).all()
        assert parsed["lut"].min().item() >= 0.0
        assert parsed["lut"].max().item() <= 1.0


def test_neutral_identity_preserves_gradient() -> None:
    image = _gradient()
    parsed = load_preset("neutral-identity")
    out = apply_lut_3d(
        image,
        parsed["lut"],
        domain_min=parsed["domain_min"],
        domain_max=parsed["domain_max"],
    )
    assert torch.allclose(out, image, atol=1e-6)


def test_warm_and_cool_presets_move_channels_opposite_directions() -> None:
    image = torch.full((1, 16, 16, 3), 0.5, dtype=torch.float32)
    warm = load_preset("warm-portrait")
    cool = load_preset("cool-portrait")
    warm_out = apply_lut_3d(image, warm["lut"])
    cool_out = apply_lut_3d(image, cool["lut"])

    assert warm_out[..., 0].mean().item() > warm_out[..., 2].mean().item()
    assert cool_out[..., 2].mean().item() > cool_out[..., 0].mean().item()


def test_intensity_blend_halfway() -> None:
    image = _gradient()
    parsed = load_preset("high-contrast")
    full = apply_lut_3d(image, parsed["lut"], strength=1.0)
    half = apply_lut_3d(image, parsed["lut"], strength=0.5)
    assert torch.allclose(half, torch.lerp(image, full, 0.5), atol=1e-5)
