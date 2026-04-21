"""Regression tests for Batch-9 Photoshop-style blend modes."""

from __future__ import annotations

import pytest
import torch

from core.blend_modes import (
    BLEND_MODES,
    apply_blend,
    blend_dissolve,
    compose_stack,
)


def _solid(color: tuple[float, float, float], size: int = 4) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(1, size, size, 3)


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_blend_mode_shape_range_and_determinism(mode: str) -> None:
    base = _solid((0.25, 0.5, 0.75))
    blend = _solid((0.8, 0.35, 0.2))
    out_a = apply_blend(mode, base, blend)
    out_b = apply_blend(mode, base, blend)
    assert out_a.shape == base.shape
    assert out_a.min().item() >= 0.0
    assert out_a.max().item() <= 1.0
    assert torch.allclose(out_a, out_b)


def test_blend_mode_list_has_27_photoshop_modes() -> None:
    assert len(BLEND_MODES) == 27
    assert BLEND_MODES[:7] == [
        "normal",
        "dissolve",
        "darken",
        "multiply",
        "color_burn",
        "linear_burn",
        "darker_color",
    ]
    assert BLEND_MODES[-4:] == ["hue", "saturation", "color", "luminosity"]


def test_normal_returns_blend() -> None:
    base = _solid((0.1, 0.2, 0.3))
    blend = _solid((0.7, 0.6, 0.5))
    assert torch.equal(apply_blend("normal", base, blend), blend)


def test_multiply_fixture() -> None:
    out = apply_blend("multiply", _solid((0.5, 0.5, 0.5)), _solid((0.5, 0.25, 0.75)))
    assert torch.allclose(out, _solid((0.25, 0.125, 0.375)))


def test_screen_fixture() -> None:
    out = apply_blend("screen", _solid((0.5, 0.5, 0.5)), _solid((0.5, 0.25, 0.75)))
    assert torch.allclose(out, _solid((0.75, 0.625, 0.875)))


def test_overlay_fixture_midtones() -> None:
    out = apply_blend("overlay", _solid((0.25, 0.5, 0.75)), _solid((0.5, 0.5, 0.5)))
    assert torch.allclose(out, _solid((0.25, 0.5, 0.75)))


def test_difference_fixture() -> None:
    out = apply_blend("difference", _solid((0.2, 0.5, 0.8)), _solid((0.7, 0.3, 0.4)))
    assert torch.allclose(out, _solid((0.5, 0.2, 0.4)))


def test_exclusion_fixture() -> None:
    out = apply_blend("exclusion", _solid((0.5, 0.5, 0.5)), _solid((0.25, 0.5, 0.75)))
    assert torch.allclose(out, _solid((0.5, 0.5, 0.5)))


def test_linear_burn_fixture_clamps() -> None:
    out = apply_blend("linear_burn", _solid((0.2, 0.6, 0.9)), _solid((0.2, 0.6, 0.4)))
    assert torch.allclose(out, _solid((0.0, 0.2, 0.3)), atol=1e-6)


def test_linear_dodge_fixture_clamps() -> None:
    out = apply_blend("linear_dodge", _solid((0.2, 0.6, 0.9)), _solid((0.2, 0.6, 0.4)))
    assert torch.allclose(out, _solid((0.4, 1.0, 1.0)), atol=1e-6)


def test_hue_component_uses_blend_hue() -> None:
    base = _solid((0.05, 0.05, 0.9))
    blend = _solid((0.9, 0.05, 0.05))
    out = apply_blend("hue", base, blend)
    assert out[..., 0].mean().item() > out[..., 2].mean().item()


def test_color_component_uses_blend_chroma() -> None:
    base = _solid((0.4, 0.4, 0.4))
    blend = _solid((0.9, 0.1, 0.1))
    out = apply_blend("color", base, blend)
    assert out[..., 0].mean().item() > out[..., 1].mean().item()


def test_luminosity_component_uses_blend_value() -> None:
    base = _solid((0.1, 0.2, 0.8))
    blend = _solid((0.9, 0.9, 0.9))
    out = apply_blend("luminosity", base, blend)
    assert out.max().item() > base.max().item()


def test_dissolve_stochastic_reproducible() -> None:
    base = _solid((0.0, 0.0, 0.0), size=12)
    blend = _solid((1.0, 1.0, 1.0), size=12)
    out_a = blend_dissolve(base, blend, opacity=0.5, seed=123)
    out_b = blend_dissolve(base, blend, opacity=0.5, seed=123)
    assert torch.equal(out_a, out_b)
    assert 0.25 < out_a.mean().item() < 0.75


def test_compose_stack_normal_lerp_opacity() -> None:
    base = _solid((0.0, 0.0, 0.0))
    layer = _solid((1.0, 1.0, 1.0))
    stack = [
        {"image": base, "mask": None},
        {"image": layer, "blend_mode": "normal", "opacity": 0.5, "fill": 1.0, "mask": None},
    ]
    out = compose_stack(stack)
    assert torch.allclose(out, _solid((0.5, 0.5, 0.5)))


def test_compose_stack_mask_limits_layer() -> None:
    base = _solid((0.0, 0.0, 0.0), size=4)
    layer = _solid((1.0, 1.0, 1.0), size=4)
    mask = torch.zeros(1, 4, 4)
    mask[:, :, 2:] = 1.0
    stack = [
        {"image": base, "mask": None},
        {"image": layer, "blend_mode": "normal", "opacity": 1.0, "fill": 1.0, "mask": mask},
    ]
    out = compose_stack(stack)
    assert out[:, :, :2].mean().item() == 0.0
    assert out[:, :, 2:].mean().item() == 1.0


def test_apply_blend_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown blend mode"):
        apply_blend("not-a-mode", _solid((0.0, 0.0, 0.0)), _solid((1.0, 1.0, 1.0)))


def test_compose_stack_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty stack"):
        compose_stack([])
