"""Regression tests for the batch-8 single-node Look preset dropdown."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test"


def _load_pack():
    if _PACK_MODULE_NAME in sys.modules:
        return sys.modules[_PACK_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PACK_MODULE_NAME,
        _PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(_PACK_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACK_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _look_select_module():
    _load_pack()
    return importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.look_select")


def _preset_options() -> list[str]:
    return list(_look_select_module().PRESET_OPTIONS)


def _solid(color: tuple[float, float, float], size: int = 16) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(1, size, size, 3)


def _gradient(size: int = 16) -> torch.Tensor:
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


def _node():
    pack = _load_pack()
    return pack.NODE_CLASS_MAPPINGS["JHPixelProLookSelect"]()


def _apply(preset: str, image: torch.Tensor, intensity: float = 1.0, protect_skin: bool = False):
    return _node().apply(image, preset, intensity, protect_skin)


def _mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def _channel_spread(image: torch.Tensor) -> float:
    return torch.mean(image.max(dim=-1).values - image.min(dim=-1).values).item()


def test_look_select_registered_and_category() -> None:
    pack = _load_pack()
    cls = pack.NODE_CLASS_MAPPINGS["JHPixelProLookSelect"]

    assert cls.CATEGORY == "ComfyUI-JH-PixelPro/looks"
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLookSelect"] == "Look: Select Preset"


def test_preset_options_contract() -> None:
    assert _preset_options() == [
        "cinematic-teal-orange",
        "warm-skin-tone",
        "moody-green",
        "faded-film",
        "golden-hour",
        "desaturated-pop",
    ]


def test_preset_json_schema_version_one_loads_all() -> None:
    for path in (_PACK_ROOT / "presets").glob("*.json"):
        preset = json.loads(path.read_text(encoding="utf-8"))
        assert preset["schema_version"] == 1
        assert preset["id"] == path.stem
        assert preset["compose_ops"]


def test_look_select_input_contract_exposes_preset_dropdown_intensity_and_protect_skin() -> None:
    spec = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProLookSelect"].INPUT_TYPES()
    assert spec["required"]["image"] == ("IMAGE",)

    preset_options, preset_meta = spec["required"]["preset"]
    assert preset_options == _preset_options()
    assert preset_meta["default"] == "cinematic-teal-orange"

    intensity_type, intensity_meta = spec["required"]["intensity"]
    assert intensity_type == "FLOAT"
    assert intensity_meta["default"] == 0.7
    assert intensity_meta["min"] == 0.0
    assert intensity_meta["max"] == 1.0
    assert intensity_meta["step"] == 0.01
    assert spec["required"]["protect_skin"] == ("BOOLEAN", {"default": False})


@pytest.mark.parametrize("preset", _preset_options())
def test_identity_at_zero_intensity(preset: str) -> None:
    image = _gradient()
    (out,) = _apply(preset, image, intensity=0.0, protect_skin=False)
    assert torch.equal(out, image)


@pytest.mark.parametrize(
    ("preset", "image", "assertion"),
    [
        (
            "cinematic-teal-orange",
            _solid((0.18, 0.18, 0.18)),
            lambda out, image: (
                out[..., 1].mean().item() > out[..., 0].mean().item()
                and out[..., 2].mean().item() > out[..., 0].mean().item()
            ),
        ),
        (
            "warm-skin-tone",
            _solid((0.78, 0.44, 0.28)),
            lambda out, image: _mean_abs_delta(out, image) > 0.01,
        ),
        (
            "moody-green",
            _solid((0.22, 0.22, 0.22)),
            lambda out, image: out[..., 1].mean().item() > out[..., 0].mean().item(),
        ),
        (
            "faded-film",
            _solid((0.03, 0.03, 0.03)),
            lambda out, image: out.mean().item() > image.mean().item() + 0.03,
        ),
        (
            "golden-hour",
            _solid((0.45, 0.45, 0.45)),
            lambda out, image: out[..., 0].mean().item() > out[..., 2].mean().item(),
        ),
        (
            "desaturated-pop",
            _solid((0.05, 0.25, 0.95)),
            lambda out, image: _channel_spread(out) < _channel_spread(image) * 0.8,
        ),
    ],
)
def test_effect_verification_at_full_intensity(preset, image, assertion) -> None:
    (out,) = _apply(preset, image, intensity=1.0, protect_skin=False)
    assert assertion(out, image)


@pytest.mark.parametrize("preset", ["warm-skin-tone", "desaturated-pop"])
def test_protect_skin_reduces_skin_effect(preset: str) -> None:
    image = _solid((0.82, 0.5, 0.36))
    (unprotected,) = _apply(preset, image, intensity=1.0, protect_skin=False)
    (protected,) = _apply(preset, image, intensity=1.0, protect_skin=True)
    assert _mean_abs_delta(protected, image) <= _mean_abs_delta(unprotected, image) * 0.5


def test_intensity_half_matches_alpha_blend() -> None:
    image = _gradient()
    (full,) = _apply("golden-hour", image, intensity=1.0, protect_skin=False)
    (half,) = _apply("golden-hour", image, intensity=0.5, protect_skin=False)
    expected = torch.lerp(image, full, 0.5)
    assert torch.allclose(half, expected, atol=1e-5)


def test_intensity_one_matches_full_apply_shape_and_range() -> None:
    image = _gradient()
    (out,) = _apply("cinematic-teal-orange", image, intensity=1.0, protect_skin=False)
    assert out.shape == image.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_protect_skin_noop_on_non_skin_blue_image() -> None:
    image = _solid((0.05, 0.25, 0.95))
    (unprotected,) = _apply("golden-hour", image, intensity=1.0, protect_skin=False)
    (protected,) = _apply("golden-hour", image, intensity=1.0, protect_skin=True)
    assert torch.allclose(protected, unprotected, atol=1e-5)


def test_dropdown_invalid_preset_raises() -> None:
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        _apply("nonexistent-preset", _gradient(), intensity=0.5, protect_skin=False)
