"""Regression tests for batch-7 JSON-driven Look preset wrapper nodes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


def _node(name: str):
    pack = _load_pack()
    return pack.NODE_CLASS_MAPPINGS[name]()


def _mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean(torch.abs(a - b)).item()


def _channel_spread(image: torch.Tensor) -> float:
    return torch.mean(image.max(dim=-1).values - image.min(dim=-1).values).item()


def test_look_nodes_registered_and_category() -> None:
    pack = _load_pack()
    expected = {
        "JHPixelProLookCinematicTealOrange": "Look: Cinematic Teal/Orange",
        "JHPixelProLookWarmSkinTone": "Look: Warm Skin Tone",
        "JHPixelProLookMoodyGreen": "Look: Moody Green",
        "JHPixelProLookFadedFilm": "Look: Faded Film",
        "JHPixelProLookGoldenHour": "Look: Golden Hour",
        "JHPixelProLookDesaturatedPop": "Look: Desaturated Pop",
    }

    for class_name, display_name in expected.items():
        assert pack.NODE_CLASS_MAPPINGS[class_name].CATEGORY == "ComfyUI-JH-PixelPro/looks"
        assert pack.NODE_DISPLAY_NAME_MAPPINGS[class_name] == display_name


def test_preset_json_schema_version_one_loads_all() -> None:
    for path in (_PACK_ROOT / "presets").glob("*.json"):
        preset = json.loads(path.read_text(encoding="utf-8"))
        assert preset["schema_version"] == 1
        assert preset["id"] == path.stem
        assert preset["compose_ops"]


def test_look_input_contract_exposes_intensity_slider_and_protect_skin() -> None:
    spec = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProLookGoldenHour"].INPUT_TYPES()
    assert spec["required"]["image"] == ("IMAGE",)
    intensity_type, intensity_meta = spec["required"]["intensity"]
    assert intensity_type == "FLOAT"
    assert intensity_meta["default"] == 1.0
    assert intensity_meta["min"] == 0.0
    assert intensity_meta["max"] == 1.0
    assert intensity_meta["display"] == "slider"
    assert spec["optional"]["protect_skin"][0] == "BOOLEAN"


def test_cinematic_teal_orange_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookCinematicTealOrange").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_warm_skin_tone_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookWarmSkinTone").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_moody_green_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookMoodyGreen").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_faded_film_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookFadedFilm").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_golden_hour_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookGoldenHour").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_desaturated_pop_identity_at_zero_intensity() -> None:
    image = _gradient()
    (out,) = _node("JHPixelProLookDesaturatedPop").apply(image, intensity=0.0)
    assert torch.equal(out, image)


def test_cinematic_teal_orange_shifts_dark_pixels_cooler() -> None:
    image = _solid((0.18, 0.18, 0.18))
    (out,) = _node("JHPixelProLookCinematicTealOrange").apply(image, intensity=1.0)
    assert out[..., 1].mean().item() > out[..., 0].mean().item()
    assert out[..., 2].mean().item() > out[..., 0].mean().item()


def test_warm_skin_tone_changes_orange_band() -> None:
    image = _solid((0.78, 0.44, 0.28))
    (out,) = _node("JHPixelProLookWarmSkinTone").apply(image, intensity=1.0)
    assert _mean_abs_delta(out, image) > 0.01


def test_moody_green_pushes_shadows_toward_green() -> None:
    image = _solid((0.22, 0.22, 0.22))
    (out,) = _node("JHPixelProLookMoodyGreen").apply(image, intensity=1.0)
    assert out[..., 1].mean().item() > out[..., 0].mean().item()


def test_faded_film_lifts_black_floor() -> None:
    image = _solid((0.03, 0.03, 0.03))
    (out,) = _node("JHPixelProLookFadedFilm").apply(image, intensity=1.0)
    assert out.mean().item() > image.mean().item() + 0.03


def test_golden_hour_warms_neutral_image() -> None:
    image = _solid((0.45, 0.45, 0.45))
    (out,) = _node("JHPixelProLookGoldenHour").apply(image, intensity=1.0)
    assert out[..., 0].mean().item() > out[..., 2].mean().item()


def test_desaturated_pop_reduces_blue_saturation() -> None:
    image = _solid((0.05, 0.25, 0.95))
    (out,) = _node("JHPixelProLookDesaturatedPop").apply(image, intensity=1.0)
    assert _channel_spread(out) < _channel_spread(image) * 0.8


def test_intensity_half_matches_alpha_blend() -> None:
    image = _gradient()
    node = _node("JHPixelProLookGoldenHour")
    (full,) = node.apply(image, intensity=1.0)
    (half,) = node.apply(image, intensity=0.5)
    expected = torch.lerp(image, full, 0.5)
    assert torch.allclose(half, expected, atol=1e-5)


def test_warm_skin_tone_protect_skin_reduces_skin_effect() -> None:
    image = _solid((0.82, 0.5, 0.36))
    node = _node("JHPixelProLookWarmSkinTone")
    (unprotected,) = node.apply(image, intensity=1.0, protect_skin=False)
    (protected,) = node.apply(image, intensity=1.0, protect_skin=True)
    assert _mean_abs_delta(protected, image) <= _mean_abs_delta(unprotected, image) * 0.5


def test_desaturated_pop_protect_skin_reduces_skin_effect() -> None:
    image = _solid((0.82, 0.5, 0.36))
    node = _node("JHPixelProLookDesaturatedPop")
    (unprotected,) = node.apply(image, intensity=1.0, protect_skin=False)
    (protected,) = node.apply(image, intensity=1.0, protect_skin=True)
    assert _mean_abs_delta(protected, image) <= _mean_abs_delta(unprotected, image) * 0.5
