"""ColorLab v1.7.0 additions: legacy golden, tone_engine=local, white balance, params_json."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest
import torch
from kornia.color import rgb_to_linear_rgb

from core.color_lab import JSON_ONLY_KEYS, apply_colorlab_pipeline, merge_params_json, param_keys
from core.local_laplacian import shadows_highlights_camera_raw

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(_FIXTURES))
from make_color_lab_golden import CASES, GOLDEN, golden_image  # noqa: E402

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_color_lab_v170"


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


def _linear_luminance(image: torch.Tensor) -> torch.Tensor:
    linear = rgb_to_linear_rgb(image.permute(0, 3, 1, 2))
    return (linear * torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(1)


def _textured(size: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    base = torch.linspace(0.2, 0.8, size).view(1, 1, size, 1).expand(1, size, size, 3)
    return (base + 0.05 * torch.randn(1, size, size, 1)).clamp(0.0, 1.0).contiguous()


# --- legacy path is frozen at v1.6.0 -------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_legacy_matches_v160_golden(case: str) -> None:
    golden = torch.load(GOLDEN)
    out = apply_colorlab_pipeline(golden_image(), CASES[case])
    assert torch.allclose(out, golden[case], atol=1e-6), (out - golden[case]).abs().max()


@pytest.mark.parametrize("case", sorted(CASES))
def test_new_defaults_are_exactly_legacy(case: str) -> None:
    image = golden_image()
    explicit = {**CASES[case], "tone_engine": "legacy", "wb_temperature": 0.0, "wb_tint": 0.0}
    explicit = merge_params_json(explicit, "")
    assert torch.equal(
        apply_colorlab_pipeline(image, explicit), apply_colorlab_pipeline(image, CASES[case])
    )


# --- tone_engine = local -------------------------------------------------------------


def test_local_engine_identity_at_zero() -> None:
    image = golden_image()
    assert torch.equal(apply_colorlab_pipeline(image, {"tone_engine": "local"}), image)


def test_local_highlights_shadows_use_local_laplacian() -> None:
    image = _textured()
    params = {"tone_engine": "local", "basic_shadows": 40.0, "basic_highlights": -50.0}
    out = apply_colorlab_pipeline(image, params)
    expected = shadows_highlights_camera_raw(image, 40.0, -50.0)
    assert torch.allclose(out, expected, atol=1e-5)
    legacy = apply_colorlab_pipeline(image, {**params, "tone_engine": "legacy"})
    assert not torch.allclose(out, legacy, atol=1e-3)


def test_local_clarity_boosts_texture_and_keeps_flat() -> None:
    image = _textured()
    out = apply_colorlab_pipeline(image, {"tone_engine": "local", "basic_clarity": 60.0})
    detail = lambda x: (x[:, :, 1:] - x[:, :, :-1]).abs().mean()  # noqa: E731
    assert detail(out) > detail(image) * 1.05
    flat = torch.full((1, 32, 32, 3), 0.4)
    assert (
        apply_colorlab_pipeline(flat, {"tone_engine": "local", "basic_clarity": 80.0}) - flat
    ).abs().max() < 1e-4


def _hazy() -> torch.Tensor:
    torch.manual_seed(1)
    scene = torch.rand(1, 48, 64, 3) * 0.6
    scene[:, :, 32:] *= torch.tensor([0.3, 0.6, 1.0])
    t = 0.45
    return (scene * t + 0.85 * (1.0 - t)).clamp(0.0, 1.0)


def test_local_dehaze_removes_veil() -> None:
    hazy = _hazy()
    out = apply_colorlab_pipeline(hazy, {"tone_engine": "local", "basic_dehaze": 70.0})
    assert out.amin(-1).mean() < hazy.amin(-1).mean() - 0.05  # veil (dark channel) drops
    assert out.std() > hazy.std() * 1.2  # contrast returns
    flat = torch.full((1, 24, 24, 3), 0.6)
    flat_out = apply_colorlab_pipeline(flat, {"tone_engine": "local", "basic_dehaze": 70.0})
    assert (flat_out - flat).abs().max() < 1e-4


def test_local_negative_dehaze_adds_veil() -> None:
    image = _textured()
    out = apply_colorlab_pipeline(image, {"tone_engine": "local", "basic_dehaze": -60.0})
    assert out.std() < image.std()


def test_unknown_tone_engine_raises() -> None:
    with pytest.raises(ValueError):
        apply_colorlab_pipeline(golden_image(), {"tone_engine": "fancy", "basic_contrast": 5.0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_local_engine_returns_on_input_device() -> None:
    image = _textured()
    out = apply_colorlab_pipeline(image, {"tone_engine": "local", "basic_shadows": 30.0})
    assert out.device.type == "cpu"


# --- white balance -------------------------------------------------------------------


def test_temperature_warms_and_keeps_luminance() -> None:
    grey = torch.full((1, 8, 8, 3), 0.4)
    warm = apply_colorlab_pipeline(grey, {"wb_temperature": 40.0})
    assert warm[0, 0, 0, 0] > warm[0, 0, 0, 2]
    assert torch.allclose(_linear_luminance(warm), _linear_luminance(grey), atol=2e-3)
    cool = apply_colorlab_pipeline(grey, {"wb_temperature": -40.0})
    assert cool[0, 0, 0, 2] > cool[0, 0, 0, 0]


def test_tint_moves_green() -> None:
    grey = torch.full((1, 8, 8, 3), 0.4)
    magenta = apply_colorlab_pipeline(grey, {"wb_tint": 50.0})
    assert magenta[0, 0, 0, 1] < magenta[0, 0, 0, 0]
    green = apply_colorlab_pipeline(grey, {"wb_tint": -50.0})
    assert green[0, 0, 0, 1] > green[0, 0, 0, 0]


# --- params_json ---------------------------------------------------------------------


def test_params_json_adds_numbers_and_sets_switches() -> None:
    merged = merge_params_json(
        {"basic_shadows": 10.0, "tone_engine": "legacy"},
        json.dumps(
            {
                "basic_shadows": 25,
                "basic_exposure": 0.3,
                "tone_engine": "local",
                "gray_enable": True,
            }
        ),
    )
    assert merged["basic_shadows"] == 35.0
    assert merged["basic_exposure"] == pytest.approx(0.3)
    assert merged["tone_engine"] == "local"
    assert merged["gray_enable"] is True


def test_params_json_ignores_unknown_keys(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        merged = merge_params_json({}, '{"basic_contrast": 5, "vignette": 30}')
    assert merged == {"basic_contrast": 5.0}
    assert "vignette" in caplog.text


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '{"basic_contrast": [1]}'])
def test_params_json_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        merge_params_json({}, bad)


def test_param_keys_cover_node_inputs() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    inputs = node.INPUT_TYPES()
    names = set(inputs["required"]) | set(inputs["optional"])
    assert names - {"image", "params_json"} == set(param_keys()) - set(JSON_ONLY_KEYS)


def test_node_keeps_optional_widget_order_and_defaults() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    optional = node.INPUT_TYPES()["optional"]
    assert list(optional) == [
        "exposure_mode",
        "tone_engine",
        "wb_temperature",
        "wb_tint",
        "params_json",
    ]
    assert optional["tone_engine"][1]["default"] == "legacy"
    assert optional["params_json"][1]["forceInput"] is True


def test_node_applies_params_json_on_top_of_sliders() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]()
    required = node.INPUT_TYPES()["required"]
    params = {k: v[1]["default"] for k, v in required.items() if k != "image"}
    image = golden_image()
    via_json, _ = node.apply(
        image, params_json='{"basic_exposure": 0.5}', **{**params, "basic_exposure": 0.25}
    )
    direct, _ = node.apply(image, **{**params, "basic_exposure": 0.75})
    assert torch.equal(via_json, direct)
