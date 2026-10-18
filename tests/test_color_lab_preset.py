"""ColorLab params_json output + "Color: Apply ColorLab Preset" node."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from core.color_lab import params_to_json, scale_params_json

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_color_lab_preset"


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


def _image() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.rand(1, 24, 32, 3)


def _colorlab_params(**overrides: object) -> dict[str, object]:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    spec = node.INPUT_TYPES()
    params = {k: v[1]["default"] for k, v in spec["required"].items() if k != "image"}
    params.update({k: v[1]["default"] for k, v in spec["optional"].items() if "default" in v[1]})
    params.update(overrides)
    return params


LOOK = {
    "basic_exposure": 0.3,
    "basic_contrast": 15.0,
    "hsl_blue_sat": -25.0,
    "grade_shadow_hue": 210.0,
    "grade_shadow_sat": 30.0,
    "wb_temperature": 12.0,
}


def test_colorlab_outputs_preset_json() -> None:
    cls = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    assert cls.RETURN_TYPES == ("IMAGE", "STRING")
    assert cls.RETURN_NAMES == ("image", "params_json")
    _, preset = cls().apply(_image(), **_colorlab_params(**LOOK))
    assert json.loads(preset) == LOOK  # only non-default values


def test_preset_into_second_colorlab_reproduces_look() -> None:
    cls = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    image = _image()
    graded, preset = cls().apply(image, **_colorlab_params(**LOOK))
    again, preset2 = cls().apply(image, params_json=preset, **_colorlab_params())
    assert torch.equal(graded, again)
    assert json.loads(preset2) == LOOK


def test_second_colorlab_sliders_add_on_top() -> None:
    cls = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    _, preset = cls().apply(_image(), **_colorlab_params(**LOOK))
    _, combined = cls().apply(_image(), params_json=preset, **_colorlab_params(basic_contrast=5.0))
    assert json.loads(combined)["basic_contrast"] == 20.0


def test_switches_are_carried() -> None:
    text = params_to_json(
        {
            "tone_engine": "local",
            "gray_enable": True,
            "exposure_mode": "display",
            "basic_dehaze": 0.0,
        }
    )
    assert json.loads(text) == {"tone_engine": "local", "gray_enable": True}


def test_apply_node_matches_colorlab() -> None:
    pack = _load_pack()
    colorlab = pack.NODE_CLASS_MAPPINGS["JHPixelProColorLab"]()
    apply = pack.NODE_CLASS_MAPPINGS["JHPixelProColorLabApply"]()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorLabApply"] == (
        "Color: Apply ColorLab Preset"
    )
    image = _image()
    graded, preset = colorlab.apply(image, **_colorlab_params(**LOOK))
    applied, out_preset = apply.apply(image, preset, 1.0)
    assert torch.equal(graded, applied)
    assert json.loads(out_preset) == LOOK


def test_apply_strength_scales_amounts_but_not_grading_hue() -> None:
    half = json.loads(scale_params_json(json.dumps(LOOK), 0.5))
    assert half["basic_exposure"] == pytest.approx(0.15)
    assert half["grade_shadow_sat"] == pytest.approx(15.0)
    assert half["grade_shadow_hue"] == 210.0
    assert json.loads(scale_params_json(json.dumps(LOOK), 0.0)) == {"grade_shadow_hue": 210.0}


def test_apply_strength_zero_is_identity_and_mask_limits() -> None:
    apply = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLabApply"]()
    image = _image()
    out, _ = apply.apply(image, json.dumps(LOOK), 0.0)
    assert torch.equal(out, image)
    mask = torch.zeros(1, 24, 32)
    mask[:, :12] = 1.0
    out, _ = apply.apply(image, json.dumps(LOOK), 1.0, mask=mask)
    assert torch.allclose(out[:, 12:], image[:, 12:])
    assert not torch.allclose(out[:, :12], image[:, :12])


def test_apply_rejects_bad_json() -> None:
    apply = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProColorLabApply"]()
    with pytest.raises(ValueError):
        apply.apply(_image(), "{oops", 1.0)
