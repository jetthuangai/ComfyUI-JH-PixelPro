from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_lr_preset"
XMP = _PACK_ROOT / "tests" / "fixtures" / "lr" / "warm-film.xmp"


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


def _run(node, *args, **kwargs):
    return node.apply(*args, **kwargs)["result"]


@pytest.fixture(scope="module")
def node_cls():
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProLightroomPreset"]


def _image() -> torch.Tensor:
    torch.manual_seed(5)
    return torch.rand(1, 32, 48, 3)


def test_registered_and_simple(node_cls) -> None:
    pack = _load_pack()
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLightroomPreset"]
        == "Color: Lightroom Preset (XMP)"
    )
    assert node_cls.RETURN_NAMES == ("image", "params_json", "report")
    spec = node_cls.INPUT_TYPES()
    assert list(spec["required"]) == ["image", "preset", "strength"]
    assert spec["optional"]["tone_engine"][1]["default"] == "local"


def test_apply_from_path(node_cls) -> None:
    image = _image()
    out, params_json, report = _run(
        node_cls(), image, "(no presets found)", 1.0, preset_path=str(XMP)
    )
    assert out.shape == image.shape and not torch.allclose(out, image)
    params = json.loads(params_json)
    assert params["tone_engine"] == "local"
    assert params["basic_exposure"] == pytest.approx(0.35)
    assert params["tone_curve"][0] == [0.0, 18.0]
    assert "JH Warm Film" in report and "Vintage 06" in report


def test_params_json_reproduces_image_in_colorlab_and_apply(node_cls) -> None:
    pack = _load_pack()
    image = _image()
    out, params_json, _ = _run(node_cls(), image, "x", 1.0, preset_path=str(XMP))
    apply = pack.NODE_CLASS_MAPPINGS["JHPixelProColorLabApply"]()
    again, _ = apply.apply(image, params_json, 1.0)
    assert torch.allclose(out, again, atol=1e-6)
    colorlab = pack.NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    spec = colorlab.INPUT_TYPES()
    sliders = {k: v[1]["default"] for k, v in spec["required"].items() if k != "image"}
    sliders.update({k: v[1]["default"] for k, v in spec["optional"].items() if "default" in v[1]})
    via_colorlab, chained = colorlab().apply(image, params_json=params_json, **sliders)
    assert torch.allclose(out, via_colorlab, atol=1e-6)
    assert json.loads(chained)["tone_curve"] == json.loads(params_json)["tone_curve"]


def test_strength_and_mask(node_cls) -> None:
    image = _image()
    same, half_json, _ = _run(node_cls(), image, "x", 0.0, preset_path=str(XMP))
    assert torch.equal(same, image)
    _, half_json, _ = _run(node_cls(), image, "x", 0.5, preset_path=str(XMP))
    assert json.loads(half_json)["basic_exposure"] == pytest.approx(0.175)
    mask = torch.zeros(1, 32, 48)
    mask[:, :, :24] = 1.0
    out, _, _ = _run(node_cls(), image, "x", 1.0, preset_path=str(XMP), mask=mask)
    assert torch.allclose(out[:, :, 24:], image[:, :, 24:])


def test_missing_preset_explains_where(node_cls) -> None:
    with pytest.raises(ValueError, match="lightroom_presets"):
        _run(node_cls(), _image(), "(no presets found)", 1.0)


def test_is_changed_tracks_file(node_cls) -> None:
    key = node_cls.IS_CHANGED("x", preset_path=str(XMP))
    assert str(XMP) in key and key.split(":")[-1].isdigit()
