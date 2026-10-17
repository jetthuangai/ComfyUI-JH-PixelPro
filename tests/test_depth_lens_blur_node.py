from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_depth_lens_blur"


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


@pytest.fixture(scope="module")
def node_cls():
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProDepthLensBlur"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProDepthLensBlur" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDepthLensBlur"] == "Depth: Lens Blur"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/depth"
    assert node_cls.RETURN_TYPES == ("IMAGE", "MASK")
    assert node_cls.RETURN_NAMES == ("image", "coc")
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(req) == [
        "image",
        "focus_depth",
        "depth_of_field",
        "max_blur",
        "bokeh",
        "layers",
        "blur_foreground",
    ]
    assert list(opt) == ["depth", "focus_mask", "keep_model_loaded"]
    assert opt["keep_model_loaded"][1]["default"] is True
    assert req["focus_depth"][1]["default"] == 0.5
    assert req["depth_of_field"][1]["default"] == 0.1
    assert (req["max_blur"][1]["default"], req["max_blur"][1]["max"]) == (24, 96)
    assert req["bokeh"][0] == ["disc", "gaussian"] and req["bokeh"][1]["default"] == "disc"
    assert (req["layers"][1]["min"], req["layers"][1]["max"]) == (2, 16)
    assert req["blur_foreground"][1]["default"] is True
    for name, value in {**req, **opt}.items():
        if name != "image":
            assert value[1]["tooltip"], name


def test_run_with_depth_and_focus_mask(node_cls) -> None:
    image = torch.rand(1, 32, 32, 3)
    depth = torch.linspace(0, 1, 32).view(1, 1, 32).expand(1, 32, 32).contiguous()
    mask = torch.zeros(1, 32, 32)
    mask[:, :, 27:30] = 1.0
    out, coc = node_cls().run(image, 0.0, 0.3, 6, "disc", 4, True, depth=depth, focus_mask=mask)
    assert out.shape == image.shape and coc.shape == (1, 32, 32)
    assert float(coc[0, 0, 28]) == pytest.approx(0.0, abs=1e-6)
    assert float(coc[0, 0, 0]) > 0.5


def test_run_without_depth_uses_estimator(node_cls, monkeypatch) -> None:
    image = torch.rand(1, 16, 16, 3)
    pack_module = sys.modules[f"{_PACK_MODULE_NAME}.core.depth_anything"]
    monkeypatch.setattr(
        pack_module, "estimate_depth", lambda image, *a, **k: torch.zeros(image.shape[:3])
    )
    out, _ = node_cls().run(image, 0.0, 0.1, 4, "gaussian", 3, True)
    assert torch.allclose(out, image, atol=1e-5)


def test_invalid_parameter_propagates(node_cls) -> None:
    with pytest.raises(ValueError, match="max_blur"):
        node_cls().run(
            torch.rand(1, 8, 8, 3), 0.5, 0.1, 0, "disc", 4, True, depth=torch.rand(1, 8, 8)
        )


def test_keep_model_loaded_off_unloads(node_cls, monkeypatch) -> None:
    module = sys.modules[f"{_PACK_MODULE_NAME}.nodes.depth_lens_blur_node"]
    calls = []
    monkeypatch.setattr(module.depth_anything, "unload", lambda: calls.append(1))
    monkeypatch.setattr(
        module.depth_anything, "estimate_depth", lambda image, *a, **k: torch.zeros(image.shape[:3])
    )
    image = torch.rand(1, 16, 16, 3)
    node_cls().run(image, 0.0, 0.1, 4, "gaussian", 3, True, keep_model_loaded=True)
    assert calls == []
    node_cls().run(image, 0.0, 0.1, 4, "gaussian", 3, True, keep_model_loaded=False)
    assert calls == [1]
    node_cls().run(
        image,
        0.0,
        0.1,
        4,
        "gaussian",
        3,
        True,
        depth=torch.zeros(1, 16, 16),
        keep_model_loaded=False,
    )
    assert calls == [1]  # external depth: no model was used
