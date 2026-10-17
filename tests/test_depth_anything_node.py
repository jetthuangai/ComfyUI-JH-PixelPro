from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_depth_anything"


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
    return _load_pack().NODE_CLASS_MAPPINGS["JHPixelProDepthAnything"]


def test_node_registered() -> None:
    pack = _load_pack()
    assert "JHPixelProDepthAnything" in pack.NODE_CLASS_MAPPINGS
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDepthAnything"] == "Depth: Depth Anything V2"


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/depth"
    assert node_cls.RETURN_TYPES == ("MASK", "IMAGE")
    assert node_cls.RETURN_NAMES == ("depth", "depth_preview")
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    required = node_cls.INPUT_TYPES()["required"]
    assert list(required) == ["image", "model", "resolution"]
    choices, opts = required["model"]
    assert choices == ["small", "base", "large"] and opts["default"] == "small"
    assert "NON-COMMERCIAL" in opts["tooltip"]
    _, res = required["resolution"]
    assert (res["default"], res["min"], res["max"], res["step"]) == (518, 266, 1036, 14)
    assert res["tooltip"]


def test_run_wraps_core(node_cls, monkeypatch) -> None:
    import sys as _sys

    module = _sys.modules[f"{_PACK_MODULE_NAME}.nodes.depth_anything_node"]
    seen = {}

    def fake(image, model="small", resolution=518):
        seen.update(model=model, resolution=resolution)
        return torch.full(image.shape[:3], 0.25)

    monkeypatch.setattr(module, "estimate_depth", fake)
    depth, preview = node_cls().run(torch.rand(2, 20, 30, 3), "base", 266)
    assert seen == {"model": "base", "resolution": 266}
    assert depth.shape == (2, 20, 30)
    assert preview.shape == (2, 20, 30, 3)
    assert torch.allclose(preview[..., 1], depth)


def test_keep_model_loaded_toggle(node_cls, monkeypatch) -> None:
    module = sys.modules[f"{_PACK_MODULE_NAME}.nodes.depth_anything_node"]
    calls = []
    monkeypatch.setattr(module, "estimate_depth", lambda image, **k: torch.zeros(image.shape[:3]))
    monkeypatch.setattr(module.depth_anything, "unload", lambda: calls.append(1))
    opt = node_cls.INPUT_TYPES()["optional"]["keep_model_loaded"]
    assert opt[0] == "BOOLEAN" and opt[1]["default"] is True and opt[1]["tooltip"]
    node_cls().run(torch.rand(1, 8, 8, 3), "small", 518)
    assert calls == []
    node_cls().run(torch.rand(1, 8, 8, 3), "small", 518, keep_model_loaded=False)
    assert calls == [1]
