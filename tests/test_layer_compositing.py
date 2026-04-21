"""Regression tests for Batch-9 LAYER_STACK compositing wrappers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from core.blend_modes import BLEND_MODES

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


def _cls(name: str):
    return _load_pack().NODE_CLASS_MAPPINGS[name]


def _solid(color: tuple[float, float, float], size: int = 8) -> torch.Tensor:
    return torch.tensor(color, dtype=torch.float32).view(1, 1, 1, 3).expand(1, size, size, 3)


def test_colorlab_registered_and_has_55_params() -> None:
    pack = _load_pack()
    cls = pack.NODE_CLASS_MAPPINGS["JHPixelProColorLab"]
    spec = cls.INPUT_TYPES()["required"]
    assert cls.CATEGORY == "ComfyUI-JH-PixelPro/color"
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorLab"] == "Color: ColorLab (ACR)"
    assert len(spec) == 56
    assert spec["basic_exposure"][1]["min"] == -5.0
    assert spec["gray_enable"] == ("BOOLEAN", {"default": False})


@pytest.mark.parametrize(
    "name",
    [
        "JHPixelProLayerStackStart",
        "JHPixelProLayerAdd",
        "JHPixelProLayerGroup",
        "JHPixelProLayerFlatten",
    ],
)
def test_compositing_nodes_registered_and_category(name: str) -> None:
    assert _cls(name).CATEGORY == "ComfyUI-JH-PixelPro/compositing"


def test_layer_add_exposes_27_blend_mode_combo() -> None:
    spec = _cls("JHPixelProLayerAdd").INPUT_TYPES()["required"]
    assert spec["blend_mode"][0] == BLEND_MODES
    assert len(spec["blend_mode"][0]) == 27


def test_stack_lifecycle_flatten() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0)))
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((1.0, 1.0, 1.0)), "normal", 0.5, 1.0, False
    )
    (out,) = _cls("JHPixelProLayerFlatten")().apply(stack2)
    assert torch.allclose(out, _solid((0.5, 0.5, 0.5)))


def test_layer_add_preserves_parent_stack_immutability() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0)))
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((1.0, 1.0, 1.0)), "normal", 1.0, 1.0, False
    )
    assert len(stack) == 1
    assert len(stack2) == 2


def test_layer_mask_limits_output() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0), 4))
    mask = torch.zeros(1, 4, 4)
    mask[:, :, 2:] = 1.0
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((1.0, 1.0, 1.0), 4), "normal", 1.0, 1.0, False, mask
    )
    (out,) = _cls("JHPixelProLayerFlatten")().apply(stack2)
    assert out[:, :, :2].mean().item() == 0.0
    assert out[:, :, 2:].mean().item() == 1.0


def test_layer_image_auto_resize_on_add() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0), 8))
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((1.0, 1.0, 1.0), 4), "normal", 1.0, 1.0, False
    )
    assert stack2[1]["image"].shape[-3:-1] == torch.Size([8, 8])


def test_layer_mask_auto_resize_on_add() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0), 8))
    mask = torch.ones(1, 4, 4)
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((1.0, 1.0, 1.0), 8), "normal", 1.0, 1.0, False, mask
    )
    assert stack2[1]["mask"].shape[-2:] == torch.Size([8, 8])


def test_clip_to_below_uses_previous_mask() -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0), 4))
    below_mask = torch.zeros(1, 4, 4)
    below_mask[:, :, :2] = 1.0
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((0.5, 0.5, 0.5), 4), "normal", 1.0, 1.0, False, below_mask
    )
    (stack3,) = _cls("JHPixelProLayerAdd")().apply(
        stack2, _solid((1.0, 1.0, 1.0), 4), "normal", 1.0, 1.0, True
    )
    (out,) = _cls("JHPixelProLayerFlatten")().apply(stack3)
    assert out[:, :, :2].mean().item() > out[:, :, 2:].mean().item()


def test_layer_group_eager_flatten_matches_manual_add() -> None:
    (parent,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.0, 0.0, 0.0), 4))
    (sub,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.2, 0.2, 0.2), 4))
    (sub2,) = _cls("JHPixelProLayerAdd")().apply(
        sub, _solid((0.8, 0.8, 0.8), 4), "normal", 0.5, 1.0, False
    )
    (grouped,) = _cls("JHPixelProLayerGroup")().apply(parent, sub2, "normal", 1.0)
    assert grouped[1]["is_group"] is True
    (out,) = _cls("JHPixelProLayerFlatten")().apply(grouped)
    assert torch.allclose(out, _solid((0.5, 0.5, 0.5), 4))


def test_layer_flatten_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty stack"):
        _cls("JHPixelProLayerFlatten")().apply([])


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_layer_add_supports_each_blend_mode(mode: str) -> None:
    (stack,) = _cls("JHPixelProLayerStackStart")().apply(_solid((0.25, 0.5, 0.75), 4))
    (stack2,) = _cls("JHPixelProLayerAdd")().apply(
        stack, _solid((0.8, 0.35, 0.2), 4), mode, 1.0, 1.0, False
    )
    (out,) = _cls("JHPixelProLayerFlatten")().apply(stack2)
    assert out.shape == stack[0]["image"].shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
