"""v1.9.0 node wrappers: Heal / Content-Aware Fill, Auto Blemish, Defringe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_v190"


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


NEW_NODES = {
    "JHPixelProHeal": ("Filters: Heal / Content-Aware Fill", "filters"),
    "JHPixelProAutoBlemish": ("Face: Auto Blemish", "face"),
    "JHPixelProDefringe": ("Color: Defringe", "color"),
}


@pytest.mark.parametrize("name", sorted(NEW_NODES))
def test_registered(name: str) -> None:
    pack = _load_pack()
    display, group = NEW_NODES[name]
    assert pack.NODE_DISPLAY_NAME_MAPPINGS[name] == display
    assert f"ComfyUI-JH-PixelPro/{group}" == pack.NODE_CLASS_MAPPINGS[name].CATEGORY


def test_pack_registers_at_least_61_nodes() -> None:
    assert len(_load_pack().NODE_CLASS_MAPPINGS) >= 61


def test_heal_node() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProHeal"]()
    image = torch.rand(1, 48, 64, 3)
    mask = torch.zeros(1, 48, 64)
    (out,) = node.heal(image, mask, "content_aware", 2, 1.5)
    assert torch.equal(out, image)
    mask[:, 20:28, 30:40] = 1.0
    (out,) = node.heal(image, mask, "smooth", 0, 0.0, context=2.0)
    assert out.shape == image.shape and not torch.equal(out, image)


def test_blemish_node_with_mask() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProAutoBlemish"]()
    flat = torch.tensor([0.86, 0.66, 0.55]).view(1, 1, 1, 3).expand(1, 64, 64, 3).contiguous()
    out, spots, count = node.retouch(flat, 0.5, 1.0, mask=torch.ones(1, 64, 64))
    assert torch.equal(out, flat) and not spots.any() and count == 0


def test_defringe_node_flat_image() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProDefringe"]()
    flat = torch.full((1, 40, 40, 3), 0.5)
    (out,) = node.defringe(flat, 1.0, 0.0, True)
    assert torch.equal(out, flat)
