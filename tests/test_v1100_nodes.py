"""v1.10.0 node wrappers: Sky Mask, HDR Merge, Cull Score."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_v1100"


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
    "JHPixelProSkyMask": ("Mask: Sky", "mask"),
    "JHPixelProHDRMerge": ("Color: HDR Merge (bracket)", "color"),
    "JHPixelProCullScore": ("QA: Cull Score", "qa"),
}


@pytest.mark.parametrize("name", sorted(NEW_NODES))
def test_registered(name: str) -> None:
    pack = _load_pack()
    display, group = NEW_NODES[name]
    assert pack.NODE_DISPLAY_NAME_MAPPINGS[name] == display
    assert f"ComfyUI-JH-PixelPro/{group}" == pack.NODE_CLASS_MAPPINGS[name].CATEGORY


def test_pack_registers_64_nodes() -> None:
    assert len(_load_pack().NODE_CLASS_MAPPINGS) == 64


def test_sky_node_shape() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProSkyMask"]()
    (mask,) = node.mask(torch.rand(2, 32, 48, 3), 0.5, True)
    assert mask.shape == (2, 32, 48)


def test_hdr_node_returns_one_image() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProHDRMerge"]()
    base = torch.rand(1, 64, 64, 3)
    (out,) = node.merge(torch.cat([base * 0.4, base, (base * 2).clamp(0, 1)]), True, True)
    assert out.shape == (1, 64, 64, 3)


def test_cull_node_never_returns_empty() -> None:
    node = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProCullScore"]()
    images = torch.stack([torch.full((64, 64, 3), v) for v in (0.3, 0.5, 0.7)])  # flat = soft
    selected, report, kept = node.cull(images, 0, 0.9, False)  # nothing reaches 0.9
    assert kept == 1 and selected.shape[0] == 1
    rows = json.loads(report)
    assert len(rows) == 3 and sum(r["keep"] for r in rows) == 1
