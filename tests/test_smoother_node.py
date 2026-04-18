"""Integration tests for the ComfyUI node wrapper ``JHPixelProEdgeAwareSmoother``.

Loads the pack via ``importlib.util.spec_from_file_location`` so relative
imports inside the hyphenated pack root resolve correctly without requiring
a running ComfyUI instance.
"""

from __future__ import annotations

import importlib.util
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


@pytest.fixture(scope="module")
def pack():
    return _load_pack()


@pytest.fixture(scope="module")
def node_cls(pack):
    return pack.NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareSmoother"]


def test_node_registered_in_mappings(pack):
    assert "JHPixelProEdgeAwareSmoother" in pack.NODE_CLASS_MAPPINGS
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProEdgeAwareSmoother"]
        == "Edge-Aware Skin Smoother"
    )


def test_class_metadata(node_cls):
    assert node_cls.CATEGORY == "image/pixelpro/filters"
    assert node_cls.RETURN_TYPES == ("IMAGE",)
    assert node_cls.RETURN_NAMES == ("image",)
    assert node_cls.FUNCTION == "run"


def test_input_types_structure(node_cls):
    types = node_cls.INPUT_TYPES()
    assert set(types["required"].keys()) == {
        "image",
        "strength",
        "sigma_color",
        "sigma_space",
    }
    assert set(types["optional"].keys()) == {"mask"}

    assert types["required"]["image"] == ("IMAGE",)
    assert types["optional"]["mask"] == ("MASK",)

    assert types["required"]["strength"][0] == "FLOAT"
    assert types["required"]["strength"][1]["default"] == 0.4
    assert types["required"]["strength"][1]["min"] == 0.0
    assert types["required"]["strength"][1]["max"] == 1.0

    assert types["required"]["sigma_color"][1]["default"] == 0.1
    assert types["required"]["sigma_space"][1]["default"] == 6.0


def test_run_bhwc_roundtrip_no_mask(node_cls):
    torch.manual_seed(0)
    image = torch.rand(1, 64, 64, 3, dtype=torch.float32)

    out = node_cls().run(image, 0.4, 0.1, 6.0)[0]

    assert isinstance(out, torch.Tensor)
    assert out.shape == image.shape
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_run_with_mask_bhw(node_cls):
    torch.manual_seed(0)
    image = torch.rand(1, 48, 48, 3, dtype=torch.float32)
    # Right half mask = 0 — those pixels must remain exactly the input.
    mask = torch.zeros(1, 48, 48, dtype=torch.float32)
    mask[:, :, :24] = 1.0

    out = node_cls().run(image, 1.0, 0.1, 6.0, mask=mask)[0]

    assert out.shape == image.shape
    assert torch.equal(out[:, :, 24:, :], image[:, :, 24:, :])


def test_run_strength_zero_identity(node_cls):
    torch.manual_seed(0)
    image = torch.rand(1, 32, 32, 3, dtype=torch.float32)

    out = node_cls().run(image, 0.0, 0.1, 6.0)[0]

    assert torch.equal(out, image)


def test_run_batch(node_cls):
    torch.manual_seed(0)
    image = torch.rand(4, 32, 32, 3, dtype=torch.float32)

    out = node_cls().run(image, 0.4, 0.1, 6.0)[0]

    assert out.shape == (4, 32, 32, 3)
    assert out.dtype == torch.float32
