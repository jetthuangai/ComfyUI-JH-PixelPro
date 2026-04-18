"""Integration tests for the ComfyUI node wrapper ``JHPixelProFrequencySeparation``.

Does NOT require a running ComfyUI instance — loads the pack via
``importlib.util.spec_from_file_location`` the same way ComfyUI does, so
relative imports inside the pack resolve correctly.
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
    return pack.NODE_CLASS_MAPPINGS["JHPixelProFrequencySeparation"]


def test_node_registered_in_mappings(pack):
    assert "JHPixelProFrequencySeparation" in pack.NODE_CLASS_MAPPINGS
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencySeparation"]
        == "GPU Frequency Separation"
    )


def test_class_metadata(node_cls):
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/filters"
    assert node_cls.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert node_cls.RETURN_NAMES == ("low", "high")
    assert node_cls.FUNCTION == "run"


def test_input_types_structure(node_cls):
    spec = node_cls.INPUT_TYPES()
    assert "required" in spec
    required = spec["required"]
    assert set(required.keys()) == {"image", "radius", "sigma", "precision"}

    assert required["image"] == ("IMAGE",)

    radius_type, radius_meta = required["radius"]
    assert radius_type == "INT"
    assert radius_meta["default"] == 8
    assert radius_meta["min"] == 1
    assert radius_meta["max"] == 128

    sigma_type, sigma_meta = required["sigma"]
    assert sigma_type == "FLOAT"
    assert sigma_meta["default"] == 0.0
    assert sigma_meta["min"] == 0.0
    assert sigma_meta["max"] == 50.0

    precision_enum, precision_meta = required["precision"]
    assert precision_enum == ["float32", "float16"]
    assert precision_meta["default"] == "float32"


def test_run_shape_and_dtype_float32(node_cls):
    node = node_cls()
    image_bhwc = torch.rand(1, 64, 96, 3)

    low, high = node.run(image=image_bhwc, radius=8, sigma=0.0, precision="float32")

    assert low.shape == image_bhwc.shape
    assert high.shape == image_bhwc.shape
    assert low.dtype == torch.float32
    assert high.dtype == torch.float32


def test_run_reconstruction_invariant_bhwc(node_cls):
    node = node_cls()
    image_bhwc = torch.rand(2, 48, 48, 3)

    low, high = node.run(image=image_bhwc, radius=6, sigma=0.0, precision="float32")
    reconstructed = low + high

    assert torch.allclose(reconstructed, image_bhwc, atol=1e-5)


def test_run_float16_precision(node_cls):
    node = node_cls()
    image_bhwc = torch.rand(1, 64, 64, 3)

    low, high = node.run(image=image_bhwc, radius=4, sigma=0.0, precision="float16")

    assert low.dtype == torch.float16
    assert high.dtype == torch.float16
    assert low.shape == image_bhwc.shape
    reconstructed = (low + high).float()
    assert torch.allclose(reconstructed, image_bhwc, atol=1e-3)


def test_run_rejects_rgba(node_cls):
    node = node_cls()
    image_bhwc_rgba = torch.rand(1, 32, 32, 4)

    with pytest.raises(ValueError, match="SplitAlpha|4"):
        node.run(image=image_bhwc_rgba, radius=4, sigma=0.0, precision="float32")


def test_run_rejects_grayscale(node_cls):
    node = node_cls()
    image_bhwc_gray = torch.rand(1, 32, 32, 1)

    with pytest.raises(ValueError, match="3-channel"):
        node.run(image=image_bhwc_gray, radius=4, sigma=0.0, precision="float32")


def test_run_rejects_zero_radius(node_cls):
    node = node_cls()
    image_bhwc = torch.rand(1, 32, 32, 3)

    with pytest.raises(ValueError, match="radius must be >= 1"):
        node.run(image=image_bhwc, radius=0, sigma=0.0, precision="float32")


def test_run_output_is_contiguous(node_cls):
    """ComfyUI downstream nodes may assume contiguous BHWC layout."""
    node = node_cls()
    image_bhwc = torch.rand(1, 32, 32, 3)

    low, high = node.run(image=image_bhwc, radius=4, sigma=0.0, precision="float32")

    assert low.is_contiguous()
    assert high.is_contiguous()


def test_run_preserves_device(node_cls):
    node = node_cls()
    image_bhwc = torch.rand(1, 32, 32, 3)

    low, high = node.run(image=image_bhwc, radius=4, sigma=0.0, precision="float32")

    assert low.device == image_bhwc.device
    assert high.device == image_bhwc.device
