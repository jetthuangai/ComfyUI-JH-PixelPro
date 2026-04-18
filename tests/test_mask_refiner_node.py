"""Integration tests for the ComfyUI node wrapper ``JHPixelProSubPixelMaskRefiner``.

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


def _disk_mask(batch: int, size: int, radius: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (size - 1) / 2.0
    mask = ((yy - center) ** 2 + (xx - center) ** 2 <= radius**2).to(dtype=torch.float32)
    return mask.unsqueeze(0).repeat(batch, 1, 1)


@pytest.fixture(scope="module")
def pack():
    return _load_pack()


@pytest.fixture(scope="module")
def node_cls(pack):
    return pack.NODE_CLASS_MAPPINGS["JHPixelProSubPixelMaskRefiner"]


def test_node_registered_in_mappings(pack):
    assert "JHPixelProSubPixelMaskRefiner" in pack.NODE_CLASS_MAPPINGS
    assert (
        pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSubPixelMaskRefiner"]
        == "Sub-Pixel Mask Refiner"
    )


def test_class_metadata(node_cls):
    assert node_cls.CATEGORY == "image/pixelpro/morphology"
    assert node_cls.RETURN_TYPES == ("MASK",)
    assert node_cls.RETURN_NAMES == ("refined_mask",)
    assert node_cls.FUNCTION == "run"


def test_input_types_structure(node_cls):
    spec = node_cls.INPUT_TYPES()
    assert "required" in spec
    required = spec["required"]
    assert set(required.keys()) == {
        "mask",
        "erosion_radius",
        "dilation_radius",
        "feather_sigma",
        "threshold",
    }

    assert required["mask"] == ("MASK",)

    er_type, er_meta = required["erosion_radius"]
    assert er_type == "INT"
    assert er_meta["default"] == 2
    assert er_meta["min"] == 0
    assert er_meta["max"] == 64

    dr_type, dr_meta = required["dilation_radius"]
    assert dr_type == "INT"
    assert dr_meta["default"] == 4
    assert dr_meta["min"] == 0
    assert dr_meta["max"] == 64

    sigma_type, sigma_meta = required["feather_sigma"]
    assert sigma_type == "FLOAT"
    assert sigma_meta["default"] == 2.0
    assert sigma_meta["min"] == 0.1
    assert sigma_meta["max"] == 32.0
    assert sigma_meta["step"] == 0.1

    thr_type, thr_meta = required["threshold"]
    assert thr_type == "FLOAT"
    assert thr_meta["default"] == 0.5
    assert thr_meta["min"] == 0.0
    assert thr_meta["max"] == 1.0
    assert thr_meta["step"] == 0.01


def test_run_shape_and_dtype_bhw(node_cls):
    node = node_cls()
    mask = _disk_mask(batch=2, size=64, radius=20)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
        threshold=0.5,
    )

    assert refined.shape == mask.shape
    assert refined.ndim == 3
    assert refined.dtype == torch.float32


def test_run_output_in_unit_range(node_cls):
    node = node_cls()
    mask = _disk_mask(batch=1, size=48, radius=14)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
        threshold=0.5,
    )

    assert refined.min().item() >= 0.0
    assert refined.max().item() <= 1.0


def test_run_preserves_bc1hw_rank(node_cls):
    """Core preserves channel dim when given BC1HW; wrapper must pass it through."""
    node = node_cls()
    mask_bhw = _disk_mask(batch=1, size=48, radius=14)
    mask_bc1hw = mask_bhw.unsqueeze(1)

    (refined,) = node.run(
        mask=mask_bc1hw,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=2.0,
        threshold=0.5,
    )

    assert refined.shape == mask_bc1hw.shape
    assert refined.ndim == 4


def test_run_zero_radii_is_clamped_blur(node_cls):
    """er=dr=0 is the documented edge case: no protected zone, output = clamped blur."""
    node = node_cls()
    mask = _disk_mask(batch=1, size=48, radius=14)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=0,
        dilation_radius=0,
        feather_sigma=1.5,
        threshold=0.5,
    )

    assert refined.shape == mask.shape
    assert refined.dtype == torch.float32
    assert refined.min().item() >= 0.0
    assert refined.max().item() <= 1.0


def test_run_preserves_device(node_cls):
    node = node_cls()
    mask = _disk_mask(batch=1, size=32, radius=10)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=1,
        dilation_radius=2,
        feather_sigma=1.0,
        threshold=0.5,
    )

    assert refined.device == mask.device


def test_run_inside_core_pinned_to_one(node_cls):
    """Pixels well inside the disk (after erosion) must stay exactly 1.0."""
    node = node_cls()
    mask = _disk_mask(batch=1, size=64, radius=20)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=3,
        dilation_radius=6,
        feather_sigma=1.5,
        threshold=0.5,
    )

    center = refined.shape[-1] // 2
    assert refined[0, center, center].item() == pytest.approx(1.0, abs=1e-6)


def test_run_outside_core_pinned_to_zero(node_cls):
    """Pixels far outside the dilated disk must stay exactly 0.0."""
    node = node_cls()
    mask = _disk_mask(batch=1, size=64, radius=10)

    (refined,) = node.run(
        mask=mask,
        erosion_radius=2,
        dilation_radius=4,
        feather_sigma=1.5,
        threshold=0.5,
    )

    assert refined[0, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
