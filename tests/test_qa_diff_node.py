from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_qa_diff"


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
    _load_pack()
    return importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.qa_diff_node").JHPixelProQADiff


def test_node_metadata(node_cls) -> None:
    assert node_cls.CATEGORY == "ComfyUI-JH-PixelPro/qa"
    assert node_cls.RETURN_TYPES == ("IMAGE", "FLOAT", "FLOAT", "STRING")
    assert node_cls.RETURN_NAMES == ("diff", "ssim", "psnr", "report")
    assert node_cls.FUNCTION == "run"


def test_input_types(node_cls) -> None:
    spec = node_cls.INPUT_TYPES()
    assert set(spec["required"]) == {"image_a", "image_b", "diff_mode", "amplify", "window_size"}
    assert spec["required"]["diff_mode"][0] == ["heatmap", "abs", "ssim_map"]
    assert spec["required"]["window_size"][1]["default"] == 11
    assert spec["required"]["window_size"][1]["step"] == 2
    assert "mask" in spec["optional"]
    assert len(node_cls.OUTPUT_TOOLTIPS) == len(node_cls.RETURN_TYPES)
    assert "item 0" in node_cls.OUTPUT_TOOLTIPS[1] and "item 0" in node_cls.OUTPUT_TOOLTIPS[2]
    for kind in spec.values():
        for cfg in kind.values():
            assert "tooltip" in cfg[1]


def test_run_report_json_safe(node_cls) -> None:
    a = torch.rand(2, 48, 48, 3) * 0.6 + 0.2
    b = a.clone()
    b[1] += 0.1
    diff, ssim, psnr, report = node_cls().run(a, b, "heatmap", 4.0, 11)
    assert diff.shape == a.shape
    data = json.loads(report)
    assert len(data) == 2
    assert psnr == 100.0 and ssim == pytest.approx(1.0, abs=1e-5)
    assert data[1]["psnr"] == pytest.approx(20.0, abs=0.01)
    assert "Infinity" not in report and "NaN" not in report


def test_run_with_mask(node_cls) -> None:
    a = torch.rand(1, 48, 48, 3) * 0.6 + 0.2
    mask = torch.zeros(1, 48, 48)
    mask[:, :, :24] = 1.0
    b = a.clone()
    b[:, :, 24:] += 0.2
    _, _, psnr, _ = node_cls().run(a, b, "abs", 4.0, 11, mask)
    assert psnr == 100.0


def test_node_registered() -> None:
    pack = _load_pack()
    assert pack.NODE_CLASS_MAPPINGS["JHPixelProQADiff"] is node_cls_from_module()
    assert pack.NODE_DISPLAY_NAME_MAPPINGS["JHPixelProQADiff"] == "QA: Image Diff (SSIM/PSNR)"


def node_cls_from_module():
    module = importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.qa_diff_node")
    return module.JHPixelProQADiff
