"""v1.8.0 node wrappers: Auto Tone, Auto Upright, Decontaminate Colors, Motion / Zoom Blur,
plus the new optional inputs of Edge-Aware Refiner and Dust & Spot."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_v180"


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


def _node(name: str):
    return _load_pack().NODE_CLASS_MAPPINGS[name]()


NEW_NODES = {
    "JHPixelProAutoTone": ("Auto: Tone & White Balance", "color"),
    "JHPixelProAutoUpright": ("Auto: Upright", "geometry"),
    "JHPixelProDecontaminateColors": ("Mask: Decontaminate Colors", "mask"),
    "JHPixelProMotionZoomBlur": ("Filters: Motion / Zoom Blur", "filters"),
}


@pytest.mark.parametrize("name", sorted(NEW_NODES))
def test_registered(name: str) -> None:
    pack = _load_pack()
    display, group = NEW_NODES[name]
    assert pack.NODE_DISPLAY_NAME_MAPPINGS[name] == display
    assert f"ComfyUI-JH-PixelPro/{group}" == pack.NODE_CLASS_MAPPINGS[name].CATEGORY


def _photo() -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    base = torch.linspace(0.05, 0.5, 48).view(1, 1, 48, 1).expand(1, 32, 48, 3)
    return base * torch.tensor([1.1, 1.0, 0.8]) + 0.02 * torch.rand(1, 32, 48, 3, generator=gen)


def test_auto_tone_outputs_colorlab_preset() -> None:
    image = _photo()
    out, preset = _node("JHPixelProAutoTone").auto(image, 1.0)
    params = json.loads(preset)
    assert params["tone_engine"] == "local" and params["exposure_mode"] == "linear"
    assert float(params["basic_exposure"]) > 0  # dark ramp gets brighter
    assert out.shape == image.shape and not torch.equal(out, image)
    # The preset reproduces the image in Apply ColorLab Preset.
    again, _ = _node("JHPixelProColorLabApply").apply(image, preset, 1.0)
    assert torch.allclose(again, out, atol=1e-6)


def test_auto_tone_strength_zero_and_batch() -> None:
    image = _photo()
    out, _ = _node("JHPixelProAutoTone").auto(image, 0.0)
    assert torch.equal(out, image)
    batch = torch.cat([image, image.flip(2)])
    out, _ = _node("JHPixelProAutoTone").auto(batch, 1.0, black_and_white=True)
    assert out.shape == batch.shape
    assert torch.allclose(out[..., 0], out[..., 1]) and torch.allclose(out[..., 1], out[..., 2])


def test_upright_node_untouched_without_lines() -> None:
    noise = torch.rand(2, 64, 80, 3, generator=torch.Generator().manual_seed(3))
    out, confidence, matrix = _node("JHPixelProAutoUpright").upright(noise, "auto", 1.0, "none")
    assert torch.equal(out, noise) and confidence == 0.0
    assert len(json.loads(matrix)) == 2


def test_decontaminate_node() -> None:
    image = torch.rand(1, 32, 32, 3)
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    (out,) = _node("JHPixelProDecontaminateColors").decontaminate(image, mask, 16, 1.0)
    assert torch.equal(out, image)  # hard mask: nothing to clean


def test_motion_zoom_node() -> None:
    image = torch.rand(1, 32, 32, 3)
    node = _node("JHPixelProMotionZoomBlur")
    (out,) = node.blur(image, "motion", 0.0, 0.0, 0.15, 0.5, 0.5)
    assert torch.equal(out, image)
    (out,) = node.blur(image, "zoom", 0.0, 0.0, 0.3, 0.5, 0.5, mask=torch.zeros(1, 32, 32))
    assert torch.equal(out, image)


def test_refiner_and_dust_new_inputs_default_to_old_behaviour() -> None:
    refiner = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareMaskRefiner"]
    assert refiner.INPUT_TYPES()["optional"]["guide_mode"][1]["default"] == "luma"
    dust = _load_pack().NODE_CLASS_MAPPINGS["JHPixelProDustSpotRemoval"]
    assert dust.INPUT_TYPES()["optional"]["fill_mode"][1]["default"] == "blur"
    mask, guide = torch.rand(1, 24, 24), torch.rand(1, 24, 24, 3)
    (old,) = refiner().refine(mask, guide, 8, 0.001, 0.0)
    (luma,) = refiner().refine(mask, guide, 8, 0.001, 0.0, guide_mode="luma")
    (rgb,) = refiner().refine(mask, guide, 8, 0.001, 0.0, guide_mode="rgb")
    assert torch.equal(old, luma) and not torch.equal(old, rgb)
