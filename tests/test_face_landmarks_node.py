from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_PACK_ROOT = Path(__file__).resolve().parent.parent
_PACK_MODULE_NAME = "comfyui_jh_pixelpro_under_test_face_landmarks"


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


def _draw_overlay():
    _load_pack()
    return importlib.import_module(f"{_PACK_MODULE_NAME}.nodes.face_landmarks_node")._draw_overlay


def test_overlay_ignores_landmarks_outside_the_frame() -> None:
    # Regression (v1.5.3): a landmark at x < 0 turned the slice ``x0:x1`` into ``0:-n`` and
    # painted a green stripe across the whole row band.
    image = torch.zeros(1, 40, 60, 3)
    points = torch.tensor([[[[-12.0, 20.0], [30.0, -5.0], [75.0, 10.0], [30.0, 20.0]]]])
    out = _draw_overlay()(image, points)
    green = out[0, ..., 1] > 0.5
    assert int(green.sum()) == 9  # only the in-frame point, as a 3x3 dot
    assert bool(green[19:22, 29:32].all())


def test_overlay_draws_normalised_points() -> None:
    image = torch.zeros(1, 21, 21, 3)
    points = torch.tensor([[[[0.5, 0.5]]]])
    out = _draw_overlay()(image, points)
    assert bool((out[0, 9:12, 9:12, 1] == 1.0).all())
