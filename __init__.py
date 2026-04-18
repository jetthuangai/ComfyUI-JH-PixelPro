"""ComfyUI-JH-PixelPro — GPU-accelerated pro image pack for ComfyUI.

Entry point loaded by ComfyUI. Concrete nodes register themselves through
``NODE_CLASS_MAPPINGS`` in child modules.
"""  # noqa: N999 — ComfyUI custom-node packs use hyphenated folder names by convention.

from __future__ import annotations

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

# ComfyUI loads this module via ``importlib.util.spec_from_file_location`` with
# ``submodule_search_locations`` set, so ``__package__`` is truthy and the
# relative import below resolves to ``.nodes``. Pytest, on the other hand, may
# walk up and try to import this file as a top-level module during collection
# (because the parent directory name contains hyphens); in that case
# ``__package__`` is empty and we skip the registration step — the node tests
# load the pack themselves via ``importlib.util.spec_from_file_location``.
if __package__:
    from .nodes import (
        JHPixelProEdgeAwareSmoother,
        JHPixelProFrequencySeparation,
        JHPixelProSubPixelMaskRefiner,
    )

    NODE_CLASS_MAPPINGS["JHPixelProFrequencySeparation"] = JHPixelProFrequencySeparation
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencySeparation"] = "GPU Frequency Separation"

    NODE_CLASS_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = JHPixelProSubPixelMaskRefiner
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = "Sub-Pixel Mask Refiner"

    NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareSmoother"] = JHPixelProEdgeAwareSmoother
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProEdgeAwareSmoother"] = "Edge-Aware Skin Smoother"

# Web extensions (JS/CSS) — not used in Phase 1.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
