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
# Category groups exposed to ComfyUI Add Node menu (under root "ComfyUI-JH-PixelPro/"):
#   filters    — pixel-domain filters (FS, ES)
#   color      — color-grade layer (Lum, ColorMatcher, ToneCurve)
#   mask       — mask creation / refinement (MR, HFDM)
#   geometry   — geometric transforms (Aligner, LensDistortion)
#   face       — face-pipeline domain (FaceDetect, UnwrapFace)
if __package__:
    from .nodes import (
        JHPixelProColorMatcher,
        JHPixelProEdgeAwareSmoother,
        JHPixelProFaceDetect,
        JHPixelProFacialAligner,
        JHPixelProFrequencySeparation,
        JHPixelProHighFreqDetailMasker,
        JHPixelProLensDistortion,
        JHPixelProLuminosityMasking,
        JHPixelProSubPixelMaskRefiner,
        JHPixelProToneCurve,
        JHPixelProUnwrapFace,
    )

    NODE_CLASS_MAPPINGS["JHPixelProFrequencySeparation"] = JHPixelProFrequencySeparation
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencySeparation"] = "GPU Frequency Separation"

    NODE_CLASS_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = JHPixelProSubPixelMaskRefiner
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = "Sub-Pixel Mask Refiner"

    NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareSmoother"] = JHPixelProEdgeAwareSmoother
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProEdgeAwareSmoother"] = "Edge-Aware Skin Smoother"

    NODE_CLASS_MAPPINGS["JHPixelProHighFreqDetailMasker"] = JHPixelProHighFreqDetailMasker
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHighFreqDetailMasker"] = "High-Frequency Detail Masker"

    NODE_CLASS_MAPPINGS["JHPixelProLuminosityMasking"] = JHPixelProLuminosityMasking
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLuminosityMasking"] = "Luminosity Masking"

    NODE_CLASS_MAPPINGS["JHPixelProFacialAligner"] = JHPixelProFacialAligner
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFacialAligner"] = "Landmark Facial Aligner"

    NODE_CLASS_MAPPINGS["JHPixelProLensDistortion"] = JHPixelProLensDistortion
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLensDistortion"] = "Lens Distortion Corrector"

    NODE_CLASS_MAPPINGS["JHPixelProFaceDetect"] = JHPixelProFaceDetect
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceDetect"] = "JHPixelProFaceDetect"

    NODE_CLASS_MAPPINGS["JHPixelProUnwrapFace"] = JHPixelProUnwrapFace
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProUnwrapFace"] = "JHPixelProUnwrapFace"

    NODE_CLASS_MAPPINGS["JHPixelProColorMatcher"] = JHPixelProColorMatcher
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorMatcher"] = "Color Matcher (LAB)"

    NODE_CLASS_MAPPINGS["JHPixelProToneCurve"] = JHPixelProToneCurve
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProToneCurve"] = "Tone Curve (RGB)"

# Web extensions (JS/CSS) — not used in Phase 1.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
