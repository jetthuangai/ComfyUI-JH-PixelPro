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
#   color      — color-grade layer (Lum, ColorMatcher, ToneCurve, HALDIdentity,
#                 LUTExport, LUTImport, HueSatRange, SatMask, ToneMatchLUT, ColorLab)
#   compositing — Photoshop-style layer stack and blend modes
#   mask       — mask creation / refinement (MR, HFDM)
#   geometry   — geometric transforms (Aligner, LensDistortion)
#   face       — face-pipeline domain (FaceDetect, UnwrapFace, Landmarks, Warp, BeautyBlend)
#   looks      — JSON-driven dropdown look presets (N-22 M6 Looks)
if __package__:
    from .nodes import (
        JHPixelProAlphaMatteExtractor,
        JHPixelProColorLab,
        JHPixelProColorMatcher,
        JHPixelProEdgeAwareMaskRefiner,
        JHPixelProEdgeAwareSmoother,
        JHPixelProFaceBeautyBlend,
        JHPixelProFaceDetect,
        JHPixelProFaceLandmarks,
        JHPixelProFaceWarp,
        JHPixelProFacialAligner,
        JHPixelProFrequencySeparation,
        JHPixelProHALDIdentity,
        JHPixelProHighFreqDetailMasker,
        JHPixelProHueSaturationRange,
        JHPixelProLayerAdd,
        JHPixelProLayerFlatten,
        JHPixelProLayerGroup,
        JHPixelProLayerStackStart,
        JHPixelProLensDistortion,
        JHPixelProLookSelect,
        JHPixelProLuminosityMasking,
        JHPixelProLUTExport,
        JHPixelProLUTImport,
        JHPixelProMaskCombine,
        JHPixelProMaskMorphology,
        JHPixelProSaturationMask,
        JHPixelProSubPixelMaskRefiner,
        JHPixelProToneCurve,
        JHPixelProToneMatchLUT,
        JHPixelProTrimapBuilder,
        JHPixelProUnwrapFace,
    )

    NODE_CLASS_MAPPINGS["JHPixelProFrequencySeparation"] = JHPixelProFrequencySeparation
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencySeparation"] = "GPU Frequency Separation"

    NODE_CLASS_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = JHPixelProSubPixelMaskRefiner
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSubPixelMaskRefiner"] = "Sub-Pixel Mask Refiner"

    NODE_CLASS_MAPPINGS["JHPixelProEdgeAwareMaskRefiner"] = JHPixelProEdgeAwareMaskRefiner
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProEdgeAwareMaskRefiner"] = "Mask: Edge-Aware Refiner"

    NODE_CLASS_MAPPINGS["JHPixelProAlphaMatteExtractor"] = JHPixelProAlphaMatteExtractor
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAlphaMatteExtractor"] = "Mask: Alpha Matte"

    NODE_CLASS_MAPPINGS["JHPixelProTrimapBuilder"] = JHPixelProTrimapBuilder
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProTrimapBuilder"] = "Mask: Trimap Builder"

    NODE_CLASS_MAPPINGS["JHPixelProMaskMorphology"] = JHPixelProMaskMorphology
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskMorphology"] = "Mask: Morphology"

    NODE_CLASS_MAPPINGS["JHPixelProMaskCombine"] = JHPixelProMaskCombine
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskCombine"] = "Mask: Combine"

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

    NODE_CLASS_MAPPINGS["JHPixelProFaceLandmarks"] = JHPixelProFaceLandmarks
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceLandmarks"] = "Face Landmarks (MediaPipe 468)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceWarp"] = JHPixelProFaceWarp
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceWarp"] = "Face Warp (Delaunay per-triangle)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceBeautyBlend"] = JHPixelProFaceBeautyBlend
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceBeautyBlend"] = "Face Beauty Blend"

    NODE_CLASS_MAPPINGS["JHPixelProUnwrapFace"] = JHPixelProUnwrapFace
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProUnwrapFace"] = "JHPixelProUnwrapFace"

    NODE_CLASS_MAPPINGS["JHPixelProColorMatcher"] = JHPixelProColorMatcher
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorMatcher"] = "Color Matcher (LAB)"

    NODE_CLASS_MAPPINGS["JHPixelProToneCurve"] = JHPixelProToneCurve
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProToneCurve"] = "Tone Curve (RGB)"

    NODE_CLASS_MAPPINGS["JHPixelProHALDIdentity"] = JHPixelProHALDIdentity
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHALDIdentity"] = "HALD Identity"

    NODE_CLASS_MAPPINGS["JHPixelProLUTExport"] = JHPixelProLUTExport
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTExport"] = "LUT Export (.cube)"

    NODE_CLASS_MAPPINGS["JHPixelProLUTImport"] = JHPixelProLUTImport
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTImport"] = "LUT Import (.cube)"

    NODE_CLASS_MAPPINGS["JHPixelProHueSaturationRange"] = JHPixelProHueSaturationRange
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHueSaturationRange"] = "Hue/Saturation per Range"

    NODE_CLASS_MAPPINGS["JHPixelProSaturationMask"] = JHPixelProSaturationMask
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSaturationMask"] = "Saturation Mask Builder"

    NODE_CLASS_MAPPINGS["JHPixelProToneMatchLUT"] = JHPixelProToneMatchLUT
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProToneMatchLUT"] = "Tone Match LUT (auto-gen .cube)"

    NODE_CLASS_MAPPINGS["JHPixelProLookSelect"] = JHPixelProLookSelect
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLookSelect"] = "Look: Select Preset"

    NODE_CLASS_MAPPINGS["JHPixelProColorLab"] = JHPixelProColorLab
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorLab"] = "Color: ColorLab (ACR)"

    NODE_CLASS_MAPPINGS["JHPixelProLayerStackStart"] = JHPixelProLayerStackStart
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerStackStart"] = "Compositing: Layer Stack Start"

    NODE_CLASS_MAPPINGS["JHPixelProLayerAdd"] = JHPixelProLayerAdd
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerAdd"] = "Compositing: Layer Add"

    NODE_CLASS_MAPPINGS["JHPixelProLayerGroup"] = JHPixelProLayerGroup
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerGroup"] = "Compositing: Layer Group (nest)"

    NODE_CLASS_MAPPINGS["JHPixelProLayerFlatten"] = JHPixelProLayerFlatten
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerFlatten"] = "Compositing: Layer Flatten"

# Web extensions (JS/CSS) — not used in Phase 1.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
