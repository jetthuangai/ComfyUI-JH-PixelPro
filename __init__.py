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
#   filters    — pixel-domain filters (FS, ES, Smart Sharpen)
#   color      — color-grade layer (Lum, ColorMatcher, ToneCurve, HALDIdentity,
#                 LUTExport, LUTImport, HueSatRange, SatMask, ToneMatchLUT, ColorLab)
#   compositing — Photoshop-style layer stack and blend modes
#   mask       — mask creation / refinement (MR, HFDM)
#   geometry   — geometric transforms (Aligner, LensDistortion)
#   face       — face-pipeline domain (FaceDetect, UnwrapFace, Landmarks, Warp, BeautyBlend)
#   looks      — JSON-driven dropdown look presets (N-22 M6 Looks)
#   qa         — before/after comparison metrics (QA Image Diff)
if __package__:
    from .nodes import (
        JHPixelProAlphaMatteExtractor,
        JHPixelProAutoAlign,
        JHPixelProAutoBlemish,
        JHPixelProAutoTone,
        JHPixelProAutoUpright,
        JHPixelProCLAHE,
        JHPixelProColorLab,
        JHPixelProColorLabApply,
        JHPixelProColorMatcher,
        JHPixelProColorMatcherRegion,
        JHPixelProCullScore,
        JHPixelProDecontaminateColors,
        JHPixelProDefringe,
        JHPixelProDepthAnything,
        JHPixelProDepthLensBlur,
        JHPixelProDustSpotRemoval,
        JHPixelProEdgeAwareMaskRefiner,
        JHPixelProEdgeAwareSmoother,
        JHPixelProFaceBeautyBlend,
        JHPixelProFaceDetect,
        JHPixelProFaceDetectLite,
        JHPixelProFaceLandmarks,
        JHPixelProFaceWarp,
        JHPixelProFaceWarpTPS,
        JHPixelProFacialAligner,
        JHPixelProFrequencyRecombine,
        JHPixelProFrequencySeparation,
        JHPixelProHALDIdentity,
        JHPixelProHDRMerge,
        JHPixelProHeal,
        JHPixelProHighFreqDetailMasker,
        JHPixelProHueSaturationRange,
        JHPixelProLayerAdd,
        JHPixelProLayerFlatten,
        JHPixelProLayerGroup,
        JHPixelProLayerStackStart,
        JHPixelProLensDistortion,
        JHPixelProLightroomPreset,
        JHPixelProLightroomPresetSheet,
        JHPixelProLookSelect,
        JHPixelProLuminosityMasking,
        JHPixelProLUTExport,
        JHPixelProLUTImport,
        JHPixelProLUTPreset,
        JHPixelProMaskCleanup,
        JHPixelProMaskCombine,
        JHPixelProMaskEdgeSmoother,
        JHPixelProMaskMorphology,
        JHPixelProMaskPreview,
        JHPixelProMotionZoomBlur,
        JHPixelProPaletteExtract,
        JHPixelProPyramidBlend,
        JHPixelProQADiff,
        JHPixelProSAMMask,
        JHPixelProSaturationMask,
        JHPixelProShadowsHighlights,
        JHPixelProSkinToneTriRegion,
        JHPixelProSkyMask,
        JHPixelProSmartSharpen,
        JHPixelProSubPixelMaskRefiner,
        JHPixelProToneCurve,
        JHPixelProToneMatchLUT,
        JHPixelProTrimapBuilder,
        JHPixelProUnwrapFace,
    )

    NODE_CLASS_MAPPINGS["JHPixelProFrequencySeparation"] = JHPixelProFrequencySeparation
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencySeparation"] = "GPU Frequency Separation"

    NODE_CLASS_MAPPINGS["JHPixelProFrequencyRecombine"] = JHPixelProFrequencyRecombine
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFrequencyRecombine"] = "Frequency Recombine (low + high)"

    NODE_CLASS_MAPPINGS["JHPixelProSmartSharpen"] = JHPixelProSmartSharpen
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSmartSharpen"] = "Filters: Smart Sharpen"

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

    NODE_CLASS_MAPPINGS["JHPixelProMaskEdgeSmoother"] = JHPixelProMaskEdgeSmoother
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskEdgeSmoother"] = "Mask: Edge Smoother"

    NODE_CLASS_MAPPINGS["JHPixelProMaskCleanup"] = JHPixelProMaskCleanup
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskCleanup"] = "Mask: Cleanup"

    NODE_CLASS_MAPPINGS["JHPixelProMaskPreview"] = JHPixelProMaskPreview
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMaskPreview"] = "Mask: Preview Overlay"

    NODE_CLASS_MAPPINGS["JHPixelProSAMMask"] = JHPixelProSAMMask
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSAMMask"] = "Mask: SAM One-Click"

    NODE_CLASS_MAPPINGS["JHPixelProDustSpotRemoval"] = JHPixelProDustSpotRemoval
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDustSpotRemoval"] = "Filters: Dust & Spot Removal"

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

    NODE_CLASS_MAPPINGS["JHPixelProAutoAlign"] = JHPixelProAutoAlign
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAutoAlign"] = "Geometry: Auto Align"

    NODE_CLASS_MAPPINGS["JHPixelProFaceDetect"] = JHPixelProFaceDetect
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceDetect"] = "Face Detect (MediaPipe)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceDetectLite"] = JHPixelProFaceDetectLite
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceDetectLite"] = "Face: Detect Lite (YuNet GPU)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceLandmarks"] = JHPixelProFaceLandmarks
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceLandmarks"] = "Face Landmarks (MediaPipe 468)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceWarp"] = JHPixelProFaceWarp
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceWarp"] = "Face Warp (Delaunay per-triangle)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceWarpTPS"] = JHPixelProFaceWarpTPS
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceWarpTPS"] = "Face Warp (TPS smooth)"

    NODE_CLASS_MAPPINGS["JHPixelProFaceBeautyBlend"] = JHPixelProFaceBeautyBlend
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProFaceBeautyBlend"] = "Face Beauty Blend"

    NODE_CLASS_MAPPINGS["JHPixelProUnwrapFace"] = JHPixelProUnwrapFace
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProUnwrapFace"] = "Unwrap Face (paste back)"

    NODE_CLASS_MAPPINGS["JHPixelProColorMatcher"] = JHPixelProColorMatcher
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorMatcher"] = "Color Matcher (LAB)"

    NODE_CLASS_MAPPINGS["JHPixelProColorMatcherRegion"] = JHPixelProColorMatcherRegion
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorMatcherRegion"] = "Color Matcher Region (LAB)"

    NODE_CLASS_MAPPINGS["JHPixelProToneCurve"] = JHPixelProToneCurve
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProToneCurve"] = "Tone Curve (RGB)"

    NODE_CLASS_MAPPINGS["JHPixelProHALDIdentity"] = JHPixelProHALDIdentity
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHALDIdentity"] = "HALD Identity"

    NODE_CLASS_MAPPINGS["JHPixelProLUTExport"] = JHPixelProLUTExport
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTExport"] = "LUT Export (.cube)"

    NODE_CLASS_MAPPINGS["JHPixelProLUTImport"] = JHPixelProLUTImport
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTImport"] = "LUT Import (.cube)"

    NODE_CLASS_MAPPINGS["JHPixelProLUTPreset"] = JHPixelProLUTPreset
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLUTPreset"] = "Color: LUT Preset"

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

    NODE_CLASS_MAPPINGS["JHPixelProColorLabApply"] = JHPixelProColorLabApply
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProColorLabApply"] = "Color: Apply ColorLab Preset"

    NODE_CLASS_MAPPINGS["JHPixelProLightroomPreset"] = JHPixelProLightroomPreset
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLightroomPreset"] = "Color: Lightroom Preset (XMP)"

    NODE_CLASS_MAPPINGS["JHPixelProLightroomPresetSheet"] = JHPixelProLightroomPresetSheet
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLightroomPresetSheet"] = (
        "Color: Lightroom Preset Contact Sheet"
    )

    from .nodes.lr_preset_routes import register_routes

    register_routes()

    NODE_CLASS_MAPPINGS["JHPixelProDepthAnything"] = JHPixelProDepthAnything
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDepthAnything"] = "Depth: Depth Anything V2"

    NODE_CLASS_MAPPINGS["JHPixelProDepthLensBlur"] = JHPixelProDepthLensBlur
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDepthLensBlur"] = "Depth: Lens Blur"

    NODE_CLASS_MAPPINGS["JHPixelProPaletteExtract"] = JHPixelProPaletteExtract
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProPaletteExtract"] = "Color: Palette Extract"

    NODE_CLASS_MAPPINGS["JHPixelProCLAHE"] = JHPixelProCLAHE
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProCLAHE"] = "Color: CLAHE Local Contrast"

    NODE_CLASS_MAPPINGS["JHPixelProSkinToneTriRegion"] = JHPixelProSkinToneTriRegion
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSkinToneTriRegion"] = "Face: Skin Tone Tri-Region"

    NODE_CLASS_MAPPINGS["JHPixelProLayerStackStart"] = JHPixelProLayerStackStart
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerStackStart"] = "Compositing: Layer Stack Start"

    NODE_CLASS_MAPPINGS["JHPixelProLayerAdd"] = JHPixelProLayerAdd
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerAdd"] = "Compositing: Layer Add"

    NODE_CLASS_MAPPINGS["JHPixelProLayerGroup"] = JHPixelProLayerGroup
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerGroup"] = "Compositing: Layer Group (nest)"

    NODE_CLASS_MAPPINGS["JHPixelProLayerFlatten"] = JHPixelProLayerFlatten
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProLayerFlatten"] = "Compositing: Layer Flatten"

    NODE_CLASS_MAPPINGS["JHPixelProQADiff"] = JHPixelProQADiff
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProQADiff"] = "QA: Image Diff (SSIM/PSNR)"

    NODE_CLASS_MAPPINGS["JHPixelProPyramidBlend"] = JHPixelProPyramidBlend
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProPyramidBlend"] = "Compositing: Pyramid Blend"

    NODE_CLASS_MAPPINGS["JHPixelProShadowsHighlights"] = JHPixelProShadowsHighlights
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProShadowsHighlights"] = "Color: Shadows/Highlights Pro"

    NODE_CLASS_MAPPINGS["JHPixelProDecontaminateColors"] = JHPixelProDecontaminateColors
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDecontaminateColors"] = "Mask: Decontaminate Colors"

    NODE_CLASS_MAPPINGS["JHPixelProAutoTone"] = JHPixelProAutoTone
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAutoTone"] = "Auto: Tone & White Balance"

    NODE_CLASS_MAPPINGS["JHPixelProAutoUpright"] = JHPixelProAutoUpright
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAutoUpright"] = "Auto: Upright"

    NODE_CLASS_MAPPINGS["JHPixelProMotionZoomBlur"] = JHPixelProMotionZoomBlur
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProMotionZoomBlur"] = "Filters: Motion / Zoom Blur"

    NODE_CLASS_MAPPINGS["JHPixelProHeal"] = JHPixelProHeal
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHeal"] = "Filters: Heal / Content-Aware Fill"

    NODE_CLASS_MAPPINGS["JHPixelProAutoBlemish"] = JHPixelProAutoBlemish
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProAutoBlemish"] = "Face: Auto Blemish"

    NODE_CLASS_MAPPINGS["JHPixelProDefringe"] = JHPixelProDefringe
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProDefringe"] = "Color: Defringe"

    NODE_CLASS_MAPPINGS["JHPixelProSkyMask"] = JHPixelProSkyMask
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProSkyMask"] = "Mask: Sky"

    NODE_CLASS_MAPPINGS["JHPixelProHDRMerge"] = JHPixelProHDRMerge
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProHDRMerge"] = "Color: HDR Merge (bracket)"

    NODE_CLASS_MAPPINGS["JHPixelProCullScore"] = JHPixelProCullScore
    NODE_DISPLAY_NAME_MAPPINGS["JHPixelProCullScore"] = "QA: Cull Score"

# Web extensions (JS/CSS) — not used in Phase 1.
WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
