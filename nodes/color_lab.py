"""ComfyUI wrapper for ACR-style ColorLab."""

from __future__ import annotations

from ..core.color_lab import GRAY_MIX_COLORS, HUE_ANCHORS, apply_colorlab_pipeline


def _float_meta(
    default: float, min_value: float, max_value: float, step: float
) -> tuple[str, dict]:
    return ("FLOAT", {"default": default, "min": min_value, "max": max_value, "step": step})


class JHPixelProColorLab:
    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        required = {
            "image": ("IMAGE",),
            "basic_exposure": _float_meta(0.0, -5.0, 5.0, 0.01),
            "basic_contrast": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_highlights": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_shadows": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_whites": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_blacks": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_texture": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_clarity": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_dehaze": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_vibrance": _float_meta(0.0, -100.0, 100.0, 1.0),
            "basic_saturation": _float_meta(0.0, -100.0, 100.0, 1.0),
        }
        for color in HUE_ANCHORS:
            required[f"hsl_{color}_hue"] = _float_meta(0.0, -100.0, 100.0, 1.0)
            required[f"hsl_{color}_sat"] = _float_meta(0.0, -100.0, 100.0, 1.0)
            required[f"hsl_{color}_lum"] = _float_meta(0.0, -100.0, 100.0, 1.0)
        for region in ("shadow", "mid", "highlight"):
            required[f"grade_{region}_hue"] = _float_meta(0.0, 0.0, 360.0, 1.0)
            required[f"grade_{region}_sat"] = _float_meta(0.0, 0.0, 100.0, 1.0)
            required[f"grade_{region}_lum"] = _float_meta(0.0, -100.0, 100.0, 1.0)
            required[f"grade_{region}_bal"] = _float_meta(0.0, -100.0, 100.0, 1.0)
        required["gray_enable"] = ("BOOLEAN", {"default": False})
        for color in GRAY_MIX_COLORS:
            required[f"gray_{color}"] = _float_meta(0.0, -200.0, 300.0, 1.0)
        return {"required": required}

    def apply(self, image, **params):
        return (apply_colorlab_pipeline(image, params),)
