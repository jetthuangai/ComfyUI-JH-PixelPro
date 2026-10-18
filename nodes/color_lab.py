"""ComfyUI wrapper for ACR-style ColorLab."""

from __future__ import annotations

import torch

from ..core.color_lab import (
    GRAY_MIX_COLORS,
    HUE_ANCHORS,
    TONE_ENGINES,
    apply_colorlab_pipeline,
    merge_params_json,
    params_to_json,
)


def _float_meta(
    default: float, min_value: float, max_value: float, step: float
) -> tuple[str, dict]:
    return ("FLOAT", {"default": default, "min": min_value, "max": max_value, "step": step})


class JHPixelProColorLab:
    """Apply the full ACR-style ColorLab stack to a ComfyUI IMAGE."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "params_json")
    OUTPUT_TOOLTIPS = (
        "Adjusted image.",
        "This node's settings (non-default values) as a preset: connect to another "
        "ColorLab's params_json or to 'Color: Apply ColorLab Preset'.",
    )
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
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
        wb_range = _float_meta(0.0, -100.0, 100.0, 1.0)[1]
        optional = {
            "exposure_mode": (
                ["display", "linear"],
                {
                    "default": "display",
                    "tooltip": (
                        "display = multiply the sRGB values by 2^EV (legacy). "
                        "linear = convert to linear light, multiply, convert back "
                        "(like a camera / Lightroom; more natural highlights)."
                    ),
                },
            ),
            # v1.7.0 additions go last so saved workflows keep their widget order; the
            # defaults reproduce v1.6.0 exactly.
            "tone_engine": (
                list(TONE_ENGINES),
                {
                    "default": "legacy",
                    "tooltip": (
                        "legacy = v1.6 behaviour. local = edge-aware Highlights/Shadows "
                        "(local Laplacian), Clarity (guided filter) and Dehaze (dark channel), "
                        "closer to Lightroom; runs on the GPU."
                    ),
                },
            ),
            "wb_temperature": ("FLOAT", {**wb_range, "tooltip": "Warmer (+) / cooler (-)."}),
            "wb_tint": ("FLOAT", {**wb_range, "tooltip": "Magenta (+) / green (-)."}),
            "params_json": (
                "STRING",
                {
                    "forceInput": True,
                    "tooltip": (
                        "Settings from another ColorLab or a Lightroom Preset node. Numbers are "
                        "added to the sliders above, so the sliders fine-tune the preset."
                    ),
                },
            ),
        }
        return {"required": required, "optional": optional}

    def apply(
        self, image: torch.Tensor, params_json: str | None = None, **params: object
    ) -> tuple[torch.Tensor, str]:
        merged = merge_params_json(params, params_json)
        return apply_colorlab_pipeline(image, merged), params_to_json(merged)
