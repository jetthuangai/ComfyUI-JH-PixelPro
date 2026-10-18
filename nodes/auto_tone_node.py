"""ComfyUI wrapper: one-click exposure, tone and white balance (ColorLab settings out)."""

from __future__ import annotations

import torch

from ..core.auto_tone import auto_tone_params
from ..core.color_lab import apply_preset, params_to_json


class JHPixelProAutoTone:
    """Measure the photo, propose ColorLab settings, apply them and output them as a preset."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "params_json")
    OUTPUT_TOOLTIPS = (
        "The corrected image.",
        "The settings that were applied (first image of a batch). Connect to a ColorLab's "
        "params_json to fine-tune them, or to Apply ColorLab Preset to reuse them.",
    )
    FUNCTION = "auto"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Photo to correct (each image of a batch is measured on its own)."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "0 = original, 1 = full automatic correction.",
                    },
                ),
            },
            "optional": {
                "exposure": ("BOOLEAN", {"default": True, "tooltip": "Fix brightness."}),
                "tone": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Set Whites, Blacks, Highlights, Shadows."},
                ),
                "white_balance": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Remove colour casts using near-grey areas."},
                ),
                "black_and_white": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Convert to black and white with an automatic colour mix.",
                    },
                ),
                "max_wb_shift": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Largest Temp/Tint change. Lower it for photos whose warm "
                        "or cool light is intended (sunsets, candlelight).",
                    },
                ),
            },
        }

    def auto(
        self,
        image: torch.Tensor,
        strength: float,
        exposure: bool = True,
        tone: bool = True,
        white_balance: bool = True,
        black_and_white: bool = False,
        max_wb_shift: float = 50.0,
    ) -> tuple[torch.Tensor, str]:
        with torch.no_grad():
            settings = auto_tone_params(
                image,
                exposure=bool(exposure),
                tone=bool(tone),
                white_balance=bool(white_balance),
                bw=bool(black_and_white),
                max_wb_shift=float(max_wb_shift),
            )
            frames, presets = [], []
            for index, params in enumerate(settings):
                out, preset = apply_preset(
                    image[index : index + 1], params_to_json(params), float(strength)
                )
                frames.append(out.to(image.device))
                presets.append(preset)
        return torch.cat(frames, dim=0), presets[0]
